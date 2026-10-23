"""Telemetry Service module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/services/telemetry_service.py

Robotics Telemetry Service

Python 3.11+
Standard library only.

Features:
    - Real-time telemetry ingestion
    - Multi-robot monitoring
    - Input validation
    - Timestamp normalization
    - Duplicate detection
    - Out-of-order sample handling
    - Bounded telemetry history
    - Latest telemetry snapshots
    - Robot telemetry statistics
    - Time-window aggregation
    - Battery, motor and vibration alerts
    - Connectivity monitoring
    - Telemetry callbacks
    - JSON / CSV / JSONL export
    - Simulation and robot service adapters

Safety:
    - Read-only telemetry monitoring
    - No hardware or actuator control
    - Alerts are advisory
"""

from __future__ import annotations

import csv
import json
import math
import threading
import uuid

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class TelemetryServiceError(Exception):
    pass


class TelemetryValidationError(TelemetryServiceError):
    pass


class TelemetryNotFoundError(TelemetryServiceError):
    pass


class TelemetryStateError(TelemetryServiceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class TelemetrySource(str, Enum):
    ROBOT = "robot"
    SIMULATION = "simulation"
    DIGITAL_TWIN = "digital_twin"
    ROS2 = "ros2"
    MQTT = "mqtt"
    KAFKA = "kafka"
    API = "api"
    OTHER = "other"


class TelemetrySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class TelemetryAlertCategory(str, Enum):
    BATTERY = "battery"
    MOTOR_TEMPERATURE = "motor_temperature"
    VIBRATION = "vibration"
    HEALTH = "health"
    NETWORK = "network"
    SENSOR = "sensor"
    DATA_QUALITY = "data_quality"


class TelemetryAggregation(str, Enum):
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    SUM = "sum"
    COUNT = "count"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(
    value: str | datetime,
) -> datetime:
    try:
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
    except (ValueError, AttributeError) as exc:
        raise TelemetryValidationError(
            "Invalid ISO timestamp"
        ) from exc

    if parsed.tzinfo is None:
        raise TelemetryValidationError(
            "Timestamp must include timezone"
        )

    return parsed.astimezone(timezone.utc)


def require_text(
    value: str,
    name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TelemetryValidationError(
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
        raise TelemetryValidationError(
            f"{name} must be finite"
        )

    result = float(value)

    if minimum is not None and result < minimum:
        raise TelemetryValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and result > maximum:
        raise TelemetryValidationError(
            f"{name} must be <= {maximum}"
        )

    return result


def json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise TelemetryValidationError(
            "Data must be JSON serializable"
        ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TelemetryServiceConfig:
    maximum_robots: int = 10000
    maximum_history_per_robot: int = 10000
    maximum_alert_history: int = 20000
    maximum_seen_ids_per_robot: int = 20000

    allow_out_of_order_samples: bool = True
    maximum_future_skew_seconds: float = 60.0

    battery_warning_soc: float = 20.0
    battery_critical_soc: float = 10.0

    motor_warning_temperature_c: float = 70.0
    motor_critical_temperature_c: float = 90.0

    vibration_warning_mm_s: float = 6.0
    vibration_critical_mm_s: float = 12.0

    health_warning_score: float = 70.0
    health_critical_score: float = 40.0

    telemetry_stale_after_seconds: float = 30.0

    export_directory: str = "data/telemetry"

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        for name in (
            "maximum_robots",
            "maximum_history_per_robot",
            "maximum_alert_history",
            "maximum_seen_ids_per_robot",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise TelemetryValidationError(
                    f"{name} must be a positive integer"
                )

        for name in (
            "battery_warning_soc",
            "battery_critical_soc",
            "health_warning_score",
            "health_critical_score",
        ):
            finite(
                getattr(self, name),
                name,
                minimum=0,
                maximum=100,
            )

        for name in (
            "motor_warning_temperature_c",
            "motor_critical_temperature_c",
            "vibration_warning_mm_s",
            "vibration_critical_mm_s",
            "maximum_future_skew_seconds",
            "telemetry_stale_after_seconds",
        ):
            finite(
                getattr(self, name),
                name,
                minimum=0,
            )

        if self.battery_critical_soc >= self.battery_warning_soc:
            raise TelemetryValidationError(
                "Battery critical threshold must be lower"
            )

        if (
            self.motor_critical_temperature_c
            <= self.motor_warning_temperature_c
        ):
            raise TelemetryValidationError(
                "Motor critical threshold must be higher"
            )

        if (
            self.vibration_critical_mm_s
            <= self.vibration_warning_mm_s
        ):
            raise TelemetryValidationError(
                "Vibration critical threshold must be higher"
            )

        if (
            self.health_critical_score
            >= self.health_warning_score
        ):
            raise TelemetryValidationError(
                "Health critical threshold must be lower"
            )

        require_text(
            self.export_directory,
            "export_directory",
        )

        if self.allow_hardware_control:
            raise TelemetryValidationError(
                "Hardware control is disabled"
            )


# ============================================================
# ROBOT TELEMETRY SAMPLE
# ============================================================

@dataclass(frozen=True)
class RoboticsTelemetrySample:
    robot_id: str

    battery_soc_percent: float
    motor_temperature_c: float
    vibration_mm_s: float

    health_score: float = 100.0

    x_m: float = 0.0
    y_m: float = 0.0
    heading_rad: float = 0.0

    linear_velocity_mps: float = 0.0
    angular_velocity_radps: float = 0.0

    energy_consumed_wh: float = 0.0
    operating_hours: float = 0.0

    sensor_available: bool = True
    network_connected: bool = True

    source: TelemetrySource = TelemetrySource.ROBOT

    timestamp: str = field(default_factory=utc_now)
    sample_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        require_text(self.robot_id, "robot_id")
        require_text(self.sample_id, "sample_id")

        parse_timestamp(self.timestamp)

        for name in (
            "battery_soc_percent",
            "health_score",
        ):
            finite(
                getattr(self, name),
                name,
                minimum=0,
                maximum=100,
            )

        for name in (
            "motor_temperature_c",
            "x_m",
            "y_m",
            "heading_rad",
            "linear_velocity_mps",
            "angular_velocity_radps",
        ):
            finite(getattr(self, name), name)

        for name in (
            "vibration_mm_s",
            "energy_consumed_wh",
            "operating_hours",
        ):
            finite(
                getattr(self, name),
                name,
                minimum=0,
            )

        if not isinstance(
            self.sensor_available,
            bool,
        ):
            raise TelemetryValidationError(
                "sensor_available must be boolean"
            )

        if not isinstance(
            self.network_connected,
            bool,
        ):
            raise TelemetryValidationError(
                "network_connected must be boolean"
            )

        object.__setattr__(
            self,
            "source",
            TelemetrySource(self.source),
        )

        object.__setattr__(
            self,
            "metadata",
            json_copy(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "robot_id": self.robot_id,
            "timestamp": self.timestamp,
            "source": self.source.value,
            "battery_soc_percent": (
                self.battery_soc_percent
            ),
            "motor_temperature_c": (
                self.motor_temperature_c
            ),
            "vibration_mm_s": self.vibration_mm_s,
            "health_score": self.health_score,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "heading_rad": self.heading_rad,
            "linear_velocity_mps": (
                self.linear_velocity_mps
            ),
            "angular_velocity_radps": (
                self.angular_velocity_radps
            ),
            "energy_consumed_wh": (
                self.energy_consumed_wh
            ),
            "operating_hours": self.operating_hours,
            "sensor_available": (
                self.sensor_available
            ),
            "network_connected": (
                self.network_connected
            ),
            "metadata": self.metadata,
        }


# ============================================================
# TELEMETRY ALERT
# ============================================================

@dataclass(frozen=True)
class TelemetryAlert:
    alert_id: str
    robot_id: str

    category: TelemetryAlertCategory
    severity: TelemetrySeverity

    message: str
    value: float | bool
    threshold: float | None

    sample_id: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "robot_id": self.robot_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "sample_id": self.sample_id,
            "created_at": self.created_at,
        }


# ============================================================
# TELEMETRY RULE ENGINE
# ============================================================

class RoboticsTelemetryRuleEngine:

    def __init__(
        self,
        config: TelemetryServiceConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        sample: RoboticsTelemetrySample,
    ) -> list[TelemetryAlert]:
        alerts: list[TelemetryAlert] = []

        def add(
            category: TelemetryAlertCategory,
            severity: TelemetrySeverity,
            message: str,
            value: float | bool,
            threshold: float | None,
        ) -> None:
            alerts.append(
                TelemetryAlert(
                    alert_id=str(uuid.uuid4()),
                    robot_id=sample.robot_id,
                    category=category,
                    severity=severity,
                    message=message,
                    value=value,
                    threshold=threshold,
                    sample_id=sample.sample_id,
                    created_at=utc_now(),
                )
            )

        soc = sample.battery_soc_percent

        if soc <= self.config.battery_critical_soc:
            add(
                TelemetryAlertCategory.BATTERY,
                TelemetrySeverity.CRITICAL,
                "Critical battery state of charge",
                soc,
                self.config.battery_critical_soc,
            )

        elif soc <= self.config.battery_warning_soc:
            add(
                TelemetryAlertCategory.BATTERY,
                TelemetrySeverity.WARNING,
                "Low battery state of charge",
                soc,
                self.config.battery_warning_soc,
            )

        temperature = sample.motor_temperature_c

        if (
            temperature
            >= self.config.motor_critical_temperature_c
        ):
            add(
                TelemetryAlertCategory.MOTOR_TEMPERATURE,
                TelemetrySeverity.CRITICAL,
                "Critical motor temperature",
                temperature,
                self.config.motor_critical_temperature_c,
            )

        elif (
            temperature
            >= self.config.motor_warning_temperature_c
        ):
            add(
                TelemetryAlertCategory.MOTOR_TEMPERATURE,
                TelemetrySeverity.WARNING,
                "High motor temperature",
                temperature,
                self.config.motor_warning_temperature_c,
            )

        vibration = sample.vibration_mm_s

        if vibration >= self.config.vibration_critical_mm_s:
            add(
                TelemetryAlertCategory.VIBRATION,
                TelemetrySeverity.CRITICAL,
                "Critical vibration detected",
                vibration,
                self.config.vibration_critical_mm_s,
            )

        elif vibration >= self.config.vibration_warning_mm_s:
            add(
                TelemetryAlertCategory.VIBRATION,
                TelemetrySeverity.WARNING,
                "High vibration detected",
                vibration,
                self.config.vibration_warning_mm_s,
            )

        health = sample.health_score

        if health <= self.config.health_critical_score:
            add(
                TelemetryAlertCategory.HEALTH,
                TelemetrySeverity.CRITICAL,
                "Critical robot health score",
                health,
                self.config.health_critical_score,
            )

        elif health <= self.config.health_warning_score:
            add(
                TelemetryAlertCategory.HEALTH,
                TelemetrySeverity.WARNING,
                "Robot health degradation",
                health,
                self.config.health_warning_score,
            )

        if not sample.network_connected:
            add(
                TelemetryAlertCategory.NETWORK,
                TelemetrySeverity.WARNING,
                "Robot network disconnected",
                False,
                None,
            )

        if not sample.sensor_available:
            add(
                TelemetryAlertCategory.SENSOR,
                TelemetrySeverity.CRITICAL,
                "Robot sensor unavailable",
                False,
                None,
            )

        return alerts


# ============================================================
# TELEMETRY STATISTICS
# ============================================================

class RoboticsTelemetryStatistics:

    NUMERIC_FIELDS = (
        "battery_soc_percent",
        "motor_temperature_c",
        "vibration_mm_s",
        "health_score",
        "x_m",
        "y_m",
        "heading_rad",
        "linear_velocity_mps",
        "angular_velocity_radps",
        "energy_consumed_wh",
        "operating_hours",
    )

    @classmethod
    def summarize(
        cls,
        samples: Iterable[RoboticsTelemetrySample],
    ) -> dict[str, Any]:
        records = list(samples)

        if not records:
            return {
                "sample_count": 0,
                "metrics": {},
            }

        metrics: dict[str, Any] = {}

        for field_name in cls.NUMERIC_FIELDS:
            values = [
                float(getattr(sample, field_name))
                for sample in records
            ]

            metrics[field_name] = {
                "minimum": round(min(values), 4),
                "maximum": round(max(values), 4),
                "average": round(
                    sum(values) / len(values),
                    4,
                ),
            }

        return {
            "sample_count": len(records),
            "first_timestamp": min(
                sample.timestamp
                for sample in records
            ),
            "last_timestamp": max(
                sample.timestamp
                for sample in records
            ),
            "source_distribution": dict(
                Counter(
                    sample.source.value
                    for sample in records
                )
            ),
            "metrics": metrics,
        }


# ============================================================
# TELEMETRY SERVICE
# ============================================================

class RoboticsTelemetryService:

    def __init__(
        self,
        config: TelemetryServiceConfig | None = None,
    ) -> None:
        self.config = (
            config or TelemetryServiceConfig()
        )

        self.history: dict[
            str,
            deque[RoboticsTelemetrySample]
        ] = {}

        self.latest: dict[
            str,
            RoboticsTelemetrySample
        ] = {}

        self.alerts: deque[TelemetryAlert] = deque(
            maxlen=self.config.maximum_alert_history
        )

        self._seen_ids: dict[
            str,
            set[str]
        ] = defaultdict(set)

        self._seen_order: dict[
            str,
            deque[str]
        ] = defaultdict(deque)

        self._callbacks: list[
            Callable[
                [RoboticsTelemetrySample, list[TelemetryAlert]],
                None,
            ]
        ] = []

        self._lock = threading.RLock()

        self.rule_engine = RoboticsTelemetryRuleEngine(
            self.config
        )

        self.statistics_engine = (
            RoboticsTelemetryStatistics()
        )

        self.ingested_count = 0
        self.duplicate_count = 0
        self.out_of_order_count = 0

    # --------------------------------------------------------
    # REGISTER CALLBACK
    # --------------------------------------------------------

    def register_callback(
        self,
        callback: Callable[
            [RoboticsTelemetrySample, list[TelemetryAlert]],
            None,
        ],
    ) -> None:
        if not callable(callback):
            raise TelemetryValidationError(
                "Callback must be callable"
            )

        with self._lock:
            self._callbacks.append(callback)

    # --------------------------------------------------------
    # TRACK SAMPLE IDS
    # --------------------------------------------------------

    def _remember_sample_id(
        self,
        robot_id: str,
        sample_id: str,
    ) -> None:
        seen = self._seen_ids[robot_id]
        order = self._seen_order[robot_id]

        seen.add(sample_id)
        order.append(sample_id)

        while (
            len(order)
            > self.config.maximum_seen_ids_per_robot
        ):
            oldest = order.popleft()
            seen.discard(oldest)

    # --------------------------------------------------------
    # INGEST SAMPLE
    # --------------------------------------------------------

    def ingest(
        self,
        sample: RoboticsTelemetrySample,
    ) -> list[TelemetryAlert]:
        if not isinstance(
            sample,
            RoboticsTelemetrySample,
        ):
            raise TelemetryValidationError(
                "Expected RoboticsTelemetrySample"
            )

        sample_time = parse_timestamp(
            sample.timestamp
        )

        future_limit = (
            datetime.now(timezone.utc)
            + timedelta(
                seconds=self.config.maximum_future_skew_seconds
            )
        )

        if sample_time > future_limit:
            raise TelemetryValidationError(
                "Telemetry timestamp too far in future"
            )

        with self._lock:
            robot_id = sample.robot_id

            if robot_id not in self.history:
                if (
                    len(self.history)
                    >= self.config.maximum_robots
                ):
                    raise TelemetryStateError(
                        "Maximum robot capacity reached"
                    )

                self.history[robot_id] = deque(
                    maxlen=(
                        self.config.maximum_history_per_robot
                    )
                )

            if sample.sample_id in self._seen_ids[robot_id]:
                self.duplicate_count += 1
                return []

            previous = self.latest.get(robot_id)

            is_out_of_order = (
                previous is not None
                and sample_time
                < parse_timestamp(previous.timestamp)
            )

            if is_out_of_order:
                self.out_of_order_count += 1

                if not self.config.allow_out_of_order_samples:
                    raise TelemetryStateError(
                        "Out-of-order telemetry rejected"
                    )

            self._remember_sample_id(
                robot_id,
                sample.sample_id,
            )

            self.history[robot_id].append(
                sample
            )

            if (
                previous is None
                or sample_time
                >= parse_timestamp(previous.timestamp)
            ):
                self.latest[robot_id] = sample

            self.ingested_count += 1

            # Historical out-of-order samples are retained,
            # but should not trigger live operational alerts.
            alerts = (
                []
                if is_out_of_order
                else self.rule_engine.evaluate(sample)
            )

            self.alerts.extend(alerts)

            callbacks = list(self._callbacks)

        # Callbacks run outside the lock to avoid
        # blocking telemetry ingestion.
        for callback in callbacks:
            try:
                callback(sample, alerts)
            except Exception:
                # Integrations should log callback errors
                # in their own observability layer.
                pass

        return alerts

    # --------------------------------------------------------
    # INGEST BATCH
    # --------------------------------------------------------

    def ingest_batch(
        self,
        samples: Iterable[RoboticsTelemetrySample],
    ) -> dict[str, int]:
        processed = 0
        generated_alerts = 0

        for sample in samples:
            alerts = self.ingest(sample)

            processed += 1
            generated_alerts += len(alerts)

        return {
            "processed": processed,
            "generated_alerts": generated_alerts,
        }

    # --------------------------------------------------------
    # GET LATEST
    # --------------------------------------------------------

    def latest_sample(
        self,
        robot_id: str,
    ) -> RoboticsTelemetrySample:
        with self._lock:
            sample = self.latest.get(robot_id)

            if sample is None:
                raise TelemetryNotFoundError(
                    f"No telemetry for {robot_id}"
                )

            return sample

    # --------------------------------------------------------
    # QUERY HISTORY
    # --------------------------------------------------------

    def query_history(
        self,
        robot_id: str,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        limit: int | None = None,
    ) -> list[RoboticsTelemetrySample]:
        with self._lock:
            if robot_id not in self.history:
                raise TelemetryNotFoundError(
                    f"No telemetry for {robot_id}"
                )

            records = list(
                self.history[robot_id]
            )

        start_time = (
            parse_timestamp(start)
            if start is not None
            else None
        )

        end_time = (
            parse_timestamp(end)
            if end is not None
            else None
        )

        if (
            start_time is not None
            and end_time is not None
            and start_time > end_time
        ):
            raise TelemetryValidationError(
                "start must not exceed end"
            )

        filtered = []

        for sample in records:
            timestamp = parse_timestamp(
                sample.timestamp
            )

            if (
                start_time is not None
                and timestamp < start_time
            ):
                continue

            if (
                end_time is not None
                and timestamp > end_time
            ):
                continue

            filtered.append(sample)

        filtered.sort(
            key=lambda sample: (
                parse_timestamp(sample.timestamp),
                sample.sample_id,
            )
        )

        if limit is not None:
            if (
                isinstance(limit, bool)
                or not isinstance(limit, int)
                or limit < 1
            ):
                raise TelemetryValidationError(
                    "limit must be positive"
                )

            filtered = filtered[-limit:]

        return filtered

    # --------------------------------------------------------
    # AGGREGATE METRIC
    # --------------------------------------------------------

    def aggregate(
        self,
        robot_id: str,
        *,
        field_name: str,
        aggregation: TelemetryAggregation,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> dict[str, Any]:
        if (
            field_name
            not in RoboticsTelemetryStatistics.NUMERIC_FIELDS
        ):
            raise TelemetryValidationError(
                f"Unsupported numeric field: {field_name}"
            )

        records = self.query_history(
            robot_id,
            start=start,
            end=end,
        )

        values = [
            float(getattr(sample, field_name))
            for sample in records
        ]

        operation = TelemetryAggregation(
            aggregation
        )

        if operation == TelemetryAggregation.COUNT:
            value: float | int | None = len(values)

        elif not values:
            value = None

        elif operation == TelemetryAggregation.AVERAGE:
            value = sum(values) / len(values)

        elif operation == TelemetryAggregation.MINIMUM:
            value = min(values)

        elif operation == TelemetryAggregation.MAXIMUM:
            value = max(values)

        elif operation == TelemetryAggregation.SUM:
            value = sum(values)

        else:
            raise TelemetryValidationError(
                "Unsupported aggregation"
            )

        return {
            "robot_id": robot_id,
            "field_name": field_name,
            "aggregation": operation.value,
            "value": (
                round(value, 4)
                if isinstance(value, float)
                else value
            ),
            "sample_count": len(values),
        }

    # --------------------------------------------------------
    # ROBOT STATISTICS
    # --------------------------------------------------------

    def robot_statistics(
        self,
        robot_id: str,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> dict[str, Any]:
        records = self.query_history(
            robot_id,
            start=start,
            end=end,
        )

        result = self.statistics_engine.summarize(
            records
        )

        result["robot_id"] = robot_id

        return result

    # --------------------------------------------------------
    # STALE TELEMETRY DETECTION
    # --------------------------------------------------------

    def stale_robots(
        self,
        *,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        current = (
            now
            if now is not None
            else datetime.now(timezone.utc)
        )

        if current.tzinfo is None:
            raise TelemetryValidationError(
                "now must be timezone-aware"
            )

        current = current.astimezone(
            timezone.utc
        )

        results = []

        with self._lock:
            latest = dict(self.latest)

        for robot_id, sample in latest.items():
            age_seconds = (
                current
                - parse_timestamp(sample.timestamp)
            ).total_seconds()

            if (
                age_seconds
                > self.config.telemetry_stale_after_seconds
            ):
                results.append({
                    "robot_id": robot_id,
                    "age_seconds": round(
                        age_seconds,
                        2,
                    ),
                    "last_timestamp": sample.timestamp,
                })

        return sorted(
            results,
            key=lambda item: -item["age_seconds"],
        )

    # --------------------------------------------------------
    # LIST ALERTS
    # --------------------------------------------------------

    def list_alerts(
        self,
        *,
        robot_id: str | None = None,
        severity: TelemetrySeverity | None = None,
        category: TelemetryAlertCategory | None = None,
    ) -> list[TelemetryAlert]:
        with self._lock:
            alerts = list(self.alerts)

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
                if alert.severity
                == TelemetrySeverity(severity)
            ]

        if category is not None:
            alerts = [
                alert
                for alert in alerts
                if alert.category
                == TelemetryAlertCategory(category)
            ]

        return alerts

    # --------------------------------------------------------
    # FLEET STATISTICS
    # --------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        with self._lock:
            latest = list(self.latest.values())
            alerts = list(self.alerts)

            return {
                "registered_robots": len(self.history),
                "ingested_samples": self.ingested_count,
                "duplicate_samples": self.duplicate_count,
                "out_of_order_samples": (
                    self.out_of_order_count
                ),
                "retained_samples": sum(
                    len(history)
                    for history in self.history.values()
                ),
                "retained_alerts": len(alerts),
                "average_battery_soc_percent": (
                    round(
                        sum(
                            sample.battery_soc_percent
                            for sample in latest
                        ) / len(latest),
                        2,
                    )
                    if latest
                    else None
                ),
                "average_health_score": (
                    round(
                        sum(
                            sample.health_score
                            for sample in latest
                        ) / len(latest),
                        2,
                    )
                    if latest
                    else None
                ),
                "alert_severity_distribution": dict(
                    Counter(
                        alert.severity.value
                        for alert in alerts
                    )
                ),
                "stale_robot_count": len(
                    self.stale_robots()
                ),
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    def dashboard(self) -> dict[str, Any]:
        with self._lock:
            latest = {
                robot_id: sample.to_dict()
                for robot_id, sample
                in self.latest.items()
            }

            recent_alerts = [
                alert.to_dict()
                for alert in list(self.alerts)[-100:]
            ]

        return {
            "generated_at": utc_now(),
            "statistics": self.statistics(),
            "latest_telemetry": latest,
            "stale_robots": self.stale_robots(),
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
        *,
        robot_id: str | None = None,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            if robot_id is None:
                records = [
                    sample
                    for history in self.history.values()
                    for sample in history
                ]
            else:
                records = list(
                    self.history.get(robot_id, ())
                )

        records.sort(
            key=lambda sample: sample.timestamp
        )

        fieldnames = [
            "sample_id",
            "robot_id",
            "timestamp",
            "source",
            "battery_soc_percent",
            "motor_temperature_c",
            "vibration_mm_s",
            "health_score",
            "x_m",
            "y_m",
            "heading_rad",
            "linear_velocity_mps",
            "angular_velocity_radps",
            "energy_consumed_wh",
            "operating_hours",
            "sensor_available",
            "network_connected",
        ]

        with output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )

            writer.writeheader()

            for sample in records:
                writer.writerow(
                    sample.to_dict()
                )

        return output

    # --------------------------------------------------------
    # EXPORT JSONL
    # --------------------------------------------------------

    def export_jsonl(
        self,
        path: str | Path,
        *,
        robot_id: str | None = None,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            if robot_id is None:
                records = [
                    sample
                    for history in self.history.values()
                    for sample in history
                ]
            else:
                records = list(
                    self.history.get(robot_id, ())
                )

        records.sort(
            key=lambda sample: sample.timestamp
        )

        with output.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for sample in records:
                handle.write(
                    json.dumps(
                        sample.to_dict(),
                        allow_nan=False,
                    )
                    + "\n"
                )

        return output


# ============================================================
# SIMULATION SERVICE ADAPTER
# ============================================================

class SimulationTelemetryAdapter:

    @staticmethod
    def from_snapshot(
        snapshot: Mapping[str, Any],
    ) -> list[RoboticsTelemetrySample]:
        results = []

        timestamp = utc_now()

        for robot_id, robot in snapshot.get(
            "robots",
            {},
        ).items():
            pose = robot.get("pose", {})
            velocity = robot.get("velocity", {})

            results.append(
                RoboticsTelemetrySample(
                    robot_id=robot_id,
                    battery_soc_percent=robot.get(
                        "battery_soc_percent",
                        100.0,
                    ),
                    motor_temperature_c=robot.get(
                        "motor_temperature_c",
                        25.0,
                    ),
                    vibration_mm_s=0.0,
                    health_score=100.0,
                    x_m=pose.get("x", 0.0),
                    y_m=pose.get("y", 0.0),
                    heading_rad=pose.get(
                        "heading_rad",
                        0.0,
                    ),
                    linear_velocity_mps=velocity.get(
                        "linear_mps",
                        0.0,
                    ),
                    angular_velocity_radps=velocity.get(
                        "angular_radps",
                        0.0,
                    ),
                    energy_consumed_wh=robot.get(
                        "energy_consumed_wh",
                        0.0,
                    ),
                    source=TelemetrySource.SIMULATION,
                    timestamp=timestamp,
                    sensor_available=robot.get(
                        "sensor_available",
                        True,
                    ),
                    network_connected=robot.get(
                        "network_connected",
                        True,
                    ),
                    metadata={
                        "session_id": snapshot.get(
                            "session_id"
                        ),
                        "simulation_time_seconds": (
                            snapshot.get(
                                "simulation_time_seconds"
                            )
                        ),
                    },
                )
            )

        return results


# ============================================================
# ROBOT SERVICE ADAPTER
# ============================================================

class RobotServiceTelemetryAdapter:
    """
    Converts RoboticsTelemetrySample into the
    RobotTelemetry model from robot_service.py.
    """

    @staticmethod
    def synchronize(
        telemetry_service: RoboticsTelemetryService,
        robot_service: Any,
    ) -> int:
        from backend.app.services.robot_service import (
            RobotTelemetry,
        )

        updated = 0

        for robot_id, sample in list(
            telemetry_service.latest.items()
        ):
            if robot_id not in robot_service.robots:
                continue

            robot_service.update_telemetry(
                RobotTelemetry(
                    robot_id=robot_id,
                    battery_soc_percent=(
                        sample.battery_soc_percent
                    ),
                    health_score=sample.health_score,
                    motor_temperature_c=(
                        sample.motor_temperature_c
                    ),
                    vibration_mm_s=(
                        sample.vibration_mm_s
                    ),
                    x_m=sample.x_m,
                    y_m=sample.y_m,
                    heading_rad=sample.heading_rad,
                    operating_hours=(
                        sample.operating_hours
                    ),
                    sensor_available=(
                        sample.sensor_available
                    ),
                    network_connected=(
                        sample.network_connected
                    ),
                    timestamp=sample.timestamp,
                )
            )

            updated += 1

        return updated


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsTelemetryService()

    service.ingest(
        RoboticsTelemetrySample(
            robot_id="robot_001",
            battery_soc_percent=87.0,
            motor_temperature_c=45.0,
            vibration_mm_s=2.1,
            health_score=94.0,
            x_m=2.0,
            y_m=3.0,
            linear_velocity_mps=0.6,
            source=TelemetrySource.SIMULATION,
        )
    )

    service.ingest(
        RoboticsTelemetrySample(
            robot_id="robot_002",
            battery_soc_percent=8.0,
            motor_temperature_c=94.0,
            vibration_mm_s=13.5,
            health_score=32.0,
            x_m=6.0,
            y_m=4.0,
            sensor_available=False,
            network_connected=False,
            source=TelemetrySource.SIMULATION,
        )
    )

    service.ingest(
        RoboticsTelemetrySample(
            robot_id="robot_001",
            battery_soc_percent=85.0,
            motor_temperature_c=47.0,
            vibration_mm_s=2.4,
            health_score=93.0,
            x_m=2.5,
            y_m=3.2,
            linear_velocity_mps=0.7,
            source=TelemetrySource.SIMULATION,
        )
    )

    json_path = service.export_json(
        "data/telemetry/telemetry_dashboard.json"
    )

    csv_path = service.export_csv(
        "data/telemetry/telemetry_history.csv"
    )

    jsonl_path = service.export_jsonl(
        "data/telemetry/telemetry_history.jsonl"
    )

    print(
        json.dumps(
            {
                "statistics": service.statistics(),
                "robot_001": service.robot_statistics(
                    "robot_001"
                ),
                "alerts": [
                    alert.to_dict()
                    for alert in service.list_alerts()
                ],
                "files": [
                    str(json_path),
                    str(csv_path),
                    str(jsonl_path),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

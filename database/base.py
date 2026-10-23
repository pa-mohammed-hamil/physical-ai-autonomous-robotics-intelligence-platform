"""Base module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/monitoring/base.py

Physical AI & Autonomous Robotics Intelligence Platform

Shared Monitoring Infrastructure

Features:
- Generic metric records and collectors
- Thread-safe bounded metric storage
- Robot, component and fleet monitoring
- Health status and severity classification
- Configurable monitoring thresholds
- P50/P90/P95/P99 percentile calculations
- Time-window filtering
- Metric aggregation
- JSON serialization
- Prometheus metrics export
- Alert generation
- Monitoring collector registry
- UTC timestamp handling
- Custom metric support

Python 3.11+
Dependencies: Standard library only

This module provides monitoring infrastructure only.
It does not control robots or implement safety interlocks.
"""

from __future__ import annotations

import json
import logging
import math
import re
import statistics
import threading
import uuid

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
)


logger = logging.getLogger(__name__)

T = TypeVar("T")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def validate_number(
    value: float,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float)
    ):
        raise TypeError(
            f"{name} must be numeric"
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"{name} must be <= {maximum}"
        )

    return value


def validate_count(
    value: int,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a nonnegative integer"
        )

    return value


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricUnit(str, Enum):
    COUNT = "count"
    PERCENT = "percent"
    RATIO = "ratio"
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"
    BYTES = "bytes"
    BYTES_PER_SECOND = "bytes_per_second"
    METERS = "meters"
    METERS_PER_SECOND = "meters_per_second"
    DEGREES = "degrees"
    CELSIUS = "celsius"
    WATTS = "watts"
    WATT_HOURS = "watt_hours"
    HOURS = "hours"
    CURRENCY = "currency"
    SCORE = "score"
    NONE = "none"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class ComparisonOperator(str, Enum):
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"


@dataclass(frozen=True)
class MetricLabels:
    robot_id: str = "unknown"
    component_id: Optional[str] = None
    device_id: Optional[str] = None
    model_id: Optional[str] = None
    environment: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        values = asdict(self)

        return {
            key: str(value)
            for key, value in values.items()
            if value is not None
        }


@dataclass(frozen=True)
class MetricRecord:
    name: str
    value: float
    metric_type: MetricType = MetricType.GAUGE
    unit: MetricUnit = MetricUnit.NONE

    labels: MetricLabels = field(
        default_factory=MetricLabels
    )

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError(
                "Metric name is required"
            )

        validate_number(
            self.value,
            "value",
        )

        ensure_utc(self.timestamp)

        if (
            self.metric_type == MetricType.COUNTER
            and self.value < 0
        ):
            raise ValueError(
                "Counter values must be nonnegative"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "metric_type": self.metric_type.value,
            "unit": self.unit.value,
            "labels": self.labels.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "metadata": serialize(self.metadata),
        }


@dataclass(frozen=True)
class MetricThreshold:
    metric_name: str
    warning_value: float
    critical_value: float

    operator: ComparisonOperator = (
        ComparisonOperator.GREATER_EQUAL
    )

    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.metric_name:
            raise ValueError(
                "Metric name is required"
            )

        validate_number(
            self.warning_value,
            "warning_value",
        )

        validate_number(
            self.critical_value,
            "critical_value",
        )

        if self.operator in (
            ComparisonOperator.GREATER_THAN,
            ComparisonOperator.GREATER_EQUAL,
        ):
            if self.critical_value <= self.warning_value:
                raise ValueError(
                    "Critical threshold must exceed warning"
                )

        else:
            if self.critical_value >= self.warning_value:
                raise ValueError(
                    "Critical threshold must be below warning"
                )

    def _matches(
        self,
        value: float,
        threshold: float,
    ) -> bool:
        if self.operator == ComparisonOperator.GREATER_THAN:
            return value > threshold

        if self.operator == ComparisonOperator.GREATER_EQUAL:
            return value >= threshold

        if self.operator == ComparisonOperator.LESS_THAN:
            return value < threshold

        return value <= threshold

    def evaluate(
        self,
        value: float,
    ) -> HealthStatus:
        validate_number(value, "metric_value")

        if not self.enabled:
            return HealthStatus.UNKNOWN

        if self._matches(
            value,
            self.critical_value,
        ):
            return HealthStatus.CRITICAL

        if self._matches(
            value,
            self.warning_value,
        ):
            return HealthStatus.WARNING

        return HealthStatus.HEALTHY


@dataclass(frozen=True)
class MonitoringAlert:
    alert_id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    observed_value: float
    threshold_value: float

    robot_id: Optional[str] = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "metric_name": self.metric_name,
            "severity": self.severity.value,
            "message": self.message,
            "observed_value": self.observed_value,
            "threshold_value": self.threshold_value,
            "robot_id": self.robot_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MetricStatistics:
    metric_name: str
    count: int

    minimum: Optional[float]
    maximum: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    standard_deviation: Optional[float]

    p50: Optional[float]
    p90: Optional[float]
    p95: Optional[float]
    p99: Optional[float]

    total: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MonitoringReport:
    report_id: str
    generated_at: datetime
    scope: str

    total_records: int
    total_metrics: int

    statistics: dict[str, MetricStatistics]
    health: dict[str, HealthStatus]
    alerts: list[MonitoringAlert]

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "scope": self.scope,
            "total_records": self.total_records,
            "total_metrics": self.total_metrics,
            "statistics": {
                name: item.to_dict()
                for name, item in self.statistics.items()
            },
            "health": {
                name: status.value
                for name, status in self.health.items()
            },
            "alerts": [
                alert.to_dict()
                for alert in self.alerts
            ],
            "metadata": serialize(self.metadata),
        }


def serialize(value: Any) -> Any:
    if value is None or isinstance(
        value, (str, bool, int)
    ):
        return value

    if isinstance(value, float):
        return (
            value
            if math.isfinite(value)
            else None
        )

    if isinstance(value, Enum):
        return serialize(value.value)

    if isinstance(value, datetime):
        return value.isoformat()

    if is_dataclass(value) and not isinstance(
        value, type
    ):
        return serialize(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            serialize(item)
            for item in value
        ]

    if hasattr(value, "to_dict"):
        return serialize(value.to_dict())

    raise TypeError(
        f"Unsupported serialization type: {type(value)}"
    )


class StatisticsCalculator:
    @staticmethod
    def percentile(
        values: Iterable[float],
        percentile: float,
    ) -> Optional[float]:
        validate_number(
            percentile,
            "percentile",
            0,
            100,
        )

        ordered = sorted(values)

        if not ordered:
            return None

        position = (
            (len(ordered) - 1)
            * percentile / 100
        )

        lower = math.floor(position)
        upper = math.ceil(position)

        if lower == upper:
            return ordered[lower]

        weight = position - lower

        return (
            ordered[lower] * (1 - weight)
            + ordered[upper] * weight
        )

    @staticmethod
    def calculate(
        metric_name: str,
        values: Iterable[float],
    ) -> MetricStatistics:
        data = list(values)

        for value in data:
            validate_number(
                value,
                "metric_value",
            )

        if not data:
            return MetricStatistics(
                metric_name=metric_name,
                count=0,
                minimum=None,
                maximum=None,
                mean=None,
                median=None,
                standard_deviation=None,
                p50=None,
                p90=None,
                p95=None,
                p99=None,
                total=0.0,
            )

        return MetricStatistics(
            metric_name=metric_name,
            count=len(data),
            minimum=min(data),
            maximum=max(data),
            mean=statistics.fmean(data),
            median=statistics.median(data),
            standard_deviation=(
                statistics.pstdev(data)
                if len(data) > 1
                else 0.0
            ),
            p50=StatisticsCalculator.percentile(
                data, 50
            ),
            p90=StatisticsCalculator.percentile(
                data, 90
            ),
            p95=StatisticsCalculator.percentile(
                data, 95
            ),
            p99=StatisticsCalculator.percentile(
                data, 99
            ),
            total=sum(data),
        )


class BaseMetricsCollector(ABC, Generic[T]):
    """
    Generic thread-safe monitoring collector.

    Subclasses define:
        record_to_metrics(record)
        get_record_timestamp(record)
        get_record_robot_id(record)

    History is bounded by max_records.
    """

    def __init__(
        self,
        name: str,
        max_records: int = 100000,
    ) -> None:
        if not name:
            raise ValueError(
                "Collector name is required"
            )

        if max_records < 1:
            raise ValueError(
                "max_records must be positive"
            )

        self.name = name

        self._records: deque[T] = deque(
            maxlen=max_records
        )

        self._lock = threading.RLock()

    @abstractmethod
    def record_to_metrics(
        self,
        record: T,
    ) -> Mapping[str, float]:
        """Convert one record into numeric metrics."""

    @abstractmethod
    def get_record_timestamp(
        self,
        record: T,
    ) -> datetime:
        """Return record timestamp."""

    @abstractmethod
    def get_record_robot_id(
        self,
        record: T,
    ) -> str:
        """Return robot identifier."""

    def record(
        self,
        record: T,
    ) -> None:
        ensure_utc(
            self.get_record_timestamp(record)
        )

        with self._lock:
            self._records.append(record)

    def get_records(
        self,
        robot_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[T]:
        if start_time is not None:
            start_time = ensure_utc(start_time)

        if end_time is not None:
            end_time = ensure_utc(end_time)

        if (
            start_time is not None
            and end_time is not None
            and start_time > end_time
        ):
            raise ValueError(
                "start_time must not exceed end_time"
            )

        with self._lock:
            records = list(self._records)

        result: list[T] = []

        for record in records:
            if (
                robot_id is not None
                and self.get_record_robot_id(record)
                != robot_id
            ):
                continue

            timestamp = ensure_utc(
                self.get_record_timestamp(record)
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

            result.append(record)

        return result

    def get_recent_records(
        self,
        minutes: float = 60,
        robot_id: Optional[str] = None,
    ) -> list[T]:
        validate_number(
            minutes,
            "minutes",
            minimum=0,
        )

        start_time = (
            utc_now()
            - timedelta(minutes=minutes)
        )

        return self.get_records(
            robot_id=robot_id,
            start_time=start_time,
        )

    def count(
        self,
        robot_id: Optional[str] = None,
    ) -> int:
        if robot_id is None:
            with self._lock:
                return len(self._records)

        return len(
            self.get_records(robot_id=robot_id)
        )

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def get_robot_ids(self) -> list[str]:
        with self._lock:
            records = list(self._records)

        return sorted({
            self.get_record_robot_id(record)
            for record in records
        })

    def get_statistics(
        self,
        robot_id: Optional[str] = None,
    ) -> dict[str, MetricStatistics]:
        records = self.get_records(
            robot_id=robot_id
        )

        grouped: dict[str, list[float]] = {}

        for record in records:
            metrics = self.record_to_metrics(
                record
            )

            for name, value in metrics.items():
                validate_number(
                    value,
                    name,
                )

                grouped.setdefault(
                    name, []
                ).append(float(value))

        return {
            name: StatisticsCalculator.calculate(
                name,
                values,
            )
            for name, values in grouped.items()
        }

    def get_robot_metrics(
        self,
        robot_id: str,
    ) -> dict[str, Any]:
        return {
            "robot_id": robot_id,
            "total_records": self.count(robot_id),
            "statistics": {
                name: stats.to_dict()
                for name, stats in self.get_statistics(
                    robot_id
                ).items()
            },
        }

    def get_fleet_metrics(self) -> dict[str, Any]:
        robot_ids = self.get_robot_ids()

        return {
            "collector": self.name,
            "total_robots": len(robot_ids),
            "total_records": self.count(),
            "statistics": {
                name: stats.to_dict()
                for name, stats in self.get_statistics().items()
            },
            "robots": {
                robot_id: self.get_robot_metrics(
                    robot_id
                )
                for robot_id in robot_ids
            },
            "evaluated_at": utc_now().isoformat(),
        }


class GenericMetricsCollector(
    BaseMetricsCollector[MetricRecord]
):
    """
    General-purpose metric collector supporting
    configurable thresholds and alert generation.
    """

    def __init__(
        self,
        name: str = "generic_metrics",
        max_records: int = 100000,
    ) -> None:
        super().__init__(
            name=name,
            max_records=max_records,
        )

        self._thresholds: dict[
            str, MetricThreshold
        ] = {}

    def record_to_metrics(
        self,
        record: MetricRecord,
    ) -> Mapping[str, float]:
        return {
            record.name: record.value
        }

    def get_record_timestamp(
        self,
        record: MetricRecord,
    ) -> datetime:
        return record.timestamp

    def get_record_robot_id(
        self,
        record: MetricRecord,
    ) -> str:
        return record.labels.robot_id

    def register_threshold(
        self,
        threshold: MetricThreshold,
    ) -> None:
        with self._lock:
            self._thresholds[
                threshold.metric_name
            ] = threshold

    def remove_threshold(
        self,
        metric_name: str,
    ) -> bool:
        with self._lock:
            return self._thresholds.pop(
                metric_name,
                None,
            ) is not None

    def evaluate_health(
        self,
        robot_id: Optional[str] = None,
    ) -> dict[str, HealthStatus]:
        records = self.get_records(
            robot_id=robot_id
        )

        with self._lock:
            thresholds = dict(self._thresholds)

        latest: dict[str, MetricRecord] = {}

        for record in records:
            current = latest.get(record.name)

            if (
                current is None
                or record.timestamp >= current.timestamp
            ):
                latest[record.name] = record

        health: dict[str, HealthStatus] = {}

        for name, record in latest.items():
            threshold = thresholds.get(name)

            health[name] = (
                threshold.evaluate(record.value)
                if threshold is not None
                else HealthStatus.UNKNOWN
            )

        return health

    def get_alerts(
        self,
        robot_id: Optional[str] = None,
    ) -> list[MonitoringAlert]:
        records = self.get_records(
            robot_id=robot_id
        )

        with self._lock:
            thresholds = dict(self._thresholds)

        latest: dict[
            tuple[str, str],
            MetricRecord,
        ] = {}

        for record in records:
            key = (
                record.labels.robot_id,
                record.name,
            )

            current = latest.get(key)

            if (
                current is None
                or record.timestamp >= current.timestamp
            ):
                latest[key] = record

        alerts: list[MonitoringAlert] = []

        for (robot, name), record in latest.items():
            threshold = thresholds.get(name)

            if threshold is None:
                continue

            health = threshold.evaluate(
                record.value
            )

            if health not in (
                HealthStatus.WARNING,
                HealthStatus.CRITICAL,
            ):
                continue

            severity = (
                AlertSeverity.CRITICAL
                if health == HealthStatus.CRITICAL
                else AlertSeverity.WARNING
            )

            limit = (
                threshold.critical_value
                if health == HealthStatus.CRITICAL
                else threshold.warning_value
            )

            alerts.append(
                MonitoringAlert(
                    alert_id=str(uuid.uuid4()),
                    metric_name=name,
                    severity=severity,
                    message=(
                        f"{name} crossed the "
                        f"{health.value} threshold"
                    ),
                    observed_value=record.value,
                    threshold_value=limit,
                    robot_id=robot,
                    timestamp=record.timestamp,
                )
            )

        return alerts

    def generate_report(
        self,
        robot_id: Optional[str] = None,
    ) -> MonitoringReport:
        statistics_data = self.get_statistics(
            robot_id=robot_id
        )

        return MonitoringReport(
            report_id=str(uuid.uuid4()),
            generated_at=utc_now(),
            scope=robot_id or "fleet",
            total_records=self.count(robot_id),
            total_metrics=len(statistics_data),
            statistics=statistics_data,
            health=self.evaluate_health(
                robot_id=robot_id
            ),
            alerts=self.get_alerts(
                robot_id=robot_id
            ),
            metadata={
                "collector": self.name,
            },
        )

    def export_json(
        self,
        robot_id: Optional[str] = None,
    ) -> str:
        report = self.generate_report(
            robot_id=robot_id
        )

        return json.dumps(
            report.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    @staticmethod
    def _prometheus_name(
        name: str,
    ) -> str:
        result = re.sub(
            r"[^a-zA-Z0-9_:]",
            "_",
            name,
        )

        if not re.match(
            r"^[a-zA-Z_:]",
            result,
        ):
            result = "_" + result

        return result

    @staticmethod
    def _escape_label(
        value: str,
    ) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace("\n", "\\n")
            .replace('"', '\\"')
        )

    def export_prometheus(self) -> str:
        """
        Export latest gauge-like snapshots.

        COUNTER records are assumed to contain
        cumulative counter values from the source.
        """

        with self._lock:
            records = list(self._records)

        latest: dict[
            tuple[str, tuple[tuple[str, str], ...]],
            MetricRecord,
        ] = {}

        for record in records:
            labels = tuple(sorted(
                record.labels.to_dict().items()
            ))

            key = (
                record.name,
                labels,
            )

            current = latest.get(key)

            if (
                current is None
                or record.timestamp >= current.timestamp
            ):
                latest[key] = record

        lines: list[str] = []
        declared: set[str] = set()
        metric_types: dict[str, MetricType] = {}

        for (raw_name, labels), record in sorted(
            latest.items(),
            key=lambda item: item[0],
        ):
            name = self._prometheus_name(
                raw_name
            )

            metric_type = (
                MetricType.COUNTER
                if record.metric_type == MetricType.COUNTER
                else MetricType.GAUGE
            )

            previous_type = metric_types.get(name)

            if (
                previous_type is not None
                and previous_type != metric_type
            ):
                logger.warning(
                    "Conflicting Prometheus metric type: %s",
                    name,
                )
                continue

            metric_types[name] = metric_type

            if name not in declared:
                lines.append(
                    f"# TYPE {name} {metric_type.value}"
                )
                declared.add(name)

            label_string = ",".join(
                f'{key}="{self._escape_label(value)}"'
                for key, value in labels
            )

            suffix = (
                f"{{{label_string}}}"
                if label_string
                else ""
            )

            lines.append(
                f"{name}{suffix} {record.value}"
            )

        return "\n".join(lines) + (
            "\n" if lines else ""
        )


class MonitoringRegistry:
    """
    Thread-safe registry of monitoring collectors.
    """

    def __init__(self) -> None:
        self._collectors: dict[str, Any] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        collector: Any,
    ) -> None:
        if not name:
            raise ValueError(
                "Collector name is required"
            )

        if not callable(
            getattr(collector, "get_fleet_metrics", None)
        ):
            raise TypeError(
                "Collector must implement get_fleet_metrics"
            )

        with self._lock:
            self._collectors[name] = collector

    def unregister(
        self,
        name: str,
    ) -> bool:
        with self._lock:
            return self._collectors.pop(
                name,
                None,
            ) is not None

    def get(
        self,
        name: str,
    ) -> Optional[Any]:
        with self._lock:
            return self._collectors.get(name)

    def list_collectors(self) -> list[str]:
        with self._lock:
            return sorted(self._collectors)

    def get_fleet_metrics(self) -> dict[str, Any]:
        with self._lock:
            collectors = dict(self._collectors)

        results: dict[str, Any] = {}
        errors: dict[str, str] = {}

        for name, collector in collectors.items():
            try:
                results[name] = serialize(
                    collector.get_fleet_metrics()
                )

            except Exception as exc:
                errors[name] = (
                    f"{type(exc).__name__}: {exc}"
                )

                logger.exception(
                    "Monitoring collector failed: %s",
                    name,
                )

        return {
            "collectors": results,
            "errors": errors,
            "evaluated_at": utc_now().isoformat(),
        }

    def clear(self) -> None:
        with self._lock:
            self._collectors.clear()


monitoring_registry = MonitoringRegistry()

base_metrics = GenericMetricsCollector(
    name="base_metrics"
)

monitoring_registry.register(
    "base_metrics",
    base_metrics,
)


def record_metric(
    name: str,
    value: float,
    robot_id: str = "unknown",
    metric_type: MetricType = MetricType.GAUGE,
    unit: MetricUnit = MetricUnit.NONE,
    component_id: Optional[str] = None,
    device_id: Optional[str] = None,
    model_id: Optional[str] = None,
    environment: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    base_metrics.record(
        MetricRecord(
            name=name,
            value=value,
            metric_type=metric_type,
            unit=unit,
            labels=MetricLabels(
                robot_id=robot_id,
                component_id=component_id,
                device_id=device_id,
                model_id=model_id,
                environment=environment,
            ),
            metadata=metadata or {},
        )
    )


def register_metric_threshold(
    metric_name: str,
    warning_value: float,
    critical_value: float,
    operator: ComparisonOperator = (
        ComparisonOperator.GREATER_EQUAL
    ),
) -> None:
    base_metrics.register_threshold(
        MetricThreshold(
            metric_name=metric_name,
            warning_value=warning_value,
            critical_value=critical_value,
            operator=operator,
        )
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
    )

    register_metric_threshold(
        metric_name="cpu_utilization_percent",
        warning_value=80.0,
        critical_value=95.0,
    )

    register_metric_threshold(
        metric_name="battery_percent",
        warning_value=20.0,
        critical_value=10.0,
        operator=ComparisonOperator.LESS_EQUAL,
    )

    record_metric(
        name="cpu_utilization_percent",
        value=72.5,
        robot_id="ROBOT-001",
        unit=MetricUnit.PERCENT,
    )

    record_metric(
        name="cpu_utilization_percent",
        value=97.0,
        robot_id="ROBOT-002",
        unit=MetricUnit.PERCENT,
    )

    record_metric(
        name="battery_percent",
        value=8.0,
        robot_id="ROBOT-002",
        unit=MetricUnit.PERCENT,
    )

    record_metric(
        name="inference_latency_ms",
        value=85.0,
        robot_id="ROBOT-001",
        unit=MetricUnit.MILLISECONDS,
    )

    print(base_metrics.export_json())

    print(base_metrics.export_prometheus())

    print(
        json.dumps(
            monitoring_registry.get_fleet_metrics(),
            indent=2,
        )
    )
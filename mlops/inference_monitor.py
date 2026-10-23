"""Inference Monitor module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/mlops/inference_monitor.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
MLOps — Inference Monitoring

Features:
    - Real-time inference event tracking
    - Model and deployment version monitoring
    - Prediction latency monitoring
    - Throughput and request-rate analytics
    - Prediction confidence monitoring
    - Error-rate monitoring
    - Model availability monitoring
    - Feature freshness monitoring
    - Input and output validation
    - Prediction distribution monitoring
    - Data drift integration signals
    - Model quality monitoring with delayed labels
    - CPU, GPU and memory monitoring metadata
    - Inference cost tracking
    - Robotics perception monitoring
    - Navigation inference monitoring
    - Reinforcement learning policy monitoring
    - Anomaly and alert detection
    - Configurable alert thresholds
    - Multi-tenant isolation
    - SQLite persistence
    - Thread-safe operations
    - JSON export
    - Synthetic inference simulation
    - Unit tests

Python 3.11+
Standard library only.

Safety:
    Monitoring results are advisory. They must not directly
    override robot emergency stops, certified safety systems,
    collision-avoidance controllers or hardware interlocks.
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
import tempfile
import threading
import unittest
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class InferenceMonitorError(Exception):
    pass


class InferenceValidationError(InferenceMonitorError):
    pass


class InferenceNotFoundError(InferenceMonitorError):
    pass


class InferenceAuthorizationError(InferenceMonitorError):
    pass


class InferenceConflictError(InferenceMonitorError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def ensure_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise InferenceValidationError(
            "Expected datetime"
        )

    if value.tzinfo is None:
        raise InferenceValidationError(
            "Datetime must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def timestamp_text(value: datetime) -> str:
    return ensure_utc(value).isoformat(
        timespec="microseconds"
    )


def parse_timestamp(value: str) -> datetime:
    return ensure_utc(
        datetime.fromisoformat(value)
    )


def validate_identifier(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
    ):
        raise InferenceValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise InferenceValidationError(
            f"{name} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InferenceValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise InferenceValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and number < minimum:
        raise InferenceValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and number > maximum:
        raise InferenceValidationError(
            f"{name} must be <= {maximum}"
        )

    return number


def validate_optional_number(
    value: float | None,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None

    return finite_number(
        value,
        name,
        minimum=minimum,
        maximum=maximum,
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: json_safe(item)
            for key, item in asdict(value).items()
        }

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

    return value


def json_text(value: Any) -> str:
    try:
        return json.dumps(
            json_safe(value),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise InferenceValidationError(
            "Value must be JSON serializable"
        ) from exc


def percentile(
    values: Sequence[float],
    percentage: float,
) -> float | None:
    if not values:
        return None

    if not 0 <= percentage <= 100:
        raise InferenceValidationError(
            "Percentile must be between 0 and 100"
        )

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        percentage / 100
    ) * (len(ordered) - 1)

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


# ============================================================
# ENUMERATIONS
# ============================================================

class InferenceTaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    POSE_ESTIMATION = "pose_estimation"
    DEPTH_ESTIMATION = "depth_estimation"
    NAVIGATION = "navigation"
    MOTION_PLANNING = "motion_planning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    LLM = "llm"
    RAG = "rag"
    MULTIMODAL = "multimodal"
    OTHER = "other"


class InferenceStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    HIGH_LATENCY = "high_latency"
    HIGH_ERROR_RATE = "high_error_rate"
    LOW_CONFIDENCE = "low_confidence"
    LOW_THROUGHPUT = "low_throughput"
    STALE_FEATURES = "stale_features"
    HIGH_CPU = "high_cpu"
    HIGH_GPU = "high_gpu"
    HIGH_MEMORY = "high_memory"
    HIGH_COST = "high_cost"
    LOW_MODEL_QUALITY = "low_model_quality"
    DATA_DRIFT = "data_drift"
    NO_INFERENCE = "no_inference"


class MonitorHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    NO_DATA = "no_data"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class InferenceMonitorConfig:
    database_path: Path = Path(
        "./data/mlops/inference_monitor.sqlite3"
    )

    default_window_seconds: int = 300
    maximum_query_limit: int = 100000
    maximum_batch_size: int = 10000

    sqlite_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "database_path",
            Path(self.database_path).expanduser().resolve(),
        )

        for name in (
            "default_window_seconds",
            "maximum_query_limit",
            "maximum_batch_size",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise InferenceValidationError(
                    f"{name} must be a positive integer"
                )

        finite_number(
            self.sqlite_timeout_seconds,
            "sqlite_timeout_seconds",
            minimum=0.001,
        )


@dataclass(frozen=True)
class InferenceAlertThresholds:
    maximum_p95_latency_ms: float = 250.0
    maximum_error_rate: float = 0.05
    minimum_mean_confidence: float = 0.60

    minimum_throughput_rps: float | None = None

    maximum_feature_age_seconds: float = 300.0

    maximum_cpu_percent: float = 90.0
    maximum_gpu_percent: float = 95.0
    maximum_memory_percent: float = 90.0

    maximum_total_cost_usd: float | None = None
    minimum_accuracy: float | None = None

    minimum_samples: int = 5

    def __post_init__(self) -> None:
        finite_number(
            self.maximum_p95_latency_ms,
            "maximum_p95_latency_ms",
            minimum=0,
        )

        finite_number(
            self.maximum_error_rate,
            "maximum_error_rate",
            minimum=0,
            maximum=1,
        )

        finite_number(
            self.minimum_mean_confidence,
            "minimum_mean_confidence",
            minimum=0,
            maximum=1,
        )

        finite_number(
            self.maximum_feature_age_seconds,
            "maximum_feature_age_seconds",
            minimum=0,
        )

        for name in (
            "maximum_cpu_percent",
            "maximum_gpu_percent",
            "maximum_memory_percent",
        ):
            finite_number(
                getattr(self, name),
                name,
                minimum=0,
                maximum=100,
            )

        validate_optional_number(
            self.minimum_throughput_rps,
            "minimum_throughput_rps",
            minimum=0,
        )

        validate_optional_number(
            self.maximum_total_cost_usd,
            "maximum_total_cost_usd",
            minimum=0,
        )

        validate_optional_number(
            self.minimum_accuracy,
            "minimum_accuracy",
            minimum=0,
            maximum=1,
        )

        if (
            isinstance(self.minimum_samples, bool)
            or not isinstance(self.minimum_samples, int)
            or self.minimum_samples <= 0
        ):
            raise InferenceValidationError(
                "minimum_samples must be positive"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class InferenceAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_write: bool = False
    can_manage: bool = False

    def __post_init__(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.user_id,
            "user_id",
        )


# ============================================================
# MODEL REGISTRATION
# ============================================================

@dataclass(frozen=True)
class MonitoredModel:
    model_id: str
    tenant_id: str

    model_name: str
    model_version: str
    deployment_id: str

    task_type: InferenceTaskType

    registered_at: datetime
    registered_by: str

    thresholds: InferenceAlertThresholds = field(
        default_factory=InferenceAlertThresholds
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# INFERENCE EVENT
# ============================================================

@dataclass(frozen=True)
class InferenceEvent:
    event_id: str
    tenant_id: str
    model_id: str

    request_id: str
    timestamp: datetime

    status: InferenceStatus
    latency_ms: float

    prediction: Any = None
    confidence: float | None = None

    actual_label: Any = None

    input_features: dict[str, Any] = field(
        default_factory=dict
    )

    feature_age_seconds: float | None = None

    cpu_percent: float | None = None
    gpu_percent: float | None = None
    memory_percent: float | None = None

    inference_cost_usd: float = 0.0

    robot_id: str | None = None
    environment_id: str | None = None

    error_code: str | None = None
    error_message: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.event_id,
            "event_id",
        )

        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.model_id,
            "model_id",
        )

        validate_identifier(
            self.request_id,
            "request_id",
        )

        ensure_utc(
            self.timestamp
        )

        finite_number(
            self.latency_ms,
            "latency_ms",
            minimum=0,
        )

        validate_optional_number(
            self.confidence,
            "confidence",
            minimum=0,
            maximum=1,
        )

        validate_optional_number(
            self.feature_age_seconds,
            "feature_age_seconds",
            minimum=0,
        )

        for name in (
            "cpu_percent",
            "gpu_percent",
            "memory_percent",
        ):
            validate_optional_number(
                getattr(self, name),
                name,
                minimum=0,
                maximum=100,
            )

        finite_number(
            self.inference_cost_usd,
            "inference_cost_usd",
            minimum=0,
        )

        json_text(
            {
                "prediction": self.prediction,
                "actual_label": self.actual_label,
                "input_features": self.input_features,
                "metadata": self.metadata,
            }
        )


# ============================================================
# MONITORING RESULTS
# ============================================================

@dataclass(frozen=True)
class InferenceMetrics:
    model_id: str

    window_start: datetime
    window_end: datetime

    total_requests: int
    successful_requests: int
    failed_requests: int

    error_rate: float
    throughput_rps: float

    mean_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None

    minimum_latency_ms: float | None
    maximum_latency_ms: float | None

    mean_confidence: float | None

    mean_feature_age_seconds: float | None
    maximum_feature_age_seconds: float | None

    mean_cpu_percent: float | None
    mean_gpu_percent: float | None
    mean_memory_percent: float | None

    total_inference_cost_usd: float

    labeled_predictions: int
    classification_accuracy: float | None

    last_inference_at: datetime | None

    status_counts: dict[str, int]


@dataclass(frozen=True)
class InferenceAlert:
    alert_id: str
    tenant_id: str
    model_id: str

    alert_type: AlertType
    severity: AlertSeverity

    message: str

    observed_value: float | None
    threshold_value: float | None

    created_at: datetime

    window_start: datetime
    window_end: datetime


@dataclass(frozen=True)
class InferenceHealthReport:
    model_id: str
    health: MonitorHealth

    metrics: InferenceMetrics

    alerts: tuple[InferenceAlert, ...]


@dataclass(frozen=True)
class PredictionDistribution:
    model_id: str
    total_predictions: int

    counts: dict[str, int]
    proportions: dict[str, float]


# ============================================================
# METRIC ANALYZER
# ============================================================

class InferenceMetricsAnalyzer:

    @staticmethod
    def analyze(
        model_id: str,
        events: Sequence[InferenceEvent],
        start_time: datetime,
        end_time: datetime,
    ) -> InferenceMetrics:
        start = ensure_utc(
            start_time
        )

        end = ensure_utc(
            end_time
        )

        if end <= start:
            raise InferenceValidationError(
                "Window end must exceed start"
            )

        total = len(events)

        successful = [
            event
            for event in events
            if event.status == InferenceStatus.SUCCESS
        ]

        success_count = len(successful)
        failed_count = total - success_count

        window_seconds = (
            end - start
        ).total_seconds()

        latencies = [
            event.latency_ms
            for event in events
        ]

        confidences = [
            event.confidence
            for event in successful
            if event.confidence is not None
        ]

        feature_ages = [
            event.feature_age_seconds
            for event in events
            if event.feature_age_seconds is not None
        ]

        def mean_optional(
            values: Sequence[float],
        ) -> float | None:
            return (
                statistics.fmean(values)
                if values
                else None
            )

        cpu_values = [
            event.cpu_percent
            for event in events
            if event.cpu_percent is not None
        ]

        gpu_values = [
            event.gpu_percent
            for event in events
            if event.gpu_percent is not None
        ]

        memory_values = [
            event.memory_percent
            for event in events
            if event.memory_percent is not None
        ]

        labeled = [
            event
            for event in successful
            if event.actual_label is not None
            and event.prediction is not None
        ]

        # Exact-match accuracy is appropriate for categorical
        # predictions, not regression or structured CV outputs.
        accuracy = (
            sum(
                event.prediction == event.actual_label
                for event in labeled
            ) / len(labeled)
            if labeled
            else None
        )

        status_counts = {
            status.value: sum(
                event.status == status
                for event in events
            )
            for status in InferenceStatus
        }

        return InferenceMetrics(
            model_id=model_id,
            window_start=start,
            window_end=end,
            total_requests=total,
            successful_requests=success_count,
            failed_requests=failed_count,
            error_rate=(
                failed_count / total
                if total
                else 0.0
            ),
            throughput_rps=total / window_seconds,
            mean_latency_ms=mean_optional(latencies),
            p50_latency_ms=percentile(latencies, 50),
            p95_latency_ms=percentile(latencies, 95),
            p99_latency_ms=percentile(latencies, 99),
            minimum_latency_ms=(
                min(latencies)
                if latencies
                else None
            ),
            maximum_latency_ms=(
                max(latencies)
                if latencies
                else None
            ),
            mean_confidence=mean_optional(confidences),
            mean_feature_age_seconds=mean_optional(
                feature_ages
            ),
            maximum_feature_age_seconds=(
                max(feature_ages)
                if feature_ages
                else None
            ),
            mean_cpu_percent=mean_optional(cpu_values),
            mean_gpu_percent=mean_optional(gpu_values),
            mean_memory_percent=mean_optional(
                memory_values
            ),
            total_inference_cost_usd=sum(
                event.inference_cost_usd
                for event in events
            ),
            labeled_predictions=len(labeled),
            classification_accuracy=accuracy,
            last_inference_at=(
                max(
                    event.timestamp
                    for event in events
                )
                if events
                else None
            ),
            status_counts=status_counts,
        )


# ============================================================
# ALERT ENGINE
# ============================================================

class InferenceAlertEngine:

    @staticmethod
    def evaluate(
        tenant_id: str,
        model_id: str,
        metrics: InferenceMetrics,
        thresholds: InferenceAlertThresholds,
        *,
        task_type: InferenceTaskType = (
            InferenceTaskType.CLASSIFICATION
        ),
    ) -> tuple[InferenceAlert, ...]:
        alerts = []

        def add(
            alert_type: AlertType,
            severity: AlertSeverity,
            message: str,
            observed: float | None,
            threshold: float | None,
        ) -> None:
            alerts.append(
                InferenceAlert(
                    alert_id=generate_id(),
                    tenant_id=tenant_id,
                    model_id=model_id,
                    alert_type=alert_type,
                    severity=severity,
                    message=message,
                    observed_value=observed,
                    threshold_value=threshold,
                    created_at=utc_now(),
                    window_start=metrics.window_start,
                    window_end=metrics.window_end,
                )
            )

        if metrics.total_requests == 0:
            add(
                AlertType.NO_INFERENCE,
                AlertSeverity.WARNING,
                "No inference events observed in window",
                0.0,
                None,
            )

            return tuple(alerts)

        enough_samples = (
            metrics.total_requests
            >= thresholds.minimum_samples
        )

        if enough_samples:
            if (
                metrics.p95_latency_ms is not None
                and metrics.p95_latency_ms
                > thresholds.maximum_p95_latency_ms
            ):
                add(
                    AlertType.HIGH_LATENCY,
                    AlertSeverity.WARNING,
                    "P95 inference latency exceeds threshold",
                    metrics.p95_latency_ms,
                    thresholds.maximum_p95_latency_ms,
                )

            if (
                metrics.error_rate
                > thresholds.maximum_error_rate
            ):
                add(
                    AlertType.HIGH_ERROR_RATE,
                    AlertSeverity.CRITICAL,
                    "Inference error rate exceeds threshold",
                    metrics.error_rate,
                    thresholds.maximum_error_rate,
                )

            if (
                metrics.mean_confidence is not None
                and metrics.mean_confidence
                < thresholds.minimum_mean_confidence
            ):
                add(
                    AlertType.LOW_CONFIDENCE,
                    AlertSeverity.WARNING,
                    "Mean prediction confidence is low",
                    metrics.mean_confidence,
                    thresholds.minimum_mean_confidence,
                )

        if (
            thresholds.minimum_throughput_rps is not None
            and metrics.throughput_rps
            < thresholds.minimum_throughput_rps
        ):
            add(
                AlertType.LOW_THROUGHPUT,
                AlertSeverity.WARNING,
                "Inference throughput below configured minimum",
                metrics.throughput_rps,
                thresholds.minimum_throughput_rps,
            )

        if (
            metrics.maximum_feature_age_seconds is not None
            and metrics.maximum_feature_age_seconds
            > thresholds.maximum_feature_age_seconds
        ):
            add(
                AlertType.STALE_FEATURES,
                AlertSeverity.WARNING,
                "Feature age exceeds freshness threshold",
                metrics.maximum_feature_age_seconds,
                thresholds.maximum_feature_age_seconds,
            )

        resource_rules = (
            (
                AlertType.HIGH_CPU,
                metrics.mean_cpu_percent,
                thresholds.maximum_cpu_percent,
            ),
            (
                AlertType.HIGH_GPU,
                metrics.mean_gpu_percent,
                thresholds.maximum_gpu_percent,
            ),
            (
                AlertType.HIGH_MEMORY,
                metrics.mean_memory_percent,
                thresholds.maximum_memory_percent,
            ),
        )

        for alert_type, observed, threshold in resource_rules:
            if (
                observed is not None
                and observed > threshold
            ):
                add(
                    alert_type,
                    AlertSeverity.WARNING,
                    f"{alert_type.value} threshold exceeded",
                    observed,
                    threshold,
                )

        if (
            thresholds.maximum_total_cost_usd is not None
            and metrics.total_inference_cost_usd
            > thresholds.maximum_total_cost_usd
        ):
            add(
                AlertType.HIGH_COST,
                AlertSeverity.WARNING,
                "Inference cost exceeds window budget",
                metrics.total_inference_cost_usd,
                thresholds.maximum_total_cost_usd,
            )

        if (
            task_type == InferenceTaskType.CLASSIFICATION
            and thresholds.minimum_accuracy is not None
            and metrics.classification_accuracy is not None
            and metrics.labeled_predictions
            >= thresholds.minimum_samples
            and metrics.classification_accuracy
            < thresholds.minimum_accuracy
        ):
            add(
                AlertType.LOW_MODEL_QUALITY,
                AlertSeverity.CRITICAL,
                "Classification accuracy below threshold",
                metrics.classification_accuracy,
                thresholds.minimum_accuracy,
            )

        return tuple(alerts)


# ============================================================
# INFERENCE MONITOR SERVICE
# ============================================================

class InferenceMonitor:

    def __init__(
        self,
        config: InferenceMonitorConfig | None = None,
    ) -> None:
        self.config = (
            config or InferenceMonitorConfig()
        )

        self._lock = threading.RLock()

        self.config.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._connection = sqlite3.connect(
            str(self.config.database_path),
            timeout=self.config.sqlite_timeout_seconds,
            check_same_thread=False,
        )

        self._connection.row_factory = sqlite3.Row

        self._connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        self._connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        self.metrics_analyzer = (
            InferenceMetricsAnalyzer()
        )

        self.alert_engine = (
            InferenceAlertEngine()
        )

        self._initialize_database()

    def __enter__(self) -> InferenceMonitor:
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # --------------------------------------------------------
    # AUTHORIZATION
    # --------------------------------------------------------

    @staticmethod
    def _authorize(
        access: InferenceAccessContext,
        permission: str,
    ) -> None:
        if not getattr(
            access,
            permission,
            False,
        ):
            raise InferenceAuthorizationError(
                f"Missing permission: {permission}"
            )

    # --------------------------------------------------------
    # DATABASE SCHEMA
    # --------------------------------------------------------

    def _initialize_database(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS monitored_models (
                    model_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    deployment_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    registered_by TEXT NOT NULL,
                    thresholds_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE (
                        tenant_id,
                        model_name,
                        model_version,
                        deployment_id
                    )
                );

                CREATE TABLE IF NOT EXISTS inference_events (
                    event_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    prediction_json TEXT,
                    confidence REAL,
                    actual_label_json TEXT,
                    input_features_json TEXT NOT NULL,
                    feature_age_seconds REAL,
                    cpu_percent REAL,
                    gpu_percent REAL,
                    memory_percent REAL,
                    inference_cost_usd REAL NOT NULL,
                    robot_id TEXT,
                    environment_id TEXT,
                    error_code TEXT,
                    error_message TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(model_id)
                        REFERENCES monitored_models(model_id),
                    UNIQUE (
                        tenant_id,
                        model_id,
                        request_id
                    )
                );

                CREATE INDEX IF NOT EXISTS
                idx_inference_events_window
                ON inference_events (
                    tenant_id,
                    model_id,
                    timestamp
                );

                CREATE TABLE IF NOT EXISTS inference_alerts (
                    alert_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    observed_value REAL,
                    threshold_value REAL,
                    created_at TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS inference_audit (
                    audit_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    model_id TEXT,
                    timestamp TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                """
            )

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        access: InferenceAccessContext,
        action: str,
        *,
        model_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO inference_audit (
                audit_id,
                tenant_id,
                user_id,
                action,
                model_id,
                timestamp,
                details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                generate_id(),
                access.tenant_id,
                access.user_id,
                action,
                model_id,
                timestamp_text(utc_now()),
                json_text(
                    dict(details or {})
                ),
            ),
        )

    # --------------------------------------------------------
    # REGISTER MODEL
    # --------------------------------------------------------

    def register_model(
        self,
        model_name: str,
        model_version: str,
        deployment_id: str,
        task_type: InferenceTaskType,
        access: InferenceAccessContext,
        *,
        thresholds: InferenceAlertThresholds | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MonitoredModel:
        self._authorize(
            access,
            "can_manage",
        )

        validate_identifier(
            model_name,
            "model_name",
        )

        validate_identifier(
            model_version,
            "model_version",
        )

        validate_identifier(
            deployment_id,
            "deployment_id",
        )

        model = MonitoredModel(
            model_id=generate_id(),
            tenant_id=access.tenant_id,
            model_name=model_name,
            model_version=model_version,
            deployment_id=deployment_id,
            task_type=task_type,
            registered_at=utc_now(),
            registered_by=access.user_id,
            thresholds=(
                thresholds
                or InferenceAlertThresholds()
            ),
            metadata=dict(metadata or {}),
        )

        with self._lock, self._connection:
            try:
                self._connection.execute(
                    """
                    INSERT INTO monitored_models (
                        model_id,
                        tenant_id,
                        model_name,
                        model_version,
                        deployment_id,
                        task_type,
                        registered_at,
                        registered_by,
                        thresholds_json,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model.model_id,
                        model.tenant_id,
                        model.model_name,
                        model.model_version,
                        model.deployment_id,
                        model.task_type.value,
                        timestamp_text(
                            model.registered_at
                        ),
                        model.registered_by,
                        json_text(model.thresholds),
                        json_text(model.metadata),
                    ),
                )

            except sqlite3.IntegrityError as exc:
                raise InferenceConflictError(
                    "Model deployment is already registered"
                ) from exc

            self._audit(
                access,
                "model_registered",
                model_id=model.model_id,
            )

        return model

    # --------------------------------------------------------
    # GET MODEL
    # --------------------------------------------------------

    def get_model(
        self,
        model_id: str,
        access: InferenceAccessContext,
    ) -> MonitoredModel:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            row = self._connection.execute(
                """
                SELECT *
                FROM monitored_models
                WHERE tenant_id = ?
                AND model_id = ?
                """,
                (
                    access.tenant_id,
                    model_id,
                ),
            ).fetchone()

            if row is None:
                raise InferenceNotFoundError(
                    "Monitored model not found"
                )

            return MonitoredModel(
                model_id=row["model_id"],
                tenant_id=row["tenant_id"],
                model_name=row["model_name"],
                model_version=row["model_version"],
                deployment_id=row["deployment_id"],
                task_type=InferenceTaskType(
                    row["task_type"]
                ),
                registered_at=parse_timestamp(
                    row["registered_at"]
                ),
                registered_by=row["registered_by"],
                thresholds=InferenceAlertThresholds(
                    **json.loads(
                        row["thresholds_json"]
                    )
                ),
                metadata=json.loads(
                    row["metadata_json"]
                ),
            )

    # --------------------------------------------------------
    # LIST MODELS
    # --------------------------------------------------------

    def list_models(
        self,
        access: InferenceAccessContext,
    ) -> tuple[MonitoredModel, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT model_id
                FROM monitored_models
                WHERE tenant_id = ?
                ORDER BY registered_at
                """,
                (
                    access.tenant_id,
                ),
            ).fetchall()

        return tuple(
            self.get_model(
                row["model_id"],
                access,
            )
            for row in rows
        )

    # --------------------------------------------------------
    # LOG INFERENCE
    # --------------------------------------------------------

    def log_inference(
        self,
        model_id: str,
        request_id: str,
        status: InferenceStatus,
        latency_ms: float,
        access: InferenceAccessContext,
        *,
        timestamp: datetime | None = None,
        prediction: Any = None,
        confidence: float | None = None,
        actual_label: Any = None,
        input_features: Mapping[str, Any] | None = None,
        feature_age_seconds: float | None = None,
        cpu_percent: float | None = None,
        gpu_percent: float | None = None,
        memory_percent: float | None = None,
        inference_cost_usd: float = 0.0,
        robot_id: str | None = None,
        environment_id: str | None = None,
        error_code: str | None = None,
        error_message: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> InferenceEvent:
        self._authorize(
            access,
            "can_write",
        )

        self.get_model(
            model_id,
            InferenceAccessContext(
                tenant_id=access.tenant_id,
                user_id=access.user_id,
                can_read=True,
            ),
        )

        event = InferenceEvent(
            event_id=generate_id(),
            tenant_id=access.tenant_id,
            model_id=model_id,
            request_id=request_id,
            timestamp=(
                ensure_utc(timestamp)
                if timestamp is not None
                else utc_now()
            ),
            status=status,
            latency_ms=latency_ms,
            prediction=prediction,
            confidence=confidence,
            actual_label=actual_label,
            input_features=dict(
                input_features or {}
            ),
            feature_age_seconds=feature_age_seconds,
            cpu_percent=cpu_percent,
            gpu_percent=gpu_percent,
            memory_percent=memory_percent,
            inference_cost_usd=inference_cost_usd,
            robot_id=robot_id,
            environment_id=environment_id,
            error_code=error_code,
            error_message=error_message,
            metadata=dict(metadata or {}),
        )

        with self._lock, self._connection:
            try:
                self._insert_event(
                    event
                )
            except sqlite3.IntegrityError as exc:
                raise InferenceConflictError(
                    "Duplicate inference request"
                ) from exc

            self._audit(
                access,
                "inference_logged",
                model_id=model_id,
                details={
                    "request_id": request_id,
                    "status": status.value,
                },
            )

        return event

    # --------------------------------------------------------
    # INTERNAL EVENT INSERT
    # --------------------------------------------------------

    def _insert_event(
        self,
        event: InferenceEvent,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO inference_events (
                event_id,
                tenant_id,
                model_id,
                request_id,
                timestamp,
                status,
                latency_ms,
                prediction_json,
                confidence,
                actual_label_json,
                input_features_json,
                feature_age_seconds,
                cpu_percent,
                gpu_percent,
                memory_percent,
                inference_cost_usd,
                robot_id,
                environment_id,
                error_code,
                error_message,
                metadata_json
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                event.event_id,
                event.tenant_id,
                event.model_id,
                event.request_id,
                timestamp_text(event.timestamp),
                event.status.value,
                event.latency_ms,
                json_text(event.prediction),
                event.confidence,
                json_text(event.actual_label),
                json_text(event.input_features),
                event.feature_age_seconds,
                event.cpu_percent,
                event.gpu_percent,
                event.memory_percent,
                event.inference_cost_usd,
                event.robot_id,
                event.environment_id,
                event.error_code,
                event.error_message,
                json_text(event.metadata),
            ),
        )

    # --------------------------------------------------------
    # BATCH LOGGING
    # --------------------------------------------------------

    def log_batch(
        self,
        model_id: str,
        events: Sequence[Mapping[str, Any]],
        access: InferenceAccessContext,
    ) -> tuple[InferenceEvent, ...]:
        self._authorize(
            access,
            "can_write",
        )

        if len(events) > self.config.maximum_batch_size:
            raise InferenceValidationError(
                "Batch exceeds configured limit"
            )

        self.get_model(
            model_id,
            InferenceAccessContext(
                tenant_id=access.tenant_id,
                user_id=access.user_id,
                can_read=True,
            ),
        )

        prepared = []

        for item in events:
            event = InferenceEvent(
                event_id=generate_id(),
                tenant_id=access.tenant_id,
                model_id=model_id,
                request_id=item["request_id"],
                timestamp=ensure_utc(
                    item.get(
                        "timestamp",
                        utc_now(),
                    )
                ),
                status=InferenceStatus(
                    item.get(
                        "status",
                        InferenceStatus.SUCCESS,
                    )
                ),
                latency_ms=item["latency_ms"],
                prediction=item.get(
                    "prediction"
                ),
                confidence=item.get(
                    "confidence"
                ),
                actual_label=item.get(
                    "actual_label"
                ),
                input_features=dict(
                    item.get(
                        "input_features",
                        {},
                    )
                ),
                feature_age_seconds=item.get(
                    "feature_age_seconds"
                ),
                cpu_percent=item.get(
                    "cpu_percent"
                ),
                gpu_percent=item.get(
                    "gpu_percent"
                ),
                memory_percent=item.get(
                    "memory_percent"
                ),
                inference_cost_usd=item.get(
                    "inference_cost_usd",
                    0.0,
                ),
                robot_id=item.get(
                    "robot_id"
                ),
                environment_id=item.get(
                    "environment_id"
                ),
                error_code=item.get(
                    "error_code"
                ),
                error_message=item.get(
                    "error_message",
                    "",
                ),
                metadata=dict(
                    item.get(
                        "metadata",
                        {},
                    )
                ),
            )

            prepared.append(
                event
            )

        with self._lock, self._connection:
            try:
                for event in prepared:
                    self._insert_event(
                        event
                    )
            except sqlite3.IntegrityError as exc:
                raise InferenceConflictError(
                    "Duplicate inference request in batch"
                ) from exc

            self._audit(
                access,
                "inference_batch_logged",
                model_id=model_id,
                details={
                    "count": len(prepared),
                },
            )

        return tuple(prepared)

    # --------------------------------------------------------
    # ROW CONVERSION
    # --------------------------------------------------------

    @staticmethod
    def _row_to_event(
        row: sqlite3.Row,
    ) -> InferenceEvent:
        return InferenceEvent(
            event_id=row["event_id"],
            tenant_id=row["tenant_id"],
            model_id=row["model_id"],
            request_id=row["request_id"],
            timestamp=parse_timestamp(
                row["timestamp"]
            ),
            status=InferenceStatus(
                row["status"]
            ),
            latency_ms=row["latency_ms"],
            prediction=json.loads(
                row["prediction_json"]
            ),
            confidence=row["confidence"],
            actual_label=json.loads(
                row["actual_label_json"]
            ),
            input_features=json.loads(
                row["input_features_json"]
            ),
            feature_age_seconds=row[
                "feature_age_seconds"
            ],
            cpu_percent=row["cpu_percent"],
            gpu_percent=row["gpu_percent"],
            memory_percent=row["memory_percent"],
            inference_cost_usd=row[
                "inference_cost_usd"
            ],
            robot_id=row["robot_id"],
            environment_id=row["environment_id"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            metadata=json.loads(
                row["metadata_json"]
            ),
        )

    # --------------------------------------------------------
    # GET EVENTS
    # --------------------------------------------------------

    def get_events(
        self,
        model_id: str,
        access: InferenceAccessContext,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 10000,
    ) -> tuple[InferenceEvent, ...]:
        self._authorize(
            access,
            "can_read",
        )

        self.get_model(
            model_id,
            access,
        )

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
            or limit > self.config.maximum_query_limit
        ):
            raise InferenceValidationError(
                "Invalid query limit"
            )

        query = """
            SELECT *
            FROM inference_events
            WHERE tenant_id = ?
            AND model_id = ?
        """

        parameters: list[Any] = [
            access.tenant_id,
            model_id,
        ]

        if start_time is not None:
            query += " AND timestamp >= ?"

            parameters.append(
                timestamp_text(
                    start_time
                )
            )

        if end_time is not None:
            query += " AND timestamp < ?"

            parameters.append(
                timestamp_text(
                    end_time
                )
            )

        query += """
            ORDER BY timestamp DESC, event_id DESC
            LIMIT ?
        """

        parameters.append(
            limit
        )

        with self._lock:
            rows = self._connection.execute(
                query,
                parameters,
            ).fetchall()

        return tuple(
            self._row_to_event(row)
            for row in rows
        )

    # --------------------------------------------------------
    # MONITORING METRICS
    # --------------------------------------------------------

    def calculate_metrics(
        self,
        model_id: str,
        access: InferenceAccessContext,
        *,
        window_seconds: int | None = None,
        end_time: datetime | None = None,
    ) -> InferenceMetrics:
        self._authorize(
            access,
            "can_read",
        )

        window = (
            window_seconds
            if window_seconds is not None
            else self.config.default_window_seconds
        )

        if (
            isinstance(window, bool)
            or not isinstance(window, int)
            or window <= 0
        ):
            raise InferenceValidationError(
                "window_seconds must be positive"
            )

        end = (
            ensure_utc(end_time)
            if end_time is not None
            else utc_now()
        )

        start = end - timedelta(
            seconds=window
        )

        # Query all events in the requested window.
        # This avoids silently computing incorrect metrics
        # from a truncated sample.
        with self._lock:
            self.get_model(
                model_id,
                access,
            )

            rows = self._connection.execute(
                """
                SELECT *
                FROM inference_events
                WHERE tenant_id = ?
                AND model_id = ?
                AND timestamp >= ?
                AND timestamp < ?
                ORDER BY timestamp ASC
                """,
                (
                    access.tenant_id,
                    model_id,
                    timestamp_text(start),
                    timestamp_text(end),
                ),
            ).fetchall()

        events = [
            self._row_to_event(row)
            for row in rows
        ]

        return self.metrics_analyzer.analyze(
            model_id,
            events,
            start,
            end,
        )

    # --------------------------------------------------------
    # HEALTH REPORT
    # --------------------------------------------------------

    def health_report(
        self,
        model_id: str,
        access: InferenceAccessContext,
        *,
        window_seconds: int | None = None,
        end_time: datetime | None = None,
        persist_alerts: bool = False,
    ) -> InferenceHealthReport:
        self._authorize(
            access,
            "can_read",
        )

        model = self.get_model(
            model_id,
            access,
        )

        metrics = self.calculate_metrics(
            model_id,
            access,
            window_seconds=window_seconds,
            end_time=end_time,
        )

        alerts = self.alert_engine.evaluate(
            access.tenant_id,
            model_id,
            metrics,
            model.thresholds,
            task_type=model.task_type,
        )

        if metrics.total_requests == 0:
            health = MonitorHealth.NO_DATA

        elif any(
            alert.severity == AlertSeverity.CRITICAL
            for alert in alerts
        ):
            health = MonitorHealth.CRITICAL

        elif alerts:
            health = MonitorHealth.DEGRADED

        else:
            health = MonitorHealth.HEALTHY

        if persist_alerts and alerts:
            self._authorize(
                access,
                "can_write",
            )

            with self._lock, self._connection:
                for alert in alerts:
                    self._connection.execute(
                        """
                        INSERT INTO inference_alerts (
                            alert_id,
                            tenant_id,
                            model_id,
                            alert_type,
                            severity,
                            message,
                            observed_value,
                            threshold_value,
                            created_at,
                            window_start,
                            window_end
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            alert.alert_id,
                            alert.tenant_id,
                            alert.model_id,
                            alert.alert_type.value,
                            alert.severity.value,
                            alert.message,
                            alert.observed_value,
                            alert.threshold_value,
                            timestamp_text(
                                alert.created_at
                            ),
                            timestamp_text(
                                alert.window_start
                            ),
                            timestamp_text(
                                alert.window_end
                            ),
                        ),
                    )

        return InferenceHealthReport(
            model_id=model_id,
            health=health,
            metrics=metrics,
            alerts=alerts,
        )

    # --------------------------------------------------------
    # DELAYED GROUND-TRUTH LABEL
    # --------------------------------------------------------

    def attach_ground_truth(
        self,
        model_id: str,
        request_id: str,
        actual_label: Any,
        access: InferenceAccessContext,
    ) -> InferenceEvent:
        self._authorize(
            access,
            "can_write",
        )

        json_text(
            actual_label
        )

        with self._lock, self._connection:
            self.get_model(
                model_id,
                InferenceAccessContext(
                    tenant_id=access.tenant_id,
                    user_id=access.user_id,
                    can_read=True,
                ),
            )

            row = self._connection.execute(
                """
                SELECT *
                FROM inference_events
                WHERE tenant_id = ?
                AND model_id = ?
                AND request_id = ?
                """,
                (
                    access.tenant_id,
                    model_id,
                    request_id,
                ),
            ).fetchone()

            if row is None:
                raise InferenceNotFoundError(
                    "Inference request not found"
                )

            existing = self._row_to_event(
                row
            )

            if existing.actual_label is not None:
                raise InferenceConflictError(
                    "Ground truth already recorded"
                )

            self._connection.execute(
                """
                UPDATE inference_events
                SET actual_label_json = ?
                WHERE tenant_id = ?
                AND model_id = ?
                AND request_id = ?
                """,
                (
                    json_text(actual_label),
                    access.tenant_id,
                    model_id,
                    request_id,
                ),
            )

            self._audit(
                access,
                "ground_truth_attached",
                model_id=model_id,
                details={
                    "request_id": request_id,
                },
            )

            updated = self._connection.execute(
                """
                SELECT *
                FROM inference_events
                WHERE tenant_id = ?
                AND model_id = ?
                AND request_id = ?
                """,
                (
                    access.tenant_id,
                    model_id,
                    request_id,
                ),
            ).fetchone()

            return self._row_to_event(
                updated
            )

    # --------------------------------------------------------
    # PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    def prediction_distribution(
        self,
        model_id: str,
        access: InferenceAccessContext,
        *,
        window_seconds: int = 3600,
        end_time: datetime | None = None,
    ) -> PredictionDistribution:
        self._authorize(
            access,
            "can_read",
        )

        end = (
            ensure_utc(end_time)
            if end_time is not None
            else utc_now()
        )

        if window_seconds <= 0:
            raise InferenceValidationError(
                "window_seconds must be positive"
            )

        start = end - timedelta(
            seconds=window_seconds
        )

        with self._lock:
            self.get_model(
                model_id,
                access,
            )

            rows = self._connection.execute(
                """
                SELECT prediction_json
                FROM inference_events
                WHERE tenant_id = ?
                AND model_id = ?
                AND status = ?
                AND timestamp >= ?
                AND timestamp < ?
                """,
                (
                    access.tenant_id,
                    model_id,
                    InferenceStatus.SUCCESS.value,
                    timestamp_text(start),
                    timestamp_text(end),
                ),
            ).fetchall()

        counts: dict[str, int] = {}

        for row in rows:
            prediction = json.loads(
                row["prediction_json"]
            )

            if prediction is None:
                continue

            key = json_text(
                prediction
            )

            counts[key] = counts.get(
                key,
                0,
            ) + 1

        total = sum(
            counts.values()
        )

        proportions = {
            name: count / total
            for name, count in counts.items()
        } if total else {}

        return PredictionDistribution(
            model_id=model_id,
            total_predictions=total,
            counts=counts,
            proportions=proportions,
        )

    # --------------------------------------------------------
    # DRIFT SIGNALS
    # --------------------------------------------------------

    def drift_signals(
        self,
        model_id: str,
        access: InferenceAccessContext,
        *,
        baseline_start: datetime,
        baseline_end: datetime,
        current_start: datetime,
        current_end: datetime,
    ) -> dict[str, Any]:
        """
        Compute lightweight prediction-distribution drift signals.

        Uses total variation distance between empirical
        categorical distributions.

        This is a drift indicator, not a statistical
        significance test. High-cardinality continuous
        predictions should be binned before comparison.
        """
        self._authorize(
            access,
            "can_read",
        )

        def distribution(
            start: datetime,
            end: datetime,
        ) -> dict[str, float]:
            if ensure_utc(end) <= ensure_utc(start):
                raise InferenceValidationError(
                    "Invalid drift window"
                )

            with self._lock:
                rows = self._connection.execute(
                    """
                    SELECT prediction_json
                    FROM inference_events
                    WHERE tenant_id = ?
                    AND model_id = ?
                    AND status = ?
                    AND timestamp >= ?
                    AND timestamp < ?
                    """,
                    (
                        access.tenant_id,
                        model_id,
                        InferenceStatus.SUCCESS.value,
                        timestamp_text(start),
                        timestamp_text(end),
                    ),
                ).fetchall()

            counts: dict[str, int] = {}

            for row in rows:
                prediction = json.loads(
                    row["prediction_json"]
                )

                if prediction is None:
                    continue

                key = json_text(
                    prediction
                )

                counts[key] = (
                    counts.get(key, 0) + 1
                )

            total = sum(
                counts.values()
            )

            return {
                key: value / total
                for key, value in counts.items()
            } if total else {}

        self.get_model(
            model_id,
            access,
        )

        baseline = distribution(
            baseline_start,
            baseline_end,
        )

        current = distribution(
            current_start,
            current_end,
        )

        if not baseline or not current:
            return {
                "model_id": model_id,
                "drift_available": False,
                "total_variation_distance": None,
                "reason": "Insufficient prediction data",
            }

        categories = (
            set(baseline)
            | set(current)
        )

        distance = 0.5 * sum(
            abs(
                baseline.get(category, 0.0)
                - current.get(category, 0.0)
            )
            for category in categories
        )

        return {
            "model_id": model_id,
            "drift_available": True,
            "total_variation_distance": distance,
            "baseline_distribution": baseline,
            "current_distribution": current,
            "drift_warning": distance > 0.2,
            "warning_threshold": 0.2,
        }

    # --------------------------------------------------------
    # JSON EXPORT
    # --------------------------------------------------------

    def export_report_json(
        self,
        model_id: str,
        access: InferenceAccessContext,
        *,
        window_seconds: int = 300,
        end_time: datetime | None = None,
    ) -> str:
        model = self.get_model(
            model_id,
            access,
        )

        report = self.health_report(
            model_id,
            access,
            window_seconds=window_seconds,
            end_time=end_time,
        )

        return json.dumps(
            json_safe(
                {
                    "model": model,
                    "health_report": report,
                }
            ),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC ROBOTICS INFERENCE GENERATOR
# ============================================================

class SyntheticInferenceFactory:

    @staticmethod
    def robot_perception_events(
        samples: int = 100,
        *,
        start_time: datetime | None = None,
        robot_id: str = "robot-001",
    ) -> list[dict[str, Any]]:
        if samples <= 0:
            raise InferenceValidationError(
                "samples must be positive"
            )

        start = (
            ensure_utc(start_time)
            if start_time is not None
            else utc_now() - timedelta(
                seconds=samples + 1
            )
        )

        classes = (
            "person",
            "forklift",
            "pallet",
            "free_space",
        )

        results = []

        for index in range(samples):
            status = (
                InferenceStatus.ERROR
                if index > 0 and index % 23 == 0
                else InferenceStatus.SUCCESS
            )

            prediction = (
                classes[index % len(classes)]
                if status == InferenceStatus.SUCCESS
                else None
            )

            latency = (
                45.0
                + 12.0 * abs(
                    math.sin(index * 0.13)
                )
            )

            if index % 17 == 0:
                latency += 220.0

            results.append(
                {
                    "request_id": f"robot-inference-{index}",
                    "timestamp": (
                        start
                        + timedelta(
                            seconds=index
                        )
                    ),
                    "status": status,
                    "latency_ms": latency,
                    "prediction": prediction,
                    "confidence": (
                        0.70
                        + 0.25 * abs(
                            math.sin(index * 0.07)
                        )
                        if prediction is not None
                        else None
                    ),
                    "input_features": {
                        "camera_frame_id": index,
                        "robot_speed_mps": (
                            0.8
                            + 0.1 * math.sin(index * 0.1)
                        ),
                    },
                    "feature_age_seconds": (
                        0.1 + index % 10
                    ),
                    "cpu_percent": (
                        45.0 + index % 20
                    ),
                    "gpu_percent": (
                        55.0 + index % 30
                    ),
                    "memory_percent": (
                        60.0 + index % 10
                    ),
                    "inference_cost_usd": 0.0001,
                    "robot_id": robot_id,
                    "environment_id": "warehouse-simulator",
                    "error_code": (
                        "SIMULATED_FAILURE"
                        if status == InferenceStatus.ERROR
                        else None
                    ),
                }
            )

        return results


# ============================================================
# UNIT TESTS
# ============================================================

class TestInferenceMonitor(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

        self.config = InferenceMonitorConfig(
            database_path=(
                Path(self.temporary.name)
                / "inference_monitor.sqlite3"
            )
        )

        self.monitor = InferenceMonitor(
            self.config
        )

        self.access = InferenceAccessContext(
            tenant_id="robotics_lab",
            user_id="mlops_engineer",
            can_read=True,
            can_write=True,
            can_manage=True,
        )

        self.model = self.monitor.register_model(
            model_name="robot-object-detector",
            model_version="v1",
            deployment_id="warehouse-edge-01",
            task_type=InferenceTaskType.OBJECT_DETECTION,
            access=self.access,
        )

    def tearDown(self) -> None:
        self.monitor.close()
        self.temporary.cleanup()

    def test_register_model(self) -> None:
        self.assertEqual(
            self.model.model_version,
            "v1",
        )

    def test_get_model(self) -> None:
        model = self.monitor.get_model(
            self.model.model_id,
            self.access,
        )

        self.assertEqual(
            model.model_name,
            "robot-object-detector",
        )

    def test_log_inference(self) -> None:
        event = self.monitor.log_inference(
            self.model.model_id,
            "request-001",
            InferenceStatus.SUCCESS,
            45.0,
            self.access,
            prediction="person",
            confidence=0.95,
        )

        self.assertEqual(
            event.status,
            InferenceStatus.SUCCESS,
        )

    def test_log_batch(self) -> None:
        events = self.monitor.log_batch(
            self.model.model_id,
            SyntheticInferenceFactory.robot_perception_events(
                samples=10
            ),
            self.access,
        )

        self.assertEqual(
            len(events),
            10,
        )

    def test_duplicate_request(self) -> None:
        self.monitor.log_inference(
            self.model.model_id,
            "request-001",
            InferenceStatus.SUCCESS,
            10.0,
            self.access,
        )

        with self.assertRaises(
            InferenceConflictError
        ):
            self.monitor.log_inference(
                self.model.model_id,
                "request-001",
                InferenceStatus.SUCCESS,
                10.0,
                self.access,
            )

    def test_metrics(self) -> None:
        end = utc_now()

        self.monitor.log_batch(
            self.model.model_id,
            SyntheticInferenceFactory.robot_perception_events(
                samples=20,
                start_time=end - timedelta(
                    seconds=30
                ),
            ),
            self.access,
        )

        metrics = self.monitor.calculate_metrics(
            self.model.model_id,
            self.access,
            window_seconds=60,
            end_time=end,
        )

        self.assertEqual(
            metrics.total_requests,
            20,
        )

    def test_error_rate(self) -> None:
        end = utc_now()

        for index in range(10):
            self.monitor.log_inference(
                self.model.model_id,
                f"request-{index}",
                (
                    InferenceStatus.ERROR
                    if index < 2
                    else InferenceStatus.SUCCESS
                ),
                50.0,
                self.access,
                timestamp=end - timedelta(
                    seconds=10
                ),
            )

        metrics = self.monitor.calculate_metrics(
            self.model.model_id,
            self.access,
            window_seconds=60,
            end_time=end,
        )

        self.assertAlmostEqual(
            metrics.error_rate,
            0.2,
        )

    def test_latency_percentiles(self) -> None:
        self.assertEqual(
            percentile([10, 20, 30], 50),
            20,
        )

        self.assertEqual(
            percentile([10, 20, 30], 100),
            30,
        )

    def test_health_report(self) -> None:
        end = utc_now()

        self.monitor.log_batch(
            self.model.model_id,
            SyntheticInferenceFactory.robot_perception_events(
                samples=20,
                start_time=end - timedelta(
                    seconds=30
                ),
            ),
            self.access,
        )

        report = self.monitor.health_report(
            self.model.model_id,
            self.access,
            window_seconds=60,
            end_time=end,
        )

        self.assertIn(
            report.health,
            tuple(MonitorHealth),
        )

    def test_no_data(self) -> None:
        report = self.monitor.health_report(
            self.model.model_id,
            self.access,
        )

        self.assertEqual(
            report.health,
            MonitorHealth.NO_DATA,
        )

    def test_attach_ground_truth(self) -> None:
        self.monitor.log_inference(
            self.model.model_id,
            "request-label",
            InferenceStatus.SUCCESS,
            20.0,
            self.access,
            prediction="person",
        )

        updated = self.monitor.attach_ground_truth(
            self.model.model_id,
            "request-label",
            "person",
            self.access,
        )

        self.assertEqual(
            updated.actual_label,
            "person",
        )

    def test_prediction_distribution(self) -> None:
        end = utc_now()

        for index in range(10):
            self.monitor.log_inference(
                self.model.model_id,
                f"distribution-{index}",
                InferenceStatus.SUCCESS,
                20.0,
                self.access,
                timestamp=end - timedelta(
                    seconds=10
                ),
                prediction=(
                    "person"
                    if index < 7
                    else "forklift"
                ),
            )

        distribution = (
            self.monitor.prediction_distribution(
                self.model.model_id,
                self.access,
                window_seconds=60,
                end_time=end,
            )
        )

        self.assertEqual(
            distribution.total_predictions,
            10,
        )

    def test_drift_signals(self) -> None:
        now = utc_now()

        baseline_start = now - timedelta(
            hours=2
        )

        baseline_end = now - timedelta(
            hours=1
        )

        current_start = now - timedelta(
            minutes=30
        )

        current_end = now

        for index in range(10):
            self.monitor.log_inference(
                self.model.model_id,
                f"baseline-{index}",
                InferenceStatus.SUCCESS,
                20.0,
                self.access,
                timestamp=baseline_start + timedelta(
                    seconds=index
                ),
                prediction="person",
            )

            self.monitor.log_inference(
                self.model.model_id,
                f"current-{index}",
                InferenceStatus.SUCCESS,
                20.0,
                self.access,
                timestamp=current_start + timedelta(
                    seconds=index
                ),
                prediction="forklift",
            )

        result = self.monitor.drift_signals(
            self.model.model_id,
            self.access,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            current_start=current_start,
            current_end=current_end,
        )

        self.assertTrue(
            result["drift_available"]
        )

        self.assertAlmostEqual(
            result["total_variation_distance"],
            1.0,
        )

    def test_tenant_isolation(self) -> None:
        other = InferenceAccessContext(
            tenant_id="another_tenant",
            user_id="other_user",
            can_read=True,
        )

        with self.assertRaises(
            InferenceNotFoundError
        ):
            self.monitor.get_model(
                self.model.model_id,
                other,
            )

    def test_json_export(self) -> None:
        exported = self.monitor.export_report_json(
            self.model.model_id,
            self.access,
        )

        payload = json.loads(
            exported
        )

        self.assertEqual(
            payload["model"]["model_name"],
            "robot-object-detector",
        )

    def test_invalid_confidence(self) -> None:
        with self.assertRaises(
            InferenceValidationError
        ):
            self.monitor.log_inference(
                self.model.model_id,
                "invalid-confidence",
                InferenceStatus.SUCCESS,
                10.0,
                self.access,
                confidence=1.5,
            )

    def test_high_error_alert(self) -> None:
        end = utc_now()

        for index in range(10):
            self.monitor.log_inference(
                self.model.model_id,
                f"error-{index}",
                InferenceStatus.ERROR,
                50.0,
                self.access,
                timestamp=end - timedelta(
                    seconds=10
                ),
            )

        report = self.monitor.health_report(
            self.model.model_id,
            self.access,
            window_seconds=60,
            end_time=end,
        )

        self.assertEqual(
            report.health,
            MonitorHealth.CRITICAL,
        )

        self.assertTrue(
            any(
                alert.alert_type
                == AlertType.HIGH_ERROR_RATE
                for alert in report.alerts
            )
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        config = InferenceMonitorConfig(
            database_path=(
                Path(temporary)
                / "inference_monitor.sqlite3"
            )
        )

        with InferenceMonitor(config) as monitor:
            access = InferenceAccessContext(
                tenant_id="physical_ai_lab",
                user_id="mlops_engineer",
                can_read=True,
                can_write=True,
                can_manage=True,
            )

            model = monitor.register_model(
                model_name="robot-perception-model",
                model_version="v1.2.0",
                deployment_id="warehouse-edge-gpu-01",
                task_type=InferenceTaskType.OBJECT_DETECTION,
                access=access,
                thresholds=InferenceAlertThresholds(
                    maximum_p95_latency_ms=150.0,
                    maximum_error_rate=0.05,
                    minimum_mean_confidence=0.65,
                    maximum_feature_age_seconds=60.0,
                    maximum_gpu_percent=90.0,
                    minimum_samples=5,
                ),
                metadata={
                    "framework": "pytorch",
                    "runtime": "onnx",
                    "robot_type": "autonomous_mobile_robot",
                    "environment": "warehouse",
                },
            )

            end = utc_now()

            events = (
                SyntheticInferenceFactory.robot_perception_events(
                    samples=100,
                    start_time=end - timedelta(
                        seconds=120
                    ),
                )
            )

            monitor.log_batch(
                model.model_id,
                events,
                access,
            )

            report = monitor.health_report(
                model.model_id,
                access,
                window_seconds=180,
                end_time=end,
                persist_alerts=True,
            )

            distribution = (
                monitor.prediction_distribution(
                    model.model_id,
                    access,
                    window_seconds=180,
                    end_time=end,
                )
            )

            print(
                "\nPHYSICAL AI INFERENCE MONITOR"
            )
            print("=" * 70)

            print(
                json.dumps(
                    json_safe(
                        {
                            "model": {
                                "model_id": model.model_id,
                                "name": model.model_name,
                                "version": model.model_version,
                                "deployment_id": model.deployment_id,
                            },
                            "health": report.health,
                            "metrics": report.metrics,
                            "alerts": report.alerts,
                            "prediction_distribution": distribution,
                            "synthetic_data": True,
                            "physical_robot_control": False,
                        }
                    ),
                    indent=2,
                    allow_nan=False,
                )
            )

    print(
        "\nRUNNING UNIT TESTS"
    )
    print("=" * 70)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestInferenceMonitor
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
"""Model Monitor module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/mlops/model_monitor.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
MLOps — Model Monitoring

Production-oriented reference implementation.

Features:
    - Model registration and monitoring configuration
    - Multi-tenant model monitoring
    - Prediction and inference telemetry ingestion
    - Latency, throughput and error-rate monitoring
    - Prediction confidence monitoring
    - Classification accuracy, precision, recall and F1
    - Regression MAE, MSE, RMSE and R²
    - Delayed ground-truth feedback
    - Input feature drift detection
    - Prediction distribution drift detection
    - PSI and total variation distance
    - Feature freshness monitoring
    - CPU, GPU and memory monitoring
    - Model cost monitoring
    - Alert generation and acknowledgement
    - Model health scoring
    - Baseline management
    - Model version comparisons
    - Robotics model categories
    - SQLite persistence
    - Thread-safe database operations
    - Tenant-scoped access control
    - JSON report export
    - Synthetic monitoring demonstration
    - Unit tests

Requirements:
    Python 3.11+
    Standard library only

Important:
    Monitoring is advisory. This module does not execute
    models, deploy models, or control physical robots.

    Safety-critical robot decisions must be handled by
    independently validated safety systems.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import statistics
import tempfile
import threading
import unittest
import uuid

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class ModelMonitorError(Exception):
    pass


class ModelValidationError(ModelMonitorError):
    pass


class ModelNotFoundError(ModelMonitorError):
    pass


class ModelAuthorizationError(ModelMonitorError):
    pass


class ModelConflictError(ModelMonitorError):
    pass


class ModelDataError(ModelMonitorError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def ensure_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ModelValidationError("Expected datetime")

    if value.tzinfo is None:
        raise ModelValidationError(
            "Datetime must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def timestamp(value: datetime) -> str:
    return ensure_utc(value).isoformat(
        timespec="microseconds"
    )


def parse_timestamp(value: str) -> datetime:
    return ensure_utc(datetime.fromisoformat(value))


def validate_name(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
    ):
        raise ModelValidationError(
            f"Invalid {field_name}"
        )

    return value.strip()


def finite(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ModelValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise ModelValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and result < minimum:
        raise ModelValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and result > maximum:
        raise ModelValidationError(
            f"{name} must be <= {maximum}"
        )

    return result


def optional_finite(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None

    return finite(
        value,
        name,
        minimum=minimum,
        maximum=maximum,
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return timestamp(value)

    if isinstance(value, Path):
        return str(value)

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

    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]

    return value


def json_encode(value: Any) -> str:
    try:
        return json.dumps(
            json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(
            "Data must be JSON serializable"
        ) from exc


def percentile(
    values: Sequence[float],
    p: float,
) -> float | None:
    if not values:
        return None

    finite(p, "percentile", minimum=0, maximum=100)

    ordered = sorted(values)

    position = (len(ordered) - 1) * p / 100

    low = math.floor(position)
    high = math.ceil(position)

    if low == high:
        return ordered[low]

    fraction = position - low

    return (
        ordered[low] * (1 - fraction)
        + ordered[high] * fraction
    )


def mean_or_none(
    values: Sequence[float],
) -> float | None:
    return statistics.fmean(values) if values else None


def categorical_key(value: Any) -> str:
    return json_encode(value)


# ============================================================
# ENUMERATIONS
# ============================================================

class ModelTask(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    POSE_ESTIMATION = "pose_estimation"
    DEPTH_ESTIMATION = "depth_estimation"
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    NAVIGATION = "navigation"
    MOTION_PLANNING = "motion_planning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    WORLD_MODEL = "world_model"
    VISION_LANGUAGE_ACTION = "vision_language_action"
    LLM = "llm"
    RAG = "rag"
    OTHER = "other"


class InferenceOutcome(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    CONFIDENCE = "confidence"
    FEATURE_FRESHNESS = "feature_freshness"
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    COST = "cost"
    QUALITY = "quality"
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    THROUGHPUT = "throughput"


class AlertStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ModelMonitorConfig:
    database_path: Path = Path(
        "./data/mlops/model_monitor.sqlite3"
    )

    default_window_seconds: int = 300
    minimum_samples: int = 10
    minimum_drift_samples: int = 20

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
            "minimum_samples",
            "minimum_drift_samples",
            "maximum_batch_size",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ModelValidationError(
                    f"{name} must be a positive integer"
                )

        finite(
            self.sqlite_timeout_seconds,
            "sqlite_timeout_seconds",
            minimum=0.001,
        )


@dataclass(frozen=True)
class ModelThresholds:
    maximum_p95_latency_ms: float = 250.0
    maximum_error_rate: float = 0.05

    minimum_confidence: float = 0.60
    maximum_feature_age_seconds: float = 300.0

    maximum_cpu_percent: float = 90.0
    maximum_gpu_percent: float = 95.0
    maximum_memory_percent: float = 90.0

    maximum_window_cost_usd: float | None = None
    minimum_throughput_rps: float | None = None

    minimum_classification_accuracy: float | None = None
    maximum_regression_rmse: float | None = None

    maximum_feature_psi: float = 0.20
    maximum_prediction_tvd: float = 0.20

    def __post_init__(self) -> None:
        finite(
            self.maximum_p95_latency_ms,
            "maximum_p95_latency_ms",
            minimum=0,
        )

        finite(
            self.maximum_error_rate,
            "maximum_error_rate",
            minimum=0,
            maximum=1,
        )

        finite(
            self.minimum_confidence,
            "minimum_confidence",
            minimum=0,
            maximum=1,
        )

        finite(
            self.maximum_feature_age_seconds,
            "maximum_feature_age_seconds",
            minimum=0,
        )

        for name in (
            "maximum_cpu_percent",
            "maximum_gpu_percent",
            "maximum_memory_percent",
        ):
            finite(
                getattr(self, name),
                name,
                minimum=0,
                maximum=100,
            )

        for name in (
            "maximum_window_cost_usd",
            "minimum_throughput_rps",
            "maximum_regression_rmse",
        ):
            optional_finite(
                getattr(self, name),
                name,
                minimum=0,
            )

        optional_finite(
            self.minimum_classification_accuracy,
            "minimum_classification_accuracy",
            minimum=0,
            maximum=1,
        )

        finite(
            self.maximum_feature_psi,
            "maximum_feature_psi",
            minimum=0,
        )

        finite(
            self.maximum_prediction_tvd,
            "maximum_prediction_tvd",
            minimum=0,
            maximum=1,
        )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class MonitorAccess:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_write: bool = False
    can_manage: bool = False

    def __post_init__(self) -> None:
        validate_name(self.tenant_id, "tenant_id")
        validate_name(self.user_id, "user_id")


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

    task: ModelTask

    created_at: datetime
    created_by: str

    thresholds: ModelThresholds = field(
        default_factory=ModelThresholds
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# TELEMETRY EVENT
# ============================================================

@dataclass(frozen=True)
class ModelInferenceEvent:
    event_id: str
    tenant_id: str
    model_id: str

    request_id: str
    timestamp: datetime

    outcome: InferenceOutcome
    latency_ms: float

    prediction: Any = None
    confidence: float | None = None
    ground_truth: Any = None

    features: dict[str, Any] = field(
        default_factory=dict
    )

    feature_age_seconds: float | None = None

    cpu_percent: float | None = None
    gpu_percent: float | None = None
    memory_percent: float | None = None

    cost_usd: float = 0.0

    robot_id: str | None = None
    environment_id: str | None = None

    error_code: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "event_id",
            "tenant_id",
            "model_id",
            "request_id",
        ):
            validate_name(getattr(self, name), name)

        ensure_utc(self.timestamp)

        finite(
            self.latency_ms,
            "latency_ms",
            minimum=0,
        )

        optional_finite(
            self.confidence,
            "confidence",
            minimum=0,
            maximum=1,
        )

        optional_finite(
            self.feature_age_seconds,
            "feature_age_seconds",
            minimum=0,
        )

        for name in (
            "cpu_percent",
            "gpu_percent",
            "memory_percent",
        ):
            optional_finite(
                getattr(self, name),
                name,
                minimum=0,
                maximum=100,
            )

        finite(
            self.cost_usd,
            "cost_usd",
            minimum=0,
        )

        json_encode(
            {
                "prediction": self.prediction,
                "ground_truth": self.ground_truth,
                "features": self.features,
                "metadata": self.metadata,
            }
        )


# ============================================================
# METRICS
# ============================================================

@dataclass(frozen=True)
class ClassificationMetrics:
    samples: int
    accuracy: float | None
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None

    confusion_matrix: dict[
        str,
        dict[str, int],
    ]


@dataclass(frozen=True)
class RegressionMetrics:
    samples: int
    mae: float | None
    mse: float | None
    rmse: float | None
    r2: float | None


@dataclass(frozen=True)
class ModelPerformanceMetrics:
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

    mean_confidence: float | None
    mean_feature_age_seconds: float | None
    maximum_feature_age_seconds: float | None

    mean_cpu_percent: float | None
    mean_gpu_percent: float | None
    mean_memory_percent: float | None

    total_cost_usd: float

    classification: ClassificationMetrics | None
    regression: RegressionMetrics | None

    last_inference_at: datetime | None

    outcome_counts: dict[str, int]


# ============================================================
# DRIFT
# ============================================================

@dataclass(frozen=True)
class FeatureDriftResult:
    feature_name: str
    psi: float | None
    drift_detected: bool

    baseline_samples: int
    current_samples: int

    reason: str = ""


@dataclass(frozen=True)
class PredictionDriftResult:
    total_variation_distance: float | None
    drift_detected: bool

    baseline_samples: int
    current_samples: int

    reason: str = ""


@dataclass(frozen=True)
class DriftReport:
    model_id: str

    feature_drift: tuple[FeatureDriftResult, ...]
    prediction_drift: PredictionDriftResult

    drift_detected: bool


# ============================================================
# ALERTS AND HEALTH
# ============================================================

@dataclass(frozen=True)
class ModelAlert:
    alert_id: str
    tenant_id: str
    model_id: str

    category: AlertCategory
    severity: AlertSeverity
    status: AlertStatus

    message: str

    observed_value: float
    threshold_value: float

    created_at: datetime

    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None


@dataclass(frozen=True)
class ModelHealthReport:
    model_id: str
    status: HealthStatus

    health_score: float

    metrics: ModelPerformanceMetrics

    drift: DriftReport | None

    alerts: tuple[ModelAlert, ...]

    generated_at: datetime


# ============================================================
# CLASSIFICATION METRIC CALCULATOR
# ============================================================

class ClassificationEvaluator:

    @staticmethod
    def evaluate(
        predictions: Sequence[Any],
        actuals: Sequence[Any],
    ) -> ClassificationMetrics:
        if len(predictions) != len(actuals):
            raise ModelValidationError(
                "Prediction and actual lengths differ"
            )

        count = len(predictions)

        if count == 0:
            return ClassificationMetrics(
                samples=0,
                accuracy=None,
                macro_precision=None,
                macro_recall=None,
                macro_f1=None,
                confusion_matrix={},
            )

        predicted_keys = [
            categorical_key(value)
            for value in predictions
        ]

        actual_keys = [
            categorical_key(value)
            for value in actuals
        ]

        labels = sorted(
            set(predicted_keys) | set(actual_keys)
        )

        confusion = {
            actual: {
                predicted: 0
                for predicted in labels
            }
            for actual in labels
        }

        for actual, predicted in zip(
            actual_keys,
            predicted_keys,
        ):
            confusion[actual][predicted] += 1

        correct = sum(
            actual == predicted
            for actual, predicted in zip(
                actual_keys,
                predicted_keys,
            )
        )

        precision_values = []
        recall_values = []
        f1_values = []

        for label in labels:
            tp = confusion[label][label]

            fp = sum(
                confusion[actual][label]
                for actual in labels
                if actual != label
            )

            fn = sum(
                confusion[label][predicted]
                for predicted in labels
                if predicted != label
            )

            precision = (
                tp / (tp + fp)
                if tp + fp > 0
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if tp + fn > 0
                else 0.0
            )

            f1 = (
                2 * precision * recall
                / (precision + recall)
                if precision + recall > 0
                else 0.0
            )

            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)

        return ClassificationMetrics(
            samples=count,
            accuracy=correct / count,
            macro_precision=statistics.fmean(
                precision_values
            ),
            macro_recall=statistics.fmean(
                recall_values
            ),
            macro_f1=statistics.fmean(
                f1_values
            ),
            confusion_matrix=confusion,
        )


# ============================================================
# REGRESSION METRIC CALCULATOR
# ============================================================

class RegressionEvaluator:

    @staticmethod
    def evaluate(
        predictions: Sequence[float],
        actuals: Sequence[float],
    ) -> RegressionMetrics:
        if len(predictions) != len(actuals):
            raise ModelValidationError(
                "Prediction and actual lengths differ"
            )

        count = len(predictions)

        if count == 0:
            return RegressionMetrics(
                samples=0,
                mae=None,
                mse=None,
                rmse=None,
                r2=None,
            )

        predicted = [
            finite(value, "prediction")
            for value in predictions
        ]

        actual = [
            finite(value, "actual")
            for value in actuals
        ]

        errors = [
            p - a
            for p, a in zip(predicted, actual)
        ]

        mae = statistics.fmean(
            abs(error)
            for error in errors
        )

        mse = statistics.fmean(
            error**2
            for error in errors
        )

        rmse = math.sqrt(mse)

        actual_mean = statistics.fmean(actual)

        total_variance = sum(
            (value - actual_mean) ** 2
            for value in actual
        )

        residual_variance = sum(
            error**2
            for error in errors
        )

        r2 = (
            1 - residual_variance / total_variance
            if total_variance > 0
            else (
                1.0
                if residual_variance == 0
                else None
            )
        )

        return RegressionMetrics(
            samples=count,
            mae=mae,
            mse=mse,
            rmse=rmse,
            r2=r2,
        )


# ============================================================
# PERFORMANCE ANALYZER
# ============================================================

class ModelPerformanceAnalyzer:

    @staticmethod
    def analyze(
        model: MonitoredModel,
        events: Sequence[ModelInferenceEvent],
        start: datetime,
        end: datetime,
    ) -> ModelPerformanceMetrics:
        start = ensure_utc(start)
        end = ensure_utc(end)

        if end <= start:
            raise ModelValidationError(
                "Invalid monitoring window"
            )

        total = len(events)

        successful = [
            event
            for event in events
            if event.outcome == InferenceOutcome.SUCCESS
        ]

        failed = total - len(successful)

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
            if event.prediction is not None
            and event.ground_truth is not None
        ]

        classification = None
        regression = None

        if model.task == ModelTask.CLASSIFICATION:
            classification = ClassificationEvaluator.evaluate(
                [event.prediction for event in labeled],
                [event.ground_truth for event in labeled],
            )

        elif model.task == ModelTask.REGRESSION:
            regression = RegressionEvaluator.evaluate(
                [event.prediction for event in labeled],
                [event.ground_truth for event in labeled],
            )

        counts = {
            outcome.value: sum(
                event.outcome == outcome
                for event in events
            )
            for outcome in InferenceOutcome
        }

        return ModelPerformanceMetrics(
            model_id=model.model_id,
            window_start=start,
            window_end=end,
            total_requests=total,
            successful_requests=len(successful),
            failed_requests=failed,
            error_rate=failed / total if total else 0.0,
            throughput_rps=(
                total / (end - start).total_seconds()
            ),
            mean_latency_ms=mean_or_none(latencies),
            p50_latency_ms=percentile(latencies, 50),
            p95_latency_ms=percentile(latencies, 95),
            p99_latency_ms=percentile(latencies, 99),
            mean_confidence=mean_or_none(confidences),
            mean_feature_age_seconds=mean_or_none(
                feature_ages
            ),
            maximum_feature_age_seconds=(
                max(feature_ages)
                if feature_ages
                else None
            ),
            mean_cpu_percent=mean_or_none(cpu_values),
            mean_gpu_percent=mean_or_none(gpu_values),
            mean_memory_percent=mean_or_none(
                memory_values
            ),
            total_cost_usd=sum(
                event.cost_usd
                for event in events
            ),
            classification=classification,
            regression=regression,
            last_inference_at=(
                max(event.timestamp for event in events)
                if events
                else None
            ),
            outcome_counts=counts,
        )


# ============================================================
# DRIFT DETECTOR
# ============================================================

class ModelDriftDetector:
    """
    Lightweight empirical drift indicators.

    Numeric PSI uses quantile bins derived from the
    baseline distribution.

    Categorical drift uses total variation distance.

    PSI and TVD are descriptive indicators, not
    calibrated statistical significance tests.
    """

    @staticmethod
    def _numeric_bins(
        baseline: Sequence[float],
        bin_count: int = 10,
    ) -> list[float]:
        ordered = sorted(baseline)

        if not ordered:
            return []

        edges = []

        for index in range(1, bin_count):
            value = percentile(
                ordered,
                index * 100 / bin_count,
            )

            if value is not None:
                edges.append(value)

        return sorted(set(edges))

    @staticmethod
    def _histogram(
        values: Sequence[float],
        edges: Sequence[float],
    ) -> list[int]:
        counts = [0] * (len(edges) + 1)

        for value in values:
            index = 0

            while (
                index < len(edges)
                and value > edges[index]
            ):
                index += 1

            counts[index] += 1

        return counts

    @staticmethod
    def population_stability_index(
        baseline: Sequence[float],
        current: Sequence[float],
        *,
        epsilon: float = 1e-6,
    ) -> float | None:
        if not baseline or not current:
            return None

        baseline_values = [
            finite(value, "baseline value")
            for value in baseline
        ]

        current_values = [
            finite(value, "current value")
            for value in current
        ]

        edges = ModelDriftDetector._numeric_bins(
            baseline_values
        )

        baseline_counts = ModelDriftDetector._histogram(
            baseline_values,
            edges,
        )

        current_counts = ModelDriftDetector._histogram(
            current_values,
            edges,
        )

        baseline_total = sum(baseline_counts)
        current_total = sum(current_counts)

        bins = len(baseline_counts)

        # Additive smoothing preserves normalized
        # probabilities and avoids division by zero.
        baseline_probabilities = [
            (count + epsilon)
            / (baseline_total + epsilon * bins)
            for count in baseline_counts
        ]

        current_probabilities = [
            (count + epsilon)
            / (current_total + epsilon * bins)
            for count in current_counts
        ]

        return sum(
            (current_p - baseline_p)
            * math.log(current_p / baseline_p)
            for baseline_p, current_p in zip(
                baseline_probabilities,
                current_probabilities,
            )
        )

    @staticmethod
    def total_variation_distance(
        baseline: Sequence[Any],
        current: Sequence[Any],
    ) -> float | None:
        if not baseline or not current:
            return None

        baseline_counts = Counter(
            categorical_key(value)
            for value in baseline
        )

        current_counts = Counter(
            categorical_key(value)
            for value in current
        )

        labels = (
            set(baseline_counts)
            | set(current_counts)
        )

        baseline_total = len(baseline)
        current_total = len(current)

        return 0.5 * sum(
            abs(
                baseline_counts[label] / baseline_total
                - current_counts[label] / current_total
            )
            for label in labels
        )

    @staticmethod
    def detect(
        model_id: str,
        baseline_events: Sequence[ModelInferenceEvent],
        current_events: Sequence[ModelInferenceEvent],
        thresholds: ModelThresholds,
        *,
        minimum_samples: int = 20,
    ) -> DriftReport:
        baseline_success = [
            event
            for event in baseline_events
            if event.outcome == InferenceOutcome.SUCCESS
        ]

        current_success = [
            event
            for event in current_events
            if event.outcome == InferenceOutcome.SUCCESS
        ]

        feature_names = sorted(
            {
                name
                for event in (
                    list(baseline_success)
                    + list(current_success)
                )
                for name in event.features
            }
        )

        feature_results = []

        for feature_name in feature_names:
            baseline_values = [
                event.features[feature_name]
                for event in baseline_success
                if feature_name in event.features
                and event.features[feature_name] is not None
            ]

            current_values = [
                event.features[feature_name]
                for event in current_success
                if feature_name in event.features
                and event.features[feature_name] is not None
            ]

            if (
                len(baseline_values) < minimum_samples
                or len(current_values) < minimum_samples
            ):
                feature_results.append(
                    FeatureDriftResult(
                        feature_name=feature_name,
                        psi=None,
                        drift_detected=False,
                        baseline_samples=len(baseline_values),
                        current_samples=len(current_values),
                        reason="Insufficient samples",
                    )
                )
                continue

            numeric = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in (
                    baseline_values + current_values
                )
            )

            if numeric:
                score = (
                    ModelDriftDetector.population_stability_index(
                        baseline_values,
                        current_values,
                    )
                )

                detected = (
                    score is not None
                    and score > thresholds.maximum_feature_psi
                )

                reason = "Numeric PSI"

            else:
                score = (
                    ModelDriftDetector.total_variation_distance(
                        baseline_values,
                        current_values,
                    )
                )

                # Reuse the prediction TVD threshold
                # for categorical feature drift.
                detected = (
                    score is not None
                    and score > thresholds.maximum_prediction_tvd
                )

                reason = "Categorical TVD"

            feature_results.append(
                FeatureDriftResult(
                    feature_name=feature_name,
                    psi=score,
                    drift_detected=detected,
                    baseline_samples=len(baseline_values),
                    current_samples=len(current_values),
                    reason=reason,
                )
            )

        baseline_predictions = [
            event.prediction
            for event in baseline_success
            if event.prediction is not None
        ]

        current_predictions = [
            event.prediction
            for event in current_success
            if event.prediction is not None
        ]

        if (
            len(baseline_predictions) < minimum_samples
            or len(current_predictions) < minimum_samples
        ):
            prediction_result = PredictionDriftResult(
                total_variation_distance=None,
                drift_detected=False,
                baseline_samples=len(baseline_predictions),
                current_samples=len(current_predictions),
                reason="Insufficient samples",
            )

        else:
            distance = (
                ModelDriftDetector.total_variation_distance(
                    baseline_predictions,
                    current_predictions,
                )
            )

            prediction_result = PredictionDriftResult(
                total_variation_distance=distance,
                drift_detected=(
                    distance is not None
                    and distance
                    > thresholds.maximum_prediction_tvd
                ),
                baseline_samples=len(baseline_predictions),
                current_samples=len(current_predictions),
                reason="Categorical prediction TVD",
            )

        detected = (
            any(
                result.drift_detected
                for result in feature_results
            )
            or prediction_result.drift_detected
        )

        return DriftReport(
            model_id=model_id,
            feature_drift=tuple(feature_results),
            prediction_drift=prediction_result,
            drift_detected=detected,
        )


# ============================================================
# ALERT ENGINE
# ============================================================

class ModelAlertEngine:

    @staticmethod
    def evaluate(
        model: MonitoredModel,
        metrics: ModelPerformanceMetrics,
        drift: DriftReport | None,
        *,
        minimum_samples: int,
    ) -> tuple[ModelAlert, ...]:
        thresholds = model.thresholds

        alerts: list[ModelAlert] = []

        def add(
            category: AlertCategory,
            severity: AlertSeverity,
            message: str,
            observed: float,
            threshold: float,
        ) -> None:
            alerts.append(
                ModelAlert(
                    alert_id=new_id(),
                    tenant_id=model.tenant_id,
                    model_id=model.model_id,
                    category=category,
                    severity=severity,
                    status=AlertStatus.OPEN,
                    message=message,
                    observed_value=observed,
                    threshold_value=threshold,
                    created_at=utc_now(),
                )
            )

        if metrics.total_requests >= minimum_samples:
            if (
                metrics.p95_latency_ms is not None
                and metrics.p95_latency_ms
                > thresholds.maximum_p95_latency_ms
            ):
                add(
                    AlertCategory.LATENCY,
                    AlertSeverity.WARNING,
                    "P95 latency exceeds threshold",
                    metrics.p95_latency_ms,
                    thresholds.maximum_p95_latency_ms,
                )

            if (
                metrics.error_rate
                > thresholds.maximum_error_rate
            ):
                add(
                    AlertCategory.ERROR_RATE,
                    AlertSeverity.CRITICAL,
                    "Inference error rate exceeds threshold",
                    metrics.error_rate,
                    thresholds.maximum_error_rate,
                )

            if (
                metrics.mean_confidence is not None
                and metrics.mean_confidence
                < thresholds.minimum_confidence
            ):
                add(
                    AlertCategory.CONFIDENCE,
                    AlertSeverity.WARNING,
                    "Prediction confidence below threshold",
                    metrics.mean_confidence,
                    thresholds.minimum_confidence,
                )

        if (
            metrics.maximum_feature_age_seconds is not None
            and metrics.maximum_feature_age_seconds
            > thresholds.maximum_feature_age_seconds
        ):
            add(
                AlertCategory.FEATURE_FRESHNESS,
                AlertSeverity.WARNING,
                "Input feature freshness threshold exceeded",
                metrics.maximum_feature_age_seconds,
                thresholds.maximum_feature_age_seconds,
            )

        resources = (
            (
                AlertCategory.CPU,
                metrics.mean_cpu_percent,
                thresholds.maximum_cpu_percent,
            ),
            (
                AlertCategory.GPU,
                metrics.mean_gpu_percent,
                thresholds.maximum_gpu_percent,
            ),
            (
                AlertCategory.MEMORY,
                metrics.mean_memory_percent,
                thresholds.maximum_memory_percent,
            ),
        )

        for category, observed, threshold in resources:
            if observed is not None and observed > threshold:
                add(
                    category,
                    AlertSeverity.WARNING,
                    f"{category.value} utilization is high",
                    observed,
                    threshold,
                )

        if (
            thresholds.maximum_window_cost_usd is not None
            and metrics.total_cost_usd
            > thresholds.maximum_window_cost_usd
        ):
            add(
                AlertCategory.COST,
                AlertSeverity.WARNING,
                "Inference cost exceeds window budget",
                metrics.total_cost_usd,
                thresholds.maximum_window_cost_usd,
            )

        if (
            thresholds.minimum_throughput_rps is not None
            and metrics.throughput_rps
            < thresholds.minimum_throughput_rps
        ):
            add(
                AlertCategory.THROUGHPUT,
                AlertSeverity.WARNING,
                "Inference throughput below threshold",
                metrics.throughput_rps,
                thresholds.minimum_throughput_rps,
            )

        classification = metrics.classification

        if (
            classification is not None
            and classification.samples >= minimum_samples
            and classification.accuracy is not None
            and thresholds.minimum_classification_accuracy
            is not None
            and classification.accuracy
            < thresholds.minimum_classification_accuracy
        ):
            add(
                AlertCategory.QUALITY,
                AlertSeverity.CRITICAL,
                "Classification accuracy below threshold",
                classification.accuracy,
                thresholds.minimum_classification_accuracy,
            )

        regression = metrics.regression

        if (
            regression is not None
            and regression.samples >= minimum_samples
            and regression.rmse is not None
            and thresholds.maximum_regression_rmse
            is not None
            and regression.rmse
            > thresholds.maximum_regression_rmse
        ):
            add(
                AlertCategory.QUALITY,
                AlertSeverity.CRITICAL,
                "Regression RMSE exceeds threshold",
                regression.rmse,
                thresholds.maximum_regression_rmse,
            )

        if drift is not None:
            for feature in drift.feature_drift:
                if (
                    feature.drift_detected
                    and feature.psi is not None
                ):
                    is_numeric = (
                        feature.reason == "Numeric PSI"
                    )

                    add(
                        AlertCategory.FEATURE_DRIFT,
                        AlertSeverity.WARNING,
                        (
                            f"Feature drift detected: "
                            f"{feature.feature_name}"
                        ),
                        feature.psi,
                        (
                            thresholds.maximum_feature_psi
                            if is_numeric
                            else thresholds.maximum_prediction_tvd
                        ),
                    )

            prediction = drift.prediction_drift

            if (
                prediction.drift_detected
                and prediction.total_variation_distance
                is not None
            ):
                add(
                    AlertCategory.PREDICTION_DRIFT,
                    AlertSeverity.WARNING,
                    "Prediction distribution drift detected",
                    prediction.total_variation_distance,
                    thresholds.maximum_prediction_tvd,
                )

        return tuple(alerts)


# ============================================================
# HEALTH EVALUATOR
# ============================================================

class ModelHealthEvaluator:

    @staticmethod
    def evaluate(
        metrics: ModelPerformanceMetrics,
        alerts: Sequence[ModelAlert],
        *,
        minimum_samples: int,
    ) -> tuple[HealthStatus, float]:
        if metrics.total_requests < minimum_samples:
            return (
                HealthStatus.INSUFFICIENT_DATA,
                0.0,
            )

        critical_count = sum(
            alert.severity == AlertSeverity.CRITICAL
            for alert in alerts
        )

        warning_count = sum(
            alert.severity == AlertSeverity.WARNING
            for alert in alerts
        )

        score = max(
            0.0,
            100.0
            - 30.0 * critical_count
            - 10.0 * warning_count,
        )

        if critical_count:
            status = HealthStatus.CRITICAL

        elif warning_count:
            status = HealthStatus.DEGRADED

        else:
            status = HealthStatus.HEALTHY

        return status, score


# ============================================================
# MODEL MONITOR SERVICE
# ============================================================

class ModelMonitor:

    def __init__(
        self,
        config: ModelMonitorConfig | None = None,
    ) -> None:
        self.config = config or ModelMonitorConfig()

        self._lock = threading.RLock()

        self.config.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._db = sqlite3.connect(
            str(self.config.database_path),
            timeout=self.config.sqlite_timeout_seconds,
            check_same_thread=False,
        )

        self._db.row_factory = sqlite3.Row

        self._db.execute(
            "PRAGMA foreign_keys = ON"
        )

        self._db.execute(
            "PRAGMA journal_mode = WAL"
        )

        self._initialize_database()

    def __enter__(self) -> ModelMonitor:
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
            self._db.close()

    # --------------------------------------------------------
    # AUTHORIZATION
    # --------------------------------------------------------

    @staticmethod
    def _authorize(
        access: MonitorAccess,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise ModelAuthorizationError(
                f"Missing permission: {permission}"
            )

    @staticmethod
    def _read_access(
        access: MonitorAccess,
    ) -> MonitorAccess:
        return MonitorAccess(
            tenant_id=access.tenant_id,
            user_id=access.user_id,
            can_read=True,
        )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    def _initialize_database(self) -> None:
        with self._lock, self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    deployment_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
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
                    outcome TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    prediction_json TEXT,
                    confidence REAL,
                    ground_truth_json TEXT,
                    features_json TEXT NOT NULL,
                    feature_age_seconds REAL,
                    cpu_percent REAL,
                    gpu_percent REAL,
                    memory_percent REAL,
                    cost_usd REAL NOT NULL,
                    robot_id TEXT,
                    environment_id TEXT,
                    error_code TEXT,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(model_id)
                        REFERENCES models(model_id),
                    UNIQUE (
                        tenant_id,
                        model_id,
                        request_id
                    )
                );

                CREATE INDEX IF NOT EXISTS
                idx_monitor_events
                ON inference_events (
                    tenant_id,
                    model_id,
                    timestamp
                );

                CREATE TABLE IF NOT EXISTS baselines (
                    baseline_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    event_count INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    FOREIGN KEY(model_id)
                        REFERENCES models(model_id)
                );

                CREATE INDEX IF NOT EXISTS
                idx_monitor_baselines
                ON baselines (
                    tenant_id,
                    model_id,
                    created_at
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    observed_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    acknowledged_at TEXT,
                    acknowledged_by TEXT
                );

                CREATE TABLE IF NOT EXISTS audit_log (
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
        access: MonitorAccess,
        action: str,
        *,
        model_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO audit_log (
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
                new_id(),
                access.tenant_id,
                access.user_id,
                action,
                model_id,
                timestamp(utc_now()),
                json_encode(dict(details or {})),
            ),
        )

    # --------------------------------------------------------
    # MODEL REGISTRATION
    # --------------------------------------------------------

    def register_model(
        self,
        model_name: str,
        model_version: str,
        deployment_id: str,
        task: ModelTask,
        access: MonitorAccess,
        *,
        thresholds: ModelThresholds | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MonitoredModel:
        self._authorize(access, "can_manage")

        model = MonitoredModel(
            model_id=new_id(),
            tenant_id=access.tenant_id,
            model_name=validate_name(
                model_name,
                "model_name",
            ),
            model_version=validate_name(
                model_version,
                "model_version",
            ),
            deployment_id=validate_name(
                deployment_id,
                "deployment_id",
            ),
            task=ModelTask(task),
            created_at=utc_now(),
            created_by=access.user_id,
            thresholds=thresholds or ModelThresholds(),
            metadata=json.loads(
                json_encode(dict(metadata or {}))
            ),
        )

        with self._lock, self._db:
            try:
                self._db.execute(
                    """
                    INSERT INTO models (
                        model_id,
                        tenant_id,
                        model_name,
                        model_version,
                        deployment_id,
                        task,
                        created_at,
                        created_by,
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
                        model.task.value,
                        timestamp(model.created_at),
                        model.created_by,
                        json_encode(model.thresholds),
                        json_encode(model.metadata),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ModelConflictError(
                    "Model deployment already registered"
                ) from exc

            self._audit(
                access,
                "model_registered",
                model_id=model.model_id,
            )

        return model

    # --------------------------------------------------------
    # MODEL LOOKUP
    # --------------------------------------------------------

    def get_model(
        self,
        model_id: str,
        access: MonitorAccess,
    ) -> MonitoredModel:
        self._authorize(access, "can_read")

        with self._lock:
            row = self._db.execute(
                """
                SELECT *
                FROM models
                WHERE tenant_id = ?
                AND model_id = ?
                """,
                (
                    access.tenant_id,
                    model_id,
                ),
            ).fetchone()

        if row is None:
            raise ModelNotFoundError(
                "Monitored model not found"
            )

        return MonitoredModel(
            model_id=row["model_id"],
            tenant_id=row["tenant_id"],
            model_name=row["model_name"],
            model_version=row["model_version"],
            deployment_id=row["deployment_id"],
            task=ModelTask(row["task"]),
            created_at=parse_timestamp(
                row["created_at"]
            ),
            created_by=row["created_by"],
            thresholds=ModelThresholds(
                **json.loads(row["thresholds_json"])
            ),
            metadata=json.loads(
                row["metadata_json"]
            ),
        )

    def list_models(
        self,
        access: MonitorAccess,
    ) -> tuple[MonitoredModel, ...]:
        self._authorize(access, "can_read")

        with self._lock:
            rows = self._db.execute(
                """
                SELECT model_id
                FROM models
                WHERE tenant_id = ?
                ORDER BY created_at
                """,
                (access.tenant_id,),
            ).fetchall()

        return tuple(
            self.get_model(
                row["model_id"],
                access,
            )
            for row in rows
        )

    # --------------------------------------------------------
    # EVENT INGESTION
    # --------------------------------------------------------

    def _insert_event(
        self,
        event: ModelInferenceEvent,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO inference_events (
                event_id,
                tenant_id,
                model_id,
                request_id,
                timestamp,
                outcome,
                latency_ms,
                prediction_json,
                confidence,
                ground_truth_json,
                features_json,
                feature_age_seconds,
                cpu_percent,
                gpu_percent,
                memory_percent,
                cost_usd,
                robot_id,
                environment_id,
                error_code,
                metadata_json
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                event.event_id,
                event.tenant_id,
                event.model_id,
                event.request_id,
                timestamp(event.timestamp),
                event.outcome.value,
                event.latency_ms,
                json_encode(event.prediction),
                event.confidence,
                json_encode(event.ground_truth),
                json_encode(event.features),
                event.feature_age_seconds,
                event.cpu_percent,
                event.gpu_percent,
                event.memory_percent,
                event.cost_usd,
                event.robot_id,
                event.environment_id,
                event.error_code,
                json_encode(event.metadata),
            ),
        )

    def log_inference(
        self,
        model_id: str,
        request_id: str,
        outcome: InferenceOutcome,
        latency_ms: float,
        access: MonitorAccess,
        *,
        event_time: datetime | None = None,
        prediction: Any = None,
        confidence: float | None = None,
        ground_truth: Any = None,
        features: Mapping[str, Any] | None = None,
        feature_age_seconds: float | None = None,
        cpu_percent: float | None = None,
        gpu_percent: float | None = None,
        memory_percent: float | None = None,
        cost_usd: float = 0.0,
        robot_id: str | None = None,
        environment_id: str | None = None,
        error_code: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelInferenceEvent:
        self._authorize(access, "can_write")

        self.get_model(
            model_id,
            self._read_access(access),
        )

        event = ModelInferenceEvent(
            event_id=new_id(),
            tenant_id=access.tenant_id,
            model_id=model_id,
            request_id=request_id,
            timestamp=event_time or utc_now(),
            outcome=InferenceOutcome(outcome),
            latency_ms=latency_ms,
            prediction=prediction,
            confidence=confidence,
            ground_truth=ground_truth,
            features=dict(features or {}),
            feature_age_seconds=feature_age_seconds,
            cpu_percent=cpu_percent,
            gpu_percent=gpu_percent,
            memory_percent=memory_percent,
            cost_usd=cost_usd,
            robot_id=robot_id,
            environment_id=environment_id,
            error_code=error_code,
            metadata=dict(metadata or {}),
        )

        with self._lock, self._db:
            try:
                self._insert_event(event)
            except sqlite3.IntegrityError as exc:
                raise ModelConflictError(
                    "Duplicate inference request"
                ) from exc

            self._audit(
                access,
                "inference_logged",
                model_id=model_id,
                details={
                    "request_id": request_id,
                    "outcome": event.outcome.value,
                },
            )

        return event

    def log_batch(
        self,
        model_id: str,
        events: Sequence[Mapping[str, Any]],
        access: MonitorAccess,
    ) -> tuple[ModelInferenceEvent, ...]:
        self._authorize(access, "can_write")

        if len(events) > self.config.maximum_batch_size:
            raise ModelValidationError(
                "Batch exceeds configured maximum"
            )

        self.get_model(
            model_id,
            self._read_access(access),
        )

        prepared = []

        for item in events:
            prepared.append(
                ModelInferenceEvent(
                    event_id=new_id(),
                    tenant_id=access.tenant_id,
                    model_id=model_id,
                    request_id=item["request_id"],
                    timestamp=item.get(
                        "timestamp",
                        utc_now(),
                    ),
                    outcome=InferenceOutcome(
                        item.get(
                            "outcome",
                            InferenceOutcome.SUCCESS,
                        )
                    ),
                    latency_ms=item["latency_ms"],
                    prediction=item.get("prediction"),
                    confidence=item.get("confidence"),
                    ground_truth=item.get("ground_truth"),
                    features=dict(
                        item.get("features", {})
                    ),
                    feature_age_seconds=item.get(
                        "feature_age_seconds"
                    ),
                    cpu_percent=item.get("cpu_percent"),
                    gpu_percent=item.get("gpu_percent"),
                    memory_percent=item.get(
                        "memory_percent"
                    ),
                    cost_usd=item.get("cost_usd", 0.0),
                    robot_id=item.get("robot_id"),
                    environment_id=item.get(
                        "environment_id"
                    ),
                    error_code=item.get("error_code"),
                    metadata=dict(
                        item.get("metadata", {})
                    ),
                )
            )

        with self._lock, self._db:
            try:
                for event in prepared:
                    self._insert_event(event)
            except sqlite3.IntegrityError as exc:
                raise ModelConflictError(
                    "Duplicate inference in batch"
                ) from exc

            self._audit(
                access,
                "inference_batch_logged",
                model_id=model_id,
                details={"count": len(prepared)},
            )

        return tuple(prepared)

    # --------------------------------------------------------
    # EVENT LOOKUP
    # --------------------------------------------------------

    @staticmethod
    def _row_to_event(
        row: sqlite3.Row,
    ) -> ModelInferenceEvent:
        return ModelInferenceEvent(
            event_id=row["event_id"],
            tenant_id=row["tenant_id"],
            model_id=row["model_id"],
            request_id=row["request_id"],
            timestamp=parse_timestamp(
                row["timestamp"]
            ),
            outcome=InferenceOutcome(
                row["outcome"]
            ),
            latency_ms=row["latency_ms"],
            prediction=json.loads(
                row["prediction_json"]
            ),
            confidence=row["confidence"],
            ground_truth=json.loads(
                row["ground_truth_json"]
            ),
            features=json.loads(
                row["features_json"]
            ),
            feature_age_seconds=row[
                "feature_age_seconds"
            ],
            cpu_percent=row["cpu_percent"],
            gpu_percent=row["gpu_percent"],
            memory_percent=row["memory_percent"],
            cost_usd=row["cost_usd"],
            robot_id=row["robot_id"],
            environment_id=row["environment_id"],
            error_code=row["error_code"],
            metadata=json.loads(
                row["metadata_json"]
            ),
        )

    def get_events(
        self,
        model_id: str,
        access: MonitorAccess,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[ModelInferenceEvent, ...]:
        self._authorize(access, "can_read")

        self.get_model(model_id, access)

        start = ensure_utc(start_time)
        end = ensure_utc(end_time)

        if end <= start:
            raise ModelValidationError(
                "Invalid time window"
            )

        with self._lock:
            rows = self._db.execute(
                """
                SELECT *
                FROM inference_events
                WHERE tenant_id = ?
                AND model_id = ?
                AND timestamp >= ?
                AND timestamp < ?
                ORDER BY timestamp, event_id
                """,
                (
                    access.tenant_id,
                    model_id,
                    timestamp(start),
                    timestamp(end),
                ),
            ).fetchall()

        return tuple(
            self._row_to_event(row)
            for row in rows
        )

    # --------------------------------------------------------
    # DELAYED LABEL FEEDBACK
    # --------------------------------------------------------

    def attach_ground_truth(
        self,
        model_id: str,
        request_id: str,
        ground_truth: Any,
        access: MonitorAccess,
    ) -> ModelInferenceEvent:
        self._authorize(access, "can_write")

        if ground_truth is None:
            raise ModelValidationError(
                "Ground truth cannot be None"
            )

        encoded = json_encode(ground_truth)

        self.get_model(
            model_id,
            self._read_access(access),
        )

        with self._lock, self._db:
            row = self._db.execute(
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
                raise ModelNotFoundError(
                    "Inference request not found"
                )

            if json.loads(
                row["ground_truth_json"]
            ) is not None:
                raise ModelConflictError(
                    "Ground truth already exists"
                )

            self._db.execute(
                """
                UPDATE inference_events
                SET ground_truth_json = ?
                WHERE tenant_id = ?
                AND model_id = ?
                AND request_id = ?
                """,
                (
                    encoded,
                    access.tenant_id,
                    model_id,
                    request_id,
                ),
            )

            self._audit(
                access,
                "ground_truth_attached",
                model_id=model_id,
                details={"request_id": request_id},
            )

            updated = self._db.execute(
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

        return self._row_to_event(updated)

    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------

    def calculate_metrics(
        self,
        model_id: str,
        access: MonitorAccess,
        *,
        window_seconds: int | None = None,
        end_time: datetime | None = None,
    ) -> ModelPerformanceMetrics:
        model = self.get_model(
            model_id,
            access,
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
            raise ModelValidationError(
                "window_seconds must be positive"
            )

        end = ensure_utc(
            end_time or utc_now()
        )

        start = end - timedelta(
            seconds=window
        )

        events = self.get_events(
            model_id,
            access,
            start_time=start,
            end_time=end,
        )

        return ModelPerformanceAnalyzer.analyze(
            model,
            events,
            start,
            end,
        )

    # --------------------------------------------------------
    # BASELINE MANAGEMENT
    # --------------------------------------------------------

    def create_baseline(
        self,
        model_id: str,
        access: MonitorAccess,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        self._authorize(access, "can_manage")

        self.get_model(
            model_id,
            self._read_access(access),
        )

        events = self.get_events(
            model_id,
            self._read_access(access),
            start_time=start_time,
            end_time=end_time,
        )

        if (
            len(events)
            < self.config.minimum_drift_samples
        ):
            raise ModelDataError(
                "Insufficient events for baseline"
            )

        baseline_id = new_id()

        # Persist an immutable data snapshot so that
        # later label updates or event changes do not
        # silently modify the historical baseline.
        snapshot = [
            json_safe(event)
            for event in events
        ]

        with self._lock, self._db:
            self._db.execute(
                """
                INSERT INTO baselines (
                    baseline_id,
                    tenant_id,
                    model_id,
                    created_at,
                    start_time,
                    end_time,
                    event_count,
                    snapshot_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    baseline_id,
                    access.tenant_id,
                    model_id,
                    timestamp(utc_now()),
                    timestamp(start_time),
                    timestamp(end_time),
                    len(events),
                    json_encode(snapshot),
                ),
            )

            self._audit(
                access,
                "baseline_created",
                model_id=model_id,
                details={
                    "baseline_id": baseline_id,
                    "event_count": len(events),
                },
            )

        return baseline_id

    def _load_baseline_events(
        self,
        model_id: str,
        baseline_id: str,
        access: MonitorAccess,
    ) -> tuple[ModelInferenceEvent, ...]:
        self.get_model(model_id, access)

        with self._lock:
            row = self._db.execute(
                """
                SELECT snapshot_json
                FROM baselines
                WHERE tenant_id = ?
                AND model_id = ?
                AND baseline_id = ?
                """,
                (
                    access.tenant_id,
                    model_id,
                    baseline_id,
                ),
            ).fetchone()

        if row is None:
            raise ModelNotFoundError(
                "Baseline not found"
            )

        payload = json.loads(
            row["snapshot_json"]
        )

        return tuple(
            ModelInferenceEvent(
                event_id=item["event_id"],
                tenant_id=item["tenant_id"],
                model_id=item["model_id"],
                request_id=item["request_id"],
                timestamp=parse_timestamp(
                    item["timestamp"]
                ),
                outcome=InferenceOutcome(
                    item["outcome"]
                ),
                latency_ms=item["latency_ms"],
                prediction=item["prediction"],
                confidence=item["confidence"],
                ground_truth=item["ground_truth"],
                features=item["features"],
                feature_age_seconds=item[
                    "feature_age_seconds"
                ],
                cpu_percent=item["cpu_percent"],
                gpu_percent=item["gpu_percent"],
                memory_percent=item["memory_percent"],
                cost_usd=item["cost_usd"],
                robot_id=item["robot_id"],
                environment_id=item["environment_id"],
                error_code=item["error_code"],
                metadata=item["metadata"],
            )
            for item in payload
        )

    # --------------------------------------------------------
    # DRIFT REPORT
    # --------------------------------------------------------

    def detect_drift(
        self,
        model_id: str,
        baseline_id: str,
        access: MonitorAccess,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> DriftReport:
        model = self.get_model(
            model_id,
            access,
        )

        baseline = self._load_baseline_events(
            model_id,
            baseline_id,
            access,
        )

        current = self.get_events(
            model_id,
            access,
            start_time=start_time,
            end_time=end_time,
        )

        return ModelDriftDetector.detect(
            model_id,
            baseline,
            current,
            model.thresholds,
            minimum_samples=(
                self.config.minimum_drift_samples
            ),
        )

    # --------------------------------------------------------
    # HEALTH REPORT
    # --------------------------------------------------------

    def health_report(
        self,
        model_id: str,
        access: MonitorAccess,
        *,
        window_seconds: int | None = None,
        end_time: datetime | None = None,
        baseline_id: str | None = None,
        persist_alerts: bool = False,
    ) -> ModelHealthReport:
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

        drift = None

        if baseline_id is not None:
            drift = self.detect_drift(
                model_id,
                baseline_id,
                access,
                start_time=metrics.window_start,
                end_time=metrics.window_end,
            )

        alerts = ModelAlertEngine.evaluate(
            model,
            metrics,
            drift,
            minimum_samples=self.config.minimum_samples,
        )

        status, score = ModelHealthEvaluator.evaluate(
            metrics,
            alerts,
            minimum_samples=self.config.minimum_samples,
        )

        if persist_alerts:
            self._authorize(access, "can_write")

            with self._lock, self._db:
                for alert in alerts:
                    self._db.execute(
                        """
                        INSERT INTO alerts (
                            alert_id,
                            tenant_id,
                            model_id,
                            category,
                            severity,
                            status,
                            message,
                            observed_value,
                            threshold_value,
                            created_at,
                            acknowledged_at,
                            acknowledged_by
                        )
                        VALUES (
                            ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            alert.alert_id,
                            alert.tenant_id,
                            alert.model_id,
                            alert.category.value,
                            alert.severity.value,
                            alert.status.value,
                            alert.message,
                            alert.observed_value,
                            alert.threshold_value,
                            timestamp(alert.created_at),
                            None,
                            None,
                        ),
                    )

                self._audit(
                    access,
                    "health_report_generated",
                    model_id=model_id,
                    details={
                        "health": status.value,
                        "alert_count": len(alerts),
                    },
                )

        return ModelHealthReport(
            model_id=model_id,
            status=status,
            health_score=score,
            metrics=metrics,
            drift=drift,
            alerts=alerts,
            generated_at=utc_now(),
        )

    # --------------------------------------------------------
    # ALERT MANAGEMENT
    # --------------------------------------------------------

    def list_alerts(
        self,
        model_id: str,
        access: MonitorAccess,
        *,
        status: AlertStatus | None = None,
    ) -> tuple[ModelAlert, ...]:
        self.get_model(model_id, access)

        query = """
            SELECT *
            FROM alerts
            WHERE tenant_id = ?
            AND model_id = ?
        """

        parameters: list[Any] = [
            access.tenant_id,
            model_id,
        ]

        if status is not None:
            query += " AND status = ?"
            parameters.append(status.value)

        query += """
            ORDER BY created_at DESC
        """

        with self._lock:
            rows = self._db.execute(
                query,
                parameters,
            ).fetchall()

        return tuple(
            self._row_to_alert(row)
            for row in rows
        )

    @staticmethod
    def _row_to_alert(
        row: sqlite3.Row,
    ) -> ModelAlert:
        return ModelAlert(
            alert_id=row["alert_id"],
            tenant_id=row["tenant_id"],
            model_id=row["model_id"],
            category=AlertCategory(
                row["category"]
            ),
            severity=AlertSeverity(
                row["severity"]
            ),
            status=AlertStatus(
                row["status"]
            ),
            message=row["message"],
            observed_value=row["observed_value"],
            threshold_value=row["threshold_value"],
            created_at=parse_timestamp(
                row["created_at"]
            ),
            acknowledged_at=(
                parse_timestamp(
                    row["acknowledged_at"]
                )
                if row["acknowledged_at"]
                else None
            ),
            acknowledged_by=row[
                "acknowledged_by"
            ],
        )

    def acknowledge_alert(
        self,
        alert_id: str,
        access: MonitorAccess,
    ) -> ModelAlert:
        self._authorize(access, "can_manage")

        with self._lock, self._db:
            row = self._db.execute(
                """
                SELECT *
                FROM alerts
                WHERE tenant_id = ?
                AND alert_id = ?
                """,
                (
                    access.tenant_id,
                    alert_id,
                ),
            ).fetchone()

            if row is None:
                raise ModelNotFoundError(
                    "Alert not found"
                )

            if row["status"] == AlertStatus.ACKNOWLEDGED.value:
                raise ModelConflictError(
                    "Alert already acknowledged"
                )

            self._db.execute(
                """
                UPDATE alerts
                SET status = ?,
                    acknowledged_at = ?,
                    acknowledged_by = ?
                WHERE tenant_id = ?
                AND alert_id = ?
                """,
                (
                    AlertStatus.ACKNOWLEDGED.value,
                    timestamp(utc_now()),
                    access.user_id,
                    access.tenant_id,
                    alert_id,
                ),
            )

            self._audit(
                access,
                "alert_acknowledged",
                model_id=row["model_id"],
                details={"alert_id": alert_id},
            )

            updated = self._db.execute(
                """
                SELECT *
                FROM alerts
                WHERE tenant_id = ?
                AND alert_id = ?
                """,
                (
                    access.tenant_id,
                    alert_id,
                ),
            ).fetchone()

        return self._row_to_alert(updated)

    # --------------------------------------------------------
    # MODEL VERSION COMPARISON
    # --------------------------------------------------------

    def compare_models(
        self,
        left_model_id: str,
        right_model_id: str,
        access: MonitorAccess,
        *,
        window_seconds: int = 3600,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        left = self.get_model(
            left_model_id,
            access,
        )

        right = self.get_model(
            right_model_id,
            access,
        )

        end = end_time or utc_now()

        left_metrics = self.calculate_metrics(
            left_model_id,
            access,
            window_seconds=window_seconds,
            end_time=end,
        )

        right_metrics = self.calculate_metrics(
            right_model_id,
            access,
            window_seconds=window_seconds,
            end_time=end,
        )

        return {
            "left_model": json_safe(left),
            "right_model": json_safe(right),
            "same_task": left.task == right.task,
            "left_metrics": json_safe(left_metrics),
            "right_metrics": json_safe(right_metrics),
            "latency_difference_ms": (
                right_metrics.mean_latency_ms
                - left_metrics.mean_latency_ms
                if (
                    right_metrics.mean_latency_ms is not None
                    and left_metrics.mean_latency_ms is not None
                )
                else None
            ),
            "error_rate_difference": (
                right_metrics.error_rate
                - left_metrics.error_rate
            ),
            "throughput_difference_rps": (
                right_metrics.throughput_rps
                - left_metrics.throughput_rps
            ),
        }

    # --------------------------------------------------------
    # JSON EXPORT
    # --------------------------------------------------------

    def export_report_json(
        self,
        model_id: str,
        access: MonitorAccess,
        *,
        window_seconds: int = 300,
        end_time: datetime | None = None,
        baseline_id: str | None = None,
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
            baseline_id=baseline_id,
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
# SYNTHETIC DATA FACTORY
# ============================================================

class SyntheticModelMonitoringFactory:

    @staticmethod
    def classification_events(
        samples: int,
        *,
        start_time: datetime,
        prefix: str = "inference",
        drift: bool = False,
    ) -> list[dict[str, Any]]:
        rng = random.Random(
            2026 if not drift else 2027
        )

        labels = [
            "person",
            "forklift",
            "pallet",
        ]

        results = []

        for index in range(samples):
            if drift:
                prediction = (
                    "forklift"
                    if index % 10 < 8
                    else "person"
                )

                temperature = rng.gauss(
                    65.0,
                    3.0,
                )

            else:
                prediction = labels[
                    index % len(labels)
                ]

                temperature = rng.gauss(
                    35.0,
                    3.0,
                )

            actual = (
                prediction
                if index % 10 != 0
                else labels[
                    (index + 1) % len(labels)
                ]
            )

            outcome = (
                InferenceOutcome.ERROR
                if index > 0 and index % 31 == 0
                else InferenceOutcome.SUCCESS
            )

            results.append(
                {
                    "request_id": f"{prefix}-{index}",
                    "timestamp": (
                        start_time
                        + timedelta(seconds=index)
                    ),
                    "outcome": outcome,
                    "latency_ms": (
                        35.0
                        + rng.random() * 30.0
                        + (
                            250.0
                            if index % 19 == 0
                            else 0.0
                        )
                    ),
                    "prediction": (
                        prediction
                        if outcome == InferenceOutcome.SUCCESS
                        else None
                    ),
                    "ground_truth": (
                        actual
                        if outcome == InferenceOutcome.SUCCESS
                        else None
                    ),
                    "confidence": (
                        0.70 + rng.random() * 0.25
                        if outcome == InferenceOutcome.SUCCESS
                        else None
                    ),
                    "features": {
                        "motor_temperature_c": temperature,
                        "robot_speed_mps": (
                            0.8 + rng.random() * 0.3
                        ),
                        "environment": "warehouse",
                    },
                    "feature_age_seconds": (
                        rng.random() * 5.0
                    ),
                    "cpu_percent": (
                        40.0 + rng.random() * 20.0
                    ),
                    "gpu_percent": (
                        55.0 + rng.random() * 20.0
                    ),
                    "memory_percent": (
                        45.0 + rng.random() * 15.0
                    ),
                    "cost_usd": 0.0001,
                    "robot_id": "robot-001",
                    "environment_id": "warehouse-sim",
                    "metadata": {
                        "synthetic": True,
                    },
                }
            )

        return results


# ============================================================
# UNIT TESTS
# ============================================================

class TestModelMonitor(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

        self.monitor = ModelMonitor(
            ModelMonitorConfig(
                database_path=(
                    Path(self.temporary.name)
                    / "monitor.sqlite3"
                ),
                minimum_samples=5,
                minimum_drift_samples=10,
            )
        )

        self.access = MonitorAccess(
            tenant_id="robotics_lab",
            user_id="mlops_engineer",
            can_read=True,
            can_write=True,
            can_manage=True,
        )

        self.model = self.monitor.register_model(
            model_name="robot-object-classifier",
            model_version="v1",
            deployment_id="edge-001",
            task=ModelTask.CLASSIFICATION,
            access=self.access,
            thresholds=ModelThresholds(
                minimum_classification_accuracy=0.75,
            ),
        )

    def tearDown(self) -> None:
        self.monitor.close()
        self.temporary.cleanup()

    def test_register_model(self) -> None:
        self.assertEqual(
            self.model.model_version,
            "v1",
        )

    def test_log_inference(self) -> None:
        event = self.monitor.log_inference(
            self.model.model_id,
            "request-001",
            InferenceOutcome.SUCCESS,
            25.0,
            self.access,
            prediction="person",
            confidence=0.95,
        )

        self.assertEqual(
            event.outcome,
            InferenceOutcome.SUCCESS,
        )

    def test_duplicate_request(self) -> None:
        self.monitor.log_inference(
            self.model.model_id,
            "duplicate",
            InferenceOutcome.SUCCESS,
            10.0,
            self.access,
        )

        with self.assertRaises(ModelConflictError):
            self.monitor.log_inference(
                self.model.model_id,
                "duplicate",
                InferenceOutcome.SUCCESS,
                10.0,
                self.access,
            )

    def test_classification_metrics(self) -> None:
        result = ClassificationEvaluator.evaluate(
            ["a", "a", "b", "b"],
            ["a", "b", "b", "b"],
        )

        self.assertAlmostEqual(
            result.accuracy,
            0.75,
        )

    def test_regression_metrics(self) -> None:
        result = RegressionEvaluator.evaluate(
            [2.0, 3.0, 4.0],
            [2.0, 3.0, 5.0],
        )

        self.assertAlmostEqual(
            result.mae,
            1 / 3,
        )

    def test_metrics(self) -> None:
        end = utc_now()

        self.monitor.log_batch(
            self.model.model_id,
            SyntheticModelMonitoringFactory.classification_events(
                20,
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

    def test_ground_truth(self) -> None:
        self.monitor.log_inference(
            self.model.model_id,
            "feedback",
            InferenceOutcome.SUCCESS,
            20.0,
            self.access,
            prediction="person",
        )

        updated = self.monitor.attach_ground_truth(
            self.model.model_id,
            "feedback",
            "person",
            self.access,
        )

        self.assertEqual(
            updated.ground_truth,
            "person",
        )

    def test_numeric_psi(self) -> None:
        baseline = [
            float(index)
            for index in range(100)
        ]

        current = [
            float(index + 100)
            for index in range(100)
        ]

        score = ModelDriftDetector.population_stability_index(
            baseline,
            current,
        )

        self.assertGreater(
            score,
            0.2,
        )

    def test_categorical_tvd(self) -> None:
        score = ModelDriftDetector.total_variation_distance(
            ["a"] * 10,
            ["b"] * 10,
        )

        self.assertAlmostEqual(
            score,
            1.0,
        )

    def test_baseline_and_drift(self) -> None:
        now = utc_now()

        baseline_start = now - timedelta(
            hours=2
        )

        current_start = now - timedelta(
            minutes=10
        )

        self.monitor.log_batch(
            self.model.model_id,
            SyntheticModelMonitoringFactory.classification_events(
                30,
                start_time=baseline_start,
                prefix="baseline",
            ),
            self.access,
        )

        baseline_id = self.monitor.create_baseline(
            self.model.model_id,
            self.access,
            start_time=baseline_start,
            end_time=baseline_start + timedelta(
                minutes=2
            ),
        )

        self.monitor.log_batch(
            self.model.model_id,
            SyntheticModelMonitoringFactory.classification_events(
                30,
                start_time=current_start,
                prefix="current",
                drift=True,
            ),
            self.access,
        )

        report = self.monitor.detect_drift(
            self.model.model_id,
            baseline_id,
            self.access,
            start_time=current_start,
            end_time=now,
        )

        self.assertTrue(
            report.drift_detected
        )

    def test_health_report(self) -> None:
        end = utc_now()

        self.monitor.log_batch(
            self.model.model_id,
            SyntheticModelMonitoringFactory.classification_events(
                20,
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
            report.status,
            tuple(HealthStatus),
        )

    def test_insufficient_data(self) -> None:
        report = self.monitor.health_report(
            self.model.model_id,
            self.access,
        )

        self.assertEqual(
            report.status,
            HealthStatus.INSUFFICIENT_DATA,
        )

    def test_tenant_isolation(self) -> None:
        other = MonitorAccess(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
        )

        with self.assertRaises(ModelNotFoundError):
            self.monitor.get_model(
                self.model.model_id,
                other,
            )

    def test_model_comparison(self) -> None:
        second = self.monitor.register_model(
            model_name="robot-object-classifier",
            model_version="v2",
            deployment_id="edge-002",
            task=ModelTask.CLASSIFICATION,
            access=self.access,
        )

        comparison = self.monitor.compare_models(
            self.model.model_id,
            second.model_id,
            self.access,
        )

        self.assertTrue(
            comparison["same_task"]
        )

    def test_export_report(self) -> None:
        exported = self.monitor.export_report_json(
            self.model.model_id,
            self.access,
        )

        payload = json.loads(exported)

        self.assertEqual(
            payload["model"]["model_name"],
            "robot-object-classifier",
        )

    def test_high_error_alert(self) -> None:
        end = utc_now()

        for index in range(10):
            self.monitor.log_inference(
                self.model.model_id,
                f"error-{index}",
                InferenceOutcome.ERROR,
                50.0,
                self.access,
                event_time=end - timedelta(
                    seconds=10
                ),
            )

        report = self.monitor.health_report(
            self.model.model_id,
            self.access,
            window_seconds=60,
            end_time=end,
            persist_alerts=True,
        )

        self.assertEqual(
            report.status,
            HealthStatus.CRITICAL,
        )

        self.assertTrue(
            any(
                alert.category == AlertCategory.ERROR_RATE
                for alert in report.alerts
            )
        )

    def test_acknowledge_alert(self) -> None:
        end = utc_now()

        for index in range(10):
            self.monitor.log_inference(
                self.model.model_id,
                f"ack-error-{index}",
                InferenceOutcome.ERROR,
                50.0,
                self.access,
                event_time=end - timedelta(
                    seconds=10
                ),
            )

        report = self.monitor.health_report(
            self.model.model_id,
            self.access,
            window_seconds=60,
            end_time=end,
            persist_alerts=True,
        )

        acknowledged = self.monitor.acknowledge_alert(
            report.alerts[0].alert_id,
            self.access,
        )

        self.assertEqual(
            acknowledged.status,
            AlertStatus.ACKNOWLEDGED,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        config = ModelMonitorConfig(
            database_path=(
                Path(temporary)
                / "model_monitor.sqlite3"
            ),
            minimum_samples=10,
            minimum_drift_samples=20,
        )

        with ModelMonitor(config) as monitor:
            access = MonitorAccess(
                tenant_id="physical_ai_lab",
                user_id="mlops_engineer",
                can_read=True,
                can_write=True,
                can_manage=True,
            )

            model = monitor.register_model(
                model_name="robot-perception-classifier",
                model_version="v1.0.0",
                deployment_id="warehouse-edge-01",
                task=ModelTask.CLASSIFICATION,
                access=access,
                thresholds=ModelThresholds(
                    maximum_p95_latency_ms=150.0,
                    maximum_error_rate=0.05,
                    minimum_confidence=0.65,
                    minimum_classification_accuracy=0.80,
                    maximum_feature_psi=0.20,
                    maximum_prediction_tvd=0.20,
                ),
                metadata={
                    "framework": "pytorch",
                    "runtime": "onnx",
                    "robot_type": "mobile_robot",
                    "environment": "warehouse",
                    "synthetic": True,
                },
            )

            now = utc_now()

            baseline_start = now - timedelta(
                hours=2
            )

            baseline_events = (
                SyntheticModelMonitoringFactory.classification_events(
                    100,
                    start_time=baseline_start,
                    prefix="baseline",
                )
            )

            monitor.log_batch(
                model.model_id,
                baseline_events,
                access,
            )

            baseline_id = monitor.create_baseline(
                model.model_id,
                access,
                start_time=baseline_start,
                end_time=baseline_start + timedelta(
                    minutes=3
                ),
            )

            current_start = now - timedelta(
                minutes=3
            )

            current_events = (
                SyntheticModelMonitoringFactory.classification_events(
                    100,
                    start_time=current_start,
                    prefix="current",
                    drift=True,
                )
            )

            monitor.log_batch(
                model.model_id,
                current_events,
                access,
            )

            report = monitor.health_report(
                model.model_id,
                access,
                window_seconds=240,
                end_time=now,
                baseline_id=baseline_id,
                persist_alerts=True,
            )

            print(
                "\nPHYSICAL AI MODEL MONITOR"
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
                            "health": report.status,
                            "health_score": report.health_score,
                            "performance": report.metrics,
                            "drift": report.drift,
                            "alerts": report.alerts,
                            "synthetic_data": True,
                            "robot_actuation_enabled": False,
                        }
                    ),
                    indent=2,
                    allow_nan=False,
                )
            )

    print("\nRUNNING UNIT TESTS")
    print("=" * 70)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestModelMonitor
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
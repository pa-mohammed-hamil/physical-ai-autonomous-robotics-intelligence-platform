# backend/app/mlops/drift_detection.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
ML / AI Model Drift Detection Engine

Features:
    - Data drift detection
    - Feature distribution drift
    - Prediction drift
    - Concept drift
    - Model performance degradation detection
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov (KS) statistic
    - Jensen-Shannon divergence
    - Wasserstein-1 distance
    - Categorical distribution drift
    - Missing-value drift
    - Numerical feature drift
    - Robotics sensor telemetry monitoring
    - Camera / LiDAR embedding drift
    - Model prediction monitoring
    - Baseline dataset management
    - Drift severity classification
    - Alert generation
    - Drift history and trend analysis
    - Multi-tenant access control
    - Thread-safe monitoring service
    - JSON persistence
    - Synthetic datasets
    - Unit tests

Python 3.11+
Standard library only.

Important:
    Statistical drift does not necessarily imply model failure.

    Concept drift requires labeled observations or another
    defensible signal of a changing input-target relationship.

    This module produces monitoring recommendations only.
    It does not automatically retrain, deploy, stop, or
    control physical robots.
"""

from __future__ import annotations

import copy
import json
import math
import os
import statistics
import tempfile
import threading
import unittest
import uuid

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class DriftDetectionError(Exception):
    pass


class DriftValidationError(DriftDetectionError):
    pass


class DriftBaselineNotFoundError(DriftDetectionError):
    pass


class DriftAuthorizationError(DriftDetectionError):
    pass


class DriftConflictError(DriftDetectionError):
    pass


class DriftPersistenceError(DriftDetectionError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def validate_identifier(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
    ):
        raise DriftValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def validate_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise DriftValidationError(
            f"{name} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DriftValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise DriftValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and number < minimum:
        raise DriftValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and number > maximum:
        raise DriftValidationError(
            f"{name} must be <= {maximum}"
        )

    return number


def is_missing(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, float):
        return math.isnan(value)

    return False


def numeric_values(
    values: Sequence[Any],
) -> list[float]:
    result = []

    for value in values:
        if is_missing(value):
            continue

        result.append(
            validate_number(
                value,
                "feature value",
            )
        )

    return result


def safe_mean(values: Sequence[float]) -> float:
    return (
        statistics.fmean(values)
        if values
        else 0.0
    )


def safe_std(values: Sequence[float]) -> float:
    return (
        statistics.pstdev(values)
        if len(values) > 1
        else 0.0
    )


def quantile(
    sorted_values: Sequence[float],
    probability: float,
) -> float:
    if not sorted_values:
        raise DriftValidationError(
            "Cannot calculate quantile of empty data"
        )

    probability = validate_number(
        probability,
        "probability",
        minimum=0,
        maximum=1,
    )

    position = (
        (len(sorted_values) - 1)
        * probability
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(sorted_values[lower])

    weight = position - lower

    return (
        sorted_values[lower] * (1 - weight)
        + sorted_values[upper] * weight
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "__dataclass_fields__"):
        return {
            name: json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(item)
            for item in value
        ]

    return value


def atomic_write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary = tempfile.mkstemp(
        prefix=".drift-",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                json_safe(payload),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )

            stream.flush()
            os.fsync(stream.fileno())

        os.replace(
            temporary,
            path,
        )

    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# ============================================================
# ENUMERATIONS
# ============================================================

class DriftType(str, Enum):
    DATA = "data"
    FEATURE = "feature"
    PREDICTION = "prediction"
    CONCEPT = "concept"
    PERFORMANCE = "performance"
    MISSING_VALUES = "missing_values"


class FeatureType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    EMBEDDING_COMPONENT = "embedding_component"


class DriftMetric(str, Enum):
    PSI = "psi"
    KS = "ks"
    JENSEN_SHANNON = "jensen_shannon"
    WASSERSTEIN = "wasserstein"
    MEAN_SHIFT = "mean_shift"
    MISSING_RATE = "missing_rate"
    ACCURACY_DROP = "accuracy_drop"
    MAE_INCREASE = "mae_increase"


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class DriftAlertStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class ModelTaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DriftDetectionConfig:
    storage_root: Path = Path(
        "./data/mlops/drift_detection"
    )

    minimum_baseline_samples: int = 30
    minimum_monitoring_samples: int = 20

    histogram_bins: int = 10

    psi_threshold: float = 0.20
    ks_threshold: float = 0.20
    js_threshold: float = 0.10

    missing_rate_threshold: float = 0.10

    accuracy_drop_threshold: float = 0.05
    mae_increase_threshold: float = 0.20

    maximum_history_records: int = 10000

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "storage_root",
            Path(self.storage_root).expanduser().resolve(),
        )

        for name in (
            "minimum_baseline_samples",
            "minimum_monitoring_samples",
            "histogram_bins",
            "maximum_history_records",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise DriftValidationError(
                    f"{name} must be a positive integer"
                )

        for name in (
            "psi_threshold",
            "ks_threshold",
            "js_threshold",
            "missing_rate_threshold",
            "accuracy_drop_threshold",
            "mae_increase_threshold",
        ):
            validate_number(
                getattr(self, name),
                name,
                minimum=0.000001,
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    feature_type: FeatureType

    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        validate_identifier(
            self.name,
            "feature name",
        )


@dataclass(frozen=True)
class DriftBaseline:
    baseline_id: str

    tenant_id: str
    model_id: str
    model_version: str

    features: tuple[FeatureDefinition, ...]

    reference_data: dict[str, tuple[Any, ...]]

    created_at: datetime
    created_by: str

    dataset_id: str | None = None
    dataset_version_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def sample_count(self) -> int:
        if not self.reference_data:
            return 0

        return len(
            next(iter(self.reference_data.values()))
        )


@dataclass(frozen=True)
class DriftMetricResult:
    metric: DriftMetric

    value: float
    threshold: float

    drift_detected: bool

    severity: DriftSeverity


@dataclass(frozen=True)
class FeatureDriftResult:
    feature_name: str
    feature_type: FeatureType

    drift_detected: bool
    severity: DriftSeverity

    metrics: tuple[DriftMetricResult, ...]

    baseline_sample_count: int
    current_sample_count: int

    baseline_missing_rate: float
    current_missing_rate: float

    baseline_mean: float | None = None
    current_mean: float | None = None

    baseline_std: float | None = None
    current_std: float | None = None


@dataclass(frozen=True)
class PerformanceMetrics:
    task_type: ModelTaskType

    sample_count: int

    accuracy: float | None = None
    mae: float | None = None
    rmse: float | None = None


@dataclass(frozen=True)
class PerformanceDriftResult:
    task_type: ModelTaskType

    baseline_metrics: PerformanceMetrics
    current_metrics: PerformanceMetrics

    drift_detected: bool
    severity: DriftSeverity

    metric_name: str
    degradation: float
    threshold: float


@dataclass(frozen=True)
class DriftReport:
    report_id: str

    tenant_id: str
    model_id: str
    model_version: str

    baseline_id: str

    created_at: datetime

    sample_count: int

    drift_detected: bool
    overall_severity: DriftSeverity

    feature_results: tuple[FeatureDriftResult, ...]

    performance_result: (
        PerformanceDriftResult | None
    ) = None

    recommendations: tuple[str, ...] = ()

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


@dataclass
class DriftAlert:
    alert_id: str
    report_id: str

    tenant_id: str
    model_id: str

    severity: DriftSeverity
    message: str

    created_at: datetime

    status: DriftAlertStatus = (
        DriftAlertStatus.OPEN
    )

    acknowledged_by: str | None = None
    resolved_by: str | None = None


@dataclass(frozen=True)
class DriftAccessContext:
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
# STATISTICAL DRIFT METRICS
# ============================================================

class DriftStatistics:

    EPSILON = 1e-8

    @staticmethod
    def missing_rate(
        values: Sequence[Any],
    ) -> float:
        if not values:
            return 0.0

        return sum(
            is_missing(value)
            for value in values
        ) / len(values)

    @staticmethod
    def histogram_edges(
        reference: Sequence[float],
        bins: int,
    ) -> list[float]:
        if not reference:
            raise DriftValidationError(
                "Reference values cannot be empty"
            )

        ordered = sorted(reference)

        edges = [
            quantile(
                ordered,
                index / bins,
            )
            for index in range(1, bins)
        ]

        return sorted(set(edges))

    @staticmethod
    def histogram_probabilities(
        values: Sequence[float],
        edges: Sequence[float],
    ) -> list[float]:
        counts = [
            0
            for _ in range(len(edges) + 1)
        ]

        for value in values:
            index = 0

            while (
                index < len(edges)
                and value > edges[index]
            ):
                index += 1

            counts[index] += 1

        total = len(values)

        if total == 0:
            return [
                0.0
                for _ in counts
            ]

        return [
            count / total
            for count in counts
        ]

    @classmethod
    def population_stability_index(
        cls,
        reference: Sequence[float],
        current: Sequence[float],
        bins: int = 10,
    ) -> float:
        if not reference or not current:
            raise DriftValidationError(
                "PSI requires nonempty samples"
            )

        edges = cls.histogram_edges(
            reference,
            bins,
        )

        expected = cls.histogram_probabilities(
            reference,
            edges,
        )

        actual = cls.histogram_probabilities(
            current,
            edges,
        )

        psi = 0.0

        for expected_probability, actual_probability in zip(
            expected,
            actual,
        ):
            expected_probability = max(
                expected_probability,
                cls.EPSILON,
            )

            actual_probability = max(
                actual_probability,
                cls.EPSILON,
            )

            psi += (
                actual_probability - expected_probability
            ) * math.log(
                actual_probability / expected_probability
            )

        return max(0.0, psi)

    @staticmethod
    def kolmogorov_smirnov(
        reference: Sequence[float],
        current: Sequence[float],
    ) -> float:
        if not reference or not current:
            raise DriftValidationError(
                "KS requires nonempty samples"
            )

        reference_sorted = sorted(reference)
        current_sorted = sorted(current)

        n = len(reference_sorted)
        m = len(current_sorted)

        i = 0
        j = 0

        maximum_difference = 0.0

        while i < n or j < m:
            if i == n:
                value = current_sorted[j]

            elif j == m:
                value = reference_sorted[i]

            else:
                value = min(
                    reference_sorted[i],
                    current_sorted[j],
                )

            while (
                i < n
                and reference_sorted[i] <= value
            ):
                i += 1

            while (
                j < m
                and current_sorted[j] <= value
            ):
                j += 1

            difference = abs(
                i / n - j / m
            )

            maximum_difference = max(
                maximum_difference,
                difference,
            )

        return maximum_difference

    @staticmethod
    def _kl_divergence(
        p: Sequence[float],
        q: Sequence[float],
    ) -> float:
        result = 0.0

        for a, b in zip(p, q):
            if a > 0:
                result += a * math.log2(
                    a / max(b, 1e-12)
                )

        return result

    @classmethod
    def jensen_shannon_divergence(
        cls,
        p: Sequence[float],
        q: Sequence[float],
    ) -> float:
        if len(p) != len(q) or not p:
            raise DriftValidationError(
                "Probability distributions must have matching sizes"
            )

        if (
            any(value < 0 for value in p)
            or any(value < 0 for value in q)
        ):
            raise DriftValidationError(
                "Probabilities cannot be negative"
            )

        p_total = sum(p)
        q_total = sum(q)

        if p_total <= 0 or q_total <= 0:
            raise DriftValidationError(
                "Probability distributions must have positive mass"
            )

        p_normalized = [
            value / p_total
            for value in p
        ]

        q_normalized = [
            value / q_total
            for value in q
        ]

        midpoint = [
            (a + b) / 2
            for a, b in zip(
                p_normalized,
                q_normalized,
            )
        ]

        result = 0.5 * (
            cls._kl_divergence(
                p_normalized,
                midpoint,
            )
            + cls._kl_divergence(
                q_normalized,
                midpoint,
            )
        )

        return max(
            0.0,
            min(1.0, result),
        )

    @classmethod
    def categorical_js(
        cls,
        reference: Sequence[Any],
        current: Sequence[Any],
    ) -> float:
        if not reference or not current:
            raise DriftValidationError(
                "Categorical JS requires nonempty samples"
            )

        reference_counts = Counter(
            str(value)
            for value in reference
        )

        current_counts = Counter(
            str(value)
            for value in current
        )

        categories = sorted(
            set(reference_counts)
            | set(current_counts)
        )

        p = [
            reference_counts.get(category, 0)
            for category in categories
        ]

        q = [
            current_counts.get(category, 0)
            for category in categories
        ]

        return cls.jensen_shannon_divergence(
            p,
            q,
        )

    @staticmethod
    def wasserstein_1d(
        reference: Sequence[float],
        current: Sequence[float],
    ) -> float:
        if not reference or not current:
            raise DriftValidationError(
                "Wasserstein distance requires nonempty samples"
            )

        x = sorted(reference)
        y = sorted(current)

        points = sorted(
            set(x) | set(y)
        )

        if len(points) == 1:
            return 0.0

        i = 0
        j = 0

        distance = 0.0

        for index in range(len(points) - 1):
            value = points[index]
            next_value = points[index + 1]

            while i < len(x) and x[i] <= value:
                i += 1

            while j < len(y) and y[j] <= value:
                j += 1

            cdf_difference = abs(
                i / len(x) - j / len(y)
            )

            distance += (
                cdf_difference
                * (next_value - value)
            )

        return distance

    @staticmethod
    def standardized_mean_shift(
        reference: Sequence[float],
        current: Sequence[float],
    ) -> float:
        if not reference or not current:
            raise DriftValidationError(
                "Mean shift requires nonempty samples"
            )

        reference_mean = safe_mean(reference)
        current_mean = safe_mean(current)

        reference_std = safe_std(reference)

        scale = max(
            reference_std,
            1e-8,
        )

        return abs(
            current_mean - reference_mean
        ) / scale


# ============================================================
# SEVERITY CLASSIFIER
# ============================================================

class DriftSeverityClassifier:

    ORDER = {
        DriftSeverity.NONE: 0,
        DriftSeverity.LOW: 1,
        DriftSeverity.MODERATE: 2,
        DriftSeverity.HIGH: 3,
        DriftSeverity.CRITICAL: 4,
    }

    @classmethod
    def classify(
        cls,
        value: float,
        threshold: float,
    ) -> DriftSeverity:
        if threshold <= 0:
            raise DriftValidationError(
                "Drift threshold must be positive"
            )

        ratio = value / threshold

        if ratio < 0.5:
            return DriftSeverity.NONE

        if ratio < 1.0:
            return DriftSeverity.LOW

        if ratio < 1.5:
            return DriftSeverity.MODERATE

        if ratio < 2.5:
            return DriftSeverity.HIGH

        return DriftSeverity.CRITICAL

    @classmethod
    def maximum(
        cls,
        severities: Sequence[DriftSeverity],
    ) -> DriftSeverity:
        if not severities:
            return DriftSeverity.NONE

        return max(
            severities,
            key=lambda severity: cls.ORDER[severity],
        )


# ============================================================
# FEATURE DRIFT DETECTOR
# ============================================================

class FeatureDriftDetector:

    def __init__(
        self,
        config: DriftDetectionConfig,
    ) -> None:
        self.config = config

        self.statistics = DriftStatistics()
        self.severity_classifier = (
            DriftSeverityClassifier()
        )

    def _metric_result(
        self,
        metric: DriftMetric,
        value: float,
        threshold: float,
    ) -> DriftMetricResult:
        return DriftMetricResult(
            metric=metric,
            value=value,
            threshold=threshold,
            drift_detected=(
                value >= threshold
            ),
            severity=(
                self.severity_classifier.classify(
                    value,
                    threshold,
                )
            ),
        )

    def detect(
        self,
        feature: FeatureDefinition,
        reference: Sequence[Any],
        current: Sequence[Any],
    ) -> FeatureDriftResult:
        if (
            len(reference)
            < self.config.minimum_baseline_samples
        ):
            raise DriftValidationError(
                "Insufficient baseline samples"
            )

        if (
            len(current)
            < self.config.minimum_monitoring_samples
        ):
            raise DriftValidationError(
                "Insufficient monitoring samples"
            )

        baseline_missing = (
            self.statistics.missing_rate(
                reference
            )
        )

        current_missing = (
            self.statistics.missing_rate(
                current
            )
        )

        missing_shift = abs(
            current_missing - baseline_missing
        )

        metrics = [
            self._metric_result(
                DriftMetric.MISSING_RATE,
                missing_shift,
                self.config.missing_rate_threshold,
            )
        ]

        baseline_mean = None
        current_mean = None

        baseline_std = None
        current_std = None

        reference_clean = [
            value
            for value in reference
            if not is_missing(value)
        ]

        current_clean = [
            value
            for value in current
            if not is_missing(value)
        ]

        if (
            feature.feature_type
            in (
                FeatureType.NUMERICAL,
                FeatureType.EMBEDDING_COMPONENT,
            )
        ):
            reference_numeric = numeric_values(
                reference_clean
            )

            current_numeric = numeric_values(
                current_clean
            )

            if reference_numeric and current_numeric:
                psi = (
                    self.statistics.population_stability_index(
                        reference_numeric,
                        current_numeric,
                        self.config.histogram_bins,
                    )
                )

                ks = (
                    self.statistics.kolmogorov_smirnov(
                        reference_numeric,
                        current_numeric,
                    )
                )

                metrics.extend(
                    [
                        self._metric_result(
                            DriftMetric.PSI,
                            psi,
                            self.config.psi_threshold,
                        ),
                        self._metric_result(
                            DriftMetric.KS,
                            ks,
                            self.config.ks_threshold,
                        ),
                    ]
                )

                baseline_mean = safe_mean(
                    reference_numeric
                )

                current_mean = safe_mean(
                    current_numeric
                )

                baseline_std = safe_std(
                    reference_numeric
                )

                current_std = safe_std(
                    current_numeric
                )

        else:
            if reference_clean and current_clean:
                js = (
                    self.statistics.categorical_js(
                        reference_clean,
                        current_clean,
                    )
                )

                metrics.append(
                    self._metric_result(
                        DriftMetric.JENSEN_SHANNON,
                        js,
                        self.config.js_threshold,
                    )
                )

        detected = any(
            metric.drift_detected
            for metric in metrics
        )

        severity = (
            self.severity_classifier.maximum(
                [
                    metric.severity
                    for metric in metrics
                ]
            )
        )

        return FeatureDriftResult(
            feature_name=feature.name,
            feature_type=feature.feature_type,
            drift_detected=detected,
            severity=severity,
            metrics=tuple(metrics),
            baseline_sample_count=len(reference),
            current_sample_count=len(current),
            baseline_missing_rate=baseline_missing,
            current_missing_rate=current_missing,
            baseline_mean=baseline_mean,
            current_mean=current_mean,
            baseline_std=baseline_std,
            current_std=current_std,
        )


# ============================================================
# MODEL PERFORMANCE MONITOR
# ============================================================

class ModelPerformanceMonitor:

    def __init__(
        self,
        config: DriftDetectionConfig,
    ) -> None:
        self.config = config

    @staticmethod
    def classification_metrics(
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
    ) -> PerformanceMetrics:
        if not y_true or len(y_true) != len(y_pred):
            raise DriftValidationError(
                "Classification labels must have equal nonzero lengths"
            )

        correct = sum(
            truth == prediction
            for truth, prediction in zip(
                y_true,
                y_pred,
            )
        )

        return PerformanceMetrics(
            task_type=ModelTaskType.CLASSIFICATION,
            sample_count=len(y_true),
            accuracy=correct / len(y_true),
        )

    @staticmethod
    def regression_metrics(
        y_true: Sequence[float],
        y_pred: Sequence[float],
    ) -> PerformanceMetrics:
        if not y_true or len(y_true) != len(y_pred):
            raise DriftValidationError(
                "Regression labels must have equal nonzero lengths"
            )

        truth = numeric_values(
            y_true
        )

        prediction = numeric_values(
            y_pred
        )

        if (
            len(truth) != len(y_true)
            or len(prediction) != len(y_pred)
        ):
            raise DriftValidationError(
                "Regression labels cannot contain missing values"
            )

        errors = [
            actual - predicted
            for actual, predicted in zip(
                truth,
                prediction,
            )
        ]

        mae = safe_mean(
            [abs(error) for error in errors]
        )

        rmse = math.sqrt(
            safe_mean(
                [error**2 for error in errors]
            )
        )

        return PerformanceMetrics(
            task_type=ModelTaskType.REGRESSION,
            sample_count=len(truth),
            mae=mae,
            rmse=rmse,
        )

    def detect(
        self,
        baseline: PerformanceMetrics,
        current: PerformanceMetrics,
    ) -> PerformanceDriftResult:
        if baseline.task_type != current.task_type:
            raise DriftValidationError(
                "Model task types must match"
            )

        if baseline.task_type == ModelTaskType.CLASSIFICATION:
            if (
                baseline.accuracy is None
                or current.accuracy is None
            ):
                raise DriftValidationError(
                    "Accuracy metrics are required"
                )

            degradation = max(
                0.0,
                baseline.accuracy - current.accuracy,
            )

            threshold = (
                self.config.accuracy_drop_threshold
            )

            metric_name = (
                DriftMetric.ACCURACY_DROP.value
            )

        else:
            if (
                baseline.mae is None
                or current.mae is None
            ):
                raise DriftValidationError(
                    "MAE metrics are required"
                )

            degradation = max(
                0.0,
                (
                    current.mae - baseline.mae
                ) / max(
                    baseline.mae,
                    1e-8,
                ),
            )

            threshold = (
                self.config.mae_increase_threshold
            )

            metric_name = (
                DriftMetric.MAE_INCREASE.value
            )

        return PerformanceDriftResult(
            task_type=baseline.task_type,
            baseline_metrics=baseline,
            current_metrics=current,
            drift_detected=(
                degradation >= threshold
            ),
            severity=(
                DriftSeverityClassifier.classify(
                    degradation,
                    threshold,
                )
            ),
            metric_name=metric_name,
            degradation=degradation,
            threshold=threshold,
        )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

class DriftRecommendationEngine:

    @staticmethod
    def generate(
        feature_results: Sequence[FeatureDriftResult],
        performance_result: (
            PerformanceDriftResult | None
        ),
    ) -> tuple[str, ...]:
        recommendations = []

        drifting_features = [
            result.feature_name
            for result in feature_results
            if result.drift_detected
        ]

        if drifting_features:
            recommendations.append(
                "Investigate distribution changes in: "
                + ", ".join(drifting_features)
            )

            recommendations.append(
                "Check upstream sensor calibration, "
                "data ingestion, preprocessing and "
                "environmental changes."
            )

        missing_features = [
            result.feature_name
            for result in feature_results
            if any(
                metric.metric == DriftMetric.MISSING_RATE
                and metric.drift_detected
                for metric in result.metrics
            )
        ]

        if missing_features:
            recommendations.append(
                "Inspect missing-value patterns and "
                "sensor availability for: "
                + ", ".join(missing_features)
            )

        if (
            performance_result is not None
            and performance_result.drift_detected
        ):
            recommendations.append(
                "Investigate labeled prediction errors "
                "and potential changes in the "
                "input-target relationship."
            )

            recommendations.append(
                "Evaluate retraining candidates offline "
                "against a representative holdout dataset."
            )

        if not recommendations:
            recommendations.append(
                "No configured drift threshold was exceeded. "
                "Continue routine monitoring."
            )

        recommendations.append(
            "Require validation and human approval "
            "before any robotics model deployment change."
        )

        return tuple(recommendations)


# ============================================================
# DRIFT DETECTION SERVICE
# ============================================================

class DriftDetectionService:

    def __init__(
        self,
        config: DriftDetectionConfig | None = None,
    ) -> None:
        self.config = (
            config or DriftDetectionConfig()
        )

        self.feature_detector = (
            FeatureDriftDetector(
                self.config
            )
        )

        self.performance_monitor = (
            ModelPerformanceMonitor(
                self.config
            )
        )

        self.recommendation_engine = (
            DriftRecommendationEngine()
        )

        self._baselines: dict[
            tuple[str, str],
            DriftBaseline,
        ] = {}

        self._reports: dict[
            tuple[str, str],
            DriftReport,
        ] = {}

        self._alerts: dict[
            tuple[str, str],
            DriftAlert,
        ] = {}

        self._lock = threading.RLock()

        self._load()

    # --------------------------------------------------------
    # AUTHORIZATION
    # --------------------------------------------------------

    @staticmethod
    def _authorize(
        access: DriftAccessContext,
        permission: str,
    ) -> None:
        if not getattr(
            access,
            permission,
            False,
        ):
            raise DriftAuthorizationError(
                f"Missing permission: {permission}"
            )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    def _storage_path(self) -> Path:
        return (
            self.config.storage_root
            / "drift_registry.json"
        )

    def _persist(self) -> None:
        atomic_write_json(
            self._storage_path(),
            {
                "schema_version": 1,
                "baselines": [
                    json_safe(item)
                    for item in self._baselines.values()
                ],
                "reports": [
                    json_safe(item)
                    for item in self._reports.values()
                ],
                "alerts": [
                    json_safe(item)
                    for item in self._alerts.values()
                ],
            },
        )

    @staticmethod
    def _parse_metric_result(
        data: Mapping[str, Any],
    ) -> DriftMetricResult:
        return DriftMetricResult(
            metric=DriftMetric(data["metric"]),
            value=data["value"],
            threshold=data["threshold"],
            drift_detected=data["drift_detected"],
            severity=DriftSeverity(
                data["severity"]
            ),
        )

    @classmethod
    def _parse_feature_result(
        cls,
        data: Mapping[str, Any],
    ) -> FeatureDriftResult:
        return FeatureDriftResult(
            feature_name=data["feature_name"],
            feature_type=FeatureType(
                data["feature_type"]
            ),
            drift_detected=data["drift_detected"],
            severity=DriftSeverity(
                data["severity"]
            ),
            metrics=tuple(
                cls._parse_metric_result(item)
                for item in data["metrics"]
            ),
            baseline_sample_count=data[
                "baseline_sample_count"
            ],
            current_sample_count=data[
                "current_sample_count"
            ],
            baseline_missing_rate=data[
                "baseline_missing_rate"
            ],
            current_missing_rate=data[
                "current_missing_rate"
            ],
            baseline_mean=data.get(
                "baseline_mean"
            ),
            current_mean=data.get(
                "current_mean"
            ),
            baseline_std=data.get(
                "baseline_std"
            ),
            current_std=data.get(
                "current_std"
            ),
        )

    @staticmethod
    def _parse_performance_metrics(
        data: Mapping[str, Any],
    ) -> PerformanceMetrics:
        return PerformanceMetrics(
            task_type=ModelTaskType(
                data["task_type"]
            ),
            sample_count=data["sample_count"],
            accuracy=data.get("accuracy"),
            mae=data.get("mae"),
            rmse=data.get("rmse"),
        )

    @classmethod
    def _parse_performance_result(
        cls,
        data: Mapping[str, Any] | None,
    ) -> PerformanceDriftResult | None:
        if data is None:
            return None

        return PerformanceDriftResult(
            task_type=ModelTaskType(
                data["task_type"]
            ),
            baseline_metrics=cls._parse_performance_metrics(
                data["baseline_metrics"]
            ),
            current_metrics=cls._parse_performance_metrics(
                data["current_metrics"]
            ),
            drift_detected=data["drift_detected"],
            severity=DriftSeverity(
                data["severity"]
            ),
            metric_name=data["metric_name"],
            degradation=data["degradation"],
            threshold=data["threshold"],
        )

    def _load(self) -> None:
        path = self._storage_path()

        if not path.exists():
            return

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            if payload.get("schema_version") != 1:
                raise DriftPersistenceError(
                    "Unsupported drift registry schema"
                )

            for item in payload.get(
                "baselines",
                [],
            ):
                features = tuple(
                    FeatureDefinition(
                        name=feature["name"],
                        feature_type=FeatureType(
                            feature["feature_type"]
                        ),
                        unit=feature.get(
                            "unit",
                            "",
                        ),
                        description=feature.get(
                            "description",
                            "",
                        ),
                    )
                    for feature in item["features"]
                )

                baseline = DriftBaseline(
                    baseline_id=item["baseline_id"],
                    tenant_id=item["tenant_id"],
                    model_id=item["model_id"],
                    model_version=item["model_version"],
                    features=features,
                    reference_data={
                        name: tuple(values)
                        for name, values
                        in item["reference_data"].items()
                    },
                    created_at=datetime.fromisoformat(
                        item["created_at"]
                    ),
                    created_by=item["created_by"],
                    dataset_id=item.get(
                        "dataset_id"
                    ),
                    dataset_version_id=item.get(
                        "dataset_version_id"
                    ),
                    metadata=item.get(
                        "metadata",
                        {},
                    ),
                )

                self._baselines[
                    (
                        baseline.tenant_id,
                        baseline.baseline_id,
                    )
                ] = baseline

            for item in payload.get(
                "reports",
                [],
            ):
                report = DriftReport(
                    report_id=item["report_id"],
                    tenant_id=item["tenant_id"],
                    model_id=item["model_id"],
                    model_version=item["model_version"],
                    baseline_id=item["baseline_id"],
                    created_at=datetime.fromisoformat(
                        item["created_at"]
                    ),
                    sample_count=item["sample_count"],
                    drift_detected=item["drift_detected"],
                    overall_severity=DriftSeverity(
                        item["overall_severity"]
                    ),
                    feature_results=tuple(
                        self._parse_feature_result(
                            feature
                        )
                        for feature in item[
                            "feature_results"
                        ]
                    ),
                    performance_result=(
                        self._parse_performance_result(
                            item.get(
                                "performance_result"
                            )
                        )
                    ),
                    recommendations=tuple(
                        item.get(
                            "recommendations",
                            [],
                        )
                    ),
                    metadata=item.get(
                        "metadata",
                        {},
                    ),
                )

                self._reports[
                    (
                        report.tenant_id,
                        report.report_id,
                    )
                ] = report

            for item in payload.get(
                "alerts",
                [],
            ):
                alert = DriftAlert(
                    alert_id=item["alert_id"],
                    report_id=item["report_id"],
                    tenant_id=item["tenant_id"],
                    model_id=item["model_id"],
                    severity=DriftSeverity(
                        item["severity"]
                    ),
                    message=item["message"],
                    created_at=datetime.fromisoformat(
                        item["created_at"]
                    ),
                    status=DriftAlertStatus(
                        item["status"]
                    ),
                    acknowledged_by=item.get(
                        "acknowledged_by"
                    ),
                    resolved_by=item.get(
                        "resolved_by"
                    ),
                )

                self._alerts[
                    (
                        alert.tenant_id,
                        alert.alert_id,
                    )
                ] = alert

        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise DriftPersistenceError(
                "Invalid drift registry"
            ) from exc

    # --------------------------------------------------------
    # BASELINE REGISTRATION
    # --------------------------------------------------------

    def register_baseline(
        self,
        model_id: str,
        model_version: str,
        features: Sequence[FeatureDefinition],
        reference_data: Mapping[str, Sequence[Any]],
        access: DriftAccessContext,
        *,
        dataset_id: str | None = None,
        dataset_version_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DriftBaseline:
        self._authorize(
            access,
            "can_write",
        )

        validate_identifier(
            model_id,
            "model_id",
        )

        validate_identifier(
            model_version,
            "model_version",
        )

        if not features:
            raise DriftValidationError(
                "At least one feature is required"
            )

        feature_names = [
            feature.name
            for feature in features
        ]

        if len(feature_names) != len(set(feature_names)):
            raise DriftValidationError(
                "Duplicate feature names"
            )

        if set(reference_data) != set(feature_names):
            raise DriftValidationError(
                "Reference data columns must match feature definitions"
            )

        lengths = {
            len(values)
            for values in reference_data.values()
        }

        if len(lengths) != 1:
            raise DriftValidationError(
                "Reference feature columns must have equal lengths"
            )

        sample_count = next(iter(lengths))

        if (
            sample_count
            < self.config.minimum_baseline_samples
        ):
            raise DriftValidationError(
                "Insufficient baseline samples"
            )

        normalized_data = {}

        for feature in features:
            values = tuple(
                reference_data[feature.name]
            )

            if feature.feature_type in (
                FeatureType.NUMERICAL,
                FeatureType.EMBEDDING_COMPONENT,
            ):
                for value in values:
                    if not is_missing(value):
                        validate_number(
                            value,
                            feature.name,
                        )

            normalized_data[
                feature.name
            ] = values

        baseline = DriftBaseline(
            baseline_id=generate_id(),
            tenant_id=access.tenant_id,
            model_id=model_id,
            model_version=model_version,
            features=tuple(features),
            reference_data=normalized_data,
            created_at=utc_now(),
            created_by=access.user_id,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            metadata=copy.deepcopy(
                dict(metadata or {})
            ),
        )

        with self._lock:
            self._baselines[
                (
                    access.tenant_id,
                    baseline.baseline_id,
                )
            ] = baseline

            self._persist()

        return copy.deepcopy(baseline)

    # --------------------------------------------------------
    # GET BASELINE
    # --------------------------------------------------------

    def get_baseline(
        self,
        baseline_id: str,
        access: DriftAccessContext,
    ) -> DriftBaseline:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            baseline = self._baselines.get(
                (
                    access.tenant_id,
                    baseline_id,
                )
            )

            if baseline is None:
                raise DriftBaselineNotFoundError(
                    "Drift baseline not found"
                )

            return copy.deepcopy(baseline)

    # --------------------------------------------------------
    # RUN DRIFT DETECTION
    # --------------------------------------------------------

    def detect_drift(
        self,
        baseline_id: str,
        current_data: Mapping[str, Sequence[Any]],
        access: DriftAccessContext,
        *,
        baseline_performance: (
            PerformanceMetrics | None
        ) = None,
        current_performance: (
            PerformanceMetrics | None
        ) = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DriftReport:
        self._authorize(
            access,
            "can_write",
        )

        with self._lock:
            baseline = self._baselines.get(
                (
                    access.tenant_id,
                    baseline_id,
                )
            )

            if baseline is None:
                raise DriftBaselineNotFoundError(
                    "Drift baseline not found"
                )

            expected_features = {
                feature.name
                for feature in baseline.features
            }

            if set(current_data) != expected_features:
                raise DriftValidationError(
                    "Monitoring data columns do not match baseline"
                )

            lengths = {
                len(values)
                for values in current_data.values()
            }

            if len(lengths) != 1:
                raise DriftValidationError(
                    "Monitoring columns must have equal lengths"
                )

            sample_count = next(iter(lengths))

            if (
                sample_count
                < self.config.minimum_monitoring_samples
            ):
                raise DriftValidationError(
                    "Insufficient monitoring samples"
                )

            feature_results = []

            for feature in baseline.features:
                result = self.feature_detector.detect(
                    feature,
                    baseline.reference_data[
                        feature.name
                    ],
                    current_data[
                        feature.name
                    ],
                )

                feature_results.append(
                    result
                )

            performance_result = None

            if (
                baseline_performance is not None
                or current_performance is not None
            ):
                if (
                    baseline_performance is None
                    or current_performance is None
                ):
                    raise DriftValidationError(
                        "Both baseline and current performance are required"
                    )

                performance_result = (
                    self.performance_monitor.detect(
                        baseline_performance,
                        current_performance,
                    )
                )

            severities = [
                result.severity
                for result in feature_results
            ]

            if performance_result is not None:
                severities.append(
                    performance_result.severity
                )

            overall_severity = (
                DriftSeverityClassifier.maximum(
                    severities
                )
            )

            detected = (
                any(
                    result.drift_detected
                    for result in feature_results
                )
                or (
                    performance_result is not None
                    and performance_result.drift_detected
                )
            )

            recommendations = (
                self.recommendation_engine.generate(
                    feature_results,
                    performance_result,
                )
            )

            report = DriftReport(
                report_id=generate_id(),
                tenant_id=access.tenant_id,
                model_id=baseline.model_id,
                model_version=baseline.model_version,
                baseline_id=baseline.baseline_id,
                created_at=utc_now(),
                sample_count=sample_count,
                drift_detected=detected,
                overall_severity=overall_severity,
                feature_results=tuple(
                    feature_results
                ),
                performance_result=performance_result,
                recommendations=recommendations,
                metadata=copy.deepcopy(
                    dict(metadata or {})
                ),
            )

            self._reports[
                (
                    access.tenant_id,
                    report.report_id,
                )
            ] = report

            if detected:
                alert = DriftAlert(
                    alert_id=generate_id(),
                    report_id=report.report_id,
                    tenant_id=access.tenant_id,
                    model_id=baseline.model_id,
                    severity=overall_severity,
                    message=(
                        f"Drift detected for model "
                        f"{baseline.model_id} "
                        f"version {baseline.model_version}"
                    ),
                    created_at=utc_now(),
                )

                self._alerts[
                    (
                        access.tenant_id,
                        alert.alert_id,
                    )
                ] = alert

            self._trim_history(
                access.tenant_id
            )

            self._persist()

            return copy.deepcopy(report)

    # --------------------------------------------------------
    # HISTORY MANAGEMENT
    # --------------------------------------------------------

    def _trim_history(
        self,
        tenant_id: str,
    ) -> None:
        tenant_reports = sorted(
            (
                report
                for report in self._reports.values()
                if report.tenant_id == tenant_id
            ),
            key=lambda report: report.created_at,
        )

        excess = (
            len(tenant_reports)
            - self.config.maximum_history_records
        )

        for report in tenant_reports[:max(0, excess)]:
            self._reports.pop(
                (
                    tenant_id,
                    report.report_id,
                ),
                None,
            )

    def get_report(
        self,
        report_id: str,
        access: DriftAccessContext,
    ) -> DriftReport:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            report = self._reports.get(
                (
                    access.tenant_id,
                    report_id,
                )
            )

            if report is None:
                raise DriftBaselineNotFoundError(
                    "Drift report not found"
                )

            return copy.deepcopy(report)

    def report_history(
        self,
        model_id: str,
        access: DriftAccessContext,
    ) -> tuple[DriftReport, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            reports = [
                copy.deepcopy(report)
                for report in self._reports.values()
                if (
                    report.tenant_id == access.tenant_id
                    and report.model_id == model_id
                )
            ]

        return tuple(
            sorted(
                reports,
                key=lambda report: report.created_at,
            )
        )

    # --------------------------------------------------------
    # ALERT MANAGEMENT
    # --------------------------------------------------------

    def list_alerts(
        self,
        access: DriftAccessContext,
        *,
        status: DriftAlertStatus | None = None,
    ) -> tuple[DriftAlert, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            alerts = []

            for alert in self._alerts.values():
                if alert.tenant_id != access.tenant_id:
                    continue

                if (
                    status is not None
                    and alert.status != status
                ):
                    continue

                alerts.append(
                    copy.deepcopy(alert)
                )

            return tuple(alerts)

    def acknowledge_alert(
        self,
        alert_id: str,
        access: DriftAccessContext,
    ) -> DriftAlert:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            alert = self._alerts.get(
                (
                    access.tenant_id,
                    alert_id,
                )
            )

            if alert is None:
                raise DriftBaselineNotFoundError(
                    "Drift alert not found"
                )

            if alert.status != DriftAlertStatus.OPEN:
                raise DriftConflictError(
                    "Alert is not open"
                )

            alert.status = (
                DriftAlertStatus.ACKNOWLEDGED
            )

            alert.acknowledged_by = (
                access.user_id
            )

            self._persist()

            return copy.deepcopy(alert)

    def resolve_alert(
        self,
        alert_id: str,
        access: DriftAccessContext,
    ) -> DriftAlert:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            alert = self._alerts.get(
                (
                    access.tenant_id,
                    alert_id,
                )
            )

            if alert is None:
                raise DriftBaselineNotFoundError(
                    "Drift alert not found"
                )

            if alert.status == DriftAlertStatus.RESOLVED:
                raise DriftConflictError(
                    "Alert is already resolved"
                )

            alert.status = (
                DriftAlertStatus.RESOLVED
            )

            alert.resolved_by = (
                access.user_id
            )

            self._persist()

            return copy.deepcopy(alert)

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    def export_report_json(
        self,
        report_id: str,
        access: DriftAccessContext,
    ) -> str:
        report = self.get_report(
            report_id,
            access,
        )

        return json.dumps(
            report.to_dict(),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC ROBOTICS DATA
# ============================================================

class SyntheticDriftDataFactory:

    @staticmethod
    def feature_definitions() -> tuple[
        FeatureDefinition,
        ...
    ]:
        return (
            FeatureDefinition(
                name="motor_temperature",
                feature_type=FeatureType.NUMERICAL,
                unit="celsius",
            ),
            FeatureDefinition(
                name="vibration_rms",
                feature_type=FeatureType.NUMERICAL,
                unit="mm/s",
            ),
            FeatureDefinition(
                name="lidar_distance",
                feature_type=FeatureType.NUMERICAL,
                unit="meters",
            ),
            FeatureDefinition(
                name="robot_operating_mode",
                feature_type=FeatureType.CATEGORICAL,
            ),
            FeatureDefinition(
                name="vision_embedding_0",
                feature_type=FeatureType.EMBEDDING_COMPONENT,
            ),
        )

    @staticmethod
    def baseline_data(
        samples: int = 200,
    ) -> dict[str, list[Any]]:
        return {
            "motor_temperature": [
                40.0 + math.sin(index * 0.13) * 2.0
                for index in range(samples)
            ],
            "vibration_rms": [
                0.20 + math.sin(index * 0.19) * 0.04
                for index in range(samples)
            ],
            "lidar_distance": [
                5.0 + math.sin(index * 0.07) * 0.5
                for index in range(samples)
            ],
            "robot_operating_mode": [
                "navigation"
                if index % 4
                else "inspection"
                for index in range(samples)
            ],
            "vision_embedding_0": [
                math.sin(index * 0.09)
                for index in range(samples)
            ],
        }

    @staticmethod
    def current_data(
        samples: int = 200,
        *,
        drift: bool = True,
    ) -> dict[str, list[Any]]:
        data = (
            SyntheticDriftDataFactory.baseline_data(
                samples
            )
        )

        if drift:
            data["motor_temperature"] = [
                value + 15.0
                for value in data[
                    "motor_temperature"
                ]
            ]

            data["vibration_rms"] = [
                value * 3.0
                for value in data[
                    "vibration_rms"
                ]
            ]

            data["robot_operating_mode"] = [
                "emergency"
                if index % 2
                else "inspection"
                for index in range(samples)
            ]

            data["vision_embedding_0"] = [
                value + 2.0
                for value in data[
                    "vision_embedding_0"
                ]
            ]

        return data


# ============================================================
# UNIT TESTS
# ============================================================

class TestDriftDetection(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

        self.root = Path(
            self.temporary.name
        )

        self.config = DriftDetectionConfig(
            storage_root=self.root / "registry",
        )

        self.service = DriftDetectionService(
            self.config
        )

        self.access = DriftAccessContext(
            tenant_id="robotics_lab",
            user_id="mlops_engineer",
            can_read=True,
            can_write=True,
            can_manage=True,
        )

        self.features = (
            SyntheticDriftDataFactory.feature_definitions()
        )

        self.reference = (
            SyntheticDriftDataFactory.baseline_data()
        )

        self.baseline = (
            self.service.register_baseline(
                model_id="robot-perception-v1",
                model_version="1.0.0",
                features=self.features,
                reference_data=self.reference,
                access=self.access,
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_baseline_registration(self) -> None:
        self.assertEqual(
            self.baseline.sample_count,
            200,
        )

    def test_psi_identical_distributions(self) -> None:
        values = [
            float(index)
            for index in range(100)
        ]

        psi = (
            DriftStatistics.population_stability_index(
                values,
                values,
            )
        )

        self.assertAlmostEqual(
            psi,
            0.0,
        )

    def test_ks_identical_distributions(self) -> None:
        values = [
            float(index)
            for index in range(100)
        ]

        ks = (
            DriftStatistics.kolmogorov_smirnov(
                values,
                values,
            )
        )

        self.assertEqual(
            ks,
            0.0,
        )

    def test_ks_shifted_distributions(self) -> None:
        reference = [
            float(index)
            for index in range(100)
        ]

        current = [
            float(index + 1000)
            for index in range(100)
        ]

        ks = (
            DriftStatistics.kolmogorov_smirnov(
                reference,
                current,
            )
        )

        self.assertEqual(
            ks,
            1.0,
        )

    def test_js_identical_distributions(self) -> None:
        result = (
            DriftStatistics.categorical_js(
                ["a", "b", "a", "b"],
                ["a", "b", "a", "b"],
            )
        )

        self.assertAlmostEqual(
            result,
            0.0,
        )

    def test_js_different_distributions(self) -> None:
        result = (
            DriftStatistics.categorical_js(
                ["a"] * 100,
                ["b"] * 100,
            )
        )

        self.assertAlmostEqual(
            result,
            1.0,
        )

    def test_wasserstein(self) -> None:
        distance = (
            DriftStatistics.wasserstein_1d(
                [0.0, 1.0],
                [1.0, 2.0],
            )
        )

        self.assertAlmostEqual(
            distance,
            1.0,
        )

    def test_no_drift(self) -> None:
        report = self.service.detect_drift(
            self.baseline.baseline_id,
            SyntheticDriftDataFactory.current_data(
                drift=False
            ),
            self.access,
        )

        self.assertFalse(
            report.drift_detected
        )

    def test_detect_drift(self) -> None:
        report = self.service.detect_drift(
            self.baseline.baseline_id,
            SyntheticDriftDataFactory.current_data(
                drift=True
            ),
            self.access,
        )

        self.assertTrue(
            report.drift_detected
        )

    def test_feature_drift(self) -> None:
        report = self.service.detect_drift(
            self.baseline.baseline_id,
            SyntheticDriftDataFactory.current_data(
                drift=True
            ),
            self.access,
        )

        results = {
            result.feature_name: result
            for result in report.feature_results
        }

        self.assertTrue(
            results[
                "motor_temperature"
            ].drift_detected
        )

    def test_alert_creation(self) -> None:
        self.service.detect_drift(
            self.baseline.baseline_id,
            SyntheticDriftDataFactory.current_data(
                drift=True
            ),
            self.access,
        )

        alerts = self.service.list_alerts(
            self.access
        )

        self.assertEqual(
            len(alerts),
            1,
        )

    def test_alert_acknowledgement(self) -> None:
        self.service.detect_drift(
            self.baseline.baseline_id,
            SyntheticDriftDataFactory.current_data(
                drift=True
            ),
            self.access,
        )

        alert = self.service.list_alerts(
            self.access
        )[0]

        acknowledged = (
            self.service.acknowledge_alert(
                alert.alert_id,
                self.access,
            )
        )

        self.assertEqual(
            acknowledged.status,
            DriftAlertStatus.ACKNOWLEDGED,
        )

    def test_alert_resolution(self) -> None:
        self.service.detect_drift(
            self.baseline.baseline_id,
            SyntheticDriftDataFactory.current_data(
                drift=True
            ),
            self.access,
        )

        alert = self.service.list_alerts(
            self.access
        )[0]

        resolved = self.service.resolve_alert(
            alert.alert_id,
            self.access,
        )

        self.assertEqual(
            resolved.status,
            DriftAlertStatus.RESOLVED,
        )

    def test_classification_performance(self) -> None:
        metrics = (
            ModelPerformanceMonitor.classification_metrics(
                [0, 1, 1, 0],
                [0, 1, 0, 0],
            )
        )

        self.assertEqual(
            metrics.accuracy,
            0.75,
        )

    def test_regression_performance(self) -> None:
        metrics = (
            ModelPerformanceMonitor.regression_metrics(
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 4.0],
            )
        )

        self.assertAlmostEqual(
            metrics.mae,
            1 / 3,
        )

    def test_performance_drift(self) -> None:
        baseline = PerformanceMetrics(
            task_type=ModelTaskType.CLASSIFICATION,
            sample_count=100,
            accuracy=0.95,
        )

        current = PerformanceMetrics(
            task_type=ModelTaskType.CLASSIFICATION,
            sample_count=100,
            accuracy=0.70,
        )

        result = (
            self.service.performance_monitor.detect(
                baseline,
                current,
            )
        )

        self.assertTrue(
            result.drift_detected
        )

    def test_persistence(self) -> None:
        report = self.service.detect_drift(
            self.baseline.baseline_id,
            SyntheticDriftDataFactory.current_data(
                drift=True
            ),
            self.access,
        )

        reloaded = DriftDetectionService(
            self.config
        )

        restored = reloaded.get_report(
            report.report_id,
            self.access,
        )

        self.assertEqual(
            restored.report_id,
            report.report_id,
        )

    def test_tenant_isolation(self) -> None:
        other = DriftAccessContext(
            tenant_id="another_tenant",
            user_id="other_user",
            can_read=True,
        )

        with self.assertRaises(
            DriftBaselineNotFoundError
        ):
            self.service.get_baseline(
                self.baseline.baseline_id,
                other,
            )

    def test_missing_rate(self) -> None:
        rate = DriftStatistics.missing_rate(
            [1.0, None, 3.0, None]
        )

        self.assertEqual(
            rate,
            0.5,
        )

    def test_insufficient_samples(self) -> None:
        with self.assertRaises(
            DriftValidationError
        ):
            self.service.detect_drift(
                self.baseline.baseline_id,
                SyntheticDriftDataFactory.current_data(
                    samples=5
                ),
                self.access,
            )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        service = DriftDetectionService(
            DriftDetectionConfig(
                storage_root=root / "registry",
            )
        )

        access = DriftAccessContext(
            tenant_id="physical_ai_lab",
            user_id="mlops_engineer",
            can_read=True,
            can_write=True,
            can_manage=True,
        )

        baseline = service.register_baseline(
            model_id="robot-perception-model",
            model_version="2.0.0",
            features=(
                SyntheticDriftDataFactory.feature_definitions()
            ),
            reference_data=(
                SyntheticDriftDataFactory.baseline_data()
            ),
            access=access,
            dataset_id="robotics-dataset",
            dataset_version_id="dataset-v1",
            metadata={
                "robot_type": "autonomous_mobile_robot",
                "environment": "warehouse",
            },
        )

        baseline_performance = PerformanceMetrics(
            task_type=ModelTaskType.CLASSIFICATION,
            sample_count=1000,
            accuracy=0.96,
        )

        current_performance = PerformanceMetrics(
            task_type=ModelTaskType.CLASSIFICATION,
            sample_count=1000,
            accuracy=0.81,
        )

        report = service.detect_drift(
            baseline_id=baseline.baseline_id,
            current_data=(
                SyntheticDriftDataFactory.current_data(
                    drift=True
                )
            ),
            access=access,
            baseline_performance=baseline_performance,
            current_performance=current_performance,
            metadata={
                "monitoring_window": "synthetic_batch",
            },
        )

        alerts = service.list_alerts(
            access
        )

        print("\nPHYSICAL AI DRIFT DETECTION ENGINE")
        print("=" * 70)

        print(
            json.dumps(
                {
                    "baseline_id": baseline.baseline_id,
                    "model_id": report.model_id,
                    "model_version": report.model_version,
                    "drift_detected": report.drift_detected,
                    "overall_severity": (
                        report.overall_severity.value
                    ),
                    "sample_count": report.sample_count,
                    "feature_drift": [
                        {
                            "feature": result.feature_name,
                            "detected": result.drift_detected,
                            "severity": result.severity.value,
                            "metrics": {
                                metric.metric.value: round(
                                    metric.value,
                                    6,
                                )
                                for metric in result.metrics
                            },
                        }
                        for result in report.feature_results
                    ],
                    "performance_drift": json_safe(
                        report.performance_result
                    ),
                    "recommendations": (
                        report.recommendations
                    ),
                    "alert_count": len(alerts),
                    "automatic_robot_control": False,
                    "automatic_model_redeployment": False,
                },
                indent=2,
                allow_nan=False,
            )
        )

    print("\nRUNNING UNIT TESTS")
    print("=" * 70)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestDriftDetection
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
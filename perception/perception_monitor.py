"""Perception Monitor module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/perception_monitor.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: perception_monitor.py

Real-Time Perception Health, Quality & Reliability Monitor

Features
--------
- Camera, LiDAR, depth, detection, segmentation and tracking monitoring
- Optical-flow pipeline monitoring
- Sensor freshness and heartbeat monitoring
- Perception latency and FPS monitoring
- Detection confidence and tracking quality monitoring
- Frame-drop and inference-failure monitoring
- Sensor data validity checks
- Multi-sensor consistency monitoring
- Perception anomaly detection
- Configurable warning and critical thresholds
- Sensor degradation detection
- Perception pipeline health scoring
- Operational readiness recommendations
- Robot and camera-specific monitoring
- Multi-tenant access control
- Alert generation and deduplication
- Bounded metrics and event history
- Thread-safe processing
- JSON health reports
- Unit tests and demonstration

Dependencies
------------
Python >= 3.11
Standard library only.

Safety
------
This module is an advisory observability component.

Health scores and readiness recommendations must never directly
authorize robot movement, disable protective systems, or replace
independently validated functional-safety mechanisms.
"""

from __future__ import annotations

import json
import math
import statistics
import threading
import time
import unittest
import uuid

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class PerceptionMonitorError(Exception):
    """Base perception monitoring exception."""


class PerceptionValidationError(PerceptionMonitorError):
    """Invalid perception telemetry or configuration."""


class PerceptionAuthorizationError(PerceptionMonitorError):
    """Access to perception monitoring data denied."""


class PerceptionCapacityError(PerceptionMonitorError):
    """Monitoring resource limit exceeded."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def validate_identifier(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise PerceptionValidationError(
            f"{field_name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise PerceptionValidationError(
            f"{field_name} must contain 1-128 characters"
        )

    return value


def validate_timestamp(
    timestamp: datetime,
) -> datetime:
    if not isinstance(timestamp, datetime):
        raise PerceptionValidationError(
            "Timestamp must be datetime"
        )

    if (
        timestamp.tzinfo is None
        or timestamp.utcoffset() is None
    ):
        raise PerceptionValidationError(
            "Timestamp must be timezone-aware"
        )

    return timestamp.astimezone(timezone.utc)


def finite_float(
    value: Any,
    field_name: str,
) -> float:
    if isinstance(value, bool):
        raise PerceptionValidationError(
            f"{field_name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PerceptionValidationError(
            f"{field_name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise PerceptionValidationError(
            f"{field_name} must be finite"
        )

    return result


def bounded_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def safe_mean(values: Sequence[float]) -> float:
    return (
        float(statistics.fmean(values))
        if values
        else 0.0
    )


def percentile(
    values: Sequence[float],
    percent: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    position = (
        (len(ordered) - 1)
        * percent / 100.0
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(ordered[lower])

    weight = position - lower

    return float(
        ordered[lower] * (1 - weight)
        + ordered[upper] * weight
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

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

    if isinstance(value, (list, tuple, deque, set, frozenset)):
        return [json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class PerceptionComponent(str, Enum):
    CAMERA = "camera"
    LIDAR = "lidar"
    DEPTH_ESTIMATION = "depth_estimation"
    OBJECT_DETECTION = "object_detection"
    OBJECT_TRACKING = "object_tracking"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    OPTICAL_FLOW = "optical_flow"
    SENSOR_FUSION = "sensor_fusion"
    POINT_CLOUD = "point_cloud"
    VISUAL_ODOMETRY = "visual_odometry"
    PERCEPTION_PIPELINE = "perception_pipeline"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    SENSOR_STALE = "sensor_stale"
    LOW_FPS = "low_fps"
    HIGH_LATENCY = "high_latency"
    HIGH_FRAME_DROP = "high_frame_drop"
    HIGH_ERROR_RATE = "high_error_rate"
    LOW_CONFIDENCE = "low_confidence"
    LOW_TRACKING_QUALITY = "low_tracking_quality"
    INVALID_DATA = "invalid_data"
    LOW_HEALTH_SCORE = "low_health_score"
    SENSOR_INCONSISTENCY = "sensor_inconsistency"


class OperationalRecommendation(str, Enum):
    NORMAL_OBSERVATION = "normal_observation"
    INCREASE_MONITORING = "increase_monitoring"
    REQUEST_OPERATOR_REVIEW = "request_operator_review"
    PERCEPTION_UNAVAILABLE = "perception_unavailable"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PerceptionMonitorConfig:
    minimum_fps: float = 10.0
    critical_fps: float = 3.0

    warning_latency_ms: float = 150.0
    critical_latency_ms: float = 500.0

    warning_frame_drop_ratio: float = 0.05
    critical_frame_drop_ratio: float = 0.20

    warning_error_ratio: float = 0.05
    critical_error_ratio: float = 0.20

    minimum_detection_confidence: float = 0.45
    minimum_tracking_quality: float = 0.50

    warning_sensor_age_s: float = 2.0
    critical_sensor_age_s: float = 5.0

    warning_health_score: float = 75.0
    critical_health_score: float = 40.0

    minimum_consistency_score: float = 0.60

    maximum_samples_per_component: int = 300
    maximum_alerts: int = 1000
    maximum_audit_events: int = 1000

    alert_cooldown_s: float = 10.0

    def __post_init__(self) -> None:
        numeric_fields = (
            "minimum_fps",
            "critical_fps",
            "warning_latency_ms",
            "critical_latency_ms",
            "warning_sensor_age_s",
            "critical_sensor_age_s",
            "warning_health_score",
            "critical_health_score",
            "alert_cooldown_s",
        )

        for name in numeric_fields:
            if finite_float(getattr(self, name), name) < 0:
                raise PerceptionValidationError(
                    f"{name} must be nonnegative"
                )

        ratio_fields = (
            "warning_frame_drop_ratio",
            "critical_frame_drop_ratio",
            "warning_error_ratio",
            "critical_error_ratio",
            "minimum_detection_confidence",
            "minimum_tracking_quality",
            "minimum_consistency_score",
        )

        for name in ratio_fields:
            value = finite_float(
                getattr(self, name),
                name,
            )

            if not 0 <= value <= 1:
                raise PerceptionValidationError(
                    f"{name} must be between 0 and 1"
                )

        for name in (
            "maximum_samples_per_component",
            "maximum_alerts",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise PerceptionValidationError(
                    f"{name} must be a positive integer"
                )

        if not self.critical_fps < self.minimum_fps:
            raise PerceptionValidationError(
                "critical_fps must be below minimum_fps"
            )

        for warning, critical in (
            (
                self.warning_latency_ms,
                self.critical_latency_ms,
            ),
            (
                self.warning_sensor_age_s,
                self.critical_sensor_age_s,
            ),
            (
                self.warning_frame_drop_ratio,
                self.critical_frame_drop_ratio,
            ),
            (
                self.warning_error_ratio,
                self.critical_error_ratio,
            ),
        ):
            if warning >= critical:
                raise PerceptionValidationError(
                    "Critical upper thresholds must exceed warnings"
                )

        if (
            self.critical_health_score
            >= self.warning_health_score
        ):
            raise PerceptionValidationError(
                "Critical health threshold must be below warning"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class PerceptionAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    can_ingest: bool = False
    can_view: bool = False
    can_manage: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.user_id, "user_id")


# ============================================================
# TELEMETRY MODEL
# ============================================================

@dataclass(frozen=True)
class PerceptionTelemetry:
    telemetry_id: str

    tenant_id: str
    robot_id: str

    component: PerceptionComponent
    sensor_id: str

    timestamp: datetime

    processing_latency_ms: float = 0.0
    fps: float | None = None

    processed_frames: int = 0
    dropped_frames: int = 0
    inference_errors: int = 0

    detection_count: int = 0
    average_detection_confidence: float | None = None

    active_tracks: int = 0
    tracking_quality: float | None = None

    valid_data_ratio: float = 1.0
    consistency_score: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "telemetry_id",
            "tenant_id",
            "robot_id",
            "sensor_id",
        ):
            validate_identifier(
                getattr(self, name),
                name,
            )

        validate_timestamp(self.timestamp)

        if not isinstance(self.component, PerceptionComponent):
            raise PerceptionValidationError(
                "Invalid perception component"
            )

        for name in (
            "processing_latency_ms",
            "fps",
        ):
            value = getattr(self, name)

            if value is not None:
                if finite_float(value, name) < 0:
                    raise PerceptionValidationError(
                        f"{name} must be nonnegative"
                    )

        for name in (
            "processed_frames",
            "dropped_frames",
            "inference_errors",
            "detection_count",
            "active_tracks",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise PerceptionValidationError(
                    f"{name} must be a nonnegative integer"
                )

        for name in (
            "average_detection_confidence",
            "tracking_quality",
            "valid_data_ratio",
            "consistency_score",
        ):
            value = getattr(self, name)

            if value is not None:
                value = finite_float(value, name)

                if not 0 <= value <= 1:
                    raise PerceptionValidationError(
                        f"{name} must be between 0 and 1"
                    )

        if self.inference_errors > self.processed_frames:
            raise PerceptionValidationError(
                "inference_errors cannot exceed processed_frames"
            )


# ============================================================
# HEALTH AND ALERT MODELS
# ============================================================

@dataclass(frozen=True)
class PerceptionMetricSummary:
    sample_count: int

    average_latency_ms: float
    p95_latency_ms: float

    average_fps: float

    frame_drop_ratio: float
    inference_error_ratio: float

    average_detection_confidence: float | None
    average_tracking_quality: float | None

    average_valid_data_ratio: float
    average_consistency_score: float | None


@dataclass(frozen=True)
class PerceptionAlert:
    alert_id: str

    tenant_id: str
    robot_id: str
    sensor_id: str

    component: PerceptionComponent

    category: AlertCategory
    severity: AlertSeverity

    message: str

    observed_value: float | None
    threshold: float | None

    timestamp: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True)
class ComponentHealthReport:
    tenant_id: str
    robot_id: str
    sensor_id: str

    component: PerceptionComponent

    status: HealthStatus
    health_score: float

    last_seen: datetime | None
    sensor_age_s: float | None

    metrics: PerceptionMetricSummary | None

    warnings: tuple[str, ...]

    recommendation: OperationalRecommendation

    generated_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True)
class RobotPerceptionHealthReport:
    report_id: str

    tenant_id: str
    robot_id: str

    overall_status: HealthStatus
    overall_health_score: float

    component_reports: tuple[ComponentHealthReport, ...]

    warning_count: int
    critical_count: int

    recommendation: OperationalRecommendation

    generated_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=lambda: {
            "advisory_only": True,
            "motion_authorization": False,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


@dataclass(frozen=True)
class PerceptionAuditEvent:
    event_id: str
    tenant_id: str
    user_id: str
    robot_id: str

    action: str
    resource_id: str

    timestamp: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# METRIC AGGREGATOR
# ============================================================

class PerceptionMetricAggregator:

    @staticmethod
    def summarize(
        samples: Sequence[PerceptionTelemetry],
    ) -> PerceptionMetricSummary:
        if not samples:
            raise PerceptionValidationError(
                "Cannot summarize empty telemetry"
            )

        latencies = [
            sample.processing_latency_ms
            for sample in samples
        ]

        fps_values = [
            sample.fps
            for sample in samples
            if sample.fps is not None
        ]

        confidence_values = [
            sample.average_detection_confidence
            for sample in samples
            if sample.average_detection_confidence is not None
        ]

        tracking_values = [
            sample.tracking_quality
            for sample in samples
            if sample.tracking_quality is not None
        ]

        consistency_values = [
            sample.consistency_score
            for sample in samples
            if sample.consistency_score is not None
        ]

        processed = sum(
            sample.processed_frames
            for sample in samples
        )

        dropped = sum(
            sample.dropped_frames
            for sample in samples
        )

        errors = sum(
            sample.inference_errors
            for sample in samples
        )

        total_frames = processed + dropped

        return PerceptionMetricSummary(
            sample_count=len(samples),
            average_latency_ms=safe_mean(latencies),
            p95_latency_ms=percentile(
                latencies,
                95,
            ),
            average_fps=safe_mean(fps_values),
            frame_drop_ratio=(
                dropped / total_frames
                if total_frames
                else 0.0
            ),
            inference_error_ratio=(
                errors / processed
                if processed
                else 0.0
            ),
            average_detection_confidence=(
                safe_mean(confidence_values)
                if confidence_values
                else None
            ),
            average_tracking_quality=(
                safe_mean(tracking_values)
                if tracking_values
                else None
            ),
            average_valid_data_ratio=safe_mean(
                [
                    sample.valid_data_ratio
                    for sample in samples
                ]
            ),
            average_consistency_score=(
                safe_mean(consistency_values)
                if consistency_values
                else None
            ),
        )


# ============================================================
# HEALTH SCORING
# ============================================================

class PerceptionHealthScorer:

    def __init__(
        self,
        config: PerceptionMonitorConfig,
    ) -> None:
        self.config = config

    def score(
        self,
        metrics: PerceptionMetricSummary,
        sensor_age_s: float,
    ) -> float:
        score = 100.0

        if metrics.p95_latency_ms >= self.config.critical_latency_ms:
            score -= 30.0

        elif metrics.p95_latency_ms >= self.config.warning_latency_ms:
            score -= 12.0

        if metrics.average_fps > 0:
            if metrics.average_fps < self.config.critical_fps:
                score -= 25.0

            elif metrics.average_fps < self.config.minimum_fps:
                score -= 10.0

        if (
            metrics.frame_drop_ratio
            >= self.config.critical_frame_drop_ratio
        ):
            score -= 25.0

        elif (
            metrics.frame_drop_ratio
            >= self.config.warning_frame_drop_ratio
        ):
            score -= 10.0

        if (
            metrics.inference_error_ratio
            >= self.config.critical_error_ratio
        ):
            score -= 30.0

        elif (
            metrics.inference_error_ratio
            >= self.config.warning_error_ratio
        ):
            score -= 12.0

        if metrics.average_detection_confidence is not None:
            if (
                metrics.average_detection_confidence
                < self.config.minimum_detection_confidence
            ):
                score -= 15.0

        if metrics.average_tracking_quality is not None:
            if (
                metrics.average_tracking_quality
                < self.config.minimum_tracking_quality
            ):
                score -= 15.0

        if metrics.average_valid_data_ratio < 0.9:
            score -= 20.0

        if metrics.average_consistency_score is not None:
            if (
                metrics.average_consistency_score
                < self.config.minimum_consistency_score
            ):
                score -= 15.0

        if sensor_age_s >= self.config.critical_sensor_age_s:
            score -= 50.0

        elif sensor_age_s >= self.config.warning_sensor_age_s:
            score -= 20.0

        return bounded_score(score)

    def classify(
        self,
        score: float,
        sensor_age_s: float,
    ) -> HealthStatus:
        if sensor_age_s >= self.config.critical_sensor_age_s:
            return HealthStatus.OFFLINE

        if score < self.config.critical_health_score:
            return HealthStatus.CRITICAL

        if score < self.config.warning_health_score:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY


# ============================================================
# ALERT RULE ENGINE
# ============================================================

class PerceptionAlertEngine:

    def __init__(
        self,
        config: PerceptionMonitorConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        telemetry: PerceptionTelemetry,
    ) -> tuple[PerceptionAlert, ...]:
        alerts = []

        def add(
            category: AlertCategory,
            severity: AlertSeverity,
            message: str,
            observed: float | None,
            threshold: float | None,
        ) -> None:
            alerts.append(
                PerceptionAlert(
                    alert_id=generate_id(),
                    tenant_id=telemetry.tenant_id,
                    robot_id=telemetry.robot_id,
                    sensor_id=telemetry.sensor_id,
                    component=telemetry.component,
                    category=category,
                    severity=severity,
                    message=message,
                    observed_value=observed,
                    threshold=threshold,
                    timestamp=telemetry.timestamp,
                )
            )

        latency = telemetry.processing_latency_ms

        if latency >= self.config.critical_latency_ms:
            add(
                AlertCategory.HIGH_LATENCY,
                AlertSeverity.CRITICAL,
                "Critical perception processing latency",
                latency,
                self.config.critical_latency_ms,
            )

        elif latency >= self.config.warning_latency_ms:
            add(
                AlertCategory.HIGH_LATENCY,
                AlertSeverity.WARNING,
                "Elevated perception processing latency",
                latency,
                self.config.warning_latency_ms,
            )

        if telemetry.fps is not None:
            if telemetry.fps < self.config.critical_fps:
                add(
                    AlertCategory.LOW_FPS,
                    AlertSeverity.CRITICAL,
                    "Critical perception frame rate",
                    telemetry.fps,
                    self.config.critical_fps,
                )

            elif telemetry.fps < self.config.minimum_fps:
                add(
                    AlertCategory.LOW_FPS,
                    AlertSeverity.WARNING,
                    "Perception frame rate below target",
                    telemetry.fps,
                    self.config.minimum_fps,
                )

        total = (
            telemetry.processed_frames
            + telemetry.dropped_frames
        )

        drop_ratio = (
            telemetry.dropped_frames / total
            if total
            else 0.0
        )

        if drop_ratio >= self.config.critical_frame_drop_ratio:
            add(
                AlertCategory.HIGH_FRAME_DROP,
                AlertSeverity.CRITICAL,
                "Critical frame drop ratio",
                drop_ratio,
                self.config.critical_frame_drop_ratio,
            )

        elif drop_ratio >= self.config.warning_frame_drop_ratio:
            add(
                AlertCategory.HIGH_FRAME_DROP,
                AlertSeverity.WARNING,
                "Elevated frame drop ratio",
                drop_ratio,
                self.config.warning_frame_drop_ratio,
            )

        error_ratio = (
            telemetry.inference_errors
            / telemetry.processed_frames
            if telemetry.processed_frames
            else 0.0
        )

        if error_ratio >= self.config.critical_error_ratio:
            add(
                AlertCategory.HIGH_ERROR_RATE,
                AlertSeverity.CRITICAL,
                "Critical inference error ratio",
                error_ratio,
                self.config.critical_error_ratio,
            )

        elif error_ratio >= self.config.warning_error_ratio:
            add(
                AlertCategory.HIGH_ERROR_RATE,
                AlertSeverity.WARNING,
                "Elevated inference error ratio",
                error_ratio,
                self.config.warning_error_ratio,
            )

        if telemetry.average_detection_confidence is not None:
            if (
                telemetry.average_detection_confidence
                < self.config.minimum_detection_confidence
            ):
                add(
                    AlertCategory.LOW_CONFIDENCE,
                    AlertSeverity.WARNING,
                    "Detection confidence below threshold",
                    telemetry.average_detection_confidence,
                    self.config.minimum_detection_confidence,
                )

        if telemetry.tracking_quality is not None:
            if (
                telemetry.tracking_quality
                < self.config.minimum_tracking_quality
            ):
                add(
                    AlertCategory.LOW_TRACKING_QUALITY,
                    AlertSeverity.WARNING,
                    "Tracking quality below threshold",
                    telemetry.tracking_quality,
                    self.config.minimum_tracking_quality,
                )

        if telemetry.valid_data_ratio < 0.9:
            add(
                AlertCategory.INVALID_DATA,
                AlertSeverity.WARNING,
                "Perception data validity degraded",
                telemetry.valid_data_ratio,
                0.9,
            )

        if telemetry.consistency_score is not None:
            if (
                telemetry.consistency_score
                < self.config.minimum_consistency_score
            ):
                add(
                    AlertCategory.SENSOR_INCONSISTENCY,
                    AlertSeverity.WARNING,
                    "Sensor consistency below threshold",
                    telemetry.consistency_score,
                    self.config.minimum_consistency_score,
                )

        return tuple(alerts)


# ============================================================
# OPERATIONAL RECOMMENDATION ENGINE
# ============================================================

class PerceptionRecommendationEngine:

    @staticmethod
    def recommend(
        status: HealthStatus,
    ) -> OperationalRecommendation:
        if status == HealthStatus.HEALTHY:
            return OperationalRecommendation.NORMAL_OBSERVATION

        if status == HealthStatus.DEGRADED:
            return OperationalRecommendation.INCREASE_MONITORING

        if status == HealthStatus.CRITICAL:
            return OperationalRecommendation.REQUEST_OPERATOR_REVIEW

        return OperationalRecommendation.PERCEPTION_UNAVAILABLE


# ============================================================
# PERCEPTION MONITOR
# ============================================================

class PerceptionMonitor:

    def __init__(
        self,
        config: PerceptionMonitorConfig | None = None,
    ) -> None:
        self.config = (
            config or PerceptionMonitorConfig()
        )

        self.aggregator = PerceptionMetricAggregator()

        self.scorer = PerceptionHealthScorer(
            self.config
        )

        self.alert_engine = PerceptionAlertEngine(
            self.config
        )

        self.recommendation_engine = (
            PerceptionRecommendationEngine()
        )

        self._samples: dict[
            tuple[str, str, PerceptionComponent, str],
            deque[PerceptionTelemetry],
        ] = {}

        self._alerts: deque[PerceptionAlert] = deque(
            maxlen=self.config.maximum_alerts
        )

        self._audit: deque[PerceptionAuditEvent] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._last_alert_time: dict[
            tuple[
                str,
                str,
                PerceptionComponent,
                str,
                AlertCategory,
            ],
            datetime,
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: PerceptionAccessContext,
        tenant_id: str,
        robot_id: str,
        permission: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise PerceptionAuthorizationError(
                "Cross-tenant access denied"
            )

        if robot_id not in access.allowed_robot_ids:
            raise PerceptionAuthorizationError(
                "Robot access denied"
            )

        if not getattr(access, permission, False):
            raise PerceptionAuthorizationError(
                f"Missing permission: {permission}"
            )

    @staticmethod
    def _key(
        telemetry: PerceptionTelemetry,
    ) -> tuple[str, str, PerceptionComponent, str]:
        return (
            telemetry.tenant_id,
            telemetry.robot_id,
            telemetry.component,
            telemetry.sensor_id,
        )

    def ingest(
        self,
        telemetry: PerceptionTelemetry,
        *,
        access: PerceptionAccessContext,
    ) -> tuple[PerceptionAlert, ...]:
        self._authorize(
            access,
            telemetry.tenant_id,
            telemetry.robot_id,
            "can_ingest",
        )

        key = self._key(telemetry)

        with self._lock:
            samples = self._samples.setdefault(
                key,
                deque(
                    maxlen=self.config.maximum_samples_per_component
                ),
            )

            if (
                samples
                and telemetry.timestamp < samples[-1].timestamp
            ):
                raise PerceptionValidationError(
                    "Out-of-order telemetry timestamp"
                )

            samples.append(telemetry)

            generated_alerts = (
                self.alert_engine.evaluate(telemetry)
            )

            accepted_alerts = []

            for alert in generated_alerts:
                cooldown_key = (
                    alert.tenant_id,
                    alert.robot_id,
                    alert.component,
                    alert.sensor_id,
                    alert.category,
                )

                previous_time = self._last_alert_time.get(
                    cooldown_key
                )

                if previous_time is not None:
                    elapsed = (
                        alert.timestamp - previous_time
                    ).total_seconds()

                    if elapsed < self.config.alert_cooldown_s:
                        continue

                self._last_alert_time[cooldown_key] = (
                    alert.timestamp
                )

                self._alerts.append(alert)
                accepted_alerts.append(alert)

            self._audit.append(
                PerceptionAuditEvent(
                    event_id=generate_id(),
                    tenant_id=telemetry.tenant_id,
                    user_id=access.user_id,
                    robot_id=telemetry.robot_id,
                    action="perception_monitor.ingest",
                    resource_id=telemetry.telemetry_id,
                )
            )

            return tuple(accepted_alerts)

    def component_health(
        self,
        *,
        robot_id: str,
        component: PerceptionComponent,
        sensor_id: str,
        access: PerceptionAccessContext,
        now: datetime | None = None,
    ) -> ComponentHealthReport:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_view",
        )

        now = validate_timestamp(
            now or utc_now()
        )

        key = (
            access.tenant_id,
            robot_id,
            component,
            sensor_id,
        )

        with self._lock:
            samples = tuple(
                self._samples.get(key, ())
            )

        if not samples:
            return ComponentHealthReport(
                tenant_id=access.tenant_id,
                robot_id=robot_id,
                sensor_id=sensor_id,
                component=component,
                status=HealthStatus.UNKNOWN,
                health_score=0.0,
                last_seen=None,
                sensor_age_s=None,
                metrics=None,
                warnings=("No telemetry received",),
                recommendation=(
                    OperationalRecommendation.PERCEPTION_UNAVAILABLE
                ),
                generated_at=now,
            )

        latest = samples[-1]

        age = max(
            0.0,
            (now - latest.timestamp).total_seconds(),
        )

        metrics = self.aggregator.summarize(
            samples
        )

        score = self.scorer.score(
            metrics,
            age,
        )

        status = self.scorer.classify(
            score,
            age,
        )

        warnings = []

        if age >= self.config.warning_sensor_age_s:
            warnings.append(
                f"Sensor telemetry age: {age:.2f}s"
            )

        if (
            metrics.p95_latency_ms
            >= self.config.warning_latency_ms
        ):
            warnings.append(
                "Perception latency exceeds warning threshold"
            )

        if (
            metrics.frame_drop_ratio
            >= self.config.warning_frame_drop_ratio
        ):
            warnings.append(
                "Frame drop ratio exceeds warning threshold"
            )

        if (
            metrics.inference_error_ratio
            >= self.config.warning_error_ratio
        ):
            warnings.append(
                "Inference error ratio exceeds warning threshold"
            )

        if (
            metrics.average_fps > 0
            and metrics.average_fps < self.config.minimum_fps
        ):
            warnings.append(
                "Frame rate below configured minimum"
            )

        return ComponentHealthReport(
            tenant_id=access.tenant_id,
            robot_id=robot_id,
            sensor_id=sensor_id,
            component=component,
            status=status,
            health_score=score,
            last_seen=latest.timestamp,
            sensor_age_s=age,
            metrics=metrics,
            warnings=tuple(warnings),
            recommendation=(
                self.recommendation_engine.recommend(status)
            ),
            generated_at=now,
        )

    def robot_health(
        self,
        *,
        robot_id: str,
        access: PerceptionAccessContext,
        now: datetime | None = None,
    ) -> RobotPerceptionHealthReport:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_view",
        )

        now = validate_timestamp(
            now or utc_now()
        )

        with self._lock:
            keys = [
                key
                for key in self._samples
                if (
                    key[0] == access.tenant_id
                    and key[1] == robot_id
                )
            ]

        reports = []

        for _, _, component, sensor_id in keys:
            reports.append(
                self.component_health(
                    robot_id=robot_id,
                    component=component,
                    sensor_id=sensor_id,
                    access=access,
                    now=now,
                )
            )

        if not reports:
            overall_status = HealthStatus.UNKNOWN
            overall_score = 0.0

        else:
            overall_score = safe_mean(
                [
                    report.health_score
                    for report in reports
                ]
            )

            statuses = {
                report.status
                for report in reports
            }

            if HealthStatus.OFFLINE in statuses:
                overall_status = HealthStatus.OFFLINE

            elif HealthStatus.CRITICAL in statuses:
                overall_status = HealthStatus.CRITICAL

            elif HealthStatus.DEGRADED in statuses:
                overall_status = HealthStatus.DEGRADED

            elif HealthStatus.UNKNOWN in statuses:
                overall_status = HealthStatus.UNKNOWN

            else:
                overall_status = HealthStatus.HEALTHY

        warning_count = sum(
            report.status == HealthStatus.DEGRADED
            for report in reports
        )

        critical_count = sum(
            report.status in (
                HealthStatus.CRITICAL,
                HealthStatus.OFFLINE,
            )
            for report in reports
        )

        return RobotPerceptionHealthReport(
            report_id=generate_id(),
            tenant_id=access.tenant_id,
            robot_id=robot_id,
            overall_status=overall_status,
            overall_health_score=overall_score,
            component_reports=tuple(reports),
            warning_count=warning_count,
            critical_count=critical_count,
            recommendation=(
                self.recommendation_engine.recommend(
                    overall_status
                )
            ),
            generated_at=now,
        )

    def alerts(
        self,
        *,
        robot_id: str,
        access: PerceptionAccessContext,
        severity: AlertSeverity | None = None,
    ) -> tuple[PerceptionAlert, ...]:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_view",
        )

        with self._lock:
            return tuple(
                alert
                for alert in self._alerts
                if (
                    alert.tenant_id == access.tenant_id
                    and alert.robot_id == robot_id
                    and (
                        severity is None
                        or alert.severity == severity
                    )
                )
            )

    def audit_events(
        self,
        *,
        robot_id: str,
        access: PerceptionAccessContext,
    ) -> tuple[PerceptionAuditEvent, ...]:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_view",
        )

        with self._lock:
            return tuple(
                event
                for event in self._audit
                if (
                    event.tenant_id == access.tenant_id
                    and event.robot_id == robot_id
                )
            )

    def reset_component(
        self,
        *,
        robot_id: str,
        component: PerceptionComponent,
        sensor_id: str,
        access: PerceptionAccessContext,
    ) -> None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_manage",
        )

        key = (
            access.tenant_id,
            robot_id,
            component,
            sensor_id,
        )

        with self._lock:
            self._samples.pop(key, None)

            for alert_key in list(self._last_alert_time):
                if alert_key[:4] == key:
                    del self._last_alert_time[alert_key]

            self._audit.append(
                PerceptionAuditEvent(
                    event_id=generate_id(),
                    tenant_id=access.tenant_id,
                    user_id=access.user_id,
                    robot_id=robot_id,
                    action="perception_monitor.reset_component",
                    resource_id=sensor_id,
                )
            )

    @staticmethod
    def export_json(
        report: RobotPerceptionHealthReport,
    ) -> str:
        return json.dumps(
            report.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# TELEMETRY FACTORY
# ============================================================

class PerceptionTelemetryFactory:

    @staticmethod
    def create(
        *,
        tenant_id: str,
        robot_id: str,
        component: PerceptionComponent,
        sensor_id: str,
        processing_latency_ms: float,
        fps: float | None = None,
        processed_frames: int = 0,
        dropped_frames: int = 0,
        inference_errors: int = 0,
        detection_count: int = 0,
        average_detection_confidence: float | None = None,
        active_tracks: int = 0,
        tracking_quality: float | None = None,
        valid_data_ratio: float = 1.0,
        consistency_score: float | None = None,
        timestamp: datetime | None = None,
    ) -> PerceptionTelemetry:
        return PerceptionTelemetry(
            telemetry_id=generate_id(),
            tenant_id=tenant_id,
            robot_id=robot_id,
            component=component,
            sensor_id=sensor_id,
            timestamp=timestamp or utc_now(),
            processing_latency_ms=processing_latency_ms,
            fps=fps,
            processed_frames=processed_frames,
            dropped_frames=dropped_frames,
            inference_errors=inference_errors,
            detection_count=detection_count,
            average_detection_confidence=(
                average_detection_confidence
            ),
            active_tracks=active_tracks,
            tracking_quality=tracking_quality,
            valid_data_ratio=valid_data_ratio,
            consistency_score=consistency_score,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestPerceptionMonitor(unittest.TestCase):

    def setUp(self) -> None:
        self.config = PerceptionMonitorConfig()

        self.monitor = PerceptionMonitor(
            self.config
        )

        self.access = PerceptionAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_ingest=True,
            can_view=True,
            can_manage=True,
        )

        self.now = utc_now()

    def make_telemetry(
        self,
        **overrides: Any,
    ) -> PerceptionTelemetry:
        values = {
            "tenant_id": "tenant_alpha",
            "robot_id": "ROBOT-001",
            "component": PerceptionComponent.CAMERA,
            "sensor_id": "CAM-001",
            "processing_latency_ms": 30.0,
            "fps": 30.0,
            "processed_frames": 100,
            "dropped_frames": 0,
            "inference_errors": 0,
            "valid_data_ratio": 1.0,
            "timestamp": self.now,
        }

        values.update(overrides)

        return PerceptionTelemetryFactory.create(
            **values
        )

    def test_valid_telemetry(self) -> None:
        telemetry = self.make_telemetry()

        self.assertEqual(
            telemetry.component,
            PerceptionComponent.CAMERA,
        )

    def test_invalid_confidence(self) -> None:
        with self.assertRaises(
            PerceptionValidationError
        ):
            self.make_telemetry(
                average_detection_confidence=1.5
            )

    def test_ingest(self) -> None:
        alerts = self.monitor.ingest(
            self.make_telemetry(),
            access=self.access,
        )

        self.assertEqual(
            len(alerts),
            0,
        )

    def test_healthy_component(self) -> None:
        self.monitor.ingest(
            self.make_telemetry(),
            access=self.access,
        )

        report = self.monitor.component_health(
            robot_id="ROBOT-001",
            component=PerceptionComponent.CAMERA,
            sensor_id="CAM-001",
            access=self.access,
            now=self.now,
        )

        self.assertEqual(
            report.status,
            HealthStatus.HEALTHY,
        )

    def test_high_latency_alert(self) -> None:
        alerts = self.monitor.ingest(
            self.make_telemetry(
                processing_latency_ms=700.0
            ),
            access=self.access,
        )

        self.assertTrue(
            any(
                alert.category == AlertCategory.HIGH_LATENCY
                for alert in alerts
            )
        )

    def test_low_fps_alert(self) -> None:
        alerts = self.monitor.ingest(
            self.make_telemetry(
                fps=2.0
            ),
            access=self.access,
        )

        self.assertTrue(
            any(
                alert.category == AlertCategory.LOW_FPS
                for alert in alerts
            )
        )

    def test_frame_drop_alert(self) -> None:
        alerts = self.monitor.ingest(
            self.make_telemetry(
                processed_frames=60,
                dropped_frames=40,
            ),
            access=self.access,
        )

        self.assertTrue(
            any(
                alert.category == AlertCategory.HIGH_FRAME_DROP
                for alert in alerts
            )
        )

    def test_inference_error_alert(self) -> None:
        alerts = self.monitor.ingest(
            self.make_telemetry(
                processed_frames=100,
                inference_errors=30,
            ),
            access=self.access,
        )

        self.assertTrue(
            any(
                alert.category == AlertCategory.HIGH_ERROR_RATE
                for alert in alerts
            )
        )

    def test_offline_sensor(self) -> None:
        self.monitor.ingest(
            self.make_telemetry(),
            access=self.access,
        )

        report = self.monitor.component_health(
            robot_id="ROBOT-001",
            component=PerceptionComponent.CAMERA,
            sensor_id="CAM-001",
            access=self.access,
            now=self.now + timedelta(seconds=10),
        )

        self.assertEqual(
            report.status,
            HealthStatus.OFFLINE,
        )

    def test_robot_health(self) -> None:
        self.monitor.ingest(
            self.make_telemetry(),
            access=self.access,
        )

        report = self.monitor.robot_health(
            robot_id="ROBOT-001",
            access=self.access,
            now=self.now,
        )

        self.assertEqual(
            report.overall_status,
            HealthStatus.HEALTHY,
        )

    def test_unknown_component(self) -> None:
        report = self.monitor.component_health(
            robot_id="ROBOT-001",
            component=PerceptionComponent.LIDAR,
            sensor_id="LIDAR-001",
            access=self.access,
            now=self.now,
        )

        self.assertEqual(
            report.status,
            HealthStatus.UNKNOWN,
        )

    def test_alert_deduplication(self) -> None:
        first = self.make_telemetry(
            processing_latency_ms=700.0
        )

        second = self.make_telemetry(
            processing_latency_ms=800.0,
            timestamp=self.now + timedelta(seconds=1),
        )

        self.monitor.ingest(
            first,
            access=self.access,
        )

        alerts = self.monitor.ingest(
            second,
            access=self.access,
        )

        self.assertFalse(
            any(
                alert.category == AlertCategory.HIGH_LATENCY
                for alert in alerts
            )
        )

    def test_tenant_isolation(self) -> None:
        with self.assertRaises(
            PerceptionAuthorizationError
        ):
            self.monitor._authorize(
                self.access,
                "tenant_beta",
                "ROBOT-001",
                "can_view",
            )

    def test_out_of_order_telemetry(self) -> None:
        self.monitor.ingest(
            self.make_telemetry(),
            access=self.access,
        )

        with self.assertRaises(
            PerceptionValidationError
        ):
            self.monitor.ingest(
                self.make_telemetry(
                    timestamp=(
                        self.now
                        - timedelta(seconds=1)
                    )
                ),
                access=self.access,
            )

    def test_reset_component(self) -> None:
        self.monitor.ingest(
            self.make_telemetry(),
            access=self.access,
        )

        self.monitor.reset_component(
            robot_id="ROBOT-001",
            component=PerceptionComponent.CAMERA,
            sensor_id="CAM-001",
            access=self.access,
        )

        report = self.monitor.component_health(
            robot_id="ROBOT-001",
            component=PerceptionComponent.CAMERA,
            sensor_id="CAM-001",
            access=self.access,
        )

        self.assertEqual(
            report.status,
            HealthStatus.UNKNOWN,
        )

    def test_json_export(self) -> None:
        self.monitor.ingest(
            self.make_telemetry(),
            access=self.access,
        )

        report = self.monitor.robot_health(
            robot_id="ROBOT-001",
            access=self.access,
            now=self.now,
        )

        payload = self.monitor.export_json(
            report
        )

        self.assertIn(
            "ROBOT-001",
            payload,
        )

    def test_percentile(self) -> None:
        self.assertAlmostEqual(
            percentile([10, 20, 30, 40], 50),
            25.0,
        )

    def test_health_score_bounds(self) -> None:
        self.assertEqual(
            bounded_score(-50),
            0.0,
        )

        self.assertEqual(
            bounded_score(150),
            100.0,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Perception Health & Reliability Monitor\n"
    )

    access = PerceptionAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
        }),
        can_ingest=True,
        can_view=True,
        can_manage=True,
    )

    monitor = PerceptionMonitor()

    timestamp = utc_now()

    samples = [
        PerceptionTelemetryFactory.create(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            component=PerceptionComponent.CAMERA,
            sensor_id="CAM-FRONT-001",
            processing_latency_ms=25.0,
            fps=30.0,
            processed_frames=100,
            dropped_frames=1,
            timestamp=timestamp,
        ),
        PerceptionTelemetryFactory.create(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            component=PerceptionComponent.LIDAR,
            sensor_id="LIDAR-001",
            processing_latency_ms=40.0,
            fps=20.0,
            processed_frames=100,
            dropped_frames=0,
            valid_data_ratio=0.98,
            timestamp=timestamp,
        ),
        PerceptionTelemetryFactory.create(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            component=PerceptionComponent.OBJECT_DETECTION,
            sensor_id="CAM-FRONT-001",
            processing_latency_ms=180.0,
            fps=8.0,
            processed_frames=100,
            dropped_frames=8,
            detection_count=12,
            average_detection_confidence=0.82,
            timestamp=timestamp,
        ),
        PerceptionTelemetryFactory.create(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            component=PerceptionComponent.OBJECT_TRACKING,
            sensor_id="CAM-FRONT-001",
            processing_latency_ms=18.0,
            fps=25.0,
            processed_frames=100,
            dropped_frames=2,
            active_tracks=7,
            tracking_quality=0.91,
            timestamp=timestamp,
        ),
        PerceptionTelemetryFactory.create(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            component=PerceptionComponent.OPTICAL_FLOW,
            sensor_id="CAM-FRONT-001",
            processing_latency_ms=45.0,
            fps=22.0,
            processed_frames=100,
            dropped_frames=3,
            valid_data_ratio=0.99,
            timestamp=timestamp,
        ),
    ]

    for sample in samples:
        alerts = monitor.ingest(
            sample,
            access=access,
        )

        print(
            f"{sample.component.value}: "
            f"latency={sample.processing_latency_ms:.1f}ms, "
            f"fps={sample.fps}, "
            f"alerts={len(alerts)}"
        )

    report = monitor.robot_health(
        robot_id="ROBOT-001",
        access=access,
        now=timestamp,
    )

    print("\nROBOT PERCEPTION HEALTH REPORT\n")

    print(
        monitor.export_json(report)
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestPerceptionMonitor
    )

    result = unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
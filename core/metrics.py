"""Metrics module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Core Metrics Module

Provides:
- Prometheus-compatible application metrics
- HTTP request counters and latency histograms
- Robot and fleet health metrics
- Sensor and telemetry metrics
- Navigation and manipulation metrics
- AI inference and model evaluation metrics
- VLA and world model metrics
- Predictive maintenance and repair metrics
- Safety and incident metrics
- Cache and event bus metrics
- Background task and resource metrics
- FastAPI /metrics endpoint integration

Install:
    pip install prometheus-client fastapi starlette

Security:
- Never use tenant IDs, user IDs, robot IDs, task IDs,
  request IDs, or other unbounded identifiers as labels.
- Metrics must not contain secrets or personal information.
- Safety metrics are observational and must not replace
  independent physical safety systems.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import math
import re
import time

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, TypeVar

from fastapi import APIRouter, FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.types import ASGIApp, Message, Receive, Scope, Send


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

METRICS_NAMESPACE = "physical_ai"

DEFAULT_LATENCY_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)

DEFAULT_INFERENCE_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)

DEFAULT_SCORE_BUCKETS = (
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    1.0,
)

MAX_LABEL_LENGTH = 64

_LABEL_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]")


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class MetricsSettings:
    enabled: bool = True
    namespace: str = METRICS_NAMESPACE
    endpoint: str = "/metrics"
    include_http_metrics: bool = True
    include_robotics_metrics: bool = True
    include_ai_metrics: bool = True
    include_safety_metrics: bool = True
    include_infrastructure_metrics: bool = True
    exclude_health_requests: bool = True

    def __post_init__(self) -> None:
        if not re.fullmatch(
            r"[a-zA-Z_:][a-zA-Z0-9_:]*",
            self.namespace,
        ):
            raise ValueError("Invalid Prometheus namespace")

        if not self.endpoint.startswith("/"):
            raise ValueError(
                "Metrics endpoint must start with '/'"
            )


# ============================================================
# Safe Label Utilities
# ============================================================


def safe_label(
    value: Any,
    *,
    allowed: set[str] | frozenset[str] | None = None,
    default: str = "unknown",
) -> str:
    """
    Normalize a bounded categorical metric label.

    For potentially unbounded values, supply an allowlist.
    Never pass raw identifiers as metric labels.
    """

    if value is None:
        return default

    normalized = str(value).strip().lower()
    normalized = _LABEL_PATTERN.sub("_", normalized)
    normalized = normalized[:MAX_LABEL_LENGTH]

    if not normalized:
        return default

    if allowed is not None and normalized not in allowed:
        return default

    return normalized


ROBOT_STATES = frozenset({
    "online",
    "offline",
    "idle",
    "ready",
    "working",
    "navigating",
    "manipulating",
    "charging",
    "maintenance",
    "degraded",
    "fault",
    "emergency_stopped",
    "unknown",
})

HEALTH_LEVELS = frozenset({
    "healthy",
    "good",
    "warning",
    "degraded",
    "critical",
    "failed",
    "unknown",
})

SEVERITY_LEVELS = frozenset({
    "info",
    "low",
    "medium",
    "high",
    "critical",
    "unknown",
})

OUTCOMES = frozenset({
    "success",
    "failure",
    "timeout",
    "cancelled",
    "rejected",
    "partial",
    "unknown",
})

AI_MODEL_TYPES = frozenset({
    "llm",
    "vlm",
    "vla",
    "world_model",
    "perception",
    "navigation",
    "manipulation",
    "policy",
    "anomaly",
    "rul",
    "forecasting",
    "embedding",
    "reranker",
    "other",
})

EVALUATION_TYPES = frozenset({
    "perception",
    "navigation",
    "manipulation",
    "vla",
    "world_model",
    "policy",
    "rag",
    "safety",
    "repair",
    "latency",
    "resource",
    "end_to_end",
    "other",
})

SENSOR_TYPES = frozenset({
    "camera",
    "depth_camera",
    "thermal_camera",
    "lidar",
    "radar",
    "imu",
    "gps",
    "gnss",
    "temperature",
    "vibration",
    "pressure",
    "force",
    "torque",
    "voltage",
    "current",
    "battery",
    "ultrasonic",
    "other",
})

MAINTENANCE_TYPES = frozenset({
    "preventive",
    "predictive",
    "corrective",
    "condition_based",
    "emergency",
    "inspection",
    "calibration",
    "other",
})

REPAIR_DECISIONS = frozenset({
    "repair",
    "replace",
    "inspect",
    "monitor",
    "defer",
    "undetermined",
})

EVENT_CATEGORIES = frozenset({
    "system",
    "robot",
    "robot_health",
    "telemetry",
    "sensor",
    "navigation",
    "manipulation",
    "perception",
    "safety",
    "incident",
    "maintenance",
    "repair",
    "parts",
    "fleet",
    "model",
    "inference",
    "world_model",
    "vla",
    "simulation",
    "evaluation",
    "security",
    "audit",
    "notification",
    "custom",
    "other",
})

CACHE_OPERATIONS = frozenset({
    "get",
    "set",
    "delete",
    "invalidate",
    "exists",
    "other",
})

CACHE_BACKENDS = frozenset({
    "memory",
    "redis",
    "other",
})


# ============================================================
# Metric Registry
# ============================================================


class MetricsRegistry:
    """
    Central metric collector.

    A dedicated registry avoids duplicate metric registration
    and allows isolated test instances.
    """

    def __init__(
        self,
        settings: MetricsSettings | None = None,
        *,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self.settings = settings or MetricsSettings()

        self.registry = registry or CollectorRegistry(
            auto_describe=True
        )

        self.namespace = self.settings.namespace

        self._initialize_http_metrics()
        self._initialize_robotics_metrics()
        self._initialize_ai_metrics()
        self._initialize_maintenance_metrics()
        self._initialize_safety_metrics()
        self._initialize_infrastructure_metrics()

    def _counter(
        self,
        name: str,
        description: str,
        labels: tuple[str, ...] = (),
    ) -> Counter:
        return Counter(
            name,
            description,
            labelnames=labels,
            namespace=self.namespace,
            registry=self.registry,
        )

    def _gauge(
        self,
        name: str,
        description: str,
        labels: tuple[str, ...] = (),
    ) -> Gauge:
        return Gauge(
            name,
            description,
            labelnames=labels,
            namespace=self.namespace,
            registry=self.registry,
        )

    def _histogram(
        self,
        name: str,
        description: str,
        labels: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS,
    ) -> Histogram:
        return Histogram(
            name,
            description,
            labelnames=labels,
            buckets=buckets,
            namespace=self.namespace,
            registry=self.registry,
        )

    # ========================================================
    # HTTP Metrics
    # ========================================================

    def _initialize_http_metrics(self) -> None:
        self.http_requests_total = self._counter(
            "http_requests_total",
            "Total HTTP requests",
            ("method", "route", "status"),
        )

        self.http_request_duration_seconds = self._histogram(
            "http_request_duration_seconds",
            "HTTP request latency in seconds",
            ("method", "route"),
        )

        self.http_requests_in_progress = self._gauge(
            "http_requests_in_progress",
            "Current HTTP requests",
            ("method",),
        )

        self.http_exceptions_total = self._counter(
            "http_exceptions_total",
            "HTTP exceptions",
            ("exception_type",),
        )

    # ========================================================
    # Robotics Metrics
    # ========================================================

    def _initialize_robotics_metrics(self) -> None:
        self.robots_total = self._gauge(
            "robots_total",
            "Number of registered robots",
        )

        self.robots_by_state = self._gauge(
            "robots_by_state",
            "Number of robots in each lifecycle state",
            ("state",),
        )

        self.robot_health_score = self._histogram(
            "robot_health_score",
            "Distribution of robot health scores",
            ("health_level",),
            buckets=(
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
            ),
        )

        self.robot_state_transitions_total = self._counter(
            "robot_state_transitions_total",
            "Robot state transitions",
            ("from_state", "to_state"),
        )

        self.robot_tasks_total = self._counter(
            "robot_tasks_total",
            "Robot tasks by outcome",
            ("task_type", "outcome"),
        )

        self.robot_task_duration_seconds = self._histogram(
            "robot_task_duration_seconds",
            "Robot task duration",
            ("task_type",),
            buckets=DEFAULT_INFERENCE_BUCKETS,
        )

        self.fleets_total = self._gauge(
            "fleets_total",
            "Number of robot fleets",
        )

        self.telemetry_messages_total = self._counter(
            "telemetry_messages_total",
            "Total telemetry messages received",
            ("transport",),
        )

        self.telemetry_anomalies_total = self._counter(
            "telemetry_anomalies_total",
            "Telemetry anomalies detected",
            ("severity",),
        )

        self.telemetry_ingestion_duration_seconds = self._histogram(
            "telemetry_ingestion_duration_seconds",
            "Telemetry ingestion duration",
        )

        self.sensor_readings_total = self._counter(
            "sensor_readings_total",
            "Sensor readings processed",
            ("sensor_type",),
        )

        self.sensor_failures_total = self._counter(
            "sensor_failures_total",
            "Sensor failures detected",
            ("sensor_type",),
        )

        self.sensor_health_score = self._histogram(
            "sensor_health_score",
            "Sensor health score distribution",
            ("sensor_type",),
            buckets=(
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
            ),
        )

        self.navigation_sessions_total = self._counter(
            "navigation_sessions_total",
            "Navigation sessions",
            ("outcome",),
        )

        self.navigation_duration_seconds = self._histogram(
            "navigation_duration_seconds",
            "Navigation execution duration",
            ("outcome",),
            buckets=DEFAULT_INFERENCE_BUCKETS,
        )

        self.navigation_distance_meters = self._histogram(
            "navigation_distance_meters",
            "Distance traveled during navigation",
            buckets=(
                0.1,
                0.5,
                1,
                2,
                5,
                10,
                25,
                50,
                100,
                250,
                500,
                1000,
            ),
        )

        self.manipulation_tasks_total = self._counter(
            "manipulation_tasks_total",
            "Manipulation tasks",
            ("task_type", "outcome"),
        )

        self.manipulation_duration_seconds = self._histogram(
            "manipulation_duration_seconds",
            "Manipulation task duration",
            ("task_type",),
            buckets=DEFAULT_INFERENCE_BUCKETS,
        )

        self.perception_predictions_total = self._counter(
            "perception_predictions_total",
            "Perception model predictions",
            ("task_type", "outcome"),
        )

        self.perception_confidence = self._histogram(
            "perception_confidence",
            "Perception prediction confidence",
            ("task_type",),
            buckets=DEFAULT_SCORE_BUCKETS,
        )

    # ========================================================
    # AI Metrics
    # ========================================================

    def _initialize_ai_metrics(self) -> None:
        self.ai_inference_requests_total = self._counter(
            "ai_inference_requests_total",
            "AI inference requests",
            ("model_type", "outcome"),
        )

        self.ai_inference_duration_seconds = self._histogram(
            "ai_inference_duration_seconds",
            "AI inference latency",
            ("model_type",),
            buckets=DEFAULT_INFERENCE_BUCKETS,
        )

        self.ai_inference_in_progress = self._gauge(
            "ai_inference_in_progress",
            "Currently running AI inference requests",
            ("model_type",),
        )

        self.ai_inference_tokens_total = self._counter(
            "ai_inference_tokens_total",
            "AI model token usage",
            ("model_type", "token_type"),
        )

        self.vla_predictions_total = self._counter(
            "vla_predictions_total",
            "VLA predictions",
            ("outcome",),
        )

        self.vla_action_rejections_total = self._counter(
            "vla_action_rejections_total",
            "Rejected VLA actions",
            ("reason",),
        )

        self.world_model_predictions_total = self._counter(
            "world_model_predictions_total",
            "World model predictions",
            ("outcome",),
        )

        self.world_model_prediction_error = self._histogram(
            "world_model_prediction_error",
            "World model prediction error",
            buckets=(
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.2,
                0.5,
                1,
                2,
                5,
                10,
            ),
        )

        self.agent_executions_total = self._counter(
            "agent_executions_total",
            "AI agent executions",
            ("agent_type", "outcome"),
        )

        self.agent_execution_duration_seconds = self._histogram(
            "agent_execution_duration_seconds",
            "AI agent execution duration",
            ("agent_type",),
            buckets=DEFAULT_INFERENCE_BUCKETS,
        )

        self.rag_queries_total = self._counter(
            "rag_queries_total",
            "RAG queries",
            ("outcome",),
        )

        self.rag_retrieval_duration_seconds = self._histogram(
            "rag_retrieval_duration_seconds",
            "RAG retrieval duration",
        )

        self.rag_groundedness_score = self._histogram(
            "rag_groundedness_score",
            "RAG answer groundedness",
            buckets=DEFAULT_SCORE_BUCKETS,
        )

        self.model_evaluations_total = self._counter(
            "model_evaluations_total",
            "Model evaluations",
            ("evaluation_type", "outcome"),
        )

        self.evaluation_score = self._histogram(
            "evaluation_score",
            "Normalized model evaluation scores",
            ("evaluation_type",),
            buckets=DEFAULT_SCORE_BUCKETS,
        )

        self.simulation_runs_total = self._counter(
            "simulation_runs_total",
            "Simulation runs",
            ("engine", "outcome"),
        )

        self.simulation_duration_seconds = self._histogram(
            "simulation_duration_seconds",
            "Simulation execution duration",
            ("engine",),
            buckets=(
                1,
                5,
                10,
                30,
                60,
                120,
                300,
                600,
                1800,
                3600,
                7200,
            ),
        )

    # ========================================================
    # Maintenance and Repair Metrics
    # ========================================================

    def _initialize_maintenance_metrics(self) -> None:
        self.maintenance_predictions_total = self._counter(
            "maintenance_predictions_total",
            "Predictive maintenance results",
            ("risk_level",),
        )

        self.maintenance_operations_total = self._counter(
            "maintenance_operations_total",
            "Maintenance operations",
            ("maintenance_type", "outcome"),
        )

        self.failure_predictions_total = self._counter(
            "failure_predictions_total",
            "Predicted component failures",
            ("risk_level",),
        )

        self.rul_prediction_hours = self._histogram(
            "rul_prediction_hours",
            "Predicted remaining useful life in hours",
            buckets=(
                1,
                6,
                12,
                24,
                48,
                72,
                168,
                336,
                720,
                1440,
                4320,
                8760,
            ),
        )

        self.repair_estimates_total = self._counter(
            "repair_estimates_total",
            "Repair cost estimates",
            ("decision",),
        )

        self.repair_operations_total = self._counter(
            "repair_operations_total",
            "Robot repair operations",
            ("outcome",),
        )

        self.repair_estimate_amount = self._histogram(
            "repair_estimate_amount",
            "Repair estimate amount in stated currency",
            ("currency",),
            buckets=(
                100,
                500,
                1000,
                2500,
                5000,
                10000,
                25000,
                50000,
                100000,
                250000,
                500000,
                1000000,
            ),
        )

        self.part_inspections_total = self._counter(
            "part_inspections_total",
            "Robot part inspections",
            ("outcome",),
        )

    # ========================================================
    # Safety Metrics
    # ========================================================

    def _initialize_safety_metrics(self) -> None:
        self.safety_alerts_total = self._counter(
            "safety_alerts_total",
            "Robot safety alerts",
            ("severity",),
        )

        self.safety_violations_total = self._counter(
            "safety_violations_total",
            "Safety policy violations",
            ("severity",),
        )

        self.emergency_stop_reports_total = self._counter(
            "emergency_stop_reports_total",
            "Reported emergency stop activations",
        )

        self.incidents_total = self._counter(
            "incidents_total",
            "Safety incidents",
            ("severity",),
        )

        self.safety_checks_total = self._counter(
            "safety_checks_total",
            "Safety validation checks",
            ("outcome",),
        )

        self.safety_action_rejections_total = self._counter(
            "safety_action_rejections_total",
            "Unsafe actions rejected",
            ("reason",),
        )

        self.active_safety_incidents = self._gauge(
            "active_safety_incidents",
            "Currently active safety incidents",
            ("severity",),
        )

    # ========================================================
    # Infrastructure Metrics
    # ========================================================

    def _initialize_infrastructure_metrics(self) -> None:
        self.cache_operations_total = self._counter(
            "cache_operations_total",
            "Cache operations",
            ("backend", "operation", "outcome"),
        )

        self.cache_operation_duration_seconds = self._histogram(
            "cache_operation_duration_seconds",
            "Cache operation latency",
            ("backend", "operation"),
        )

        self.event_bus_events_total = self._counter(
            "event_bus_events_total",
            "Event bus events",
            ("category", "outcome"),
        )

        self.event_bus_queue_size = self._gauge(
            "event_bus_queue_size",
            "Current event bus queue size",
        )

        self.event_bus_handler_duration_seconds = self._histogram(
            "event_bus_handler_duration_seconds",
            "Event handler processing latency",
            ("category",),
        )

        self.background_tasks_active = self._gauge(
            "background_tasks_active",
            "Number of active background tasks",
        )

        self.background_task_failures_total = self._counter(
            "background_task_failures_total",
            "Background task failures",
            ("task_category",),
        )

        self.application_uptime_seconds = self._gauge(
            "application_uptime_seconds",
            "Application uptime in seconds",
        )

        self.resource_cpu_utilization = self._gauge(
            "resource_cpu_utilization_percent",
            "CPU utilization percentage",
        )

        self.resource_memory_bytes = self._gauge(
            "resource_memory_bytes",
            "Memory usage in bytes",
        )

        self.resource_gpu_utilization = self._gauge(
            "resource_gpu_utilization_percent",
            "GPU utilization percentage",
        )

        self.resource_gpu_memory_bytes = self._gauge(
            "resource_gpu_memory_bytes",
            "GPU memory usage in bytes",
        )

    # ========================================================
    # Metric Export
    # ========================================================

    def export(self) -> bytes:
        return generate_latest(self.registry)

    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST


# ============================================================
# Metric Recorder
# ============================================================


class MetricsRecorder:
    """
    Domain-oriented metric recording interface.

    Labels are normalized and bounded to prevent
    unbounded Prometheus time-series cardinality.
    """

    def __init__(
        self,
        registry: MetricsRegistry,
    ) -> None:
        self.registry = registry
        self.started_at = time.monotonic()

    @staticmethod
    def _nonnegative(value: float) -> float:
        number = float(value)

        if not math.isfinite(number) or number < 0:
            raise ValueError(
                "Metric value must be finite and nonnegative"
            )

        return number

    @staticmethod
    def _score(
        value: float,
        *,
        maximum: float = 1.0,
    ) -> float:
        number = float(value)

        if not math.isfinite(number):
            raise ValueError("Metric score must be finite")

        if not 0 <= number <= maximum:
            raise ValueError(
                f"Score must be between 0 and {maximum}"
            )

        return number

    def record_http_request(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        if not self.registry.settings.include_http_metrics:
            return

        method_label = safe_label(
            method,
            allowed={
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "head",
                "options",
            },
        )

        # Route must be a registered route template,
        # never a raw URL containing IDs.
        route_label = (
            route
            if route.startswith("/")
            and len(route) <= 120
            and not re.search(r"/\d+(?:/|$)", route)
            else "unmatched"
        )

        status_label = str(int(status_code))

        self.registry.http_requests_total.labels(
            method=method_label,
            route=route_label,
            status=status_label,
        ).inc()

        self.registry.http_request_duration_seconds.labels(
            method=method_label,
            route=route_label,
        ).observe(
            self._nonnegative(duration_seconds)
        )

    def set_robot_count(self, count: int) -> None:
        self.registry.robots_total.set(
            self._nonnegative(count)
        )

    def set_fleet_count(self, count: int) -> None:
        self.registry.fleets_total.set(
            self._nonnegative(count)
        )

    def set_robots_by_state(
        self,
        state: str,
        count: int,
    ) -> None:
        self.registry.robots_by_state.labels(
            state=safe_label(
                state,
                allowed=ROBOT_STATES,
            )
        ).set(
            self._nonnegative(count)
        )

    def record_robot_health(
        self,
        *,
        score: float,
        health_level: str,
    ) -> None:
        self.registry.robot_health_score.labels(
            health_level=safe_label(
                health_level,
                allowed=HEALTH_LEVELS,
            )
        ).observe(
            self._score(score, maximum=100)
        )

    def record_robot_state_transition(
        self,
        *,
        previous_state: str,
        current_state: str,
    ) -> None:
        self.registry.robot_state_transitions_total.labels(
            from_state=safe_label(
                previous_state,
                allowed=ROBOT_STATES,
            ),
            to_state=safe_label(
                current_state,
                allowed=ROBOT_STATES,
            ),
        ).inc()

    def record_robot_task(
        self,
        *,
        task_type: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        task = safe_label(
            task_type,
            allowed={
                "navigation",
                "manipulation",
                "inspection",
                "delivery",
                "maintenance",
                "charging",
                "simulation",
                "other",
            },
            default="other",
        )

        result = safe_label(
            outcome,
            allowed=OUTCOMES,
        )

        self.registry.robot_tasks_total.labels(
            task_type=task,
            outcome=result,
        ).inc()

        self.registry.robot_task_duration_seconds.labels(
            task_type=task
        ).observe(
            self._nonnegative(duration_seconds)
        )

    def record_telemetry(
        self,
        *,
        transport: str,
        count: int = 1,
    ) -> None:
        transport_label = safe_label(
            transport,
            allowed={
                "http",
                "websocket",
                "mqtt",
                "kafka",
                "ros",
                "ros2",
                "grpc",
                "other",
            },
        )

        self.registry.telemetry_messages_total.labels(
            transport=transport_label
        ).inc(
            self._nonnegative(count)
        )

    def record_telemetry_anomaly(
        self,
        severity: str,
    ) -> None:
        self.registry.telemetry_anomalies_total.labels(
            severity=safe_label(
                severity,
                allowed=SEVERITY_LEVELS,
            )
        ).inc()

    def record_sensor_reading(
        self,
        sensor_type: str,
    ) -> None:
        self.registry.sensor_readings_total.labels(
            sensor_type=safe_label(
                sensor_type,
                allowed=SENSOR_TYPES,
                default="other",
            )
        ).inc()

    def record_sensor_failure(
        self,
        sensor_type: str,
    ) -> None:
        self.registry.sensor_failures_total.labels(
            sensor_type=safe_label(
                sensor_type,
                allowed=SENSOR_TYPES,
                default="other",
            )
        ).inc()

    def record_sensor_health(
        self,
        *,
        sensor_type: str,
        score: float,
    ) -> None:
        self.registry.sensor_health_score.labels(
            sensor_type=safe_label(
                sensor_type,
                allowed=SENSOR_TYPES,
                default="other",
            )
        ).observe(
            self._score(score, maximum=100)
        )

    def record_navigation(
        self,
        *,
        outcome: str,
        duration_seconds: float,
        distance_meters: float | None = None,
    ) -> None:
        result = safe_label(
            outcome,
            allowed=OUTCOMES,
        )

        self.registry.navigation_sessions_total.labels(
            outcome=result
        ).inc()

        self.registry.navigation_duration_seconds.labels(
            outcome=result
        ).observe(
            self._nonnegative(duration_seconds)
        )

        if distance_meters is not None:
            self.registry.navigation_distance_meters.observe(
                self._nonnegative(distance_meters)
            )

    def record_manipulation(
        self,
        *,
        task_type: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        task = safe_label(
            task_type,
            allowed={
                "pick",
                "place",
                "pick_and_place",
                "grasp",
                "release",
                "push",
                "pull",
                "lift",
                "sort",
                "assembly",
                "inspection",
                "tool_use",
                "other",
            },
            default="other",
        )

        result = safe_label(
            outcome,
            allowed=OUTCOMES,
        )

        self.registry.manipulation_tasks_total.labels(
            task_type=task,
            outcome=result,
        ).inc()

        self.registry.manipulation_duration_seconds.labels(
            task_type=task
        ).observe(
            self._nonnegative(duration_seconds)
        )

    def record_inference(
        self,
        *,
        model_type: str,
        outcome: str,
        duration_seconds: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        model = safe_label(
            model_type,
            allowed=AI_MODEL_TYPES,
            default="other",
        )

        result = safe_label(
            outcome,
            allowed=OUTCOMES,
        )

        self.registry.ai_inference_requests_total.labels(
            model_type=model,
            outcome=result,
        ).inc()

        self.registry.ai_inference_duration_seconds.labels(
            model_type=model
        ).observe(
            self._nonnegative(duration_seconds)
        )

        if input_tokens:
            self.registry.ai_inference_tokens_total.labels(
                model_type=model,
                token_type="input",
            ).inc(
                self._nonnegative(input_tokens)
            )

        if output_tokens:
            self.registry.ai_inference_tokens_total.labels(
                model_type=model,
                token_type="output",
            ).inc(
                self._nonnegative(output_tokens)
            )

    def record_vla_prediction(
        self,
        outcome: str,
    ) -> None:
        self.registry.vla_predictions_total.labels(
            outcome=safe_label(
                outcome,
                allowed=OUTCOMES,
            )
        ).inc()

    def record_world_model_prediction(
        self,
        *,
        outcome: str,
        prediction_error: float | None = None,
    ) -> None:
        self.registry.world_model_predictions_total.labels(
            outcome=safe_label(
                outcome,
                allowed=OUTCOMES,
            )
        ).inc()

        if prediction_error is not None:
            self.registry.world_model_prediction_error.observe(
                self._nonnegative(prediction_error)
            )

    def record_rag_query(
        self,
        *,
        outcome: str,
        retrieval_seconds: float,
        groundedness: float | None = None,
    ) -> None:
        self.registry.rag_queries_total.labels(
            outcome=safe_label(
                outcome,
                allowed=OUTCOMES,
            )
        ).inc()

        self.registry.rag_retrieval_duration_seconds.observe(
            self._nonnegative(retrieval_seconds)
        )

        if groundedness is not None:
            self.registry.rag_groundedness_score.observe(
                self._score(groundedness)
            )

    def record_evaluation(
        self,
        *,
        evaluation_type: str,
        outcome: str,
        score: float | None = None,
    ) -> None:
        category = safe_label(
            evaluation_type,
            allowed=EVALUATION_TYPES,
            default="other",
        )

        self.registry.model_evaluations_total.labels(
            evaluation_type=category,
            outcome=safe_label(
                outcome,
                allowed=OUTCOMES,
            ),
        ).inc()

        if score is not None:
            self.registry.evaluation_score.labels(
                evaluation_type=category
            ).observe(
                self._score(score)
            )

    def record_maintenance_prediction(
        self,
        *,
        risk_level: str,
        remaining_useful_life_hours: float | None = None,
    ) -> None:
        risk = safe_label(
            risk_level,
            allowed={
                "low",
                "moderate",
                "high",
                "critical",
                "unknown",
            },
        )

        self.registry.maintenance_predictions_total.labels(
            risk_level=risk
        ).inc()

        if remaining_useful_life_hours is not None:
            self.registry.rul_prediction_hours.observe(
                self._nonnegative(
                    remaining_useful_life_hours
                )
            )

    def record_repair_estimate(
        self,
        *,
        decision: str,
        estimated_cost: float,
        currency: str = "INR",
    ) -> None:
        decision_label = safe_label(
            decision,
            allowed=REPAIR_DECISIONS,
            default="undetermined",
        )

        currency_label = safe_label(
            currency,
            allowed={
                "inr",
                "usd",
                "eur",
                "gbp",
                "aed",
                "cad",
                "aud",
                "sgd",
                "jpy",
            },
        )

        self.registry.repair_estimates_total.labels(
            decision=decision_label
        ).inc()

        self.registry.repair_estimate_amount.labels(
            currency=currency_label
        ).observe(
            self._nonnegative(estimated_cost)
        )

    def record_safety_alert(
        self,
        severity: str,
    ) -> None:
        self.registry.safety_alerts_total.labels(
            severity=safe_label(
                severity,
                allowed=SEVERITY_LEVELS,
            )
        ).inc()

    def record_safety_violation(
        self,
        severity: str,
    ) -> None:
        self.registry.safety_violations_total.labels(
            severity=safe_label(
                severity,
                allowed=SEVERITY_LEVELS,
            )
        ).inc()

    def record_emergency_stop(self) -> None:
        self.registry.emergency_stop_reports_total.inc()

    def record_safety_check(
        self,
        outcome: str,
    ) -> None:
        self.registry.safety_checks_total.labels(
            outcome=safe_label(
                outcome,
                allowed=OUTCOMES,
            )
        ).inc()

    def record_incident(
        self,
        severity: str,
    ) -> None:
        self.registry.incidents_total.labels(
            severity=safe_label(
                severity,
                allowed=SEVERITY_LEVELS,
            )
        ).inc()

    def set_active_incidents(
        self,
        *,
        severity: str,
        count: int,
    ) -> None:
        self.registry.active_safety_incidents.labels(
            severity=safe_label(
                severity,
                allowed=SEVERITY_LEVELS,
            )
        ).set(
            self._nonnegative(count)
        )

    def record_cache_operation(
        self,
        *,
        backend: str,
        operation: str,
        outcome: str,
        duration_seconds: float | None = None,
    ) -> None:
        backend_label = safe_label(
            backend,
            allowed=CACHE_BACKENDS,
            default="other",
        )

        operation_label = safe_label(
            operation,
            allowed=CACHE_OPERATIONS,
            default="other",
        )

        outcome_label = safe_label(
            outcome,
            allowed={
                "hit",
                "miss",
                "success",
                "failure",
                "unknown",
            },
        )

        self.registry.cache_operations_total.labels(
            backend=backend_label,
            operation=operation_label,
            outcome=outcome_label,
        ).inc()

        if duration_seconds is not None:
            self.registry.cache_operation_duration_seconds.labels(
                backend=backend_label,
                operation=operation_label,
            ).observe(
                self._nonnegative(duration_seconds)
            )

    def record_event(
        self,
        *,
        category: str,
        outcome: str,
        duration_seconds: float | None = None,
    ) -> None:
        category_label = safe_label(
            category,
            allowed=EVENT_CATEGORIES,
            default="other",
        )

        self.registry.event_bus_events_total.labels(
            category=category_label,
            outcome=safe_label(
                outcome,
                allowed=OUTCOMES,
            ),
        ).inc()

        if duration_seconds is not None:
            self.registry.event_bus_handler_duration_seconds.labels(
                category=category_label
            ).observe(
                self._nonnegative(duration_seconds)
            )

    def set_event_queue_size(self, size: int) -> None:
        self.registry.event_bus_queue_size.set(
            self._nonnegative(size)
        )

    def set_background_task_count(self, count: int) -> None:
        self.registry.background_tasks_active.set(
            self._nonnegative(count)
        )

    def update_uptime(self) -> None:
        self.registry.application_uptime_seconds.set(
            time.monotonic() - self.started_at
        )

    def update_resource_usage(
        self,
        *,
        cpu_percent: float | None = None,
        memory_bytes: float | None = None,
        gpu_percent: float | None = None,
        gpu_memory_bytes: float | None = None,
    ) -> None:
        if cpu_percent is not None:
            self.registry.resource_cpu_utilization.set(
                self._score(cpu_percent, maximum=100)
            )

        if memory_bytes is not None:
            self.registry.resource_memory_bytes.set(
                self._nonnegative(memory_bytes)
            )

        if gpu_percent is not None:
            self.registry.resource_gpu_utilization.set(
                self._score(gpu_percent, maximum=100)
            )

        if gpu_memory_bytes is not None:
            self.registry.resource_gpu_memory_bytes.set(
                self._nonnegative(gpu_memory_bytes)
            )


# ============================================================
# Timing Utilities
# ============================================================


@contextmanager
def measure_duration(
    histogram: Histogram,
) -> Iterator[None]:
    """
    Record execution duration in seconds.

    Example:
        with measure_duration(
            registry.telemetry_ingestion_duration_seconds
        ):
            process_telemetry()
    """

    start = time.perf_counter()

    try:
        yield
    finally:
        histogram.observe(
            time.perf_counter() - start
        )


def track_execution(
    histogram: Histogram,
) -> Callable[[F], F]:
    """
    Decorator for synchronous and asynchronous functions.
    """

    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                started = time.perf_counter()

                try:
                    return await func(*args, **kwargs)
                finally:
                    histogram.observe(
                        time.perf_counter() - started
                    )

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            started = time.perf_counter()

            try:
                return func(*args, **kwargs)
            finally:
                histogram.observe(
                    time.perf_counter() - started
                )

        return sync_wrapper  # type: ignore[return-value]

    return decorator


# ============================================================
# FastAPI Metrics Middleware
# ============================================================


class MetricsMiddleware:
    """
    Pure ASGI HTTP metrics middleware.

    Uses registered route templates rather than raw paths
    to prevent unbounded label cardinality.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        recorder: MetricsRecorder,
        exclude_paths: set[str] | None = None,
    ) -> None:
        self.app = app
        self.recorder = recorder
        self.exclude_paths = exclude_paths or {
            "/metrics",
            "/health",
            "/ready",
            "/live",
        }

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")

        if path in self.exclude_paths:
            await self.app(scope, receive, send)
            return

        method = safe_label(
            scope.get("method", "GET"),
            allowed={
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "head",
                "options",
            },
        )

        started = time.perf_counter()
        status_code = 500

        self.recorder.registry.http_requests_in_progress.labels(
            method=method
        ).inc()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = int(message["status"])

            await send(message)

        try:
            await self.app(
                scope,
                receive,
                send_wrapper,
            )

        except Exception:
            status_code = 500

            self.recorder.registry.http_exceptions_total.labels(
                exception_type="unhandled"
            ).inc()

            raise

        finally:
            duration = time.perf_counter() - started

            route = scope.get("route")
            route_path = getattr(
                route,
                "path",
                "unmatched",
            )

            try:
                self.recorder.record_http_request(
                    method=method,
                    route=route_path,
                    status_code=status_code,
                    duration_seconds=duration,
                )
            finally:
                self.recorder.registry.http_requests_in_progress.labels(
                    method=method
                ).dec()


# ============================================================
# Prometheus Metrics Router
# ============================================================


def create_metrics_router(
    metrics_registry: MetricsRegistry,
    *,
    endpoint: str = "/metrics",
) -> APIRouter:
    router = APIRouter(
        tags=["Monitoring"],
        include_in_schema=False,
    )

    @router.get(endpoint)
    async def prometheus_metrics() -> Response:
        return Response(
            content=metrics_registry.export(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return router


def setup_metrics(
    app: FastAPI,
    *,
    settings: MetricsSettings | None = None,
    metrics_registry: MetricsRegistry | None = None,
) -> MetricsRecorder:
    """
    Configure Prometheus metrics for a FastAPI application.

    Example:
        app = FastAPI()

        recorder = setup_metrics(
            app,
            settings=MetricsSettings(
                enabled=True,
                endpoint="/metrics",
            ),
        )

    Protect /metrics using network-level access controls
    or authenticated monitoring infrastructure.
    """

    config = settings or MetricsSettings()

    registry = metrics_registry or MetricsRegistry(config)

    recorder = MetricsRecorder(registry)

    app.state.metrics_registry = registry
    app.state.metrics_recorder = recorder

    if not config.enabled:
        return recorder

    app.include_router(
        create_metrics_router(
            registry,
            endpoint=config.endpoint,
        )
    )

    if config.include_http_metrics:
        excluded = {config.endpoint}

        if config.exclude_health_requests:
            excluded.update({
                "/health",
                "/ready",
                "/live",
            })

        app.add_middleware(
            MetricsMiddleware,
            recorder=recorder,
            exclude_paths=excluded,
        )

    logger.info(
        "Prometheus metrics initialized at %s",
        config.endpoint,
    )

    return recorder


# ============================================================
# Global Metrics Instance
# ============================================================


_metrics_registry: MetricsRegistry | None = None
_metrics_recorder: MetricsRecorder | None = None


def get_metrics_registry() -> MetricsRegistry:
    global _metrics_registry

    if _metrics_registry is None:
        _metrics_registry = MetricsRegistry()

    return _metrics_registry


def get_metrics_recorder() -> MetricsRecorder:
    global _metrics_recorder

    if _metrics_recorder is None:
        _metrics_recorder = MetricsRecorder(
            get_metrics_registry()
        )

    return _metrics_recorder


def export_metrics() -> bytes:
    recorder = get_metrics_recorder()
    recorder.update_uptime()

    return get_metrics_registry().export()


# ============================================================
# Exports
# ============================================================


__all__ = [
    "METRICS_NAMESPACE",
    "DEFAULT_LATENCY_BUCKETS",
    "DEFAULT_INFERENCE_BUCKETS",
    "DEFAULT_SCORE_BUCKETS",
    "MetricsSettings",
    "MetricsRegistry",
    "MetricsRecorder",
    "MetricsMiddleware",
    "safe_label",
    "measure_duration",
    "track_execution",
    "create_metrics_router",
    "setup_metrics",
    "get_metrics_registry",
    "get_metrics_recorder",
    "export_metrics",
]
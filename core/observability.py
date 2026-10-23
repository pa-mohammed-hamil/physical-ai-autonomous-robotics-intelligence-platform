"""Observability module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Core Observability

Features:
- OpenTelemetry distributed tracing
- FastAPI request tracing
- Structured trace and span context
- Prometheus metrics integration
- Application health and readiness checks
- Robotics telemetry observability
- AI inference, RAG, VLA and world-model tracing
- Agent execution monitoring
- Navigation and manipulation tracing
- Safety event observability
- Predictive maintenance monitoring
- Database, cache and event-bus instrumentation
- OTLP trace export
- Graceful telemetry shutdown

Install:
    pip install \
        opentelemetry-api \
        opentelemetry-sdk \
        opentelemetry-exporter-otlp-proto-http \
        opentelemetry-instrumentation-fastapi \
        prometheus-client

Security:
- Do not record credentials, prompts, images or raw telemetry.
- Do not place user, tenant or robot IDs in metric labels.
- Avoid sensitive information in trace attributes.
- Observability must not control physical robot actuation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import time

from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Iterator, Mapping

from fastapi import FastAPI

from opentelemetry import baggage, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.trace import (
    Span,
    Status,
    StatusCode,
    Tracer,
)

from app.core.logging import (
    get_logger,
    logging_context,
    redact_sensitive,
)
from app.core.metrics import (
    MetricsRecorder,
    MetricsRegistry,
    MetricsSettings,
    get_metrics_recorder,
    setup_metrics,
)


logger = get_logger(__name__)


# ============================================================
# Constants
# ============================================================

SERVICE_NAME = "physical-ai-autonomous-robotics"
SERVICE_VERSION = "1.0.0"

DEFAULT_OTLP_ENDPOINT = (
    "http://localhost:4318/v1/traces"
)

DEFAULT_TRACE_SAMPLE_RATE = 0.1

MAX_ATTRIBUTE_LENGTH = 256
MAX_EVENT_ATTRIBUTES = 16

SENSITIVE_ATTRIBUTE_TERMS = frozenset({
    "password",
    "secret",
    "token",
    "credential",
    "authorization",
    "cookie",
    "api_key",
    "private_key",
    "prompt",
    "completion",
    "image",
    "audio",
    "video",
    "raw_data",
    "payload",
    "document_content",
    "personal_data",
})


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def monotonic_time() -> float:
    return time.perf_counter()


def validate_duration(value: float) -> float:
    result = float(value)

    if not math.isfinite(result) or result < 0:
        raise ValueError(
            "Duration must be finite and nonnegative"
        )

    return result


def safe_attribute_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, str):
        return value[:MAX_ATTRIBUTE_LENGTH]

    if isinstance(value, (tuple, list)):
        if not value:
            return []

        if all(isinstance(item, str) for item in value):
            return [
                item[:MAX_ATTRIBUTE_LENGTH]
                for item in value[:MAX_EVENT_ATTRIBUTES]
            ]

        return None

    return None


def sanitize_attributes(
    attributes: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not attributes:
        return {}

    result: dict[str, Any] = {}

    for key, value in attributes.items():
        normalized_key = str(key).lower()

        if any(
            term in normalized_key
            for term in SENSITIVE_ATTRIBUTE_TERMS
        ):
            continue

        safe_value = safe_attribute_value(value)

        if safe_value is not None:
            result[str(key)[:128]] = safe_value

    return result


def current_trace_id() -> str | None:
    span = trace.get_current_span()
    context = span.get_span_context()

    if not context.is_valid:
        return None

    return format(context.trace_id, "032x")


def current_span_id() -> str | None:
    span = trace.get_current_span()
    context = span.get_span_context()

    if not context.is_valid:
        return None

    return format(context.span_id, "016x")


# ============================================================
# Enums
# ============================================================


class ObservabilityComponent(str, Enum):
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    EVENT_BUS = "event_bus"
    ROBOT = "robot"
    FLEET = "fleet"
    SENSOR = "sensor"
    TELEMETRY = "telemetry"
    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    VLA = "vla"
    WORLD_MODEL = "world_model"
    AGENT = "agent"
    RAG = "rag"
    SIMULATION = "simulation"
    EVALUATION = "evaluation"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    SAFETY = "safety"
    INFRASTRUCTURE = "infrastructure"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class OperationOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# ============================================================
# Settings
# ============================================================


@dataclass(frozen=True)
class ObservabilitySettings:
    enabled: bool = True

    service_name: str = SERVICE_NAME
    service_version: str = SERVICE_VERSION
    environment: str = "development"

    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_fastapi_instrumentation: bool = True

    enable_otlp_export: bool = False
    enable_console_trace_export: bool = False

    otlp_endpoint: str = DEFAULT_OTLP_ENDPOINT

    trace_sample_rate: float = DEFAULT_TRACE_SAMPLE_RATE

    metrics_endpoint: str = "/metrics"

    enable_health_checks: bool = True

    health_check_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not 0 <= self.trace_sample_rate <= 1:
            raise ValueError(
                "trace_sample_rate must be between 0 and 1"
            )

        if self.health_check_timeout_seconds <= 0:
            raise ValueError(
                "health_check_timeout_seconds must be positive"
            )

        if not self.metrics_endpoint.startswith("/"):
            raise ValueError(
                "metrics_endpoint must start with '/'"
            )


# ============================================================
# Health Check Models
# ============================================================


@dataclass(frozen=True)
class HealthCheckResult:
    component: str
    status: HealthStatus
    latency_ms: float
    checked_at: datetime = field(default_factory=utc_now)
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 3),
            "checked_at": self.checked_at.isoformat(),
            "details": sanitize_attributes(self.details),
        }


HealthCheckCallback = Callable[[], Any]


@dataclass
class HealthCheckDefinition:
    name: str
    callback: HealthCheckCallback
    critical: bool = True
    timeout_seconds: float = 5.0


# ============================================================
# Health Check Manager
# ============================================================


class HealthCheckManager:
    """
    Executes registered health checks.

    Synchronous callbacks are executed in worker threads
    to avoid blocking the event loop.
    """

    def __init__(self) -> None:
        self._checks: dict[
            str, HealthCheckDefinition
        ] = {}

    def register(
        self,
        name: str,
        callback: HealthCheckCallback,
        *,
        critical: bool = True,
        timeout_seconds: float = 5.0,
    ) -> None:
        if name in self._checks:
            raise ValueError(
                f"Health check already registered: {name}"
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be positive"
            )

        self._checks[name] = HealthCheckDefinition(
            name=name,
            callback=callback,
            critical=critical,
            timeout_seconds=timeout_seconds,
        )

    async def _execute(
        self,
        definition: HealthCheckDefinition,
    ) -> HealthCheckResult:
        started = monotonic_time()

        try:
            if inspect.iscoroutinefunction(
                definition.callback
            ):
                operation = definition.callback()
            else:
                operation = asyncio.to_thread(
                    definition.callback
                )

            result = await asyncio.wait_for(
                operation,
                timeout=definition.timeout_seconds,
            )

            if inspect.isawaitable(result):
                result = await asyncio.wait_for(
                    result,
                    timeout=definition.timeout_seconds,
                )

            if isinstance(result, HealthCheckResult):
                return result

            if isinstance(result, Mapping):
                raw_status = result.get(
                    "status",
                    "unknown",
                )

                try:
                    status = HealthStatus(
                        str(raw_status).lower()
                    )
                except ValueError:
                    status = HealthStatus.UNKNOWN

                details = {
                    str(key): value
                    for key, value in result.items()
                    if key != "status"
                }

            elif result is True:
                status = HealthStatus.HEALTHY
                details = {}

            elif result is False:
                status = HealthStatus.UNHEALTHY
                details = {}

            else:
                status = HealthStatus.UNKNOWN
                details = {}

            return HealthCheckResult(
                component=definition.name,
                status=status,
                latency_ms=(
                    monotonic_time() - started
                ) * 1000,
                details=details,
            )

        except asyncio.TimeoutError:
            return HealthCheckResult(
                component=definition.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(
                    monotonic_time() - started
                ) * 1000,
                details={"reason": "timeout"},
            )

        except Exception:
            logger.exception(
                "Health check failed: %s",
                definition.name,
            )

            return HealthCheckResult(
                component=definition.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(
                    monotonic_time() - started
                ) * 1000,
                details={"reason": "check_failed"},
            )

    async def check_all(self) -> dict[str, Any]:
        definitions = list(self._checks.values())

        results = await asyncio.gather(
            *[
                self._execute(definition)
                for definition in definitions
            ]
        )

        result_map = {
            result.component: result
            for result in results
        }

        critical_healthy = all(
            result_map[definition.name].status
            == HealthStatus.HEALTHY
            for definition in definitions
            if definition.critical
        )

        all_healthy = all(
            result.status == HealthStatus.HEALTHY
            for result in results
        )

        if not critical_healthy:
            overall = HealthStatus.UNHEALTHY
        elif not all_healthy:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return {
            "status": overall.value,
            "ready": critical_healthy,
            "checked_at": utc_now().isoformat(),
            "components": {
                name: result.to_dict()
                for name, result in result_map.items()
            },
        }


# ============================================================
# Distributed Tracing
# ============================================================


class TracingManager:
    """
    Manages OpenTelemetry tracer initialization and shutdown.
    """

    def __init__(
        self,
        settings: ObservabilitySettings,
    ) -> None:
        self.settings = settings

        self.provider: TracerProvider | None = None
        self.tracer: Tracer = trace.get_tracer(
            settings.service_name
        )

        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        if (
            not self.settings.enabled
            or not self.settings.enable_tracing
        ):
            self._initialized = True
            return

        from opentelemetry.sdk.trace.sampling import (
            ParentBased,
            TraceIdRatioBased,
        )

        resource = Resource.create({
            "service.name": self.settings.service_name,
            "service.version": self.settings.service_version,
            "deployment.environment.name": (
                self.settings.environment
            ),
        })

        provider = TracerProvider(
            resource=resource,
            sampler=ParentBased(
                TraceIdRatioBased(
                    self.settings.trace_sample_rate
                )
            ),
        )

        if self.settings.enable_otlp_export:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(
                endpoint=self.settings.otlp_endpoint,
            )

            provider.add_span_processor(
                BatchSpanProcessor(exporter)
            )

        if self.settings.enable_console_trace_export:
            provider.add_span_processor(
                BatchSpanProcessor(
                    ConsoleSpanExporter()
                )
            )

        trace.set_tracer_provider(provider)

        self.provider = provider
        self.tracer = trace.get_tracer(
            self.settings.service_name,
            self.settings.service_version,
        )

        self._initialized = True

        logger.info(
            "Distributed tracing initialized",
            extra={
                "tracing_enabled": True,
                "otlp_enabled": (
                    self.settings.enable_otlp_export
                ),
            },
        )

    @contextmanager
    def span(
        self,
        name: str,
        *,
        component: ObservabilityComponent,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[Span]:
        safe_attributes = sanitize_attributes(
            attributes
        )

        safe_attributes["app.component"] = (
            component.value
        )

        with self.tracer.start_as_current_span(
            name,
            attributes=safe_attributes,
        ) as span:
            trace_id = current_trace_id()

            with logging_context(
                trace_id=trace_id
            ):
                try:
                    yield span

                    span.set_status(
                        Status(StatusCode.OK)
                    )

                except Exception as exc:
                    span.record_exception(
                        exc,
                        attributes={
                            "exception.escaped": True,
                        },
                    )

                    span.set_status(
                        Status(
                            StatusCode.ERROR,
                            "Operation failed",
                        )
                    )

                    raise

    def shutdown(self) -> None:
        if self.provider is not None:
            self.provider.shutdown()

            logger.info(
                "Distributed tracing shut down"
            )


# ============================================================
# Domain Observability Service
# ============================================================


class RoboticsObservability:
    """
    Unified observability interface for robotics workflows.
    """

    def __init__(
        self,
        *,
        tracing: TracingManager,
        metrics: MetricsRecorder,
    ) -> None:
        self.tracing = tracing
        self.metrics = metrics

    @contextmanager
    def observe_operation(
        self,
        operation: str,
        *,
        component: ObservabilityComponent,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[Span]:
        started = monotonic_time()

        with self.tracing.span(
            operation,
            component=component,
            attributes=attributes,
        ) as span:
            try:
                yield span

            finally:
                duration = monotonic_time() - started

                span.set_attribute(
                    "operation.duration_ms",
                    duration * 1000,
                )

    def record_robot_health(
        self,
        *,
        score: float,
        health_level: str,
    ) -> None:
        self.metrics.record_robot_health(
            score=score,
            health_level=health_level,
        )

    def record_sensor_health(
        self,
        *,
        sensor_type: str,
        score: float,
    ) -> None:
        self.metrics.record_sensor_health(
            sensor_type=sensor_type,
            score=score,
        )

    def record_navigation(
        self,
        *,
        outcome: str,
        duration_seconds: float,
        distance_meters: float | None = None,
    ) -> None:
        self.metrics.record_navigation(
            outcome=outcome,
            duration_seconds=duration_seconds,
            distance_meters=distance_meters,
        )

    def record_manipulation(
        self,
        *,
        task_type: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        self.metrics.record_manipulation(
            task_type=task_type,
            outcome=outcome,
            duration_seconds=duration_seconds,
        )

    def record_ai_inference(
        self,
        *,
        model_type: str,
        outcome: str,
        duration_seconds: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.metrics.record_inference(
            model_type=model_type,
            outcome=outcome,
            duration_seconds=duration_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def record_world_model(
        self,
        *,
        outcome: str,
        prediction_error: float | None = None,
    ) -> None:
        self.metrics.record_world_model_prediction(
            outcome=outcome,
            prediction_error=prediction_error,
        )

    def record_vla(
        self,
        *,
        outcome: str,
    ) -> None:
        self.metrics.record_vla_prediction(
            outcome=outcome
        )

    def record_rag(
        self,
        *,
        outcome: str,
        retrieval_seconds: float,
        groundedness: float | None = None,
    ) -> None:
        self.metrics.record_rag_query(
            outcome=outcome,
            retrieval_seconds=retrieval_seconds,
            groundedness=groundedness,
        )

    def record_maintenance(
        self,
        *,
        risk_level: str,
        remaining_useful_life_hours: float | None = None,
    ) -> None:
        self.metrics.record_maintenance_prediction(
            risk_level=risk_level,
            remaining_useful_life_hours=(
                remaining_useful_life_hours
            ),
        )

    def record_repair_estimate(
        self,
        *,
        decision: str,
        estimated_cost: float,
        currency: str = "INR",
    ) -> None:
        self.metrics.record_repair_estimate(
            decision=decision,
            estimated_cost=estimated_cost,
            currency=currency,
        )

    def record_safety_alert(
        self,
        *,
        severity: str,
        alert_code: str,
    ) -> None:
        self.metrics.record_safety_alert(
            severity=severity
        )

        logger.critical(
            "Safety alert observed",
            extra={
                "event_type": "safety.alert",
                "severity": severity,
                "alert_code": alert_code,
            },
        )

    def record_emergency_stop(self) -> None:
        self.metrics.record_emergency_stop()

        logger.critical(
            "Emergency stop event observed"
        )


# ============================================================
# Observability Manager
# ============================================================


class ObservabilityManager:
    """
    Main observability lifecycle manager.
    """

    def __init__(
        self,
        settings: ObservabilitySettings | None = None,
    ) -> None:
        self.settings = (
            settings or ObservabilitySettings()
        )

        self.tracing = TracingManager(
            self.settings
        )

        self.metrics_registry = MetricsRegistry(
            MetricsSettings(
                enabled=self.settings.enable_metrics,
                endpoint=self.settings.metrics_endpoint,
            )
        )

        self.metrics = MetricsRecorder(
            self.metrics_registry
        )

        self.robotics = RoboticsObservability(
            tracing=self.tracing,
            metrics=self.metrics,
        )

        self.health = HealthCheckManager()

        self.started_at: datetime | None = None
        self._initialized = False

    def initialize(
        self,
        app: FastAPI | None = None,
    ) -> None:
        if self._initialized:
            return

        if self.settings.enabled:
            self.tracing.initialize()

            if app is not None:
                if self.settings.enable_metrics:
                    setup_metrics(
                        app,
                        settings=MetricsSettings(
                            enabled=True,
                            endpoint=(
                                self.settings.metrics_endpoint
                            ),
                        ),
                        metrics_registry=(
                            self.metrics_registry
                        ),
                    )

                    app.state.metrics_recorder = (
                        self.metrics
                    )

                if (
                    self.settings.enable_tracing
                    and self.settings.enable_fastapi_instrumentation
                ):
                    try:
                        from opentelemetry.instrumentation.fastapi import (
                            FastAPIInstrumentor,
                        )

                        FastAPIInstrumentor.instrument_app(
                            app,
                            tracer_provider=(
                                self.tracing.provider
                            ),
                        )

                    except ImportError:
                        logger.warning(
                            "FastAPI tracing instrumentation "
                            "is not installed"
                        )

        self.started_at = utc_now()
        self._initialized = True

        logger.info(
            "Application observability initialized"
        )

    async def readiness(self) -> dict[str, Any]:
        if not self._initialized:
            return {
                "status": "unhealthy",
                "ready": False,
                "reason": "not_initialized",
            }

        if not self.settings.enable_health_checks:
            return {
                "status": "healthy",
                "ready": True,
            }

        return await self.health.check_all()

    def liveness(self) -> dict[str, Any]:
        return {
            "status": (
                "healthy"
                if self._initialized
                else "unhealthy"
            ),
            "alive": self._initialized,
            "service": self.settings.service_name,
            "timestamp": utc_now().isoformat(),
        }

    def get_status(self) -> dict[str, Any]:
        self.metrics.update_uptime()

        return {
            "service": self.settings.service_name,
            "version": self.settings.service_version,
            "environment": self.settings.environment,
            "initialized": self._initialized,
            "tracing_enabled": (
                self.settings.enabled
                and self.settings.enable_tracing
            ),
            "metrics_enabled": (
                self.settings.enabled
                and self.settings.enable_metrics
            ),
            "started_at": (
                self.started_at.isoformat()
                if self.started_at
                else None
            ),
            "timestamp": utc_now().isoformat(),
        }

    async def shutdown(self) -> None:
        if not self._initialized:
            return

        self.tracing.shutdown()
        self._initialized = False

        logger.info(
            "Application observability stopped"
        )


# ============================================================
# Decorators
# ============================================================


def traced_operation(
    name: str,
    *,
    component: ObservabilityComponent,
    attributes: Mapping[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Trace synchronous or asynchronous operations.

    Example:
        @traced_operation(
            "robot.health.predict",
            component=ObservabilityComponent.ROBOT,
        )
        async def predict_health():
            ...
    """

    def decorator(
        func: Callable[..., Any],
    ) -> Callable[..., Any]:
        from functools import wraps

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                manager = get_observability_manager()

                with manager.tracing.span(
                    name,
                    component=component,
                    attributes=attributes,
                ):
                    return await func(
                        *args,
                        **kwargs,
                    )

            return async_wrapper

        @wraps(func)
        def sync_wrapper(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            manager = get_observability_manager()

            with manager.tracing.span(
                name,
                component=component,
                attributes=attributes,
            ):
                return func(
                    *args,
                    **kwargs,
                )

        return sync_wrapper

    return decorator


# ============================================================
# Singleton
# ============================================================


_observability_manager: ObservabilityManager | None = None


def configure_observability(
    settings: ObservabilitySettings | None = None,
) -> ObservabilityManager:
    global _observability_manager

    if (
        _observability_manager is not None
        and _observability_manager._initialized
    ):
        raise RuntimeError(
            "Observability is already initialized"
        )

    _observability_manager = ObservabilityManager(
        settings
    )

    return _observability_manager


def get_observability_manager() -> ObservabilityManager:
    global _observability_manager

    if _observability_manager is None:
        _observability_manager = ObservabilityManager()

    return _observability_manager


def get_robotics_observability() -> RoboticsObservability:
    return get_observability_manager().robotics


def initialize_observability(
    app: FastAPI | None = None,
) -> ObservabilityManager:
    manager = get_observability_manager()
    manager.initialize(app)
    return manager


async def shutdown_observability() -> None:
    manager = get_observability_manager()
    await manager.shutdown()


# ============================================================
# FastAPI Integration
# ============================================================


def register_observability_endpoints(
    app: FastAPI,
) -> None:
    """
    Register health endpoints.

    Use during application setup, before startup.
    """

    @app.get(
        "/health/live",
        tags=["Health"],
    )
    async def liveness_endpoint() -> dict[str, Any]:
        return get_observability_manager().liveness()

    @app.get(
        "/health/ready",
        tags=["Health"],
    )
    async def readiness_endpoint() -> dict[str, Any]:
        return await get_observability_manager().readiness()

    @app.get(
        "/health/observability",
        tags=["Health"],
    )
    async def observability_status() -> dict[str, Any]:
        return get_observability_manager().get_status()


# ============================================================
# Exports
# ============================================================


__all__ = [
    "SERVICE_NAME",
    "SERVICE_VERSION",
    "DEFAULT_OTLP_ENDPOINT",
    "ObservabilityComponent",
    "HealthStatus",
    "OperationOutcome",
    "ObservabilitySettings",
    "HealthCheckResult",
    "HealthCheckDefinition",
    "HealthCheckManager",
    "TracingManager",
    "RoboticsObservability",
    "ObservabilityManager",
    "traced_operation",
    "sanitize_attributes",
    "safe_attribute_value",
    "current_trace_id",
    "current_span_id",
    "configure_observability",
    "get_observability_manager",
    "get_robotics_observability",
    "initialize_observability",
    "shutdown_observability",
    "register_observability_endpoints",
]
"""Tracing module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Core Distributed Tracing

Features:
- OpenTelemetry distributed tracing
- OTLP HTTP trace export
- Parent-based trace sampling
- Synchronous and asynchronous tracing decorators
- FastAPI request instrumentation
- Trace and span context propagation
- Robotics operation tracing
- AI inference and agent execution tracing
- RAG, VLA and world-model tracing
- Navigation and manipulation tracing
- Sensor and telemetry tracing
- Simulation and predictive maintenance tracing
- Safety event tracing
- Structured trace events
- Sensitive attribute filtering
- Graceful shutdown

Dependencies:
    pip install \
        opentelemetry-api \
        opentelemetry-sdk \
        opentelemetry-exporter-otlp-proto-http \
        opentelemetry-instrumentation-fastapi

Security:
- Never record passwords, API keys or access tokens.
- Never record raw prompts, images, audio or video.
- Avoid high-cardinality and sensitive trace attributes.
- Tracing must not authorize or execute physical actions.
"""

from __future__ import annotations

import functools
import inspect
import logging
import math
import threading
import time

from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Mapping,
    ParamSpec,
    TypeVar,
)

from opentelemetry import context, propagate, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.trace.sampling import (
    ParentBased,
    TraceIdRatioBased,
)
from opentelemetry.trace import (
    Span,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


# ============================================================
# Constants
# ============================================================

SERVICE_NAME = "physical-ai-autonomous-robotics"
SERVICE_VERSION = "1.0.0"

DEFAULT_OTLP_ENDPOINT = (
    "http://localhost:4318/v1/traces"
)

MAX_ATTRIBUTE_LENGTH = 256
MAX_ATTRIBUTE_COUNT = 32
MAX_EVENT_NAME_LENGTH = 128

SENSITIVE_TERMS = frozenset({
    "password",
    "secret",
    "token",
    "authorization",
    "cookie",
    "credential",
    "api_key",
    "private_key",
    "prompt",
    "completion",
    "message_content",
    "document_content",
    "image",
    "video",
    "audio",
    "raw_payload",
    "raw_telemetry",
    "personal_data",
    "email",
    "phone",
    "address",
    "user_id",
    "tenant_id",
    "organization_id",
    "robot_id",
    "fleet_id",
})


# ============================================================
# Enums
# ============================================================


class TraceComponent(str, Enum):
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
    LLM = "llm"
    SIMULATION = "simulation"
    EVALUATION = "evaluation"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    SAFETY = "safety"
    INFRASTRUCTURE = "infrastructure"


class TraceOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class TracingSettings:
    enabled: bool = True

    service_name: str = SERVICE_NAME
    service_version: str = SERVICE_VERSION
    environment: str = "development"

    sample_rate: float = 0.1

    enable_otlp_export: bool = False
    enable_console_export: bool = False

    otlp_endpoint: str = DEFAULT_OTLP_ENDPOINT

    export_timeout_seconds: float = 5.0

    max_attribute_length: int = MAX_ATTRIBUTE_LENGTH

    record_exceptions: bool = True

    enable_fastapi_instrumentation: bool = True

    def __post_init__(self) -> None:
        if not self.service_name.strip():
            raise ValueError(
                "service_name cannot be empty"
            )

        if not 0 <= self.sample_rate <= 1:
            raise ValueError(
                "sample_rate must be between 0 and 1"
            )

        if self.export_timeout_seconds <= 0:
            raise ValueError(
                "export_timeout_seconds must be positive"
            )

        if self.max_attribute_length <= 0:
            raise ValueError(
                "max_attribute_length must be positive"
            )

        if self.enable_otlp_export:
            if not self.otlp_endpoint.startswith(
                ("http://", "https://")
            ):
                raise ValueError(
                    "OTLP endpoint must be an HTTP(S) URL"
                )


# ============================================================
# Attribute Sanitization
# ============================================================


def is_sensitive_attribute(key: str) -> bool:
    normalized = key.strip().lower()

    return any(
        term in normalized
        for term in SENSITIVE_TERMS
    )


def sanitize_trace_value(
    value: Any,
    *,
    max_length: int = MAX_ATTRIBUTE_LENGTH,
) -> Any | None:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, str):
        return value[:max_length]

    if isinstance(value, (list, tuple)):
        if not value:
            return []

        if all(
            isinstance(item, str)
            for item in value
        ):
            return [
                item[:max_length]
                for item in value[:16]
            ]

        if all(
            isinstance(item, bool)
            for item in value
        ):
            return list(value[:16])

        if all(
            isinstance(item, int)
            and not isinstance(item, bool)
            for item in value
        ):
            return list(value[:16])

        if all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        ):
            return [
                float(item)
                for item in value[:16]
            ]

    return None


def sanitize_trace_attributes(
    attributes: Mapping[str, Any] | None,
    *,
    max_length: int = MAX_ATTRIBUTE_LENGTH,
) -> dict[str, Any]:
    if not attributes:
        return {}

    result: dict[str, Any] = {}

    for key, value in attributes.items():
        if len(result) >= MAX_ATTRIBUTE_COUNT:
            break

        normalized_key = str(key)

        if is_sensitive_attribute(normalized_key):
            continue

        safe_value = sanitize_trace_value(
            value,
            max_length=max_length,
        )

        if safe_value is not None:
            result[
                normalized_key[:128]
            ] = safe_value

    return result


# ============================================================
# Trace Context
# ============================================================


@dataclass(frozen=True)
class TraceContextInfo:
    trace_id: str | None
    span_id: str | None
    sampled: bool
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "sampled": self.sampled,
            "valid": self.valid,
        }


def get_trace_context() -> TraceContextInfo:
    span = trace.get_current_span()
    span_context = span.get_span_context()

    if not span_context.is_valid:
        return TraceContextInfo(
            trace_id=None,
            span_id=None,
            sampled=False,
            valid=False,
        )

    return TraceContextInfo(
        trace_id=format(
            span_context.trace_id,
            "032x",
        ),
        span_id=format(
            span_context.span_id,
            "016x",
        ),
        sampled=bool(
            span_context.trace_flags.sampled
        ),
        valid=True,
    )


def get_trace_id() -> str | None:
    return get_trace_context().trace_id


def get_span_id() -> str | None:
    return get_trace_context().span_id


def inject_trace_headers(
    headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Inject W3C trace context into outbound headers.

    Use only for trusted outbound destinations.
    """

    carrier = dict(headers or {})

    propagate.inject(carrier)

    return carrier


@contextmanager
def extracted_trace_context(
    headers: Mapping[str, str],
) -> Iterator[None]:
    """
    Extract upstream trace context.

    Only use with trusted transport boundaries and
    approved propagators.
    """

    extracted = propagate.extract(dict(headers))

    token = context.attach(extracted)

    try:
        yield
    finally:
        context.detach(token)


# ============================================================
# Tracing Manager
# ============================================================


class TracingManager:
    """
    Manages OpenTelemetry tracing lifecycle.

    Configure once during application startup.

    OpenTelemetry's global tracer provider is process-wide.
    Applications should avoid creating competing providers.
    """

    def __init__(
        self,
        settings: TracingSettings | None = None,
    ) -> None:
        self.settings = settings or TracingSettings()

        self.provider: TracerProvider | None = None

        self.tracer: Tracer = trace.get_tracer(
            self.settings.service_name
        )

        self._initialized = False
        self._owns_provider = False
        self._lock = threading.RLock()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return

            if not self.settings.enabled:
                self._initialized = True
                return

            resource = Resource.create({
                "service.name": self.settings.service_name,
                "service.version": (
                    self.settings.service_version
                ),
                "deployment.environment.name": (
                    self.settings.environment
                ),
            })

            sampler = ParentBased(
                TraceIdRatioBased(
                    self.settings.sample_rate
                )
            )

            provider = TracerProvider(
                resource=resource,
                sampler=sampler,
            )

            if self.settings.enable_otlp_export:
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )
                except ImportError as exc:
                    raise RuntimeError(
                        "Install "
                        "opentelemetry-exporter-otlp-proto-http"
                    ) from exc

                exporter = OTLPSpanExporter(
                    endpoint=self.settings.otlp_endpoint,
                    timeout=(
                        self.settings.export_timeout_seconds
                    ),
                )

                provider.add_span_processor(
                    BatchSpanProcessor(exporter)
                )

            if self.settings.enable_console_export:
                provider.add_span_processor(
                    BatchSpanProcessor(
                        ConsoleSpanExporter()
                    )
                )

            current_provider = (
                trace.get_tracer_provider()
            )

            # The global provider can be configured only
            # once. If another component owns it, reuse
            # that provider instead of overriding it.
            if isinstance(
                current_provider,
                TracerProvider,
            ):
                provider.shutdown()

                self.provider = current_provider
                self._owns_provider = False

                logger.info(
                    "Using existing OpenTelemetry "
                    "tracer provider"
                )
            else:
                trace.set_tracer_provider(provider)

                if trace.get_tracer_provider() is provider:
                    self.provider = provider
                    self._owns_provider = True
                else:
                    provider.shutdown()
                    self.provider = None
                    self._owns_provider = False

            self.tracer = trace.get_tracer(
                self.settings.service_name,
                self.settings.service_version,
            )

            self._initialized = True

            logger.info(
                "Tracing initialized",
                extra={
                    "service": self.settings.service_name,
                    "sample_rate": self.settings.sample_rate,
                    "otlp_export": (
                        self.settings.enable_otlp_export
                    ),
                },
            )

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        component: TraceComponent | str,
        attributes: Mapping[str, Any] | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        record_exception: bool | None = None,
    ) -> Iterator[Span]:
        """
        Start an operation span.

        Span status is automatically set to ERROR when
        an exception escapes the context.
        """

        component_name = (
            component.value
            if isinstance(component, TraceComponent)
            else str(component)
        )

        safe_attributes = sanitize_trace_attributes(
            attributes,
            max_length=(
                self.settings.max_attribute_length
            ),
        )

        safe_attributes["app.component"] = (
            component_name[:128]
        )

        should_record_exception = (
            self.settings.record_exceptions
            if record_exception is None
            else record_exception
        )

        with self.tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=safe_attributes,
            record_exception=should_record_exception,
            set_status_on_exception=True,
        ) as span:
            started = time.perf_counter()

            try:
                yield span
            except BaseException:
                raise
            else:
                if span.is_recording():
                    span.set_status(
                        Status(StatusCode.OK)
                    )
            finally:
                if span.is_recording():
                    span.set_attribute(
                        "operation.duration_ms",
                        (
                            time.perf_counter() - started
                        ) * 1000,
                    )

    @asynccontextmanager
    async def start_async_span(
        self,
        name: str,
        *,
        component: TraceComponent | str,
        attributes: Mapping[str, Any] | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> AsyncIterator[Span]:
        with self.start_span(
            name,
            component=component,
            attributes=attributes,
            kind=kind,
        ) as span:
            yield span

    def add_event(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        span = trace.get_current_span()

        if not span.is_recording():
            return

        span.add_event(
            name[:MAX_EVENT_NAME_LENGTH],
            attributes=sanitize_trace_attributes(
                attributes,
                max_length=(
                    self.settings.max_attribute_length
                ),
            ),
        )

    def set_attributes(
        self,
        attributes: Mapping[str, Any],
    ) -> None:
        span = trace.get_current_span()

        if not span.is_recording():
            return

        safe_attributes = sanitize_trace_attributes(
            attributes,
            max_length=(
                self.settings.max_attribute_length
            ),
        )

        for key, value in safe_attributes.items():
            span.set_attribute(key, value)

    def record_error(
        self,
        *,
        error_type: str,
        message: str = "Operation failed",
    ) -> None:
        span = trace.get_current_span()

        if not span.is_recording():
            return

        span.add_event(
            "operation.error",
            attributes={
                "error.type": error_type[:128],
            },
        )

        span.set_status(
            Status(
                StatusCode.ERROR,
                message[:MAX_ATTRIBUTE_LENGTH],
            )
        )

    def force_flush(
        self,
        timeout_millis: int = 5000,
    ) -> bool:
        if self.provider is None:
            return True

        return bool(
            self.provider.force_flush(
                timeout_millis=timeout_millis
            )
        )

    def shutdown(self) -> None:
        with self._lock:
            if not self._initialized:
                return

            if (
                self.provider is not None
                and self._owns_provider
            ):
                self.provider.shutdown()

            self.provider = None
            self._owns_provider = False
            self._initialized = False

            logger.info("Tracing shutdown completed")


# ============================================================
# Robotics Tracing
# ============================================================


class RoboticsTracer:
    """
    Domain-specific tracing interface.

    These methods record observability metadata only.
    They never execute robot commands.
    """

    def __init__(
        self,
        manager: TracingManager,
    ) -> None:
        self.manager = manager

    @contextmanager
    def robot_operation(
        self,
        operation: str,
        *,
        robot_type: str = "unknown",
        mode: str = "simulation",
    ) -> Iterator[Span]:
        with self.manager.start_span(
            f"robot.{operation}",
            component=TraceComponent.ROBOT,
            attributes={
                "robot.type": robot_type,
                "robot.mode": mode,
            },
        ) as span:
            yield span

    @contextmanager
    def sensor_processing(
        self,
        *,
        sensor_type: str,
        operation: str = "process",
    ) -> Iterator[Span]:
        with self.manager.start_span(
            f"sensor.{operation}",
            component=TraceComponent.SENSOR,
            attributes={
                "sensor.type": sensor_type,
            },
        ) as span:
            yield span

    @contextmanager
    def perception(
        self,
        *,
        task_type: str,
        model_family: str = "unknown",
    ) -> Iterator[Span]:
        with self.manager.start_span(
            "perception.inference",
            component=TraceComponent.PERCEPTION,
            attributes={
                "perception.task_type": task_type,
                "model.family": model_family,
            },
        ) as span:
            yield span

    @contextmanager
    def navigation(
        self,
        *,
        planner: str,
        execution_mode: str = "simulation",
    ) -> Iterator[Span]:
        with self.manager.start_span(
            "navigation.operation",
            component=TraceComponent.NAVIGATION,
            attributes={
                "navigation.planner": planner,
                "navigation.mode": execution_mode,
            },
        ) as span:
            yield span

    @contextmanager
    def manipulation(
        self,
        *,
        task_type: str,
        execution_mode: str = "simulation",
    ) -> Iterator[Span]:
        with self.manager.start_span(
            "manipulation.operation",
            component=TraceComponent.MANIPULATION,
            attributes={
                "manipulation.task_type": task_type,
                "manipulation.mode": execution_mode,
            },
        ) as span:
            yield span

    @contextmanager
    def vla_inference(
        self,
        *,
        model_family: str,
        action_execution: bool = False,
    ) -> Iterator[Span]:
        with self.manager.start_span(
            "vla.inference",
            component=TraceComponent.VLA,
            attributes={
                "vla.model_family": model_family,
                "vla.action_execution_requested": (
                    action_execution
                ),
            },
        ) as span:
            yield span

    @contextmanager
    def world_model_prediction(
        self,
        *,
        model_family: str,
        horizon_steps: int,
    ) -> Iterator[Span]:
        with self.manager.start_span(
            "world_model.predict",
            component=TraceComponent.WORLD_MODEL,
            attributes={
                "world_model.family": model_family,
                "world_model.horizon_steps": (
                    horizon_steps
                ),
            },
        ) as span:
            yield span

    @contextmanager
    def agent_execution(
        self,
        *,
        agent_type: str,
        tool_name: str | None = None,
    ) -> Iterator[Span]:
        attributes: dict[str, Any] = {
            "agent.type": agent_type,
        }

        if tool_name:
            attributes["agent.tool_name"] = tool_name

        with self.manager.start_span(
            "agent.execute",
            component=TraceComponent.AGENT,
            attributes=attributes,
        ) as span:
            yield span

    @contextmanager
    def llm_inference(
        self,
        *,
        provider: str,
        model_family: str,
        operation: str = "generate",
    ) -> Iterator[Span]:
        with self.manager.start_span(
            f"llm.{operation}",
            component=TraceComponent.LLM,
            attributes={
                "llm.provider": provider,
                "llm.model_family": model_family,
            },
        ) as span:
            yield span

    @contextmanager
    def rag_query(
        self,
        *,
        retrieval_strategy: str,
        top_k: int,
    ) -> Iterator[Span]:
        with self.manager.start_span(
            "rag.query",
            component=TraceComponent.RAG,
            attributes={
                "rag.retrieval_strategy": (
                    retrieval_strategy
                ),
                "rag.top_k": top_k,
            },
        ) as span:
            yield span

    @contextmanager
    def simulation(
        self,
        *,
        engine: str,
        scenario_type: str,
    ) -> Iterator[Span]:
        with self.manager.start_span(
            "simulation.run",
            component=TraceComponent.SIMULATION,
            attributes={
                "simulation.engine": engine,
                "simulation.scenario_type": (
                    scenario_type
                ),
            },
        ) as span:
            yield span

    @contextmanager
    def maintenance_prediction(
        self,
        *,
        prediction_type: str,
        asset_type: str,
    ) -> Iterator[Span]:
        with self.manager.start_span(
            "maintenance.predict",
            component=TraceComponent.MAINTENANCE,
            attributes={
                "maintenance.prediction_type": (
                    prediction_type
                ),
                "maintenance.asset_type": asset_type,
            },
        ) as span:
            yield span

    @contextmanager
    def repair_estimation(
        self,
        *,
        component_type: str,
        estimate_type: str = "repair_or_replace",
    ) -> Iterator[Span]:
        with self.manager.start_span(
            "repair.estimate",
            component=TraceComponent.REPAIR,
            attributes={
                "repair.component_type": (
                    component_type
                ),
                "repair.estimate_type": (
                    estimate_type
                ),
            },
        ) as span:
            yield span

    def safety_event(
        self,
        *,
        event_type: str,
        severity: str,
        outcome: TraceOutcome = TraceOutcome.REJECTED,
    ) -> None:
        self.manager.add_event(
            "robotics.safety_event",
            attributes={
                "safety.event_type": event_type,
                "safety.severity": severity,
                "safety.outcome": outcome.value,
            },
        )

        logger.warning(
            "Robotics safety event observed",
            extra={
                "event_type": event_type,
                "severity": severity,
                "outcome": outcome.value,
            },
        )


# ============================================================
# Tracing Decorator
# ============================================================


def traced(
    operation: str,
    *,
    component: TraceComponent | str,
    attributes: Mapping[str, Any] | None = None,
) -> Callable[
    [Callable[P, R]],
    Callable[P, R],
]:
    """
    Trace synchronous and asynchronous functions.

    Example:

        @traced(
            "robot.health.predict",
            component=TraceComponent.MAINTENANCE,
        )
        async def predict_health():
            ...
    """

    def decorator(
        func: Callable[P, R],
    ) -> Callable[P, R]:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(
                *args: P.args,
                **kwargs: P.kwargs,
            ) -> Any:
                manager = get_tracing_manager()

                with manager.start_span(
                    operation,
                    component=component,
                    attributes=attributes,
                ):
                    return await func(
                        *args,
                        **kwargs,
                    )

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            manager = get_tracing_manager()

            with manager.start_span(
                operation,
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
# FastAPI Instrumentation
# ============================================================


def instrument_fastapi(
    app: Any,
    *,
    manager: TracingManager | None = None,
    excluded_urls: str = (
        "/health/live,/health/ready,/metrics"
    ),
) -> None:
    """
    Instrument FastAPI HTTP requests.

    Call once during application setup, before startup
    and before the ASGI middleware stack is finalized.
    """

    tracing_manager = (
        manager or get_tracing_manager()
    )

    if not tracing_manager.settings.enabled:
        return

    if not (
        tracing_manager.settings
        .enable_fastapi_instrumentation
    ):
        return

    try:
        from opentelemetry.instrumentation.fastapi import (
            FastAPIInstrumentor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install opentelemetry-instrumentation-fastapi"
        ) from exc

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=(
            tracing_manager.provider
            or trace.get_tracer_provider()
        ),
        excluded_urls=excluded_urls,
    )

    logger.info(
        "FastAPI tracing instrumentation configured"
    )


# ============================================================
# Singleton
# ============================================================


_tracing_manager: TracingManager | None = None
_tracing_lock = threading.RLock()


def configure_tracing(
    settings: TracingSettings | None = None,
) -> TracingManager:
    global _tracing_manager

    with _tracing_lock:
        if (
            _tracing_manager is not None
            and _tracing_manager.initialized
        ):
            raise RuntimeError(
                "Tracing is already initialized"
            )

        _tracing_manager = TracingManager(
            settings
        )

        return _tracing_manager


def get_tracing_manager() -> TracingManager:
    global _tracing_manager

    with _tracing_lock:
        if _tracing_manager is None:
            _tracing_manager = TracingManager()

        return _tracing_manager


def get_robotics_tracer() -> RoboticsTracer:
    return RoboticsTracer(
        get_tracing_manager()
    )


def initialize_tracing(
    settings: TracingSettings | None = None,
) -> TracingManager:
    manager = (
        configure_tracing(settings)
        if settings is not None
        else get_tracing_manager()
    )

    manager.initialize()

    return manager


def shutdown_tracing() -> None:
    global _tracing_manager

    with _tracing_lock:
        if _tracing_manager is not None:
            _tracing_manager.shutdown()
            _tracing_manager = None


# ============================================================
# Exports
# ============================================================


__all__ = [
    "SERVICE_NAME",
    "SERVICE_VERSION",
    "DEFAULT_OTLP_ENDPOINT",
    "TraceComponent",
    "TraceOutcome",
    "TracingSettings",
    "TraceContextInfo",
    "TracingManager",
    "RoboticsTracer",
    "is_sensitive_attribute",
    "sanitize_trace_value",
    "sanitize_trace_attributes",
    "get_trace_context",
    "get_trace_id",
    "get_span_id",
    "inject_trace_headers",
    "extracted_trace_context",
    "traced",
    "instrument_fastapi",
    "configure_tracing",
    "get_tracing_manager",
    "get_robotics_tracer",
    "initialize_tracing",
    "shutdown_tracing",
]
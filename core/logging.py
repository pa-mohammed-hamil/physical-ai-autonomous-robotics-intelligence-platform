"""Logging module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Core Logging Configuration

Features:
- Structured JSON logging
- Human-readable console logging
- Request and correlation IDs
- Tenant, robot, fleet, and task context
- ContextVar-based async context isolation
- Sensitive-data redaction
- Exception logging
- Configurable log levels
- Rotating file handlers
- UTC timestamps
- FastAPI request logging middleware
- Robotics event and performance logging
- Audit logging integration

Security:
- Never log credentials, tokens, or secrets.
- Avoid logging raw sensor images, audio, or PII.
- Tenant context must come from trusted authentication.
- Audit records should be stored in a durable audit system.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.config
import os
import re
import sys
import time
import uuid

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator, Mapping

from starlette.types import ASGIApp, Message, Receive, Scope, Send


# ============================================================
# Constants
# ============================================================

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "json"
DEFAULT_LOG_DIRECTORY = "logs"
DEFAULT_LOG_FILE = "physical_ai.log"

DEFAULT_MAX_LOG_BYTES = 20 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 5

SERVICE_NAME = "physical-ai-autonomous-robotics"

SENSITIVE_FIELD_NAMES = frozenset({
    "password",
    "password_hash",
    "secret",
    "client_secret",
    "api_key",
    "access_token",
    "refresh_token",
    "id_token",
    "authorization",
    "cookie",
    "set-cookie",
    "private_key",
    "secret_key",
    "session_token",
    "session_id",
    "mfa_secret",
    "otp",
    "pin",
    "credential",
    "credentials",
    "database_url",
    "redis_url",
    "connection_string",
})

REDACTED = "[REDACTED]"


# ============================================================
# Context Variables
# ============================================================

_request_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("request_id", default=None)
)

_correlation_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("correlation_id", default=None)
)

_trace_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("trace_id", default=None)
)

_tenant_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("tenant_id", default=None)
)

_organization_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("organization_id", default=None)
)

_user_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("user_id", default=None)
)

_robot_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("robot_id", default=None)
)

_fleet_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("fleet_id", default=None)
)

_task_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("task_id", default=None)
)

_model_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("model_id", default=None)
)

_simulation_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("simulation_id", default=None)
)

_context_variables: dict[
    str, contextvars.ContextVar[str | None]
] = {
    "request_id": _request_id,
    "correlation_id": _correlation_id,
    "trace_id": _trace_id,
    "tenant_id": _tenant_id,
    "organization_id": _organization_id,
    "user_id": _user_id,
    "robot_id": _robot_id,
    "fleet_id": _fleet_id,
    "task_id": _task_id,
    "model_id": _model_id,
    "simulation_id": _simulation_id,
}


# ============================================================
# Logging Settings
# ============================================================


@dataclass(frozen=True)
class LoggingSettings:
    level: str = DEFAULT_LOG_LEVEL
    format: str = DEFAULT_LOG_FORMAT

    service_name: str = SERVICE_NAME
    environment: str = "development"

    enable_console: bool = True
    enable_file: bool = False

    log_directory: str = DEFAULT_LOG_DIRECTORY
    log_filename: str = DEFAULT_LOG_FILE

    max_file_bytes: int = DEFAULT_MAX_LOG_BYTES
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT

    include_context: bool = True
    redact_sensitive_data: bool = True

    log_http_requests: bool = True
    log_health_requests: bool = False

    def __post_init__(self) -> None:
        if self.level.upper() not in logging._nameToLevel:
            raise ValueError(
                f"Invalid logging level: {self.level}"
            )

        if self.format not in {"json", "console"}:
            raise ValueError(
                "Logging format must be json or console"
            )

        if self.max_file_bytes <= 0:
            raise ValueError(
                "max_file_bytes must be positive"
            )

        if self.backup_count < 0:
            raise ValueError(
                "backup_count cannot be negative"
            )


# ============================================================
# Context Management
# ============================================================


def generate_request_id() -> str:
    return str(uuid.uuid4())


def set_log_context(**values: str | None) -> None:
    """
    Set context for the current asynchronous execution.

    Trusted application services should set tenant_id,
    user_id, and organization_id after authentication.
    """

    for key, value in values.items():
        variable = _context_variables.get(key)

        if variable is None:
            raise ValueError(
                f"Unsupported logging context field: {key}"
            )

        variable.set(
            str(value) if value is not None else None
        )


def get_log_context() -> dict[str, str]:
    return {
        name: value
        for name, variable in _context_variables.items()
        if (value := variable.get()) is not None
    }


def clear_log_context() -> None:
    for variable in _context_variables.values():
        variable.set(None)


@contextmanager
def logging_context(
    **values: str | None,
) -> Iterator[None]:
    """
    Temporarily bind structured logging context.

    Example:
        with logging_context(
            tenant_id="tenant-1",
            robot_id="robot-12",
        ):
            logger.info("Robot health updated")
    """

    tokens: list[
        tuple[
            contextvars.ContextVar[str | None],
            contextvars.Token[str | None],
        ]
    ] = []

    try:
        for key, value in values.items():
            variable = _context_variables.get(key)

            if variable is None:
                raise ValueError(
                    f"Unsupported context field: {key}"
                )

            token = variable.set(
                str(value) if value is not None else None
            )

            tokens.append((variable, token))

        yield

    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


# ============================================================
# Sensitive Data Redaction
# ============================================================


_SECRET_PATTERN = re.compile(
    r"(?i)\b("
    r"password|passwd|api[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|"
    r"client[_-]?secret|secret[_-]?key|"
    r"authorization"
    r")\s*[:=]\s*"
    r"([\"']?)([^\s,;\"']+)\2"
)

_BEARER_PATTERN = re.compile(
    r"(?i)\bBearer\s+[A-Za-z0-9._~+/\-=]+"
)

_URL_CREDENTIAL_PATTERN = re.compile(
    r"(?i)([a-z][a-z0-9+.-]*://)"
    r"([^:/@\s]+):([^@\s]+)@"
)


def redact_string(value: str) -> str:
    value = _BEARER_PATTERN.sub(
        f"Bearer {REDACTED}",
        value,
    )

    value = _SECRET_PATTERN.sub(
        lambda match: (
            f"{match.group(1)}={REDACTED}"
        ),
        value,
    )

    value = _URL_CREDENTIAL_PATTERN.sub(
        lambda match: (
            f"{match.group(1)}{REDACTED}@"
        ),
        value,
    )

    return value


def redact_sensitive(
    value: Any,
    *,
    max_depth: int = 8,
    _depth: int = 0,
) -> Any:
    """
    Best-effort redaction for structured log values.

    This is defense in depth, not a guarantee that arbitrary
    secrets embedded in free-form text can be detected.
    """

    if _depth >= max_depth:
        return "[MAX_DEPTH]"

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}

        for key, item in value.items():
            name = str(key)
            normalized = name.lower().replace("-", "_")

            if (
                normalized in SENSITIVE_FIELD_NAMES
                or normalized.endswith("_token")
                or normalized.endswith("_secret")
                or normalized.endswith("_password")
                or normalized.endswith("_key")
            ):
                result[name] = REDACTED
            else:
                result[name] = redact_sensitive(
                    item,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )

        return result

    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            redact_sensitive(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in value
        ]

    if isinstance(value, str):
        return redact_string(value)

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    return redact_string(str(value))


# ============================================================
# Context Filter
# ============================================================


class ContextFilter(logging.Filter):
    """
    Add ContextVar values to each log record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        context = get_log_context()

        for key in _context_variables:
            if not hasattr(record, key):
                setattr(
                    record,
                    key,
                    context.get(key),
                )

        return True


# ============================================================
# JSON Formatter
# ============================================================


class JSONFormatter(logging.Formatter):
    """
    Structured JSON formatter with UTC timestamps.
    """

    RESERVED_FIELDS = frozenset(
        vars(
            logging.LogRecord(
                name="",
                level=0,
                pathname="",
                lineno=0,
                msg="",
                args=(),
                exc_info=None,
            )
        )
    )

    def __init__(
        self,
        *,
        service_name: str = SERVICE_NAME,
        environment: str = "development",
        include_context: bool = True,
        redact: bool = True,
    ) -> None:
        super().__init__()

        self.service_name = service_name
        self.environment = environment
        self.include_context = include_context
        self.redact = redact

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(
            record.created,
            tz=timezone.utc,
        ).isoformat()

        message = record.getMessage()

        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": message,
            "service": self.service_name,
            "environment": self.environment,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread,
        }

        if self.include_context:
            for name in _context_variables:
                value = getattr(record, name, None)

                if value is not None:
                    payload[name] = value

        for key, value in record.__dict__.items():
            if (
                key not in self.RESERVED_FIELDS
                and key not in payload
                and not key.startswith("_")
            ):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(
                record.exc_info
            )

        if record.stack_info:
            payload["stack"] = record.stack_info

        if self.redact:
            payload = redact_sensitive(payload)

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        )


# ============================================================
# Console Formatter
# ============================================================


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable formatter for local development.
    """

    def __init__(
        self,
        *,
        include_context: bool = True,
        redact: bool = True,
    ) -> None:
        super().__init__(
            fmt=(
                "%(asctime)s | %(levelname)-8s | "
                "%(name)s | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self.converter = time.gmtime
        self.include_context = include_context
        self.redact = redact

    def format(self, record: logging.LogRecord) -> str:
        # Copy to avoid mutating the record shared with
        # other logging handlers.
        copied = logging.makeLogRecord(
            record.__dict__.copy()
        )

        if self.redact:
            copied.msg = redact_string(
                record.getMessage()
            )
            copied.args = ()

        result = super().format(copied)

        if self.include_context:
            context = {
                key: getattr(record, key, None)
                for key in _context_variables
                if getattr(record, key, None) is not None
            }

            if context:
                if self.redact:
                    context = redact_sensitive(context)

                result += (
                    " | context="
                    + json.dumps(
                        context,
                        default=str,
                        ensure_ascii=False,
                    )
                )

        return result


# ============================================================
# Logging Setup
# ============================================================


def setup_logging(
    settings: LoggingSettings | None = None,
) -> None:
    """
    Configure application logging.

    Call once during application startup.

    Example:
        setup_logging(
            LoggingSettings(
                level="INFO",
                format="json",
                enable_file=True,
            )
        )
    """

    config = settings or LoggingSettings()

    root_logger = logging.getLogger()

    level = logging.getLevelName(
        config.level.upper()
    )

    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    context_filter = ContextFilter()

    if config.format == "json":
        formatter: logging.Formatter = JSONFormatter(
            service_name=config.service_name,
            environment=config.environment,
            include_context=config.include_context,
            redact=config.redact_sensitive_data,
        )
    else:
        formatter = ConsoleFormatter(
            include_context=config.include_context,
            redact=config.redact_sensitive_data,
        )

    if config.enable_console:
        console_handler = logging.StreamHandler(
            sys.stdout
        )

        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(context_filter)

        root_logger.addHandler(console_handler)

    if config.enable_file:
        log_directory = Path(
            config.log_directory
        )

        log_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_handler = RotatingFileHandler(
            filename=(
                log_directory / config.log_filename
            ),
            maxBytes=config.max_file_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )

        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)

        root_logger.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(
        logging.WARNING
    )

    logging.getLogger("httpx").setLevel(
        logging.WARNING
    )

    logging.getLogger("urllib3").setLevel(
        logging.WARNING
    )

    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.WARNING
    )

    logging.getLogger(__name__).info(
        "Application logging initialized",
        extra={
            "log_format": config.format,
            "log_level": config.level,
            "environment_name": config.environment,
        },
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_log_level(
    level: str,
    *,
    logger_name: str | None = None,
) -> None:
    normalized = level.upper()

    if normalized not in logging._nameToLevel:
        raise ValueError(
            f"Invalid logging level: {level}"
        )

    target = logging.getLogger(logger_name)

    target.setLevel(normalized)


# ============================================================
# Structured Robotics Logger
# ============================================================


class RoboticsLogger:
    """
    Convenience logger for robotics domain events.

    Avoid including raw camera frames, sensor payloads,
    credentials, or sensitive user information.
    """

    def __init__(
        self,
        name: str = "physical_ai.robotics",
    ) -> None:
        self.logger = get_logger(name)

    def robot_state_changed(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        previous_state: str,
        current_state: str,
    ) -> None:
        self.logger.info(
            "Robot state changed",
            extra={
                "event_type": "robot.state.changed",
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "previous_state": previous_state,
                "current_state": current_state,
            },
        )

    def robot_health_updated(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        health_score: float,
        health_status: str,
    ) -> None:
        self.logger.info(
            "Robot health updated",
            extra={
                "event_type": "robot.health.updated",
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "health_score": health_score,
                "health_status": health_status,
            },
        )

    def sensor_anomaly_detected(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
        metric_name: str,
        anomaly_score: float,
    ) -> None:
        self.logger.warning(
            "Sensor anomaly detected",
            extra={
                "event_type": "sensor.anomaly.detected",
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "sensor_id": sensor_id,
                "metric_name": metric_name,
                "anomaly_score": anomaly_score,
            },
        )

    def inference_completed(
        self,
        *,
        tenant_id: str,
        model_id: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        log_method = (
            self.logger.info
            if success
            else self.logger.error
        )

        log_method(
            "AI inference completed",
            extra={
                "event_type": "model.inference.completed",
                "tenant_id": tenant_id,
                "model_id": model_id,
                "latency_ms": latency_ms,
                "success": success,
            },
        )

    def maintenance_prediction(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        component_id: str,
        failure_probability: float,
        remaining_useful_life_hours: float | None,
    ) -> None:
        self.logger.info(
            "Predictive maintenance result",
            extra={
                "event_type": "maintenance.prediction",
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "component_id": component_id,
                "failure_probability": failure_probability,
                "remaining_useful_life_hours": (
                    remaining_useful_life_hours
                ),
            },
        )

    def safety_alert(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        severity: str,
        alert_code: str,
        incident_id: str | None = None,
    ) -> None:
        self.logger.critical(
            "Robot safety alert",
            extra={
                "event_type": "safety.alert",
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "severity": severity,
                "alert_code": alert_code,
                "incident_id": incident_id,
                "requires_human_review": True,
            },
        )

    def repair_estimate_created(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        estimate_id: str,
        estimated_cost: float,
        currency: str,
    ) -> None:
        self.logger.info(
            "Repair estimate created",
            extra={
                "event_type": "repair.estimate.created",
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "estimate_id": estimate_id,
                "estimated_cost": estimated_cost,
                "currency": currency,
            },
        )


# ============================================================
# Performance Logging
# ============================================================


@contextmanager
def log_execution_time(
    operation_name: str,
    *,
    logger_name: str = "physical_ai.performance",
    warning_threshold_ms: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """
    Measure synchronous operation execution time.
    """

    performance_logger = get_logger(logger_name)
    started = time.perf_counter()
    success = False

    try:
        yield
        success = True

    finally:
        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        level = logging.INFO

        if not success:
            level = logging.ERROR

        elif (
            warning_threshold_ms is not None
            and elapsed_ms >= warning_threshold_ms
        ):
            level = logging.WARNING

        performance_logger.log(
            level,
            "Operation execution completed",
            extra={
                "operation_name": operation_name,
                "duration_ms": round(elapsed_ms, 3),
                "success": success,
                "metadata": redact_sensitive(
                    dict(metadata or {})
                ),
            },
        )


# ============================================================
# FastAPI / ASGI Request Logging Middleware
# ============================================================


class RequestLoggingMiddleware:
    """
    Pure ASGI middleware for request logging.

    - Generates a trusted internal request ID.
    - Does not trust client-provided tenant/user identifiers.
    - Does not log request or response bodies.
    - Avoids logging query strings.
    - Preserves async context isolation.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        log_health_requests: bool = False,
    ) -> None:
        self.app = app
        self.logger = get_logger(
            "physical_ai.http"
        )
        self.log_health_requests = log_health_requests

        self.excluded_paths = {
            "/health",
            "/ready",
            "/live",
            "/metrics",
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
        method = scope.get("method", "UNKNOWN")

        should_log = (
            self.log_health_requests
            or path not in self.excluded_paths
        )

        request_id = generate_request_id()

        tokens = [
            (
                _request_id,
                _request_id.set(request_id),
            )
        ]

        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(
            message: Message,
        ) -> None:
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]

                headers = list(
                    message.get("headers", [])
                )

                headers.append(
                    (
                        b"x-request-id",
                        request_id.encode("ascii"),
                    )
                )

                message = {
                    **message,
                    "headers": headers,
                }

            await send(message)

        try:
            await self.app(
                scope,
                receive,
                send_wrapper,
            )

        except Exception:
            if should_log:
                self.logger.exception(
                    "Unhandled HTTP request exception",
                    extra={
                        "http_method": method,
                        "http_path": path,
                        "http_status_code": 500,
                    },
                )

            raise

        finally:
            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000.0

            if should_log:
                level = (
                    logging.ERROR
                    if status_code >= 500
                    else logging.WARNING
                    if status_code >= 400
                    else logging.INFO
                )

                self.logger.log(
                    level,
                    "HTTP request completed",
                    extra={
                        "http_method": method,
                        "http_path": path,
                        "http_status_code": status_code,
                        "duration_ms": round(
                            elapsed_ms, 3
                        ),
                    },
                )

            for variable, token in reversed(tokens):
                variable.reset(token)


# ============================================================
# Audit Logging
# ============================================================


class AuditLogger:
    """
    Structured audit event logger.

    Production audit trails should also be persisted to
    an append-only, access-controlled audit store.
    """

    def __init__(self) -> None:
        self.logger = get_logger(
            "physical_ai.audit"
        )

    def record(
        self,
        *,
        action: str,
        tenant_id: str,
        actor_id: str,
        resource_type: str,
        resource_id: str,
        outcome: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.logger.info(
            "Audit event recorded",
            extra={
                "event_type": "audit.event",
                "audit_action": action,
                "tenant_id": tenant_id,
                "actor_id": actor_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "outcome": outcome,
                "audit_metadata": redact_sensitive(
                    dict(metadata or {})
                ),
                "audit_timestamp": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            },
        )


# ============================================================
# Global Logger Instances
# ============================================================


robotics_logger = RoboticsLogger()
audit_logger = AuditLogger()


# ============================================================
# Exports
# ============================================================


__all__ = [
    "LoggingSettings",
    "ContextFilter",
    "JSONFormatter",
    "ConsoleFormatter",
    "RequestLoggingMiddleware",
    "RoboticsLogger",
    "AuditLogger",
    "setup_logging",
    "get_logger",
    "set_log_level",
    "set_log_context",
    "get_log_context",
    "clear_log_context",
    "logging_context",
    "generate_request_id",
    "redact_string",
    "redact_sensitive",
    "log_execution_time",
    "robotics_logger",
    "audit_logger",
]
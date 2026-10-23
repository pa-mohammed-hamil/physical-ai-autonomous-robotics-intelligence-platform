"""Middleware module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Core Middleware

Features:
- Request and correlation ID management
- Request execution time tracking
- Structured HTTP access logging
- Security response headers
- Request body size limits
- Trusted host validation
- CORS configuration
- Global exception handling
- API version headers
- Health endpoint exclusions
- Optional Prometheus metrics integration
- Safe FastAPI middleware registration

Security:
- Never trust client-provided tenant/user/robot identifiers.
- Authentication and authorization must be enforced separately.
- Physical robot commands require dedicated safety validation.
- Request bodies and credentials are never logged.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import logging_context

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

DEFAULT_MAX_REQUEST_BODY_BYTES = 20 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_API_VERSION = "1.0.0"

REQUEST_ID_HEADER = b"x-request-id"
CORRELATION_ID_HEADER = b"x-correlation-id"
PROCESS_TIME_HEADER = b"x-process-time"

DEFAULT_EXCLUDED_PATHS = frozenset({
    "/health",
    "/health/live",
    "/health/ready",
    "/live",
    "/ready",
    "/metrics",
    "/favicon.ico",
})

SAFE_HTTP_METHODS = frozenset({
    "GET",
    "HEAD",
    "OPTIONS",
})

VALID_HTTP_METHODS = frozenset({
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "HEAD",
    "OPTIONS",
})

_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_request_id() -> str:
    return str(uuid.uuid4())


def sanitize_request_id(value: str | None) -> str:
    if value and _ID_PATTERN.fullmatch(value):
        return value

    return generate_request_id()


def get_header(
    scope: Scope,
    header_name: bytes,
) -> str | None:
    for name, value in scope.get("headers", []):
        if name.lower() == header_name.lower():
            return value.decode("latin-1")

    return None


def set_response_header(
    message: Message,
    name: bytes,
    value: str,
) -> Message:
    headers = [
        (key, item)
        for key, item in message.get("headers", [])
        if key.lower() != name.lower()
    ]

    headers.append((name, value.encode("latin-1")))

    return {
        **message,
        "headers": headers,
    }


def get_route_template(scope: Scope) -> str:
    route = scope.get("route")
    template = getattr(route, "path", None)

    if isinstance(template, str) and template.startswith("/"):
        return template

    return "unmatched"


def error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str | None = None,
    details: Any = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
        },
        "timestamp": utc_now().isoformat(),
    }

    if request_id is not None:
        payload["request_id"] = request_id

    if details is not None:
        payload["error"]["details"] = details

    return JSONResponse(
        status_code=status_code,
        content=payload,
    )


# ============================================================
# Middleware Configuration
# ============================================================


@dataclass(frozen=True)
class MiddlewareSettings:
    environment: str = "development"

    enable_request_context: bool = True
    enable_access_logging: bool = True
    enable_security_headers: bool = True
    enable_body_size_limit: bool = True
    enable_request_timeout: bool = False
    enable_cors: bool = True
    enable_trusted_hosts: bool = True

    enable_metrics: bool = False

    max_request_body_bytes: int = (
        DEFAULT_MAX_REQUEST_BODY_BYTES
    )

    request_timeout_seconds: float = (
        DEFAULT_REQUEST_TIMEOUT_SECONDS
    )

    api_version: str = DEFAULT_API_VERSION

    allowed_hosts: tuple[str, ...] = (
        "localhost",
        "127.0.0.1",
        "testserver",
    )

    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    )

    cors_methods: tuple[str, ...] = (
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
    )

    cors_headers: tuple[str, ...] = (
        "Authorization",
        "Content-Type",
        "X-Request-ID",
        "X-Correlation-ID",
    )

    cors_allow_credentials: bool = False

    excluded_paths: frozenset[str] = field(
        default_factory=lambda: DEFAULT_EXCLUDED_PATHS
    )

    hsts_max_age_seconds: int = 31_536_000
    enable_hsts: bool = False

    def __post_init__(self) -> None:
        if self.max_request_body_bytes <= 0:
            raise ValueError(
                "max_request_body_bytes must be positive"
            )

        if self.request_timeout_seconds <= 0:
            raise ValueError(
                "request_timeout_seconds must be positive"
            )

        if self.hsts_max_age_seconds < 0:
            raise ValueError(
                "hsts_max_age_seconds cannot be negative"
            )

        if (
            self.cors_allow_credentials
            and "*" in self.cors_origins
        ):
            raise ValueError(
                "Wildcard CORS origins cannot be used "
                "with credentials"
            )

        if (
            self.environment.lower() == "production"
            and self.enable_trusted_hosts
            and "*" in self.allowed_hosts
        ):
            raise ValueError(
                "Wildcard trusted hosts are not allowed "
                "in production"
            )


# ============================================================
# Request Context Middleware
# ============================================================


class RequestContextMiddleware:
    """
    Assigns request and correlation identifiers.

    A server-generated request ID is authoritative.
    A validated incoming correlation ID may be propagated.

    Tenant/user identity must be supplied by trusted
    authentication middleware or dependencies.
    """

    def __init__(
        self,
        app: ASGIApp,
    ) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = generate_request_id()

        incoming_correlation = get_header(
            scope,
            CORRELATION_ID_HEADER,
        )

        correlation_id = sanitize_request_id(
            incoming_correlation
        ) if incoming_correlation else request_id

        scope.setdefault("state", {})

        scope["state"]["request_id"] = request_id
        scope["state"]["correlation_id"] = correlation_id

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = set_response_header(
                    message,
                    REQUEST_ID_HEADER,
                    request_id,
                )

                message = set_response_header(
                    message,
                    CORRELATION_ID_HEADER,
                    correlation_id,
                )

            await send(message)

        with logging_context(
            request_id=request_id,
            correlation_id=correlation_id,
        ):
            await self.app(
                scope,
                receive,
                send_wrapper,
            )


# ============================================================
# Access Logging Middleware
# ============================================================


class AccessLoggingMiddleware:
    """
    Records HTTP request metadata without request bodies,
    query parameters, authorization headers, or cookies.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        excluded_paths: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.excluded_paths = frozenset(excluded_paths)

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

        if path in self.excluded_paths:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "UNKNOWN")

        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = int(message["status"])

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)

        except Exception:
            logger.exception(
                "Unhandled HTTP request failure",
                extra={
                    "http_method": method,
                    "http_route": get_route_template(scope),
                },
            )
            raise

        finally:
            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000.0

            level = (
                logging.ERROR
                if status_code >= 500
                else logging.WARNING
                if status_code >= 400
                else logging.INFO
            )

            logger.log(
                level,
                "HTTP request completed",
                extra={
                    "http_method": method,
                    "http_route": get_route_template(scope),
                    "http_status_code": status_code,
                    "duration_ms": round(elapsed_ms, 3),
                },
            )


# ============================================================
# Process Time Middleware
# ============================================================


class ProcessTimeMiddleware:
    def __init__(
        self,
        app: ASGIApp,
    ) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                elapsed = time.perf_counter() - started

                message = set_response_header(
                    message,
                    PROCESS_TIME_HEADER,
                    f"{elapsed:.6f}",
                )

            await send(message)

        await self.app(
            scope,
            receive,
            send_wrapper,
        )


# ============================================================
# Security Headers Middleware
# ============================================================


class SecurityHeadersMiddleware:
    """
    Adds common HTTP security headers.

    HSTS should only be enabled when HTTPS is enforced
    for the entire intended domain.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        enable_hsts: bool = False,
        hsts_max_age_seconds: int = 31_536_000,
    ) -> None:
        self.app = app
        self.enable_hsts = enable_hsts
        self.hsts_max_age_seconds = hsts_max_age_seconds

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                security_headers = {
                    b"x-content-type-options": "nosniff",
                    b"x-frame-options": "DENY",
                    b"referrer-policy": "strict-origin-when-cross-origin",
                    b"permissions-policy": (
                        "camera=(), microphone=(), "
                        "geolocation=()"
                    ),
                    b"content-security-policy": (
                        "default-src 'none'; "
                        "frame-ancestors 'none'"
                    ),
                    b"cache-control": "no-store",
                }

                if (
                    self.enable_hsts
                    and scope.get("scheme") == "https"
                ):
                    security_headers[
                        b"strict-transport-security"
                    ] = (
                        f"max-age={self.hsts_max_age_seconds}"
                    )

                for name, value in security_headers.items():
                    message = set_response_header(
                        message,
                        name,
                        value,
                    )

            await send(message)

        await self.app(
            scope,
            receive,
            send_wrapper,
        )


# ============================================================
# Request Body Size Middleware
# ============================================================


class RequestBodySizeLimitMiddleware:
    """
    Rejects requests exceeding the configured body limit.

    Checks Content-Length and counts received body bytes.

    Infrastructure proxies should enforce compatible limits.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    ) -> None:
        if max_body_bytes <= 0:
            raise ValueError(
                "max_body_bytes must be positive"
            )

        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = get_header(
            scope,
            b"content-length",
        )

        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = -1

            if declared_size < 0:
                response = error_response(
                    status_code=400,
                    code="INVALID_CONTENT_LENGTH",
                    message="Invalid Content-Length header",
                )
                await response(scope, receive, send)
                return

            if declared_size > self.max_body_bytes:
                response = error_response(
                    status_code=413,
                    code="REQUEST_BODY_TOO_LARGE",
                    message="Request body exceeds size limit",
                )
                await response(scope, receive, send)
                return

        received_bytes = 0
        exceeded = False

        async def limited_receive() -> Message:
            nonlocal received_bytes, exceeded

            message = await receive()

            if message["type"] == "http.request":
                received_bytes += len(
                    message.get("body", b"")
                )

                if received_bytes > self.max_body_bytes:
                    exceeded = True
                    raise RequestBodyTooLargeError(
                        "Request body exceeds size limit"
                    )

            return message

        response_started = False

        async def tracked_send(message: Message) -> None:
            nonlocal response_started

            if message["type"] == "http.response.start":
                response_started = True

            await send(message)

        try:
            await self.app(
                scope,
                limited_receive,
                tracked_send,
            )

        except RequestBodyTooLargeError:
            if response_started:
                raise

            response = error_response(
                status_code=413,
                code="REQUEST_BODY_TOO_LARGE",
                message="Request body exceeds size limit",
            )

            await response(scope, receive, send)


class RequestBodyTooLargeError(Exception):
    pass


# ============================================================
# API Version Middleware
# ============================================================


class APIVersionMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        version: str = DEFAULT_API_VERSION,
    ) -> None:
        self.app = app
        self.version = version

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = set_response_header(
                    message,
                    b"x-api-version",
                    self.version,
                )

            await send(message)

        await self.app(scope, receive, send_wrapper)


# ============================================================
# Request Timeout Middleware
# ============================================================


class RequestTimeoutMiddleware:
    """
    Optional request timeout.

    This is appropriate for bounded API operations, but not
    for long-running streaming or robot control sessions.

    Timeout cancellation is not a hardware stop mechanism.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        timeout_seconds: float = (
            DEFAULT_REQUEST_TIMEOUT_SECONDS
        ),
        excluded_paths: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.timeout_seconds = timeout_seconds
        self.excluded_paths = frozenset(excluded_paths)

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] != "http"
            or scope.get("path") in self.excluded_paths
        ):
            await self.app(scope, receive, send)
            return

        response_started = False

        async def tracked_send(message: Message) -> None:
            nonlocal response_started

            if message["type"] == "http.response.start":
                response_started = True

            await send(message)

        try:
            await asyncio.wait_for(
                self.app(scope, receive, tracked_send),
                timeout=self.timeout_seconds,
            )

        except asyncio.TimeoutError:
            if response_started:
                raise

            logger.warning(
                "HTTP request timed out",
                extra={
                    "http_method": scope.get("method"),
                    "http_route": get_route_template(scope),
                    "timeout_seconds": self.timeout_seconds,
                },
            )

            response = error_response(
                status_code=504,
                code="REQUEST_TIMEOUT",
                message="Request processing timed out",
            )

            await response(scope, receive, send)


# ============================================================
# Global Exception Handlers
# ============================================================


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    message = (
        exc.detail
        if isinstance(exc.detail, str)
        else "HTTP request failed"
    )

    response = error_response(
        status_code=exc.status_code,
        code=f"HTTP_{exc.status_code}",
        message=message,
        request_id=request_id,
    )

    for key, value in (exc.headers or {}).items():
        response.headers[key] = value

    return response


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    errors = []

    for error in exc.errors():
        errors.append({
            "field": ".".join(
                str(part)
                for part in error.get("loc", ())
            ),
            "message": error.get(
                "msg",
                "Invalid input",
            ),
            "type": error.get(
                "type",
                "validation_error",
            ),
        })

    return error_response(
        status_code=422,
        code="VALIDATION_ERROR",
        message="Request validation failed",
        request_id=getattr(
            request.state,
            "request_id",
            None,
        ),
        details=errors,
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception(
        "Unhandled application exception",
        extra={
            "http_method": request.method,
            "http_route": get_route_template(
                request.scope
            ),
        },
    )

    return error_response(
        status_code=500,
        code="INTERNAL_SERVER_ERROR",
        message="An internal server error occurred",
        request_id=getattr(
            request.state,
            "request_id",
            None,
        ),
    )


def register_exception_handlers(
    app: FastAPI,
) -> None:
    app.add_exception_handler(
        HTTPException,
        http_exception_handler,
    )

    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,
    )

    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )


# ============================================================
# Middleware Registration
# ============================================================


def setup_middleware(
    app: FastAPI,
    *,
    settings: MiddlewareSettings | None = None,
) -> None:
    """
    Register application middleware.

    Call before the FastAPI application starts.

    Example:
        app = FastAPI()

        setup_middleware(
            app,
            settings=MiddlewareSettings(
                environment="development",
                enable_cors=True,
            ),
        )

    Starlette executes the last registered middleware first.
    """

    config = settings or MiddlewareSettings()

    register_exception_handlers(app)

    if config.enable_request_timeout:
        app.add_middleware(
            RequestTimeoutMiddleware,
            timeout_seconds=config.request_timeout_seconds,
            excluded_paths=config.excluded_paths,
        )

    if config.enable_body_size_limit:
        app.add_middleware(
            RequestBodySizeLimitMiddleware,
            max_body_bytes=config.max_request_body_bytes,
        )

    app.add_middleware(
        APIVersionMiddleware,
        version=config.api_version,
    )

    if config.enable_security_headers:
        app.add_middleware(
            SecurityHeadersMiddleware,
            enable_hsts=config.enable_hsts,
            hsts_max_age_seconds=config.hsts_max_age_seconds,
        )

    app.add_middleware(ProcessTimeMiddleware)

    if config.enable_access_logging:
        app.add_middleware(
            AccessLoggingMiddleware,
            excluded_paths=config.excluded_paths,
        )

    if config.enable_request_context:
        app.add_middleware(RequestContextMiddleware)

    if config.enable_trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=list(config.allowed_hosts),
        )

    if config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_credentials=config.cors_allow_credentials,
            allow_methods=list(config.cors_methods),
            allow_headers=list(config.cors_headers),
            expose_headers=[
                "X-Request-ID",
                "X-Correlation-ID",
                "X-Process-Time",
                "X-API-Version",
            ],
        )

    if config.enable_metrics:
        from app.core.metrics import (
            MetricsMiddleware,
            get_metrics_recorder,
        )

        app.add_middleware(
            MetricsMiddleware,
            recorder=get_metrics_recorder(),
            exclude_paths=set(config.excluded_paths),
        )

    logger.info(
        "Application middleware configured",
        extra={
            "environment_name": config.environment,
            "cors_enabled": config.enable_cors,
            "metrics_enabled": config.enable_metrics,
            "body_limit_bytes": config.max_request_body_bytes,
        },
    )


# ============================================================
# Request Context Accessors
# ============================================================


def get_request_id(request: Request) -> str | None:
    return getattr(
        request.state,
        "request_id",
        None,
    )


def get_correlation_id(
    request: Request,
) -> str | None:
    return getattr(
        request.state,
        "correlation_id",
        None,
    )


def get_request_metadata(
    request: Request,
) -> dict[str, Any]:
    return {
        "request_id": get_request_id(request),
        "correlation_id": get_correlation_id(request),
        "method": request.method,
        "route": get_route_template(request.scope),
        "timestamp": utc_now().isoformat(),
    }


# ============================================================
# Exports
# ============================================================


__all__ = [
    "MiddlewareSettings",
    "RequestContextMiddleware",
    "AccessLoggingMiddleware",
    "ProcessTimeMiddleware",
    "SecurityHeadersMiddleware",
    "RequestBodySizeLimitMiddleware",
    "RequestBodyTooLargeError",
    "APIVersionMiddleware",
    "RequestTimeoutMiddleware",
    "setup_middleware",
    "register_exception_handlers",
    "http_exception_handler",
    "validation_exception_handler",
    "unhandled_exception_handler",
    "get_request_id",
    "get_correlation_id",
    "get_request_metadata",
    "generate_request_id",
    "sanitize_request_id",
    "get_route_template",
    "error_response",
]
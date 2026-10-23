"""Exception Handlers module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Global FastAPI Exception Handlers

Features:
- Standardized JSON error responses
- Authentication and authorization errors
- JWT, session, refresh token and API key errors
- Tenant isolation and RBAC errors
- Request validation errors
- Rate limiting errors
- Database and service errors
- Safe production error messages
- Request and correlation IDs
- Structured exception logging
- Security-conscious error handling
- No sensitive information disclosure

Python 3.11+
FastAPI / Starlette
"""

from __future__ import annotations

import logging
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

MAX_REQUEST_ID_LENGTH = 128
MAX_ERROR_MESSAGE_LENGTH = 500

SENSITIVE_FIELDS = frozenset({
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "private_key",
    "client_secret",
    "session_token",
    "password_hash",
    "credential_hash",
})

SAFE_HTTP_MESSAGES = {
    400: "Bad request",
    401: "Authentication required",
    403: "Access denied",
    404: "Resource not found",
    405: "Method not allowed",
    408: "Request timeout",
    409: "Resource conflict",
    413: "Request payload too large",
    415: "Unsupported media type",
    422: "Request validation failed",
    429: "Too many requests",
    500: "Internal server error",
    501: "Operation not implemented",
    502: "Upstream service error",
    503: "Service temporarily unavailable",
    504: "Upstream service timeout",
}


# ============================================================
# Enums
# ============================================================


class ErrorCode(str, Enum):
    BAD_REQUEST = "BAD_REQUEST"
    VALIDATION_ERROR = "VALIDATION_ERROR"

    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    TOKEN_REVOKED = "TOKEN_REVOKED"
    TOKEN_REUSE_DETECTED = "TOKEN_REUSE_DETECTED"

    ACCESS_DENIED = "ACCESS_DENIED"
    PERMISSION_DENIED = "PERMISSION_DENIED"

    TENANT_ACCESS_DENIED = "TENANT_ACCESS_DENIED"
    TENANT_NOT_FOUND = "TENANT_NOT_FOUND"

    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"

    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    SESSION_EXPIRED = "SESSION_EXPIRED"
    SESSION_REVOKED = "SESSION_REVOKED"

    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"

    OPERATION_NOT_ALLOWED = "OPERATION_NOT_ALLOWED"
    PHYSICAL_CONTROL_DENIED = "PHYSICAL_CONTROL_DENIED"

    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_request_id(request: Request) -> str:
    """
    Use a server-generated request ID.

    An incoming header is not inherently trusted.
    Middleware may provide a validated request ID
    through request.state.request_id.
    """

    value = getattr(
        request.state,
        "request_id",
        None,
    )

    if (
        isinstance(value, str)
        and 1 <= len(value) <= MAX_REQUEST_ID_LENGTH
        and all(
            character.isalnum()
            or character in "-_.:"
            for character in value
        )
    ):
        return value

    return uuid.uuid4().hex


def sanitize_text(
    value: Any,
    *,
    max_length: int = MAX_ERROR_MESSAGE_LENGTH,
) -> str:
    if not isinstance(value, str):
        return "Request could not be processed"

    return "".join(
        character
        for character in value
        if character.isprintable()
    )[:max_length]


def sanitize_validation_errors(
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Never include submitted input values.

    Validation exception objects may contain
    passwords, tokens, request bodies or other
    sensitive information.
    """

    sanitized: list[dict[str, Any]] = []

    for error in errors[:50]:
        location = error.get("loc", ())
        error_type = error.get(
            "type",
            "validation_error",
        )

        safe_location = []

        for part in location:
            part_text = str(part)

            if part_text.lower() in SENSITIVE_FIELDS:
                safe_location.append("[redacted]")
            else:
                safe_location.append(
                    sanitize_text(
                        part_text,
                        max_length=100,
                    )
                )

        sanitized.append({
            "location": safe_location,
            "type": sanitize_text(
                str(error_type),
                max_length=100,
            ),
            "message": "Invalid field value",
        })

    return sanitized


# ============================================================
# Error Response Model
# ============================================================


@dataclass(frozen=True)
class APIError:
    code: ErrorCode
    message: str
    status_code: int

    request_id: str
    timestamp: datetime = field(
        default_factory=utc_now
    )

    details: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        error = {
            "code": self.code.value,
            "message": self.message,
            "status_code": self.status_code,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
        }

        if self.details is not None:
            error["details"] = dict(self.details)

        return {
            "success": False,
            "error": error,
        }


def error_response(
    request: Request,
    *,
    status_code: int,
    code: ErrorCode,
    message: str,
    details: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    request_id = get_request_id(request)

    error = APIError(
        code=code,
        message=sanitize_text(message),
        status_code=status_code,
        request_id=request_id,
        details=details,
    )

    response_headers = {
        "X-Request-ID": request_id,
        "Cache-Control": "no-store",
    }

    if headers:
        for key, value in headers.items():
            if (
                isinstance(key, str)
                and isinstance(value, str)
                and "\r" not in value
                and "\n" not in value
            ):
                response_headers[key] = value

    return JSONResponse(
        status_code=status_code,
        content=error.to_dict(),
        headers=response_headers,
    )


# ============================================================
# Application Exceptions
# ============================================================


class ApplicationError(Exception):
    status_code = 500
    error_code = ErrorCode.INTERNAL_SERVER_ERROR
    public_message = "Internal server error"

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message or self.public_message
        )


class BadRequestError(ApplicationError):
    status_code = 400
    error_code = ErrorCode.BAD_REQUEST
    public_message = "Invalid request"


class AuthenticationRequiredError(ApplicationError):
    status_code = 401
    error_code = ErrorCode.AUTHENTICATION_REQUIRED
    public_message = "Authentication required"


class AccessDeniedError(ApplicationError):
    status_code = 403
    error_code = ErrorCode.ACCESS_DENIED
    public_message = "Access denied"


class ResourceNotFoundError(ApplicationError):
    status_code = 404
    error_code = ErrorCode.RESOURCE_NOT_FOUND
    public_message = "Resource not found"


class ResourceConflictError(ApplicationError):
    status_code = 409
    error_code = ErrorCode.RESOURCE_CONFLICT
    public_message = "Resource conflict"


class RateLimitExceededError(ApplicationError):
    status_code = 429
    error_code = ErrorCode.RATE_LIMIT_EXCEEDED
    public_message = "Too many requests"


class ServiceUnavailableError(ApplicationError):
    status_code = 503
    error_code = ErrorCode.SERVICE_UNAVAILABLE
    public_message = "Service temporarily unavailable"


class PhysicalControlDeniedError(ApplicationError):
    status_code = 403
    error_code = ErrorCode.PHYSICAL_CONTROL_DENIED
    public_message = "Physical control operation denied"


# ============================================================
# Exception Mapping
# ============================================================


@dataclass(frozen=True)
class ExceptionMapping:
    status_code: int
    error_code: ErrorCode
    message: str


EXCEPTION_MAPPINGS: dict[
    tuple[str, str],
    ExceptionMapping,
] = {}


def register_exception_mapping(
    module_name: str,
    exception_name: str,
    *,
    status_code: int,
    error_code: ErrorCode,
    message: str,
) -> None:
    EXCEPTION_MAPPINGS[
        (module_name, exception_name)
    ] = ExceptionMapping(
        status_code=status_code,
        error_code=error_code,
        message=message,
    )


def _register_default_mappings() -> None:
    mappings = [
        (
            "jwt",
            "JWTAuthenticationError",
            401,
            ErrorCode.INVALID_CREDENTIALS,
            "Authentication failed",
        ),
        (
            "jwt",
            "JWTExpiredError",
            401,
            ErrorCode.TOKEN_EXPIRED,
            "Access token expired",
        ),
        (
            "jwt",
            "JWTRevokedError",
            401,
            ErrorCode.TOKEN_REVOKED,
            "Access token revoked",
        ),
        (
            "jwt",
            "JWTSignatureError",
            401,
            ErrorCode.INVALID_CREDENTIALS,
            "Invalid access token",
        ),
        (
            "jwt",
            "JWTPermissionError",
            403,
            ErrorCode.PERMISSION_DENIED,
            "Permission denied",
        ),
        (
            "jwt",
            "JWTConfigurationError",
            503,
            ErrorCode.CONFIGURATION_ERROR,
            "Authentication service unavailable",
        ),
        (
            "rbac",
            "RBACPermissionDenied",
            403,
            ErrorCode.PERMISSION_DENIED,
            "Permission denied",
        ),
        (
            "rbac",
            "RBACRoleNotFound",
            404,
            ErrorCode.RESOURCE_NOT_FOUND,
            "Role not found",
        ),
        (
            "rbac",
            "RBACConflictError",
            409,
            ErrorCode.RESOURCE_CONFLICT,
            "Role conflict",
        ),
        (
            "rbac",
            "RBACValidationError",
            400,
            ErrorCode.VALIDATION_ERROR,
            "Invalid role or permission",
        ),
        (
            "tenant_isolation",
            "TenantAccessDenied",
            403,
            ErrorCode.TENANT_ACCESS_DENIED,
            "Tenant access denied",
        ),
        (
            "tenant_isolation",
            "TenantNotFoundError",
            404,
            ErrorCode.TENANT_NOT_FOUND,
            "Tenant not found",
        ),
        (
            "tenant_isolation",
            "TenantResourceNotFound",
            404,
            ErrorCode.RESOURCE_NOT_FOUND,
            "Resource not found",
        ),
        (
            "tenant_isolation",
            "TenantValidationError",
            400,
            ErrorCode.VALIDATION_ERROR,
            "Invalid tenant request",
        ),
        (
            "sessions",
            "SessionAuthenticationError",
            401,
            ErrorCode.INVALID_CREDENTIALS,
            "Session authentication failed",
        ),
        (
            "sessions",
            "SessionExpiredError",
            401,
            ErrorCode.SESSION_EXPIRED,
            "Session expired",
        ),
        (
            "sessions",
            "SessionRevokedError",
            401,
            ErrorCode.SESSION_REVOKED,
            "Session revoked",
        ),
        (
            "sessions",
            "SessionNotFoundError",
            404,
            ErrorCode.RESOURCE_NOT_FOUND,
            "Session not found",
        ),
        (
            "sessions",
            "SessionConflictError",
            409,
            ErrorCode.RESOURCE_CONFLICT,
            "Session conflict",
        ),
        (
            "sessions",
            "SessionLimitError",
            409,
            ErrorCode.RESOURCE_CONFLICT,
            "Maximum active sessions reached",
        ),
        (
            "refresh_tokens",
            "RefreshTokenAuthenticationError",
            401,
            ErrorCode.INVALID_CREDENTIALS,
            "Refresh token authentication failed",
        ),
        (
            "refresh_tokens",
            "RefreshTokenExpiredError",
            401,
            ErrorCode.TOKEN_EXPIRED,
            "Refresh token expired",
        ),
        (
            "refresh_tokens",
            "RefreshTokenRevokedError",
            401,
            ErrorCode.TOKEN_REVOKED,
            "Refresh token revoked",
        ),
        (
            "refresh_tokens",
            "RefreshTokenReuseError",
            401,
            ErrorCode.TOKEN_REUSE_DETECTED,
            "Refresh token reuse detected",
        ),
        (
            "refresh_tokens",
            "RefreshTokenConflictError",
            409,
            ErrorCode.RESOURCE_CONFLICT,
            "Refresh token conflict",
        ),
        (
            "service_accounts",
            "ServiceAccountAuthenticationError",
            401,
            ErrorCode.INVALID_CREDENTIALS,
            "Service authentication failed",
        ),
        (
            "service_accounts",
            "ServiceAccountPermissionError",
            403,
            ErrorCode.PERMISSION_DENIED,
            "Service account permission denied",
        ),
        (
            "service_accounts",
            "ServiceAccountNotFoundError",
            404,
            ErrorCode.RESOURCE_NOT_FOUND,
            "Service account not found",
        ),
        (
            "service_accounts",
            "ServiceAccountConflictError",
            409,
            ErrorCode.RESOURCE_CONFLICT,
            "Service account conflict",
        ),
        (
            "service_accounts",
            "ServiceAccountExpiredError",
            401,
            ErrorCode.TOKEN_EXPIRED,
            "Service credential expired",
        ),
        (
            "service_accounts",
            "ServiceAccountRevokedError",
            401,
            ErrorCode.TOKEN_REVOKED,
            "Service credential revoked",
        ),
        (
            "authorization",
            "AuthorizationDeniedError",
            403,
            ErrorCode.ACCESS_DENIED,
            "Authorization denied",
        ),
        (
            "authorization",
            "AuthorizationValidationError",
            400,
            ErrorCode.VALIDATION_ERROR,
            "Invalid authorization request",
        ),
        (
            "passwords",
            "PasswordValidationError",
            400,
            ErrorCode.VALIDATION_ERROR,
            "Password does not meet security requirements",
        ),
        (
            "passwords",
            "PasswordReuseError",
            400,
            ErrorCode.VALIDATION_ERROR,
            "Password cannot be reused",
        ),
        (
            "passwords",
            "PasswordResetExpiredError",
            400,
            ErrorCode.TOKEN_EXPIRED,
            "Password reset token expired",
        ),
        (
            "passwords",
            "PasswordResetInvalidError",
            400,
            ErrorCode.INVALID_CREDENTIALS,
            "Invalid password reset token",
        ),
    ]

    for (
        module_name,
        exception_name,
        status_code,
        error_code,
        message,
    ) in mappings:
        register_exception_mapping(
            module_name,
            exception_name,
            status_code=status_code,
            error_code=error_code,
            message=message,
        )


_register_default_mappings()


def resolve_exception_mapping(
    exc: Exception,
) -> ExceptionMapping | None:
    """
    Resolve by actual exception class and module.

    Class names alone are not trusted because
    unrelated modules can define identical names.
    """

    for cls in type(exc).__mro__:
        module_name = cls.__module__.rsplit(".", 1)[-1]

        mapping = EXCEPTION_MAPPINGS.get(
            (module_name, cls.__name__)
        )

        if mapping is not None:
            return mapping

    return None


# ============================================================
# FastAPI HTTP Exception Handler
# ============================================================


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    status_code = exc.status_code

    if status_code == 401:
        code = ErrorCode.AUTHENTICATION_REQUIRED

    elif status_code == 403:
        code = ErrorCode.ACCESS_DENIED

    elif status_code == 404:
        code = ErrorCode.RESOURCE_NOT_FOUND

    elif status_code == 409:
        code = ErrorCode.RESOURCE_CONFLICT

    elif status_code == 422:
        code = ErrorCode.VALIDATION_ERROR

    elif status_code == 429:
        code = ErrorCode.RATE_LIMIT_EXCEEDED

    elif status_code == 503:
        code = ErrorCode.SERVICE_UNAVAILABLE

    elif 400 <= status_code < 500:
        code = ErrorCode.BAD_REQUEST

    else:
        code = ErrorCode.INTERNAL_SERVER_ERROR

    message = SAFE_HTTP_MESSAGES.get(
        status_code,
        "Request failed",
    )

    # Do not expose arbitrary exception details.
    # Only known-safe HTTP response messages
    # are included in production responses.

    allowed_headers = {}

    for key, value in (exc.headers or {}).items():
        if key.lower() in {
            "www-authenticate",
            "retry-after",
            "allow",
        }:
            allowed_headers[key] = value

    return error_response(
        request,
        status_code=status_code,
        code=code,
        message=message,
        headers=allowed_headers,
    )


# ============================================================
# Request Validation Handler
# ============================================================


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    details = {
        "fields": sanitize_validation_errors(
            exc.errors()
        )
    }

    return error_response(
        request,
        status_code=422,
        code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        details=details,
    )


# ============================================================
# Application Error Handler
# ============================================================


async def application_exception_handler(
    request: Request,
    exc: ApplicationError,
) -> JSONResponse:
    logger.warning(
        "Application error: code=%s request_id=%s",
        exc.error_code.value,
        get_request_id(request),
    )

    return error_response(
        request,
        status_code=exc.status_code,
        code=exc.error_code,
        message=exc.public_message,
    )


# ============================================================
# Global Exception Handler
# ============================================================


async def global_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    mapping = resolve_exception_mapping(exc)

    if mapping is not None:
        logger.warning(
            "Handled service exception: "
            "type=%s status=%s request_id=%s",
            type(exc).__name__,
            mapping.status_code,
            get_request_id(request),
        )

        headers = None

        if mapping.status_code == 401:
            headers = {
                "WWW-Authenticate": "Bearer"
            }

        return error_response(
            request,
            status_code=mapping.status_code,
            code=mapping.error_code,
            message=mapping.message,
            headers=headers,
        )

    logger.error(
        "Unhandled application exception: "
        "request_id=%s path=%s method=%s",
        get_request_id(request),
        request.url.path,
        request.method,
        exc_info=(
            type(exc),
            exc,
            exc.__traceback__,
        ),
    )

    return error_response(
        request,
        status_code=500,
        code=ErrorCode.INTERNAL_SERVER_ERROR,
        message="Internal server error",
    )


# ============================================================
# Exception Registration
# ============================================================


def register_exception_handlers(
    app: FastAPI,
) -> None:
    """
    Register global exception handlers.

    Usage:
        app = FastAPI()
        register_exception_handlers(app)
    """

    app.add_exception_handler(
        StarletteHTTPException,
        http_exception_handler,
    )

    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,
    )

    app.add_exception_handler(
        ApplicationError,
        application_exception_handler,
    )

    app.add_exception_handler(
        Exception,
        global_exception_handler,
    )


# ============================================================
# Exports
# ============================================================


__all__ = [
    "ErrorCode",
    "APIError",
    "ApplicationError",
    "BadRequestError",
    "AuthenticationRequiredError",
    "AccessDeniedError",
    "ResourceNotFoundError",
    "ResourceConflictError",
    "RateLimitExceededError",
    "ServiceUnavailableError",
    "PhysicalControlDeniedError",
    "ExceptionMapping",
    "EXCEPTION_MAPPINGS",
    "utc_now",
    "get_request_id",
    "sanitize_text",
    "sanitize_validation_errors",
    "error_response",
    "register_exception_mapping",
    "resolve_exception_mapping",
    "http_exception_handler",
    "validation_exception_handler",
    "application_exception_handler",
    "global_exception_handler",
    "register_exception_handlers",
]
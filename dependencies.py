"""Dependencies module.

TODO: implement and test; no real robot control is enabled.
"""

"""
FastAPI Dependency Injection
============================

Physical AI & Autonomous Robotics Intelligence Platform

Features:
    - Application configuration dependency
    - Async PostgreSQL database sessions
    - Redis client dependency
    - JWT authentication
    - Current user dependency
    - RBAC authorization
    - Robot access authorization
    - Robot command policy checks
    - Analytics dependency providers
    - AI provider configuration
    - Request ID support

Security:
    - Fail-closed authorization
    - Explicit robot permissions
    - JWT signature verification
    - JWT expiration verification
    - Restricted robot command access
    - No direct agent hardware control

Important:
    Software authorization is not a substitute for
    physical robot safety interlocks.
"""

from __future__ import annotations

import logging
import uuid

from dataclasses import dataclass
from functools import lru_cache
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Callable,
    Literal,
)

import jwt

from fastapi import (
    Depends,
    Header,
    HTTPException,
    Request,
    status,
)
from jwt import InvalidTokenError
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import (
    Settings,
    get_settings,
)


logger = logging.getLogger(__name__)


# ======================================================
# CONFIGURATION
# ======================================================

def get_config() -> Settings:
    """Return validated application settings."""
    return get_settings()


ConfigDep = Annotated[
    Settings,
    Depends(get_config),
]


# ======================================================
# DATABASE
# ======================================================

@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[
    AsyncSession
]:
    config = get_settings()

    engine = create_async_engine(
        config.database_dsn,
        echo=config.DB_ECHO,
        pool_size=config.DB_POOL_SIZE,
        max_overflow=config.DB_MAX_OVERFLOW,
        pool_timeout=config.DB_POOL_TIMEOUT_SECONDS,
        pool_recycle=config.DB_POOL_RECYCLE_SECONDS,
        pool_pre_ping=True,
    )

    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def get_db() -> AsyncIterator[AsyncSession]:
    """
    Yield one database session per request.

    Route/service code must explicitly commit writes.
    Uncommitted changes are rolled back on failure.
    """
    factory = get_session_factory()

    async with factory() as session:
        try:
            yield session

        except Exception:
            await session.rollback()
            raise


DatabaseDep = Annotated[
    AsyncSession,
    Depends(get_db),
]


# ======================================================
# REDIS
# ======================================================

@lru_cache(maxsize=1)
def get_redis_client() -> Redis:
    config = get_settings()

    if not config.REDIS_ENABLED:
        raise RuntimeError(
            "Redis is disabled in configuration"
        )

    return Redis.from_url(
        config.redis_dsn,
        encoding="utf-8",
        decode_responses=True,
        socket_timeout=(
            config.REDIS_SOCKET_TIMEOUT_SECONDS
        ),
    )


async def get_redis() -> Redis:
    """
    Provide a shared Redis client.

    The application lifespan should close this client
    during shutdown when Redis is enabled.
    """
    if not get_settings().REDIS_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis service is disabled",
        )

    return get_redis_client()


RedisDep = Annotated[
    Redis,
    Depends(get_redis),
]


# ======================================================
# AUTHENTICATED USER
# ======================================================

class CurrentUser(BaseModel):
    user_id: str = Field(min_length=1)
    username: str | None = None
    email: str | None = None

    roles: set[str] = Field(
        default_factory=set
    )

    permissions: set[str] = Field(
        default_factory=set
    )

    robot_ids: set[str] = Field(
        default_factory=set
    )

    is_active: bool = True

    is_service_account: bool = False


def extract_bearer_token(
    authorization: str | None,
) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, separator, token = (
        authorization.partition(" ")
    )

    if (
        not separator
        or scheme.lower() != "bearer"
        or not token.strip()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token.strip()


def decode_access_token(
    token: str,
    config: Settings,
) -> dict[str, Any]:
    """
    Decode and validate a signed JWT.

    Required claims:
        sub, exp, iat, token_type

    Production should additionally enforce token
    revocation, issuer/audience and current account
    permissions from a trusted identity store.
    """
    try:
        payload = jwt.decode(
            token,
            config.JWT_SECRET_KEY.get_secret_value(),
            algorithms=[config.JWT_ALGORITHM],
            options={
                "require": [
                    "sub",
                    "exp",
                    "iat",
                    "token_type",
                ],
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
            },
        )

    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if payload.get("token_type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Expected access token",
        )

    if not isinstance(payload.get("sub"), str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid subject claim",
        )

    if not payload["sub"].strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty subject claim",
        )

    return payload


def _string_set(
    payload: dict[str, Any],
    field_name: str,
) -> set[str]:
    value = payload.get(field_name, [])

    if not isinstance(value, list):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid {field_name} claim",
        )

    if not all(
        isinstance(item, str) and item.strip()
        for item in value
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid {field_name} entries",
        )

    return set(value)


async def get_current_user(
    config: ConfigDep,
    authorization: Annotated[
        str | None,
        Header(alias="Authorization"),
    ] = None,
) -> CurrentUser:
    """
    Authenticate a user from a bearer token.

    JWT authorization claims are treated as signed
    snapshots. Production systems should load current
    roles, permissions and robot assignments from
    a trusted database or identity provider.
    """
    token = extract_bearer_token(
        authorization
    )

    payload = decode_access_token(
        token,
        config,
    )

    user = CurrentUser(
        user_id=payload["sub"],
        username=payload.get("username"),
        email=payload.get("email"),
        roles=_string_set(
            payload,
            "roles",
        ),
        permissions=_string_set(
            payload,
            "permissions",
        ),
        robot_ids=_string_set(
            payload,
            "robot_ids",
        ),
        is_active=payload.get(
            "is_active",
            True,
        ),
        is_service_account=payload.get(
            "is_service_account",
            False,
        ),
    )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    return user


CurrentUserDep = Annotated[
    CurrentUser,
    Depends(get_current_user),
]


# ======================================================
# ROLE-BASED AUTHORIZATION
# ======================================================

class Roles:
    ADMIN = "admin"
    ROBOT_OPERATOR = "robot_operator"
    ROBOT_TECHNICIAN = "robot_technician"
    SAFETY_OFFICER = "safety_officer"
    AI_ENGINEER = "ai_engineer"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Permissions:
    ROBOT_READ = "robot:read"
    ROBOT_COMMAND = "robot:command"
    ROBOT_MAINTENANCE = "robot:maintenance"

    TELEMETRY_READ = "telemetry:read"

    SAFETY_READ = "safety:read"
    SAFETY_MANAGE = "safety:manage"

    ANALYTICS_READ = "analytics:read"
    REPORT_EXPORT = "report:export"

    MODEL_READ = "model:read"
    MODEL_MANAGE = "model:manage"

    ADMIN_ACCESS = "admin:access"


def require_roles(
    *required_roles: str,
) -> Callable[..., Any]:
    """
    Require at least one listed role.
    """
    async def dependency(
        user: CurrentUserDep,
    ) -> CurrentUser:
        if not set(required_roles).intersection(
            user.roles
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role privileges",
            )

        return user

    return dependency


def require_permissions(
    *required_permissions: str,
) -> Callable[..., Any]:
    """
    Require every listed permission.

    Roles do not automatically grant permissions here;
    permissions must be assigned by the identity system.
    """
    async def dependency(
        user: CurrentUserDep,
    ) -> CurrentUser:
        missing = set(
            required_permissions
        ) - user.permissions

        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )

        return user

    return dependency


# ======================================================
# ROBOT ACCESS AUTHORIZATION
# ======================================================

def ensure_robot_access(
    user: CurrentUser,
    robot_id: str,
    *,
    permission: str = Permissions.ROBOT_READ,
) -> None:
    """
    Verify robot-specific authorization.

    Robot assignments should be loaded from an
    authoritative database in production.
    """
    if permission not in user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing robot permission",
        )

    if robot_id not in user.robot_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Robot access denied",
        )


async def get_authorized_robot_reader(
    robot_id: str,
    user: CurrentUserDep,
) -> CurrentUser:
    ensure_robot_access(
        user,
        robot_id,
        permission=Permissions.ROBOT_READ,
    )

    return user


RobotReaderDep = Annotated[
    CurrentUser,
    Depends(get_authorized_robot_reader),
]


# ======================================================
# ROBOT COMMAND POLICY
# ======================================================

@dataclass(frozen=True)
class RobotCommandAuthorization:
    user_id: str
    robot_id: str
    permission: str
    operator_approval_required: bool
    audit_required: bool
    expiration_required: bool
    simulation_validation_required: bool


async def get_robot_command_authorization(
    robot_id: str,
    user: CurrentUserDep,
    config: ConfigDep,
) -> RobotCommandAuthorization:
    """
    Perform API-level authorization checks.

    This does not authorize actual hardware actuation.

    A separate command execution service must verify:
        - Current operator approval
        - Robot identity and command target
        - Command freshness and replay protection
        - Robot operational state
        - Hardware safety interlocks
        - Independent safety controller status
        - Command-specific operational constraints
    """
    if not config.ROBOT_COMMANDS_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Robot commands are disabled",
        )

    if Roles.ROBOT_OPERATOR not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Robot operator role required",
        )

    ensure_robot_access(
        user,
        robot_id,
        permission=Permissions.ROBOT_COMMAND,
    )

    return RobotCommandAuthorization(
        user_id=user.user_id,
        robot_id=robot_id,
        permission=Permissions.ROBOT_COMMAND,
        operator_approval_required=(
            config.REQUIRE_ROBOT_OPERATOR_APPROVAL
        ),
        audit_required=(
            config.REQUIRE_COMMAND_AUDIT
        ),
        expiration_required=(
            config.REQUIRE_COMMAND_EXPIRATION
        ),
        simulation_validation_required=(
            config.REQUIRE_SIMULATION_VALIDATION
        ),
    )


RobotCommandDep = Annotated[
    RobotCommandAuthorization,
    Depends(get_robot_command_authorization),
]


# ======================================================
# ANALYTICS DEPENDENCIES
# ======================================================

def get_safety_analytics():
    from app.analytics.safety_analytics import (
        SafetyAnalytics,
    )

    config = get_settings()

    return SafetyAnalytics(
        incident_rate_warning=(
            config.SAFETY_INCIDENT_RATE_WARNING
        ),
        incident_rate_critical=(
            config.SAFETY_INCIDENT_RATE_CRITICAL
        ),
        rate_base_hours=(
            config.SAFETY_INCIDENT_RATE_BASE_HOURS
        ),
    )


def get_task_success_analytics():
    from app.analytics.task_success_analytics import (
        TaskSuccessAnalytics,
    )

    config = get_settings()

    return TaskSuccessAnalytics(
        success_warning_percentage=(
            config.TASK_SUCCESS_WARNING_PERCENTAGE
        ),
        retry_warning_percentage=(
            config.TASK_RETRY_WARNING_PERCENTAGE
        ),
        sla_warning_percentage=(
            config.TASK_SLA_WARNING_PERCENTAGE
        ),
    )


def get_robot_utilization_analytics():
    from app.analytics.robot_utilization import (
        RobotUtilizationAnalytics,
        UtilizationThresholds,
    )

    config = get_settings()

    thresholds = UtilizationThresholds(
        underutilized_below=(
            config.UTILIZATION_LOW_THRESHOLD
        ),
        high_at_or_above=(
            config.UTILIZATION_HIGH_THRESHOLD
        ),
    )

    return RobotUtilizationAnalytics(
        thresholds=thresholds
    )


# These engines are in-memory. New instances are
# empty until populated from persistent repositories.
# Do not treat them as shared application state.


# ======================================================
# AI PROVIDER CONFIGURATION
# ======================================================

@dataclass(frozen=True)
class AIProviderSettings:
    provider: str
    model: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: int


def get_ai_provider_settings(
    config: ConfigDep,
) -> AIProviderSettings:
    if not config.AI_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI features are disabled",
        )

    return AIProviderSettings(
        provider=config.DEFAULT_LLM_PROVIDER,
        model=config.DEFAULT_LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
        timeout_seconds=config.LLM_TIMEOUT_SECONDS,
    )


AIProviderDep = Annotated[
    AIProviderSettings,
    Depends(get_ai_provider_settings),
]


# ======================================================
# REQUEST ID
# ======================================================

async def get_request_id(
    request: Request,
) -> str:
    """
    Return a server-controlled request ID.

    Middleware may assign request.state.request_id.
    Otherwise a new identifier is generated.
    """
    existing = getattr(
        request.state,
        "request_id",
        None,
    )

    if existing is not None:
        return str(existing)

    generated = str(uuid.uuid4())

    request.state.request_id = generated

    return generated


RequestIdDep = Annotated[
    str,
    Depends(get_request_id),
]


__all__ = [
    "get_config",
    "ConfigDep",
    "get_session_factory",
    "get_db",
    "DatabaseDep",
    "get_redis_client",
    "get_redis",
    "RedisDep",
    "CurrentUser",
    "get_current_user",
    "CurrentUserDep",
    "Roles",
    "Permissions",
    "require_roles",
    "require_permissions",
    "ensure_robot_access",
    "get_authorized_robot_reader",
    "RobotReaderDep",
    "RobotCommandAuthorization",
    "get_robot_command_authorization",
    "RobotCommandDep",
    "get_safety_analytics",
    "get_task_success_analytics",
    "get_robot_utilization_analytics",
    "AIProviderSettings",
    "get_ai_provider_settings",
    "AIProviderDep",
    "get_request_id",
    "RequestIdDep",
]

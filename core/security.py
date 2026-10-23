"""Security module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Core Security Module

Features:
- Password hashing and verification
- JWT access and refresh tokens
- OAuth2 bearer authentication
- Token type, issuer, audience and expiry validation
- Authenticated principal construction
- Role and permission validation
- Tenant and organization isolation
- API key generation and verification
- Security audit logging
- FastAPI authentication dependencies
- Robot safety authorization boundaries

Dependencies:
    pip install fastapi pyjwt pwdlib[argon2]

Production requirements:
- Configure JWT_SECRET_KEY using a secret manager.
- Use HTTPS.
- Persist refresh token revocation and API key records.
- Enforce database-backed user and tenant membership checks.
- Never allow an LLM or AI agent to approve its own actions.
- Hardware safety interlocks must remain independent.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import uuid

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Any, Callable, Mapping, Protocol

import jwt

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from jwt import InvalidTokenError
from pwdlib import PasswordHash

from app.core.permissions import (
    AuthorizationPrincipal,
    Permission,
    PermissionService,
    ResourceContext,
    Role,
    get_permission_service,
)


logger = logging.getLogger(__name__)

password_hasher = PasswordHash.recommended()

bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="BearerAuth",
)


# ============================================================
# Constants
# ============================================================

JWT_ALGORITHM = "HS256"

DEFAULT_ACCESS_TOKEN_MINUTES = 30
DEFAULT_REFRESH_TOKEN_DAYS = 7

MIN_SECRET_LENGTH = 32
MAX_PASSWORD_LENGTH = 1024

API_KEY_PREFIX = "pai_"


# ============================================================
# Exceptions
# ============================================================


class SecurityError(Exception):
    pass


class AuthenticationError(SecurityError):
    pass


class InvalidCredentialsError(AuthenticationError):
    pass


class TokenExpiredError(AuthenticationError):
    pass


class TokenRevokedError(AuthenticationError):
    pass


class InvalidTokenTypeError(AuthenticationError):
    pass


class AccountDisabledError(AuthenticationError):
    pass


class SecurityConfigurationError(SecurityError):
    pass


# ============================================================
# Enums
# ============================================================


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"


class AuthenticationMethod(str, Enum):
    PASSWORD = "password"
    ACCESS_TOKEN = "access_token"
    API_KEY = "api_key"
    SERVICE_ACCOUNT = "service_account"


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class SecuritySettings:
    jwt_secret_key: str

    jwt_algorithm: str = JWT_ALGORITHM

    jwt_issuer: str = "physical-ai-platform"
    jwt_audience: str = "physical-ai-api"

    access_token_minutes: int = (
        DEFAULT_ACCESS_TOKEN_MINUTES
    )

    refresh_token_days: int = (
        DEFAULT_REFRESH_TOKEN_DAYS
    )

    clock_skew_seconds: int = 30

    require_active_account: bool = True

    def __post_init__(self) -> None:
        if len(self.jwt_secret_key) < MIN_SECRET_LENGTH:
            raise SecurityConfigurationError(
                "JWT secret must contain at least "
                "32 characters"
            )

        if self.jwt_algorithm != JWT_ALGORITHM:
            raise SecurityConfigurationError(
                "Only HS256 is supported by this module"
            )

        if self.access_token_minutes <= 0:
            raise ValueError(
                "access_token_minutes must be positive"
            )

        if self.refresh_token_days <= 0:
            raise ValueError(
                "refresh_token_days must be positive"
            )

        if self.clock_skew_seconds < 0:
            raise ValueError(
                "clock_skew_seconds cannot be negative"
            )

    @classmethod
    def from_environment(cls) -> "SecuritySettings":
        secret = os.getenv("JWT_SECRET_KEY")

        if not secret:
            raise SecurityConfigurationError(
                "JWT_SECRET_KEY is not configured"
            )

        return cls(
            jwt_secret_key=secret,
            jwt_issuer=os.getenv(
                "JWT_ISSUER",
                "physical-ai-platform",
            ),
            jwt_audience=os.getenv(
                "JWT_AUDIENCE",
                "physical-ai-api",
            ),
            access_token_minutes=int(
                os.getenv(
                    "ACCESS_TOKEN_MINUTES",
                    "30",
                )
            ),
            refresh_token_days=int(
                os.getenv(
                    "REFRESH_TOKEN_DAYS",
                    "7",
                )
            ),
        )


# ============================================================
# Password Security
# ============================================================


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("Password cannot be empty")

    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError("Password is too long")

    return password_hasher.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    if not plain_password or not hashed_password:
        return False

    try:
        return password_hasher.verify(
            plain_password,
            hashed_password,
        )
    except Exception:
        logger.warning(
            "Password verification failed"
        )
        return False


def password_needs_rehash(
    hashed_password: str,
) -> bool:
    return password_hasher.check_needs_rehash(
        hashed_password
    )


# ============================================================
# Authentication Models
# ============================================================


@dataclass(frozen=True)
class AuthenticatedUser:
    """
    User record loaded from a trusted identity store.
    """

    user_id: str
    tenant_id: str
    organization_id: str | None

    roles: frozenset[Role]

    is_active: bool = True
    is_service_account: bool = False

    allowed_robot_ids: frozenset[str] = frozenset()
    allowed_fleet_ids: frozenset[str] = frozenset()

    token_version: int = 0


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str

    token_type: str = "bearer"

    expires_in: int = (
        DEFAULT_ACCESS_TOKEN_MINUTES * 60
    )


@dataclass(frozen=True)
class TokenClaims:
    subject: str
    tenant_id: str
    token_type: TokenType
    token_id: str

    issued_at: datetime
    expires_at: datetime

    token_version: int = 0


# ============================================================
# Identity Store Protocol
# ============================================================


class IdentityStore(Protocol):
    """
    Implement using the application's user repository.

    Implementations must load current roles, tenant
    membership, account status and resource assignments
    from trusted persistent storage.
    """

    async def get_user(
        self,
        user_id: str,
    ) -> AuthenticatedUser | None:
        ...

    async def is_token_revoked(
        self,
        token_id: str,
    ) -> bool:
        ...


# ============================================================
# JWT Service
# ============================================================


class JWTService:
    def __init__(
        self,
        settings: SecuritySettings,
    ) -> None:
        self.settings = settings

    def _create_token(
        self,
        *,
        user: AuthenticatedUser,
        token_type: TokenType,
        expires_delta: timedelta,
    ) -> str:
        now = datetime.now(timezone.utc)
        expires_at = now + expires_delta

        payload = {
            "sub": user.user_id,
            "tenant_id": user.tenant_id,
            "type": token_type.value,
            "jti": uuid.uuid4().hex,
            "iat": now,
            "nbf": now,
            "exp": expires_at,
            "iss": self.settings.jwt_issuer,
            "aud": self.settings.jwt_audience,
            "token_version": user.token_version,
        }

        return jwt.encode(
            payload,
            self.settings.jwt_secret_key,
            algorithm=self.settings.jwt_algorithm,
        )

    def create_access_token(
        self,
        user: AuthenticatedUser,
    ) -> str:
        return self._create_token(
            user=user,
            token_type=TokenType.ACCESS,
            expires_delta=timedelta(
                minutes=self.settings.access_token_minutes
            ),
        )

    def create_refresh_token(
        self,
        user: AuthenticatedUser,
    ) -> str:
        return self._create_token(
            user=user,
            token_type=TokenType.REFRESH,
            expires_delta=timedelta(
                days=self.settings.refresh_token_days
            ),
        )

    def create_token_pair(
        self,
        user: AuthenticatedUser,
    ) -> TokenPair:
        return TokenPair(
            access_token=self.create_access_token(user),
            refresh_token=self.create_refresh_token(user),
            expires_in=(
                self.settings.access_token_minutes * 60
            ),
        )

    def decode_token(
        self,
        token: str,
        *,
        expected_type: TokenType = TokenType.ACCESS,
    ) -> TokenClaims:
        try:
            payload = jwt.decode(
                token,
                self.settings.jwt_secret_key,
                algorithms=[self.settings.jwt_algorithm],
                audience=self.settings.jwt_audience,
                issuer=self.settings.jwt_issuer,
                leeway=self.settings.clock_skew_seconds,
                options={
                    "require": [
                        "sub",
                        "tenant_id",
                        "type",
                        "jti",
                        "iat",
                        "nbf",
                        "exp",
                        "iss",
                        "aud",
                    ]
                },
            )

        except jwt.ExpiredSignatureError as exc:
            raise TokenExpiredError(
                "Authentication token expired"
            ) from exc

        except InvalidTokenError as exc:
            raise InvalidCredentialsError(
                "Invalid authentication token"
            ) from exc

        if payload.get("type") != expected_type.value:
            raise InvalidTokenTypeError(
                "Unexpected authentication token type"
            )

        subject = payload.get("sub")
        tenant_id = payload.get("tenant_id")
        token_id = payload.get("jti")

        if not all(
            isinstance(value, str) and value
            for value in (
                subject,
                tenant_id,
                token_id,
            )
        ):
            raise InvalidCredentialsError(
                "Invalid token claims"
            )

        return TokenClaims(
            subject=subject,
            tenant_id=tenant_id,
            token_type=expected_type,
            token_id=token_id,
            issued_at=datetime.fromtimestamp(
                payload["iat"],
                tz=timezone.utc,
            ),
            expires_at=datetime.fromtimestamp(
                payload["exp"],
                tz=timezone.utc,
            ),
            token_version=int(
                payload.get("token_version", 0)
            ),
        )


# ============================================================
# API Key Security
# ============================================================


def generate_api_key() -> str:
    """
    Return a new API key.

    Store only its hash in persistent storage.
    Display the plaintext key once.
    """

    return API_KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    if not api_key.startswith(API_KEY_PREFIX):
        raise InvalidCredentialsError(
            "Invalid API key format"
        )

    return hashlib.sha256(
        api_key.encode("utf-8")
    ).hexdigest()


def verify_api_key(
    provided_key: str,
    stored_hash: str,
) -> bool:
    try:
        candidate_hash = hash_api_key(
            provided_key
        )
    except InvalidCredentialsError:
        return False

    return hmac.compare_digest(
        candidate_hash,
        stored_hash,
    )


# ============================================================
# Authentication Service
# ============================================================


class AuthenticationService:
    """
    Validates JWTs against the current identity store.

    The JWT proves token authenticity; database records
    determine current account status and permissions.
    """

    def __init__(
        self,
        *,
        jwt_service: JWTService,
        identity_store: IdentityStore,
    ) -> None:
        self.jwt_service = jwt_service
        self.identity_store = identity_store

    async def authenticate_access_token(
        self,
        token: str,
    ) -> AuthorizationPrincipal:
        claims = self.jwt_service.decode_token(
            token,
            expected_type=TokenType.ACCESS,
        )

        if await self.identity_store.is_token_revoked(
            claims.token_id
        ):
            raise TokenRevokedError(
                "Authentication token revoked"
            )

        user = await self.identity_store.get_user(
            claims.subject
        )

        if user is None:
            raise InvalidCredentialsError(
                "User account not found"
            )

        if not user.is_active:
            raise AccountDisabledError(
                "User account disabled"
            )

        if not hmac.compare_digest(
            user.tenant_id,
            claims.tenant_id,
        ):
            raise InvalidCredentialsError(
                "Token tenant mismatch"
            )

        if user.token_version != claims.token_version:
            raise TokenRevokedError(
                "Authentication token is no longer valid"
            )

        return AuthorizationPrincipal(
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            organization_id=user.organization_id,
            roles=user.roles,
            allowed_robot_ids=user.allowed_robot_ids,
            allowed_fleet_ids=user.allowed_fleet_ids,
            is_active=user.is_active,
            is_service_account=user.is_service_account,
            is_human=not user.is_service_account,
        )

    async def refresh_access_token(
        self,
        refresh_token: str,
    ) -> TokenPair:
        claims = self.jwt_service.decode_token(
            refresh_token,
            expected_type=TokenType.REFRESH,
        )

        if await self.identity_store.is_token_revoked(
            claims.token_id
        ):
            raise TokenRevokedError(
                "Refresh token revoked"
            )

        user = await self.identity_store.get_user(
            claims.subject
        )

        if (
            user is None
            or not user.is_active
            or user.tenant_id != claims.tenant_id
            or user.token_version != claims.token_version
        ):
            raise InvalidCredentialsError(
                "Refresh token is no longer valid"
            )

        # Production refresh-token rotation should
        # atomically revoke the old token before issuing
        # a replacement, using the persistent token store.
        return self.jwt_service.create_token_pair(user)


# ============================================================
# FastAPI Authentication Dependencies
# ============================================================


_authentication_service: AuthenticationService | None = None


def configure_authentication(
    *,
    settings: SecuritySettings,
    identity_store: IdentityStore,
) -> AuthenticationService:
    global _authentication_service

    service = AuthenticationService(
        jwt_service=JWTService(settings),
        identity_store=identity_store,
    )

    _authentication_service = service

    return service


def get_authentication_service() -> AuthenticationService:
    if _authentication_service is None:
        raise RuntimeError(
            "Authentication service is not configured"
        )

    return _authentication_service


async def get_current_user(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> AuthorizationPrincipal:
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    try:
        principal = (
            await get_authentication_service()
            .authenticate_access_token(
                credentials.credentials
            )
        )

    except AuthenticationError as exc:
        logger.warning(
            "Authentication rejected",
            extra={
                "reason": type(exc).__name__,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired credentials",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        ) from exc

    request.state.auth_principal = principal

    return principal


def require_authenticated_permission(
    permission: Permission,
) -> Callable[..., Any]:
    """
    FastAPI dependency enforcing authentication and RBAC.

    Resource-specific access must also be checked after
    loading the resource from trusted storage.
    """

    async def dependency(
        principal: Annotated[
            AuthorizationPrincipal,
            Depends(get_current_user),
        ],
    ) -> AuthorizationPrincipal:
        decision = get_permission_service().authorize(
            principal=principal,
            permission=permission,
        )

        if not decision.allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )

        return principal

    return dependency


# ============================================================
# Tenant and Resource Isolation
# ============================================================


def require_tenant_access(
    principal: AuthorizationPrincipal,
    *,
    tenant_id: str,
) -> None:
    if not principal.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive account",
        )

    if not hmac.compare_digest(
        principal.tenant_id,
        tenant_id,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Resource access denied",
        )


def authorize_resource_action(
    *,
    principal: AuthorizationPrincipal,
    permission: Permission,
    resource: ResourceContext,
    human_approved: bool = False,
) -> None:
    """
    Enforce resource authorization.

    human_approved must originate from a verified,
    persistent approval workflow, never a client boolean.
    """

    decision = get_permission_service().authorize(
        principal=principal,
        permission=permission,
        resource=resource,
        human_approved=human_approved,
    )

    if not decision.allowed:
        logger.warning(
            "Resource authorization denied",
            extra={
                "permission": permission.value,
                "reason": decision.reason,
                "resource_type": resource.resource_type,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Resource access denied",
        )


# ============================================================
# Security Audit Events
# ============================================================


def log_security_event(
    *,
    event_type: str,
    outcome: str,
    actor_id: str | None = None,
    tenant_id: str | None = None,
    resource_type: str | None = None,
    reason: str | None = None,
) -> None:
    """
    Emit structured security metadata without secrets.

    For compliance-grade auditing, persist events to an
    append-only audit repository as well.
    """

    logger.info(
        "Security event",
        extra={
            "event_type": event_type,
            "outcome": outcome,
            "actor_id": actor_id,
            "tenant_id": tenant_id,
            "resource_type": resource_type,
            "reason": reason,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    )


# ============================================================
# Exports
# ============================================================


__all__ = [
    "SecuritySettings",
    "SecurityError",
    "AuthenticationError",
    "InvalidCredentialsError",
    "TokenExpiredError",
    "TokenRevokedError",
    "InvalidTokenTypeError",
    "AccountDisabledError",
    "SecurityConfigurationError",
    "TokenType",
    "AuthenticationMethod",
    "AuthenticatedUser",
    "TokenPair",
    "TokenClaims",
    "IdentityStore",
    "JWTService",
    "AuthenticationService",
    "hash_password",
    "verify_password",
    "password_needs_rehash",
    "generate_api_key",
    "hash_api_key",
    "verify_api_key",
    "configure_authentication",
    "get_authentication_service",
    "get_current_user",
    "require_authenticated_permission",
    "require_tenant_access",
    "authorize_resource_action",
    "log_security_event",
]
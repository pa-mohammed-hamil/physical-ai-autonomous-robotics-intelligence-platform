"""
FastAPI dependencies for authentication, authorization,
tenant isolation, and service access.

Physical AI & Autonomous Robotics Intelligence Platform

Security:
- Bearer JWT verification
- Active session verification
- Tenant membership checks
- RBAC permission enforcement
- Resource-level authorization
- No implicit administrative bypass
- No direct physical robot control authorization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Callable

from fastapi import Depends, Header, HTTPException, Path, Request, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from app.services.jwt import (
    JWTAuthenticationError,
    JWTConfigurationError,
    JWTPrincipal,
    JWTPermissionError,
    JWTService,
    get_jwt_service,
)
from app.services.rbac import (
    RBACError,
    RBACPermissionDenied,
    RBACService,
    get_rbac_service,
)
from app.services.tenant_isolation import (
    TenantAccessAction,
    TenantAccessDenied,
    TenantAccessRequest,
    TenantIsolationError,
    TenantIsolationService,
    TenantPrincipal,
    TenantResourceType,
    get_tenant_isolation_service,
)
from app.services.sessions import (
    SessionService,
    get_session_service,
)
from app.services.authentication import (
    AuthenticationService,
    get_authentication_service,
)
from app.services.authorization import (
    AuthorizationService,
    get_authorization_service,
)
from app.services.api_keys import (
    APIKeyService,
    get_api_key_service,
)
from app.services.audit import (
    AuditService,
    get_audit_service,
)


# ============================================================
# Security Scheme
# ============================================================

bearer_scheme = HTTPBearer(
    scheme_name="JWTBearer",
    description="JWT access token",
    auto_error=False,
)


# ============================================================
# Errors
# ============================================================


def unauthorized(
    detail: str = "Authentication required",
) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def forbidden(
    detail: str = "Access denied",
) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=detail,
    )


def service_unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Security service unavailable",
    )


# ============================================================
# Service Dependencies
# ============================================================


def jwt_service_dependency() -> JWTService:
    try:
        return get_jwt_service()
    except JWTConfigurationError as exc:
        raise service_unavailable() from exc


def rbac_service_dependency() -> RBACService:
    try:
        return get_rbac_service()
    except Exception as exc:
        raise service_unavailable() from exc


def tenant_service_dependency() -> TenantIsolationService:
    try:
        return get_tenant_isolation_service()
    except Exception as exc:
        raise service_unavailable() from exc


def session_service_dependency() -> SessionService:
    try:
        return get_session_service()
    except Exception as exc:
        raise service_unavailable() from exc


def authentication_service_dependency() -> AuthenticationService:
    try:
        return get_authentication_service()
    except Exception as exc:
        raise service_unavailable() from exc


def authorization_service_dependency() -> AuthorizationService:
    try:
        return get_authorization_service()
    except Exception as exc:
        raise service_unavailable() from exc


def api_key_service_dependency() -> APIKeyService:
    try:
        return get_api_key_service()
    except Exception as exc:
        raise service_unavailable() from exc


def audit_service_dependency() -> AuditService:
    try:
        return get_audit_service()
    except Exception as exc:
        raise service_unavailable() from exc


# ============================================================
# Authentication Principal
# ============================================================


async def get_current_principal(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    jwt_service: Annotated[
        JWTService,
        Depends(jwt_service_dependency),
    ],
) -> JWTPrincipal:
    if credentials is None:
        raise unauthorized()

    if credentials.scheme.lower() != "bearer":
        raise unauthorized(
            "Bearer authentication required"
        )

    token = credentials.credentials

    if not token:
        raise unauthorized(
            "Access token is missing"
        )

    try:
        return await jwt_service.verify_access_token(
            token
        )

    except JWTPermissionError as exc:
        raise forbidden() from exc

    except JWTAuthenticationError as exc:
        raise unauthorized(
            "Invalid or expired access token"
        ) from exc


CurrentPrincipal = Annotated[
    JWTPrincipal,
    Depends(get_current_principal),
]


# ============================================================
# Tenant Context
# ============================================================


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str
    user_id: str
    session_id: str
    principal: JWTPrincipal


async def get_tenant_context(
    principal: CurrentPrincipal,
    tenant_service: Annotated[
        TenantIsolationService,
        Depends(tenant_service_dependency),
    ],
) -> TenantContext:
    tenant_principal = TenantPrincipal(
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        session_id=principal.session_id,
        authenticated=True,
        is_active=True,
    )

    try:
        valid = (
            await tenant_service.verify_tenant_membership(
                tenant_principal
            )
        )
    except TenantIsolationError as exc:
        raise forbidden(
            "Tenant membership verification failed"
        ) from exc

    if not valid:
        raise forbidden(
            "Tenant membership is inactive"
        )

    return TenantContext(
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        session_id=principal.session_id,
        principal=principal,
    )


CurrentTenant = Annotated[
    TenantContext,
    Depends(get_tenant_context),
]


# ============================================================
# RBAC Permission Dependency
# ============================================================


def require_permission(
    permission: str,
) -> Callable[..., Any]:
    """
    Example:

        @router.get(
            "/robots",
            dependencies=[
                Depends(require_permission("robot:list"))
            ],
        )
        async def list_robots():
            ...
    """

    async def dependency(
        context: CurrentTenant,
        rbac_service: Annotated[
            RBACService,
            Depends(rbac_service_dependency),
        ],
    ) -> TenantContext:
        try:
            await rbac_service.require_permission(
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                permission=permission,
            )

        except (
            RBACPermissionDenied,
            RBACError,
        ) as exc:
            raise forbidden(
                "Required permission is missing"
            ) from exc

        return context

    return dependency


# ============================================================
# Multiple Permissions
# ============================================================


def require_all_permissions(
    *permissions: str,
) -> Callable[..., Any]:
    async def dependency(
        context: CurrentTenant,
        rbac_service: Annotated[
            RBACService,
            Depends(rbac_service_dependency),
        ],
    ) -> TenantContext:
        try:
            for permission in permissions:
                await rbac_service.require_permission(
                    tenant_id=context.tenant_id,
                    user_id=context.user_id,
                    permission=permission,
                )

        except RBACError as exc:
            raise forbidden(
                "One or more permissions are missing"
            ) from exc

        return context

    return dependency


def require_any_permission(
    *permissions: str,
) -> Callable[..., Any]:
    async def dependency(
        context: CurrentTenant,
        rbac_service: Annotated[
            RBACService,
            Depends(rbac_service_dependency),
        ],
    ) -> TenantContext:
        if not permissions:
            raise forbidden(
                "No permissions configured"
            )

        for permission in permissions:
            try:
                decision = (
                    await rbac_service.check_permission(
                        tenant_id=context.tenant_id,
                        user_id=context.user_id,
                        permission=permission,
                    )
                )

                if decision.allowed:
                    return context

            except RBACError:
                continue

        raise forbidden(
            "No matching permission"
        )

    return dependency


# ============================================================
# Tenant Resource Authorization
# ============================================================


def require_resource_access(
    *,
    resource_type: TenantResourceType,
    action: TenantAccessAction,
    path_parameter: str = "resource_id",
    permission: str | None = None,
) -> Callable[..., Any]:
    """
    Enforce RBAC and tenant resource isolation.

    Example:

        @router.get(
            "/robots/{robot_id}",
            dependencies=[
                Depends(
                    require_resource_access(
                        resource_type=TenantResourceType.ROBOT,
                        action=TenantAccessAction.READ,
                        path_parameter="robot_id",
                        permission="robot:read",
                    )
                )
            ],
        )
        async def get_robot(robot_id: str):
            ...
    """

    async def dependency(
        request: Request,
        context: CurrentTenant,
        tenant_service: Annotated[
            TenantIsolationService,
            Depends(tenant_service_dependency),
        ],
        rbac_service: Annotated[
            RBACService,
            Depends(rbac_service_dependency),
        ],
    ) -> TenantContext:
        resource_id = request.path_params.get(
            path_parameter
        )

        if not isinstance(resource_id, str):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Resource identifier is missing",
            )

        required_permission = (
            permission
            or f"{resource_type.value}:{action.value}"
        )

        try:
            await rbac_service.require_permission(
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                permission=required_permission,
            )

            await tenant_service.require_access(
                TenantAccessRequest(
                    principal=TenantPrincipal(
                        tenant_id=context.tenant_id,
                        user_id=context.user_id,
                        session_id=context.session_id,
                    ),
                    resource_type=resource_type,
                    resource_id=resource_id,
                    action=action,
                    requested_tenant_id=context.tenant_id,
                )
            )

        except (
            RBACError,
            TenantAccessDenied,
            TenantIsolationError,
        ) as exc:
            raise forbidden(
                "Resource access denied"
            ) from exc

        return context

    return dependency


# ============================================================
# Tenant Header Verification
# ============================================================


async def verify_tenant_header(
    context: CurrentTenant,
    x_tenant_id: Annotated[
        str | None,
        Header(alias="X-Tenant-ID"),
    ] = None,
) -> TenantContext:
    """
    Optional tenant header.

    The header can confirm the authenticated
    tenant but cannot switch tenant identity.
    """

    if (
        x_tenant_id is not None
        and x_tenant_id != context.tenant_id
    ):
        raise forbidden(
            "Tenant header does not match "
            "authenticated tenant"
        )

    return context


# ============================================================
# Active Session Dependency
# ============================================================


async def get_active_session(
    context: CurrentTenant,
    session_service: Annotated[
        SessionService,
        Depends(session_service_dependency),
    ],
) -> Any:
    try:
        return await session_service.require_session(
            session_id=context.session_id,
            tenant_id=context.tenant_id,
            user_id=context.user_id,
        )

    except Exception as exc:
        raise unauthorized(
            "Session is invalid or expired"
        ) from exc


# ============================================================
# Convenience Dependencies
# ============================================================


async def require_authenticated_user(
    principal: CurrentPrincipal,
) -> JWTPrincipal:
    return principal


async def require_active_tenant(
    context: CurrentTenant,
) -> TenantContext:
    return context


async def require_tenant_admin(
    context: Annotated[
        TenantContext,
        Depends(require_permission("user:manage")),
    ],
) -> TenantContext:
    return context


async def require_robot_read(
    context: Annotated[
        TenantContext,
        Depends(require_permission("robot:read")),
    ],
) -> TenantContext:
    return context


async def require_fleet_read(
    context: Annotated[
        TenantContext,
        Depends(require_permission("fleet:read")),
    ],
) -> TenantContext:
    return context


async def require_telemetry_read(
    context: Annotated[
        TenantContext,
        Depends(require_permission("telemetry:read")),
    ],
) -> TenantContext:
    return context


async def require_maintenance_read(
    context: Annotated[
        TenantContext,
        Depends(require_permission("maintenance:read")),
    ],
) -> TenantContext:
    return context


async def require_model_read(
    context: Annotated[
        TenantContext,
        Depends(require_permission("model:read")),
    ],
) -> TenantContext:
    return context


async def require_simulation_execute(
    context: Annotated[
        TenantContext,
        Depends(require_permission("simulation:execute")),
    ],
) -> TenantContext:
    """
    Offline simulation execution only.

    Never use this dependency for real robot motion.
    """
    return context


# ============================================================
# Type Aliases
# ============================================================


JWTServiceDep = Annotated[
    JWTService,
    Depends(jwt_service_dependency),
]

RBACServiceDep = Annotated[
    RBACService,
    Depends(rbac_service_dependency),
]

TenantIsolationServiceDep = Annotated[
    TenantIsolationService,
    Depends(tenant_service_dependency),
]

SessionServiceDep = Annotated[
    SessionService,
    Depends(session_service_dependency),
]

AuthenticationServiceDep = Annotated[
    AuthenticationService,
    Depends(authentication_service_dependency),
]

AuthorizationServiceDep = Annotated[
    AuthorizationService,
    Depends(authorization_service_dependency),
]

APIKeyServiceDep = Annotated[
    APIKeyService,
    Depends(api_key_service_dependency),
]

AuditServiceDep = Annotated[
    AuditService,
    Depends(audit_service_dependency),
]


# ============================================================
# Exports
# ============================================================


__all__ = [
    "bearer_scheme",
    "unauthorized",
    "forbidden",
    "service_unavailable",
    "jwt_service_dependency",
    "rbac_service_dependency",
    "tenant_service_dependency",
    "session_service_dependency",
    "authentication_service_dependency",
    "authorization_service_dependency",
    "api_key_service_dependency",
    "audit_service_dependency",
    "get_current_principal",
    "get_tenant_context",
    "get_active_session",
    "verify_tenant_header",
    "require_permission",
    "require_all_permissions",
    "require_any_permission",
    "require_resource_access",
    "require_authenticated_user",
    "require_active_tenant",
    "require_tenant_admin",
    "require_robot_read",
    "require_fleet_read",
    "require_telemetry_read",
    "require_maintenance_read",
    "require_model_read",
    "require_simulation_execute",
    "TenantContext",
    "CurrentPrincipal",
    "CurrentTenant",
    "JWTServiceDep",
    "RBACServiceDep",
    "TenantIsolationServiceDep",
    "SessionServiceDep",
    "AuthenticationServiceDep",
    "AuthorizationServiceDep",
    "APIKeyServiceDep",
    "AuditServiceDep",
]
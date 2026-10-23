"""Permissions module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Core Permissions and Authorization

Features:
- Role-based access control (RBAC)
- Fine-grained permissions
- Tenant and organization isolation
- Robot and fleet resource authorization
- FastAPI permission dependencies
- Scoped access policies
- Human approval requirements
- Safety-critical action restrictions
- Authorization audit logging

Security:
- Default deny
- No implicit cross-tenant access
- Client-provided roles are never trusted
- AI agents cannot authorize themselves
- Authorization does not replace hardware safety controls
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Callable, FrozenSet, Mapping

from fastapi import Depends, HTTPException, Request, status

logger = logging.getLogger(__name__)


# ============================================================
# Roles
# ============================================================


class Role(str, Enum):
    SUPER_ADMIN = "super_admin"
    ORGANIZATION_ADMIN = "organization_admin"
    FLEET_MANAGER = "fleet_manager"
    ROBOT_OPERATOR = "robot_operator"
    MAINTENANCE_ENGINEER = "maintenance_engineer"
    SAFETY_OFFICER = "safety_officer"
    AI_ENGINEER = "ai_engineer"
    DATA_SCIENTIST = "data_scientist"
    RESEARCHER = "researcher"
    AUDITOR = "auditor"
    VIEWER = "viewer"
    SERVICE_ACCOUNT = "service_account"


# ============================================================
# Permissions
# ============================================================


class Permission(str, Enum):

    # Organization
    ORGANIZATION_READ = "organization:read"
    ORGANIZATION_CREATE = "organization:create"
    ORGANIZATION_UPDATE = "organization:update"
    ORGANIZATION_DELETE = "organization:delete"

    # Users
    USER_READ = "user:read"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_ROLE_ASSIGN = "user:role_assign"

    # Robots
    ROBOT_READ = "robot:read"
    ROBOT_CREATE = "robot:create"
    ROBOT_UPDATE = "robot:update"
    ROBOT_DELETE = "robot:delete"
    ROBOT_CONTROL = "robot:control"
    ROBOT_START = "robot:start"
    ROBOT_STOP = "robot:stop"
    ROBOT_RESET = "robot:reset"
    ROBOT_CONFIGURE = "robot:configure"

    # Fleets
    FLEET_READ = "fleet:read"
    FLEET_CREATE = "fleet:create"
    FLEET_UPDATE = "fleet:update"
    FLEET_DELETE = "fleet:delete"
    FLEET_CONTROL = "fleet:control"

    # Telemetry
    TELEMETRY_READ = "telemetry:read"
    TELEMETRY_EXPORT = "telemetry:export"
    TELEMETRY_INGEST = "telemetry:ingest"

    # Sensors
    SENSOR_READ = "sensor:read"
    SENSOR_CREATE = "sensor:create"
    SENSOR_UPDATE = "sensor:update"
    SENSOR_CALIBRATE = "sensor:calibrate"

    # Navigation
    NAVIGATION_READ = "navigation:read"
    NAVIGATION_PLAN = "navigation:plan"
    NAVIGATION_EXECUTE = "navigation:execute"

    # Manipulation
    MANIPULATION_READ = "manipulation:read"
    MANIPULATION_PLAN = "manipulation:plan"
    MANIPULATION_EXECUTE = "manipulation:execute"

    # AI models
    MODEL_READ = "model:read"
    MODEL_CREATE = "model:create"
    MODEL_UPDATE = "model:update"
    MODEL_DELETE = "model:delete"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"
    MODEL_INFERENCE = "model:inference"

    # VLA
    VLA_READ = "vla:read"
    VLA_INFERENCE = "vla:inference"
    VLA_EXECUTE = "vla:execute"

    # World models
    WORLD_MODEL_READ = "world_model:read"
    WORLD_MODEL_TRAIN = "world_model:train"
    WORLD_MODEL_PREDICT = "world_model:predict"

    # Agents
    AGENT_READ = "agent:read"
    AGENT_EXECUTE = "agent:execute"
    AGENT_CONFIGURE = "agent:configure"

    # RAG
    RAG_READ = "rag:read"
    RAG_QUERY = "rag:query"
    RAG_INGEST = "rag:ingest"
    RAG_DELETE = "rag:delete"

    # Simulation
    SIMULATION_READ = "simulation:read"
    SIMULATION_CREATE = "simulation:create"
    SIMULATION_EXECUTE = "simulation:execute"
    SIMULATION_DELETE = "simulation:delete"

    # Evaluations
    EVALUATION_READ = "evaluation:read"
    EVALUATION_CREATE = "evaluation:create"
    EVALUATION_EXECUTE = "evaluation:execute"

    # Maintenance
    MAINTENANCE_READ = "maintenance:read"
    MAINTENANCE_CREATE = "maintenance:create"
    MAINTENANCE_UPDATE = "maintenance:update"
    MAINTENANCE_APPROVE = "maintenance:approve"

    # Repair
    REPAIR_READ = "repair:read"
    REPAIR_ESTIMATE = "repair:estimate"
    REPAIR_CREATE = "repair:create"
    REPAIR_APPROVE = "repair:approve"
    REPAIR_COMPLETE = "repair:complete"

    # Parts
    PART_READ = "part:read"
    PART_CREATE = "part:create"
    PART_UPDATE = "part:update"
    PART_INSPECT = "part:inspect"
    PART_PRICE_ESTIMATE = "part:price_estimate"

    # Safety
    SAFETY_READ = "safety:read"
    SAFETY_ALERT_CREATE = "safety:alert_create"
    SAFETY_POLICY_READ = "safety:policy_read"
    SAFETY_POLICY_UPDATE = "safety:policy_update"
    SAFETY_APPROVE = "safety:approve"
    SAFETY_EMERGENCY_STOP = "safety:emergency_stop"
    SAFETY_RESET_APPROVE = "safety:reset_approve"

    # Incidents
    INCIDENT_READ = "incident:read"
    INCIDENT_CREATE = "incident:create"
    INCIDENT_UPDATE = "incident:update"
    INCIDENT_CLOSE = "incident:close"

    # Audit
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"

    # System
    SYSTEM_READ = "system:read"
    SYSTEM_CONFIGURE = "system:configure"
    SYSTEM_METRICS_READ = "system:metrics_read"


# ============================================================
# Safety-Critical Permissions
# ============================================================


SAFETY_CRITICAL_PERMISSIONS: FrozenSet[Permission] = frozenset({
    Permission.ROBOT_CONTROL,
    Permission.ROBOT_START,
    Permission.ROBOT_RESET,
    Permission.FLEET_CONTROL,
    Permission.NAVIGATION_EXECUTE,
    Permission.MANIPULATION_EXECUTE,
    Permission.VLA_EXECUTE,
    Permission.SAFETY_POLICY_UPDATE,
    Permission.SAFETY_RESET_APPROVE,
})


HUMAN_APPROVAL_PERMISSIONS: FrozenSet[Permission] = frozenset({
    Permission.ROBOT_RESET,
    Permission.MODEL_DEPLOY,
    Permission.MAINTENANCE_APPROVE,
    Permission.REPAIR_APPROVE,
    Permission.SAFETY_POLICY_UPDATE,
    Permission.SAFETY_RESET_APPROVE,
})


# ============================================================
# Role Permission Mapping
# ============================================================


ROLE_PERMISSIONS: dict[Role, FrozenSet[Permission]] = {
    Role.SUPER_ADMIN: frozenset(Permission),

    Role.ORGANIZATION_ADMIN: frozenset({
        Permission.ORGANIZATION_READ,
        Permission.ORGANIZATION_UPDATE,
        Permission.USER_READ,
        Permission.USER_CREATE,
        Permission.USER_UPDATE,
        Permission.USER_DELETE,
        Permission.USER_ROLE_ASSIGN,
        Permission.ROBOT_READ,
        Permission.ROBOT_CREATE,
        Permission.ROBOT_UPDATE,
        Permission.FLEET_READ,
        Permission.FLEET_CREATE,
        Permission.FLEET_UPDATE,
        Permission.TELEMETRY_READ,
        Permission.SENSOR_READ,
        Permission.MODEL_READ,
        Permission.MAINTENANCE_READ,
        Permission.REPAIR_READ,
        Permission.PART_READ,
        Permission.SAFETY_READ,
        Permission.INCIDENT_READ,
        Permission.AUDIT_READ,
        Permission.SYSTEM_READ,
        Permission.SYSTEM_METRICS_READ,
    }),

    Role.FLEET_MANAGER: frozenset({
        Permission.ROBOT_READ,
        Permission.ROBOT_UPDATE,
        Permission.ROBOT_CONTROL,
        Permission.ROBOT_START,
        Permission.ROBOT_STOP,
        Permission.FLEET_READ,
        Permission.FLEET_CREATE,
        Permission.FLEET_UPDATE,
        Permission.FLEET_CONTROL,
        Permission.TELEMETRY_READ,
        Permission.TELEMETRY_EXPORT,
        Permission.SENSOR_READ,
        Permission.NAVIGATION_READ,
        Permission.NAVIGATION_PLAN,
        Permission.NAVIGATION_EXECUTE,
        Permission.MANIPULATION_READ,
        Permission.MAINTENANCE_READ,
        Permission.MAINTENANCE_CREATE,
        Permission.REPAIR_READ,
        Permission.SAFETY_READ,
        Permission.INCIDENT_READ,
        Permission.INCIDENT_CREATE,
    }),

    Role.ROBOT_OPERATOR: frozenset({
        Permission.ROBOT_READ,
        Permission.ROBOT_CONTROL,
        Permission.ROBOT_START,
        Permission.ROBOT_STOP,
        Permission.FLEET_READ,
        Permission.TELEMETRY_READ,
        Permission.SENSOR_READ,
        Permission.NAVIGATION_READ,
        Permission.NAVIGATION_PLAN,
        Permission.NAVIGATION_EXECUTE,
        Permission.MANIPULATION_READ,
        Permission.MANIPULATION_PLAN,
        Permission.MANIPULATION_EXECUTE,
        Permission.VLA_READ,
        Permission.VLA_INFERENCE,
        Permission.SAFETY_READ,
        Permission.SAFETY_EMERGENCY_STOP,
        Permission.INCIDENT_READ,
        Permission.INCIDENT_CREATE,
    }),

    Role.MAINTENANCE_ENGINEER: frozenset({
        Permission.ROBOT_READ,
        Permission.ROBOT_UPDATE,
        Permission.ROBOT_STOP,
        Permission.TELEMETRY_READ,
        Permission.SENSOR_READ,
        Permission.SENSOR_UPDATE,
        Permission.SENSOR_CALIBRATE,
        Permission.MAINTENANCE_READ,
        Permission.MAINTENANCE_CREATE,
        Permission.MAINTENANCE_UPDATE,
        Permission.REPAIR_READ,
        Permission.REPAIR_ESTIMATE,
        Permission.REPAIR_CREATE,
        Permission.REPAIR_COMPLETE,
        Permission.PART_READ,
        Permission.PART_CREATE,
        Permission.PART_UPDATE,
        Permission.PART_INSPECT,
        Permission.PART_PRICE_ESTIMATE,
        Permission.SAFETY_READ,
        Permission.INCIDENT_READ,
    }),

    Role.SAFETY_OFFICER: frozenset({
        Permission.ROBOT_READ,
        Permission.ROBOT_STOP,
        Permission.FLEET_READ,
        Permission.TELEMETRY_READ,
        Permission.SENSOR_READ,
        Permission.SAFETY_READ,
        Permission.SAFETY_ALERT_CREATE,
        Permission.SAFETY_POLICY_READ,
        Permission.SAFETY_POLICY_UPDATE,
        Permission.SAFETY_APPROVE,
        Permission.SAFETY_EMERGENCY_STOP,
        Permission.SAFETY_RESET_APPROVE,
        Permission.INCIDENT_READ,
        Permission.INCIDENT_CREATE,
        Permission.INCIDENT_UPDATE,
        Permission.INCIDENT_CLOSE,
        Permission.AUDIT_READ,
    }),

    Role.AI_ENGINEER: frozenset({
        Permission.ROBOT_READ,
        Permission.TELEMETRY_READ,
        Permission.SENSOR_READ,
        Permission.MODEL_READ,
        Permission.MODEL_CREATE,
        Permission.MODEL_UPDATE,
        Permission.MODEL_TRAIN,
        Permission.MODEL_DEPLOY,
        Permission.MODEL_INFERENCE,
        Permission.VLA_READ,
        Permission.VLA_INFERENCE,
        Permission.WORLD_MODEL_READ,
        Permission.WORLD_MODEL_TRAIN,
        Permission.WORLD_MODEL_PREDICT,
        Permission.AGENT_READ,
        Permission.AGENT_EXECUTE,
        Permission.AGENT_CONFIGURE,
        Permission.RAG_READ,
        Permission.RAG_QUERY,
        Permission.RAG_INGEST,
        Permission.SIMULATION_READ,
        Permission.SIMULATION_CREATE,
        Permission.SIMULATION_EXECUTE,
        Permission.EVALUATION_READ,
        Permission.EVALUATION_CREATE,
        Permission.EVALUATION_EXECUTE,
        Permission.SYSTEM_METRICS_READ,
    }),

    Role.DATA_SCIENTIST: frozenset({
        Permission.ROBOT_READ,
        Permission.TELEMETRY_READ,
        Permission.TELEMETRY_EXPORT,
        Permission.SENSOR_READ,
        Permission.MODEL_READ,
        Permission.MODEL_CREATE,
        Permission.MODEL_TRAIN,
        Permission.MODEL_INFERENCE,
        Permission.WORLD_MODEL_READ,
        Permission.WORLD_MODEL_PREDICT,
        Permission.SIMULATION_READ,
        Permission.SIMULATION_EXECUTE,
        Permission.EVALUATION_READ,
        Permission.EVALUATION_CREATE,
        Permission.EVALUATION_EXECUTE,
    }),

    Role.RESEARCHER: frozenset({
        Permission.ROBOT_READ,
        Permission.TELEMETRY_READ,
        Permission.SENSOR_READ,
        Permission.MODEL_READ,
        Permission.MODEL_INFERENCE,
        Permission.VLA_READ,
        Permission.VLA_INFERENCE,
        Permission.WORLD_MODEL_READ,
        Permission.WORLD_MODEL_PREDICT,
        Permission.RAG_READ,
        Permission.RAG_QUERY,
        Permission.SIMULATION_READ,
        Permission.SIMULATION_CREATE,
        Permission.SIMULATION_EXECUTE,
        Permission.EVALUATION_READ,
        Permission.EVALUATION_CREATE,
        Permission.EVALUATION_EXECUTE,
    }),

    Role.AUDITOR: frozenset({
        Permission.ORGANIZATION_READ,
        Permission.USER_READ,
        Permission.ROBOT_READ,
        Permission.FLEET_READ,
        Permission.TELEMETRY_READ,
        Permission.MODEL_READ,
        Permission.SAFETY_READ,
        Permission.INCIDENT_READ,
        Permission.MAINTENANCE_READ,
        Permission.REPAIR_READ,
        Permission.AUDIT_READ,
        Permission.AUDIT_EXPORT,
    }),

    Role.VIEWER: frozenset({
        Permission.ROBOT_READ,
        Permission.FLEET_READ,
        Permission.TELEMETRY_READ,
        Permission.SENSOR_READ,
        Permission.SAFETY_READ,
        Permission.MAINTENANCE_READ,
        Permission.REPAIR_READ,
    }),

    Role.SERVICE_ACCOUNT: frozenset({
        Permission.ROBOT_READ,
        Permission.FLEET_READ,
        Permission.TELEMETRY_INGEST,
        Permission.SENSOR_READ,
        Permission.MODEL_INFERENCE,
        Permission.RAG_QUERY,
        Permission.SAFETY_ALERT_CREATE,
    }),
}


# ============================================================
# Authorization Models
# ============================================================


@dataclass(frozen=True)
class AuthorizationPrincipal:
    """
    Identity established by trusted authentication.

    Never construct from unverified request JSON or headers.
    """

    user_id: str
    tenant_id: str
    organization_id: str | None = None

    roles: FrozenSet[Role] = field(
        default_factory=frozenset
    )

    explicit_permissions: FrozenSet[Permission] = field(
        default_factory=frozenset
    )

    denied_permissions: FrozenSet[Permission] = field(
        default_factory=frozenset
    )

    allowed_robot_ids: FrozenSet[str] = field(
        default_factory=frozenset
    )

    allowed_fleet_ids: FrozenSet[str] = field(
        default_factory=frozenset
    )

    is_active: bool = True
    is_service_account: bool = False
    is_human: bool = True

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise ValueError("user_id is required")

        if not self.tenant_id.strip():
            raise ValueError("tenant_id is required")


@dataclass(frozen=True)
class ResourceContext:
    resource_type: str
    resource_id: str

    tenant_id: str
    organization_id: str | None = None

    robot_id: str | None = None
    fleet_id: str | None = None


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    permission: Permission
    reason: str
    requires_human_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "permission": self.permission.value,
            "reason": self.reason,
            "requires_human_approval": (
                self.requires_human_approval
            ),
        }


# ============================================================
# Authorization Exceptions
# ============================================================


class AuthorizationError(Exception):
    pass


class PermissionDeniedError(AuthorizationError):
    pass


class TenantAccessDeniedError(AuthorizationError):
    pass


class HumanApprovalRequiredError(AuthorizationError):
    pass


# ============================================================
# Permission Resolution
# ============================================================


def get_role_permissions(
    role: Role,
) -> FrozenSet[Permission]:
    return ROLE_PERMISSIONS.get(
        role,
        frozenset(),
    )


def resolve_permissions(
    principal: AuthorizationPrincipal,
) -> FrozenSet[Permission]:
    permissions: set[Permission] = set(
        principal.explicit_permissions
    )

    for role in principal.roles:
        permissions.update(
            get_role_permissions(role)
        )

    permissions.difference_update(
        principal.denied_permissions
    )

    return frozenset(permissions)


def has_permission(
    principal: AuthorizationPrincipal,
    permission: Permission,
) -> bool:
    if not principal.is_active:
        return False

    return permission in resolve_permissions(
        principal
    )


def has_any_permission(
    principal: AuthorizationPrincipal,
    permissions: set[Permission] | FrozenSet[Permission],
) -> bool:
    return any(
        has_permission(principal, permission)
        for permission in permissions
    )


def has_all_permissions(
    principal: AuthorizationPrincipal,
    permissions: set[Permission] | FrozenSet[Permission],
) -> bool:
    return all(
        has_permission(principal, permission)
        for permission in permissions
    )


# ============================================================
# Authorization Service
# ============================================================


class PermissionService:
    """
    Central authorization policy evaluator.

    Role permissions grant eligibility only.

    Actual physical robot commands must also pass:
    - Robot ownership and assignment checks
    - Safety policy validation
    - Human approval where required
    - Independent hardware safety interlocks
    """

    def authorize(
        self,
        *,
        principal: AuthorizationPrincipal,
        permission: Permission,
        resource: ResourceContext | None = None,
        human_approved: bool = False,
    ) -> AuthorizationDecision:

        if not principal.is_active:
            return AuthorizationDecision(
                allowed=False,
                permission=permission,
                reason="inactive_principal",
            )

        if not has_permission(
            principal,
            permission,
        ):
            return AuthorizationDecision(
                allowed=False,
                permission=permission,
                reason="permission_not_granted",
            )

        if resource is not None:
            if (
                principal.tenant_id
                != resource.tenant_id
            ):
                return AuthorizationDecision(
                    allowed=False,
                    permission=permission,
                    reason="tenant_mismatch",
                )

            if (
                principal.organization_id is not None
                and resource.organization_id is not None
                and principal.organization_id
                != resource.organization_id
            ):
                return AuthorizationDecision(
                    allowed=False,
                    permission=permission,
                    reason="organization_mismatch",
                )

            if (
                resource.robot_id is not None
                and resource.robot_id
                not in principal.allowed_robot_ids
                and Role.SUPER_ADMIN not in principal.roles
            ):
                return AuthorizationDecision(
                    allowed=False,
                    permission=permission,
                    reason="robot_scope_denied",
                )

            if (
                resource.fleet_id is not None
                and resource.fleet_id
                not in principal.allowed_fleet_ids
                and Role.SUPER_ADMIN not in principal.roles
            ):
                return AuthorizationDecision(
                    allowed=False,
                    permission=permission,
                    reason="fleet_scope_denied",
                )

        if (
            permission in SAFETY_CRITICAL_PERMISSIONS
            and principal.is_service_account
        ):
            return AuthorizationDecision(
                allowed=False,
                permission=permission,
                reason="service_account_physical_control_denied",
            )

        if permission in HUMAN_APPROVAL_PERMISSIONS:
            if not principal.is_human:
                return AuthorizationDecision(
                    allowed=False,
                    permission=permission,
                    reason="human_actor_required",
                    requires_human_approval=True,
                )

            if not human_approved:
                return AuthorizationDecision(
                    allowed=False,
                    permission=permission,
                    reason="human_approval_required",
                    requires_human_approval=True,
                )

        return AuthorizationDecision(
            allowed=True,
            permission=permission,
            reason="authorized",
        )

    def require(
        self,
        *,
        principal: AuthorizationPrincipal,
        permission: Permission,
        resource: ResourceContext | None = None,
        human_approved: bool = False,
    ) -> None:
        decision = self.authorize(
            principal=principal,
            permission=permission,
            resource=resource,
            human_approved=human_approved,
        )

        if decision.allowed:
            return

        logger.warning(
            "Authorization denied",
            extra={
                "permission": permission.value,
                "reason": decision.reason,
                "resource_type": (
                    resource.resource_type
                    if resource
                    else None
                ),
            },
        )

        if decision.requires_human_approval:
            raise HumanApprovalRequiredError(
                decision.reason
            )

        if decision.reason == "tenant_mismatch":
            raise TenantAccessDeniedError(
                decision.reason
            )

        raise PermissionDeniedError(
            decision.reason
        )


# ============================================================
# FastAPI Dependencies
# ============================================================


async def get_current_principal(
    request: Request,
) -> AuthorizationPrincipal:
    """
    Requires authentication middleware to populate:

        request.state.auth_principal

    The value must be an AuthorizationPrincipal created
    after verifying authentication credentials.

    This dependency intentionally does not trust request
    headers, cookies or JSON without authentication.
    """

    principal = getattr(
        request.state,
        "auth_principal",
        None,
    )

    if not isinstance(
        principal,
        AuthorizationPrincipal,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    if not principal.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive account",
        )

    return principal


def require_permission(
    permission: Permission,
) -> Callable[..., Any]:

    async def dependency(
        principal: Annotated[
            AuthorizationPrincipal,
            Depends(get_current_principal),
        ],
    ) -> AuthorizationPrincipal:

        if not has_permission(
            principal,
            permission,
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )

        return principal

    return dependency


def require_any_permission(
    *permissions: Permission,
) -> Callable[..., Any]:

    required = frozenset(permissions)

    async def dependency(
        principal: Annotated[
            AuthorizationPrincipal,
            Depends(get_current_principal),
        ],
    ) -> AuthorizationPrincipal:

        if not has_any_permission(
            principal,
            required,
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )

        return principal

    return dependency


def require_all_permissions(
    *permissions: Permission,
) -> Callable[..., Any]:

    required = frozenset(permissions)

    async def dependency(
        principal: Annotated[
            AuthorizationPrincipal,
            Depends(get_current_principal),
        ],
    ) -> AuthorizationPrincipal:

        if not has_all_permissions(
            principal,
            required,
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )

        return principal

    return dependency


def require_role(
    *roles: Role,
) -> Callable[..., Any]:

    required = frozenset(roles)

    async def dependency(
        principal: Annotated[
            AuthorizationPrincipal,
            Depends(get_current_principal),
        ],
    ) -> AuthorizationPrincipal:

        if not principal.roles.intersection(required):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Required role not assigned",
            )

        return principal

    return dependency


# ============================================================
# Resource Authorization Helpers
# ============================================================


def require_robot_access(
    principal: AuthorizationPrincipal,
    *,
    tenant_id: str,
    robot_id: str,
    permission: Permission = Permission.ROBOT_READ,
) -> None:

    resource = ResourceContext(
        resource_type="robot",
        resource_id=robot_id,
        tenant_id=tenant_id,
        robot_id=robot_id,
    )

    PermissionService().require(
        principal=principal,
        permission=permission,
        resource=resource,
    )


def require_fleet_access(
    principal: AuthorizationPrincipal,
    *,
    tenant_id: str,
    fleet_id: str,
    permission: Permission = Permission.FLEET_READ,
) -> None:

    resource = ResourceContext(
        resource_type="fleet",
        resource_id=fleet_id,
        tenant_id=tenant_id,
        fleet_id=fleet_id,
    )

    PermissionService().require(
        principal=principal,
        permission=permission,
        resource=resource,
    )


def require_safety_approval(
    principal: AuthorizationPrincipal,
    *,
    permission: Permission,
    human_approved: bool,
    resource: ResourceContext | None = None,
) -> None:

    if permission not in HUMAN_APPROVAL_PERMISSIONS:
        raise ValueError(
            "Permission does not require human approval"
        )

    PermissionService().require(
        principal=principal,
        permission=permission,
        resource=resource,
        human_approved=human_approved,
    )


# ============================================================
# Singleton
# ============================================================


_permission_service: PermissionService | None = None


def get_permission_service() -> PermissionService:
    global _permission_service

    if _permission_service is None:
        _permission_service = PermissionService()

    return _permission_service


# ============================================================
# Exports
# ============================================================


__all__ = [
    "Role",
    "Permission",
    "ROLE_PERMISSIONS",
    "SAFETY_CRITICAL_PERMISSIONS",
    "HUMAN_APPROVAL_PERMISSIONS",
    "AuthorizationPrincipal",
    "ResourceContext",
    "AuthorizationDecision",
    "AuthorizationError",
    "PermissionDeniedError",
    "TenantAccessDeniedError",
    "HumanApprovalRequiredError",
    "PermissionService",
    "get_role_permissions",
    "resolve_permissions",
    "has_permission",
    "has_any_permission",
    "has_all_permissions",
    "get_current_principal",
    "require_permission",
    "require_any_permission",
    "require_all_permissions",
    "require_role",
    "require_robot_access",
    "require_fleet_access",
    "require_safety_approval",
    "get_permission_service",
]
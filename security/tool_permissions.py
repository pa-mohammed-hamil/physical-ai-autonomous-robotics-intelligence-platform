
"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/security/tool_permissions.py

Python 3.11+
Standard library only.

Features:
    - AI agent tool allowlisting
    - Role-based tool permissions
    - Agent-specific tool grants
    - Tenant-scoped authorization
    - Robot/resource scope restrictions
    - Read/write/execute permission separation
    - Sensitive operation approval gates
    - Tool argument validation hooks
    - Time-limited permissions
    - Permission revocation
    - Deny-by-default policy
    - Security audit logging
    - JSON report export

IMPORTANT:
    This module authorizes requests; it does not
    execute tools or physical robot commands.

    The caller must supply a verified identity
    from a trusted authentication layer.

    Human approvals and physical robot safety
    must be independently enforced.
"""

from __future__ import annotations

import json
import re
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class ToolPermissionError(Exception):
    pass


class ToolValidationError(ToolPermissionError):
    pass


class ToolAccessDenied(ToolPermissionError):
    pass


class ToolNotFoundError(ToolPermissionError):
    pass


class ToolGrantError(ToolPermissionError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ToolPermission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    NAVIGATION = "navigation"
    PERCEPTION = "perception"
    MANIPULATION = "manipulation"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    RESEARCH = "research"
    DATA = "data"
    EVALUATION = "evaluation"
    SIMULATION = "simulation"
    VIEWER = "viewer"


class ToolCategory(str, Enum):
    ROBOT = "robot"
    NAVIGATION = "navigation"
    PERCEPTION = "perception"
    MANIPULATION = "manipulation"
    SIMULATION = "simulation"
    TELEMETRY = "telemetry"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    MODEL = "model"
    FILE = "file"
    RAG = "rag"
    WEB = "web"
    DATA = "data"
    EVALUATION = "evaluation"
    REPORT = "report"


class ToolDecisionStatus(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    APPROVAL_REQUIRED = "approval_required"


class ToolRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def validate_identifier(
    value: str,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
            value,
        )
        or value in {".", ".."}
    ):
        raise ToolValidationError(
            f"Invalid {field_name}"
        )

    return value


def validate_tool_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[a-z][a-z0-9_]*"
            r"(?:\.[a-z][a-z0-9_]*){1,3}",
            value,
        )
    ):
        raise ToolValidationError(
            "Invalid tool name"
        )

    return value


def parse_expiration(
    value: datetime | None,
) -> datetime | None:
    if value is None:
        return None

    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ToolValidationError(
            "Expiration must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ToolValidationError(
            "Invalid JSON-compatible value"
        ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ToolPermissionConfig:
    maximum_tools: int = 1000
    maximum_agents: int = 10000
    maximum_grants: int = 100000
    maximum_audit_events: int = 20000

    maximum_arguments_bytes: int = 65536

    default_grant_ttl_seconds: int = 3600

    require_sensitive_approval: bool = True

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        for name in (
            "maximum_tools",
            "maximum_agents",
            "maximum_grants",
            "maximum_audit_events",
            "maximum_arguments_bytes",
            "default_grant_ttl_seconds",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ToolValidationError(
                    f"{name} must be a positive integer"
                )

        if self.allow_hardware_control:
            raise ToolValidationError(
                "Physical robot control cannot be "
                "enabled through this module"
            )


# ============================================================
# TOOL DEFINITION
# ============================================================

@dataclass(frozen=True)
class RegisteredTool:
    name: str
    category: ToolCategory

    allowed_permissions: frozenset[
        ToolPermission
    ]

    allowed_roles: frozenset[
        AgentRole
    ]

    risk_level: ToolRiskLevel = ToolRiskLevel.LOW

    requires_approval: bool = False

    resource_scoped: bool = False

    description: str = ""

    def __post_init__(self) -> None:
        validate_tool_name(self.name)

        if not self.allowed_permissions:
            raise ToolValidationError(
                "Tool requires at least one permission"
            )

        if not self.allowed_roles:
            raise ToolValidationError(
                "Tool requires at least one allowed role"
            )


# ============================================================
# AGENT PRINCIPAL
# ============================================================

@dataclass(frozen=True)
class AgentPrincipal:
    agent_id: str
    role: AgentRole
    tenant_id: str

    active: bool = True

    allowed_robot_ids: frozenset[str] = field(
        default_factory=frozenset
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.agent_id,
            "agent_id",
        )

        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        for robot_id in self.allowed_robot_ids:
            validate_identifier(
                robot_id,
                "robot_id",
            )


# ============================================================
# TOOL GRANT
# ============================================================

@dataclass
class ToolGrant:
    grant_id: str

    agent_id: str
    tenant_id: str

    tool_name: str

    permissions: frozenset[
        ToolPermission
    ]

    resource_ids: frozenset[str] = field(
        default_factory=frozenset
    )

    expires_at: datetime | None = None

    active: bool = True

    granted_by: str = ""

    created_at: str = field(
        default_factory=utc_iso
    )

    def is_valid(self) -> bool:
        if not self.active:
            return False

        if (
            self.expires_at is not None
            and utc_now() >= self.expires_at
        ):
            return False

        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "tool_name": self.tool_name,
            "permissions": sorted(
                item.value
                for item in self.permissions
            ),
            "resource_ids": sorted(
                self.resource_ids
            ),
            "expires_at": (
                self.expires_at.isoformat()
                if self.expires_at
                else None
            ),
            "active": self.active,
            "granted_by": self.granted_by,
            "created_at": self.created_at,
        }


# ============================================================
# TOOL REQUEST
# ============================================================

@dataclass(frozen=True)
class ToolAccessRequest:
    agent_id: str
    tenant_id: str

    tool_name: str
    permission: ToolPermission

    resource_id: str | None = None

    arguments: dict[str, Any] = field(
        default_factory=dict
    )

    request_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.agent_id,
            "agent_id",
        )

        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_tool_name(
            self.tool_name
        )

        if self.resource_id is not None:
            validate_identifier(
                self.resource_id,
                "resource_id",
            )

        if not isinstance(
            self.arguments,
            dict,
        ):
            raise ToolValidationError(
                "Tool arguments must be an object"
            )


# ============================================================
# AUTHORIZATION DECISION
# ============================================================

@dataclass(frozen=True)
class ToolAccessDecision:
    request_id: str

    status: ToolDecisionStatus

    reason: str

    tool_name: str
    agent_id: str
    tenant_id: str

    evaluated_at: str = field(
        default_factory=utc_iso
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "reason": self.reason,
            "tool_name": self.tool_name,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "evaluated_at": self.evaluated_at,
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class ToolPermissionAuditEvent:
    event_id: str
    timestamp: str

    action: str
    agent_id: str
    tenant_id: str
    tool_name: str

    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "tool_name": self.tool_name,
            "details": json_copy(self.details),
        }


# ============================================================
# CENTRAL TOOL PERMISSION SERVICE
# ============================================================

class RoboticsToolPermissionService:

    def __init__(
        self,
        config: ToolPermissionConfig | None = None,
    ) -> None:
        self.config = (
            config or ToolPermissionConfig()
        )

        self.tools: dict[
            str,
            RegisteredTool
        ] = {}

        self.agents: dict[
            str,
            AgentPrincipal
        ] = {}

        self.grants: dict[
            str,
            ToolGrant
        ] = {}

        self.audit_events: list[
            ToolPermissionAuditEvent
        ] = []

        self.argument_validators: dict[
            str,
            Callable[
                [Mapping[str, Any]],
                Mapping[str, Any],
            ]
        ] = {}

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        action: str,
        *,
        agent_id: str,
        tenant_id: str,
        tool_name: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.audit_events.append(
            ToolPermissionAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_iso(),
                action=action,
                agent_id=agent_id,
                tenant_id=tenant_id,
                tool_name=tool_name,
                details=json_copy(
                    dict(details or {})
                ),
            )
        )

        if (
            len(self.audit_events)
            > self.config.maximum_audit_events
        ):
            self.audit_events.pop(0)

    # --------------------------------------------------------
    # REGISTER TOOL
    # --------------------------------------------------------

    def register_tool(
        self,
        tool: RegisteredTool,
    ) -> None:
        with self._lock:
            if tool.name in self.tools:
                raise ToolValidationError(
                    "Tool already registered"
                )

            if (
                len(self.tools)
                >= self.config.maximum_tools
            ):
                raise ToolValidationError(
                    "Tool capacity reached"
                )

            self.tools[tool.name] = tool

    # --------------------------------------------------------
    # REGISTER AGENT
    # --------------------------------------------------------

    def register_agent(
        self,
        principal: AgentPrincipal,
    ) -> None:
        with self._lock:
            if principal.agent_id in self.agents:
                raise ToolValidationError(
                    "Agent already registered"
                )

            if (
                len(self.agents)
                >= self.config.maximum_agents
            ):
                raise ToolValidationError(
                    "Agent capacity reached"
                )

            self.agents[
                principal.agent_id
            ] = principal

    # --------------------------------------------------------
    # REGISTER ARGUMENT VALIDATOR
    # --------------------------------------------------------

    def register_argument_validator(
        self,
        tool_name: str,
        validator: Callable[
            [Mapping[str, Any]],
            Mapping[str, Any],
        ],
    ) -> None:
        validate_tool_name(
            tool_name
        )

        if not callable(validator):
            raise ToolValidationError(
                "Validator must be callable"
            )

        with self._lock:
            if tool_name not in self.tools:
                raise ToolNotFoundError(
                    "Tool not registered"
                )

            self.argument_validators[
                tool_name
            ] = validator

    # --------------------------------------------------------
    # GRANT TOOL PERMISSION
    # --------------------------------------------------------

    def grant_permission(
        self,
        *,
        agent_id: str,
        tool_name: str,
        permissions: frozenset[
            ToolPermission
        ],
        granted_by: str,
        resource_ids: frozenset[str] = frozenset(),
        ttl_seconds: int | None = None,
    ) -> ToolGrant:
        validate_identifier(
            granted_by,
            "granted_by",
        )

        with self._lock:
            agent = self.agents.get(
                agent_id
            )

            tool = self.tools.get(
                tool_name
            )

            if agent is None or not agent.active:
                raise ToolGrantError(
                    "Active agent not found"
                )

            if tool is None:
                raise ToolNotFoundError(
                    "Tool not found"
                )

            if agent.role not in tool.allowed_roles:
                raise ToolGrantError(
                    "Agent role cannot use tool"
                )

            if (
                not permissions
                or not permissions.issubset(
                    tool.allowed_permissions
                )
            ):
                raise ToolGrantError(
                    "Invalid tool permission grant"
                )

            for resource_id in resource_ids:
                validate_identifier(
                    resource_id,
                    "resource_id",
                )

            if (
                tool.resource_scoped
                and not resource_ids
            ):
                raise ToolGrantError(
                    "Resource-scoped tool requires "
                    "explicit resource IDs"
                )

            if (
                len(self.grants)
                >= self.config.maximum_grants
            ):
                raise ToolGrantError(
                    "Grant capacity reached"
                )

            ttl = (
                ttl_seconds
                if ttl_seconds is not None
                else self.config.default_grant_ttl_seconds
            )

            if (
                isinstance(ttl, bool)
                or not isinstance(ttl, int)
                or ttl <= 0
            ):
                raise ToolGrantError(
                    "Invalid grant TTL"
                )

            grant = ToolGrant(
                grant_id=str(uuid.uuid4()),
                agent_id=agent_id,
                tenant_id=agent.tenant_id,
                tool_name=tool_name,
                permissions=frozenset(
                    permissions
                ),
                resource_ids=frozenset(
                    resource_ids
                ),
                expires_at=(
                    utc_now()
                    + timedelta(seconds=ttl)
                ),
                granted_by=granted_by,
            )

            self.grants[
                grant.grant_id
            ] = grant

            self._audit(
                "permission_granted",
                agent_id=agent_id,
                tenant_id=agent.tenant_id,
                tool_name=tool_name,
                details={
                    "grant_id": grant.grant_id,
                    "permissions": sorted(
                        item.value
                        for item in permissions
                    ),
                },
            )

            return grant

    # --------------------------------------------------------
    # REVOKE GRANT
    # --------------------------------------------------------

    def revoke_grant(
        self,
        grant_id: str,
        *,
        revoked_by: str,
    ) -> None:
        validate_identifier(
            revoked_by,
            "revoked_by",
        )

        with self._lock:
            grant = self.grants.get(
                grant_id
            )

            if grant is None:
                raise ToolGrantError(
                    "Grant not found"
                )

            grant.active = False

            self._audit(
                "permission_revoked",
                agent_id=grant.agent_id,
                tenant_id=grant.tenant_id,
                tool_name=grant.tool_name,
                details={
                    "grant_id": grant_id,
                    "revoked_by": revoked_by,
                },
            )

    # --------------------------------------------------------
    # EVALUATE ACCESS
    # --------------------------------------------------------

    def evaluate(
        self,
        request: ToolAccessRequest,
    ) -> ToolAccessDecision:
        with self._lock:
            agent = self.agents.get(
                request.agent_id
            )

            tool = self.tools.get(
                request.tool_name
            )

            def decision(
                status: ToolDecisionStatus,
                reason: str,
            ) -> ToolAccessDecision:
                result = ToolAccessDecision(
                    request_id=request.request_id,
                    status=status,
                    reason=reason,
                    tool_name=request.tool_name,
                    agent_id=request.agent_id,
                    tenant_id=request.tenant_id,
                )

                self._audit(
                    "access_evaluated",
                    agent_id=request.agent_id,
                    tenant_id=request.tenant_id,
                    tool_name=request.tool_name,
                    details={
                        "status": status.value,
                        "reason": reason,
                    },
                )

                return result

            if agent is None or not agent.active:
                return decision(
                    ToolDecisionStatus.DENIED,
                    "Agent is not registered or active",
                )

            if agent.tenant_id != request.tenant_id:
                return decision(
                    ToolDecisionStatus.DENIED,
                    "Cross-tenant access denied",
                )

            if tool is None:
                return decision(
                    ToolDecisionStatus.DENIED,
                    "Tool is not registered",
                )

            if agent.role not in tool.allowed_roles:
                return decision(
                    ToolDecisionStatus.DENIED,
                    "Agent role is not permitted",
                )

            if (
                request.permission
                not in tool.allowed_permissions
            ):
                return decision(
                    ToolDecisionStatus.DENIED,
                    "Operation is not permitted",
                )

            if (
                tool.resource_scoped
                and request.resource_id is None
            ):
                return decision(
                    ToolDecisionStatus.DENIED,
                    "Resource ID is required",
                )

            try:
                arguments = json_copy(
                    request.arguments
                )

                encoded = json.dumps(
                    arguments,
                    allow_nan=False,
                ).encode("utf-8")

                if (
                    len(encoded)
                    > self.config.maximum_arguments_bytes
                ):
                    raise ToolValidationError(
                        "Tool arguments exceed size limit"
                    )

                validator = self.argument_validators.get(
                    request.tool_name
                )

                if validator is None:
                    return decision(
                        ToolDecisionStatus.DENIED,
                        "No argument validator registered",
                    )

                validator(arguments)

            except (
                ToolValidationError,
                TypeError,
                ValueError,
            ):
                return decision(
                    ToolDecisionStatus.DENIED,
                    "Tool arguments failed validation",
                )

            eligible_grants = [
                grant
                for grant in self.grants.values()
                if (
                    grant.agent_id == request.agent_id
                    and grant.tenant_id == request.tenant_id
                    and grant.tool_name == request.tool_name
                    and request.permission in grant.permissions
                    and grant.is_valid()
                )
            ]

            if not eligible_grants:
                return decision(
                    ToolDecisionStatus.DENIED,
                    "No active permission grant",
                )

            if tool.resource_scoped:
                eligible_grants = [
                    grant
                    for grant in eligible_grants
                    if request.resource_id
                    in grant.resource_ids
                ]

                if not eligible_grants:
                    return decision(
                        ToolDecisionStatus.DENIED,
                        "Resource not included in grant",
                    )

                if (
                    tool.category
                    in {
                        ToolCategory.ROBOT,
                        ToolCategory.NAVIGATION,
                        ToolCategory.PERCEPTION,
                        ToolCategory.MANIPULATION,
                        ToolCategory.TELEMETRY,
                    }
                    and request.resource_id
                    not in agent.allowed_robot_ids
                ):
                    return decision(
                        ToolDecisionStatus.DENIED,
                        "Robot is outside agent scope",
                    )

            if (
                tool.requires_approval
                or (
                    self.config.require_sensitive_approval
                    and tool.risk_level
                    in {
                        ToolRiskLevel.HIGH,
                        ToolRiskLevel.CRITICAL,
                    }
                )
            ):
                return decision(
                    ToolDecisionStatus.APPROVAL_REQUIRED,
                    "Independent human approval required",
                )

            return decision(
                ToolDecisionStatus.ALLOWED,
                "Tool access permitted",
            )

    # --------------------------------------------------------
    # REQUIRE ALLOWED
    # --------------------------------------------------------

    def require_allowed(
        self,
        request: ToolAccessRequest,
    ) -> ToolAccessDecision:
        result = self.evaluate(
            request
        )

        if result.status != ToolDecisionStatus.ALLOWED:
            raise ToolAccessDenied(
                f"Tool access not allowed: "
                f"{result.status.value} - {result.reason}"
            )

        return result

    # --------------------------------------------------------
    # LIST AGENT GRANTS
    # --------------------------------------------------------

    def list_agent_grants(
        self,
        agent_id: str,
    ) -> list[ToolGrant]:
        with self._lock:
            return [
                grant
                for grant in self.grants.values()
                if (
                    grant.agent_id == agent_id
                    and grant.is_valid()
                )
            ]

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "registered_tools": len(
                    self.tools
                ),
                "registered_agents": len(
                    self.agents
                ),
                "total_grants": len(
                    self.grants
                ),
                "active_grants": sum(
                    grant.is_valid()
                    for grant in self.grants.values()
                ),
                "audit_events": len(
                    self.audit_events
                ),
                "hardware_control_enabled": False,
            }

    # --------------------------------------------------------
    # EXPORT AUDIT
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_iso(),
                "statistics": self.statistics(),
                "grants": [
                    grant.to_dict()
                    for grant in self.grants.values()
                ],
                "audit_events": [
                    event.to_dict()
                    for event in self.audit_events
                ],
            }

        output.write_text(
            json.dumps(
                payload,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output


# ============================================================
# DEFAULT TOOL DEFINITIONS
# ============================================================

def register_default_tools(
    service: RoboticsToolPermissionService,
) -> None:
    definitions = [
        RegisteredTool(
            name="robot.get_status",
            category=ToolCategory.ROBOT,
            allowed_permissions=frozenset({
                ToolPermission.READ,
            }),
            allowed_roles=frozenset({
                AgentRole.SUPERVISOR,
                AgentRole.PLANNER,
                AgentRole.NAVIGATION,
                AgentRole.MAINTENANCE,
                AgentRole.REPAIR,
                AgentRole.VIEWER,
            }),
            resource_scoped=True,
        ),

        RegisteredTool(
            name="navigation.plan",
            category=ToolCategory.NAVIGATION,
            allowed_permissions=frozenset({
                ToolPermission.EXECUTE,
            }),
            allowed_roles=frozenset({
                AgentRole.SUPERVISOR,
                AgentRole.PLANNER,
                AgentRole.NAVIGATION,
            }),
            resource_scoped=True,
            risk_level=ToolRiskLevel.MEDIUM,
            description="Plan simulated robot navigation",
        ),

        RegisteredTool(
            name="simulation.create",
            category=ToolCategory.SIMULATION,
            allowed_permissions=frozenset({
                ToolPermission.EXECUTE,
            }),
            allowed_roles=frozenset({
                AgentRole.SUPERVISOR,
                AgentRole.SIMULATION,
                AgentRole.EVALUATION,
            }),
            resource_scoped=False,
            risk_level=ToolRiskLevel.LOW,
        ),

        RegisteredTool(
            name="model.deploy",
            category=ToolCategory.MODEL,
            allowed_permissions=frozenset({
                ToolPermission.EXECUTE,
            }),
            allowed_roles=frozenset({
                AgentRole.SUPERVISOR,
                AgentRole.EVALUATION,
            }),
            resource_scoped=True,
            risk_level=ToolRiskLevel.CRITICAL,
            requires_approval=True,
        ),

        RegisteredTool(
            name="repair.create_work_order",
            category=ToolCategory.REPAIR,
            allowed_permissions=frozenset({
                ToolPermission.WRITE,
            }),
            allowed_roles=frozenset({
                AgentRole.SUPERVISOR,
                AgentRole.MAINTENANCE,
                AgentRole.REPAIR,
            }),
            resource_scoped=True,
            risk_level=ToolRiskLevel.MEDIUM,
        ),
    ]

    for tool in definitions:
        service.register_tool(
            tool
        )


# ============================================================
# INPUT VALIDATION INTEGRATION
# ============================================================

def attach_input_validators(
    service: RoboticsToolPermissionService,
) -> None:
    from backend.app.security.input_validation import (
        RoboticsInputValidationService,
    )

    validator = RoboticsInputValidationService()

    for tool_name in (
        "robot.get_status",
        "navigation.plan",
        "simulation.create",
    ):
        service.register_argument_validator(
            tool_name,
            lambda arguments, name=tool_name: (
                validator.validate_tool_arguments(
                    name,
                    arguments,
                )
            ),
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsToolPermissionService()

    register_default_tools(
        service
    )

    attach_input_validators(
        service
    )

    service.register_agent(
        AgentPrincipal(
            agent_id="navigation_agent_001",
            role=AgentRole.NAVIGATION,
            tenant_id="tenant_alpha",
            allowed_robot_ids=frozenset({
                "robot_001",
            }),
        )
    )

    service.grant_permission(
        agent_id="navigation_agent_001",
        tool_name="navigation.plan",
        permissions=frozenset({
            ToolPermission.EXECUTE,
        }),
        resource_ids=frozenset({
            "robot_001",
        }),
        granted_by="security_admin_001",
        ttl_seconds=3600,
    )

    allowed_request = ToolAccessRequest(
        agent_id="navigation_agent_001",
        tenant_id="tenant_alpha",
        tool_name="navigation.plan",
        permission=ToolPermission.EXECUTE,
        resource_id="robot_001",
        arguments={
            "robot_id": "robot_001",
            "goal_x": 10.0,
            "goal_y": 5.0,
        },
    )

    denied_request = ToolAccessRequest(
        agent_id="navigation_agent_001",
        tenant_id="tenant_beta",
        tool_name="navigation.plan",
        permission=ToolPermission.EXECUTE,
        resource_id="robot_001",
        arguments={
            "robot_id": "robot_001",
            "goal_x": 10.0,
            "goal_y": 5.0,
        },
    )

    allowed = service.evaluate(
        allowed_request
    )

    denied = service.evaluate(
        denied_request
    )

    report = service.export_json(
        "data/security/tool_permissions_report.json"
    )

    print(
        json.dumps(
            {
                "allowed_request": allowed.to_dict(),
                "denied_request": denied.to_dict(),
                "statistics": service.statistics(),
                "report": str(report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Agents module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: backend/app/api/endpoints/agents.py

Secure Agentic AI REST API.

Features:
- List available AI agents
- Retrieve agent information
- Create agent execution requests
- Execute authorized AI workflows
- Retrieve execution status and results
- Retrieve execution history
- Cancel supported agent executions
- Agent health and performance metrics
- Multi-agent workflow visibility
- Tenant isolation and RBAC
- Request validation and structured responses
- Audit logging through injectable audit service

Safety:
- No direct robot actuation
- No autonomous physical command execution
- No emergency-stop override
- No safety interlock bypass
- AI recommendations require independent authorization
  before any physical execution

Python 3.11+
FastAPI
Pydantic v2
"""

from __future__ import annotations

import logging
import math
import re
import secrets

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Annotated, Mapping, Protocol

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.api.dependencies import (
    CurrentTenant,
)

from app.services.rbac import (
    RBACPrincipal,
    get_rbac_service,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agents",
    tags=["AI Agents"],
)

MAX_AGENT_NAME_LENGTH = 100
MAX_PROMPT_LENGTH = 12_000
MAX_CONTEXT_ITEMS = 50
MAX_CONTEXT_BYTES = 32_768
MAX_TOOL_NAMES = 30
MAX_WORKFLOW_STEPS = 50
MAX_PAGE_SIZE = 100

AGENT_READ_PERMISSION = "agent:read"
AGENT_EXECUTE_PERMISSION = "agent:execute"
AGENT_CANCEL_PERMISSION = "agent:cancel"
AGENT_METRICS_PERMISSION = "agent:metrics"

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)


# ============================================================
# Exceptions
# ============================================================


class AgentAPIError(Exception):
    pass


class AgentValidationError(AgentAPIError):
    pass


class AgentAuthorizationError(AgentAPIError):
    pass


class AgentNotFoundError(AgentAPIError):
    pass


class AgentExecutionNotFoundError(AgentAPIError):
    pass


class AgentExecutionConflictError(AgentAPIError):
    pass


class AgentServiceUnavailableError(AgentAPIError):
    pass


# ============================================================
# Enums
# ============================================================


class AgentType(str, Enum):
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    RESEARCH = "research"
    ROBOT_HEALTH = "robot_health"
    DIAGNOSTICS = "diagnostics"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    REPAIR = "repair"
    PERCEPTION = "perception"
    NAVIGATION_ANALYSIS = "navigation_analysis"
    MANIPULATION_ANALYSIS = "manipulation_analysis"
    WORLD_MODEL = "world_model"
    SIMULATION = "simulation"
    SAFETY_ANALYSIS = "safety_analysis"
    FLEET = "fleet"
    DATA_ANALYSIS = "data_analysis"
    EVALUATION = "evaluation"
    DOCUMENT = "document"
    RAG = "rag"
    REPORT = "report"
    VERIFICATION = "verification"


class AgentExecutionStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class AgentHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class AgentWorkflowStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentToolAccess(str, Enum):
    READ_ONLY = "read_only"
    ANALYSIS = "analysis"
    SIMULATION = "simulation"


TERMINAL_EXECUTION_STATUSES = frozenset({
    AgentExecutionStatus.COMPLETED,
    AgentExecutionStatus.FAILED,
    AgentExecutionStatus.CANCELLED,
    AgentExecutionStatus.TIMED_OUT,
})


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_identifier(
    value: str,
    name: str = "identifier",
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Invalid {name}")

    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid {name}")

    return value


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def validate_context(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    import json

    if len(context) > MAX_CONTEXT_ITEMS:
        raise ValueError(
            "Too many context items"
        )

    try:
        encoded = json.dumps(
            dict(context),
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Context must contain JSON-compatible values"
        ) from exc

    if len(encoded) > MAX_CONTEXT_BYTES:
        raise ValueError(
            "Context exceeds maximum size"
        )

    return dict(context)


def raise_service_error(
    exc: Exception,
) -> None:
    if isinstance(exc, AgentNotFoundError):
        raise HTTPException(
            status_code=404,
            detail="Agent not found",
        ) from exc

    if isinstance(exc, AgentExecutionNotFoundError):
        raise HTTPException(
            status_code=404,
            detail="Agent execution not found",
        ) from exc

    if isinstance(exc, AgentExecutionConflictError):
        raise HTTPException(
            status_code=409,
            detail="Agent execution conflict",
        ) from exc

    if isinstance(exc, AgentValidationError):
        raise HTTPException(
            status_code=422,
            detail=str(exc)[:300],
        ) from exc

    if isinstance(exc, AgentAuthorizationError):
        raise HTTPException(
            status_code=403,
            detail="Agent access denied",
        ) from exc

    logger.error(
        "Agent service operation failed: %s",
        type(exc).__name__,
    )

    raise HTTPException(
        status_code=503,
        detail="Agent service unavailable",
    ) from exc


# ============================================================
# Base Models
# ============================================================


class AgentBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        from_attributes=True,
    )


class AgentCapability(AgentBaseModel):
    name: str = Field(
        min_length=1,
        max_length=100,
    )

    description: str = Field(
        min_length=1,
        max_length=500,
    )

    tool_access: AgentToolAccess = (
        AgentToolAccess.READ_ONLY
    )


class AgentInformation(AgentBaseModel):
    agent_id: str

    name: str = Field(
        min_length=1,
        max_length=MAX_AGENT_NAME_LENGTH,
    )

    agent_type: AgentType

    description: str = Field(
        min_length=1,
        max_length=1000,
    )

    version: str = Field(
        min_length=1,
        max_length=50,
    )

    enabled: bool = True

    health_status: AgentHealthStatus = (
        AgentHealthStatus.UNKNOWN
    )

    capabilities: list[AgentCapability] = Field(
        default_factory=list,
        max_length=50,
    )

    allowed_tools: list[str] = Field(
        default_factory=list,
        max_length=MAX_TOOL_NAMES,
    )

    supports_multi_agent: bool = False
    supports_streaming: bool = False

    physical_control_enabled: bool = False

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(
        cls,
        value: str,
    ) -> str:
        return validate_identifier(
            value,
            "agent_id",
        )

    @model_validator(mode="after")
    def enforce_read_only_control(
        self,
    ) -> "AgentInformation":
        if self.physical_control_enabled:
            raise ValueError(
                "Direct physical control is not supported"
            )

        return self


class AgentExecutionRequest(AgentBaseModel):
    agent_id: str

    prompt: str = Field(
        min_length=1,
        max_length=MAX_PROMPT_LENGTH,
    )

    robot_id: str | None = None
    fleet_id: str | None = None
    simulation_id: str | None = None

    context: dict[str, Any] = Field(
        default_factory=dict,
    )

    allowed_tools: list[str] = Field(
        default_factory=list,
        max_length=MAX_TOOL_NAMES,
    )

    max_steps: int = Field(
        default=10,
        ge=1,
        le=MAX_WORKFLOW_STEPS,
    )

    timeout_seconds: int = Field(
        default=120,
        ge=5,
        le=600,
    )

    require_verification: bool = True

    allow_physical_execution: bool = False

    @field_validator(
        "agent_id",
        "robot_id",
        "fleet_id",
        "simulation_id",
    )
    @classmethod
    def validate_ids(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        return validate_identifier(value)

    @field_validator("context")
    @classmethod
    def validate_context_data(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        return validate_context(value)

    @field_validator("allowed_tools")
    @classmethod
    def validate_tools(
        cls,
        values: list[str],
    ) -> list[str]:
        for value in values:
            validate_identifier(
                value,
                "tool_name",
            )

        if len(set(values)) != len(values):
            raise ValueError(
                "Duplicate tool names"
            )

        return values

    @model_validator(mode="after")
    def enforce_safety(
        self,
    ) -> "AgentExecutionRequest":
        if self.allow_physical_execution:
            raise ValueError(
                "Physical execution is not allowed "
                "through the Agent API"
            )

        return self


class AgentWorkflowStep(AgentBaseModel):
    step_id: str

    agent_id: str

    description: str = Field(
        min_length=1,
        max_length=1000,
    )

    status: AgentWorkflowStepStatus

    started_at: datetime | None = None
    completed_at: datetime | None = None

    duration_ms: float | None = Field(
        default=None,
        ge=0,
    )

    error_message: str | None = Field(
        default=None,
        max_length=1000,
    )

    @field_validator(
        "started_at",
        "completed_at",
    )
    @classmethod
    def validate_timestamps(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return ensure_utc(value) if value else None


class AgentCitation(AgentBaseModel):
    source_id: str
    title: str

    document_id: str | None = None
    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    relevance_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


class AgentExecutionResult(AgentBaseModel):
    execution_id: str

    tenant_id: str
    user_id: str
    agent_id: str

    status: AgentExecutionStatus

    answer: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    citations: list[AgentCitation] = Field(
        default_factory=list,
        max_length=100,
    )

    workflow_steps: list[AgentWorkflowStep] = Field(
        default_factory=list,
        max_length=MAX_WORKFLOW_STEPS,
    )

    risk_level: AgentRiskLevel = (
        AgentRiskLevel.LOW
    )

    requires_human_review: bool = False

    physical_execution_authorized: bool = False

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    started_at: datetime | None = None
    completed_at: datetime | None = None

    error_message: str | None = Field(
        default=None,
        max_length=1000,
    )

    @field_validator(
        "execution_id",
        "tenant_id",
        "user_id",
        "agent_id",
    )
    @classmethod
    def validate_ids(
        cls,
        value: str,
    ) -> str:
        return validate_identifier(value)

    @field_validator(
        "created_at",
        "started_at",
        "completed_at",
    )
    @classmethod
    def validate_timestamps(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return ensure_utc(value) if value else None

    @model_validator(mode="after")
    def enforce_safety(
        self,
    ) -> "AgentExecutionResult":
        if self.physical_execution_authorized:
            raise ValueError(
                "Agent results cannot authorize "
                "physical execution"
            )

        return self


class AgentExecutionSummary(AgentBaseModel):
    execution_id: str
    agent_id: str

    status: AgentExecutionStatus

    created_at: datetime
    completed_at: datetime | None = None

    duration_ms: float | None = Field(
        default=None,
        ge=0,
    )


class AgentMetrics(AgentBaseModel):
    agent_id: str

    total_executions: int = Field(
        default=0,
        ge=0,
    )

    successful_executions: int = Field(
        default=0,
        ge=0,
    )

    failed_executions: int = Field(
        default=0,
        ge=0,
    )

    cancelled_executions: int = Field(
        default=0,
        ge=0,
    )

    average_latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    average_tool_calls: float | None = Field(
        default=None,
        ge=0,
    )

    average_confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    evaluated_success_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    last_execution_at: datetime | None = None


class AgentHealthResponse(AgentBaseModel):
    agent_id: str
    status: AgentHealthStatus
    available: bool
    message: str | None = None
    checked_at: datetime = Field(
        default_factory=utc_now,
    )


class AgentListResponse(AgentBaseModel):
    items: list[AgentInformation]
    total: int
    limit: int
    offset: int


class AgentExecutionListResponse(AgentBaseModel):
    items: list[AgentExecutionSummary]
    total: int
    limit: int
    offset: int


class AgentOperationResponse(AgentBaseModel):
    success: bool
    execution_id: str
    status: AgentExecutionStatus
    message: str


# ============================================================
# Service Interfaces
# ============================================================


class AgentApplicationService(Protocol):
    async def list_agents(
        self,
        *,
        tenant_id: str,
        agent_type: AgentType | None,
        enabled_only: bool,
        limit: int,
        offset: int,
    ) -> tuple[list[AgentInformation], int]:
        ...

    async def get_agent(
        self,
        *,
        tenant_id: str,
        agent_id: str,
    ) -> AgentInformation | None:
        ...

    async def create_execution(
        self,
        *,
        tenant_id: str,
        user_id: str,
        request: AgentExecutionRequest,
    ) -> AgentExecutionResult:
        ...

    async def get_execution(
        self,
        *,
        tenant_id: str,
        user_id: str,
        execution_id: str,
    ) -> AgentExecutionResult | None:
        ...

    async def list_executions(
        self,
        *,
        tenant_id: str,
        user_id: str,
        agent_id: str | None,
        execution_status: AgentExecutionStatus | None,
        limit: int,
        offset: int,
    ) -> tuple[list[AgentExecutionSummary], int]:
        ...

    async def cancel_execution(
        self,
        *,
        tenant_id: str,
        user_id: str,
        execution_id: str,
    ) -> AgentExecutionResult:
        ...

    async def get_metrics(
        self,
        *,
        tenant_id: str,
        agent_id: str,
    ) -> AgentMetrics:
        ...

    async def get_health(
        self,
        *,
        tenant_id: str,
        agent_id: str,
    ) -> AgentHealthResponse:
        ...


class AgentResourceAuthorizer(Protocol):
    async def require_access(
        self,
        *,
        tenant_id: str,
        user_id: str,
        robot_id: str | None,
        fleet_id: str | None,
        simulation_id: str | None,
    ) -> None:
        """
        Verify referenced resources exist and belong to
        the tenant, and that the user may read them.
        """
        ...


class AgentAuditSink(Protocol):
    async def record(
        self,
        *,
        tenant_id: str,
        user_id: str,
        action: str,
        resource_id: str,
        metadata: Mapping[str, Any],
    ) -> None:
        ...


# ============================================================
# Service Configuration
# ============================================================


_agent_service: AgentApplicationService | None = None
_resource_authorizer: AgentResourceAuthorizer | None = None
_agent_audit_sink: AgentAuditSink | None = None


def configure_agent_api(
    *,
    service: AgentApplicationService,
    resource_authorizer: AgentResourceAuthorizer,
    audit_sink: AgentAuditSink | None = None,
) -> None:
    global _agent_service
    global _resource_authorizer
    global _agent_audit_sink

    _agent_service = service
    _resource_authorizer = resource_authorizer
    _agent_audit_sink = audit_sink


def get_agent_service() -> AgentApplicationService:
    if _agent_service is None:
        raise HTTPException(
            status_code=503,
            detail="Agent service not configured",
        )

    return _agent_service


def get_agent_resource_authorizer() -> AgentResourceAuthorizer:
    if _resource_authorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Agent resource authorization unavailable",
        )

    return _resource_authorizer


# ============================================================
# RBAC
# ============================================================


async def require_agent_permission(
    tenant: CurrentTenant,
    permission: str,
) -> None:
    try:
        decision = await get_rbac_service().check_permission(
            RBACPrincipal(
                tenant_id=tenant.tenant_id,
                user_id=tenant.user_id,
                authenticated=True,
                is_active=True,
            ),
            permission,
        )

    except Exception as exc:
        logger.error(
            "Agent permission check unavailable: %s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail="Authorization service unavailable",
        ) from exc

    if not decision.allowed:
        raise HTTPException(
            status_code=403,
            detail="Agent permission denied",
        )


# ============================================================
# Audit
# ============================================================


async def audit_agent_action(
    *,
    tenant_id: str,
    user_id: str,
    action: str,
    resource_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    if _agent_audit_sink is None:
        logger.warning(
            "Agent audit sink not configured: %s",
            action,
        )
        return

    try:
        await _agent_audit_sink.record(
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource_id=resource_id,
            metadata=dict(metadata or {}),
        )

    except Exception:
        logger.exception(
            "Agent audit recording failed"
        )


# ============================================================
# List Agents
# ============================================================


@router.get(
    "",
    response_model=AgentListResponse,
)
async def list_agents(
    tenant: CurrentTenant,
    agent_type: AgentType | None = Query(
        default=None,
    ),
    enabled_only: bool = Query(
        default=True,
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=MAX_PAGE_SIZE,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    service: AgentApplicationService = Depends(
        get_agent_service
    ),
) -> AgentListResponse:
    await require_agent_permission(
        tenant,
        AGENT_READ_PERMISSION,
    )

    try:
        items, total = await service.list_agents(
            tenant_id=tenant.tenant_id,
            agent_type=agent_type,
            enabled_only=enabled_only,
            limit=limit,
            offset=offset,
        )

        return AgentListResponse(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )

    except Exception as exc:
        raise_service_error(exc)


# ============================================================
# List Execution History
# ============================================================


@router.get(
    "/executions",
    response_model=AgentExecutionListResponse,
)
async def list_agent_executions(
    tenant: CurrentTenant,
    agent_id: str | None = Query(
        default=None,
    ),
    execution_status: AgentExecutionStatus | None = Query(
        default=None,
        alias="status",
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=MAX_PAGE_SIZE,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    service: AgentApplicationService = Depends(
        get_agent_service
    ),
) -> AgentExecutionListResponse:
    await require_agent_permission(
        tenant,
        AGENT_READ_PERMISSION,
    )

    if agent_id is not None:
        validate_identifier(
            agent_id,
            "agent_id",
        )

    try:
        items, total = await service.list_executions(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            agent_id=agent_id,
            execution_status=execution_status,
            limit=limit,
            offset=offset,
        )

        return AgentExecutionListResponse(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )

    except Exception as exc:
        raise_service_error(exc)


# ============================================================
# Get Execution
# ============================================================


@router.get(
    "/executions/{execution_id}",
    response_model=AgentExecutionResult,
)
async def get_agent_execution(
    execution_id: str,
    tenant: CurrentTenant,
    service: AgentApplicationService = Depends(
        get_agent_service
    ),
) -> AgentExecutionResult:
    await require_agent_permission(
        tenant,
        AGENT_READ_PERMISSION,
    )

    validate_identifier(
        execution_id,
        "execution_id",
    )

    try:
        result = await service.get_execution(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            execution_id=execution_id,
        )

        if result is None:
            raise AgentExecutionNotFoundError()

        if (
            result.tenant_id != tenant.tenant_id
            or result.user_id != tenant.user_id
        ):
            raise AgentAuthorizationError()

        return result

    except Exception as exc:
        raise_service_error(exc)


# ============================================================
# Cancel Execution
# ============================================================


@router.post(
    "/executions/{execution_id}/cancel",
    response_model=AgentOperationResponse,
)
async def cancel_agent_execution(
    execution_id: str,
    tenant: CurrentTenant,
    service: AgentApplicationService = Depends(
        get_agent_service
    ),
) -> AgentOperationResponse:
    await require_agent_permission(
        tenant,
        AGENT_CANCEL_PERMISSION,
    )

    validate_identifier(
        execution_id,
        "execution_id",
    )

    try:
        result = await service.cancel_execution(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            execution_id=execution_id,
        )

        if (
            result.tenant_id != tenant.tenant_id
            or result.user_id != tenant.user_id
        ):
            raise AgentAuthorizationError()

        await audit_agent_action(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            action="agent.execution.cancel",
            resource_id=execution_id,
        )

        return AgentOperationResponse(
            success=True,
            execution_id=result.execution_id,
            status=result.status,
            message="Cancellation request processed",
        )

    except Exception as exc:
        raise_service_error(exc)


# ============================================================
# Get Agent
# ============================================================


@router.get(
    "/{agent_id}",
    response_model=AgentInformation,
)
async def get_agent(
    agent_id: str,
    tenant: CurrentTenant,
    service: AgentApplicationService = Depends(
        get_agent_service
    ),
) -> AgentInformation:
    await require_agent_permission(
        tenant,
        AGENT_READ_PERMISSION,
    )

    validate_identifier(
        agent_id,
        "agent_id",
    )

    try:
        agent = await service.get_agent(
            tenant_id=tenant.tenant_id,
            agent_id=agent_id,
        )

        if agent is None:
            raise AgentNotFoundError()

        return agent

    except Exception as exc:
        raise_service_error(exc)


# ============================================================
# Execute Agent
# ============================================================


@router.post(
    "/execute",
    response_model=AgentExecutionResult,
    status_code=status.HTTP_202_ACCEPTED,
)
async def execute_agent(
    payload: AgentExecutionRequest,
    tenant: CurrentTenant,
    service: AgentApplicationService = Depends(
        get_agent_service
    ),
    resource_authorizer: AgentResourceAuthorizer = Depends(
        get_agent_resource_authorizer
    ),
) -> AgentExecutionResult:
    await require_agent_permission(
        tenant,
        AGENT_EXECUTE_PERMISSION,
    )

    try:
        agent = await service.get_agent(
            tenant_id=tenant.tenant_id,
            agent_id=payload.agent_id,
        )

        if agent is None:
            raise AgentNotFoundError()

        if not agent.enabled:
            raise AgentExecutionConflictError(
                "Agent is disabled"
            )

        if agent.physical_control_enabled:
            raise AgentAuthorizationError(
                "Physical control is prohibited"
            )

        await resource_authorizer.require_access(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            robot_id=payload.robot_id,
            fleet_id=payload.fleet_id,
            simulation_id=payload.simulation_id,
        )

        allowed_tools = set(agent.allowed_tools)

        if not set(payload.allowed_tools).issubset(
            allowed_tools
        ):
            raise AgentAuthorizationError(
                "Requested tools are not permitted"
            )

        result = await service.create_execution(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            request=payload,
        )

        if (
            result.tenant_id != tenant.tenant_id
            or result.user_id != tenant.user_id
            or result.agent_id != payload.agent_id
        ):
            raise AgentAuthorizationError(
                "Execution ownership mismatch"
            )

        await audit_agent_action(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            action="agent.execution.create",
            resource_id=result.execution_id,
            metadata={
                "agent_id": payload.agent_id,
                "robot_id": payload.robot_id,
                "fleet_id": payload.fleet_id,
                "simulation_id": payload.simulation_id,
            },
        )

        return result

    except Exception as exc:
        raise_service_error(exc)


# ============================================================
# Agent Metrics
# ============================================================


@router.get(
    "/{agent_id}/metrics",
    response_model=AgentMetrics,
)
async def get_agent_metrics(
    agent_id: str,
    tenant: CurrentTenant,
    service: AgentApplicationService = Depends(
        get_agent_service
    ),
) -> AgentMetrics:
    await require_agent_permission(
        tenant,
        AGENT_METRICS_PERMISSION,
    )

    validate_identifier(
        agent_id,
        "agent_id",
    )

    try:
        return await service.get_metrics(
            tenant_id=tenant.tenant_id,
            agent_id=agent_id,
        )

    except Exception as exc:
        raise_service_error(exc)


# ============================================================
# Agent Health
# ============================================================


@router.get(
    "/{agent_id}/health",
    response_model=AgentHealthResponse,
)
async def get_agent_health(
    agent_id: str,
    tenant: CurrentTenant,
    service: AgentApplicationService = Depends(
        get_agent_service
    ),
) -> AgentHealthResponse:
    await require_agent_permission(
        tenant,
        AGENT_READ_PERMISSION,
    )

    validate_identifier(
        agent_id,
        "agent_id",
    )

    try:
        return await service.get_health(
            tenant_id=tenant.tenant_id,
            agent_id=agent_id,
        )

    except Exception as exc:
        raise_service_error(exc)


# ============================================================
# Exports
# ============================================================


__all__ = [
    "router",
    "AgentAPIError",
    "AgentValidationError",
    "AgentAuthorizationError",
    "AgentNotFoundError",
    "AgentExecutionNotFoundError",
    "AgentExecutionConflictError",
    "AgentServiceUnavailableError",
    "AgentType",
    "AgentExecutionStatus",
    "AgentHealthStatus",
    "AgentWorkflowStepStatus",
    "AgentRiskLevel",
    "AgentToolAccess",
    "AgentCapability",
    "AgentInformation",
    "AgentExecutionRequest",
    "AgentWorkflowStep",
    "AgentCitation",
    "AgentExecutionResult",
    "AgentExecutionSummary",
    "AgentMetrics",
    "AgentHealthResponse",
    "AgentListResponse",
    "AgentExecutionListResponse",
    "AgentOperationResponse",
    "AgentApplicationService",
    "AgentResourceAuthorizer",
    "AgentAuditSink",
    "configure_agent_api",
    "get_agent_service",
    "get_agent_resource_authorizer",
    "require_agent_permission",
    "audit_agent_action",
    "list_agents",
    "list_agent_executions",
    "get_agent_execution",
    "cancel_agent_execution",
    "get_agent",
    "execute_agent",
    "get_agent_metrics",
    "get_agent_health",
]
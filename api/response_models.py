"""
Physical AI & Autonomous Robotics Intelligence Platform
API Response Models

Features:
- Standardized FastAPI responses
- Pydantic v2 models
- Success and error responses
- Pagination
- Authentication and session responses
- Robot and fleet responses
- Telemetry and sensor responses
- Maintenance and repair responses
- AI inference and RAG responses
- Evaluation and simulation responses
- Tenant-aware metadata
- UTC timestamps
- Safe serialization

Python 3.11+
FastAPI
Pydantic >= 2.0
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# Constants
# ============================================================

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 500

T = TypeVar("T")


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError(
            "Datetime must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# Base Model
# ============================================================


class APIBaseModel(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        populate_by_name=True,
    )


# ============================================================
# Enums
# ============================================================


class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PENDING = "pending"


class OperationStatus(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    COMPLETED = "completed"
    PROCESSING = "processing"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RobotOperationalStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    IDLE = "idle"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    EMERGENCY_STOP = "emergency_stop"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AIExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NEEDS_REVIEW = "needs_review"


# ============================================================
# Common Metadata
# ============================================================


class ResponseMetadata(APIBaseModel):
    request_id: str | None = None
    correlation_id: str | None = None

    tenant_id: str | None = None

    timestamp: datetime = Field(
        default_factory=utc_now
    )

    processing_time_ms: float | None = Field(
        default=None,
        ge=0,
    )

    api_version: str = "v1"

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        return ensure_utc(value)


# ============================================================
# Standard Responses
# ============================================================


class SuccessResponse(
    APIBaseModel,
    Generic[T],
):
    success: bool = True
    status: ResponseStatus = ResponseStatus.SUCCESS

    message: str = "Operation completed successfully"

    data: T

    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata
    )


class MessageResponse(APIBaseModel):
    success: bool = True
    message: str

    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata
    )


class OperationResponse(APIBaseModel):
    success: bool = True

    operation_id: str | None = None
    operation_status: OperationStatus

    message: str

    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata
    )


# ============================================================
# Error Responses
# ============================================================


class ErrorDetail(APIBaseModel):
    code: str
    message: str
    status_code: int = Field(
        ge=400,
        le=599,
    )

    request_id: str | None = None

    timestamp: datetime = Field(
        default_factory=utc_now
    )

    details: dict[str, Any] | None = None

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        return ensure_utc(value)


class ErrorResponse(APIBaseModel):
    success: bool = False
    error: ErrorDetail

    @model_validator(mode="after")
    def enforce_failure(self) -> "ErrorResponse":
        if self.success:
            raise ValueError(
                "Error response cannot indicate success"
            )

        return self


class ValidationFieldError(APIBaseModel):
    location: list[str | int]
    type: str
    message: str


class ValidationErrorResponse(APIBaseModel):
    success: bool = False
    error: ErrorDetail


# ============================================================
# Pagination
# ============================================================


class PaginationParams(APIBaseModel):
    page: int = Field(
        default=DEFAULT_PAGE,
        ge=1,
    )

    page_size: int = Field(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
    )


class PaginationMetadata(APIBaseModel):
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)

    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=0)

    has_next: bool
    has_previous: bool


class PaginatedResponse(
    APIBaseModel,
    Generic[T],
):
    success: bool = True
    status: ResponseStatus = ResponseStatus.SUCCESS

    items: list[T]

    pagination: PaginationMetadata

    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata
    )


def create_pagination_metadata(
    *,
    page: int,
    page_size: int,
    total_items: int,
) -> PaginationMetadata:
    if page < 1:
        raise ValueError(
            "Page must be greater than zero"
        )

    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ValueError(
            "Invalid page size"
        )

    if total_items < 0:
        raise ValueError(
            "Total items cannot be negative"
        )

    total_pages = (
        (total_items + page_size - 1) // page_size
    )

    return PaginationMetadata(
        page=page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_previous=page > 1,
    )


# ============================================================
# Authentication Responses
# ============================================================


class UserResponse(APIBaseModel):
    user_id: str
    tenant_id: str

    email: str | None = None

    roles: list[str] = Field(
        default_factory=list
    )

    permissions: list[str] = Field(
        default_factory=list
    )

    is_active: bool = True


class TokenResponse(APIBaseModel):
    access_token: str = Field(
        repr=False
    )

    token_type: str = "bearer"

    expires_in: int = Field(
        gt=0
    )

    refresh_token: str | None = Field(
        default=None,
        repr=False,
    )


class LoginResponse(APIBaseModel):
    success: bool = True

    user: UserResponse
    tokens: TokenResponse

    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata
    )


class SessionResponse(APIBaseModel):
    session_id: str
    user_id: str
    tenant_id: str

    status: str

    created_at: datetime
    expires_at: datetime

    device_id: str | None = None
    device_name: str | None = None

    last_seen_at: datetime | None = None


# ============================================================
# Tenant Responses
# ============================================================


class TenantResponse(APIBaseModel):
    tenant_id: str
    name: str
    status: str

    created_at: datetime | None = None


class TenantAccessResponse(APIBaseModel):
    allowed: bool
    tenant_id: str
    resource_type: str
    action: str
    reason: str


# ============================================================
# Robot Responses
# ============================================================


class RobotResponse(APIBaseModel):
    robot_id: str
    tenant_id: str

    name: str
    robot_type: str | None = None

    fleet_id: str | None = None

    operational_status: RobotOperationalStatus = (
        RobotOperationalStatus.UNKNOWN
    )

    health_status: HealthStatus = (
        HealthStatus.UNKNOWN
    )

    battery_percentage: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    last_seen_at: datetime | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class RobotHealthResponse(APIBaseModel):
    robot_id: str
    tenant_id: str

    health_status: HealthStatus

    health_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    battery_percentage: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    cpu_temperature_c: float | None = None
    motor_temperature_c: float | None = None

    active_faults: list[str] = Field(
        default_factory=list
    )

    recommendations: list[str] = Field(
        default_factory=list
    )

    evaluated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# Fleet Responses
# ============================================================


class FleetResponse(APIBaseModel):
    fleet_id: str
    tenant_id: str
    name: str

    robot_count: int = Field(
        default=0,
        ge=0,
    )

    online_robots: int = Field(
        default=0,
        ge=0,
    )

    offline_robots: int = Field(
        default=0,
        ge=0,
    )

    health_status: HealthStatus = (
        HealthStatus.UNKNOWN
    )


# ============================================================
# Sensor Responses
# ============================================================


class SensorResponse(APIBaseModel):
    sensor_id: str
    robot_id: str
    tenant_id: str

    sensor_type: str

    status: HealthStatus

    last_reading_at: datetime | None = None


class SensorReadingResponse(APIBaseModel):
    sensor_id: str
    robot_id: str
    tenant_id: str

    value: float
    unit: str

    timestamp: datetime

    quality_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


# ============================================================
# Telemetry Responses
# ============================================================


class TelemetryResponse(APIBaseModel):
    telemetry_id: str

    robot_id: str
    tenant_id: str

    timestamp: datetime

    measurements: dict[str, float]

    source: str | None = None


class TelemetrySummaryResponse(APIBaseModel):
    robot_id: str
    tenant_id: str

    start_time: datetime
    end_time: datetime

    sample_count: int = Field(
        ge=0
    )

    averages: dict[str, float] = Field(
        default_factory=dict
    )

    minimums: dict[str, float] = Field(
        default_factory=dict
    )

    maximums: dict[str, float] = Field(
        default_factory=dict
    )


# ============================================================
# Alert Responses
# ============================================================


class AlertResponse(APIBaseModel):
    alert_id: str
    tenant_id: str

    robot_id: str | None = None

    severity: AlertSeverity

    title: str
    message: str

    acknowledged: bool = False

    created_at: datetime = Field(
        default_factory=utc_now
    )

    acknowledged_at: datetime | None = None


# ============================================================
# Maintenance Responses
# ============================================================


class MaintenanceResponse(APIBaseModel):
    maintenance_id: str
    tenant_id: str
    robot_id: str

    maintenance_type: str
    status: str

    description: str | None = None

    scheduled_at: datetime | None = None
    completed_at: datetime | None = None

    estimated_cost: float | None = Field(
        default=None,
        ge=0,
    )

    currency: str = "USD"


class PredictiveMaintenanceResponse(APIBaseModel):
    robot_id: str
    tenant_id: str

    predicted_failure: bool

    failure_probability: float = Field(
        ge=0,
        le=1,
    )

    remaining_useful_life_hours: float | None = Field(
        default=None,
        ge=0,
    )

    affected_components: list[str] = Field(
        default_factory=list
    )

    recommendations: list[str] = Field(
        default_factory=list
    )

    model_id: str | None = None

    evaluated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# Repair Responses
# ============================================================


class RepairEstimateResponse(APIBaseModel):
    estimate_id: str
    tenant_id: str
    robot_id: str

    issue_description: str

    parts_cost: float = Field(
        ge=0
    )

    labor_cost: float = Field(
        ge=0
    )

    additional_cost: float = Field(
        default=0,
        ge=0,
    )

    total_estimated_cost: float = Field(
        ge=0
    )

    currency: str = "USD"

    estimated_repair_hours: float | None = Field(
        default=None,
        ge=0,
    )

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    requires_human_review: bool = True


class RobotPartResponse(APIBaseModel):
    part_id: str
    tenant_id: str

    name: str
    manufacturer: str | None = None

    compatible_robot_ids: list[str] = Field(
        default_factory=list
    )

    estimated_price: float | None = Field(
        default=None,
        ge=0,
    )

    currency: str = "USD"

    availability: str | None = None


# ============================================================
# AI Model Responses
# ============================================================


class AIModelResponse(APIBaseModel):
    model_id: str
    name: str
    version: str

    provider: str | None = None

    capabilities: list[str] = Field(
        default_factory=list
    )

    status: str

    context_window: int | None = Field(
        default=None,
        ge=1,
    )


class AIUsageResponse(APIBaseModel):
    input_tokens: int = Field(
        default=0,
        ge=0,
    )

    output_tokens: int = Field(
        default=0,
        ge=0,
    )

    total_tokens: int = Field(
        default=0,
        ge=0,
    )

    estimated_cost_usd: float | None = Field(
        default=None,
        ge=0,
    )


class CitationResponse(APIBaseModel):
    citation_id: str
    source_id: str

    title: str | None = None
    url: str | None = None

    page: int | None = Field(
        default=None,
        ge=1,
    )

    snippet: str | None = None


class AIResponse(APIBaseModel):
    response_id: str

    tenant_id: str
    conversation_id: str | None = None

    status: AIExecutionStatus

    answer: str

    model_id: str | None = None

    citations: list[CitationResponse] = Field(
        default_factory=list
    )

    usage: AIUsageResponse | None = None

    grounded: bool | None = None

    requires_human_review: bool = False

    generated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# RAG Responses
# ============================================================


class RAGDocumentResponse(APIBaseModel):
    document_id: str
    tenant_id: str

    filename: str
    content_type: str

    processing_status: str

    chunk_count: int = Field(
        default=0,
        ge=0,
    )

    uploaded_at: datetime = Field(
        default_factory=utc_now
    )


class RAGSearchResultResponse(APIBaseModel):
    document_id: str
    chunk_id: str

    content: str

    score: float | None = None

    source: str | None = None
    page: int | None = None


class RAGQueryResponse(APIBaseModel):
    query: str

    answer: str

    results: list[RAGSearchResultResponse] = Field(
        default_factory=list
    )

    citations: list[CitationResponse] = Field(
        default_factory=list
    )

    grounded: bool = False

    latency_ms: float | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# Simulation Responses
# ============================================================


class SimulationResponse(APIBaseModel):
    simulation_id: str
    tenant_id: str

    name: str
    status: str

    environment: str | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None

    metrics: dict[str, float] = Field(
        default_factory=dict
    )

    # Simulation execution does not authorize
    # real-world robot actuation.
    physical_execution_authorized: bool = False

    @field_validator(
        "physical_execution_authorized"
    )
    @classmethod
    def reject_physical_authorization(
        cls,
        value: bool,
    ) -> bool:
        if value:
            raise ValueError(
                "Simulation responses cannot authorize "
                "physical robot execution"
            )

        return False


# ============================================================
# Evaluation Responses
# ============================================================


class EvaluationMetricResponse(APIBaseModel):
    metric_name: str
    metric_value: float

    threshold: float | None = None

    passed: bool | None = None


class EvaluationResponse(APIBaseModel):
    evaluation_id: str
    tenant_id: str

    model_id: str | None = None

    status: str

    metrics: list[EvaluationMetricResponse] = Field(
        default_factory=list
    )

    overall_score: float | None = None

    passed: bool | None = None

    evaluated_at: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# System Health Responses
# ============================================================


class ComponentHealthResponse(APIBaseModel):
    component: str
    status: HealthStatus

    latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    message: str | None = None


class SystemHealthResponse(APIBaseModel):
    status: HealthStatus

    service: str = (
        "physical-ai-autonomous-robotics-intelligence"
    )

    version: str

    uptime_seconds: float | None = Field(
        default=None,
        ge=0,
    )

    components: list[ComponentHealthResponse] = Field(
        default_factory=list
    )

    timestamp: datetime = Field(
        default_factory=utc_now
    )


# ============================================================
# Response Factories
# ============================================================


def success_response(
    data: T,
    *,
    message: str = "Operation completed successfully",
    request_id: str | None = None,
    tenant_id: str | None = None,
    processing_time_ms: float | None = None,
) -> SuccessResponse[T]:
    return SuccessResponse[T](
        data=data,
        message=message,
        metadata=ResponseMetadata(
            request_id=request_id,
            tenant_id=tenant_id,
            processing_time_ms=processing_time_ms,
        ),
    )


def message_response(
    message: str,
    *,
    request_id: str | None = None,
    tenant_id: str | None = None,
) -> MessageResponse:
    return MessageResponse(
        message=message,
        metadata=ResponseMetadata(
            request_id=request_id,
            tenant_id=tenant_id,
        ),
    )


def paginated_response(
    items: list[T],
    *,
    page: int,
    page_size: int,
    total_items: int,
    request_id: str | None = None,
    tenant_id: str | None = None,
) -> PaginatedResponse[T]:
    return PaginatedResponse[T](
        items=items,
        pagination=create_pagination_metadata(
            page=page,
            page_size=page_size,
            total_items=total_items,
        ),
        metadata=ResponseMetadata(
            request_id=request_id,
            tenant_id=tenant_id,
        ),
    )


def error_response(
    *,
    code: str,
    message: str,
    status_code: int,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            status_code=status_code,
            request_id=request_id,
            details=details,
        )
    )


# ============================================================
# Exports
# ============================================================


__all__ = [
    "APIBaseModel",
    "ResponseStatus",
    "OperationStatus",
    "HealthStatus",
    "RobotOperationalStatus",
    "AlertSeverity",
    "AIExecutionStatus",
    "ResponseMetadata",
    "SuccessResponse",
    "MessageResponse",
    "OperationResponse",
    "ErrorDetail",
    "ErrorResponse",
    "ValidationFieldError",
    "ValidationErrorResponse",
    "PaginationParams",
    "PaginationMetadata",
    "PaginatedResponse",
    "UserResponse",
    "TokenResponse",
    "LoginResponse",
    "SessionResponse",
    "TenantResponse",
    "TenantAccessResponse",
    "RobotResponse",
    "RobotHealthResponse",
    "FleetResponse",
    "SensorResponse",
    "SensorReadingResponse",
    "TelemetryResponse",
    "TelemetrySummaryResponse",
    "AlertResponse",
    "MaintenanceResponse",
    "PredictiveMaintenanceResponse",
    "RepairEstimateResponse",
    "RobotPartResponse",
    "AIModelResponse",
    "AIUsageResponse",
    "CitationResponse",
    "AIResponse",
    "RAGDocumentResponse",
    "RAGSearchResultResponse",
    "RAGQueryResponse",
    "SimulationResponse",
    "EvaluationMetricResponse",
    "EvaluationResponse",
    "ComponentHealthResponse",
    "SystemHealthResponse",
    "utc_now",
    "ensure_utc",
    "create_pagination_metadata",
    "success_response",
    "message_response",
    "paginated_response",
    "error_response",
]
"""Simulation Agent module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/api/endpoints/simulation.py

Physical AI & Autonomous Robotics Intelligence Platform
Simulation & Digital Twin API

Python 3.11+ | FastAPI | Pydantic v2

Features:
- Simulation environment management
- Physics engine configuration
- Robot simulation scenarios
- Virtual robot and sensor configuration
- Simulation run creation and execution requests
- Pause, resume and cancellation
- Simulation status and progress
- Virtual telemetry snapshots
- Simulation results and metrics
- Collision and safety-event reporting
- Replay metadata
- Simulation comparison
- Simulation evaluation
- Simulation statistics
- Tenant isolation, RBAC and auditing

IMPORTANT:
    This module is simulation-only.

    It does not issue commands to physical robots.
    A separate simulation worker must enforce resource
    limits, sandbox isolation and run state transitions.

    Simulation results cannot certify real-world safety.
"""

from __future__ import annotations

import logging
import re

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.api.dependencies import CurrentTenant


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/simulation",
    tags=["Robotics Simulation"],
)

READ = "simulation:read"
CREATE = "simulation:create"
UPDATE = "simulation:update"
RUN = "simulation:run"
CONTROL = "simulation:control"
EVALUATE = "simulation:evaluate"
ANALYTICS = "simulation:analytics"

ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")

MAX_PAGE_SIZE = 100
MAX_ROBOTS_PER_SCENARIO = 100
MAX_SENSORS_PER_ROBOT = 100


# ============================================================
# Enumerations
# ============================================================

class SimulationEngine(str, Enum):
    ISAAC_SIM = "isaac_sim"
    GAZEBO = "gazebo"
    MUJOCO = "mujoco"
    PYBULLET = "pybullet"
    WEBOTS = "webots"
    CUSTOM = "custom"


class EnvironmentType(str, Enum):
    WAREHOUSE = "warehouse"
    FACTORY = "factory"
    OFFICE = "office"
    HOSPITAL = "hospital"
    OUTDOOR = "outdoor"
    URBAN = "urban"
    LABORATORY = "laboratory"
    CUSTOM = "custom"


class SimulationStatus(str, Enum):
    QUEUED = "queued"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SimulationAction(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"


class SimulationTaskType(str, Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PICK_AND_PLACE = "pick_and_place"
    INSPECTION = "inspection"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    MULTI_ROBOT = "multi_robot"
    PERCEPTION = "perception"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    CUSTOM = "custom"


class VirtualSensorType(str, Enum):
    CAMERA = "camera"
    DEPTH_CAMERA = "depth_camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    ENCODER = "encoder"
    FORCE_TORQUE = "force_torque"
    PROXIMITY = "proximity"
    OTHER = "other"


class SimulationEventType(str, Enum):
    COLLISION = "collision"
    NEAR_MISS = "near_miss"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    SENSOR_FAILURE = "sensor_failure"
    ROBOT_FAULT = "robot_fault"
    SAFETY_VIOLATION = "safety_violation"
    CUSTOM = "custom"


class EvaluationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


# ============================================================
# Base Models
# ============================================================

class APIModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        allow_inf_nan=False,
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str) -> str:
    if not ID_PATTERN.fullmatch(value):
        raise ValueError("Invalid resource identifier")
    return value


def checked_id(value: str) -> str:
    try:
        return validate_id(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid resource identifier",
        ) from exc


# ============================================================
# Exceptions
# ============================================================

class SimulationError(Exception):
    pass


class SimulationNotFound(SimulationError):
    pass


class SimulationForbidden(SimulationError):
    pass


class SimulationConflict(SimulationError):
    pass


class SimulationInvalid(SimulationError):
    pass


class SimulationUnavailable(SimulationError):
    pass


def handle_error(exc: Exception) -> None:
    if isinstance(exc, HTTPException):
        raise exc

    if isinstance(exc, SimulationNotFound):
        raise HTTPException(
            status_code=404,
            detail="Simulation resource not found",
        ) from exc

    if isinstance(exc, SimulationForbidden):
        raise HTTPException(
            status_code=403,
            detail="Simulation resource access denied",
        ) from exc

    if isinstance(exc, SimulationConflict):
        raise HTTPException(
            status_code=409,
            detail="Simulation state conflict",
        ) from exc

    if isinstance(exc, SimulationInvalid):
        raise HTTPException(
            status_code=422,
            detail="Invalid simulation request",
        ) from exc

    if isinstance(exc, SimulationUnavailable):
        raise HTTPException(
            status_code=503,
            detail="Simulation service unavailable",
        ) from exc

    logger.exception("Unexpected simulation API error")

    raise HTTPException(
        status_code=500,
        detail="Internal simulation service error",
    ) from exc


# ============================================================
# Geometry
# ============================================================

class Vector3(APIModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Quaternion(APIModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    @model_validator(mode="after")
    def validate_normalization(self) -> "Quaternion":
        import math

        norm = math.sqrt(
            self.x**2
            + self.y**2
            + self.z**2
            + self.w**2
        )

        if not math.isclose(
            norm,
            1.0,
            rel_tol=1e-3,
            abs_tol=1e-3,
        ):
            raise ValueError(
                "Quaternion must be normalized"
            )

        return self


class Pose3D(APIModel):
    position: Vector3
    orientation: Quaternion


# ============================================================
# Physics Configuration
# ============================================================

class PhysicsConfiguration(APIModel):
    gravity: Vector3 = Field(
        default_factory=lambda: Vector3(
            x=0,
            y=0,
            z=-9.81,
        )
    )

    time_step_seconds: float = Field(
        default=0.01,
        gt=0,
        le=0.1,
    )

    solver_iterations: int = Field(
        default=20,
        ge=1,
        le=1000,
    )

    friction_coefficient: float = Field(
        default=0.6,
        ge=0,
        le=5,
    )

    restitution: float = Field(
        default=0.1,
        ge=0,
        le=1,
    )

    deterministic_mode: bool = True


# ============================================================
# Environment
# ============================================================

class SimulationEnvironmentCreate(APIModel):
    name: str = Field(
        min_length=3,
        max_length=200,
    )

    description: str | None = Field(
        default=None,
        max_length=3000,
    )

    environment_type: EnvironmentType

    engine: SimulationEngine

    asset_reference: str | None = None

    physics: PhysicsConfiguration = Field(
        default_factory=PhysicsConfiguration
    )

    @field_validator("asset_reference")
    @classmethod
    def validate_asset(
        cls,
        value: str | None,
    ) -> str | None:
        return validate_id(value) if value else None


class SimulationEnvironment(APIModel):
    environment_id: str
    tenant_id: str

    name: str
    description: str | None = None

    environment_type: EnvironmentType

    engine: SimulationEngine

    asset_reference: str | None = None

    physics: PhysicsConfiguration

    version: int = Field(ge=1)

    created_by: str
    created_at: datetime
    updated_at: datetime


class SimulationEnvironmentList(APIModel):
    items: list[SimulationEnvironment]

    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


# ============================================================
# Virtual Sensors and Robots
# ============================================================

class VirtualSensorConfiguration(APIModel):
    name: str = Field(
        min_length=1,
        max_length=128,
    )

    sensor_type: VirtualSensorType

    sampling_rate_hz: float = Field(
        gt=0,
        le=100000,
    )

    noise_enabled: bool = False

    noise_stddev: float = Field(
        default=0,
        ge=0,
    )

    pose: Pose3D | None = None


class VirtualRobotConfiguration(APIModel):
    robot_id: str

    robot_model_reference: str

    initial_pose: Pose3D

    sensors: list[VirtualSensorConfiguration] = Field(
        default_factory=list,
        max_length=MAX_SENSORS_PER_ROBOT,
    )

    @field_validator(
        "robot_id",
        "robot_model_reference",
    )
    @classmethod
    def validate_references(cls, value: str) -> str:
        return validate_id(value)


# ============================================================
# Scenario
# ============================================================

class SimulationScenarioCreate(APIModel):
    name: str = Field(
        min_length=3,
        max_length=200,
    )

    description: str | None = Field(
        default=None,
        max_length=3000,
    )

    environment_id: str

    task_type: SimulationTaskType

    robots: list[VirtualRobotConfiguration] = Field(
        min_length=1,
        max_length=MAX_ROBOTS_PER_SCENARIO,
    )

    maximum_duration_seconds: int = Field(
        default=600,
        ge=1,
        le=86400,
    )

    random_seed: int = Field(
        default=42,
        ge=0,
        le=2**32 - 1,
    )

    @field_validator("environment_id")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        return validate_id(value)

    @model_validator(mode="after")
    def validate_unique_robots(
        self,
    ) -> "SimulationScenarioCreate":
        ids = [robot.robot_id for robot in self.robots]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate robot IDs in scenario"
            )

        return self


class SimulationScenario(APIModel):
    scenario_id: str
    tenant_id: str

    name: str
    description: str | None = None

    environment_id: str

    task_type: SimulationTaskType

    robots: list[VirtualRobotConfiguration]

    maximum_duration_seconds: int

    random_seed: int

    version: int = Field(ge=1)

    created_by: str
    created_at: datetime
    updated_at: datetime


class SimulationScenarioList(APIModel):
    items: list[SimulationScenario]

    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


# ============================================================
# Simulation Runs
# ============================================================

class SimulationRunCreate(APIModel):
    scenario_id: str

    model_id: str | None = None

    record_telemetry: bool = True
    record_events: bool = True
    generate_replay: bool = True

    @field_validator("scenario_id", "model_id")
    @classmethod
    def validate_references(
        cls,
        value: str | None,
    ) -> str | None:
        return validate_id(value) if value else None


class SimulationRun(APIModel):
    run_id: str
    tenant_id: str

    scenario_id: str

    model_id: str | None = None

    status: SimulationStatus

    progress_percent: float = Field(
        ge=0,
        le=100,
    )

    simulated_time_seconds: float = Field(
        default=0,
        ge=0,
    )

    created_by: str

    created_at: datetime

    started_at: datetime | None = None
    completed_at: datetime | None = None

    error_message: str | None = None


class SimulationRunList(APIModel):
    items: list[SimulationRun]

    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class SimulationControlRequest(APIModel):
    action: SimulationAction

    reason: str = Field(
        min_length=3,
        max_length=1000,
    )


class SimulationControlResponse(APIModel):
    run_id: str

    status: SimulationStatus

    action: SimulationAction

    accepted: bool

    message: str

    updated_at: datetime


# ============================================================
# Simulation Telemetry
# ============================================================

class SimulatedRobotState(APIModel):
    robot_id: str

    pose: Pose3D

    linear_velocity: Vector3

    angular_velocity: Vector3

    battery_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    collision_detected: bool = False


class SimulationTelemetrySnapshot(APIModel):
    run_id: str
    tenant_id: str

    simulated_time_seconds: float = Field(ge=0)

    robots: list[SimulatedRobotState]

    timestamp: datetime


# ============================================================
# Simulation Events
# ============================================================

class SimulationEvent(APIModel):
    event_id: str

    run_id: str

    event_type: SimulationEventType

    robot_id: str | None = None

    description: str

    simulated_time_seconds: float = Field(ge=0)

    created_at: datetime


class SimulationEventList(APIModel):
    run_id: str

    items: list[SimulationEvent]

    total: int = Field(ge=0)


# ============================================================
# Simulation Metrics
# ============================================================

class SimulationMetrics(APIModel):
    run_id: str

    task_success_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    navigation_success_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    collision_count: int = Field(default=0, ge=0)

    near_miss_count: int = Field(default=0, ge=0)

    safety_violation_count: int = Field(
        default=0,
        ge=0,
    )

    total_distance_m: float | None = Field(
        default=None,
        ge=0,
    )

    average_task_duration_seconds: float | None = Field(
        default=None,
        ge=0,
    )

    simulated_duration_seconds: float = Field(
        ge=0,
    )

    wall_clock_duration_seconds: float | None = Field(
        default=None,
        ge=0,
    )


class SimulationResult(APIModel):
    run_id: str
    tenant_id: str

    scenario_id: str

    status: SimulationStatus

    metrics: SimulationMetrics

    warnings: list[str] = Field(
        default_factory=list,
    )

    generated_at: datetime


# ============================================================
# Replay
# ============================================================

class SimulationReplay(APIModel):
    replay_id: str
    tenant_id: str

    run_id: str

    available: bool

    frame_count: int = Field(ge=0)

    duration_seconds: float = Field(ge=0)

    artifact_id: str | None = None

    created_at: datetime


# ============================================================
# Comparison
# ============================================================

class SimulationComparisonRequest(APIModel):
    run_ids: list[str] = Field(
        min_length=2,
        max_length=10,
    )

    @field_validator("run_ids")
    @classmethod
    def validate_runs(
        cls,
        values: list[str],
    ) -> list[str]:
        for value in values:
            validate_id(value)

        if len(values) != len(set(values)):
            raise ValueError("Duplicate run IDs")

        return values


class SimulationComparisonResult(APIModel):
    comparison_id: str
    tenant_id: str

    run_ids: list[str]

    metrics: list[SimulationMetrics]

    summary: str | None = None

    generated_at: datetime


# ============================================================
# Evaluation
# ============================================================

class SimulationEvaluationRequest(APIModel):
    run_id: str

    minimum_task_success_rate: float = Field(
        default=0.9,
        ge=0,
        le=1,
    )

    maximum_collisions: int = Field(
        default=0,
        ge=0,
    )

    maximum_safety_violations: int = Field(
        default=0,
        ge=0,
    )

    @field_validator("run_id")
    @classmethod
    def validate_run(cls, value: str) -> str:
        return validate_id(value)


class SimulationEvaluationResult(APIModel):
    evaluation_id: str
    tenant_id: str

    run_id: str

    status: EvaluationStatus

    passed_checks: list[str] = Field(
        default_factory=list,
    )

    failed_checks: list[str] = Field(
        default_factory=list,
    )

    warnings: list[str] = Field(
        default_factory=list,
    )

    evaluated_at: datetime


# ============================================================
# Statistics
# ============================================================

class SimulationStatistics(APIModel):
    tenant_id: str

    total_environments: int = Field(ge=0)
    total_scenarios: int = Field(ge=0)

    total_runs: int = Field(ge=0)

    queued_runs: int = Field(ge=0)
    running_runs: int = Field(ge=0)
    completed_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)

    total_simulated_hours: float = Field(ge=0)

    generated_at: datetime = Field(
        default_factory=utc_now,
    )


# ============================================================
# Service Contract
# ============================================================

class SimulationService(Protocol):

    async def create_environment(
        self,
        *,
        tenant_id: str,
        user_id: str,
        payload: SimulationEnvironmentCreate,
    ) -> SimulationEnvironment:
        ...

    async def list_environments(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int,
        offset: int,
    ) -> tuple[list[SimulationEnvironment], int]:
        ...

    async def get_environment(
        self,
        *,
        tenant_id: str,
        user_id: str,
        environment_id: str,
    ) -> SimulationEnvironment | None:
        ...

    async def create_scenario(
        self,
        *,
        tenant_id: str,
        user_id: str,
        payload: SimulationScenarioCreate,
    ) -> SimulationScenario:
        ...

    async def list_scenarios(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int,
        offset: int,
    ) -> tuple[list[SimulationScenario], int]:
        ...

    async def get_scenario(
        self,
        *,
        tenant_id: str,
        user_id: str,
        scenario_id: str,
    ) -> SimulationScenario | None:
        ...

    async def create_run(
        self,
        *,
        tenant_id: str,
        user_id: str,
        payload: SimulationRunCreate,
    ) -> SimulationRun:
        ...

    async def list_runs(
        self,
        *,
        tenant_id: str,
        user_id: str,
        scenario_id: str | None,
        run_status: SimulationStatus | None,
        limit: int,
        offset: int,
    ) -> tuple[list[SimulationRun], int]:
        ...

    async def get_run(
        self,
        *,
        tenant_id: str,
        user_id: str,
        run_id: str,
    ) -> SimulationRun | None:
        ...

    async def control_run(
        self,
        *,
        tenant_id: str,
        user_id: str,
        run_id: str,
        payload: SimulationControlRequest,
    ) -> SimulationControlResponse:
        ...

    async def get_telemetry(
        self,
        *,
        tenant_id: str,
        user_id: str,
        run_id: str,
    ) -> SimulationTelemetrySnapshot:
        ...

    async def get_events(
        self,
        *,
        tenant_id: str,
        user_id: str,
        run_id: str,
    ) -> SimulationEventList:
        ...

    async def get_result(
        self,
        *,
        tenant_id: str,
        user_id: str,
        run_id: str,
    ) -> SimulationResult:
        ...

    async def get_replay(
        self,
        *,
        tenant_id: str,
        user_id: str,
        run_id: str,
    ) -> SimulationReplay:
        ...

    async def compare_runs(
        self,
        *,
        tenant_id: str,
        user_id: str,
        payload: SimulationComparisonRequest,
    ) -> SimulationComparisonResult:
        ...

    async def evaluate_run(
        self,
        *,
        tenant_id: str,
        user_id: str,
        payload: SimulationEvaluationRequest,
    ) -> SimulationEvaluationResult:
        ...

    async def get_statistics(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> SimulationStatistics:
        ...


# ============================================================
# Authorization and Audit
# ============================================================

class SimulationAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        ...

    async def require_access(
        self,
        *,
        tenant_id: str,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
    ) -> None:
        ...


class SimulationAuditSink(Protocol):

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
# Dependency Configuration
# ============================================================

_service: SimulationService | None = None
_authorizer: SimulationAuthorizer | None = None
_audit: SimulationAuditSink | None = None


def configure_simulation_api(
    *,
    service: SimulationService,
    authorizer: SimulationAuthorizer,
    audit_sink: SimulationAuditSink,
) -> None:
    global _service, _authorizer, _audit

    _service = service
    _authorizer = authorizer
    _audit = audit_sink


def get_service() -> SimulationService:
    if _service is None:
        raise HTTPException(
            status_code=503,
            detail="Simulation service not configured",
        )
    return _service


def get_authorizer() -> SimulationAuthorizer:
    if _authorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Simulation authorization not configured",
        )
    return _authorizer


def get_audit() -> SimulationAuditSink:
    if _audit is None:
        raise HTTPException(
            status_code=503,
            detail="Simulation audit sink not configured",
        )
    return _audit


# ============================================================
# Security Helpers
# ============================================================

async def authorize(
    tenant: CurrentTenant,
    authorizer: SimulationAuthorizer,
    permission: str,
) -> None:
    try:
        await authorizer.require_permission(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            permission=permission,
        )
    except SimulationForbidden as exc:
        raise HTTPException(
            status_code=403,
            detail="Simulation permission denied",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Authorization unavailable",
        ) from exc


async def require_resource(
    tenant: CurrentTenant,
    authorizer: SimulationAuthorizer,
    resource_type: str,
    resource_id: str,
    action: str,
) -> None:
    try:
        await authorizer.require_access(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
        )
    except SimulationNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail="Resource not found",
        ) from exc
    except SimulationForbidden as exc:
        raise HTTPException(
            status_code=403,
            detail="Resource access denied",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Resource authorization unavailable",
        ) from exc


def verify_tenant(
    tenant: CurrentTenant,
    resource_tenant_id: str,
) -> None:
    if tenant.tenant_id != resource_tenant_id:
        raise SimulationForbidden()


async def audit_action(
    audit: SimulationAuditSink,
    tenant: CurrentTenant,
    action: str,
    resource_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    await audit.record(
        tenant_id=tenant.tenant_id,
        user_id=tenant.user_id,
        action=action,
        resource_id=resource_id,
        metadata=dict(metadata or {}),
    )


async def require_run(
    *,
    tenant: CurrentTenant,
    service: SimulationService,
    authorizer: SimulationAuthorizer,
    run_id: str,
    action: str,
) -> SimulationRun:
    checked_id(run_id)

    await require_resource(
        tenant,
        authorizer,
        "simulation_run",
        run_id,
        action,
    )

    run = await service.get_run(
        tenant_id=tenant.tenant_id,
        user_id=tenant.user_id,
        run_id=run_id,
    )

    if run is None:
        raise SimulationNotFound()

    verify_tenant(tenant, run.tenant_id)

    if run.run_id != run_id:
        raise SimulationForbidden()

    return run


# ============================================================
# Environment Endpoints
# ============================================================

@router.post(
    "/environments",
    response_model=SimulationEnvironment,
    status_code=status.HTTP_201_CREATED,
)
async def create_environment(
    payload: SimulationEnvironmentCreate,
    tenant: CurrentTenant,
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
    audit: SimulationAuditSink = Depends(get_audit),
):
    await authorize(tenant, authorizer, CREATE)

    if payload.asset_reference:
        await require_resource(
            tenant,
            authorizer,
            "simulation_asset",
            payload.asset_reference,
            "read",
        )

    try:
        result = await service.create_environment(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            payload=payload,
        )

        verify_tenant(tenant, result.tenant_id)

        await audit_action(
            audit,
            tenant,
            "simulation.environment.create",
            result.environment_id,
            {"engine": payload.engine.value},
        )

        return result

    except Exception as exc:
        handle_error(exc)


@router.get(
    "/environments",
    response_model=SimulationEnvironmentList,
)
async def list_environments(
    tenant: CurrentTenant,
    limit: int = Query(default=20, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
):
    await authorize(tenant, authorizer, READ)

    try:
        items, total = await service.list_environments(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            limit=limit,
            offset=offset,
        )

        for item in items:
            verify_tenant(tenant, item.tenant_id)

            await require_resource(
                tenant,
                authorizer,
                "simulation_environment",
                item.environment_id,
                "read",
            )

        return SimulationEnvironmentList(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )

    except Exception as exc:
        handle_error(exc)


@router.get(
    "/environments/{environment_id}",
    response_model=SimulationEnvironment,
)
async def get_environment(
    environment_id: str,
    tenant: CurrentTenant,
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
):
    await authorize(tenant, authorizer, READ)
    checked_id(environment_id)

    await require_resource(
        tenant,
        authorizer,
        "simulation_environment",
        environment_id,
        "read",
    )

    try:
        result = await service.get_environment(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            environment_id=environment_id,
        )

        if result is None:
            raise SimulationNotFound()

        verify_tenant(tenant, result.tenant_id)

        if result.environment_id != environment_id:
            raise SimulationForbidden()

        return result

    except Exception as exc:
        handle_error(exc)


# ============================================================
# Scenario Endpoints
# ============================================================

@router.post(
    "/scenarios",
    response_model=SimulationScenario,
    status_code=status.HTTP_201_CREATED,
)
async def create_scenario(
    payload: SimulationScenarioCreate,
    tenant: CurrentTenant,
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
    audit: SimulationAuditSink = Depends(get_audit),
):
    await authorize(tenant, authorizer, CREATE)

    await require_resource(
        tenant,
        authorizer,
        "simulation_environment",
        payload.environment_id,
        "read",
    )

    for robot in payload.robots:
        await require_resource(
            tenant,
            authorizer,
            "robot",
            robot.robot_id,
            "read",
        )

        await require_resource(
            tenant,
            authorizer,
            "simulation_asset",
            robot.robot_model_reference,
            "read",
        )

    try:
        result = await service.create_scenario(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            payload=payload,
        )

        verify_tenant(tenant, result.tenant_id)

        if result.environment_id != payload.environment_id:
            raise SimulationForbidden()

        await audit_action(
            audit,
            tenant,
            "simulation.scenario.create",
            result.scenario_id,
            {"task_type": payload.task_type.value},
        )

        return result

    except Exception as exc:
        handle_error(exc)


@router.get(
    "/scenarios",
    response_model=SimulationScenarioList,
)
async def list_scenarios(
    tenant: CurrentTenant,
    limit: int = Query(default=20, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
):
    await authorize(tenant, authorizer, READ)

    try:
        items, total = await service.list_scenarios(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            limit=limit,
            offset=offset,
        )

        for item in items:
            verify_tenant(tenant, item.tenant_id)

            await require_resource(
                tenant,
                authorizer,
                "simulation_scenario",
                item.scenario_id,
                "read",
            )

        return SimulationScenarioList(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )

    except Exception as exc:
        handle_error(exc)


@router.get(
    "/scenarios/{scenario_id}",
    response_model=SimulationScenario,
)
async def get_scenario(
    scenario_id: str,
    tenant: CurrentTenant,
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
):
    await authorize(tenant, authorizer, READ)
    checked_id(scenario_id)

    await require_resource(
        tenant,
        authorizer,
        "simulation_scenario",
        scenario_id,
        "read",
    )

    try:
        result = await service.get_scenario(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            scenario_id=scenario_id,
        )

        if result is None:
            raise SimulationNotFound()

        verify_tenant(tenant, result.tenant_id)

        if result.scenario_id != scenario_id:
            raise SimulationForbidden()

        return result

    except Exception as exc:
        handle_error(exc)


# ============================================================
# Run Endpoints
# ============================================================

@router.post(
    "/runs",
    response_model=SimulationRun,
    status_code=status.HTTP_201_CREATED,
)
async def create_simulation_run(
    payload: SimulationRunCreate,
    tenant: CurrentTenant,
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
    audit: SimulationAuditSink = Depends(get_audit),
):
    await authorize(tenant, authorizer, RUN)

    await require_resource(
        tenant,
        authorizer,
        "simulation_scenario",
        payload.scenario_id,
        "run",
    )

    if payload.model_id:
        await require_resource(
            tenant,
            authorizer,
            "model",
            payload.model_id,
            "read",
        )

    try:
        # Concrete service must enqueue a sandboxed job.
        # It must never connect simulation controls
        # to a physical robot command channel.

        result = await service.create_run(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            payload=payload,
        )

        verify_tenant(tenant, result.tenant_id)

        if result.scenario_id != payload.scenario_id:
            raise SimulationForbidden()

        await audit_action(
            audit,
            tenant,
            "simulation.run.create",
            result.run_id,
            {"scenario_id": payload.scenario_id},
        )

        return result

    except Exception as exc:
        handle_error(exc)


@router.get(
    "/runs",
    response_model=SimulationRunList,
)
async def list_simulation_runs(
    tenant: CurrentTenant,
    scenario_id: str | None = Query(default=None),
    run_status: SimulationStatus | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
):
    await authorize(tenant, authorizer, READ)

    if scenario_id:
        checked_id(scenario_id)

        await require_resource(
            tenant,
            authorizer,
            "simulation_scenario",
            scenario_id,
            "read",
        )

    try:
        items, total = await service.list_runs(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            scenario_id=scenario_id,
            run_status=run_status,
            limit=limit,
            offset=offset,
        )

        for item in items:
            verify_tenant(tenant, item.tenant_id)

            await require_resource(
                tenant,
                authorizer,
                "simulation_run",
                item.run_id,
                "read",
            )

        return SimulationRunList(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )

    except Exception as exc:
        handle_error(exc)


@router.get(
    "/runs/{run_id}",
    response_model=SimulationRun,
)
async def get_simulation_run(
    run_id: str,
    tenant: CurrentTenant,
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
):
    await authorize(tenant, authorizer, READ)

    try:
        return await require_run(
            tenant=tenant,
            service=service,
            authorizer=authorizer,
            run_id=run_id,
            action="read",
        )

    except Exception as exc:
        handle_error(exc)


# ============================================================
# Run Control
# ============================================================

@router.post(
    "/runs/{run_id}/control",
    response_model=SimulationControlResponse,
)
async def control_simulation_run(
    run_id: str,
    payload: SimulationControlRequest,
    tenant: CurrentTenant,
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
    audit: SimulationAuditSink = Depends(get_audit),
):
    await authorize(tenant, authorizer, CONTROL)

    try:
        await require_run(
            tenant=tenant,
            service=service,
            authorizer=authorizer,
            run_id=run_id,
            action="control",
        )

        # Worker must atomically validate transitions:
        # RUNNING -> PAUSED
        # PAUSED -> RUNNING
        # QUEUED/RUNNING/PAUSED -> CANCELLED
        #
        # These are virtual simulation operations only.

        result = await service.control_run(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            run_id=run_id,
            payload=payload,
        )

        if (
            result.run_id != run_id
            or result.action != payload.action
        ):
            raise SimulationForbidden()

        await audit_action(
            audit,
            tenant,
            "simulation.run.control",
            run_id,
            {"action": payload.action.value},
        )

        return result

    except Exception as exc:
        handle_error(exc)


# ============================================================
# Run Telemetry
# ============================================================

@router.get(
    "/runs/{run_id}/telemetry",
    response_model=SimulationTelemetrySnapshot,
)
async def get_simulation_telemetry(
    run_id: str,
    tenant: CurrentTenant,
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
):
    await authorize(tenant, authorizer, READ)

    try:
        await require_run(
            tenant=tenant,
            service=service,
            authorizer=authorizer,
            run_id=run_id,
            action="read",
        )

        result = await service.get_telemetry(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            run_id=run_id,
        )

        verify_tenant(tenant, result.tenant_id)

        if result.run_id != run_id:
            raise SimulationForbidden()

        return result

    except Exception as exc:
        handle_error(exc)


# ============================================================
# Run Events
# ============================================================

@router.get(
    "/runs/{run_id}/events",
    response_model=SimulationEventList,
)
async def get_simulation_events(
    run_id: str,
    tenant: CurrentTenant,
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
):
    await authorize(tenant, authorizer, READ)

    try:
        await require_run(
            tenant=tenant,
            service=service,
            authorizer=authorizer,
            run_id=run_id,
            action="read",
        )

        result = await service.get_events(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            run_id=run_id,
        )

        if result.run_id != run_id:
            raise SimulationForbidden()

        for event in result.items:
            if event.run_id != run_id:
                raise SimulationForbidden()

        return result

    except Exception as exc:
        handle_error(exc)


# ============================================================
# Run Results
# ============================================================

@router.get(
    "/runs/{run_id}/results",
    response_model=SimulationResult,
)
async def get_simulation_result(
    run_id: str,
    tenant: CurrentTenant,
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
):
    await authorize(tenant, authorizer, READ)

    try:
        run = await require_run(
            tenant=tenant,
            service=service,
            authorizer=authorizer,
            run_id=run_id,
            action="read",
        )

        if run.status not in {
            SimulationStatus.COMPLETED,
            SimulationStatus.FAILED,
            SimulationStatus.CANCELLED,
        }:
            raise SimulationConflict(
                "Simulation results are not finalized"
            )

        result = await service.get_result(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            run_id=run_id,
        )

        verify_tenant(tenant, result.tenant_id)

        if (
            result.run_id != run_id
            or result.scenario_id != run.scenario_id
        ):
            raise SimulationForbidden()

        return result

    except Exception as exc:
        handle_error(exc)


# ============================================================
# Replay
# ============================================================

@router.get(
    "/runs/{run_id}/replay",
    response_model=SimulationReplay,
)
async def get_simulation_replay(
    run_id: str,
    tenant: CurrentTenant,
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
):
    await authorize(tenant, authorizer, READ)

    try:
        await require_run(
            tenant=tenant,
            service=service,
            authorizer=authorizer,
            run_id=run_id,
            action="read",
        )

        result = await service.get_replay(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            run_id=run_id,
        )

        verify_tenant(tenant, result.tenant_id)

        if result.run_id != run_id:
            raise SimulationForbidden()

        return result

    except Exception as exc:
        handle_error(exc)


# ============================================================
# Comparison
# ============================================================

@router.post(
    "/compare",
    response_model=SimulationComparisonResult,
)
async def compare_simulation_runs(
    payload: SimulationComparisonRequest,
    tenant: CurrentTenant,
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
):
    await authorize(tenant, authorizer, READ)

    for run_id in payload.run_ids:
        await require_run(
            tenant=tenant,
            service=service,
            authorizer=authorizer,
            run_id=run_id,
            action="read",
        )

    try:
        result = await service.compare_runs(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            payload=payload,
        )

        verify_tenant(tenant, result.tenant_id)

        if set(result.run_ids) != set(payload.run_ids):
            raise SimulationForbidden()

        return result

    except Exception as exc:
        handle_error(exc)


# ============================================================
# Evaluation
# ============================================================

@router.post(
    "/evaluate",
    response_model=SimulationEvaluationResult,
)
async def evaluate_simulation_run(
    payload: SimulationEvaluationRequest,
    tenant: CurrentTenant,
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
    audit: SimulationAuditSink = Depends(get_audit),
):
    await authorize(tenant, authorizer, EVALUATE)

    try:
        run = await require_run(
            tenant=tenant,
            service=service,
            authorizer=authorizer,
            run_id=payload.run_id,
            action="evaluate",
        )

        if run.status != SimulationStatus.COMPLETED:
            raise SimulationConflict(
                "Only completed runs can be evaluated"
            )

        result = await service.evaluate_run(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
            payload=payload,
        )

        verify_tenant(tenant, result.tenant_id)

        if result.run_id != payload.run_id:
            raise SimulationForbidden()

        await audit_action(
            audit,
            tenant,
            "simulation.evaluate",
            result.evaluation_id,
            {"run_id": payload.run_id},
        )

        return result

    except Exception as exc:
        handle_error(exc)


# ============================================================
# Statistics
# ============================================================

@router.get(
    "/statistics",
    response_model=SimulationStatistics,
)
async def simulation_statistics(
    tenant: CurrentTenant,
    service: SimulationService = Depends(get_service),
    authorizer: SimulationAuthorizer = Depends(
        get_authorizer
    ),
):
    await authorize(tenant, authorizer, ANALYTICS)

    try:
        result = await service.get_statistics(
            tenant_id=tenant.tenant_id,
            user_id=tenant.user_id,
        )

        verify_tenant(tenant, result.tenant_id)

        return result

    except Exception as exc:
        handle_error(exc)


__all__ = [
    "router",
    "SimulationEngine",
    "EnvironmentType",
    "SimulationStatus",
    "SimulationAction",
    "SimulationTaskType",
    "VirtualSensorType",
    "SimulationEventType",
    "EvaluationStatus",
    "Vector3",
    "Quaternion",
    "Pose3D",
    "PhysicsConfiguration",
    "SimulationEnvironmentCreate",
    "SimulationEnvironment",
    "SimulationEnvironmentList",
    "VirtualSensorConfiguration",
    "VirtualRobotConfiguration",
    "SimulationScenarioCreate",
    "SimulationScenario",
    "SimulationScenarioList",
    "SimulationRunCreate",
    "SimulationRun",
    "SimulationRunList",
    "SimulationControlRequest",
    "SimulationControlResponse",
    "SimulationTelemetrySnapshot",
    "SimulationEvent",
    "SimulationEventList",
    "SimulationMetrics",
    "SimulationResult",
    "SimulationReplay",
    "SimulationComparisonRequest",
    "SimulationComparisonResult",
    "SimulationEvaluationRequest",
    "SimulationEvaluationResult",
    "SimulationStatistics",
    "SimulationService",
    "SimulationAuthorizer",
    "SimulationAuditSink",
    "configure_simulation_api",
]
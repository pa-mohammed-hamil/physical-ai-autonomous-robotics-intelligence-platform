"""Fleet Agent module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/fleet_agent.py

Physical AI & Autonomous Robotics Intelligence Platform
Autonomous Robot Fleet Intelligence Agent

Features:
- Multi-robot fleet monitoring
- Robot availability and health assessment
- Robot-to-task matching
- Capability-aware task allocation
- Fleet workload balancing
- Battery-aware scheduling
- Maintenance-aware scheduling
- Robot utilization analysis
- Fleet anomaly detection
- Mission planning recommendations
- Fleet performance analytics
- Fleet operational cost estimation
- Offline robot detection
- Human-reviewed fleet recommendations
- Multi-tenant isolation
- Audit logging

Safety:
- This agent generates recommendations only.
- It does not directly control robots.
- Task allocation is advisory until approved.
- Safety-critical robots require human review.
- Hardware emergency-stop systems remain independent.

Python 3.11+
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import re
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

MAX_ROBOTS_PER_FLEET = 10_000
MAX_TASKS_PER_REQUEST = 500
MAX_CAPABILITIES = 100
MAX_METADATA_FIELDS = 50
MAX_RECOMMENDATIONS = 500

DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 120

MONEY_QUANTUM = Decimal("0.01")

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)


# ============================================================
# Utilities
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def validate_identifier(value: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError("Invalid identifier")
    return value


def validate_score(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Score must be finite")

    if not 0 <= value <= 1:
        raise ValueError("Score must be between 0 and 1")

    return value


def validate_nonnegative(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "Value must be finite and nonnegative"
        )

    return value


def validate_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError(
            "Datetime must be timezone-aware"
        )

    return value


def money(value: Decimal | int | str) -> Decimal:
    if isinstance(value, float):
        raise TypeError(
            "Use Decimal or string for money"
        )

    result = Decimal(value)

    if not result.is_finite():
        raise ValueError(
            "Monetary value must be finite"
        )

    return result.quantize(
        MONEY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


# ============================================================
# Enumerations
# ============================================================

class RobotType(str, Enum):
    MOBILE_ROBOT = "mobile_robot"
    AMR = "amr"
    AGV = "agv"
    INDUSTRIAL_ARM = "industrial_arm"
    COBOT = "cobot"
    HUMANOID = "humanoid"
    QUADRUPED = "quadruped"
    DRONE = "drone"
    INSPECTION_ROBOT = "inspection_robot"
    WAREHOUSE_ROBOT = "warehouse_robot"
    SERVICE_ROBOT = "service_robot"
    CUSTOM = "custom"


class RobotOperationalStatus(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    CHARGING = "charging"
    PAUSED = "paused"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    EMERGENCY_STOP = "emergency_stop"
    OFFLINE = "offline"


class RobotHealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RobotCapability(str, Enum):
    NAVIGATION = "navigation"
    MAPPING = "mapping"
    SLAM = "slam"
    OBJECT_DETECTION = "object_detection"
    OBJECT_TRACKING = "object_tracking"
    MANIPULATION = "manipulation"
    GRASPING = "grasping"
    PICK_AND_PLACE = "pick_and_place"
    INSPECTION = "inspection"
    DELIVERY = "delivery"
    TRANSPORT = "transport"
    WELDING = "welding"
    ASSEMBLY = "assembly"
    CLEANING = "cleaning"
    INVENTORY = "inventory"
    HUMAN_INTERACTION = "human_interaction"
    VLA = "vla"
    CUSTOM = "custom"


class FleetTaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class FleetTaskStatus(str, Enum):
    PENDING = "pending"
    PLANNED = "planned"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FleetRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class FleetRecommendationType(str, Enum):
    TASK_ASSIGNMENT = "task_assignment"
    CHARGING = "charging"
    MAINTENANCE = "maintenance"
    WORKLOAD_BALANCING = "workload_balancing"
    HEALTH_INSPECTION = "health_inspection"
    SAFETY_REVIEW = "safety_review"
    FLEET_OPTIMIZATION = "fleet_optimization"


class FleetRecommendationStatus(str, Enum):
    PROPOSED = "proposed"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    REJECTED = "rejected"


class FleetOptimizationStrategy(str, Enum):
    BALANCED = "balanced"
    MINIMUM_COST = "minimum_cost"
    MAXIMUM_AVAILABILITY = "maximum_availability"
    BATTERY_EFFICIENT = "battery_efficient"
    MINIMUM_UTILIZATION = "minimum_utilization"


# ============================================================
# Exceptions
# ============================================================

class FleetAgentError(Exception):
    pass


class FleetValidationError(FleetAgentError):
    pass


class FleetRobotNotFoundError(FleetAgentError):
    pass


class FleetTaskNotFoundError(FleetAgentError):
    pass


class FleetUnavailableError(FleetAgentError):
    pass


class FleetConflictError(FleetAgentError):
    pass


class FleetPermissionError(FleetAgentError):
    pass


# ============================================================
# Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class FleetAgentScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Geometry
# ============================================================

@dataclass(slots=True)
class FleetPosition:
    x: float
    y: float
    z: float = 0.0

    frame_id: str = "map"

    def __post_init__(self) -> None:
        for value in (self.x, self.y, self.z):
            if not math.isfinite(value):
                raise FleetValidationError(
                    "Position coordinates must be finite"
                )

        if not self.frame_id.strip():
            raise FleetValidationError(
                "Coordinate frame is required"
            )

    def distance_to(
        self,
        other: "FleetPosition",
    ) -> float:
        if self.frame_id != other.frame_id:
            raise FleetValidationError(
                "Cannot compare different coordinate frames"
            )

        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )


# ============================================================
# Robot Models
# ============================================================

@dataclass(slots=True)
class FleetRobot:
    robot_id: str
    tenant_id: str
    fleet_id: str

    name: str
    robot_type: RobotType

    capabilities: list[RobotCapability]

    operational_status: RobotOperationalStatus
    health_status: RobotHealthStatus

    battery_percentage: float = 100.0

    position: FleetPosition | None = None

    maximum_payload_kg: float = 0.0

    maximum_concurrent_tasks: int = 1
    active_tasks: int = 0

    utilization_percentage: float = 0.0

    operating_cost_per_hour: Decimal = Decimal("0")

    maintenance_required: bool = False
    safety_review_required: bool = False

    enabled: bool = True

    last_heartbeat_at: datetime | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(self.robot_id)
        validate_identifier(self.tenant_id)
        validate_identifier(self.fleet_id)

        if not self.name.strip():
            raise FleetValidationError(
                "Robot name is required"
            )

        if len(self.capabilities) > MAX_CAPABILITIES:
            raise FleetValidationError(
                "Too many robot capabilities"
            )

        if not 0 <= self.battery_percentage <= 100:
            raise FleetValidationError(
                "Battery percentage must be 0–100"
            )

        validate_nonnegative(self.maximum_payload_kg)

        if self.maximum_concurrent_tasks < 1:
            raise FleetValidationError(
                "Maximum concurrent tasks must be positive"
            )

        if not (
            0 <= self.active_tasks
            <= self.maximum_concurrent_tasks
        ):
            raise FleetValidationError(
                "Invalid active task count"
            )

        if not 0 <= self.utilization_percentage <= 100:
            raise FleetValidationError(
                "Utilization percentage must be 0–100"
            )

        self.operating_cost_per_hour = money(
            self.operating_cost_per_hour
        )

        if self.operating_cost_per_hour < 0:
            raise FleetValidationError(
                "Operating cost cannot be negative"
            )

        if len(self.metadata) > MAX_METADATA_FIELDS:
            raise FleetValidationError(
                "Too many metadata fields"
            )

        if self.last_heartbeat_at is not None:
            validate_timestamp(self.last_heartbeat_at)

    @property
    def available_task_capacity(self) -> int:
        return max(
            0,
            self.maximum_concurrent_tasks
            - self.active_tasks,
        )


# ============================================================
# Fleet Task Models
# ============================================================

@dataclass(slots=True)
class FleetTask:
    task_id: str
    tenant_id: str
    fleet_id: str

    name: str
    description: str

    required_capabilities: list[RobotCapability]

    priority: FleetTaskPriority = (
        FleetTaskPriority.NORMAL
    )

    status: FleetTaskStatus = (
        FleetTaskStatus.PENDING
    )

    required_payload_kg: float = 0.0

    estimated_duration_minutes: float = 30.0

    minimum_battery_percentage: float = 20.0

    target_position: FleetPosition | None = None

    assigned_robot_id: str | None = None

    safety_critical: bool = False

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(self.task_id)
        validate_identifier(self.tenant_id)
        validate_identifier(self.fleet_id)

        if not self.name.strip():
            raise FleetValidationError(
                "Task name is required"
            )

        if len(self.required_capabilities) > MAX_CAPABILITIES:
            raise FleetValidationError(
                "Too many required capabilities"
            )

        validate_nonnegative(self.required_payload_kg)

        if (
            not math.isfinite(self.estimated_duration_minutes)
            or self.estimated_duration_minutes <= 0
        ):
            raise FleetValidationError(
                "Estimated duration must be positive"
            )

        if not 0 <= self.minimum_battery_percentage <= 100:
            raise FleetValidationError(
                "Minimum battery percentage must be 0–100"
            )

        if self.assigned_robot_id is not None:
            validate_identifier(self.assigned_robot_id)


# ============================================================
# Fleet Recommendation Models
# ============================================================

@dataclass(slots=True)
class FleetRecommendation:
    recommendation_id: str
    tenant_id: str
    fleet_id: str

    recommendation_type: FleetRecommendationType

    title: str
    description: str

    robot_id: str | None = None
    task_id: str | None = None

    confidence: float = 1.0

    risk_level: FleetRiskLevel = FleetRiskLevel.LOW

    requires_human_review: bool = False

    status: FleetRecommendationStatus = (
        FleetRecommendationStatus.PROPOSED
    )

    estimated_cost: Decimal | None = None

    reasoning: list[str] = field(
        default_factory=list
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(self.recommendation_id)
        validate_identifier(self.tenant_id)
        validate_identifier(self.fleet_id)

        validate_score(self.confidence)

        if self.robot_id is not None:
            validate_identifier(self.robot_id)

        if self.task_id is not None:
            validate_identifier(self.task_id)

        if self.estimated_cost is not None:
            self.estimated_cost = money(
                self.estimated_cost
            )

            if self.estimated_cost < 0:
                raise FleetValidationError(
                    "Estimated cost cannot be negative"
                )


@dataclass(slots=True)
class FleetAssignmentResult:
    task_id: str
    robot_id: str | None

    matched: bool
    score: float

    estimated_operating_cost: Decimal | None

    recommendation: FleetRecommendation | None

    reasons: list[str] = field(
        default_factory=list
    )


@dataclass(slots=True)
class FleetOptimizationResult:
    optimization_id: str
    tenant_id: str
    fleet_id: str

    strategy: FleetOptimizationStrategy

    assignments: list[FleetAssignmentResult]

    unassigned_task_ids: list[str]

    estimated_total_cost: Decimal

    requires_human_review: bool

    generated_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(slots=True)
class FleetHealthSummary:
    tenant_id: str
    fleet_id: str

    total_robots: int

    healthy_robots: int
    warning_robots: int
    degraded_robots: int
    critical_robots: int
    unknown_health_robots: int

    offline_robots: int
    working_robots: int
    idle_robots: int
    charging_robots: int

    maintenance_required_count: int

    average_battery_percentage: float
    average_utilization_percentage: float

    fleet_health_score: float

    generated_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(slots=True)
class FleetCostSummary:
    tenant_id: str
    fleet_id: str

    total_hourly_operating_cost: Decimal

    active_hourly_operating_cost: Decimal

    estimated_daily_cost: Decimal

    estimated_monthly_cost: Decimal

    generated_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(slots=True)
class FleetAlert:
    alert_id: str
    tenant_id: str
    fleet_id: str

    robot_id: str

    severity: FleetRiskLevel

    title: str
    description: str

    created_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# Repository Protocols
# ============================================================

class FleetAgentRepository(Protocol):

    async def list_robots(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
    ) -> list[FleetRobot]:
        ...

    async def get_robot(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
    ) -> FleetRobot | None:
        ...

    async def list_tasks(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
    ) -> list[FleetTask]:
        ...

    async def save_recommendation(
        self,
        *,
        recommendation: FleetRecommendation,
    ) -> None:
        ...

    async def save_optimization(
        self,
        *,
        optimization: FleetOptimizationResult,
    ) -> None:
        ...


class FleetAuditSink(Protocol):

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


class FleetAuthorizer(Protocol):

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


# ============================================================
# Fleet Intelligence Agent
# ============================================================

class FleetAgent:

    def __init__(
        self,
        *,
        repository: FleetAgentRepository,
        authorizer: FleetAuthorizer,
        audit_sink: FleetAuditSink | None = None,
        heartbeat_timeout_seconds: int = (
            DEFAULT_HEARTBEAT_TIMEOUT_SECONDS
        ),
    ) -> None:

        if heartbeat_timeout_seconds <= 0:
            raise ValueError(
                "Heartbeat timeout must be positive"
            )

        self.repository = repository
        self.authorizer = authorizer
        self.audit_sink = audit_sink

        self.heartbeat_timeout_seconds = (
            heartbeat_timeout_seconds
        )

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    async def _authorize(
        self,
        *,
        scope: FleetAgentScope,
        fleet_id: str,
        action: str,
    ) -> None:

        validate_identifier(fleet_id)

        await self.authorizer.require_access(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            resource_type="fleet",
            resource_id=fleet_id,
            action=action,
        )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    async def _audit(
        self,
        *,
        scope: FleetAgentScope,
        action: str,
        resource_id: str,
        metadata: Mapping[str, Any],
    ) -> None:

        if self.audit_sink is None:
            return

        await self.audit_sink.record(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            action=action,
            resource_id=resource_id,
            metadata=metadata,
        )

    # --------------------------------------------------------
    # Load Fleet Robots
    # --------------------------------------------------------

    async def get_fleet_robots(
        self,
        *,
        scope: FleetAgentScope,
        fleet_id: str,
    ) -> list[FleetRobot]:

        await self._authorize(
            scope=scope,
            fleet_id=fleet_id,
            action="read",
        )

        robots = await self.repository.list_robots(
            tenant_id=scope.tenant_id,
            fleet_id=fleet_id,
        )

        if len(robots) > MAX_ROBOTS_PER_FLEET:
            raise FleetValidationError(
                "Fleet exceeds supported robot count"
            )

        for robot in robots:
            if (
                robot.tenant_id != scope.tenant_id
                or robot.fleet_id != fleet_id
            ):
                raise FleetPermissionError(
                    "Cross-tenant fleet data detected"
                )

        return robots

    # --------------------------------------------------------
    # Heartbeat
    # --------------------------------------------------------

    def _heartbeat_fresh(
        self,
        robot: FleetRobot,
    ) -> bool:

        if robot.last_heartbeat_at is None:
            return False

        elapsed = (
            utc_now() - robot.last_heartbeat_at
        ).total_seconds()

        return (
            0 <= elapsed
            <= self.heartbeat_timeout_seconds
        )

    # --------------------------------------------------------
    # Robot Availability
    # --------------------------------------------------------

    def is_robot_available(
        self,
        robot: FleetRobot,
        *,
        minimum_battery_percentage: float = 20.0,
    ) -> bool:

        if not robot.enabled:
            return False

        if robot.operational_status not in (
            RobotOperationalStatus.IDLE,
            RobotOperationalStatus.WORKING,
        ):
            return False

        if robot.health_status != RobotHealthStatus.HEALTHY:
            return False

        if robot.maintenance_required:
            return False

        if robot.safety_review_required:
            return False

        if not self._heartbeat_fresh(robot):
            return False

        if (
            robot.battery_percentage
            < minimum_battery_percentage
        ):
            return False

        if robot.available_task_capacity <= 0:
            return False

        return True

    # --------------------------------------------------------
    # Robot Task Compatibility
    # --------------------------------------------------------

    def check_task_compatibility(
        self,
        *,
        robot: FleetRobot,
        task: FleetTask,
    ) -> tuple[bool, list[str]]:

        reasons: list[str] = []

        if robot.tenant_id != task.tenant_id:
            reasons.append("Tenant mismatch")

        if robot.fleet_id != task.fleet_id:
            reasons.append("Fleet mismatch")

        if not self.is_robot_available(
            robot,
            minimum_battery_percentage=(
                task.minimum_battery_percentage
            ),
        ):
            reasons.append(
                "Robot is not operationally available"
            )

        missing_capabilities = (
            set(task.required_capabilities)
            - set(robot.capabilities)
        )

        if missing_capabilities:
            reasons.append(
                "Missing capabilities: "
                + ", ".join(
                    sorted(
                        capability.value
                        for capability
                        in missing_capabilities
                    )
                )
            )

        if (
            task.required_payload_kg
            > robot.maximum_payload_kg
        ):
            reasons.append(
                "Required payload exceeds robot capacity"
            )

        if (
            task.target_position is not None
            and robot.position is not None
            and task.target_position.frame_id
            != robot.position.frame_id
        ):
            reasons.append(
                "Robot and task positions use "
                "different coordinate frames"
            )

        if task.safety_critical:
            reasons.append(
                "Safety-critical task requires human review"
            )

        return (
            len([
                reason
                for reason in reasons
                if reason
                != "Safety-critical task requires human review"
            ]) == 0,
            reasons,
        )

    # --------------------------------------------------------
    # Robot Scoring
    # --------------------------------------------------------

    def score_robot_for_task(
        self,
        *,
        robot: FleetRobot,
        task: FleetTask,
        strategy: FleetOptimizationStrategy,
    ) -> float:

        compatible, _ = self.check_task_compatibility(
            robot=robot,
            task=task,
        )

        if not compatible:
            return 0.0

        battery_score = (
            robot.battery_percentage / 100
        )

        utilization_score = (
            1.0
            - robot.utilization_percentage / 100
        )

        capacity_score = (
            robot.available_task_capacity
            / robot.maximum_concurrent_tasks
        )

        distance_score = 0.5

        if (
            robot.position is not None
            and task.target_position is not None
        ):
            distance = robot.position.distance_to(
                task.target_position
            )

            distance_score = 1.0 / (
                1.0 + distance / 100.0
            )

        hourly_cost = float(
            robot.operating_cost_per_hour
        )

        cost_score = 1.0 / (
            1.0 + hourly_cost / 100.0
        )

        if strategy == FleetOptimizationStrategy.MINIMUM_COST:
            score = (
                0.55 * cost_score
                + 0.20 * battery_score
                + 0.15 * distance_score
                + 0.10 * capacity_score
            )

        elif (
            strategy
            == FleetOptimizationStrategy.BATTERY_EFFICIENT
        ):
            score = (
                0.55 * battery_score
                + 0.20 * distance_score
                + 0.15 * utilization_score
                + 0.10 * capacity_score
            )

        elif (
            strategy
            == FleetOptimizationStrategy.MINIMUM_UTILIZATION
        ):
            score = (
                0.60 * utilization_score
                + 0.20 * capacity_score
                + 0.10 * battery_score
                + 0.10 * distance_score
            )

        elif (
            strategy
            == FleetOptimizationStrategy.MAXIMUM_AVAILABILITY
        ):
            score = (
                0.45 * capacity_score
                + 0.30 * battery_score
                + 0.15 * utilization_score
                + 0.10 * distance_score
            )

        else:
            score = (
                0.25 * battery_score
                + 0.25 * utilization_score
                + 0.20 * capacity_score
                + 0.15 * distance_score
                + 0.15 * cost_score
            )

        return round(
            max(0.0, min(1.0, score)),
            6,
        )

    # --------------------------------------------------------
    # Single Task Assignment Recommendation
    # --------------------------------------------------------

    async def recommend_robot_for_task(
        self,
        *,
        scope: FleetAgentScope,
        fleet_id: str,
        task: FleetTask,
        strategy: FleetOptimizationStrategy = (
            FleetOptimizationStrategy.BALANCED
        ),
    ) -> FleetAssignmentResult:

        await self._authorize(
            scope=scope,
            fleet_id=fleet_id,
            action="plan",
        )

        if (
            task.tenant_id != scope.tenant_id
            or task.fleet_id != fleet_id
        ):
            raise FleetPermissionError(
                "Task is outside authorized fleet"
            )

        robots = await self.get_fleet_robots(
            scope=scope,
            fleet_id=fleet_id,
        )

        candidates: list[
            tuple[float, FleetRobot]
        ] = []

        for robot in robots:
            score = self.score_robot_for_task(
                robot=robot,
                task=task,
                strategy=strategy,
            )

            if score > 0:
                candidates.append(
                    (score, robot)
                )

        if not candidates:
            return FleetAssignmentResult(
                task_id=task.task_id,
                robot_id=None,
                matched=False,
                score=0.0,
                estimated_operating_cost=None,
                recommendation=None,
                reasons=[
                    "No available robot satisfies "
                    "the task requirements"
                ],
            )

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1].battery_percentage,
                item[1].robot_id,
            ),
            reverse=True,
        )

        best_score, best_robot = candidates[0]

        estimated_cost = money(
            best_robot.operating_cost_per_hour
            * Decimal(
                str(task.estimated_duration_minutes)
            )
            / Decimal("60")
        )

        requires_review = (
            task.safety_critical
            or task.priority
            == FleetTaskPriority.CRITICAL
        )

        recommendation = FleetRecommendation(
            recommendation_id=generate_id(),
            tenant_id=scope.tenant_id,
            fleet_id=fleet_id,
            recommendation_type=(
                FleetRecommendationType.TASK_ASSIGNMENT
            ),
            title="Robot task assignment recommendation",
            description=(
                f"Consider robot {best_robot.robot_id} "
                f"for task {task.task_id}."
            ),
            robot_id=best_robot.robot_id,
            task_id=task.task_id,
            confidence=best_score,
            risk_level=(
                FleetRiskLevel.HIGH
                if requires_review
                else FleetRiskLevel.LOW
            ),
            requires_human_review=requires_review,
            status=(
                FleetRecommendationStatus.REVIEW_REQUIRED
                if requires_review
                else FleetRecommendationStatus.PROPOSED
            ),
            estimated_cost=estimated_cost,
            reasoning=[
                "Required capabilities are satisfied",
                "Robot health and battery meet "
                "the configured thresholds",
                "Robot has available task capacity",
                f"Selection strategy: {strategy.value}",
                "Assignment is advisory and "
                "does not dispatch the robot",
            ],
        )

        await self.repository.save_recommendation(
            recommendation=recommendation
        )

        await self._audit(
            scope=scope,
            action="fleet_agent.recommend_assignment",
            resource_id=recommendation.recommendation_id,
            metadata={
                "fleet_id": fleet_id,
                "task_id": task.task_id,
                "robot_id": best_robot.robot_id,
            },
        )

        return FleetAssignmentResult(
            task_id=task.task_id,
            robot_id=best_robot.robot_id,
            matched=True,
            score=best_score,
            estimated_operating_cost=estimated_cost,
            recommendation=recommendation,
            reasons=recommendation.reasoning,
        )

    # --------------------------------------------------------
    # Multi-Task Fleet Optimization
    # --------------------------------------------------------

    async def optimize_fleet_tasks(
        self,
        *,
        scope: FleetAgentScope,
        fleet_id: str,
        tasks: Sequence[FleetTask],
        strategy: FleetOptimizationStrategy = (
            FleetOptimizationStrategy.BALANCED
        ),
    ) -> FleetOptimizationResult:

        await self._authorize(
            scope=scope,
            fleet_id=fleet_id,
            action="plan",
        )

        if len(tasks) > MAX_TASKS_PER_REQUEST:
            raise FleetValidationError(
                "Too many tasks in optimization request"
            )

        task_ids = [task.task_id for task in tasks]

        if len(set(task_ids)) != len(task_ids):
            raise FleetValidationError(
                "Duplicate task IDs"
            )

        for task in tasks:
            if (
                task.tenant_id != scope.tenant_id
                or task.fleet_id != fleet_id
            ):
                raise FleetPermissionError(
                    "Task is outside authorized fleet"
                )

        robots = await self.get_fleet_robots(
            scope=scope,
            fleet_id=fleet_id,
        )

        # Work on copies to avoid mutating live robot state.
        simulated_robots = {
            robot.robot_id: copy.deepcopy(robot)
            for robot in robots
        }

        priority_order = {
            FleetTaskPriority.CRITICAL: 4,
            FleetTaskPriority.HIGH: 3,
            FleetTaskPriority.NORMAL: 2,
            FleetTaskPriority.LOW: 1,
        }

        ordered_tasks = sorted(
            tasks,
            key=lambda task: (
                -priority_order[task.priority],
                task.created_at,
                task.task_id,
            ),
        )

        assignments: list[FleetAssignmentResult] = []
        unassigned: list[str] = []

        total_cost = Decimal("0")
        requires_review = False

        for task in ordered_tasks:
            candidates = []

            for robot in simulated_robots.values():
                score = self.score_robot_for_task(
                    robot=robot,
                    task=task,
                    strategy=strategy,
                )

                if score > 0:
                    candidates.append(
                        (score, robot)
                    )

            if not candidates:
                unassigned.append(task.task_id)

                assignments.append(
                    FleetAssignmentResult(
                        task_id=task.task_id,
                        robot_id=None,
                        matched=False,
                        score=0.0,
                        estimated_operating_cost=None,
                        recommendation=None,
                        reasons=[
                            "No compatible robot with "
                            "remaining simulated capacity"
                        ],
                    )
                )

                continue

            candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1].battery_percentage,
                    item[1].robot_id,
                ),
                reverse=True,
            )

            score, robot = candidates[0]

            robot.active_tasks += 1
            robot.operational_status = (
                RobotOperationalStatus.WORKING
            )

            estimated_cost = money(
                robot.operating_cost_per_hour
                * Decimal(
                    str(task.estimated_duration_minutes)
                )
                / Decimal("60")
            )

            total_cost += estimated_cost

            task_review = (
                task.safety_critical
                or task.priority
                == FleetTaskPriority.CRITICAL
            )

            requires_review = (
                requires_review or task_review
            )

            recommendation = FleetRecommendation(
                recommendation_id=generate_id(),
                tenant_id=scope.tenant_id,
                fleet_id=fleet_id,
                recommendation_type=(
                    FleetRecommendationType.TASK_ASSIGNMENT
                ),
                title="Proposed fleet task assignment",
                description=(
                    f"Proposed robot {robot.robot_id} "
                    f"for task {task.task_id}."
                ),
                robot_id=robot.robot_id,
                task_id=task.task_id,
                confidence=score,
                risk_level=(
                    FleetRiskLevel.HIGH
                    if task_review
                    else FleetRiskLevel.LOW
                ),
                requires_human_review=task_review,
                status=(
                    FleetRecommendationStatus.REVIEW_REQUIRED
                    if task_review
                    else FleetRecommendationStatus.PROPOSED
                ),
                estimated_cost=estimated_cost,
                reasoning=[
                    "Capability-aware robot matching",
                    "Battery and health screening",
                    "Simulated task-capacity reservation",
                    "No physical task dispatch performed",
                ],
            )

            await self.repository.save_recommendation(
                recommendation=recommendation
            )

            assignments.append(
                FleetAssignmentResult(
                    task_id=task.task_id,
                    robot_id=robot.robot_id,
                    matched=True,
                    score=score,
                    estimated_operating_cost=estimated_cost,
                    recommendation=recommendation,
                    reasons=recommendation.reasoning,
                )
            )

        result = FleetOptimizationResult(
            optimization_id=generate_id(),
            tenant_id=scope.tenant_id,
            fleet_id=fleet_id,
            strategy=strategy,
            assignments=assignments,
            unassigned_task_ids=unassigned,
            estimated_total_cost=money(total_cost),
            requires_human_review=requires_review,
        )

        await self.repository.save_optimization(
            optimization=result
        )

        await self._audit(
            scope=scope,
            action="fleet_agent.optimize_tasks",
            resource_id=result.optimization_id,
            metadata={
                "fleet_id": fleet_id,
                "strategy": strategy.value,
                "task_count": len(tasks),
                "unassigned_count": len(unassigned),
            },
        )

        return result

    # --------------------------------------------------------
    # Fleet Health Analysis
    # --------------------------------------------------------

    async def analyze_fleet_health(
        self,
        *,
        scope: FleetAgentScope,
        fleet_id: str,
    ) -> FleetHealthSummary:

        robots = await self.get_fleet_robots(
            scope=scope,
            fleet_id=fleet_id,
        )

        total = len(robots)

        healthy = sum(
            robot.health_status
            == RobotHealthStatus.HEALTHY
            for robot in robots
        )

        warning = sum(
            robot.health_status
            == RobotHealthStatus.WARNING
            for robot in robots
        )

        degraded = sum(
            robot.health_status
            == RobotHealthStatus.DEGRADED
            for robot in robots
        )

        critical = sum(
            robot.health_status
            == RobotHealthStatus.CRITICAL
            for robot in robots
        )

        unknown = sum(
            robot.health_status
            == RobotHealthStatus.UNKNOWN
            for robot in robots
        )

        offline = sum(
            not self._heartbeat_fresh(robot)
            or robot.operational_status
            == RobotOperationalStatus.OFFLINE
            for robot in robots
        )

        working = sum(
            robot.operational_status
            == RobotOperationalStatus.WORKING
            for robot in robots
        )

        idle = sum(
            robot.operational_status
            == RobotOperationalStatus.IDLE
            for robot in robots
        )

        charging = sum(
            robot.operational_status
            == RobotOperationalStatus.CHARGING
            for robot in robots
        )

        maintenance = sum(
            robot.maintenance_required
            for robot in robots
        )

        average_battery = (
            sum(
                robot.battery_percentage
                for robot in robots
            ) / total
            if total
            else 0.0
        )

        average_utilization = (
            sum(
                robot.utilization_percentage
                for robot in robots
            ) / total
            if total
            else 0.0
        )

        health_weights = {
            RobotHealthStatus.HEALTHY: 1.0,
            RobotHealthStatus.WARNING: 0.75,
            RobotHealthStatus.DEGRADED: 0.4,
            RobotHealthStatus.CRITICAL: 0.0,
            RobotHealthStatus.UNKNOWN: 0.0,
        }

        fleet_health_score = (
            sum(
                (
                    health_weights[robot.health_status]
                    if self._heartbeat_fresh(robot)
                    else 0.0
                )
                for robot in robots
            ) / total
            if total
            else 0.0
        )

        return FleetHealthSummary(
            tenant_id=scope.tenant_id,
            fleet_id=fleet_id,
            total_robots=total,
            healthy_robots=healthy,
            warning_robots=warning,
            degraded_robots=degraded,
            critical_robots=critical,
            unknown_health_robots=unknown,
            offline_robots=offline,
            working_robots=working,
            idle_robots=idle,
            charging_robots=charging,
            maintenance_required_count=maintenance,
            average_battery_percentage=round(
                average_battery,
                2,
            ),
            average_utilization_percentage=round(
                average_utilization,
                2,
            ),
            fleet_health_score=round(
                fleet_health_score,
                6,
            ),
        )

    # --------------------------------------------------------
    # Fleet Alert Detection
    # --------------------------------------------------------

    async def detect_fleet_alerts(
        self,
        *,
        scope: FleetAgentScope,
        fleet_id: str,
    ) -> list[FleetAlert]:

        robots = await self.get_fleet_robots(
            scope=scope,
            fleet_id=fleet_id,
        )

        alerts: list[FleetAlert] = []

        for robot in robots:

            def add_alert(
                severity: FleetRiskLevel,
                title: str,
                description: str,
            ) -> None:

                alerts.append(
                    FleetAlert(
                        alert_id=generate_id(),
                        tenant_id=scope.tenant_id,
                        fleet_id=fleet_id,
                        robot_id=robot.robot_id,
                        severity=severity,
                        title=title,
                        description=description,
                    )
                )

            if not self._heartbeat_fresh(robot):
                add_alert(
                    FleetRiskLevel.HIGH,
                    "Robot heartbeat unavailable",
                    "Robot heartbeat is missing or stale.",
                )

            if robot.battery_percentage <= 10:
                add_alert(
                    FleetRiskLevel.CRITICAL,
                    "Critical battery level",
                    "Robot battery is at or below 10%.",
                )

            elif robot.battery_percentage <= 20:
                add_alert(
                    FleetRiskLevel.MODERATE,
                    "Low battery level",
                    "Robot battery is at or below 20%.",
                )

            if robot.health_status == RobotHealthStatus.CRITICAL:
                add_alert(
                    FleetRiskLevel.CRITICAL,
                    "Critical robot health",
                    "Robot requires urgent inspection.",
                )

            if robot.maintenance_required:
                add_alert(
                    FleetRiskLevel.MODERATE,
                    "Maintenance required",
                    "Robot has a pending maintenance requirement.",
                )

            if robot.safety_review_required:
                add_alert(
                    FleetRiskLevel.HIGH,
                    "Safety review required",
                    "Robot must undergo safety review "
                    "before new assignments.",
                )

            if (
                robot.operational_status
                == RobotOperationalStatus.EMERGENCY_STOP
            ):
                add_alert(
                    FleetRiskLevel.CRITICAL,
                    "Robot emergency stop active",
                    "Emergency-stop status reported. "
                    "Follow the approved safety procedure.",
                )

        return alerts

    # --------------------------------------------------------
    # Charging Recommendations
    # --------------------------------------------------------

    async def recommend_charging(
        self,
        *,
        scope: FleetAgentScope,
        fleet_id: str,
        threshold_percentage: float = 25.0,
    ) -> list[FleetRecommendation]:

        if not 0 <= threshold_percentage <= 100:
            raise FleetValidationError(
                "Invalid charging threshold"
            )

        robots = await self.get_fleet_robots(
            scope=scope,
            fleet_id=fleet_id,
        )

        recommendations: list[FleetRecommendation] = []

        for robot in robots:

            if robot.battery_percentage > threshold_percentage:
                continue

            if (
                robot.operational_status
                == RobotOperationalStatus.CHARGING
            ):
                continue

            critical = robot.battery_percentage <= 10

            recommendation = FleetRecommendation(
                recommendation_id=generate_id(),
                tenant_id=scope.tenant_id,
                fleet_id=fleet_id,
                recommendation_type=(
                    FleetRecommendationType.CHARGING
                ),
                title="Robot charging recommendation",
                description=(
                    f"Robot {robot.robot_id} battery is "
                    f"{robot.battery_percentage:.1f}%. "
                    "Schedule charging when operationally safe."
                ),
                robot_id=robot.robot_id,
                confidence=1.0,
                risk_level=(
                    FleetRiskLevel.CRITICAL
                    if critical
                    else FleetRiskLevel.MODERATE
                ),
                requires_human_review=critical,
                status=(
                    FleetRecommendationStatus.REVIEW_REQUIRED
                    if critical
                    else FleetRecommendationStatus.PROPOSED
                ),
                reasoning=[
                    "Battery is below the configured threshold",
                    "Charging must follow approved "
                    "robot and battery procedures",
                ],
            )

            recommendations.append(recommendation)

            await self.repository.save_recommendation(
                recommendation=recommendation
            )

        return recommendations

    # --------------------------------------------------------
    # Maintenance Recommendations
    # --------------------------------------------------------

    async def recommend_maintenance(
        self,
        *,
        scope: FleetAgentScope,
        fleet_id: str,
    ) -> list[FleetRecommendation]:

        robots = await self.get_fleet_robots(
            scope=scope,
            fleet_id=fleet_id,
        )

        recommendations: list[FleetRecommendation] = []

        for robot in robots:

            needs_maintenance = (
                robot.maintenance_required
                or robot.health_status in (
                    RobotHealthStatus.DEGRADED,
                    RobotHealthStatus.CRITICAL,
                )
            )

            if not needs_maintenance:
                continue

            critical = (
                robot.health_status
                == RobotHealthStatus.CRITICAL
            )

            recommendation = FleetRecommendation(
                recommendation_id=generate_id(),
                tenant_id=scope.tenant_id,
                fleet_id=fleet_id,
                recommendation_type=(
                    FleetRecommendationType.MAINTENANCE
                ),
                title="Robot maintenance recommendation",
                description=(
                    f"Robot {robot.robot_id} requires "
                    "maintenance assessment."
                ),
                robot_id=robot.robot_id,
                confidence=1.0,
                risk_level=(
                    FleetRiskLevel.CRITICAL
                    if critical
                    else FleetRiskLevel.MODERATE
                ),
                requires_human_review=True,
                status=(
                    FleetRecommendationStatus.REVIEW_REQUIRED
                ),
                reasoning=[
                    "Robot health or maintenance status "
                    "indicates inspection is needed",
                    "A qualified technician must validate "
                    "maintenance actions",
                ],
            )

            recommendations.append(recommendation)

            await self.repository.save_recommendation(
                recommendation=recommendation
            )

        return recommendations

    # --------------------------------------------------------
    # Fleet Cost Analysis
    # --------------------------------------------------------

    async def analyze_fleet_cost(
        self,
        *,
        scope: FleetAgentScope,
        fleet_id: str,
        daily_operating_hours: Decimal = Decimal("8"),
        monthly_operating_days: Decimal = Decimal("22"),
    ) -> FleetCostSummary:

        if (
            not daily_operating_hours.is_finite()
            or not monthly_operating_days.is_finite()
            or daily_operating_hours < 0
            or daily_operating_hours > 24
            or monthly_operating_days < 0
            or monthly_operating_days > 31
        ):
            raise FleetValidationError(
                "Invalid operating schedule"
            )

        robots = await self.get_fleet_robots(
            scope=scope,
            fleet_id=fleet_id,
        )

        total_hourly_cost = money(
            sum(
                (
                    robot.operating_cost_per_hour
                    for robot in robots
                    if robot.enabled
                ),
                Decimal("0"),
            )
        )

        active_hourly_cost = money(
            sum(
                (
                    robot.operating_cost_per_hour
                    for robot in robots
                    if (
                        robot.enabled
                        and robot.operational_status
                        == RobotOperationalStatus.WORKING
                    )
                ),
                Decimal("0"),
            )
        )

        daily_cost = money(
            total_hourly_cost
            * daily_operating_hours
        )

        monthly_cost = money(
            daily_cost
            * monthly_operating_days
        )

        return FleetCostSummary(
            tenant_id=scope.tenant_id,
            fleet_id=fleet_id,
            total_hourly_operating_cost=total_hourly_cost,
            active_hourly_operating_cost=active_hourly_cost,
            estimated_daily_cost=daily_cost,
            estimated_monthly_cost=monthly_cost,
        )

    # --------------------------------------------------------
    # Fleet Summary
    # --------------------------------------------------------

    async def generate_fleet_summary(
        self,
        *,
        scope: FleetAgentScope,
        fleet_id: str,
    ) -> dict[str, Any]:

        health = await self.analyze_fleet_health(
            scope=scope,
            fleet_id=fleet_id,
        )

        alerts = await self.detect_fleet_alerts(
            scope=scope,
            fleet_id=fleet_id,
        )

        costs = await self.analyze_fleet_cost(
            scope=scope,
            fleet_id=fleet_id,
        )

        critical_alerts = sum(
            alert.severity == FleetRiskLevel.CRITICAL
            for alert in alerts
        )

        return {
            "tenant_id": scope.tenant_id,
            "fleet_id": fleet_id,
            "generated_at": utc_now().isoformat(),
            "robots": {
                "total": health.total_robots,
                "healthy": health.healthy_robots,
                "working": health.working_robots,
                "idle": health.idle_robots,
                "offline": health.offline_robots,
                "maintenance_required": (
                    health.maintenance_required_count
                ),
            },
            "health": {
                "fleet_health_score": (
                    health.fleet_health_score
                ),
                "average_battery_percentage": (
                    health.average_battery_percentage
                ),
                "average_utilization_percentage": (
                    health.average_utilization_percentage
                ),
            },
            "alerts": {
                "total": len(alerts),
                "critical": critical_alerts,
            },
            "costs": {
                "hourly": str(
                    costs.total_hourly_operating_cost
                ),
                "daily_estimate": str(
                    costs.estimated_daily_cost
                ),
                "monthly_estimate": str(
                    costs.estimated_monthly_cost
                ),
                "currency": (
                    "configured_by_fleet_repository"
                ),
            },
            "safety": {
                "autonomous_dispatch_enabled": False,
                "physical_robot_control": False,
                "recommendations_only": True,
            },
        }


# ============================================================
# Factory
# ============================================================

def create_fleet_agent(
    *,
    repository: FleetAgentRepository,
    authorizer: FleetAuthorizer,
    audit_sink: FleetAuditSink | None = None,
    heartbeat_timeout_seconds: int = (
        DEFAULT_HEARTBEAT_TIMEOUT_SECONDS
    ),
) -> FleetAgent:

    return FleetAgent(
        repository=repository,
        authorizer=authorizer,
        audit_sink=audit_sink,
        heartbeat_timeout_seconds=(
            heartbeat_timeout_seconds
        ),
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "RobotType",
    "RobotOperationalStatus",
    "RobotHealthStatus",
    "RobotCapability",
    "FleetTaskPriority",
    "FleetTaskStatus",
    "FleetRiskLevel",
    "FleetRecommendationType",
    "FleetRecommendationStatus",
    "FleetOptimizationStrategy",
    "FleetAgentScope",
    "FleetPosition",
    "FleetRobot",
    "FleetTask",
    "FleetRecommendation",
    "FleetAssignmentResult",
    "FleetOptimizationResult",
    "FleetHealthSummary",
    "FleetCostSummary",
    "FleetAlert",
    "FleetAgentRepository",
    "FleetAuditSink",
    "FleetAuthorizer",
    "FleetAgent",
    "FleetAgentError",
    "FleetValidationError",
    "FleetRobotNotFoundError",
    "FleetTaskNotFoundError",
    "FleetUnavailableError",
    "FleetConflictError",
    "FleetPermissionError",
    "create_fleet_agent",
]
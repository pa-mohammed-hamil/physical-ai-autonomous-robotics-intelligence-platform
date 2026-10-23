"""Navigation Agent module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/navigation_agent.py

Physical AI & Autonomous Robotics Intelligence Platform
Autonomous Navigation Intelligence Agent

Python 3.11+

Features:
- Autonomous robot navigation planning
- Goal-based navigation
- Global route planning
- Local trajectory planning
- SLAM and localization integration
- LiDAR and obstacle intelligence
- Dynamic obstacle detection
- Collision risk assessment
- Workspace and geofence validation
- Path optimization and waypoint generation
- Robot battery and health validation
- Navigation simulation
- Route replanning
- Multi-robot fleet awareness
- Human-in-the-loop review
- Tenant-aware authorization
- Navigation telemetry and audit logging
- Explainable navigation reports

Safety:
- Planning and simulation only.
- No direct motor or actuator commands.
- No emergency-stop override.
- No autonomous approval for physical execution.
- Physical deployment requires an independent,
  certified safety and robot-control system.
"""

from __future__ import annotations

import asyncio
import copy
import heapq
import math
import re
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


# ============================================================
# Constants
# ============================================================

MAX_ROUTE_WAYPOINTS = 5000
MAX_OBSTACLES = 2000
MAX_BATCH_REQUESTS = 50
MAX_REPLAN_ATTEMPTS = 5

DEFAULT_ROBOT_RADIUS_M = 0.35
DEFAULT_SAFETY_MARGIN_M = 0.30
DEFAULT_MAX_SPEED_MPS = 1.0
DEFAULT_GRID_RESOLUTION_M = 0.25

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
    if not isinstance(value, str):
        raise NavigationValidationError(
            "Identifier must be a string"
        )

    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise NavigationValidationError(
            f"Invalid identifier: {value!r}"
        )

    return value


def validate_finite(value: float) -> float:
    if not math.isfinite(value):
        raise NavigationValidationError(
            "Value must be finite"
        )

    return value


def validate_nonnegative(value: float) -> float:
    validate_finite(value)

    if value < 0:
        raise NavigationValidationError(
            "Value must be nonnegative"
        )

    return value


def validate_positive(value: float) -> float:
    validate_finite(value)

    if value <= 0:
        raise NavigationValidationError(
            "Value must be positive"
        )

    return value


def validate_probability(value: float) -> float:
    validate_finite(value)

    if not 0.0 <= value <= 1.0:
        raise NavigationValidationError(
            "Probability must be between 0 and 1"
        )

    return value


def validate_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise NavigationValidationError(
            "Datetime must be timezone-aware"
        )

    return value


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, value))


# ============================================================
# Exceptions
# ============================================================

class NavigationAgentError(Exception):
    pass


class NavigationValidationError(NavigationAgentError):
    pass


class NavigationNotFoundError(NavigationAgentError):
    pass


class NavigationPermissionError(NavigationAgentError):
    pass


class NavigationPlanningError(NavigationAgentError):
    pass


class NavigationSafetyError(NavigationAgentError):
    pass


class NavigationServiceUnavailableError(
    NavigationAgentError
):
    pass


# ============================================================
# Enumerations
# ============================================================

class NavigationMode(str, Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    WAREHOUSE = "warehouse"
    INDUSTRIAL = "industrial"
    AUTONOMOUS_MOBILE_ROBOT = "amr"
    DELIVERY = "delivery"
    INSPECTION = "inspection"
    SIMULATION = "simulation"


class NavigationTaskType(str, Enum):
    GO_TO_GOAL = "go_to_goal"
    FOLLOW_WAYPOINTS = "follow_waypoints"
    PATROL = "patrol"
    INSPECTION_ROUTE = "inspection_route"
    RETURN_TO_BASE = "return_to_base"
    DOCKING_APPROACH = "docking_approach"
    REPLAN = "replan"


class NavigationStatus(str, Enum):
    DRAFT = "draft"
    PLANNING = "planning"
    PLANNED = "planned"
    VALIDATION_FAILED = "validation_failed"
    SIMULATION_FAILED = "simulation_failed"
    REVIEW_REQUIRED = "review_required"
    APPROVED_FOR_SIMULATION = "approved_for_simulation"
    REJECTED = "rejected"


class NavigationRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class LocalizationStatus(str, Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    LOST = "lost"
    UNKNOWN = "unknown"


class ObstacleType(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    HUMAN = "human"
    VEHICLE = "vehicle"
    ROBOT = "robot"
    UNKNOWN = "unknown"


class CollisionStatus(str, Enum):
    CLEAR = "clear"
    COLLISION = "collision"
    UNKNOWN = "unknown"


class PathPlanningAlgorithm(str, Enum):
    ASTAR = "astar"
    DIJKSTRA = "dijkstra"
    RRT = "rrt"
    RRT_STAR = "rrt_star"
    PRM = "prm"
    HYBRID_ASTAR = "hybrid_astar"
    EXTERNAL = "external"


class NavigationSafetyDecision(str, Enum):
    ALLOW_SIMULATION = "allow_simulation"
    REVIEW_REQUIRED = "review_required"
    REJECT = "reject"


class NavigationSimulationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    NOT_RUN = "not_run"


# ============================================================
# Security Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class NavigationScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Geometry
# ============================================================

@dataclass(frozen=True, slots=True)
class Vector2:
    x: float
    y: float

    def __post_init__(self) -> None:
        validate_finite(self.x)
        validate_finite(self.y)

    def distance_to(self, other: Vector2) -> float:
        return math.hypot(
            self.x - other.x,
            self.y - other.y,
        )


@dataclass(frozen=True, slots=True)
class Pose2D:
    position: Vector2
    yaw_rad: float
    frame_id: str

    def __post_init__(self) -> None:
        validate_finite(self.yaw_rad)
        validate_identifier(self.frame_id)


@dataclass(frozen=True, slots=True)
class Velocity2D:
    linear_x_mps: float
    linear_y_mps: float
    angular_z_radps: float

    def __post_init__(self) -> None:
        validate_finite(self.linear_x_mps)
        validate_finite(self.linear_y_mps)
        validate_finite(self.angular_z_radps)


@dataclass(frozen=True, slots=True)
class NavigationBounds:
    minimum: Vector2
    maximum: Vector2

    def __post_init__(self) -> None:
        if not (
            self.minimum.x < self.maximum.x
            and self.minimum.y < self.maximum.y
        ):
            raise NavigationValidationError(
                "Invalid navigation bounds"
            )

    def contains(self, point: Vector2) -> bool:
        return (
            self.minimum.x <= point.x <= self.maximum.x
            and self.minimum.y <= point.y <= self.maximum.y
        )


# ============================================================
# Map Models
# ============================================================

@dataclass(slots=True)
class OccupancyGrid:
    map_id: str
    frame_id: str

    width: int
    height: int

    resolution_m: float

    origin: Vector2

    data: list[int]

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(self.map_id)
        validate_identifier(self.frame_id)

        if self.width <= 0 or self.height <= 0:
            raise NavigationValidationError(
                "Grid dimensions must be positive"
            )

        validate_positive(self.resolution_m)
        validate_timestamp(self.timestamp)

        if len(self.data) != self.width * self.height:
            raise NavigationValidationError(
                "Occupancy grid data size mismatch"
            )

        for value in self.data:
            if not -1 <= value <= 100:
                raise NavigationValidationError(
                    "Occupancy values must be -1 to 100"
                )

    def in_bounds(self, x: int, y: int) -> bool:
        return (
            0 <= x < self.width
            and 0 <= y < self.height
        )

    def index(self, x: int, y: int) -> int:
        return y * self.width + x

    def occupancy(self, x: int, y: int) -> int:
        if not self.in_bounds(x, y):
            return 100

        return self.data[self.index(x, y)]

    def world_to_grid(
        self,
        point: Vector2,
    ) -> tuple[int, int]:

        x = math.floor(
            (point.x - self.origin.x)
            / self.resolution_m
        )

        y = math.floor(
            (point.y - self.origin.y)
            / self.resolution_m
        )

        return x, y

    def grid_to_world(
        self,
        x: int,
        y: int,
    ) -> Vector2:

        return Vector2(
            x=self.origin.x
            + (x + 0.5) * self.resolution_m,
            y=self.origin.y
            + (y + 0.5) * self.resolution_m,
        )


# ============================================================
# Obstacles
# ============================================================

@dataclass(frozen=True, slots=True)
class NavigationObstacle:
    obstacle_id: str

    obstacle_type: ObstacleType

    position: Vector2
    radius_m: float

    velocity: Vector2 = Vector2(0.0, 0.0)

    confidence: float = 1.0

    def __post_init__(self) -> None:
        validate_identifier(self.obstacle_id)
        validate_nonnegative(self.radius_m)
        validate_probability(self.confidence)


# ============================================================
# Robot State
# ============================================================

@dataclass(slots=True)
class RobotNavigationState:
    tenant_id: str
    robot_id: str

    timestamp: datetime

    pose: Pose2D
    velocity: Velocity2D

    localization_status: LocalizationStatus

    battery_soc: float | None

    robot_radius_m: float = DEFAULT_ROBOT_RADIUS_M

    maximum_speed_mps: float = DEFAULT_MAX_SPEED_MPS

    emergency_stop_active: bool = False
    safety_fault_active: bool = False

    robot_available: bool = True

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.robot_id)
        validate_timestamp(self.timestamp)

        validate_positive(self.robot_radius_m)
        validate_positive(self.maximum_speed_mps)

        if self.battery_soc is not None:
            validate_probability(self.battery_soc)


# ============================================================
# Navigation Request
# ============================================================

@dataclass(slots=True)
class NavigationRequest:
    tenant_id: str
    robot_id: str

    task_type: NavigationTaskType

    goal_pose: Pose2D

    navigation_mode: NavigationMode = (
        NavigationMode.INDOOR
    )

    algorithm: PathPlanningAlgorithm = (
        PathPlanningAlgorithm.ASTAR
    )

    safety_margin_m: float = (
        DEFAULT_SAFETY_MARGIN_M
    )

    maximum_speed_mps: float | None = None

    require_simulation: bool = True

    simulation_only: bool = True

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.robot_id)

        validate_nonnegative(self.safety_margin_m)

        if self.maximum_speed_mps is not None:
            validate_positive(self.maximum_speed_mps)

        if not self.simulation_only:
            raise NavigationValidationError(
                "NavigationAgent supports "
                "simulation-only planning"
            )


# ============================================================
# Route Models
# ============================================================

@dataclass(frozen=True, slots=True)
class NavigationWaypoint:
    waypoint_id: str

    pose: Pose2D

    distance_from_start_m: float

    estimated_time_seconds: float

    def __post_init__(self) -> None:
        validate_identifier(self.waypoint_id)

        validate_nonnegative(
            self.distance_from_start_m
        )

        validate_nonnegative(
            self.estimated_time_seconds
        )


@dataclass(slots=True)
class NavigationRoute:
    route_id: str

    robot_id: str

    waypoints: list[NavigationWaypoint]

    total_distance_m: float

    estimated_duration_seconds: float

    algorithm: PathPlanningAlgorithm

    map_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.route_id)
        validate_identifier(self.robot_id)
        validate_identifier(self.map_id)

        validate_nonnegative(self.total_distance_m)
        validate_nonnegative(
            self.estimated_duration_seconds
        )

        if len(self.waypoints) > MAX_ROUTE_WAYPOINTS:
            raise NavigationValidationError(
                "Too many route waypoints"
            )


# ============================================================
# Safety Models
# ============================================================

@dataclass(frozen=True, slots=True)
class NavigationSafetyAssessment:
    decision: NavigationSafetyDecision

    risk_level: NavigationRiskLevel

    collision_status: CollisionStatus

    localization_status: LocalizationStatus

    battery_sufficient: bool | None

    goal_within_bounds: bool

    requires_human_review: bool

    findings: tuple[str, ...]


@dataclass(slots=True)
class NavigationSimulationResult:
    simulation_id: str

    status: NavigationSimulationStatus

    collision_detected: bool | None

    goal_reached: bool | None

    estimated_duration_seconds: float | None

    findings: list[str] = field(
        default_factory=list
    )


@dataclass(slots=True)
class NavigationPlan:
    plan_id: str

    tenant_id: str
    robot_id: str

    task_type: NavigationTaskType

    status: NavigationStatus

    start_pose: Pose2D
    goal_pose: Pose2D

    route: NavigationRoute | None

    safety_assessment: NavigationSafetyAssessment

    simulation_result: (
        NavigationSimulationResult | None
    )

    requires_human_review: bool

    reasoning: list[str]

    created_at: datetime = field(
        default_factory=utc_now
    )

    simulation_only: bool = True


# ============================================================
# Provider Protocols
# ============================================================

class NavigationStateProvider(Protocol):

    async def get_robot_state(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> RobotNavigationState | None:
        ...


class NavigationMapProvider(Protocol):

    async def get_occupancy_grid(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> OccupancyGrid | None:
        ...


class NavigationObstacleProvider(Protocol):

    async def get_obstacles(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> list[NavigationObstacle]:
        ...


class NavigationSimulator(Protocol):

    async def simulate(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        plan: NavigationPlan,
    ) -> NavigationSimulationResult:
        ...


class NavigationAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        robot_id: str,
        permission: str,
    ) -> None:
        ...


class NavigationPlanRepository(Protocol):

    async def save(
        self,
        *,
        plan: NavigationPlan,
    ) -> None:
        ...


class NavigationAuditSink(Protocol):

    async def record(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        action: str,
        resource_id: str,
        metadata: Mapping[str, Any],
    ) -> None:
        ...


# ============================================================
# A* Global Path Planner
# ============================================================

class AStarPathPlanner:

    def __init__(
        self,
        *,
        occupied_threshold: int = 65,
        allow_unknown: bool = False,
    ) -> None:

        if not 0 <= occupied_threshold <= 100:
            raise NavigationValidationError(
                "Invalid occupancy threshold"
            )

        self.occupied_threshold = occupied_threshold
        self.allow_unknown = allow_unknown

    def is_free(
        self,
        grid: OccupancyGrid,
        x: int,
        y: int,
    ) -> bool:

        value = grid.occupancy(x, y)

        if value == -1:
            return self.allow_unknown

        return value < self.occupied_threshold

    @staticmethod
    def heuristic(
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> float:

        return math.hypot(
            a[0] - b[0],
            a[1] - b[1],
        )

    @staticmethod
    def neighbors(
        x: int,
        y: int,
    ) -> list[tuple[int, int]]:

        return [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
            (x + 1, y + 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
            (x - 1, y - 1),
        ]

    def plan(
        self,
        *,
        grid: OccupancyGrid,
        start: Vector2,
        goal: Vector2,
        robot_radius_m: float,
        safety_margin_m: float,
    ) -> list[Vector2]:

        start_cell = grid.world_to_grid(start)
        goal_cell = grid.world_to_grid(goal)

        if not grid.in_bounds(*start_cell):
            raise NavigationPlanningError(
                "Start is outside the map"
            )

        if not grid.in_bounds(*goal_cell):
            raise NavigationPlanningError(
                "Goal is outside the map"
            )

        inflation_cells = math.ceil(
            (robot_radius_m + safety_margin_m)
            / grid.resolution_m
        )

        blocked = self._inflate_obstacles(
            grid=grid,
            inflation_cells=inflation_cells,
        )

        if start_cell in blocked:
            raise NavigationPlanningError(
                "Start is in an occupied or unsafe cell"
            )

        if goal_cell in blocked:
            raise NavigationPlanningError(
                "Goal is in an occupied or unsafe cell"
            )

        open_heap: list[
            tuple[float, int, tuple[int, int]]
        ] = []

        counter = 0

        heapq.heappush(
            open_heap,
            (
                self.heuristic(start_cell, goal_cell),
                counter,
                start_cell,
            ),
        )

        came_from: dict[
            tuple[int, int],
            tuple[int, int],
        ] = {}

        g_score = {start_cell: 0.0}

        closed: set[tuple[int, int]] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in closed:
                continue

            if current == goal_cell:
                cells = self._reconstruct_path(
                    came_from,
                    current,
                )

                return [
                    grid.grid_to_world(x, y)
                    for x, y in cells
                ]

            closed.add(current)

            for neighbor in self.neighbors(*current):
                if not grid.in_bounds(*neighbor):
                    continue

                if neighbor in blocked:
                    continue

                if neighbor in closed:
                    continue

                dx = neighbor[0] - current[0]
                dy = neighbor[1] - current[1]

                if dx != 0 and dy != 0:
                    adjacent_a = (
                        current[0] + dx,
                        current[1],
                    )
                    adjacent_b = (
                        current[0],
                        current[1] + dy,
                    )

                    if (
                        adjacent_a in blocked
                        or adjacent_b in blocked
                    ):
                        continue

                step_cost = math.hypot(dx, dy)

                tentative = (
                    g_score[current] + step_cost
                )

                if tentative < g_score.get(
                    neighbor,
                    math.inf,
                ):
                    came_from[neighbor] = current

                    g_score[neighbor] = tentative

                    counter += 1

                    priority = (
                        tentative
                        + self.heuristic(
                            neighbor,
                            goal_cell,
                        )
                    )

                    heapq.heappush(
                        open_heap,
                        (
                            priority,
                            counter,
                            neighbor,
                        ),
                    )

        raise NavigationPlanningError(
            "No collision-free route was found"
        )

    def _inflate_obstacles(
        self,
        *,
        grid: OccupancyGrid,
        inflation_cells: int,
    ) -> set[tuple[int, int]]:

        blocked: set[tuple[int, int]] = set()

        occupied: list[tuple[int, int]] = []

        for y in range(grid.height):
            for x in range(grid.width):
                if not self.is_free(grid, x, y):
                    occupied.append((x, y))

        for ox, oy in occupied:
            for dy in range(
                -inflation_cells,
                inflation_cells + 1,
            ):
                for dx in range(
                    -inflation_cells,
                    inflation_cells + 1,
                ):
                    if (
                        dx * dx + dy * dy
                        > inflation_cells * inflation_cells
                    ):
                        continue

                    nx = ox + dx
                    ny = oy + dy

                    if grid.in_bounds(nx, ny):
                        blocked.add((nx, ny))

        return blocked

    @staticmethod
    def _reconstruct_path(
        came_from: Mapping[
            tuple[int, int],
            tuple[int, int],
        ],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:

        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()

        return path


# ============================================================
# Navigation Agent
# ============================================================

class NavigationAgent:

    def __init__(
        self,
        *,
        state_provider: NavigationStateProvider,
        map_provider: NavigationMapProvider,
        obstacle_provider: NavigationObstacleProvider,
        authorizer: NavigationAuthorizer,
        simulator: NavigationSimulator | None = None,
        repository: NavigationPlanRepository | None = None,
        audit_sink: NavigationAuditSink | None = None,
        path_planner: AStarPathPlanner | None = None,
    ) -> None:

        self.state_provider = state_provider
        self.map_provider = map_provider
        self.obstacle_provider = obstacle_provider

        self.authorizer = authorizer
        self.simulator = simulator
        self.repository = repository
        self.audit_sink = audit_sink

        self.path_planner = (
            path_planner or AStarPathPlanner()
        )

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    async def _authorize(
        self,
        *,
        scope: NavigationScope,
        robot_id: str,
    ) -> None:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            robot_id=robot_id,
            permission="navigation:plan",
        )

    # --------------------------------------------------------
    # Path Distance
    # --------------------------------------------------------

    @staticmethod
    def calculate_path_distance(
        points: Sequence[Vector2],
    ) -> float:

        return sum(
            points[index - 1].distance_to(
                points[index]
            )
            for index in range(1, len(points))
        )

    # --------------------------------------------------------
    # Dynamic Obstacle Check
    # --------------------------------------------------------

    @staticmethod
    def check_dynamic_obstacles(
        *,
        points: Sequence[Vector2],
        obstacles: Sequence[NavigationObstacle],
        robot_radius_m: float,
        safety_margin_m: float,
    ) -> CollisionStatus:

        if not points:
            return CollisionStatus.UNKNOWN

        for obstacle in obstacles:
            if obstacle.obstacle_type in (
                ObstacleType.DYNAMIC,
                ObstacleType.HUMAN,
                ObstacleType.VEHICLE,
                ObstacleType.ROBOT,
            ):
                # A static snapshot cannot certify a moving
                # obstacle trajectory. Return UNKNOWN even
                # when there is no current intersection.
                return CollisionStatus.UNKNOWN

            required_distance = (
                obstacle.radius_m
                + robot_radius_m
                + safety_margin_m
            )

            for point in points:
                if (
                    point.distance_to(obstacle.position)
                    <= required_distance
                ):
                    return CollisionStatus.COLLISION

        return CollisionStatus.CLEAR

    # --------------------------------------------------------
    # Safety Assessment
    # --------------------------------------------------------

    @staticmethod
    def assess_safety(
        *,
        state: RobotNavigationState,
        goal_within_bounds: bool,
        collision_status: CollisionStatus,
        route_available: bool,
    ) -> NavigationSafetyAssessment:

        findings: list[str] = []

        if state.emergency_stop_active:
            findings.append(
                "Robot emergency stop is active."
            )

        if state.safety_fault_active:
            findings.append(
                "Robot safety fault is active."
            )

        if not state.robot_available:
            findings.append(
                "Robot is unavailable."
            )

        if state.localization_status == LocalizationStatus.LOST:
            findings.append(
                "Robot localization is lost."
            )

        if state.localization_status == LocalizationStatus.DEGRADED:
            findings.append(
                "Robot localization is degraded."
            )

        if state.localization_status == LocalizationStatus.UNKNOWN:
            findings.append(
                "Robot localization status is unknown."
            )

        if not goal_within_bounds:
            findings.append(
                "Goal is outside the navigation map."
            )

        if not route_available:
            findings.append(
                "No valid route is available."
            )

        if collision_status != CollisionStatus.CLEAR:
            findings.append(
                f"Collision status: {collision_status.value}."
            )

        battery_sufficient: bool | None = None

        if state.battery_soc is not None:
            battery_sufficient = (
                state.battery_soc >= 0.20
            )

            if not battery_sufficient:
                findings.append(
                    "Robot battery is below planning threshold."
                )

        hard_reject = (
            state.emergency_stop_active
            or state.safety_fault_active
            or not state.robot_available
            or state.localization_status
            == LocalizationStatus.LOST
            or not goal_within_bounds
            or not route_available
            or collision_status == CollisionStatus.COLLISION
            or battery_sufficient is False
        )

        if hard_reject:
            return NavigationSafetyAssessment(
                decision=NavigationSafetyDecision.REJECT,
                risk_level=NavigationRiskLevel.CRITICAL,
                collision_status=collision_status,
                localization_status=state.localization_status,
                battery_sufficient=battery_sufficient,
                goal_within_bounds=goal_within_bounds,
                requires_human_review=True,
                findings=tuple(findings),
            )

        uncertain = (
            collision_status == CollisionStatus.UNKNOWN
            or state.localization_status
            != LocalizationStatus.GOOD
            or battery_sufficient is None
        )

        if uncertain:
            return NavigationSafetyAssessment(
                decision=(
                    NavigationSafetyDecision.REVIEW_REQUIRED
                ),
                risk_level=NavigationRiskLevel.HIGH,
                collision_status=collision_status,
                localization_status=state.localization_status,
                battery_sufficient=battery_sufficient,
                goal_within_bounds=goal_within_bounds,
                requires_human_review=True,
                findings=tuple(findings),
            )

        return NavigationSafetyAssessment(
            decision=NavigationSafetyDecision.ALLOW_SIMULATION,
            risk_level=NavigationRiskLevel.LOW,
            collision_status=collision_status,
            localization_status=state.localization_status,
            battery_sufficient=battery_sufficient,
            goal_within_bounds=goal_within_bounds,
            requires_human_review=False,
            findings=tuple(findings),
        )

    # --------------------------------------------------------
    # Build Route
    # --------------------------------------------------------

    @staticmethod
    def build_route(
        *,
        robot_id: str,
        map_id: str,
        frame_id: str,
        points: Sequence[Vector2],
        speed_mps: float,
        algorithm: PathPlanningAlgorithm,
    ) -> NavigationRoute:

        validate_positive(speed_mps)

        if len(points) > MAX_ROUTE_WAYPOINTS:
            raise NavigationValidationError(
                "Route exceeds waypoint limit"
            )

        waypoints: list[NavigationWaypoint] = []

        distance = 0.0

        for index, point in enumerate(points):
            if index > 0:
                distance += points[
                    index - 1
                ].distance_to(point)

            if index < len(points) - 1:
                next_point = points[index + 1]
                yaw = math.atan2(
                    next_point.y - point.y,
                    next_point.x - point.x,
                )
            else:
                yaw = (
                    waypoints[-1].pose.yaw_rad
                    if waypoints
                    else 0.0
                )

            waypoints.append(
                NavigationWaypoint(
                    waypoint_id=generate_id(),
                    pose=Pose2D(
                        position=point,
                        yaw_rad=yaw,
                        frame_id=frame_id,
                    ),
                    distance_from_start_m=distance,
                    estimated_time_seconds=(
                        distance / speed_mps
                    ),
                )
            )

        return NavigationRoute(
            route_id=generate_id(),
            robot_id=robot_id,
            waypoints=waypoints,
            total_distance_m=distance,
            estimated_duration_seconds=(
                distance / speed_mps
            ),
            algorithm=algorithm,
            map_id=map_id,
        )

    # --------------------------------------------------------
    # Navigation Planning
    # --------------------------------------------------------

    async def plan_navigation(
        self,
        *,
        scope: NavigationScope,
        request: NavigationRequest,
    ) -> NavigationPlan:

        if request.tenant_id != scope.tenant_id:
            raise NavigationPermissionError(
                "Tenant mismatch"
            )

        await self._authorize(
            scope=scope,
            robot_id=request.robot_id,
        )

        state = await self.state_provider.get_robot_state(
            tenant_id=scope.tenant_id,
            robot_id=request.robot_id,
        )

        if state is None:
            raise NavigationNotFoundError(
                "Robot navigation state not found"
            )

        if (
            state.tenant_id != scope.tenant_id
            or state.robot_id != request.robot_id
        ):
            raise NavigationPermissionError(
                "Robot state scope mismatch"
            )

        grid = await self.map_provider.get_occupancy_grid(
            tenant_id=scope.tenant_id,
            robot_id=request.robot_id,
        )

        if grid is None:
            raise NavigationNotFoundError(
                "Navigation map not found"
            )

        if (
            state.pose.frame_id != grid.frame_id
            or request.goal_pose.frame_id != grid.frame_id
        ):
            raise NavigationValidationError(
                "Pose frames do not match map frame; "
                "an authorized transform is required"
            )

        obstacles = (
            await self.obstacle_provider.get_obstacles(
                tenant_id=scope.tenant_id,
                robot_id=request.robot_id,
            )
        )

        if len(obstacles) > MAX_OBSTACLES:
            raise NavigationValidationError(
                "Too many obstacles"
            )

        goal_cell = grid.world_to_grid(
            request.goal_pose.position
        )

        goal_within_bounds = grid.in_bounds(
            *goal_cell
        )

        speed = min(
            request.maximum_speed_mps
            or state.maximum_speed_mps,
            state.maximum_speed_mps,
        )

        route: NavigationRoute | None = None

        collision_status = CollisionStatus.UNKNOWN

        reasoning: list[str] = []

        try:
            if request.algorithm != PathPlanningAlgorithm.ASTAR:
                raise NavigationPlanningError(
                    "Only A* is implemented by the built-in planner"
                )

            points = self.path_planner.plan(
                grid=grid,
                start=state.pose.position,
                goal=request.goal_pose.position,
                robot_radius_m=state.robot_radius_m,
                safety_margin_m=request.safety_margin_m,
            )

            collision_status = (
                self.check_dynamic_obstacles(
                    points=points,
                    obstacles=obstacles,
                    robot_radius_m=state.robot_radius_m,
                    safety_margin_m=request.safety_margin_m,
                )
            )

            route = self.build_route(
                robot_id=request.robot_id,
                map_id=grid.map_id,
                frame_id=grid.frame_id,
                points=points,
                speed_mps=speed,
                algorithm=request.algorithm,
            )

            reasoning.append(
                f"Global route contains {len(points)} waypoints."
            )

            reasoning.append(
                f"Estimated route distance: "
                f"{route.total_distance_m:.2f} meters."
            )

        except NavigationPlanningError as exc:
            reasoning.append(str(exc))

        safety = self.assess_safety(
            state=state,
            goal_within_bounds=goal_within_bounds,
            collision_status=collision_status,
            route_available=route is not None,
        )

        if safety.decision == NavigationSafetyDecision.REJECT:
            status = NavigationStatus.VALIDATION_FAILED

        elif (
            safety.decision
            == NavigationSafetyDecision.REVIEW_REQUIRED
        ):
            status = NavigationStatus.REVIEW_REQUIRED

        else:
            status = NavigationStatus.PLANNED

        plan = NavigationPlan(
            plan_id=generate_id(),
            tenant_id=scope.tenant_id,
            robot_id=request.robot_id,
            task_type=request.task_type,
            status=status,
            start_pose=state.pose,
            goal_pose=request.goal_pose,
            route=route,
            safety_assessment=safety,
            simulation_result=None,
            requires_human_review=(
                safety.requires_human_review
            ),
            reasoning=reasoning + list(safety.findings),
            simulation_only=True,
        )

        if (
            request.require_simulation
            and safety.decision
            != NavigationSafetyDecision.REJECT
        ):
            if self.simulator is None:
                plan.status = NavigationStatus.REVIEW_REQUIRED
                plan.requires_human_review = True
                plan.reasoning.append(
                    "Navigation simulator is unavailable."
                )

            else:
                result = await self.simulator.simulate(
                    tenant_id=scope.tenant_id,
                    robot_id=request.robot_id,
                    plan=plan,
                )

                plan.simulation_result = result

                if (
                    result.status
                    == NavigationSimulationStatus.FAILED
                ):
                    plan.status = (
                        NavigationStatus.SIMULATION_FAILED
                    )
                    plan.requires_human_review = True

                elif (
                    result.status
                    == NavigationSimulationStatus.PASSED
                    and result.collision_detected is False
                    and result.goal_reached is True
                    and safety.decision
                    == NavigationSafetyDecision.ALLOW_SIMULATION
                ):
                    plan.status = (
                        NavigationStatus.APPROVED_FOR_SIMULATION
                    )

                else:
                    plan.status = (
                        NavigationStatus.REVIEW_REQUIRED
                    )
                    plan.requires_human_review = True

        if self.repository is not None:
            await self.repository.save(plan=plan)

        if self.audit_sink is not None:
            await self.audit_sink.record(
                tenant_id=scope.tenant_id,
                actor_id=scope.user_id,
                action="navigation_agent.plan_created",
                resource_id=plan.plan_id,
                metadata={
                    "robot_id": request.robot_id,
                    "status": plan.status.value,
                    "task_type": request.task_type.value,
                    "simulation_only": True,
                },
            )

        return plan

    # --------------------------------------------------------
    # Replanning
    # --------------------------------------------------------

    async def replan_navigation(
        self,
        *,
        scope: NavigationScope,
        previous_plan: NavigationPlan,
    ) -> NavigationPlan:

        if previous_plan.tenant_id != scope.tenant_id:
            raise NavigationPermissionError(
                "Previous plan tenant mismatch"
            )

        request = NavigationRequest(
            tenant_id=scope.tenant_id,
            robot_id=previous_plan.robot_id,
            task_type=NavigationTaskType.REPLAN,
            goal_pose=previous_plan.goal_pose,
            require_simulation=True,
            simulation_only=True,
        )

        return await self.plan_navigation(
            scope=scope,
            request=request,
        )

    # --------------------------------------------------------
    # Batch Planning
    # --------------------------------------------------------

    async def plan_batch(
        self,
        *,
        scope: NavigationScope,
        requests: Sequence[NavigationRequest],
    ) -> list[NavigationPlan]:

        if len(requests) > MAX_BATCH_REQUESTS:
            raise NavigationValidationError(
                "Batch size exceeds limit"
            )

        semaphore = asyncio.Semaphore(5)

        async def process(
            request: NavigationRequest,
        ) -> NavigationPlan:

            async with semaphore:
                return await self.plan_navigation(
                    scope=scope,
                    request=request,
                )

        return list(
            await asyncio.gather(
                *(process(request) for request in requests)
            )
        )

    # --------------------------------------------------------
    # Explainable Report
    # --------------------------------------------------------

    @staticmethod
    def generate_report(
        plan: NavigationPlan,
    ) -> dict[str, Any]:

        route = plan.route

        return {
            "plan_id": plan.plan_id,
            "tenant_id": plan.tenant_id,
            "robot_id": plan.robot_id,
            "task_type": plan.task_type.value,
            "status": plan.status.value,
            "created_at": plan.created_at.isoformat(),
            "route": (
                {
                    "route_id": route.route_id,
                    "map_id": route.map_id,
                    "algorithm": route.algorithm.value,
                    "distance_m": route.total_distance_m,
                    "estimated_duration_seconds": (
                        route.estimated_duration_seconds
                    ),
                    "waypoint_count": len(route.waypoints),
                    "waypoints": [
                        {
                            "waypoint_id": waypoint.waypoint_id,
                            "x": waypoint.pose.position.x,
                            "y": waypoint.pose.position.y,
                            "yaw_rad": waypoint.pose.yaw_rad,
                            "distance_from_start_m": (
                                waypoint.distance_from_start_m
                            ),
                            "estimated_time_seconds": (
                                waypoint.estimated_time_seconds
                            ),
                        }
                        for waypoint in route.waypoints
                    ],
                }
                if route is not None
                else None
            ),
            "safety": {
                "decision": (
                    plan.safety_assessment.decision.value
                ),
                "risk_level": (
                    plan.safety_assessment.risk_level.value
                ),
                "collision_status": (
                    plan.safety_assessment.collision_status.value
                ),
                "localization_status": (
                    plan.safety_assessment.localization_status.value
                ),
                "battery_sufficient": (
                    plan.safety_assessment.battery_sufficient
                ),
                "goal_within_bounds": (
                    plan.safety_assessment.goal_within_bounds
                ),
                "findings": list(
                    plan.safety_assessment.findings
                ),
            },
            "simulation": (
                {
                    "simulation_id": (
                        plan.simulation_result.simulation_id
                    ),
                    "status": (
                        plan.simulation_result.status.value
                    ),
                    "collision_detected": (
                        plan.simulation_result.collision_detected
                    ),
                    "goal_reached": (
                        plan.simulation_result.goal_reached
                    ),
                    "findings": (
                        plan.simulation_result.findings
                    ),
                }
                if plan.simulation_result is not None
                else None
            ),
            "reasoning": plan.reasoning,
            "requires_human_review": (
                plan.requires_human_review
            ),
            "simulation_only": True,
            "physical_execution_enabled": False,
        }


# ============================================================
# Factory
# ============================================================

def create_navigation_agent(
    *,
    state_provider: NavigationStateProvider,
    map_provider: NavigationMapProvider,
    obstacle_provider: NavigationObstacleProvider,
    authorizer: NavigationAuthorizer,
    simulator: NavigationSimulator | None = None,
    repository: NavigationPlanRepository | None = None,
    audit_sink: NavigationAuditSink | None = None,
    path_planner: AStarPathPlanner | None = None,
) -> NavigationAgent:

    return NavigationAgent(
        state_provider=state_provider,
        map_provider=map_provider,
        obstacle_provider=obstacle_provider,
        authorizer=authorizer,
        simulator=simulator,
        repository=repository,
        audit_sink=audit_sink,
        path_planner=path_planner,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "NavigationAgentError",
    "NavigationValidationError",
    "NavigationNotFoundError",
    "NavigationPermissionError",
    "NavigationPlanningError",
    "NavigationSafetyError",
    "NavigationServiceUnavailableError",
    "NavigationMode",
    "NavigationTaskType",
    "NavigationStatus",
    "NavigationRiskLevel",
    "LocalizationStatus",
    "ObstacleType",
    "CollisionStatus",
    "PathPlanningAlgorithm",
    "NavigationSafetyDecision",
    "NavigationSimulationStatus",
    "NavigationScope",
    "Vector2",
    "Pose2D",
    "Velocity2D",
    "NavigationBounds",
    "OccupancyGrid",
    "NavigationObstacle",
    "RobotNavigationState",
    "NavigationRequest",
    "NavigationWaypoint",
    "NavigationRoute",
    "NavigationSafetyAssessment",
    "NavigationSimulationResult",
    "NavigationPlan",
    "NavigationStateProvider",
    "NavigationMapProvider",
    "NavigationObstacleProvider",
    "NavigationSimulator",
    "NavigationAuthorizer",
    "NavigationPlanRepository",
    "NavigationAuditSink",
    "AStarPathPlanner",
    "NavigationAgent",
    "create_navigation_agent",
]
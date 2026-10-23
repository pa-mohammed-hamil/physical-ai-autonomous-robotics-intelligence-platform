"""Manipulation Agent module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/manipulation_agent.py

Physical AI & Autonomous Robotics Intelligence Platform
Autonomous Manipulation Planning Agent

Python 3.11+

Capabilities:
- Robotic arm manipulation planning
- Pick-and-place task planning
- Object grasp planning
- Object detection and pose integration
- End-effector compatibility checking
- Manipulation trajectory planning
- Inverse kinematics service integration
- Collision checking
- Workspace boundary validation
- Payload and reachability checks
- Force and torque constraint validation
- Object placement planning
- Manipulation risk assessment
- Simulation-based plan evaluation
- Human-in-the-loop review
- Multi-tenant authorization
- Audit logging
- Explainable manipulation plans

Safety:
- Produces manipulation plans only.
- Does not execute physical robot commands.
- Does not bypass robot safety controllers.
- Does not reset emergency stops.
- Requires independent hardware safety validation
  before any real-world execution.

External robotics integrations are defined through Protocols.
"""

from __future__ import annotations

import asyncio
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

MAX_MANIPULATION_STEPS = 100
MAX_TRAJECTORY_WAYPOINTS = 2000
MAX_GRASP_CANDIDATES = 100
MAX_BATCH_SIZE = 50

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
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def validate_finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Value must be finite")
    return value


def validate_nonnegative(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("Value must be finite and nonnegative")
    return value


def validate_positive(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Value must be finite and positive")
    return value


def validate_probability(value: float) -> float:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probability must be between 0 and 1")
    return value


def validate_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Datetime must be timezone-aware")
    return value


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, value))


# ============================================================
# Enumerations
# ============================================================

class ManipulationTaskType(str, Enum):
    PICK = "pick"
    PLACE = "place"
    PICK_AND_PLACE = "pick_and_place"
    GRASP = "grasp"
    RELEASE = "release"
    PUSH = "push"
    PULL = "pull"
    INSERT = "insert"
    ASSEMBLE = "assemble"
    SORT = "sort"
    STACK = "stack"
    INSPECT = "inspect"
    TOOL_USE = "tool_use"
    CUSTOM = "custom"


class ManipulationStepType(str, Enum):
    MOVE_TO_PREGRASP = "move_to_pregrasp"
    ALIGN_GRIPPER = "align_gripper"
    APPROACH_OBJECT = "approach_object"
    CLOSE_GRIPPER = "close_gripper"
    VERIFY_GRASP = "verify_grasp"
    LIFT_OBJECT = "lift_object"
    TRANSPORT_OBJECT = "transport_object"
    MOVE_TO_PLACE = "move_to_place"
    LOWER_OBJECT = "lower_object"
    OPEN_GRIPPER = "open_gripper"
    RETREAT = "retreat"
    VERIFY_PLACEMENT = "verify_placement"
    HOLD_POSITION = "hold_position"
    CUSTOM = "custom"


class EndEffectorType(str, Enum):
    PARALLEL_GRIPPER = "parallel_gripper"
    THREE_FINGER_GRIPPER = "three_finger_gripper"
    SUCTION_GRIPPER = "suction_gripper"
    MAGNETIC_GRIPPER = "magnetic_gripper"
    SOFT_GRIPPER = "soft_gripper"
    ROBOT_HAND = "robot_hand"
    TOOL = "tool"
    CUSTOM = "custom"


class GraspType(str, Enum):
    PARALLEL = "parallel"
    PINCH = "pinch"
    POWER = "power"
    SUCTION = "suction"
    TOP_DOWN = "top_down"
    SIDE = "side"
    CUSTOM = "custom"


class ManipulationPlanStatus(str, Enum):
    DRAFT = "draft"
    PLANNING = "planning"
    PLANNED = "planned"
    VALIDATION_FAILED = "validation_failed"
    SIMULATION_FAILED = "simulation_failed"
    REVIEW_REQUIRED = "review_required"
    APPROVED_FOR_SIMULATION = "approved_for_simulation"
    REJECTED = "rejected"


class ManipulationRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CollisionStatus(str, Enum):
    CLEAR = "clear"
    COLLISION = "collision"
    UNKNOWN = "unknown"


class ReachabilityStatus(str, Enum):
    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


class SimulationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    NOT_RUN = "not_run"


class SafetyDecision(str, Enum):
    ALLOW_SIMULATION = "allow_simulation"
    REVIEW_REQUIRED = "review_required"
    REJECT = "reject"


# ============================================================
# Exceptions
# ============================================================

class ManipulationAgentError(Exception):
    pass


class ManipulationValidationError(ManipulationAgentError):
    pass


class ManipulationNotFoundError(ManipulationAgentError):
    pass


class ManipulationPermissionError(ManipulationAgentError):
    pass


class ManipulationServiceUnavailableError(
    ManipulationAgentError
):
    pass


class ManipulationPlanningError(ManipulationAgentError):
    pass


# ============================================================
# Security
# ============================================================

@dataclass(frozen=True, slots=True)
class ManipulationScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Geometry Models
# ============================================================

@dataclass(frozen=True, slots=True)
class Vector3:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        validate_finite(self.x)
        validate_finite(self.y)
        validate_finite(self.z)

    def distance_to(self, other: Vector3) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )


@dataclass(frozen=True, slots=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.z, self.w)

        for value in values:
            validate_finite(value)

        norm = math.sqrt(
            sum(value * value for value in values)
        )

        if not math.isclose(
            norm,
            1.0,
            rel_tol=1e-4,
            abs_tol=1e-4,
        ):
            raise ManipulationValidationError(
                "Quaternion must be normalized"
            )


@dataclass(frozen=True, slots=True)
class Pose3D:
    position: Vector3
    orientation: Quaternion
    frame_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.frame_id)


@dataclass(frozen=True, slots=True)
class BoundingBox3D:
    center: Vector3
    dimensions: Vector3

    def __post_init__(self) -> None:
        for value in (
            self.dimensions.x,
            self.dimensions.y,
            self.dimensions.z,
        ):
            validate_positive(value)


@dataclass(frozen=True, slots=True)
class WorkspaceBounds:
    minimum: Vector3
    maximum: Vector3

    def __post_init__(self) -> None:
        if not (
            self.minimum.x < self.maximum.x
            and self.minimum.y < self.maximum.y
            and self.minimum.z < self.maximum.z
        ):
            raise ManipulationValidationError(
                "Invalid workspace bounds"
            )

    def contains(self, position: Vector3) -> bool:
        return (
            self.minimum.x <= position.x <= self.maximum.x
            and self.minimum.y <= position.y <= self.maximum.y
            and self.minimum.z <= position.z <= self.maximum.z
        )


# ============================================================
# Robot and Object Models
# ============================================================

@dataclass(slots=True)
class ManipulationObject:
    object_id: str
    label: str

    pose: Pose3D
    bounding_box: BoundingBox3D

    mass_kg: float | None = None
    fragile: bool = False
    hazardous: bool = False

    perception_confidence: float = 1.0

    graspable: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(self.object_id)
        validate_probability(self.perception_confidence)

        if self.mass_kg is not None:
            validate_nonnegative(self.mass_kg)


@dataclass(slots=True)
class EndEffectorConfiguration:
    end_effector_id: str
    end_effector_type: EndEffectorType

    maximum_payload_kg: float

    maximum_grip_force_n: float | None = None
    maximum_opening_m: float | None = None

    supports_force_feedback: bool = False
    supports_tactile_feedback: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.end_effector_id)
        validate_positive(self.maximum_payload_kg)

        for value in (
            self.maximum_grip_force_n,
            self.maximum_opening_m,
        ):
            if value is not None:
                validate_positive(value)


@dataclass(slots=True)
class RobotManipulationState:
    tenant_id: str
    robot_id: str

    timestamp: datetime

    base_pose: Pose3D
    end_effector_pose: Pose3D

    joint_positions: list[float]

    workspace_bounds: WorkspaceBounds

    end_effector: EndEffectorConfiguration

    emergency_stop_active: bool = False
    safety_fault_active: bool = False

    robot_available: bool = True

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.robot_id)
        validate_timestamp(self.timestamp)

        if not self.joint_positions:
            raise ManipulationValidationError(
                "Joint positions are required"
            )

        for position in self.joint_positions:
            validate_finite(position)


# ============================================================
# Grasp Models
# ============================================================

@dataclass(frozen=True, slots=True)
class GraspCandidate:
    grasp_id: str

    object_id: str

    grasp_type: GraspType
    grasp_pose: Pose3D

    confidence: float

    estimated_grip_force_n: float | None = None

    approach_distance_m: float = 0.10
    lift_distance_m: float = 0.10

    def __post_init__(self) -> None:
        validate_identifier(self.grasp_id)
        validate_identifier(self.object_id)

        validate_probability(self.confidence)
        validate_nonnegative(self.approach_distance_m)
        validate_nonnegative(self.lift_distance_m)

        if self.estimated_grip_force_n is not None:
            validate_nonnegative(
                self.estimated_grip_force_n
            )


@dataclass(slots=True)
class GraspPlanningResult:
    robot_id: str
    object_id: str

    candidates: list[GraspCandidate]

    selected_grasp: GraspCandidate | None

    success: bool

    reasoning: list[str] = field(
        default_factory=list
    )


# ============================================================
# Trajectory Models
# ============================================================

@dataclass(frozen=True, slots=True)
class TrajectoryWaypoint:
    time_from_start_seconds: float

    joint_positions: tuple[float, ...]

    end_effector_pose: Pose3D | None = None

    def __post_init__(self) -> None:
        validate_nonnegative(
            self.time_from_start_seconds
        )

        if not self.joint_positions:
            raise ManipulationValidationError(
                "Waypoint joint positions are required"
            )

        for value in self.joint_positions:
            validate_finite(value)


@dataclass(slots=True)
class ManipulationTrajectory:
    trajectory_id: str
    robot_id: str

    waypoints: list[TrajectoryWaypoint]

    estimated_duration_seconds: float

    planning_frame_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.trajectory_id)
        validate_identifier(self.robot_id)
        validate_identifier(self.planning_frame_id)

        validate_nonnegative(
            self.estimated_duration_seconds
        )

        if len(self.waypoints) > MAX_TRAJECTORY_WAYPOINTS:
            raise ManipulationValidationError(
                "Too many trajectory waypoints"
            )

        times = [
            waypoint.time_from_start_seconds
            for waypoint in self.waypoints
        ]

        if times != sorted(times):
            raise ManipulationValidationError(
                "Trajectory timestamps must be ordered"
            )


# ============================================================
# Task Models
# ============================================================

@dataclass(slots=True)
class ManipulationTaskRequest:
    tenant_id: str
    robot_id: str

    task_type: ManipulationTaskType

    object_id: str

    target_pose: Pose3D | None = None

    instruction: str = ""

    maximum_allowed_force_n: float | None = None

    require_simulation: bool = True

    simulation_only: bool = True

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.robot_id)
        validate_identifier(self.object_id)

        if not self.simulation_only:
            raise ManipulationValidationError(
                "ManipulationAgent supports "
                "simulation-only planning"
            )

        if self.maximum_allowed_force_n is not None:
            validate_positive(
                self.maximum_allowed_force_n
            )

        if self.task_type in (
            ManipulationTaskType.PLACE,
            ManipulationTaskType.PICK_AND_PLACE,
            ManipulationTaskType.STACK,
        ) and self.target_pose is None:
            raise ManipulationValidationError(
                "Target pose is required for placement tasks"
            )


@dataclass(slots=True)
class ManipulationStep:
    step_id: str

    step_type: ManipulationStepType

    description: str

    target_pose: Pose3D | None = None

    trajectory: ManipulationTrajectory | None = None

    estimated_duration_seconds: float = 0.0

    requires_verification: bool = False


@dataclass(frozen=True, slots=True)
class ManipulationSafetyAssessment:
    decision: SafetyDecision

    risk_level: ManipulationRiskLevel

    collision_status: CollisionStatus
    reachability_status: ReachabilityStatus

    payload_compatible: bool | None
    workspace_valid: bool | None

    requires_human_review: bool

    findings: tuple[str, ...]


@dataclass(slots=True)
class ManipulationSimulationResult:
    simulation_id: str

    status: SimulationStatus

    success_probability: float | None

    collision_detected: bool | None

    object_slippage_detected: bool | None

    estimated_duration_seconds: float | None

    findings: list[str] = field(
        default_factory=list
    )


@dataclass(slots=True)
class ManipulationPlan:
    plan_id: str

    tenant_id: str
    robot_id: str
    object_id: str

    task_type: ManipulationTaskType

    status: ManipulationPlanStatus

    steps: list[ManipulationStep]

    selected_grasp: GraspCandidate | None

    safety_assessment: ManipulationSafetyAssessment

    simulation_result: (
        ManipulationSimulationResult | None
    )

    estimated_duration_seconds: float

    requires_human_review: bool

    reasoning: list[str]

    created_at: datetime = field(
        default_factory=utc_now
    )

    simulation_only: bool = True


# ============================================================
# Integration Protocols
# ============================================================

class ManipulationStateProvider(Protocol):

    async def get_robot_state(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> RobotManipulationState | None:
        ...


class ManipulationPerceptionProvider(Protocol):

    async def get_object(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        object_id: str,
    ) -> ManipulationObject | None:
        ...


class GraspPlanner(Protocol):

    async def generate_grasps(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        target_object: ManipulationObject,
        end_effector: EndEffectorConfiguration,
    ) -> list[GraspCandidate]:
        ...


class InverseKinematicsProvider(Protocol):

    async def check_reachability(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        target_pose: Pose3D,
    ) -> ReachabilityStatus:
        ...


class ManipulationTrajectoryPlanner(Protocol):

    async def plan_trajectory(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        start_pose: Pose3D,
        target_pose: Pose3D,
    ) -> ManipulationTrajectory | None:
        ...


class ManipulationCollisionChecker(Protocol):

    async def check_collision(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        trajectory: ManipulationTrajectory,
    ) -> CollisionStatus:
        ...


class ManipulationSimulator(Protocol):

    async def simulate(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        plan: ManipulationPlan,
    ) -> ManipulationSimulationResult:
        ...


class ManipulationAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        robot_id: str,
        permission: str,
    ) -> None:
        ...


class ManipulationAuditSink(Protocol):

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


class ManipulationPlanRepository(Protocol):

    async def save(
        self,
        *,
        plan: ManipulationPlan,
    ) -> None:
        ...


# ============================================================
# Manipulation Agent
# ============================================================

class ManipulationAgent:

    def __init__(
        self,
        *,
        state_provider: ManipulationStateProvider,
        perception_provider: ManipulationPerceptionProvider,
        grasp_planner: GraspPlanner,
        ik_provider: InverseKinematicsProvider,
        trajectory_planner: ManipulationTrajectoryPlanner,
        collision_checker: ManipulationCollisionChecker,
        authorizer: ManipulationAuthorizer,
        simulator: ManipulationSimulator | None = None,
        repository: ManipulationPlanRepository | None = None,
        audit_sink: ManipulationAuditSink | None = None,
    ) -> None:

        self.state_provider = state_provider
        self.perception_provider = perception_provider
        self.grasp_planner = grasp_planner
        self.ik_provider = ik_provider
        self.trajectory_planner = trajectory_planner
        self.collision_checker = collision_checker
        self.authorizer = authorizer
        self.simulator = simulator
        self.repository = repository
        self.audit_sink = audit_sink

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    async def _authorize(
        self,
        *,
        scope: ManipulationScope,
        robot_id: str,
        permission: str,
    ) -> None:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            robot_id=robot_id,
            permission=permission,
        )

    # --------------------------------------------------------
    # Payload Check
    # --------------------------------------------------------

    @staticmethod
    def check_payload(
        *,
        target_object: ManipulationObject,
        end_effector: EndEffectorConfiguration,
    ) -> bool | None:

        if target_object.mass_kg is None:
            return None

        return (
            target_object.mass_kg
            <= end_effector.maximum_payload_kg
        )

    # --------------------------------------------------------
    # Grasp Compatibility
    # --------------------------------------------------------

    @staticmethod
    def grasp_compatible(
        *,
        grasp: GraspCandidate,
        end_effector: EndEffectorConfiguration,
        maximum_allowed_force_n: float | None,
    ) -> bool:

        type_mapping = {
            EndEffectorType.PARALLEL_GRIPPER: {
                GraspType.PARALLEL,
                GraspType.PINCH,
                GraspType.TOP_DOWN,
                GraspType.SIDE,
            },
            EndEffectorType.THREE_FINGER_GRIPPER: {
                GraspType.PINCH,
                GraspType.POWER,
                GraspType.TOP_DOWN,
                GraspType.SIDE,
            },
            EndEffectorType.SUCTION_GRIPPER: {
                GraspType.SUCTION,
                GraspType.TOP_DOWN,
            },
            EndEffectorType.SOFT_GRIPPER: {
                GraspType.POWER,
                GraspType.PINCH,
            },
            EndEffectorType.ROBOT_HAND: {
                GraspType.POWER,
                GraspType.PINCH,
                GraspType.SIDE,
            },
        }

        allowed = type_mapping.get(
            end_effector.end_effector_type
        )

        if allowed is not None:
            if grasp.grasp_type not in allowed:
                return False

        force = grasp.estimated_grip_force_n

        if force is not None:
            if (
                end_effector.maximum_grip_force_n is not None
                and force > end_effector.maximum_grip_force_n
            ):
                return False

            if (
                maximum_allowed_force_n is not None
                and force > maximum_allowed_force_n
            ):
                return False

        return True

    # --------------------------------------------------------
    # Grasp Selection
    # --------------------------------------------------------

    async def select_grasp(
        self,
        *,
        scope: ManipulationScope,
        robot_state: RobotManipulationState,
        target_object: ManipulationObject,
        maximum_allowed_force_n: float | None,
    ) -> GraspPlanningResult:

        if not target_object.graspable:
            return GraspPlanningResult(
                robot_id=robot_state.robot_id,
                object_id=target_object.object_id,
                candidates=[],
                selected_grasp=None,
                success=False,
                reasoning=["Object is not marked graspable."],
            )

        candidates = await self.grasp_planner.generate_grasps(
            tenant_id=scope.tenant_id,
            robot_id=robot_state.robot_id,
            target_object=target_object,
            end_effector=robot_state.end_effector,
        )

        if len(candidates) > MAX_GRASP_CANDIDATES:
            raise ManipulationValidationError(
                "Too many grasp candidates"
            )

        compatible = [
            grasp
            for grasp in candidates
            if grasp.object_id == target_object.object_id
            and self.grasp_compatible(
                grasp=grasp,
                end_effector=robot_state.end_effector,
                maximum_allowed_force_n=(
                    maximum_allowed_force_n
                ),
            )
        ]

        compatible.sort(
            key=lambda grasp: grasp.confidence,
            reverse=True,
        )

        for grasp in compatible:
            reachability = (
                await self.ik_provider.check_reachability(
                    tenant_id=scope.tenant_id,
                    robot_id=robot_state.robot_id,
                    target_pose=grasp.grasp_pose,
                )
            )

            if reachability == ReachabilityStatus.REACHABLE:
                return GraspPlanningResult(
                    robot_id=robot_state.robot_id,
                    object_id=target_object.object_id,
                    candidates=candidates,
                    selected_grasp=grasp,
                    success=True,
                    reasoning=[
                        "Grasp is compatible with the end effector.",
                        "Inverse kinematics reports reachable pose.",
                    ],
                )

        return GraspPlanningResult(
            robot_id=robot_state.robot_id,
            object_id=target_object.object_id,
            candidates=candidates,
            selected_grasp=None,
            success=False,
            reasoning=[
                "No compatible and reachable grasp was found."
            ],
        )

    # --------------------------------------------------------
    # Step Construction
    # --------------------------------------------------------

    @staticmethod
    def _step(
        step_type: ManipulationStepType,
        description: str,
        *,
        target_pose: Pose3D | None = None,
        requires_verification: bool = False,
    ) -> ManipulationStep:

        return ManipulationStep(
            step_id=generate_id(),
            step_type=step_type,
            description=description,
            target_pose=target_pose,
            requires_verification=requires_verification,
        )

    def build_task_steps(
        self,
        *,
        request: ManipulationTaskRequest,
        grasp: GraspCandidate | None,
    ) -> list[ManipulationStep]:

        steps: list[ManipulationStep] = []

        pick_tasks = {
            ManipulationTaskType.PICK,
            ManipulationTaskType.PICK_AND_PLACE,
            ManipulationTaskType.GRASP,
            ManipulationTaskType.SORT,
            ManipulationTaskType.STACK,
        }

        place_tasks = {
            ManipulationTaskType.PLACE,
            ManipulationTaskType.PICK_AND_PLACE,
            ManipulationTaskType.SORT,
            ManipulationTaskType.STACK,
        }

        if request.task_type in pick_tasks:
            if grasp is None:
                raise ManipulationPlanningError(
                    "Grasp required for picking task"
                )

            steps.extend(
                [
                    self._step(
                        ManipulationStepType.MOVE_TO_PREGRASP,
                        "Move to pregrasp pose.",
                        target_pose=grasp.grasp_pose,
                    ),
                    self._step(
                        ManipulationStepType.ALIGN_GRIPPER,
                        "Align end effector with target object.",
                        target_pose=grasp.grasp_pose,
                    ),
                    self._step(
                        ManipulationStepType.APPROACH_OBJECT,
                        "Approach target object.",
                        target_pose=grasp.grasp_pose,
                    ),
                    self._step(
                        ManipulationStepType.CLOSE_GRIPPER,
                        "Close gripper using approved force limits.",
                    ),
                    self._step(
                        ManipulationStepType.VERIFY_GRASP,
                        "Verify object grasp.",
                        requires_verification=True,
                    ),
                    self._step(
                        ManipulationStepType.LIFT_OBJECT,
                        "Lift object within workspace constraints.",
                    ),
                ]
            )

        if request.task_type in place_tasks:
            if request.target_pose is None:
                raise ManipulationPlanningError(
                    "Placement target is required"
                )

            steps.extend(
                [
                    self._step(
                        ManipulationStepType.TRANSPORT_OBJECT,
                        "Transport object to placement region.",
                        target_pose=request.target_pose,
                    ),
                    self._step(
                        ManipulationStepType.MOVE_TO_PLACE,
                        "Align with target placement pose.",
                        target_pose=request.target_pose,
                    ),
                    self._step(
                        ManipulationStepType.LOWER_OBJECT,
                        "Lower object into placement region.",
                        target_pose=request.target_pose,
                    ),
                    self._step(
                        ManipulationStepType.OPEN_GRIPPER,
                        "Release object.",
                    ),
                    self._step(
                        ManipulationStepType.RETREAT,
                        "Retreat end effector.",
                    ),
                    self._step(
                        ManipulationStepType.VERIFY_PLACEMENT,
                        "Verify final object placement.",
                        requires_verification=True,
                    ),
                ]
            )

        if request.task_type == ManipulationTaskType.RELEASE:
            steps.append(
                self._step(
                    ManipulationStepType.OPEN_GRIPPER,
                    "Release currently held object.",
                )
            )

        if not steps:
            raise ManipulationPlanningError(
                "Task type requires a specialized planner"
            )

        if len(steps) > MAX_MANIPULATION_STEPS:
            raise ManipulationValidationError(
                "Too many manipulation steps"
            )

        return steps

    # --------------------------------------------------------
    # Safety Assessment
    # --------------------------------------------------------

    @staticmethod
    def assess_safety(
        *,
        robot_state: RobotManipulationState,
        target_object: ManipulationObject,
        payload_compatible: bool | None,
        workspace_valid: bool,
        reachability: ReachabilityStatus,
        collision: CollisionStatus,
    ) -> ManipulationSafetyAssessment:

        findings: list[str] = []

        if robot_state.emergency_stop_active:
            findings.append("Emergency stop is active.")

        if robot_state.safety_fault_active:
            findings.append("Robot safety fault is active.")

        if not robot_state.robot_available:
            findings.append("Robot is unavailable.")

        if payload_compatible is False:
            findings.append("Payload exceeds rated capacity.")

        if payload_compatible is None:
            findings.append("Object mass is unknown.")

        if not workspace_valid:
            findings.append("Target lies outside workspace bounds.")

        if reachability != ReachabilityStatus.REACHABLE:
            findings.append(
                f"Reachability is {reachability.value}."
            )

        if collision != CollisionStatus.CLEAR:
            findings.append(
                f"Collision check is {collision.value}."
            )

        if target_object.hazardous:
            findings.append(
                "Object is marked hazardous."
            )

        if target_object.fragile:
            findings.append(
                "Object is marked fragile."
            )

        hard_reject = (
            robot_state.emergency_stop_active
            or robot_state.safety_fault_active
            or not robot_state.robot_available
            or payload_compatible is False
            or not workspace_valid
            or reachability == ReachabilityStatus.UNREACHABLE
            or collision == CollisionStatus.COLLISION
        )

        if hard_reject:
            return ManipulationSafetyAssessment(
                decision=SafetyDecision.REJECT,
                risk_level=ManipulationRiskLevel.CRITICAL,
                collision_status=collision,
                reachability_status=reachability,
                payload_compatible=payload_compatible,
                workspace_valid=workspace_valid,
                requires_human_review=True,
                findings=tuple(findings),
            )

        uncertain = (
            payload_compatible is None
            or reachability == ReachabilityStatus.UNKNOWN
            or collision == CollisionStatus.UNKNOWN
            or target_object.hazardous
            or target_object.fragile
            or target_object.perception_confidence < 0.85
        )

        if uncertain:
            return ManipulationSafetyAssessment(
                decision=SafetyDecision.REVIEW_REQUIRED,
                risk_level=ManipulationRiskLevel.HIGH,
                collision_status=collision,
                reachability_status=reachability,
                payload_compatible=payload_compatible,
                workspace_valid=workspace_valid,
                requires_human_review=True,
                findings=tuple(findings),
            )

        return ManipulationSafetyAssessment(
            decision=SafetyDecision.ALLOW_SIMULATION,
            risk_level=ManipulationRiskLevel.LOW,
            collision_status=collision,
            reachability_status=reachability,
            payload_compatible=payload_compatible,
            workspace_valid=workspace_valid,
            requires_human_review=False,
            findings=tuple(findings),
        )

    # --------------------------------------------------------
    # Plan Manipulation
    # --------------------------------------------------------

    async def plan_manipulation(
        self,
        *,
        scope: ManipulationScope,
        request: ManipulationTaskRequest,
    ) -> ManipulationPlan:

        if request.tenant_id != scope.tenant_id:
            raise ManipulationPermissionError(
                "Tenant mismatch"
            )

        await self._authorize(
            scope=scope,
            robot_id=request.robot_id,
            permission="manipulation:plan",
        )

        robot_state = await self.state_provider.get_robot_state(
            tenant_id=scope.tenant_id,
            robot_id=request.robot_id,
        )

        if robot_state is None:
            raise ManipulationNotFoundError(
                "Robot state not found"
            )

        if (
            robot_state.tenant_id != scope.tenant_id
            or robot_state.robot_id != request.robot_id
        ):
            raise ManipulationPermissionError(
                "Robot state scope mismatch"
            )

        target_object = (
            await self.perception_provider.get_object(
                tenant_id=scope.tenant_id,
                robot_id=request.robot_id,
                object_id=request.object_id,
            )
        )

        if target_object is None:
            raise ManipulationNotFoundError(
                "Target object not found"
            )

        if target_object.object_id != request.object_id:
            raise ManipulationValidationError(
                "Perception provider returned wrong object"
            )

        payload_compatible = self.check_payload(
            target_object=target_object,
            end_effector=robot_state.end_effector,
        )

        workspace_valid = (
            robot_state.workspace_bounds.contains(
                target_object.pose.position
            )
        )

        if request.target_pose is not None:
            workspace_valid = (
                workspace_valid
                and robot_state.workspace_bounds.contains(
                    request.target_pose.position
                )
            )

        grasp: GraspCandidate | None = None

        pick_tasks = {
            ManipulationTaskType.PICK,
            ManipulationTaskType.PICK_AND_PLACE,
            ManipulationTaskType.GRASP,
            ManipulationTaskType.SORT,
            ManipulationTaskType.STACK,
        }

        reasoning: list[str] = []

        if request.task_type in pick_tasks:
            grasp_result = await self.select_grasp(
                scope=scope,
                robot_state=robot_state,
                target_object=target_object,
                maximum_allowed_force_n=(
                    request.maximum_allowed_force_n
                ),
            )

            grasp = grasp_result.selected_grasp
            reasoning.extend(grasp_result.reasoning)

        steps: list[ManipulationStep] = []

        if (
            request.task_type not in pick_tasks
            or grasp is not None
        ):
            steps = self.build_task_steps(
                request=request,
                grasp=grasp,
            )

        planning_target = (
            grasp.grasp_pose
            if grasp is not None
            else request.target_pose or target_object.pose
        )

        reachability = (
            await self.ik_provider.check_reachability(
                tenant_id=scope.tenant_id,
                robot_id=request.robot_id,
                target_pose=planning_target,
            )
        )

        collision = CollisionStatus.UNKNOWN
        planned_trajectory: ManipulationTrajectory | None = None

        if (
            reachability == ReachabilityStatus.REACHABLE
            and workspace_valid
            and not robot_state.emergency_stop_active
            and not robot_state.safety_fault_active
        ):
            planned_trajectory = (
                await self.trajectory_planner.plan_trajectory(
                    tenant_id=scope.tenant_id,
                    robot_id=request.robot_id,
                    start_pose=robot_state.end_effector_pose,
                    target_pose=planning_target,
                )
            )

            if planned_trajectory is not None:
                collision = (
                    await self.collision_checker.check_collision(
                        tenant_id=scope.tenant_id,
                        robot_id=request.robot_id,
                        trajectory=planned_trajectory,
                    )
                )

                if steps:
                    steps[0].trajectory = planned_trajectory
                    steps[0].estimated_duration_seconds = (
                        planned_trajectory
                        .estimated_duration_seconds
                    )

        safety = self.assess_safety(
            robot_state=robot_state,
            target_object=target_object,
            payload_compatible=payload_compatible,
            workspace_valid=workspace_valid,
            reachability=reachability,
            collision=collision,
        )

        if request.task_type in pick_tasks and grasp is None:
            safety = ManipulationSafetyAssessment(
                decision=SafetyDecision.REJECT,
                risk_level=ManipulationRiskLevel.HIGH,
                collision_status=collision,
                reachability_status=reachability,
                payload_compatible=payload_compatible,
                workspace_valid=workspace_valid,
                requires_human_review=True,
                findings=(
                    *safety.findings,
                    "No valid grasp was found.",
                ),
            )

        if planned_trajectory is None:
            safety = ManipulationSafetyAssessment(
                decision=SafetyDecision.REJECT,
                risk_level=ManipulationRiskLevel.HIGH,
                collision_status=collision,
                reachability_status=reachability,
                payload_compatible=payload_compatible,
                workspace_valid=workspace_valid,
                requires_human_review=True,
                findings=(
                    *safety.findings,
                    "No validated trajectory was generated.",
                ),
            )

        if safety.decision == SafetyDecision.REJECT:
            status = ManipulationPlanStatus.VALIDATION_FAILED
        elif safety.decision == SafetyDecision.REVIEW_REQUIRED:
            status = ManipulationPlanStatus.REVIEW_REQUIRED
        else:
            status = ManipulationPlanStatus.PLANNED

        estimated_duration = sum(
            step.estimated_duration_seconds
            for step in steps
        )

        plan = ManipulationPlan(
            plan_id=generate_id(),
            tenant_id=scope.tenant_id,
            robot_id=request.robot_id,
            object_id=request.object_id,
            task_type=request.task_type,
            status=status,
            steps=steps,
            selected_grasp=grasp,
            safety_assessment=safety,
            simulation_result=None,
            estimated_duration_seconds=estimated_duration,
            requires_human_review=(
                safety.requires_human_review
            ),
            reasoning=reasoning + list(safety.findings),
            simulation_only=True,
        )

        if (
            request.require_simulation
            and safety.decision != SafetyDecision.REJECT
        ):
            if self.simulator is None:
                plan.status = (
                    ManipulationPlanStatus.REVIEW_REQUIRED
                )
                plan.requires_human_review = True
                plan.reasoning.append(
                    "Simulation service is unavailable."
                )
            else:
                result = await self.simulator.simulate(
                    tenant_id=scope.tenant_id,
                    robot_id=request.robot_id,
                    plan=plan,
                )

                plan.simulation_result = result

                if result.status == SimulationStatus.FAILED:
                    plan.status = (
                        ManipulationPlanStatus.SIMULATION_FAILED
                    )
                    plan.requires_human_review = True

                elif (
                    result.status == SimulationStatus.PASSED
                    and result.collision_detected is False
                    and result.object_slippage_detected is False
                    and safety.decision
                    == SafetyDecision.ALLOW_SIMULATION
                ):
                    plan.status = (
                        ManipulationPlanStatus
                        .APPROVED_FOR_SIMULATION
                    )

                else:
                    plan.status = (
                        ManipulationPlanStatus.REVIEW_REQUIRED
                    )
                    plan.requires_human_review = True

        if self.repository is not None:
            await self.repository.save(plan=plan)

        if self.audit_sink is not None:
            await self.audit_sink.record(
                tenant_id=scope.tenant_id,
                actor_id=scope.user_id,
                action="manipulation_agent.plan_created",
                resource_id=plan.plan_id,
                metadata={
                    "robot_id": request.robot_id,
                    "object_id": request.object_id,
                    "task_type": request.task_type.value,
                    "status": plan.status.value,
                    "simulation_only": True,
                },
            )

        return plan

    # --------------------------------------------------------
    # Batch Planning
    # --------------------------------------------------------

    async def plan_batch(
        self,
        *,
        scope: ManipulationScope,
        requests: Sequence[ManipulationTaskRequest],
    ) -> list[ManipulationPlan]:

        if len(requests) > MAX_BATCH_SIZE:
            raise ManipulationValidationError(
                "Batch size exceeds limit"
            )

        semaphore = asyncio.Semaphore(5)

        async def process(
            request: ManipulationTaskRequest,
        ) -> ManipulationPlan:
            async with semaphore:
                return await self.plan_manipulation(
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
        plan: ManipulationPlan,
    ) -> dict[str, Any]:

        return {
            "plan_id": plan.plan_id,
            "tenant_id": plan.tenant_id,
            "robot_id": plan.robot_id,
            "object_id": plan.object_id,
            "task_type": plan.task_type.value,
            "status": plan.status.value,
            "created_at": plan.created_at.isoformat(),
            "estimated_duration_seconds": (
                plan.estimated_duration_seconds
            ),
            "selected_grasp": (
                {
                    "grasp_id": plan.selected_grasp.grasp_id,
                    "grasp_type": (
                        plan.selected_grasp.grasp_type.value
                    ),
                    "confidence": (
                        plan.selected_grasp.confidence
                    ),
                }
                if plan.selected_grasp is not None
                else None
            ),
            "steps": [
                {
                    "step_id": step.step_id,
                    "step_type": step.step_type.value,
                    "description": step.description,
                    "estimated_duration_seconds": (
                        step.estimated_duration_seconds
                    ),
                    "requires_verification": (
                        step.requires_verification
                    ),
                    "trajectory_id": (
                        step.trajectory.trajectory_id
                        if step.trajectory is not None
                        else None
                    ),
                }
                for step in plan.steps
            ],
            "safety": {
                "decision": (
                    plan.safety_assessment.decision.value
                ),
                "risk_level": (
                    plan.safety_assessment.risk_level.value
                ),
                "collision_status": (
                    plan.safety_assessment
                    .collision_status.value
                ),
                "reachability_status": (
                    plan.safety_assessment
                    .reachability_status.value
                ),
                "payload_compatible": (
                    plan.safety_assessment
                    .payload_compatible
                ),
                "workspace_valid": (
                    plan.safety_assessment
                    .workspace_valid
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
                    "success_probability": (
                        plan.simulation_result
                        .success_probability
                    ),
                    "collision_detected": (
                        plan.simulation_result
                        .collision_detected
                    ),
                    "object_slippage_detected": (
                        plan.simulation_result
                        .object_slippage_detected
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
            "physical_robot_execution": False,
            "simulation_only": True,
        }


# ============================================================
# Factory
# ============================================================

def create_manipulation_agent(
    *,
    state_provider: ManipulationStateProvider,
    perception_provider: ManipulationPerceptionProvider,
    grasp_planner: GraspPlanner,
    ik_provider: InverseKinematicsProvider,
    trajectory_planner: ManipulationTrajectoryPlanner,
    collision_checker: ManipulationCollisionChecker,
    authorizer: ManipulationAuthorizer,
    simulator: ManipulationSimulator | None = None,
    repository: ManipulationPlanRepository | None = None,
    audit_sink: ManipulationAuditSink | None = None,
) -> ManipulationAgent:

    return ManipulationAgent(
        state_provider=state_provider,
        perception_provider=perception_provider,
        grasp_planner=grasp_planner,
        ik_provider=ik_provider,
        trajectory_planner=trajectory_planner,
        collision_checker=collision_checker,
        authorizer=authorizer,
        simulator=simulator,
        repository=repository,
        audit_sink=audit_sink,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "ManipulationTaskType",
    "ManipulationStepType",
    "EndEffectorType",
    "GraspType",
    "ManipulationPlanStatus",
    "ManipulationRiskLevel",
    "CollisionStatus",
    "ReachabilityStatus",
    "SimulationStatus",
    "SafetyDecision",
    "ManipulationAgentError",
    "ManipulationValidationError",
    "ManipulationNotFoundError",
    "ManipulationPermissionError",
    "ManipulationServiceUnavailableError",
    "ManipulationPlanningError",
    "ManipulationScope",
    "Vector3",
    "Quaternion",
    "Pose3D",
    "BoundingBox3D",
    "WorkspaceBounds",
    "ManipulationObject",
    "EndEffectorConfiguration",
    "RobotManipulationState",
    "GraspCandidate",
    "GraspPlanningResult",
    "TrajectoryWaypoint",
    "ManipulationTrajectory",
    "ManipulationTaskRequest",
    "ManipulationStep",
    "ManipulationSafetyAssessment",
    "ManipulationSimulationResult",
    "ManipulationPlan",
    "ManipulationStateProvider",
    "ManipulationPerceptionProvider",
    "GraspPlanner",
    "InverseKinematicsProvider",
    "ManipulationTrajectoryPlanner",
    "ManipulationCollisionChecker",
    "ManipulationSimulator",
    "ManipulationAuthorizer",
    "ManipulationAuditSink",
    "ManipulationPlanRepository",
    "ManipulationAgent",
    "create_manipulation_agent",
]
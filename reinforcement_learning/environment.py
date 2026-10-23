"""Environment module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/environment.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Simulation Environment Management Engine

Python 3.11+
Dependencies: Standard library only

Features:
    - Robotics simulation environment creation
    - 2D and 3D environment configuration
    - Warehouse, factory, laboratory and outdoor environments
    - Robot spawn positions and goal locations
    - Static and dynamic obstacle definitions
    - Terrain, gravity, friction and lighting configuration
    - Sensor simulation settings
    - Domain randomization integration
    - Curriculum learning integration
    - Environment reset and deterministic seeding
    - Simulation time-step management
    - Collision and proximity evaluation
    - Robot movement simulation
    - Episode termination and reward calculation
    - Environment snapshots
    - Multi-robot simulation
    - Multi-tenant isolation
    - Thread-safe in-memory storage
    - Audit logging
    - JSON export

Integrations:
    action_space.py
    curriculum.py
    domain_randomization.py
    checkpoints.py
    simulator.py
    simulation_adapter.py
    robot_state.py
    sensor_manager.py
    navigation_service.py
    safety_monitor.py
    evaluation_service.py

Safety:
    Simulation-only environment.

    This module never sends commands to physical robots.
    Its simplified kinematics and collision calculations
    are not substitutes for validated physics engines,
    hardware safety systems or deployment certification.
"""

from __future__ import annotations

import copy
import json
import math
import random
import threading
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class EnvironmentError(Exception):
    pass


class EnvironmentValidationError(EnvironmentError):
    pass


class EnvironmentNotFound(EnvironmentError):
    pass


class EnvironmentConflict(EnvironmentError):
    pass


class EnvironmentPermissionDenied(EnvironmentError):
    pass


# ============================================================
# ENUMS
# ============================================================

class EnvironmentType(str, Enum):
    WAREHOUSE = "warehouse"
    FACTORY = "factory"
    LABORATORY = "laboratory"
    OFFICE = "office"
    HOSPITAL = "hospital"
    OUTDOOR = "outdoor"
    URBAN = "urban"
    TEST_ARENA = "test_arena"
    CUSTOM = "custom"


class EnvironmentStatus(str, Enum):
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"


class ObstacleType(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class TerrainType(str, Enum):
    FLAT = "flat"
    ROUGH = "rough"
    SLOPE = "slope"
    MIXED = "mixed"


class EpisodeTermination(str, Enum):
    NONE = "none"
    GOAL_REACHED = "goal_reached"
    COLLISION = "collision"
    OUT_OF_BOUNDS = "out_of_bounds"
    TIME_LIMIT = "time_limit"
    MANUAL = "manual"


class ActionType(str, Enum):
    VELOCITY = "velocity"
    STOP = "stop"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            character.isalnum() or character in "._-"
            for character in value
        )
    ):
        raise EnvironmentValidationError(
            f"Invalid {name}"
        )

    return value


def validate_text(
    value: str,
    name: str,
    *,
    allow_empty: bool = False,
    maximum: int = 4000,
) -> str:
    if not isinstance(value, str):
        raise EnvironmentValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if not allow_empty and not result:
        raise EnvironmentValidationError(
            f"{name} cannot be empty"
        )

    if len(result) > maximum:
        raise EnvironmentValidationError(
            f"{name} exceeds maximum length"
        )

    return result


def number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise EnvironmentValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise EnvironmentValidationError(
            f"{name} must be finite"
        )

    return result


def nonnegative(value: Any, name: str) -> float:
    result = number(value, name)

    if result < 0:
        raise EnvironmentValidationError(
            f"{name} cannot be negative"
        )

    return result


def positive(value: Any, name: str) -> float:
    result = number(value, name)

    if result <= 0:
        raise EnvironmentValidationError(
            f"{name} must be positive"
        )

    return result


def validate_integer(
    value: int,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 10000000,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise EnvironmentValidationError(
            f"Invalid {name}"
        )

    return value


def distance_3d(
    first: "Vector3",
    second: "Vector3",
) -> float:
    return math.sqrt(
        (first.x - second.x) ** 2
        + (first.y - second.y) ** 2
        + (first.z - second.z) ** 2
    )


def json_mapping(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EnvironmentValidationError(
            "Expected mapping"
        )

    try:
        result = copy.deepcopy(dict(value))

        json.dumps(
            result,
            allow_nan=False,
        )

    except (
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise EnvironmentValidationError(
            "Mapping must contain JSON-compatible values"
        ) from exc

    return result


# ============================================================
# GEOMETRY
# ============================================================

@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float = 0.0

    def __post_init__(self) -> None:
        number(self.x, "x")
        number(self.y, "y")
        number(self.z, "z")

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


@dataclass(frozen=True)
class EnvironmentBounds:
    minimum: Vector3
    maximum: Vector3

    def __post_init__(self) -> None:
        if not isinstance(self.minimum, Vector3):
            raise EnvironmentValidationError(
                "Invalid minimum bounds"
            )

        if not isinstance(self.maximum, Vector3):
            raise EnvironmentValidationError(
                "Invalid maximum bounds"
            )

        if (
            self.minimum.x >= self.maximum.x
            or self.minimum.y >= self.maximum.y
            or self.minimum.z > self.maximum.z
        ):
            raise EnvironmentValidationError(
                "Invalid environment dimensions"
            )

    def contains(
        self,
        position: Vector3,
        radius: float = 0.0,
    ) -> bool:
        return (
            self.minimum.x + radius
            <= position.x
            <= self.maximum.x - radius
            and self.minimum.y + radius
            <= position.y
            <= self.maximum.y - radius
            and self.minimum.z
            <= position.z
            <= self.maximum.z
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum": self.minimum.to_dict(),
            "maximum": self.maximum.to_dict(),
        }


# ============================================================
# PHYSICS CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PhysicsConfig:
    gravity: float = 9.81
    friction: float = 0.7
    restitution: float = 0.0
    linear_damping: float = 0.05
    time_step: float = 0.05

    def __post_init__(self) -> None:
        nonnegative(self.gravity, "gravity")
        nonnegative(self.friction, "friction")
        nonnegative(self.restitution, "restitution")
        nonnegative(
            self.linear_damping,
            "linear_damping",
        )

        positive(self.time_step, "time_step")

        if self.time_step > 1:
            raise EnvironmentValidationError(
                "Time step must not exceed one second"
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "gravity": self.gravity,
            "friction": self.friction,
            "restitution": self.restitution,
            "linear_damping": self.linear_damping,
            "time_step": self.time_step,
        }


# ============================================================
# SENSOR CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SensorSimulationConfig:
    camera_enabled: bool = True
    lidar_enabled: bool = True
    imu_enabled: bool = True

    camera_noise_std: float = 0.01
    lidar_noise_std: float = 0.02
    imu_noise_std: float = 0.005

    sensor_dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "camera_enabled",
            "lidar_enabled",
            "imu_enabled",
        ):
            if not isinstance(
                getattr(self, name),
                bool,
            ):
                raise EnvironmentValidationError(
                    f"{name} must be boolean"
                )

        for name in (
            "camera_noise_std",
            "lidar_noise_std",
            "imu_noise_std",
        ):
            nonnegative(
                getattr(self, name),
                name,
            )

        probability = number(
            self.sensor_dropout_probability,
            "sensor_dropout_probability",
        )

        if not 0 <= probability <= 1:
            raise EnvironmentValidationError(
                "Dropout probability must be in [0, 1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_enabled": self.camera_enabled,
            "lidar_enabled": self.lidar_enabled,
            "imu_enabled": self.imu_enabled,
            "camera_noise_std": self.camera_noise_std,
            "lidar_noise_std": self.lidar_noise_std,
            "imu_noise_std": self.imu_noise_std,
            "sensor_dropout_probability": (
                self.sensor_dropout_probability
            ),
        }


# ============================================================
# OBSTACLE DEFINITION
# ============================================================

@dataclass(frozen=True)
class SimulationObstacle:
    obstacle_id: str
    name: str

    position: Vector3
    radius: float

    obstacle_type: ObstacleType = ObstacleType.STATIC

    velocity: Vector3 = field(
        default_factory=lambda: Vector3(0, 0, 0)
    )

    collision_enabled: bool = True

    def __post_init__(self) -> None:
        validate_id(self.obstacle_id, "obstacle_id")
        validate_text(self.name, "name")

        if not isinstance(self.position, Vector3):
            raise EnvironmentValidationError(
                "Invalid obstacle position"
            )

        positive(self.radius, "radius")

        if not isinstance(
            self.obstacle_type,
            ObstacleType,
        ):
            raise EnvironmentValidationError(
                "Invalid obstacle type"
            )

        if not isinstance(self.velocity, Vector3):
            raise EnvironmentValidationError(
                "Invalid obstacle velocity"
            )

        if not isinstance(
            self.collision_enabled,
            bool,
        ):
            raise EnvironmentValidationError(
                "collision_enabled must be boolean"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "obstacle_id": self.obstacle_id,
            "name": self.name,
            "position": self.position.to_dict(),
            "radius": self.radius,
            "obstacle_type": self.obstacle_type.value,
            "velocity": self.velocity.to_dict(),
            "collision_enabled": self.collision_enabled,
        }


# ============================================================
# ROBOT DEFINITION
# ============================================================

@dataclass(frozen=True)
class SimulationRobot:
    robot_id: str
    name: str

    spawn_position: Vector3
    goal_position: Vector3

    radius: float = 0.35

    maximum_speed: float = 1.5

    goal_tolerance: float = 0.5

    def __post_init__(self) -> None:
        validate_id(self.robot_id, "robot_id")
        validate_text(self.name, "name")

        if not isinstance(
            self.spawn_position,
            Vector3,
        ):
            raise EnvironmentValidationError(
                "Invalid spawn position"
            )

        if not isinstance(
            self.goal_position,
            Vector3,
        ):
            raise EnvironmentValidationError(
                "Invalid goal position"
            )

        positive(self.radius, "radius")
        positive(self.maximum_speed, "maximum_speed")
        positive(self.goal_tolerance, "goal_tolerance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "name": self.name,
            "spawn_position": (
                self.spawn_position.to_dict()
            ),
            "goal_position": (
                self.goal_position.to_dict()
            ),
            "radius": self.radius,
            "maximum_speed": self.maximum_speed,
            "goal_tolerance": self.goal_tolerance,
        }


# ============================================================
# ENVIRONMENT DEFINITION
# ============================================================

@dataclass(frozen=True)
class EnvironmentDefinition:
    environment_id: str

    tenant_id: str

    name: str

    environment_type: EnvironmentType

    bounds: EnvironmentBounds

    robots: tuple[SimulationRobot, ...]

    obstacles: tuple[SimulationObstacle, ...] = ()

    physics: PhysicsConfig = field(
        default_factory=PhysicsConfig
    )

    sensors: SensorSimulationConfig = field(
        default_factory=SensorSimulationConfig
    )

    terrain: TerrainType = TerrainType.FLAT

    lighting_intensity: float = 1.0

    maximum_steps: int = 1000

    random_seed: int = 42

    curriculum_stage_id: str | None = None

    domain_randomization_profile_id: str | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    description: str = ""

    def __post_init__(self) -> None:
        validate_id(
            self.environment_id,
            "environment_id",
        )

        validate_id(
            self.tenant_id,
            "tenant_id",
        )

        validate_text(self.name, "name")

        if not isinstance(
            self.environment_type,
            EnvironmentType,
        ):
            raise EnvironmentValidationError(
                "Invalid environment type"
            )

        if not isinstance(
            self.bounds,
            EnvironmentBounds,
        ):
            raise EnvironmentValidationError(
                "Invalid environment bounds"
            )

        if not self.robots:
            raise EnvironmentValidationError(
                "Environment requires at least one robot"
            )

        robot_ids = set()

        for robot in self.robots:
            if not isinstance(
                robot,
                SimulationRobot,
            ):
                raise EnvironmentValidationError(
                    "Invalid robot definition"
                )

            if robot.robot_id in robot_ids:
                raise EnvironmentValidationError(
                    "Duplicate robot ID"
                )

            robot_ids.add(robot.robot_id)

            if not self.bounds.contains(
                robot.spawn_position,
                robot.radius,
            ):
                raise EnvironmentValidationError(
                    "Robot spawn outside environment"
                )

            if not self.bounds.contains(
                robot.goal_position,
                robot.radius,
            ):
                raise EnvironmentValidationError(
                    "Robot goal outside environment"
                )

        obstacle_ids = set()

        for obstacle in self.obstacles:
            if not isinstance(
                obstacle,
                SimulationObstacle,
            ):
                raise EnvironmentValidationError(
                    "Invalid obstacle definition"
                )

            if obstacle.obstacle_id in obstacle_ids:
                raise EnvironmentValidationError(
                    "Duplicate obstacle ID"
                )

            obstacle_ids.add(obstacle.obstacle_id)

            if not self.bounds.contains(
                obstacle.position,
                obstacle.radius,
            ):
                raise EnvironmentValidationError(
                    "Obstacle outside environment"
                )

        if not isinstance(self.physics, PhysicsConfig):
            raise EnvironmentValidationError(
                "Invalid physics configuration"
            )

        if not isinstance(
            self.sensors,
            SensorSimulationConfig,
        ):
            raise EnvironmentValidationError(
                "Invalid sensor configuration"
            )

        if not isinstance(self.terrain, TerrainType):
            raise EnvironmentValidationError(
                "Invalid terrain type"
            )

        nonnegative(
            self.lighting_intensity,
            "lighting_intensity",
        )

        validate_integer(
            self.maximum_steps,
            "maximum_steps",
            minimum=1,
        )

        validate_integer(
            self.random_seed,
            "random_seed",
            maximum=2**63 - 1,
        )

        if self.curriculum_stage_id is not None:
            validate_id(
                self.curriculum_stage_id,
                "curriculum_stage_id",
            )

        if (
            self.domain_randomization_profile_id
            is not None
        ):
            validate_id(
                self.domain_randomization_profile_id,
                "domain_randomization_profile_id",
            )

        json_mapping(self.metadata)

        validate_text(
            self.description,
            "description",
            allow_empty=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "environment_type": (
                self.environment_type.value
            ),
            "bounds": self.bounds.to_dict(),
            "robots": [
                robot.to_dict()
                for robot in self.robots
            ],
            "obstacles": [
                obstacle.to_dict()
                for obstacle in self.obstacles
            ],
            "physics": self.physics.to_dict(),
            "sensors": self.sensors.to_dict(),
            "terrain": self.terrain.value,
            "lighting_intensity": (
                self.lighting_intensity
            ),
            "maximum_steps": self.maximum_steps,
            "random_seed": self.random_seed,
            "curriculum_stage_id": (
                self.curriculum_stage_id
            ),
            "domain_randomization_profile_id": (
                self.domain_randomization_profile_id
            ),
            "metadata": copy.deepcopy(
                dict(self.metadata)
            ),
            "description": self.description,
            "simulation_only": True,
        }


# ============================================================
# ROBOT ACTION
# ============================================================

@dataclass(frozen=True)
class RobotAction:
    robot_id: str

    action_type: ActionType

    linear_velocity_x: float = 0.0
    linear_velocity_y: float = 0.0
    linear_velocity_z: float = 0.0

    def __post_init__(self) -> None:
        validate_id(self.robot_id, "robot_id")

        if not isinstance(
            self.action_type,
            ActionType,
        ):
            raise EnvironmentValidationError(
                "Invalid action type"
            )

        for name in (
            "linear_velocity_x",
            "linear_velocity_y",
            "linear_velocity_z",
        ):
            number(
                getattr(self, name),
                name,
            )

    def velocity(self) -> Vector3:
        if self.action_type == ActionType.STOP:
            return Vector3(0, 0, 0)

        return Vector3(
            self.linear_velocity_x,
            self.linear_velocity_y,
            self.linear_velocity_z,
        )


# ============================================================
# RUNTIME ROBOT STATE
# ============================================================

@dataclass
class RobotRuntimeState:
    robot_id: str

    position: Vector3

    velocity: Vector3 = field(
        default_factory=lambda: Vector3(0, 0, 0)
    )

    goal_reached: bool = False
    collided: bool = False
    out_of_bounds: bool = False

    total_reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "goal_reached": self.goal_reached,
            "collided": self.collided,
            "out_of_bounds": self.out_of_bounds,
            "total_reward": round(
                self.total_reward,
                6,
            ),
        }


# ============================================================
# STEP RESULT
# ============================================================

@dataclass(frozen=True)
class EnvironmentStepResult:
    environment_id: str

    step_number: int

    simulation_time: float

    observations: Mapping[str, Any]

    rewards: Mapping[str, float]

    terminated: bool

    termination_reason: EpisodeTermination

    collisions: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "step_number": self.step_number,
            "simulation_time": self.simulation_time,
            "observations": copy.deepcopy(
                dict(self.observations)
            ),
            "rewards": dict(self.rewards),
            "terminated": self.terminated,
            "termination_reason": (
                self.termination_reason.value
            ),
            "collisions": [
                dict(item)
                for item in self.collisions
            ],
            "simulation_only": True,
        }


# ============================================================
# ENVIRONMENT RUNTIME
# ============================================================

@dataclass
class EnvironmentRuntime:
    definition: EnvironmentDefinition

    status: EnvironmentStatus = (
        EnvironmentStatus.CREATED
    )

    robot_states: dict[
        str,
        RobotRuntimeState,
    ] = field(default_factory=dict)

    obstacle_positions: dict[
        str,
        Vector3,
    ] = field(default_factory=dict)

    step_number: int = 0

    simulation_time: float = 0.0

    episode_number: int = 0

    seed: int = 42

    termination_reason: EpisodeTermination = (
        EpisodeTermination.NONE
    )

    last_collisions: list[
        dict[str, Any]
    ] = field(default_factory=list)

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition.to_dict(),
            "status": self.status.value,
            "robot_states": {
                key: value.to_dict()
                for key, value in self.robot_states.items()
            },
            "obstacle_positions": {
                key: value.to_dict()
                for key, value in self.obstacle_positions.items()
            },
            "step_number": self.step_number,
            "simulation_time": self.simulation_time,
            "episode_number": self.episode_number,
            "seed": self.seed,
            "termination_reason": (
                self.termination_reason.value
            ),
            "last_collisions": copy.deepcopy(
                self.last_collisions
            ),
            "updated_at": self.updated_at.isoformat(),
            "simulation_only": True,
        }


# ============================================================
# ENGINE CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class EnvironmentEngineConfig:
    maximum_environments: int = 10000
    maximum_robots_per_environment: int = 100
    maximum_obstacles_per_environment: int = 10000
    maximum_audit_events: int = 20000

    collision_penalty: float = -100.0
    goal_reward: float = 100.0
    progress_reward_scale: float = 5.0
    step_penalty: float = -0.01

    terminate_on_collision: bool = True

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_environments",
            "maximum_robots_per_environment",
            "maximum_obstacles_per_environment",
            "maximum_audit_events",
        ):
            validate_integer(
                getattr(self, name),
                name,
                minimum=1,
            )

        for name in (
            "collision_penalty",
            "goal_reward",
            "progress_reward_scale",
            "step_penalty",
        ):
            number(
                getattr(self, name),
                name,
            )

        if not isinstance(
            self.terminate_on_collision,
            bool,
        ):
            raise EnvironmentValidationError(
                "terminate_on_collision must be boolean"
            )

        if self.simulation_only is not True:
            raise EnvironmentValidationError(
                "Only simulation mode is supported"
            )


# ============================================================
# ENVIRONMENT ENGINE
# ============================================================

class RoboticsEnvironmentEngine:

    def __init__(
        self,
        config: EnvironmentEngineConfig | None = None,
    ) -> None:
        self.config = (
            config or EnvironmentEngineConfig()
        )

        self._environments: dict[
            tuple[str, str],
            EnvironmentRuntime,
        ] = {}

        self._audit: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _log(
        self,
        *,
        tenant_id: str,
        environment_id: str,
        action: str,
        actor_id: str = "system",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._audit.append(
            {
                "event_id": new_id(),
                "timestamp": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "environment_id": environment_id,
                "action": action,
                "actor_id": actor_id,
                "details": copy.deepcopy(
                    dict(details or {})
                ),
            }
        )

    # --------------------------------------------------------
    # LOOKUP
    # --------------------------------------------------------

    def _runtime(
        self,
        tenant_id: str,
        environment_id: str,
    ) -> EnvironmentRuntime:
        validate_id(tenant_id, "tenant_id")
        validate_id(
            environment_id,
            "environment_id",
        )

        runtime = self._environments.get(
            (tenant_id, environment_id)
        )

        if runtime is None:
            raise EnvironmentNotFound(
                "Environment not found"
            )

        return runtime

    # --------------------------------------------------------
    # CREATE ENVIRONMENT
    # --------------------------------------------------------

    def create_environment(
        self,
        definition: EnvironmentDefinition,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise EnvironmentPermissionDenied(
                "Authorized environment creation required"
            )

        if not isinstance(
            definition,
            EnvironmentDefinition,
        ):
            raise EnvironmentValidationError(
                "Invalid environment definition"
            )

        validate_id(actor_id, "actor_id")

        if (
            len(definition.robots)
            > self.config.maximum_robots_per_environment
        ):
            raise EnvironmentValidationError(
                "Too many robots"
            )

        if (
            len(definition.obstacles)
            > self.config.maximum_obstacles_per_environment
        ):
            raise EnvironmentValidationError(
                "Too many obstacles"
            )

        key = (
            definition.tenant_id,
            definition.environment_id,
        )

        with self._lock:
            if key in self._environments:
                raise EnvironmentConflict(
                    "Environment already exists"
                )

            if (
                len(self._environments)
                >= self.config.maximum_environments
            ):
                raise EnvironmentConflict(
                    "Environment capacity reached"
                )

            self._validate_initial_collisions(
                definition
            )

            runtime = EnvironmentRuntime(
                definition=definition,
                seed=definition.random_seed,
            )

            self._initialize_runtime(runtime)

            self._environments[key] = runtime

            self._log(
                tenant_id=definition.tenant_id,
                environment_id=definition.environment_id,
                action="environment_created",
                actor_id=actor_id,
                details={
                    "robot_count": len(
                        definition.robots
                    ),
                    "obstacle_count": len(
                        definition.obstacles
                    ),
                },
            )

            return runtime.to_dict()

    def _validate_initial_collisions(
        self,
        definition: EnvironmentDefinition,
    ) -> None:
        robots = definition.robots
        obstacles = definition.obstacles

        for index, robot in enumerate(robots):
            for other in robots[index + 1:]:
                if distance_3d(
                    robot.spawn_position,
                    other.spawn_position,
                ) <= robot.radius + other.radius:
                    raise EnvironmentValidationError(
                        "Robot spawn positions overlap"
                    )

            for obstacle in obstacles:
                if not obstacle.collision_enabled:
                    continue

                if distance_3d(
                    robot.spawn_position,
                    obstacle.position,
                ) <= robot.radius + obstacle.radius:
                    raise EnvironmentValidationError(
                        "Robot spawn overlaps obstacle"
                    )

    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    def _initialize_runtime(
        self,
        runtime: EnvironmentRuntime,
    ) -> None:
        runtime.robot_states = {
            robot.robot_id: RobotRuntimeState(
                robot_id=robot.robot_id,
                position=robot.spawn_position,
            )
            for robot in runtime.definition.robots
        }

        runtime.obstacle_positions = {
            obstacle.obstacle_id: obstacle.position
            for obstacle in runtime.definition.obstacles
        }

        runtime.step_number = 0
        runtime.simulation_time = 0.0

        runtime.termination_reason = (
            EpisodeTermination.NONE
        )

        runtime.last_collisions = []

        runtime.status = EnvironmentStatus.READY
        runtime.updated_at = utc_now()

    # --------------------------------------------------------
    # RESET ENVIRONMENT
    # --------------------------------------------------------

    def reset_environment(
        self,
        *,
        tenant_id: str,
        environment_id: str,
        seed: int | None = None,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise EnvironmentPermissionDenied(
                "Authorized environment reset required"
            )

        validate_id(actor_id, "actor_id")

        with self._lock:
            runtime = self._runtime(
                tenant_id,
                environment_id,
            )

            if seed is not None:
                runtime.seed = validate_integer(
                    seed,
                    "seed",
                    maximum=2**63 - 1,
                )

            self._initialize_runtime(runtime)

            runtime.episode_number += 1

            self._log(
                tenant_id=tenant_id,
                environment_id=environment_id,
                action="environment_reset",
                actor_id=actor_id,
                details={
                    "episode_number": (
                        runtime.episode_number
                    ),
                    "seed": runtime.seed,
                },
            )

            return runtime.to_dict()

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def start_environment(
        self,
        *,
        tenant_id: str,
        environment_id: str,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise EnvironmentPermissionDenied(
                "Authorized simulation start required"
            )

        validate_id(actor_id, "actor_id")

        with self._lock:
            runtime = self._runtime(
                tenant_id,
                environment_id,
            )

            if runtime.status not in (
                EnvironmentStatus.READY,
                EnvironmentStatus.PAUSED,
            ):
                raise EnvironmentConflict(
                    "Environment cannot start "
                    "from current status"
                )

            runtime.status = EnvironmentStatus.RUNNING
            runtime.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                environment_id=environment_id,
                action="simulation_started",
                actor_id=actor_id,
            )

            return runtime.to_dict()

    # --------------------------------------------------------
    # PAUSE
    # --------------------------------------------------------

    def pause_environment(
        self,
        *,
        tenant_id: str,
        environment_id: str,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise EnvironmentPermissionDenied(
                "Authorized simulation pause required"
            )

        validate_id(actor_id, "actor_id")

        with self._lock:
            runtime = self._runtime(
                tenant_id,
                environment_id,
            )

            if runtime.status != EnvironmentStatus.RUNNING:
                raise EnvironmentConflict(
                    "Environment is not running"
                )

            runtime.status = EnvironmentStatus.PAUSED
            runtime.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                environment_id=environment_id,
                action="simulation_paused",
                actor_id=actor_id,
            )

            return runtime.to_dict()

    # --------------------------------------------------------
    # STEP SIMULATION
    # --------------------------------------------------------

    def step(
        self,
        *,
        tenant_id: str,
        environment_id: str,
        actions: Mapping[str, RobotAction],
    ) -> dict[str, Any]:
        if not isinstance(actions, Mapping):
            raise EnvironmentValidationError(
                "actions must be a mapping"
            )

        with self._lock:
            runtime = self._runtime(
                tenant_id,
                environment_id,
            )

            if runtime.status != EnvironmentStatus.RUNNING:
                raise EnvironmentConflict(
                    "Simulation is not running"
                )

            definition = runtime.definition
            dt = definition.physics.time_step

            robots_by_id = {
                robot.robot_id: robot
                for robot in definition.robots
            }

            for robot_id, action in actions.items():
                if robot_id not in robots_by_id:
                    raise EnvironmentValidationError(
                        f"Unknown robot: {robot_id}"
                    )

                if not isinstance(action, RobotAction):
                    raise EnvironmentValidationError(
                        "Invalid robot action"
                    )

                if action.robot_id != robot_id:
                    raise EnvironmentValidationError(
                        "Robot action ID mismatch"
                    )

            previous_distances = {
                robot.robot_id: distance_3d(
                    runtime.robot_states[
                        robot.robot_id
                    ].position,
                    robot.goal_position,
                )
                for robot in definition.robots
            }

            # Simplified kinematic simulation.
            # All robot positions are updated before
            # evaluating collisions and rewards.

            proposed_states = copy.deepcopy(
                runtime.robot_states
            )

            for robot in definition.robots:
                state = proposed_states[
                    robot.robot_id
                ]

                if (
                    state.goal_reached
                    or state.collided
                    or state.out_of_bounds
                ):
                    state.velocity = Vector3(0, 0, 0)
                    continue

                action = actions.get(
                    robot.robot_id,
                    RobotAction(
                        robot_id=robot.robot_id,
                        action_type=ActionType.STOP,
                    ),
                )

                requested_velocity = action.velocity()

                speed = math.sqrt(
                    requested_velocity.x ** 2
                    + requested_velocity.y ** 2
                    + requested_velocity.z ** 2
                )

                if speed > robot.maximum_speed:
                    scale = robot.maximum_speed / speed

                    velocity = Vector3(
                        requested_velocity.x * scale,
                        requested_velocity.y * scale,
                        requested_velocity.z * scale,
                    )

                else:
                    velocity = requested_velocity

                state.velocity = velocity

                state.position = Vector3(
                    state.position.x + velocity.x * dt,
                    state.position.y + velocity.y * dt,
                    state.position.z + velocity.z * dt,
                )

            proposed_obstacles = {}

            for obstacle in definition.obstacles:
                current = runtime.obstacle_positions[
                    obstacle.obstacle_id
                ]

                if (
                    obstacle.obstacle_type
                    == ObstacleType.DYNAMIC
                ):
                    next_position = Vector3(
                        current.x + obstacle.velocity.x * dt,
                        current.y + obstacle.velocity.y * dt,
                        current.z + obstacle.velocity.z * dt,
                    )

                    if definition.bounds.contains(
                        next_position,
                        obstacle.radius,
                    ):
                        proposed_obstacles[
                            obstacle.obstacle_id
                        ] = next_position

                    else:
                        proposed_obstacles[
                            obstacle.obstacle_id
                        ] = current

                else:
                    proposed_obstacles[
                        obstacle.obstacle_id
                    ] = current

            collisions = []

            # Robot-to-obstacle collision checks.

            for robot in definition.robots:
                state = proposed_states[
                    robot.robot_id
                ]

                if not definition.bounds.contains(
                    state.position,
                    robot.radius,
                ):
                    state.out_of_bounds = True

                for obstacle in definition.obstacles:
                    if not obstacle.collision_enabled:
                        continue

                    obstacle_position = proposed_obstacles[
                        obstacle.obstacle_id
                    ]

                    if distance_3d(
                        state.position,
                        obstacle_position,
                    ) <= robot.radius + obstacle.radius:
                        state.collided = True

                        collisions.append(
                            {
                                "robot_id": robot.robot_id,
                                "obstacle_id": (
                                    obstacle.obstacle_id
                                ),
                                "collision_type": (
                                    "robot_obstacle"
                                ),
                            }
                        )

            # Robot-to-robot collision checks.

            for index, robot in enumerate(
                definition.robots
            ):
                for other in definition.robots[
                    index + 1:
                ]:
                    first_state = proposed_states[
                        robot.robot_id
                    ]

                    second_state = proposed_states[
                        other.robot_id
                    ]

                    if distance_3d(
                        first_state.position,
                        second_state.position,
                    ) <= robot.radius + other.radius:
                        first_state.collided = True
                        second_state.collided = True

                        collisions.append(
                            {
                                "robot_id": robot.robot_id,
                                "other_robot_id": (
                                    other.robot_id
                                ),
                                "collision_type": (
                                    "robot_robot"
                                ),
                            }
                        )

            rewards = {}

            for robot in definition.robots:
                state = proposed_states[
                    robot.robot_id
                ]

                current_distance = distance_3d(
                    state.position,
                    robot.goal_position,
                )

                progress = (
                    previous_distances[robot.robot_id]
                    - current_distance
                )

                reward = (
                    self.config.step_penalty
                    + progress
                    * self.config.progress_reward_scale
                )

                if state.collided or state.out_of_bounds:
                    reward += self.config.collision_penalty

                if (
                    current_distance
                    <= robot.goal_tolerance
                    and not state.collided
                    and not state.out_of_bounds
                    and not state.goal_reached
                ):
                    state.goal_reached = True
                    reward += self.config.goal_reward

                state.total_reward += reward

                rewards[robot.robot_id] = round(
                    reward,
                    6,
                )

            runtime.robot_states = proposed_states
            runtime.obstacle_positions = proposed_obstacles

            runtime.step_number += 1
            runtime.simulation_time += dt

            runtime.last_collisions = collisions

            all_goals_reached = all(
                state.goal_reached
                for state in runtime.robot_states.values()
            )

            any_out_of_bounds = any(
                state.out_of_bounds
                for state in runtime.robot_states.values()
            )

            any_collision = any(
                state.collided
                for state in runtime.robot_states.values()
            )

            terminated = False

            reason = EpisodeTermination.NONE

            if any_out_of_bounds:
                terminated = True
                reason = EpisodeTermination.OUT_OF_BOUNDS

            elif (
                any_collision
                and self.config.terminate_on_collision
            ):
                terminated = True
                reason = EpisodeTermination.COLLISION

            elif all_goals_reached:
                terminated = True
                reason = EpisodeTermination.GOAL_REACHED

            elif (
                runtime.step_number
                >= definition.maximum_steps
            ):
                terminated = True
                reason = EpisodeTermination.TIME_LIMIT

            if terminated:
                runtime.status = (
                    EnvironmentStatus.COMPLETED
                )

                runtime.termination_reason = reason

            runtime.updated_at = utc_now()

            observations = self._observations(runtime)

            result = EnvironmentStepResult(
                environment_id=environment_id,
                step_number=runtime.step_number,
                simulation_time=round(
                    runtime.simulation_time,
                    6,
                ),
                observations=observations,
                rewards=rewards,
                terminated=terminated,
                termination_reason=reason,
                collisions=tuple(collisions),
            )

            if terminated:
                self._log(
                    tenant_id=tenant_id,
                    environment_id=environment_id,
                    action="episode_completed",
                    details={
                        "reason": reason.value,
                        "steps": runtime.step_number,
                        "collisions": len(collisions),
                    },
                )

            return result.to_dict()

    # --------------------------------------------------------
    # OBSERVATIONS
    # --------------------------------------------------------

    def _observations(
        self,
        runtime: EnvironmentRuntime,
    ) -> dict[str, Any]:
        observations = {}

        for robot in runtime.definition.robots:
            state = runtime.robot_states[
                robot.robot_id
            ]

            nearest_obstacle_distance = None

            for obstacle in runtime.definition.obstacles:
                obstacle_position = (
                    runtime.obstacle_positions[
                        obstacle.obstacle_id
                    ]
                )

                distance = max(
                    0.0,
                    distance_3d(
                        state.position,
                        obstacle_position,
                    )
                    - robot.radius
                    - obstacle.radius,
                )

                if (
                    nearest_obstacle_distance is None
                    or distance
                    < nearest_obstacle_distance
                ):
                    nearest_obstacle_distance = distance

            observations[
                robot.robot_id
            ] = {
                "position": state.position.to_dict(),
                "velocity": state.velocity.to_dict(),
                "goal_position": (
                    robot.goal_position.to_dict()
                ),
                "distance_to_goal": round(
                    distance_3d(
                        state.position,
                        robot.goal_position,
                    ),
                    6,
                ),
                "nearest_obstacle_distance": (
                    nearest_obstacle_distance
                ),
                "goal_reached": state.goal_reached,
                "collided": state.collided,
                "out_of_bounds": state.out_of_bounds,
                "total_reward": state.total_reward,
            }

        return observations

    def get_observations(
        self,
        *,
        tenant_id: str,
        environment_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            runtime = self._runtime(
                tenant_id,
                environment_id,
            )

            return self._observations(runtime)

    # --------------------------------------------------------
    # GET ENVIRONMENT
    # --------------------------------------------------------

    def get_environment(
        self,
        *,
        tenant_id: str,
        environment_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._runtime(
                tenant_id,
                environment_id,
            ).to_dict()

    # --------------------------------------------------------
    # LIST ENVIRONMENTS
    # --------------------------------------------------------

    def list_environments(
        self,
        *,
        tenant_id: str,
        status: EnvironmentStatus | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if status is not None and not isinstance(
            status,
            EnvironmentStatus,
        ):
            raise EnvironmentValidationError(
                "Invalid status"
            )

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            runtimes = [
                runtime
                for (tenant, _), runtime
                in self._environments.items()
                if (
                    tenant == tenant_id
                    and (
                        status is None
                        or runtime.status == status
                    )
                )
            ]

            runtimes.sort(
                key=lambda item: item.updated_at,
                reverse=True,
            )

            return [
                runtime.to_dict()
                for runtime in runtimes[:limit]
            ]

    # --------------------------------------------------------
    # ENVIRONMENT SNAPSHOT
    # --------------------------------------------------------

    def snapshot(
        self,
        *,
        tenant_id: str,
        environment_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            runtime = self._runtime(
                tenant_id,
                environment_id,
            )

            return {
                "snapshot_id": new_id(),
                "captured_at": utc_now().isoformat(),
                "environment": runtime.to_dict(),
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # ENVIRONMENT SUMMARY
    # --------------------------------------------------------

    def environment_summary(
        self,
        *,
        tenant_id: str,
        environment_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            runtime = self._runtime(
                tenant_id,
                environment_id,
            )

            states = list(
                runtime.robot_states.values()
            )

            return {
                "environment_id": environment_id,
                "tenant_id": tenant_id,
                "status": runtime.status.value,
                "environment_type": (
                    runtime.definition.environment_type.value
                ),
                "robot_count": len(states),
                "obstacle_count": len(
                    runtime.definition.obstacles
                ),
                "robots_at_goal": sum(
                    state.goal_reached
                    for state in states
                ),
                "robots_collided": sum(
                    state.collided
                    for state in states
                ),
                "robots_out_of_bounds": sum(
                    state.out_of_bounds
                    for state in states
                ),
                "step_number": runtime.step_number,
                "simulation_time": (
                    runtime.simulation_time
                ),
                "episode_number": (
                    runtime.episode_number
                ),
                "termination_reason": (
                    runtime.termination_reason.value
                ),
                "total_reward": round(
                    sum(
                        state.total_reward
                        for state in states
                    ),
                    6,
                ),
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        *,
        tenant_id: str,
        environment_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if environment_id is not None:
            validate_id(
                environment_id,
                "environment_id",
            )

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            events = [
                copy.deepcopy(event)
                for event in self._audit
                if (
                    event["tenant_id"] == tenant_id
                    and (
                        environment_id is None
                        or event["environment_id"]
                        == environment_id
                    )
                )
            ]

            return events[-limit:]

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
    ) -> Path:
        validate_id(tenant_id, "tenant_id")

        output = Path(path)

        with self._lock:
            environments = [
                runtime.to_dict()
                for (tenant, _), runtime
                in self._environments.items()
                if tenant == tenant_id
            ]

            audit = [
                copy.deepcopy(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ]

        payload = {
            "generated_at": utc_now().isoformat(),
            "tenant_id": tenant_id,
            "environments": environments,
            "audit_events": audit,
            "simulation_only": True,
            "physical_execution_authorized": False,
        }

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

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
# DEFAULT WAREHOUSE ENVIRONMENT
# ============================================================

def create_warehouse_environment(
    *,
    tenant_id: str,
    environment_id: str = "warehouse_simulation",
) -> EnvironmentDefinition:

    robots = (
        SimulationRobot(
            robot_id="robot_001",
            name="Autonomous Mobile Robot 1",
            spawn_position=Vector3(2, 2, 0),
            goal_position=Vector3(18, 18, 0),
            radius=0.35,
            maximum_speed=1.5,
            goal_tolerance=0.5,
        ),
        SimulationRobot(
            robot_id="robot_002",
            name="Autonomous Mobile Robot 2",
            spawn_position=Vector3(2, 5, 0),
            goal_position=Vector3(18, 5, 0),
            radius=0.35,
            maximum_speed=1.2,
            goal_tolerance=0.5,
        ),
    )

    obstacles = (
        SimulationObstacle(
            obstacle_id="shelf_001",
            name="Warehouse Shelf 1",
            position=Vector3(8, 8, 0),
            radius=1.0,
        ),
        SimulationObstacle(
            obstacle_id="shelf_002",
            name="Warehouse Shelf 2",
            position=Vector3(12, 12, 0),
            radius=1.0,
        ),
        SimulationObstacle(
            obstacle_id="moving_cart_001",
            name="Moving Warehouse Cart",
            position=Vector3(10, 4, 0),
            radius=0.5,
            obstacle_type=ObstacleType.DYNAMIC,
            velocity=Vector3(0, 0.2, 0),
        ),
    )

    return EnvironmentDefinition(
        environment_id=environment_id,
        tenant_id=tenant_id,
        name="Autonomous Warehouse Simulation",
        environment_type=EnvironmentType.WAREHOUSE,
        bounds=EnvironmentBounds(
            minimum=Vector3(0, 0, 0),
            maximum=Vector3(20, 20, 5),
        ),
        robots=robots,
        obstacles=obstacles,
        physics=PhysicsConfig(
            gravity=9.81,
            friction=0.7,
            restitution=0.0,
            linear_damping=0.05,
            time_step=0.1,
        ),
        sensors=SensorSimulationConfig(
            camera_enabled=True,
            lidar_enabled=True,
            imu_enabled=True,
            camera_noise_std=0.02,
            lidar_noise_std=0.03,
            imu_noise_std=0.005,
        ),
        terrain=TerrainType.FLAT,
        lighting_intensity=1.0,
        maximum_steps=500,
        random_seed=42,
        curriculum_stage_id="stage_003",
        domain_randomization_profile_id=(
            "robot_domain_randomization"
        ),
        metadata={
            "scenario": "warehouse_navigation",
            "training_mode": True,
            "multi_robot": True,
        },
        description=(
            "Multi-robot warehouse navigation simulation "
            "with shelves, moving carts and goal locations."
        ),
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    engine = RoboticsEnvironmentEngine()

    definition = create_warehouse_environment(
        tenant_id="tenant_alpha",
    )

    engine.create_environment(
        definition,
        authorized=True,
        actor_id="simulation_admin",
    )

    engine.start_environment(
        tenant_id="tenant_alpha",
        environment_id="warehouse_simulation",
        authorized=True,
        actor_id="simulation_admin",
    )

    print("\nINITIAL OBSERVATIONS")

    print(
        json.dumps(
            engine.get_observations(
                tenant_id="tenant_alpha",
                environment_id="warehouse_simulation",
            ),
            indent=2,
        )
    )

    for step_index in range(10):
        result = engine.step(
            tenant_id="tenant_alpha",
            environment_id="warehouse_simulation",
            actions={
                "robot_001": RobotAction(
                    robot_id="robot_001",
                    action_type=ActionType.VELOCITY,
                    linear_velocity_x=0.5,
                    linear_velocity_y=0.5,
                ),
                "robot_002": RobotAction(
                    robot_id="robot_002",
                    action_type=ActionType.VELOCITY,
                    linear_velocity_x=0.6,
                    linear_velocity_y=0.0,
                ),
            },
        )

        print(
            f"\nSIMULATION STEP {step_index + 1}"
        )

        print(
            json.dumps(
                {
                    "step_number": result["step_number"],
                    "rewards": result["rewards"],
                    "terminated": result["terminated"],
                    "termination_reason": (
                        result["termination_reason"]
                    ),
                },
                indent=2,
            )
        )

        if result["terminated"]:
            break

    print("\nENVIRONMENT SUMMARY")

    print(
        json.dumps(
            engine.environment_summary(
                tenant_id="tenant_alpha",
                environment_id="warehouse_simulation",
            ),
            indent=2,
        )
    )

    print("\nENVIRONMENT SNAPSHOT")

    snapshot = engine.snapshot(
        tenant_id="tenant_alpha",
        environment_id="warehouse_simulation",
    )

    print(
        json.dumps(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "captured_at": snapshot["captured_at"],
                "status": snapshot["environment"]["status"],
            },
            indent=2,
        )
    )

    output = engine.export_json(
        "data/robotics/environment_report.json",
        tenant_id="tenant_alpha",
    )

    print("\nExported:", output)


if __name__ == "__main__":
    main()
# backend/app/robotics/control/reward_shaping.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Reward Shaping Engine

Python 3.11+
Dependencies: Standard library only

Features:
    - Potential-Based Reward Shaping (PBRS)
    - Goal-distance rewards
    - Navigation progress rewards
    - Collision penalties
    - Human proximity safety penalties
    - Obstacle clearance rewards
    - Energy consumption penalties
    - Smooth motion rewards
    - Velocity tracking rewards
    - Task completion rewards
    - Manipulation accuracy rewards
    - Time efficiency penalties
    - Configurable reward weights
    - Reward clipping
    - Episode reward tracking
    - Reward breakdown and explainability
    - Reward normalization
    - Multi-tenant isolation
    - Thread-safe operations
    - Audit logging
    - JSON report export

Compatible with:
    ppo_trainer.py
    offline_rl.py
    policy.py
    policy_evaluation.py
    replay_buffer.py
    environment.py
    curriculum.py
    domain_randomization.py

Safety:
    Simulation and offline learning only.

    Reward shaping is not a substitute for physical
    safety constraints, independent safety monitors,
    emergency-stop systems, or human authorization.

    A learned policy must not directly control
    physical robot hardware through this module.
"""

from __future__ import annotations

import json
import math
import threading
import uuid

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class RewardShapingError(Exception):
    pass


class RewardValidationError(RewardShapingError):
    pass


class RewardProfileNotFound(RewardShapingError):
    pass


class RewardConflict(RewardShapingError):
    pass


class RewardPermissionDenied(RewardShapingError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RewardComponent(str, Enum):
    BASE_REWARD = "base_reward"
    POTENTIAL_SHAPING = "potential_shaping"
    GOAL_PROGRESS = "goal_progress"
    GOAL_REACHED = "goal_reached"
    COLLISION = "collision"
    HUMAN_PROXIMITY = "human_proximity"
    OBSTACLE_CLEARANCE = "obstacle_clearance"
    ENERGY_CONSUMPTION = "energy_consumption"
    SMOOTH_MOTION = "smooth_motion"
    VELOCITY_TRACKING = "velocity_tracking"
    MANIPULATION_ACCURACY = "manipulation_accuracy"
    TASK_COMPLETION = "task_completion"
    TIME_PENALTY = "time_penalty"
    CUSTOM = "custom"


class RewardProfileStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class RewardTaskType(str, Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    MOBILE_MANIPULATION = "mobile_manipulation"
    INSPECTION = "inspection"
    WAREHOUSE = "warehouse"
    CUSTOM = "custom"


class RewardNormalizationMode(str, Enum):
    NONE = "none"
    RUNNING_STANDARDIZATION = "running_standardization"


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
        raise RewardValidationError(f"Invalid {name}")

    return value


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float)
    ):
        raise RewardValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise RewardValidationError(
            f"{name} must be finite"
        )

    return result


def nonnegative(value: Any, name: str) -> float:
    result = finite_number(value, name)

    if result < 0:
        raise RewardValidationError(
            f"{name} must be nonnegative"
        )

    return result


def probability(value: Any, name: str) -> float:
    result = finite_number(value, name)

    if not 0.0 <= result <= 1.0:
        raise RewardValidationError(
            f"{name} must be between 0 and 1"
        )

    return result


def vector(
    values: Sequence[float],
    name: str,
    dimension: int | None = None,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise RewardValidationError(
            f"{name} must be a numeric vector"
        )

    try:
        result = tuple(
            finite_number(value, name)
            for value in values
        )
    except TypeError as exc:
        raise RewardValidationError(
            f"{name} must be a sequence"
        ) from exc

    if not result:
        raise RewardValidationError(
            f"{name} cannot be empty"
        )

    if dimension is not None and len(result) != dimension:
        raise RewardValidationError(
            f"{name} requires {dimension} dimensions"
        )

    return result


def euclidean_distance(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    first = vector(a, "first_vector")
    second = vector(
        b,
        "second_vector",
        len(first),
    )

    return math.sqrt(
        sum(
            (x - y) ** 2
            for x, y in zip(first, second)
        )
    )


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, deque)):
        return [
            json_safe(item)
            for item in value
        ]

    return value


# ============================================================
# REWARD WEIGHTS
# ============================================================

@dataclass(frozen=True)
class RewardWeights:
    potential_shaping: float = 1.0

    goal_progress: float = 0.0
    goal_reached: float = 10.0

    collision: float = 20.0
    human_proximity: float = 10.0

    obstacle_clearance: float = 1.0

    energy_consumption: float = 0.05
    smooth_motion: float = 0.1

    velocity_tracking: float = 0.5
    manipulation_accuracy: float = 1.0

    task_completion: float = 10.0
    time_penalty: float = 0.01

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            nonnegative(
                getattr(self, name),
                name,
            )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RewardShapingConfig:
    gamma: float = 0.99

    goal_tolerance: float = 0.25

    human_safe_distance: float = 1.5
    obstacle_safe_distance: float = 0.75

    maximum_velocity_error: float = 5.0

    reward_minimum: float = -100.0
    reward_maximum: float = 100.0

    clip_rewards: bool = True

    normalization: RewardNormalizationMode = (
        RewardNormalizationMode.NONE
    )

    normalization_epsilon: float = 1e-8

    maximum_episode_history: int = 10_000
    maximum_audit_events: int = 20_000

    simulation_only: bool = True

    def __post_init__(self) -> None:
        probability(self.gamma, "gamma")

        for name in (
            "goal_tolerance",
            "human_safe_distance",
            "obstacle_safe_distance",
            "maximum_velocity_error",
            "normalization_epsilon",
        ):
            if finite_number(
                getattr(self, name),
                name,
            ) <= 0:
                raise RewardValidationError(
                    f"{name} must be positive"
                )

        if self.reward_minimum >= self.reward_maximum:
            raise RewardValidationError(
                "Invalid reward clipping bounds"
            )

        for name in (
            "maximum_episode_history",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise RewardValidationError(
                    f"{name} must be a positive integer"
                )

        if not isinstance(
            self.normalization,
            RewardNormalizationMode,
        ):
            raise RewardValidationError(
                "Invalid normalization mode"
            )

        if self.simulation_only is not True:
            raise RewardValidationError(
                "Physical robot execution is unsupported"
            )


# ============================================================
# REWARD PROFILE
# ============================================================

@dataclass
class RewardProfile:
    profile_id: str
    tenant_id: str
    robot_id: str

    name: str
    task_type: RewardTaskType

    config: RewardShapingConfig
    weights: RewardWeights

    status: RewardProfileStatus = (
        RewardProfileStatus.DRAFT
    )

    version: int = 1

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# REWARD OBSERVATION
# ============================================================

@dataclass(frozen=True)
class RewardObservation:
    position: tuple[float, ...]
    goal_position: tuple[float, ...]

    velocity: tuple[float, ...] | None = None
    target_velocity: tuple[float, ...] | None = None

    previous_velocity: tuple[float, ...] | None = None

    nearest_obstacle_distance: float | None = None
    nearest_human_distance: float | None = None

    energy_consumption: float = 0.0

    manipulation_error: float | None = None

    collision_detected: bool = False
    task_completed: bool = False

    def __post_init__(self) -> None:
        position = vector(
            self.position,
            "position",
        )

        vector(
            self.goal_position,
            "goal_position",
            len(position),
        )

        for name in (
            "velocity",
            "target_velocity",
            "previous_velocity",
        ):
            value = getattr(self, name)

            if value is not None:
                vector(
                    value,
                    name,
                    len(position),
                )

        for name in (
            "nearest_obstacle_distance",
            "nearest_human_distance",
            "manipulation_error",
        ):
            value = getattr(self, name)

            if value is not None:
                nonnegative(value, name)

        nonnegative(
            self.energy_consumption,
            "energy_consumption",
        )

        if not isinstance(self.collision_detected, bool):
            raise RewardValidationError(
                "collision_detected must be boolean"
            )

        if not isinstance(self.task_completed, bool):
            raise RewardValidationError(
                "task_completed must be boolean"
            )


# ============================================================
# REWARD REQUEST
# ============================================================

@dataclass(frozen=True)
class RewardRequest:
    environment_id: str
    episode_id: str
    step_index: int

    current: RewardObservation
    next: RewardObservation

    base_reward: float = 0.0

    terminated: bool = False
    truncated: bool = False

    time_delta_seconds: float = 1.0

    custom_rewards: Mapping[str, float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_id(
            self.environment_id,
            "environment_id",
        )

        validate_id(
            self.episode_id,
            "episode_id",
        )

        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise RewardValidationError(
                "step_index must be nonnegative"
            )

        finite_number(
            self.base_reward,
            "base_reward",
        )

        if finite_number(
            self.time_delta_seconds,
            "time_delta_seconds",
        ) <= 0:
            raise RewardValidationError(
                "time_delta_seconds must be positive"
            )

        if self.terminated and self.truncated:
            raise RewardValidationError(
                "Transition cannot be both "
                "terminated and truncated"
            )

        for name, value in self.custom_rewards.items():
            validate_id(name, "custom_reward_name")
            finite_number(value, name)


# ============================================================
# REWARD RESULT
# ============================================================

@dataclass(frozen=True)
class RewardResult:
    result_id: str

    tenant_id: str
    profile_id: str

    environment_id: str
    episode_id: str
    step_index: int

    base_reward: float
    shaping_reward: float

    raw_total_reward: float
    final_reward: float

    components: dict[str, float]

    current_goal_distance: float
    next_goal_distance: float

    goal_reached: bool
    collision_detected: bool

    terminated: bool
    truncated: bool

    clipped: bool
    normalized: bool

    created_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# RUNNING STATISTICS
# ============================================================

class RunningStatistics:
    """
    Welford online mean and variance.

    Used for optional reward normalization.

    Note:
        Reward normalization changes reward scaling
        and may alter the learning objective.
    """

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        value = finite_number(
            value,
            "running_statistic_value",
        )

        self.count += 1

        delta = value - self.mean

        self.mean += delta / self.count

        delta2 = value - self.mean

        self.m2 += delta * delta2

    def variance(self) -> float:
        if self.count < 2:
            return 0.0

        return max(
            0.0,
            self.m2 / self.count,
        )

    def standard_deviation(self) -> float:
        return math.sqrt(
            self.variance()
        )

    def normalize(
        self,
        value: float,
        epsilon: float,
    ) -> float:
        if self.count < 2:
            return value

        return (
            value - self.mean
        ) / max(
            self.standard_deviation(),
            epsilon,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance(),
            "standard_deviation": (
                self.standard_deviation()
            ),
        }


# ============================================================
# POTENTIAL-BASED REWARD SHAPING
# ============================================================

class PotentialBasedRewardShaper:
    """
    Implements:

        F(s, s') = gamma * Phi(s') - Phi(s)

    Default potential:

        Phi(s) = -distance(robot, goal)

    With suitable assumptions, potential-based
    shaping preserves optimal policies.

    Non-potential auxiliary rewards, reward clipping,
    and normalization do not generally preserve
    that guarantee.
    """

    @staticmethod
    def potential(
        observation: RewardObservation,
    ) -> float:
        return -euclidean_distance(
            observation.position,
            observation.goal_position,
        )

    @classmethod
    def calculate(
        cls,
        current: RewardObservation,
        next_observation: RewardObservation,
        gamma: float,
        *,
        terminated: bool = False,
    ) -> float:
        current_potential = cls.potential(
            current
        )

        next_potential = (
            0.0
            if terminated
            else cls.potential(next_observation)
        )

        return (
            gamma * next_potential
            - current_potential
        )


# ============================================================
# REWARD CALCULATOR
# ============================================================

class RewardCalculator:

    @staticmethod
    def calculate(
        *,
        tenant_id: str,
        profile: RewardProfile,
        request: RewardRequest,
        statistics: RunningStatistics | None = None,
    ) -> RewardResult:
        config = profile.config
        weights = profile.weights

        current = request.current
        next_observation = request.next

        current_distance = euclidean_distance(
            current.position,
            current.goal_position,
        )

        next_distance = euclidean_distance(
            next_observation.position,
            next_observation.goal_position,
        )

        components: dict[str, float] = {}

        # ----------------------------------------------------
        # BASE ENVIRONMENT REWARD
        # ----------------------------------------------------

        components[
            RewardComponent.BASE_REWARD.value
        ] = request.base_reward

        # ----------------------------------------------------
        # POTENTIAL-BASED SHAPING
        # ----------------------------------------------------

        potential_reward = (
            PotentialBasedRewardShaper.calculate(
                current,
                next_observation,
                config.gamma,
                terminated=request.terminated,
            )
        )

        components[
            RewardComponent.POTENTIAL_SHAPING.value
        ] = (
            weights.potential_shaping
            * potential_reward
        )

        # ----------------------------------------------------
        # GOAL PROGRESS
        # ----------------------------------------------------

        progress = (
            current_distance - next_distance
        )

        components[
            RewardComponent.GOAL_PROGRESS.value
        ] = (
            weights.goal_progress
            * progress
        )

        # ----------------------------------------------------
        # GOAL REACHED
        # ----------------------------------------------------

        goal_reached = (
            next_distance <= config.goal_tolerance
        )

        components[
            RewardComponent.GOAL_REACHED.value
        ] = (
            weights.goal_reached
            if goal_reached
            else 0.0
        )

        # ----------------------------------------------------
        # COLLISION PENALTY
        # ----------------------------------------------------

        components[
            RewardComponent.COLLISION.value
        ] = (
            -weights.collision
            if next_observation.collision_detected
            else 0.0
        )

        # ----------------------------------------------------
        # HUMAN PROXIMITY
        # ----------------------------------------------------

        human_distance = (
            next_observation.nearest_human_distance
        )

        human_penalty = 0.0

        if human_distance is not None:
            if human_distance < config.human_safe_distance:
                proximity_ratio = (
                    config.human_safe_distance
                    - human_distance
                ) / config.human_safe_distance

                human_penalty = (
                    -weights.human_proximity
                    * proximity_ratio
                )

        components[
            RewardComponent.HUMAN_PROXIMITY.value
        ] = human_penalty

        # ----------------------------------------------------
        # OBSTACLE CLEARANCE
        # ----------------------------------------------------

        obstacle_distance = (
            next_observation.nearest_obstacle_distance
        )

        obstacle_reward = 0.0

        if obstacle_distance is not None:
            if obstacle_distance < config.obstacle_safe_distance:
                risk_ratio = (
                    config.obstacle_safe_distance
                    - obstacle_distance
                ) / config.obstacle_safe_distance

                obstacle_reward = (
                    -weights.obstacle_clearance
                    * risk_ratio
                )

        components[
            RewardComponent.OBSTACLE_CLEARANCE.value
        ] = obstacle_reward

        # ----------------------------------------------------
        # ENERGY CONSUMPTION
        # ----------------------------------------------------

        components[
            RewardComponent.ENERGY_CONSUMPTION.value
        ] = (
            -weights.energy_consumption
            * next_observation.energy_consumption
        )

        # ----------------------------------------------------
        # SMOOTH MOTION
        # ----------------------------------------------------

        smooth_motion_reward = 0.0

        if (
            next_observation.velocity is not None
            and current.velocity is not None
        ):
            velocity_change = euclidean_distance(
                current.velocity,
                next_observation.velocity,
            )

            smooth_motion_reward = (
                -weights.smooth_motion
                * velocity_change
            )

        components[
            RewardComponent.SMOOTH_MOTION.value
        ] = smooth_motion_reward

        # ----------------------------------------------------
        # VELOCITY TRACKING
        # ----------------------------------------------------

        velocity_reward = 0.0

        if (
            next_observation.velocity is not None
            and next_observation.target_velocity is not None
        ):
            velocity_error = euclidean_distance(
                next_observation.velocity,
                next_observation.target_velocity,
            )

            normalized_error = min(
                velocity_error
                / config.maximum_velocity_error,
                1.0,
            )

            velocity_reward = (
                -weights.velocity_tracking
                * normalized_error
            )

        components[
            RewardComponent.VELOCITY_TRACKING.value
        ] = velocity_reward

        # ----------------------------------------------------
        # MANIPULATION ACCURACY
        # ----------------------------------------------------

        manipulation_reward = 0.0

        if next_observation.manipulation_error is not None:
            manipulation_reward = (
                -weights.manipulation_accuracy
                * next_observation.manipulation_error
            )

        components[
            RewardComponent.MANIPULATION_ACCURACY.value
        ] = manipulation_reward

        # ----------------------------------------------------
        # TASK COMPLETION
        # ----------------------------------------------------

        components[
            RewardComponent.TASK_COMPLETION.value
        ] = (
            weights.task_completion
            if next_observation.task_completed
            else 0.0
        )

        # ----------------------------------------------------
        # TIME PENALTY
        # ----------------------------------------------------

        components[
            RewardComponent.TIME_PENALTY.value
        ] = (
            -weights.time_penalty
            * request.time_delta_seconds
        )

        # ----------------------------------------------------
        # CUSTOM REWARDS
        # ----------------------------------------------------

        for name, value in request.custom_rewards.items():
            components[
                f"{RewardComponent.CUSTOM.value}:{name}"
            ] = finite_number(
                value,
                name,
            )

        # ----------------------------------------------------
        # TOTAL REWARD
        # ----------------------------------------------------

        raw_total = sum(
            components.values()
        )

        shaping_reward = (
            raw_total - request.base_reward
        )

        final_reward = raw_total
        normalized = False

        if (
            config.normalization
            == RewardNormalizationMode.RUNNING_STANDARDIZATION
            and statistics is not None
        ):
            statistics.update(
                raw_total
            )

            final_reward = statistics.normalize(
                raw_total,
                config.normalization_epsilon,
            )

            normalized = True

        clipped = False

        if config.clip_rewards:
            clipped_reward = clamp(
                final_reward,
                config.reward_minimum,
                config.reward_maximum,
            )

            clipped = (
                clipped_reward != final_reward
            )

            final_reward = clipped_reward

        return RewardResult(
            result_id=new_id(),
            tenant_id=tenant_id,
            profile_id=profile.profile_id,
            environment_id=request.environment_id,
            episode_id=request.episode_id,
            step_index=request.step_index,
            base_reward=request.base_reward,
            shaping_reward=shaping_reward,
            raw_total_reward=raw_total,
            final_reward=final_reward,
            components=components,
            current_goal_distance=current_distance,
            next_goal_distance=next_distance,
            goal_reached=goal_reached,
            collision_detected=(
                next_observation.collision_detected
            ),
            terminated=request.terminated,
            truncated=request.truncated,
            clipped=clipped,
            normalized=normalized,
        )


# ============================================================
# EPISODE TRACKER
# ============================================================

@dataclass
class EpisodeRewardSummary:
    episode_id: str
    environment_id: str
    profile_id: str

    steps: int = 0

    total_base_reward: float = 0.0
    total_shaping_reward: float = 0.0
    total_final_reward: float = 0.0

    component_totals: dict[str, float] = field(
        default_factory=dict
    )

    collisions: int = 0
    goal_reached: bool = False

    completed: bool = False
    terminated: bool = False
    truncated: bool = False

    started_at: datetime = field(
        default_factory=utc_now
    )

    completed_at: datetime | None = None

    def record(
        self,
        result: RewardResult,
    ) -> None:
        if self.completed:
            raise RewardConflict(
                "Cannot record after episode completion"
            )

        if result.step_index != self.steps:
            raise RewardConflict(
                "Episode steps must be sequential"
            )

        self.steps += 1

        self.total_base_reward += (
            result.base_reward
        )

        self.total_shaping_reward += (
            result.shaping_reward
        )

        self.total_final_reward += (
            result.final_reward
        )

        for name, value in result.components.items():
            self.component_totals[name] = (
                self.component_totals.get(name, 0.0)
                + value
            )

        if result.collision_detected:
            self.collisions += 1

        self.goal_reached = (
            self.goal_reached or result.goal_reached
        )

        if result.terminated or result.truncated:
            self.completed = True
            self.terminated = result.terminated
            self.truncated = result.truncated
            self.completed_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# REWARD SHAPING ENGINE
# ============================================================

class RoboticsRewardShapingEngine:

    def __init__(self) -> None:
        self._profiles: dict[
            tuple[str, str],
            RewardProfile,
        ] = {}

        self._episodes: dict[
            tuple[str, str, str, str],
            EpisodeRewardSummary,
        ] = {}

        self._statistics: dict[
            tuple[str, str],
            RunningStatistics,
        ] = {}

        self._recent_results: dict[
            tuple[str, str],
            deque[RewardResult],
        ] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        self._audit: deque[
            dict[str, Any]
        ] = deque(maxlen=20_000)

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _log(
        self,
        *,
        tenant_id: str,
        action: str,
        entity_id: str,
        actor_id: str = "system",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._audit.append(
            {
                "event_id": new_id(),
                "timestamp": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "action": action,
                "entity_id": entity_id,
                "actor_id": actor_id,
                "details": json_safe(
                    dict(details or {})
                ),
            }
        )

    # --------------------------------------------------------
    # REGISTER PROFILE
    # --------------------------------------------------------

    def register_profile(
        self,
        *,
        profile: RewardProfile,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise RewardPermissionDenied(
                "Authorized profile registration required"
            )

        if not isinstance(profile, RewardProfile):
            raise RewardValidationError(
                "Invalid reward profile"
            )

        for name in (
            "profile_id",
            "tenant_id",
            "robot_id",
        ):
            validate_id(
                getattr(profile, name),
                name,
            )

        if not profile.name.strip():
            raise RewardValidationError(
                "Profile name cannot be empty"
            )

        key = (
            profile.tenant_id,
            profile.profile_id,
        )

        with self._lock:
            if key in self._profiles:
                raise RewardConflict(
                    "Reward profile already exists"
                )

            self._profiles[key] = profile

            self._statistics[key] = RunningStatistics()

            self._log(
                tenant_id=profile.tenant_id,
                action="reward_profile_registered",
                entity_id=profile.profile_id,
                actor_id=actor_id,
            )

            return profile.to_dict()

    # --------------------------------------------------------
    # GET PROFILE
    # --------------------------------------------------------

    def _get_profile(
        self,
        tenant_id: str,
        profile_id: str,
    ) -> RewardProfile:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            profile_id,
            "profile_id",
        )

        profile = self._profiles.get(
            (tenant_id, profile_id)
        )

        if profile is None:
            raise RewardProfileNotFound(
                "Reward profile not found"
            )

        return profile

    def get_profile(
        self,
        *,
        tenant_id: str,
        profile_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get_profile(
                tenant_id,
                profile_id,
            ).to_dict()

    # --------------------------------------------------------
    # ACTIVATE PROFILE
    # --------------------------------------------------------

    def activate_profile(
        self,
        *,
        tenant_id: str,
        profile_id: str,
        reviewer_id: str,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise RewardPermissionDenied(
                "Human authorization required"
            )

        validate_id(
            reviewer_id,
            "reviewer_id",
        )

        with self._lock:
            profile = self._get_profile(
                tenant_id,
                profile_id,
            )

            if profile.status == RewardProfileStatus.ARCHIVED:
                raise RewardConflict(
                    "Archived profile cannot be activated"
                )

            profile.status = RewardProfileStatus.ACTIVE
            profile.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                action="reward_profile_activated",
                entity_id=profile_id,
                actor_id=reviewer_id,
            )

            return profile.to_dict()

    # --------------------------------------------------------
    # CALCULATE REWARD
    # --------------------------------------------------------

    def calculate_reward(
        self,
        *,
        tenant_id: str,
        profile_id: str,
        request: RewardRequest,
    ) -> dict[str, Any]:
        if not isinstance(request, RewardRequest):
            raise RewardValidationError(
                "Invalid reward request"
            )

        with self._lock:
            profile = self._get_profile(
                tenant_id,
                profile_id,
            )

            if profile.status != RewardProfileStatus.ACTIVE:
                raise RewardConflict(
                    "Reward profile must be active"
                )

            episode_key = (
                tenant_id,
                profile_id,
                request.environment_id,
                request.episode_id,
            )

            episode = self._episodes.get(
                episode_key
            )

            if episode is None:
                if request.step_index != 0:
                    raise RewardConflict(
                        "New episode must begin at step 0"
                    )

                episode = EpisodeRewardSummary(
                    episode_id=request.episode_id,
                    environment_id=request.environment_id,
                    profile_id=profile_id,
                )

                self._episodes[episode_key] = episode

            if episode.completed:
                raise RewardConflict(
                    "Episode is already complete"
                )

            if request.step_index != episode.steps:
                raise RewardConflict(
                    "Unexpected episode step index"
                )

            statistics = self._statistics[
                (tenant_id, profile_id)
            ]

            result = RewardCalculator.calculate(
                tenant_id=tenant_id,
                profile=profile,
                request=request,
                statistics=statistics,
            )

            episode.record(result)

            self._recent_results[
                (tenant_id, profile_id)
            ].append(result)

            if result.terminated or result.truncated:
                self._log(
                    tenant_id=tenant_id,
                    action="reward_episode_completed",
                    entity_id=request.episode_id,
                    details={
                        "profile_id": profile_id,
                        "steps": episode.steps,
                        "total_reward": (
                            episode.total_final_reward
                        ),
                        "collisions": episode.collisions,
                    },
                )

            return result.to_dict()

    # --------------------------------------------------------
    # GET EPISODE
    # --------------------------------------------------------

    def get_episode(
        self,
        *,
        tenant_id: str,
        profile_id: str,
        environment_id: str,
        episode_id: str,
    ) -> dict[str, Any]:
        key = (
            tenant_id,
            profile_id,
            environment_id,
            episode_id,
        )

        with self._lock:
            episode = self._episodes.get(key)

            if episode is None:
                raise RewardProfileNotFound(
                    "Episode not found"
                )

            return episode.to_dict()

    # --------------------------------------------------------
    # PROFILE SUMMARY
    # --------------------------------------------------------

    def profile_summary(
        self,
        *,
        tenant_id: str,
        profile_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            profile = self._get_profile(
                tenant_id,
                profile_id,
            )

            episodes = [
                episode
                for key, episode in self._episodes.items()
                if key[0] == tenant_id
                and key[1] == profile_id
            ]

            completed = [
                episode
                for episode in episodes
                if episode.completed
            ]

            total_rewards = [
                episode.total_final_reward
                for episode in completed
            ]

            goal_successes = sum(
                episode.goal_reached
                for episode in completed
            )

            return {
                "tenant_id": tenant_id,
                "profile_id": profile_id,
                "status": profile.status.value,
                "task_type": profile.task_type.value,
                "episode_count": len(episodes),
                "completed_episodes": len(completed),
                "mean_episode_reward": (
                    sum(total_rewards) / len(total_rewards)
                    if total_rewards
                    else 0.0
                ),
                "goal_success_rate": (
                    goal_successes / len(completed)
                    if completed
                    else 0.0
                ),
                "total_collisions": sum(
                    episode.collisions
                    for episode in episodes
                ),
                "normalization_statistics": (
                    self._statistics[
                        (tenant_id, profile_id)
                    ].to_dict()
                ),
                "simulation_only": True,
                "physical_deployment_approved": False,
            }

    # --------------------------------------------------------
    # REWARD COMPONENT ANALYSIS
    # --------------------------------------------------------

    def component_analysis(
        self,
        *,
        tenant_id: str,
        profile_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            self._get_profile(
                tenant_id,
                profile_id,
            )

            results = list(
                self._recent_results[
                    (tenant_id, profile_id)
                ]
            )

            component_totals: dict[str, float] = (
                defaultdict(float)
            )

            for result in results:
                for name, value in result.components.items():
                    component_totals[name] += value

            component_means = {
                name: (
                    value / len(results)
                    if results
                    else 0.0
                )
                for name, value in component_totals.items()
            }

            return {
                "profile_id": profile_id,
                "sample_count": len(results),
                "component_totals": dict(
                    component_totals
                ),
                "component_means": component_means,
                "dominant_components": sorted(
                    component_totals,
                    key=lambda name: abs(
                        component_totals[name]
                    ),
                    reverse=True,
                ),
            }

    # --------------------------------------------------------
    # LIST PROFILES
    # --------------------------------------------------------

    def list_profiles(
        self,
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        with self._lock:
            return [
                profile.to_dict()
                for (tenant, _), profile
                in self._profiles.items()
                if tenant == tenant_id
            ]

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
        ):
            raise RewardValidationError(
                "limit must be positive"
            )

        with self._lock:
            return [
                dict(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ][-limit:]

    # --------------------------------------------------------
    # EXPORT REPORT
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
    ) -> Path:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        output = Path(path)

        with self._lock:
            profiles = [
                profile.to_dict()
                for (tenant, _), profile
                in self._profiles.items()
                if tenant == tenant_id
            ]

            episodes = [
                episode.to_dict()
                for key, episode in self._episodes.items()
                if key[0] == tenant_id
            ]

            audit = [
                dict(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ]

        payload = {
            "generated_at": utc_now().isoformat(),
            "tenant_id": tenant_id,
            "profiles": profiles,
            "episodes": episodes,
            "audit_events": audit,
            "simulation_only": True,
            "physical_deployment_approved": False,
        }

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                json_safe(payload),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output


# ============================================================
# DEFAULT NAVIGATION REWARD PROFILE
# ============================================================

def create_navigation_reward_profile(
    *,
    tenant_id: str,
    robot_id: str,
    profile_id: str = "navigation_reward_v1",
) -> RewardProfile:
    return RewardProfile(
        profile_id=profile_id,
        tenant_id=tenant_id,
        robot_id=robot_id,
        name="Autonomous Navigation Reward Shaping",
        task_type=RewardTaskType.NAVIGATION,
        config=RewardShapingConfig(
            gamma=0.99,
            goal_tolerance=0.25,
            human_safe_distance=1.5,
            obstacle_safe_distance=0.75,
            reward_minimum=-100.0,
            reward_maximum=100.0,
            clip_rewards=True,
        ),
        weights=RewardWeights(
            potential_shaping=1.0,
            goal_progress=0.0,
            goal_reached=10.0,
            collision=25.0,
            human_proximity=15.0,
            obstacle_clearance=2.0,
            energy_consumption=0.05,
            smooth_motion=0.1,
            velocity_tracking=0.5,
            manipulation_accuracy=0.0,
            task_completion=10.0,
            time_penalty=0.01,
        ),
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    engine = RoboticsRewardShapingEngine()

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"
    profile_id = "navigation_reward_v1"

    print("\nREGISTER REWARD PROFILE")

    profile = create_navigation_reward_profile(
        tenant_id=tenant_id,
        robot_id=robot_id,
        profile_id=profile_id,
    )

    print(
        json.dumps(
            engine.register_profile(
                profile=profile,
                authorized=True,
                actor_id="training_admin",
            ),
            indent=2,
        )
    )

    print("\nACTIVATE REWARD PROFILE")

    engine.activate_profile(
        tenant_id=tenant_id,
        profile_id=profile_id,
        reviewer_id="simulation_reviewer",
        authorized=True,
    )

    print("\nSIMULATE NAVIGATION EPISODE")

    positions = [
        (0.0, 0.0),
        (1.0, 0.5),
        (2.0, 1.0),
        (3.0, 2.0),
        (4.0, 4.0),
    ]

    goal = (4.0, 4.0)

    for index in range(len(positions) - 1):
        current = RewardObservation(
            position=positions[index],
            goal_position=goal,
            velocity=(1.0, 0.5),
            nearest_obstacle_distance=2.0,
            nearest_human_distance=3.0,
            energy_consumption=0.2,
        )

        final_step = (
            index == len(positions) - 2
        )

        next_observation = RewardObservation(
            position=positions[index + 1],
            goal_position=goal,
            velocity=(1.0, 0.5),
            target_velocity=(1.0, 0.5),
            nearest_obstacle_distance=1.5,
            nearest_human_distance=2.5,
            energy_consumption=0.2,
            task_completed=final_step,
        )

        request = RewardRequest(
            environment_id="warehouse_sim_01",
            episode_id="episode_001",
            step_index=index,
            current=current,
            next=next_observation,
            base_reward=0.0,
            terminated=final_step,
            time_delta_seconds=1.0,
        )

        result = engine.calculate_reward(
            tenant_id=tenant_id,
            profile_id=profile_id,
            request=request,
        )

        print(
            json.dumps(
                {
                    "step": index,
                    "final_reward": result["final_reward"],
                    "components": result["components"],
                },
                indent=2,
            )
        )

    print("\nEPISODE SUMMARY")

    print(
        json.dumps(
            engine.get_episode(
                tenant_id=tenant_id,
                profile_id=profile_id,
                environment_id="warehouse_sim_01",
                episode_id="episode_001",
            ),
            indent=2,
        )
    )

    print("\nPROFILE SUMMARY")

    print(
        json.dumps(
            engine.profile_summary(
                tenant_id=tenant_id,
                profile_id=profile_id,
            ),
            indent=2,
        )
    )

    print("\nREWARD COMPONENT ANALYSIS")

    print(
        json.dumps(
            engine.component_analysis(
                tenant_id=tenant_id,
                profile_id=profile_id,
            ),
            indent=2,
        )
    )

    print("\nEXPORT REPORT")

    output = engine.export_json(
        "data/robotics/reward_shaping_report.json",
        tenant_id=tenant_id,
    )

    print(output)


if __name__ == "__main__":
    main()
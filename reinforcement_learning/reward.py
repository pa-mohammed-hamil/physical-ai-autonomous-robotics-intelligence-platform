"""Reward module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/reward.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Reward Calculation and Management Engine

Python 3.11+
Dependencies: Standard library only

Features:
- Configurable reward functions
- Navigation and manipulation rewards
- Goal completion rewards
- Collision and safety penalties
- Energy and time penalties
- Reward component weighting
- Reward clipping
- Episode return calculation
- Discounted returns
- Generalized Advantage Estimation (GAE)
- Reward statistics
- Multi-tenant isolation
- Audit logging
- JSON report export

Compatible with:
    reward_shaping.py
    ppo_trainer.py
    policy.py
    policy_evaluation.py
    replay_buffer.py
    offline_rl.py
    environment.py
    curriculum.py

Safety:
    Simulation and offline training only.
    This module never sends robot control commands.
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

class RewardError(Exception):
    pass


class RewardValidationError(RewardError):
    pass


class RewardNotFound(RewardError):
    pass


class RewardConflict(RewardError):
    pass


class RewardPermissionDenied(RewardError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RewardType(str, Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INSPECTION = "inspection"
    TASK_COMPLETION = "task_completion"
    CUSTOM = "custom"


class RewardStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class RewardComponent(str, Enum):
    BASE = "base"
    PROGRESS = "progress"
    GOAL = "goal"
    COLLISION = "collision"
    HUMAN_PROXIMITY = "human_proximity"
    OBSTACLE_PROXIMITY = "obstacle_proximity"
    ENERGY = "energy"
    TIME = "time"
    SMOOTHNESS = "smoothness"
    MANIPULATION = "manipulation"
    TASK_COMPLETION = "task_completion"
    CUSTOM = "custom"


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
            char.isalnum() or char in "._-"
            for char in value
        )
    ):
        raise RewardValidationError(
            f"Invalid {name}"
        )

    return value


def finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
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


def validate_vector(
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
            f"{name} must have {dimension} dimensions"
        )

    return result


def euclidean_distance(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    a = validate_vector(first, "first")
    b = validate_vector(
        second,
        "second",
        len(a),
    )

    return math.sqrt(
        sum(
            (x - y) ** 2
            for x, y in zip(a, b)
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
# REWARD CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RewardWeights:
    progress: float = 1.0
    goal: float = 10.0
    collision: float = 25.0
    human_proximity: float = 15.0
    obstacle_proximity: float = 2.0
    energy: float = 0.05
    time: float = 0.01
    smoothness: float = 0.1
    manipulation: float = 1.0
    task_completion: float = 10.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            nonnegative(
                getattr(self, name),
                name,
            )


@dataclass(frozen=True)
class RewardConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95

    goal_tolerance: float = 0.25
    human_safe_distance: float = 1.5
    obstacle_safe_distance: float = 0.75

    clip_rewards: bool = True

    reward_minimum: float = -100.0
    reward_maximum: float = 100.0

    maximum_episode_steps: int = 10_000
    maximum_history: int = 10_000

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in ("gamma", "gae_lambda"):
            value = finite_number(
                getattr(self, name),
                name,
            )

            if not 0.0 <= value <= 1.0:
                raise RewardValidationError(
                    f"{name} must be in [0, 1]"
                )

        for name in (
            "goal_tolerance",
            "human_safe_distance",
            "obstacle_safe_distance",
        ):
            if finite_number(
                getattr(self, name),
                name,
            ) <= 0:
                raise RewardValidationError(
                    f"{name} must be positive"
                )

        if (
            finite_number(
                self.reward_minimum,
                "reward_minimum",
            )
            >= finite_number(
                self.reward_maximum,
                "reward_maximum",
            )
        ):
            raise RewardValidationError(
                "Invalid reward bounds"
            )

        for name in (
            "maximum_episode_steps",
            "maximum_history",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise RewardValidationError(
                    f"{name} must be positive"
                )

        if self.simulation_only is not True:
            raise RewardValidationError(
                "Physical robot execution unsupported"
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
    reward_type: RewardType

    config: RewardConfig = field(
        default_factory=RewardConfig
    )

    weights: RewardWeights = field(
        default_factory=RewardWeights
    )

    status: RewardStatus = RewardStatus.DRAFT

    version: int = 1

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "tenant_id",
            "robot_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if not self.name.strip():
            raise RewardValidationError(
                "Profile name cannot be empty"
            )

        if not isinstance(
            self.reward_type,
            RewardType,
        ):
            raise RewardValidationError(
                "Invalid reward type"
            )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# ROBOT STATE
# ============================================================

@dataclass(frozen=True)
class RobotRewardState:
    position: tuple[float, ...]
    goal_position: tuple[float, ...]

    velocity: tuple[float, ...] | None = None

    nearest_obstacle_distance: float | None = None
    nearest_human_distance: float | None = None

    energy_consumption: float = 0.0
    manipulation_error: float | None = None

    collision_detected: bool = False
    task_completed: bool = False

    def __post_init__(self) -> None:
        position = validate_vector(
            self.position,
            "position",
        )

        validate_vector(
            self.goal_position,
            "goal_position",
            len(position),
        )

        if self.velocity is not None:
            validate_vector(
                self.velocity,
                "velocity",
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


# ============================================================
# REWARD REQUEST
# ============================================================

@dataclass(frozen=True)
class RewardRequest:
    environment_id: str
    episode_id: str
    step_index: int

    current_state: RobotRewardState
    next_state: RobotRewardState

    base_reward: float = 0.0

    terminated: bool = False
    truncated: bool = False

    delta_time: float = 1.0

    custom_components: Mapping[str, float] = field(
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
            self.delta_time,
            "delta_time",
        ) <= 0:
            raise RewardValidationError(
                "delta_time must be positive"
            )

        if self.terminated and self.truncated:
            raise RewardValidationError(
                "Transition cannot be both "
                "terminated and truncated"
            )

        for name, value in self.custom_components.items():
            validate_id(
                name,
                "custom_component",
            )

            finite_number(
                value,
                name,
            )


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

    components: dict[str, float]

    raw_reward: float
    final_reward: float

    goal_reached: bool
    collision_detected: bool

    terminated: bool
    truncated: bool

    created_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# REWARD CALCULATOR
# ============================================================

class RewardCalculator:

    @staticmethod
    def calculate(
        *,
        profile: RewardProfile,
        request: RewardRequest,
    ) -> RewardResult:
        config = profile.config
        weights = profile.weights

        current = request.current_state
        next_state = request.next_state

        previous_distance = euclidean_distance(
            current.position,
            current.goal_position,
        )

        next_distance = euclidean_distance(
            next_state.position,
            next_state.goal_position,
        )

        components: dict[str, float] = {}

        # Base environment reward
        components[RewardComponent.BASE.value] = (
            request.base_reward
        )

        # Navigation progress
        progress = (
            previous_distance - next_distance
        )

        components[RewardComponent.PROGRESS.value] = (
            weights.progress * progress
        )

        # Goal completion
        goal_reached = (
            next_distance <= config.goal_tolerance
        )

        components[RewardComponent.GOAL.value] = (
            weights.goal if goal_reached else 0.0
        )

        # Collision
        components[RewardComponent.COLLISION.value] = (
            -weights.collision
            if next_state.collision_detected
            else 0.0
        )

        # Human proximity
        human_penalty = 0.0

        human_distance = (
            next_state.nearest_human_distance
        )

        if (
            human_distance is not None
            and human_distance
            < config.human_safe_distance
        ):
            ratio = (
                config.human_safe_distance
                - human_distance
            ) / config.human_safe_distance

            human_penalty = (
                -weights.human_proximity * ratio
            )

        components[
            RewardComponent.HUMAN_PROXIMITY.value
        ] = human_penalty

        # Obstacle proximity
        obstacle_penalty = 0.0

        obstacle_distance = (
            next_state.nearest_obstacle_distance
        )

        if (
            obstacle_distance is not None
            and obstacle_distance
            < config.obstacle_safe_distance
        ):
            ratio = (
                config.obstacle_safe_distance
                - obstacle_distance
            ) / config.obstacle_safe_distance

            obstacle_penalty = (
                -weights.obstacle_proximity * ratio
            )

        components[
            RewardComponent.OBSTACLE_PROXIMITY.value
        ] = obstacle_penalty

        # Energy usage
        components[RewardComponent.ENERGY.value] = (
            -weights.energy
            * next_state.energy_consumption
        )

        # Time penalty
        components[RewardComponent.TIME.value] = (
            -weights.time
            * request.delta_time
        )

        # Motion smoothness
        smoothness_penalty = 0.0

        if (
            current.velocity is not None
            and next_state.velocity is not None
        ):
            velocity_change = euclidean_distance(
                current.velocity,
                next_state.velocity,
            )

            smoothness_penalty = (
                -weights.smoothness
                * velocity_change
            )

        components[
            RewardComponent.SMOOTHNESS.value
        ] = smoothness_penalty

        # Manipulation accuracy
        manipulation_penalty = 0.0

        if next_state.manipulation_error is not None:
            manipulation_penalty = (
                -weights.manipulation
                * next_state.manipulation_error
            )

        components[
            RewardComponent.MANIPULATION.value
        ] = manipulation_penalty

        # Task completion
        components[
            RewardComponent.TASK_COMPLETION.value
        ] = (
            weights.task_completion
            if next_state.task_completed
            else 0.0
        )

        # Custom rewards
        for name, value in request.custom_components.items():
            components[
                f"{RewardComponent.CUSTOM.value}:{name}"
            ] = finite_number(
                value,
                name,
            )

        raw_reward = sum(
            components.values()
        )

        final_reward = raw_reward

        if config.clip_rewards:
            final_reward = clamp(
                raw_reward,
                config.reward_minimum,
                config.reward_maximum,
            )

        return RewardResult(
            result_id=new_id(),
            tenant_id=profile.tenant_id,
            profile_id=profile.profile_id,
            environment_id=request.environment_id,
            episode_id=request.episode_id,
            step_index=request.step_index,
            components=components,
            raw_reward=raw_reward,
            final_reward=final_reward,
            goal_reached=goal_reached,
            collision_detected=(
                next_state.collision_detected
            ),
            terminated=request.terminated,
            truncated=request.truncated,
        )


# ============================================================
# DISCOUNTED RETURNS
# ============================================================

class ReturnCalculator:

    @staticmethod
    def discounted_returns(
        rewards: Sequence[float],
        *,
        gamma: float = 0.99,
        bootstrap_value: float = 0.0,
        terminated: Sequence[bool] | None = None,
    ) -> list[float]:
        gamma = finite_number(
            gamma,
            "gamma",
        )

        if not 0 <= gamma <= 1:
            raise RewardValidationError(
                "gamma must be in [0, 1]"
            )

        bootstrap_value = finite_number(
            bootstrap_value,
            "bootstrap_value",
        )

        values = [
            finite_number(reward, "reward")
            for reward in rewards
        ]

        if terminated is None:
            terminated = [
                False
            ] * len(values)

        if len(terminated) != len(values):
            raise RewardValidationError(
                "terminated length mismatch"
            )

        returns = [
            0.0
        ] * len(values)

        running_return = bootstrap_value

        for index in reversed(range(len(values))):
            if terminated[index]:
                running_return = 0.0

            running_return = (
                values[index]
                + gamma * running_return
            )

            returns[index] = running_return

        return returns

    @staticmethod
    def generalized_advantage_estimation(
        rewards: Sequence[float],
        values: Sequence[float],
        next_values: Sequence[float],
        terminated: Sequence[bool],
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[list[float], list[float]]:
        """
        GAE:
            delta_t = r_t
                      + gamma * V(s_{t+1}) * (1-done_t)
                      - V(s_t)

            A_t = delta_t
                  + gamma * lambda
                    * (1-done_t) * A_{t+1}

        Note:
            For truncated episodes, provide the
            correct bootstrap value and reset the
            advantage recursion at the boundary.
            This helper treats `terminated` as the
            recursion boundary mask.
        """

        size = len(rewards)

        if not (
            len(values) == size
            and len(next_values) == size
            and len(terminated) == size
        ):
            raise RewardValidationError(
                "GAE input lengths must match"
            )

        gamma = finite_number(
            gamma,
            "gamma",
        )

        gae_lambda = finite_number(
            gae_lambda,
            "gae_lambda",
        )

        if not (
            0 <= gamma <= 1
            and 0 <= gae_lambda <= 1
        ):
            raise RewardValidationError(
                "Invalid GAE parameters"
            )

        advantages = [0.0] * size

        running_advantage = 0.0

        for index in reversed(range(size)):
            reward = finite_number(
                rewards[index],
                "reward",
            )

            value = finite_number(
                values[index],
                "value",
            )

            next_value = finite_number(
                next_values[index],
                "next_value",
            )

            nonterminal = (
                0.0 if terminated[index] else 1.0
            )

            delta = (
                reward
                + gamma * next_value * nonterminal
                - value
            )

            running_advantage = (
                delta
                + gamma
                * gae_lambda
                * nonterminal
                * running_advantage
            )

            advantages[index] = running_advantage

        returns = [
            advantage + finite_number(value, "value")
            for advantage, value
            in zip(advantages, values)
        ]

        return advantages, returns


# ============================================================
# EPISODE TRACKER
# ============================================================

@dataclass
class EpisodeRewardTracker:
    episode_id: str
    environment_id: str
    profile_id: str

    rewards: list[float] = field(
        default_factory=list
    )

    component_totals: dict[str, float] = field(
        default_factory=dict
    )

    collisions: int = 0
    goal_reached: bool = False

    completed: bool = False

    started_at: datetime = field(
        default_factory=utc_now
    )

    completed_at: datetime | None = None

    def record(self, result: RewardResult) -> None:
        if self.completed:
            raise RewardConflict(
                "Episode already completed"
            )

        if result.step_index != len(self.rewards):
            raise RewardConflict(
                "Episode step order mismatch"
            )

        self.rewards.append(
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
            self.goal_reached
            or result.goal_reached
        )

        if result.terminated or result.truncated:
            self.completed = True
            self.completed_at = utc_now()

    def summary(
        self,
        gamma: float = 0.99,
    ) -> dict[str, Any]:
        returns = ReturnCalculator.discounted_returns(
            self.rewards,
            gamma=gamma,
        )

        return {
            "episode_id": self.episode_id,
            "environment_id": self.environment_id,
            "profile_id": self.profile_id,
            "steps": len(self.rewards),
            "total_reward": sum(self.rewards),
            "mean_reward": (
                sum(self.rewards) / len(self.rewards)
                if self.rewards
                else 0.0
            ),
            "discounted_return": (
                returns[0]
                if returns
                else 0.0
            ),
            "collisions": self.collisions,
            "goal_reached": self.goal_reached,
            "completed": self.completed,
            "component_totals": dict(
                self.component_totals
            ),
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at
                else None
            ),
        }


# ============================================================
# ROBOTICS REWARD ENGINE
# ============================================================

class RoboticsRewardEngine:

    def __init__(self) -> None:
        self._profiles: dict[
            tuple[str, str],
            RewardProfile,
        ] = {}

        self._episodes: dict[
            tuple[str, str, str, str],
            EpisodeRewardTracker,
        ] = {}

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
        profile: RewardProfile,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise RewardPermissionDenied(
                "Authorized registration required"
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
            raise RewardNotFound(
                "Reward profile not found"
            )

        return profile

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

            if profile.status == RewardStatus.ARCHIVED:
                raise RewardConflict(
                    "Archived profile cannot be activated"
                )

            profile.status = RewardStatus.ACTIVE
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
        with self._lock:
            profile = self._get_profile(
                tenant_id,
                profile_id,
            )

            if profile.status != RewardStatus.ACTIVE:
                raise RewardConflict(
                    "Reward profile must be active"
                )

            key = (
                tenant_id,
                profile_id,
                request.environment_id,
                request.episode_id,
            )

            tracker = self._episodes.get(key)

            if tracker is None:
                if request.step_index != 0:
                    raise RewardConflict(
                        "Episode must start at step 0"
                    )

                tracker = EpisodeRewardTracker(
                    episode_id=request.episode_id,
                    environment_id=request.environment_id,
                    profile_id=profile_id,
                )

                self._episodes[key] = tracker

            if len(tracker.rewards) >= (
                profile.config.maximum_episode_steps
            ):
                raise RewardConflict(
                    "Maximum episode steps exceeded"
                )

            result = RewardCalculator.calculate(
                profile=profile,
                request=request,
            )

            tracker.record(result)

            if tracker.completed:
                self._log(
                    tenant_id=tenant_id,
                    action="reward_episode_completed",
                    entity_id=request.episode_id,
                    details={
                        "profile_id": profile_id,
                        "total_reward": sum(
                            tracker.rewards
                        ),
                        "steps": len(
                            tracker.rewards
                        ),
                    },
                )

            return result.to_dict()

    # --------------------------------------------------------
    # EPISODE SUMMARY
    # --------------------------------------------------------

    def episode_summary(
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
            tracker = self._episodes.get(key)

            if tracker is None:
                raise RewardNotFound(
                    "Episode not found"
                )

            profile = self._get_profile(
                tenant_id,
                profile_id,
            )

            return tracker.summary(
                gamma=profile.config.gamma
            )

    # --------------------------------------------------------
    # REWARD STATISTICS
    # --------------------------------------------------------

    def reward_statistics(
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
                tracker
                for key, tracker in self._episodes.items()
                if key[0] == tenant_id
                and key[1] == profile_id
            ]

            completed = [
                episode
                for episode in episodes
                if episode.completed
            ]

            returns = [
                sum(episode.rewards)
                for episode in completed
            ]

            successes = sum(
                episode.goal_reached
                for episode in completed
            )

            return {
                "tenant_id": tenant_id,
                "profile_id": profile_id,
                "reward_type": profile.reward_type.value,
                "episodes": len(episodes),
                "completed_episodes": len(completed),
                "mean_episode_reward": (
                    sum(returns) / len(returns)
                    if returns
                    else 0.0
                ),
                "maximum_episode_reward": (
                    max(returns)
                    if returns
                    else 0.0
                ),
                "minimum_episode_reward": (
                    min(returns)
                    if returns
                    else 0.0
                ),
                "success_rate": (
                    successes / len(completed)
                    if completed
                    else 0.0
                ),
                "total_collisions": sum(
                    episode.collisions
                    for episode in episodes
                ),
                "simulation_only": True,
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
                tracker.summary()
                for key, tracker in self._episodes.items()
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
# DEFAULT PROFILE FACTORY
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
        name="Autonomous Navigation Reward",
        reward_type=RewardType.NAVIGATION,
        config=RewardConfig(
            gamma=0.99,
            gae_lambda=0.95,
            goal_tolerance=0.25,
            human_safe_distance=1.5,
            obstacle_safe_distance=0.75,
        ),
        weights=RewardWeights(
            progress=1.0,
            goal=10.0,
            collision=25.0,
            human_proximity=15.0,
            obstacle_proximity=2.0,
            energy=0.05,
            time=0.01,
            smoothness=0.1,
            manipulation=0.0,
            task_completion=10.0,
        ),
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    engine = RoboticsRewardEngine()

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"
    profile_id = "navigation_reward_v1"

    profile = create_navigation_reward_profile(
        tenant_id=tenant_id,
        robot_id=robot_id,
        profile_id=profile_id,
    )

    engine.register_profile(
        profile,
        authorized=True,
        actor_id="training_admin",
    )

    engine.activate_profile(
        tenant_id=tenant_id,
        profile_id=profile_id,
        reviewer_id="simulation_reviewer",
        authorized=True,
    )

    positions = [
        (0.0, 0.0),
        (1.0, 0.5),
        (2.0, 1.0),
        (3.0, 2.0),
        (4.0, 4.0),
    ]

    goal = (4.0, 4.0)

    for index in range(len(positions) - 1):
        final_step = (
            index == len(positions) - 2
        )

        current = RobotRewardState(
            position=positions[index],
            goal_position=goal,
            velocity=(1.0, 0.5),
            nearest_obstacle_distance=2.0,
            nearest_human_distance=3.0,
            energy_consumption=0.1,
        )

        next_state = RobotRewardState(
            position=positions[index + 1],
            goal_position=goal,
            velocity=(1.0, 0.5),
            nearest_obstacle_distance=1.5,
            nearest_human_distance=2.5,
            energy_consumption=0.2,
            task_completed=final_step,
        )

        request = RewardRequest(
            environment_id="warehouse_sim_01",
            episode_id="episode_001",
            step_index=index,
            current_state=current,
            next_state=next_state,
            terminated=final_step,
            delta_time=1.0,
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
                    "reward": result["final_reward"],
                    "components": result["components"],
                },
                indent=2,
            )
        )

    print("\nEPISODE SUMMARY")

    print(
        json.dumps(
            engine.episode_summary(
                tenant_id=tenant_id,
                profile_id=profile_id,
                environment_id="warehouse_sim_01",
                episode_id="episode_001",
            ),
            indent=2,
        )
    )

    print("\nREWARD STATISTICS")

    print(
        json.dumps(
            engine.reward_statistics(
                tenant_id=tenant_id,
                profile_id=profile_id,
            ),
            indent=2,
        )
    )

    print("\nGAE EXAMPLE")

    advantages, returns = (
        ReturnCalculator.generalized_advantage_estimation(
            rewards=[1.0, 0.5, 2.0, 3.0],
            values=[0.4, 0.5, 0.8, 1.0],
            next_values=[0.5, 0.8, 1.0, 0.0],
            terminated=[False, False, False, True],
        )
    )

    print(
        json.dumps(
            {
                "advantages": advantages,
                "returns": returns,
            },
            indent=2,
        )
    )

    print("\nEXPORT REPORT")

    output = engine.export_json(
        "data/robotics/reward_report.json",
        tenant_id=tenant_id,
    )

    print(output)


if __name__ == "__main__":
    main()
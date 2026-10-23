"""Policy Evaluation module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/policy_evaluation.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Policy Evaluation Engine

Python 3.11+
Dependencies:
    pip install numpy

Features:
    - Reinforcement learning policy evaluation
    - Behavioral cloning policy evaluation
    - Offline RL policy evaluation
    - Discrete and continuous action evaluation
    - Simulation episode evaluation
    - Reward and return analysis
    - Discounted return calculation
    - Success and failure rates
    - Collision and safety violation metrics
    - Action accuracy and imitation error
    - Policy comparison and ranking
    - Bootstrap confidence intervals
    - Generalization across simulation scenarios
    - Robustness and domain randomization evaluation
    - Latency and inference performance
    - Policy evaluation thresholds
    - Human-reviewed simulation approval
    - Multi-tenant isolation
    - Audit logging
    - JSON report export

Integrations:
    environment.py
    observation_space.py
    action_space.py
    imitation_learning.py
    offline_rl.py
    curriculum.py
    domain_randomization.py
    simulator.py
    simulation_adapter.py
    evaluation_service.py
    model_registry.py
    safety_monitor.py

Safety:
    Evaluation is limited to simulation and offline data.

    Metrics do not certify physical robot safety.

    The engine does not execute robot commands, control
    actuators, or approve physical deployment.

    Safety metrics depend on the integrity of externally
    supplied simulation records.
"""

from __future__ import annotations

import copy
import json
import math
import random
import statistics
import threading
import uuid

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# ============================================================
# EXCEPTIONS
# ============================================================

class PolicyEvaluationError(Exception):
    pass


class PolicyValidationError(PolicyEvaluationError):
    pass


class PolicyNotFound(PolicyEvaluationError):
    pass


class PolicyConflict(PolicyEvaluationError):
    pass


class PolicyPermissionDenied(PolicyEvaluationError):
    pass


# ============================================================
# ENUMS
# ============================================================

class PolicyType(str, Enum):
    BEHAVIORAL_CLONING = "behavioral_cloning"
    OFFLINE_RL = "offline_rl"
    ONLINE_RL = "online_rl"
    HEURISTIC = "heuristic"
    HYBRID = "hybrid"


class ActionSpaceType(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class EvaluationMode(str, Enum):
    SIMULATION = "simulation"
    OFFLINE_DATASET = "offline_dataset"


class EvaluationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"


class PolicyReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED_FOR_SIMULATION = "approved_for_simulation"
    REJECTED = "rejected"


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
        raise PolicyValidationError(f"Invalid {name}")

    return value


def finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, str))
        or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        )
    ):
        raise PolicyValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise PolicyValidationError(
            f"{name} must be finite"
        )

    return result


def positive_integer(
    value: int,
    name: str,
    minimum: int = 1,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise PolicyValidationError(
            f"{name} must be >= {minimum}"
        )

    return value


def probability(value: Any, name: str) -> float:
    result = finite_number(value, name)

    if not 0.0 <= result <= 1.0:
        raise PolicyValidationError(
            f"{name} must be between 0 and 1"
        )

    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        raise PolicyValidationError(
            "Non-finite JSON value"
        )

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PolicyEvaluationConfig:
    discount_factor: float = 0.99

    minimum_episodes: int = 10
    maximum_episodes: int = 100_000

    maximum_steps_per_episode: int = 100_000

    bootstrap_samples: int = 1_000
    confidence_level: float = 0.95

    random_seed: int = 42

    maximum_audit_events: int = 20_000

    simulation_only: bool = True
    human_approval_required: bool = True

    def __post_init__(self) -> None:
        gamma = finite_number(
            self.discount_factor,
            "discount_factor",
        )

        if not 0.0 <= gamma <= 1.0:
            raise PolicyValidationError(
                "Discount factor must be in [0, 1]"
            )

        for name in (
            "minimum_episodes",
            "maximum_episodes",
            "maximum_steps_per_episode",
            "bootstrap_samples",
            "maximum_audit_events",
        ):
            positive_integer(
                getattr(self, name),
                name,
            )

        if self.maximum_episodes < self.minimum_episodes:
            raise PolicyValidationError(
                "Maximum episodes below minimum"
            )

        level = probability(
            self.confidence_level,
            "confidence_level",
        )

        if not 0.0 < level < 1.0:
            raise PolicyValidationError(
                "Confidence level must be in (0, 1)"
            )

        positive_integer(
            self.random_seed,
            "random_seed",
            minimum=0,
        )

        if self.simulation_only is not True:
            raise PolicyValidationError(
                "Physical robot evaluation is unsupported"
            )

        if self.human_approval_required is not True:
            raise PolicyValidationError(
                "Human approval cannot be disabled"
            )


# ============================================================
# POLICY PROFILE
# ============================================================

@dataclass(frozen=True)
class PolicyProfile:
    policy_id: str
    tenant_id: str
    robot_id: str

    name: str
    version: str

    policy_type: PolicyType
    action_space_type: ActionSpaceType

    observation_dimension: int
    action_dimension: int

    model_reference: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "policy_id",
            "tenant_id",
            "robot_id",
        ):
            validate_id(getattr(self, name), name)

        if not self.name.strip():
            raise PolicyValidationError(
                "Policy name cannot be empty"
            )

        if not self.version.strip():
            raise PolicyValidationError(
                "Policy version cannot be empty"
            )

        if not isinstance(self.policy_type, PolicyType):
            raise PolicyValidationError(
                "Invalid policy type"
            )

        if not isinstance(
            self.action_space_type,
            ActionSpaceType,
        ):
            raise PolicyValidationError(
                "Invalid action space type"
            )

        positive_integer(
            self.observation_dimension,
            "observation_dimension",
        )

        positive_integer(
            self.action_dimension,
            "action_dimension",
        )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


# ============================================================
# EVALUATION STEP
# ============================================================

@dataclass(frozen=True)
class PolicyEvaluationStep:
    timestep: int

    reward: float

    action: int | tuple[float, ...]

    expert_action: int | tuple[float, ...] | None = None

    terminated: bool = False
    truncated: bool = False

    collision: bool = False
    safety_violation: bool = False

    inference_latency_ms: float = 0.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        positive_integer(
            self.timestep,
            "timestep",
            minimum=0,
        )

        finite_number(self.reward, "reward")

        if finite_number(
            self.inference_latency_ms,
            "inference_latency_ms",
        ) < 0:
            raise PolicyValidationError(
                "Inference latency cannot be negative"
            )

        for name in (
            "terminated",
            "truncated",
            "collision",
            "safety_violation",
        ):
            if not isinstance(getattr(self, name), bool):
                raise PolicyValidationError(
                    f"{name} must be boolean"
                )

        if self.terminated and self.truncated:
            raise PolicyValidationError(
                "Step cannot be terminated and truncated"
            )

        for name in ("action", "expert_action"):
            action = getattr(self, name)

            if action is None and name == "expert_action":
                continue

            if isinstance(action, tuple):
                if not action:
                    raise PolicyValidationError(
                        f"{name} cannot be empty"
                    )

                for value in action:
                    finite_number(value, name)

            elif (
                isinstance(action, bool)
                or not isinstance(action, int)
                or action < 0
            ):
                raise PolicyValidationError(
                    f"Invalid {name}"
                )

        try:
            json.dumps(
                json_safe(self.metadata),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise PolicyValidationError(
                "Invalid step metadata"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


# ============================================================
# EVALUATION EPISODE
# ============================================================

@dataclass(frozen=True)
class PolicyEvaluationEpisode:
    episode_id: str
    tenant_id: str
    robot_id: str
    policy_id: str

    scenario_id: str

    steps: tuple[PolicyEvaluationStep, ...]

    success: bool

    environment_seed: int | None = None

    domain_randomization_enabled: bool = False

    source: EvaluationMode = EvaluationMode.SIMULATION

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "tenant_id",
            "robot_id",
            "policy_id",
            "scenario_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if not self.steps:
            raise PolicyValidationError(
                "Evaluation episode cannot be empty"
            )

        if not isinstance(self.success, bool):
            raise PolicyValidationError(
                "success must be boolean"
            )

        if not isinstance(self.source, EvaluationMode):
            raise PolicyValidationError(
                "Invalid evaluation mode"
            )

        if not isinstance(
            self.domain_randomization_enabled,
            bool,
        ):
            raise PolicyValidationError(
                "Invalid domain randomization flag"
            )

        for index, step in enumerate(self.steps):
            if not isinstance(step, PolicyEvaluationStep):
                raise PolicyValidationError(
                    "Invalid episode step"
                )

            if step.timestep != index:
                raise PolicyValidationError(
                    "Timesteps must start at zero and "
                    "increase consecutively"
                )

            if index < len(self.steps) - 1:
                if step.terminated or step.truncated:
                    raise PolicyValidationError(
                        "Terminal step must be last"
                    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                **{
                    key: value
                    for key, value in self.__dict__.items()
                    if key != "steps"
                },
                "steps": [
                    step.to_dict()
                    for step in self.steps
                ],
            }
        )


# ============================================================
# EVALUATION THRESHOLDS
# ============================================================

@dataclass(frozen=True)
class PolicyEvaluationThresholds:
    minimum_success_rate: float = 0.80

    minimum_mean_return: float = 0.0

    maximum_collision_rate: float = 0.05

    maximum_safety_violation_rate: float = 0.0

    maximum_mean_latency_ms: float = 100.0

    minimum_scenario_count: int = 1

    def __post_init__(self) -> None:
        for name in (
            "minimum_success_rate",
            "maximum_collision_rate",
            "maximum_safety_violation_rate",
        ):
            probability(
                getattr(self, name),
                name,
            )

        finite_number(
            self.minimum_mean_return,
            "minimum_mean_return",
        )

        if finite_number(
            self.maximum_mean_latency_ms,
            "maximum_mean_latency_ms",
        ) < 0:
            raise PolicyValidationError(
                "Maximum latency cannot be negative"
            )

        positive_integer(
            self.minimum_scenario_count,
            "minimum_scenario_count",
        )


# ============================================================
# EVALUATION REPORT
# ============================================================

@dataclass
class PolicyEvaluationReport:
    evaluation_id: str

    tenant_id: str
    robot_id: str
    policy_id: str

    episode_count: int
    scenario_count: int

    metrics: dict[str, Any]

    scenario_metrics: dict[str, Any]

    threshold_results: dict[str, bool]

    status: EvaluationStatus

    review_status: PolicyReviewStatus = (
        PolicyReviewStatus.PENDING
    )

    reviewer_id: str | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                **self.__dict__,
                "simulation_only": True,
                "physical_deployment_approved": False,
            }
        )


# ============================================================
# METRICS
# ============================================================

class PolicyMetrics:

    @staticmethod
    def discounted_return(
        rewards: Sequence[float],
        gamma: float,
    ) -> float:
        total = 0.0
        discount = 1.0

        for reward in rewards:
            total += discount * reward
            discount *= gamma

        return total

    @staticmethod
    def bootstrap_mean_interval(
        values: Sequence[float],
        *,
        confidence_level: float,
        samples: int,
        seed: int,
    ) -> tuple[float, float]:
        data = np.asarray(
            values,
            dtype=np.float64,
        )

        if data.size == 0:
            raise PolicyValidationError(
                "Cannot bootstrap empty data"
            )

        if data.size == 1:
            result = float(data[0])
            return result, result

        rng = np.random.default_rng(seed)

        means = np.empty(
            samples,
            dtype=np.float64,
        )

        for index in range(samples):
            selected = rng.integers(
                0,
                len(data),
                size=len(data),
            )

            means[index] = float(
                np.mean(data[selected])
            )

        alpha = 1.0 - confidence_level

        lower, upper = np.quantile(
            means,
            [
                alpha / 2.0,
                1.0 - alpha / 2.0,
            ],
        )

        return float(lower), float(upper)

    @staticmethod
    def imitation_metrics(
        episodes: Sequence[PolicyEvaluationEpisode],
        policy: PolicyProfile,
    ) -> dict[str, Any]:
        discrete_matches = []
        continuous_errors = []

        for episode in episodes:
            for step in episode.steps:
                if step.expert_action is None:
                    continue

                if (
                    policy.action_space_type
                    == ActionSpaceType.DISCRETE
                ):
                    discrete_matches.append(
                        int(step.action == step.expert_action)
                    )

                else:
                    actual = np.asarray(
                        step.action,
                        dtype=np.float64,
                    )

                    expert = np.asarray(
                        step.expert_action,
                        dtype=np.float64,
                    )

                    continuous_errors.append(
                        actual - expert
                    )

        if (
            policy.action_space_type
            == ActionSpaceType.DISCRETE
        ):
            return {
                "expert_labeled_steps": len(
                    discrete_matches
                ),
                "action_accuracy": (
                    float(np.mean(discrete_matches))
                    if discrete_matches
                    else None
                ),
            }

        if not continuous_errors:
            return {
                "expert_labeled_steps": 0,
                "action_mse": None,
                "action_mae": None,
            }

        errors = np.stack(
            continuous_errors
        )

        return {
            "expert_labeled_steps": len(
                continuous_errors
            ),
            "action_mse": float(
                np.mean(errors ** 2)
            ),
            "action_mae": float(
                np.mean(np.abs(errors))
            ),
        }


# ============================================================
# POLICY EVALUATION ENGINE
# ============================================================

class RoboticsPolicyEvaluationEngine:

    def __init__(
        self,
        config: PolicyEvaluationConfig | None = None,
    ) -> None:
        self.config = (
            config or PolicyEvaluationConfig()
        )

        self._policies: dict[
            tuple[str, str],
            PolicyProfile,
        ] = {}

        self._episodes: dict[
            tuple[str, str],
            dict[str, PolicyEvaluationEpisode],
        ] = defaultdict(dict)

        self._reports: dict[
            tuple[str, str],
            PolicyEvaluationReport,
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
    # REGISTER POLICY
    # --------------------------------------------------------

    def register_policy(
        self,
        policy: PolicyProfile,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PolicyPermissionDenied(
                "Authorized policy registration required"
            )

        if not isinstance(policy, PolicyProfile):
            raise PolicyValidationError(
                "Invalid policy profile"
            )

        validate_id(actor_id, "actor_id")

        key = (
            policy.tenant_id,
            policy.policy_id,
        )

        with self._lock:
            if key in self._policies:
                raise PolicyConflict(
                    "Policy already registered"
                )

            self._policies[key] = policy

            self._log(
                tenant_id=policy.tenant_id,
                action="policy_registered",
                entity_id=policy.policy_id,
                actor_id=actor_id,
                details={
                    "robot_id": policy.robot_id,
                    "version": policy.version,
                },
            )

            return policy.to_dict()

    # --------------------------------------------------------
    # GET POLICY
    # --------------------------------------------------------

    def _get_policy(
        self,
        tenant_id: str,
        policy_id: str,
    ) -> PolicyProfile:
        validate_id(tenant_id, "tenant_id")
        validate_id(policy_id, "policy_id")

        policy = self._policies.get(
            (tenant_id, policy_id)
        )

        if policy is None:
            raise PolicyNotFound(
                "Policy not found"
            )

        return policy

    def get_policy(
        self,
        *,
        tenant_id: str,
        policy_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get_policy(
                tenant_id,
                policy_id,
            ).to_dict()

    # --------------------------------------------------------
    # VALIDATE ACTION
    # --------------------------------------------------------

    @staticmethod
    def _validate_action(
        action: int | tuple[float, ...],
        policy: PolicyProfile,
    ) -> None:
        if (
            policy.action_space_type
            == ActionSpaceType.DISCRETE
        ):
            if (
                isinstance(action, bool)
                or not isinstance(action, int)
                or not 0 <= action < policy.action_dimension
            ):
                raise PolicyValidationError(
                    "Discrete action out of range"
                )

        else:
            if (
                not isinstance(action, tuple)
                or len(action) != policy.action_dimension
            ):
                raise PolicyValidationError(
                    "Continuous action dimension mismatch"
                )

            for value in action:
                finite_number(
                    value,
                    "continuous action",
                )

    # --------------------------------------------------------
    # ADD EVALUATION EPISODE
    # --------------------------------------------------------

    def add_episode(
        self,
        episode: PolicyEvaluationEpisode,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PolicyPermissionDenied(
                "Authorized episode ingestion required"
            )

        if not isinstance(
            episode,
            PolicyEvaluationEpisode,
        ):
            raise PolicyValidationError(
                "Invalid evaluation episode"
            )

        with self._lock:
            policy = self._get_policy(
                episode.tenant_id,
                episode.policy_id,
            )

            if policy.robot_id != episode.robot_id:
                raise PolicyConflict(
                    "Episode robot mismatch"
                )

            if (
                len(episode.steps)
                > self.config.maximum_steps_per_episode
            ):
                raise PolicyValidationError(
                    "Episode exceeds maximum steps"
                )

            for step in episode.steps:
                self._validate_action(
                    step.action,
                    policy,
                )

                if step.expert_action is not None:
                    self._validate_action(
                        step.expert_action,
                        policy,
                    )

            key = (
                episode.tenant_id,
                episode.policy_id,
            )

            episodes = self._episodes[key]

            if episode.episode_id in episodes:
                raise PolicyConflict(
                    "Duplicate episode ID"
                )

            if (
                len(episodes)
                >= self.config.maximum_episodes
            ):
                raise PolicyConflict(
                    "Episode capacity reached"
                )

            episodes[
                episode.episode_id
            ] = episode

            self._log(
                tenant_id=episode.tenant_id,
                action="evaluation_episode_added",
                entity_id=episode.episode_id,
                actor_id=actor_id,
                details={
                    "policy_id": episode.policy_id,
                    "scenario_id": episode.scenario_id,
                    "success": episode.success,
                },
            )

            return {
                "episode_id": episode.episode_id,
                "policy_id": episode.policy_id,
                "scenario_id": episode.scenario_id,
                "step_count": len(episode.steps),
                "success": episode.success,
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # BATCH EPISODE INGESTION
    # --------------------------------------------------------

    def add_episodes(
        self,
        episodes: Sequence[PolicyEvaluationEpisode],
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PolicyPermissionDenied(
                "Authorized ingestion required"
            )

        added = []

        for episode in episodes:
            result = self.add_episode(
                episode,
                authorized=True,
                actor_id=actor_id,
            )

            added.append(
                result["episode_id"]
            )

        return {
            "added_count": len(added),
            "episode_ids": added,
        }

    # --------------------------------------------------------
    # COMPUTE METRICS
    # --------------------------------------------------------

    def _compute_metrics(
        self,
        episodes: Sequence[PolicyEvaluationEpisode],
        policy: PolicyProfile,
    ) -> dict[str, Any]:
        if not episodes:
            raise PolicyValidationError(
                "No episodes to evaluate"
            )

        returns = []
        discounted_returns = []

        lengths = []
        successes = []

        collisions = []
        safety_violations = []

        latency_values = []

        domain_randomized_successes = []
        standard_successes = []

        for episode in episodes:
            rewards = [
                step.reward
                for step in episode.steps
            ]

            returns.append(
                float(sum(rewards))
            )

            discounted_returns.append(
                PolicyMetrics.discounted_return(
                    rewards,
                    self.config.discount_factor,
                )
            )

            lengths.append(
                len(episode.steps)
            )

            successes.append(
                int(episode.success)
            )

            collisions.append(
                int(
                    any(
                        step.collision
                        for step in episode.steps
                    )
                )
            )

            safety_violations.append(
                int(
                    any(
                        step.safety_violation
                        for step in episode.steps
                    )
                )
            )

            latency_values.extend(
                step.inference_latency_ms
                for step in episode.steps
            )

            if episode.domain_randomization_enabled:
                domain_randomized_successes.append(
                    int(episode.success)
                )
            else:
                standard_successes.append(
                    int(episode.success)
                )

        return_interval = (
            PolicyMetrics.bootstrap_mean_interval(
                returns,
                confidence_level=(
                    self.config.confidence_level
                ),
                samples=self.config.bootstrap_samples,
                seed=self.config.random_seed,
            )
        )

        success_interval = (
            PolicyMetrics.bootstrap_mean_interval(
                successes,
                confidence_level=(
                    self.config.confidence_level
                ),
                samples=self.config.bootstrap_samples,
                seed=self.config.random_seed + 1,
            )
        )

        metrics = {
            "episode_count": len(episodes),
            "mean_return": float(
                np.mean(returns)
            ),
            "median_return": float(
                np.median(returns)
            ),
            "return_std": float(
                np.std(returns)
            ),
            "minimum_return": float(
                np.min(returns)
            ),
            "maximum_return": float(
                np.max(returns)
            ),
            "mean_discounted_return": float(
                np.mean(discounted_returns)
            ),
            "mean_episode_length": float(
                np.mean(lengths)
            ),
            "success_rate": float(
                np.mean(successes)
            ),
            "collision_rate": float(
                np.mean(collisions)
            ),
            "safety_violation_rate": float(
                np.mean(safety_violations)
            ),
            "mean_inference_latency_ms": float(
                np.mean(latency_values)
            ),
            "p95_inference_latency_ms": float(
                np.percentile(
                    latency_values,
                    95,
                )
            ),
            "mean_return_confidence_interval": {
                "lower": return_interval[0],
                "upper": return_interval[1],
            },
            "success_rate_confidence_interval": {
                "lower": success_interval[0],
                "upper": success_interval[1],
            },
            "domain_randomized_success_rate": (
                float(
                    np.mean(
                        domain_randomized_successes
                    )
                )
                if domain_randomized_successes
                else None
            ),
            "standard_success_rate": (
                float(
                    np.mean(
                        standard_successes
                    )
                )
                if standard_successes
                else None
            ),
        }

        metrics.update(
            PolicyMetrics.imitation_metrics(
                episodes,
                policy,
            )
        )

        return metrics

    # --------------------------------------------------------
    # SCENARIO METRICS
    # --------------------------------------------------------

    def _scenario_metrics(
        self,
        episodes: Sequence[PolicyEvaluationEpisode],
    ) -> dict[str, Any]:
        grouped = defaultdict(list)

        for episode in episodes:
            grouped[
                episode.scenario_id
            ].append(episode)

        result = {}

        for scenario_id, group in grouped.items():
            returns = [
                sum(
                    step.reward
                    for step in episode.steps
                )
                for episode in group
            ]

            result[scenario_id] = {
                "episode_count": len(group),
                "success_rate": float(
                    np.mean(
                        [
                            episode.success
                            for episode in group
                        ]
                    )
                ),
                "mean_return": float(
                    np.mean(returns)
                ),
                "collision_rate": float(
                    np.mean(
                        [
                            any(
                                step.collision
                                for step in episode.steps
                            )
                            for episode in group
                        ]
                    )
                ),
                "safety_violation_rate": float(
                    np.mean(
                        [
                            any(
                                step.safety_violation
                                for step in episode.steps
                            )
                            for episode in group
                        ]
                    )
                ),
            }

        return result

    # --------------------------------------------------------
    # EVALUATE POLICY
    # --------------------------------------------------------

    def evaluate_policy(
        self,
        *,
        tenant_id: str,
        policy_id: str,
        thresholds: PolicyEvaluationThresholds | None = None,
        scenario_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        thresholds = (
            thresholds or PolicyEvaluationThresholds()
        )

        with self._lock:
            policy = self._get_policy(
                tenant_id,
                policy_id,
            )

            episodes = list(
                self._episodes.get(
                    (tenant_id, policy_id),
                    {},
                ).values()
            )

            if scenario_ids is not None:
                selected = set(scenario_ids)

                for scenario_id in selected:
                    validate_id(
                        scenario_id,
                        "scenario_id",
                    )

                episodes = [
                    episode
                    for episode in episodes
                    if episode.scenario_id in selected
                ]

            if not episodes:
                raise PolicyValidationError(
                    "No evaluation episodes available"
                )

            metrics = self._compute_metrics(
                episodes,
                policy,
            )

            scenario_metrics = self._scenario_metrics(
                episodes
            )

            threshold_results = {
                "success_rate": (
                    metrics["success_rate"]
                    >= thresholds.minimum_success_rate
                ),
                "mean_return": (
                    metrics["mean_return"]
                    >= thresholds.minimum_mean_return
                ),
                "collision_rate": (
                    metrics["collision_rate"]
                    <= thresholds.maximum_collision_rate
                ),
                "safety_violation_rate": (
                    metrics["safety_violation_rate"]
                    <= thresholds.maximum_safety_violation_rate
                ),
                "inference_latency": (
                    metrics["mean_inference_latency_ms"]
                    <= thresholds.maximum_mean_latency_ms
                ),
                "scenario_coverage": (
                    len(scenario_metrics)
                    >= thresholds.minimum_scenario_count
                ),
            }

            if (
                len(episodes)
                < self.config.minimum_episodes
            ):
                status = (
                    EvaluationStatus.INSUFFICIENT_DATA
                )

            elif all(
                threshold_results.values()
            ):
                status = EvaluationStatus.PASSED

            else:
                status = EvaluationStatus.FAILED

            report = PolicyEvaluationReport(
                evaluation_id=new_id(),
                tenant_id=tenant_id,
                robot_id=policy.robot_id,
                policy_id=policy_id,
                episode_count=len(episodes),
                scenario_count=len(scenario_metrics),
                metrics=metrics,
                scenario_metrics=scenario_metrics,
                threshold_results=threshold_results,
                status=status,
            )

            self._reports[
                (tenant_id, report.evaluation_id)
            ] = report

            self._log(
                tenant_id=tenant_id,
                action="policy_evaluated",
                entity_id=report.evaluation_id,
                details={
                    "policy_id": policy_id,
                    "status": status.value,
                    "episode_count": len(episodes),
                },
            )

            return report.to_dict()

    # --------------------------------------------------------
    # GET EVALUATION REPORT
    # --------------------------------------------------------

    def get_report(
        self,
        *,
        tenant_id: str,
        evaluation_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            report = self._reports.get(
                (tenant_id, evaluation_id)
            )

            if report is None:
                raise PolicyNotFound(
                    "Evaluation report not found"
                )

            return report.to_dict()

    # --------------------------------------------------------
    # HUMAN REVIEW
    # --------------------------------------------------------

    def review_evaluation(
        self,
        *,
        tenant_id: str,
        evaluation_id: str,
        reviewer_id: str,
        approve: bool,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PolicyPermissionDenied(
                "Authorized human review required"
            )

        validate_id(reviewer_id, "reviewer_id")

        with self._lock:
            report = self._reports.get(
                (tenant_id, evaluation_id)
            )

            if report is None:
                raise PolicyNotFound(
                    "Evaluation report not found"
                )

            if (
                report.review_status
                != PolicyReviewStatus.PENDING
            ):
                raise PolicyConflict(
                    "Evaluation already reviewed"
                )

            if (
                approve
                and report.status
                != EvaluationStatus.PASSED
            ):
                raise PolicyConflict(
                    "Only passing evaluations can be approved"
                )

            report.review_status = (
                PolicyReviewStatus.APPROVED_FOR_SIMULATION
                if approve
                else PolicyReviewStatus.REJECTED
            )

            report.reviewer_id = reviewer_id

            self._log(
                tenant_id=tenant_id,
                action="evaluation_reviewed",
                entity_id=evaluation_id,
                actor_id=reviewer_id,
                details={
                    "approved": approve,
                },
            )

            return {
                "evaluation_id": evaluation_id,
                "policy_id": report.policy_id,
                "review_status": report.review_status.value,
                "reviewer_id": reviewer_id,
                "simulation_approved": approve,
                "physical_deployment_approved": False,
            }

    # --------------------------------------------------------
    # COMPARE POLICIES
    # --------------------------------------------------------

    def compare_policies(
        self,
        *,
        tenant_id: str,
        evaluation_ids: Sequence[str],
    ) -> dict[str, Any]:
        if len(evaluation_ids) < 2:
            raise PolicyValidationError(
                "At least two evaluations required"
            )

        with self._lock:
            comparisons = []

            for evaluation_id in evaluation_ids:
                report = self._reports.get(
                    (tenant_id, evaluation_id)
                )

                if report is None:
                    raise PolicyNotFound(
                        f"Evaluation not found: {evaluation_id}"
                    )

                metrics = report.metrics

                comparisons.append(
                    {
                        "evaluation_id": evaluation_id,
                        "policy_id": report.policy_id,
                        "status": report.status.value,
                        "success_rate": (
                            metrics["success_rate"]
                        ),
                        "mean_return": (
                            metrics["mean_return"]
                        ),
                        "collision_rate": (
                            metrics["collision_rate"]
                        ),
                        "safety_violation_rate": (
                            metrics["safety_violation_rate"]
                        ),
                        "mean_latency_ms": (
                            metrics[
                                "mean_inference_latency_ms"
                            ]
                        ),
                    }
                )

            comparisons.sort(
                key=lambda item: (
                    item["status"]
                    != EvaluationStatus.PASSED.value,
                    item["safety_violation_rate"],
                    item["collision_rate"],
                    -item["success_rate"],
                    -item["mean_return"],
                    item["mean_latency_ms"],
                )
            )

            return {
                "tenant_id": tenant_id,
                "ranking": [
                    {
                        "rank": index + 1,
                        **item,
                    }
                    for index, item in enumerate(
                        comparisons
                    )
                ],
                "ranking_note": (
                    "Descriptive ranking only. "
                    "For a controlled comparison, use "
                    "the same scenario distribution and "
                    "independent evaluation episodes."
                ),
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # LIST REPORTS
    # --------------------------------------------------------

    def list_reports(
        self,
        *,
        tenant_id: str,
        policy_id: str | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        with self._lock:
            return [
                report.to_dict()
                for (tenant, _), report
                in self._reports.items()
                if (
                    tenant == tenant_id
                    and (
                        policy_id is None
                        or report.policy_id == policy_id
                    )
                )
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
        validate_id(tenant_id, "tenant_id")

        positive_integer(
            limit,
            "limit",
        )

        with self._lock:
            return [
                copy.deepcopy(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ][-limit:]

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
            policies = [
                policy.to_dict()
                for (tenant, _), policy
                in self._policies.items()
                if tenant == tenant_id
            ]

            reports = [
                report.to_dict()
                for (tenant, _), report
                in self._reports.items()
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
            "policies": policies,
            "evaluation_reports": reports,
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
# SYNTHETIC SIMULATION EPISODES
# ============================================================

def create_synthetic_evaluation_episodes(
    *,
    tenant_id: str,
    robot_id: str,
    policy_id: str,
    count: int = 50,
    steps_per_episode: int = 30,
    seed: int = 42,
    success_probability: float = 0.90,
) -> list[PolicyEvaluationEpisode]:
    """
    Generate synthetic evaluation records for testing.

    These episodes do not run an actual robot policy
    or simulator. They must not be presented as real
    performance evidence.
    """

    positive_integer(count, "count")
    positive_integer(
        steps_per_episode,
        "steps_per_episode",
    )

    probability(
        success_probability,
        "success_probability",
    )

    rng = np.random.default_rng(seed)

    episodes = []

    for episode_index in range(count):
        scenario_id = (
            "warehouse_navigation"
            if episode_index % 2 == 0
            else "obstacle_avoidance"
        )

        success = bool(
            rng.random() < success_probability
        )

        collision = bool(
            rng.random() < 0.02
        )

        safety_violation = bool(
            rng.random() < 0.005
        )

        steps = []

        for timestep in range(steps_per_episode):
            action = int(
                rng.integers(0, 4)
            )

            expert_action = (
                action
                if rng.random() < 0.90
                else int(rng.integers(0, 4))
            )

            reward = float(
                rng.normal(
                    1.0 if success else -0.2,
                    0.15,
                )
            )

            steps.append(
                PolicyEvaluationStep(
                    timestep=timestep,
                    reward=reward,
                    action=action,
                    expert_action=expert_action,
                    terminated=(
                        timestep == steps_per_episode - 1
                    ),
                    collision=(
                        collision and timestep == 10
                    ),
                    safety_violation=(
                        safety_violation
                        and timestep == 15
                    ),
                    inference_latency_ms=float(
                        rng.uniform(2.0, 15.0)
                    ),
                    metadata={
                        "synthetic": True,
                    },
                )
            )

        episodes.append(
            PolicyEvaluationEpisode(
                episode_id=(
                    f"{policy_id}_episode_"
                    f"{episode_index + 1:05d}"
                ),
                tenant_id=tenant_id,
                robot_id=robot_id,
                policy_id=policy_id,
                scenario_id=scenario_id,
                steps=tuple(steps),
                success=success,
                environment_seed=(
                    seed + episode_index
                ),
                domain_randomization_enabled=(
                    episode_index % 3 == 0
                ),
            )
        )

    return episodes


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    config = PolicyEvaluationConfig(
        discount_factor=0.99,
        minimum_episodes=10,
        bootstrap_samples=500,
        confidence_level=0.95,
        random_seed=42,
    )

    engine = RoboticsPolicyEvaluationEngine(
        config
    )

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"

    policy_a = PolicyProfile(
        policy_id="navigation_bc_v1",
        tenant_id=tenant_id,
        robot_id=robot_id,
        name="Navigation Behavioral Cloning",
        version="1.0.0",
        policy_type=PolicyType.BEHAVIORAL_CLONING,
        action_space_type=ActionSpaceType.DISCRETE,
        observation_dimension=4,
        action_dimension=4,
    )

    policy_b = PolicyProfile(
        policy_id="navigation_cql_v1",
        tenant_id=tenant_id,
        robot_id=robot_id,
        name="Navigation Conservative Q-Learning",
        version="1.0.0",
        policy_type=PolicyType.OFFLINE_RL,
        action_space_type=ActionSpaceType.DISCRETE,
        observation_dimension=4,
        action_dimension=4,
    )

    print("\nREGISTER POLICIES")

    for policy in (policy_a, policy_b):
        print(
            json.dumps(
                engine.register_policy(
                    policy,
                    authorized=True,
                    actor_id="evaluation_admin",
                ),
                indent=2,
            )
        )

    print("\nADD SYNTHETIC EVALUATION EPISODES")

    episodes_a = create_synthetic_evaluation_episodes(
        tenant_id=tenant_id,
        robot_id=robot_id,
        policy_id=policy_a.policy_id,
        count=50,
        seed=42,
        success_probability=0.85,
    )

    episodes_b = create_synthetic_evaluation_episodes(
        tenant_id=tenant_id,
        robot_id=robot_id,
        policy_id=policy_b.policy_id,
        count=50,
        seed=43,
        success_probability=0.95,
    )

    engine.add_episodes(
        episodes_a,
        authorized=True,
        actor_id="evaluation_admin",
    )

    engine.add_episodes(
        episodes_b,
        authorized=True,
        actor_id="evaluation_admin",
    )

    thresholds = PolicyEvaluationThresholds(
        minimum_success_rate=0.80,
        minimum_mean_return=0.0,
        maximum_collision_rate=0.10,
        maximum_safety_violation_rate=0.05,
        maximum_mean_latency_ms=50.0,
        minimum_scenario_count=2,
    )

    print("\nEVALUATE POLICIES")

    report_a = engine.evaluate_policy(
        tenant_id=tenant_id,
        policy_id=policy_a.policy_id,
        thresholds=thresholds,
    )

    report_b = engine.evaluate_policy(
        tenant_id=tenant_id,
        policy_id=policy_b.policy_id,
        thresholds=thresholds,
    )

    print(
        json.dumps(
            {
                "policy_a": {
                    "status": report_a["status"],
                    "metrics": report_a["metrics"],
                },
                "policy_b": {
                    "status": report_b["status"],
                    "metrics": report_b["metrics"],
                },
            },
            indent=2,
        )
    )

    print("\nCOMPARE POLICIES")

    comparison = engine.compare_policies(
        tenant_id=tenant_id,
        evaluation_ids=[
            report_a["evaluation_id"],
            report_b["evaluation_id"],
        ],
    )

    print(
        json.dumps(
            comparison,
            indent=2,
        )
    )

    print("\nEXPORT REPORT")

    output = engine.export_json(
        "data/robotics/policy_evaluation_report.json",
        tenant_id=tenant_id,
    )

    print(output)


if __name__ == "__main__":
    main()
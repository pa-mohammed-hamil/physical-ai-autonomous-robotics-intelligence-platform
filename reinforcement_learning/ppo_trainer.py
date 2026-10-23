"""Ppo Trainer module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/ppo_trainer.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Proximal Policy Optimization (PPO) Trainer

Python 3.11+
Dependencies:
    pip install numpy torch

Features:
    - Proximal Policy Optimization (PPO-Clip)
    - Discrete and continuous action policies
    - Actor-Critic neural networks
    - Generalized Advantage Estimation (GAE-Lambda)
    - Discounted return calculation
    - On-policy rollout collection
    - Parallel-environment rollout support
    - Mini-batch optimization
    - Multiple PPO update epochs
    - Clipped policy objective
    - Value function clipping
    - Entropy regularization
    - Advantage normalization
    - Gradient clipping
    - Approximate KL divergence monitoring
    - Early stopping using target KL
    - Explained variance
    - Learning-rate scheduling
    - Observation normalization
    - Action masking for discrete policies
    - Squashed Gaussian continuous actions
    - Checkpoint saving and loading
    - Training metrics and evaluation reports
    - Multi-tenant policy isolation
    - Audit logging
    - Simulation-only training

Integrations:
    policy.py
    policy_evaluation.py
    observation_space.py
    action_space.py
    environment.py
    curriculum.py
    domain_randomization.py
    simulation_adapter.py
    checkpoints.py

Safety:
    PPO training and inference are simulation-only.

    This module does not connect to physical robots,
    motor controllers, actuators, or ROS2 command topics.

    PPO training does not certify physical robot safety.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
import threading
import uuid

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal


# ============================================================
# EXCEPTIONS
# ============================================================

class PPOError(Exception):
    pass


class PPOValidationError(PPOError):
    pass


class PPONotFound(PPOError):
    pass


class PPOConflict(PPOError):
    pass


class PPOPermissionDenied(PPOError):
    pass


# ============================================================
# ENUMS
# ============================================================

class PPOActionType(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class PPOTrainingStatus(str, Enum):
    CREATED = "created"
    COLLECTING = "collecting"
    READY = "ready"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    APPROVED_FOR_SIMULATION = "approved_for_simulation"
    PAUSED = "paused"
    ARCHIVED = "archived"


class PPOSelectionMode(str, Enum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"


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
        raise PPOValidationError(f"Invalid {name}")

    return value


def finite_float(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, str))
        or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        )
    ):
        raise PPOValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise PPOValidationError(
            f"{name} must be finite"
        )

    return result


def positive_int(
    value: Any,
    name: str,
    minimum: int = 1,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise PPOValidationError(
            f"{name} must be >= {minimum}"
        )

    return value


def validate_vector(
    value: Sequence[float],
    dimension: int,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(
            value,
            dtype=np.float32,
        )
    except (TypeError, ValueError) as exc:
        raise PPOValidationError(
            f"Invalid {name}"
        ) from exc

    if array.shape != (dimension,):
        raise PPOValidationError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.isfinite(array).all():
        raise PPOValidationError(
            f"{name} contains non-finite values"
        )

    return array


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        raise PPOValidationError(
            "Non-finite JSON value"
        )

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PPOConfig:
    observation_dimension: int
    action_dimension: int

    action_type: PPOActionType = PPOActionType.DISCRETE

    hidden_dimensions: tuple[int, ...] = (128, 128)

    learning_rate: float = 3e-4

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2

    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01

    max_gradient_norm: float = 0.5

    update_epochs: int = 10
    minibatch_size: int = 64

    rollout_steps: int = 2048

    target_kl: float = 0.02

    normalize_advantages: bool = True

    minimum_log_std: float = -5.0
    maximum_log_std: float = 2.0

    observation_clip: float = 10.0

    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None

    random_seed: int = 42

    device: str = "cpu"

    maximum_audit_events: int = 20_000

    simulation_only: bool = True
    human_approval_required: bool = True

    def __post_init__(self) -> None:
        for name in (
            "observation_dimension",
            "action_dimension",
            "update_epochs",
            "minibatch_size",
            "rollout_steps",
            "maximum_audit_events",
        ):
            positive_int(getattr(self, name), name)

        if not isinstance(self.action_type, PPOActionType):
            raise PPOValidationError(
                "Invalid action type"
            )

        if not self.hidden_dimensions:
            raise PPOValidationError(
                "Hidden layers cannot be empty"
            )

        for dimension in self.hidden_dimensions:
            positive_int(dimension, "hidden_dimension")

        for name in (
            "learning_rate",
            "clip_epsilon",
            "value_clip_epsilon",
            "max_gradient_norm",
            "target_kl",
            "observation_clip",
        ):
            if finite_float(getattr(self, name), name) <= 0:
                raise PPOValidationError(
                    f"{name} must be positive"
                )

        for name in ("gamma", "gae_lambda"):
            value = finite_float(
                getattr(self, name),
                name,
            )

            if not 0.0 <= value <= 1.0:
                raise PPOValidationError(
                    f"{name} must be in [0, 1]"
                )

        for name in (
            "value_loss_coefficient",
            "entropy_coefficient",
        ):
            if finite_float(
                getattr(self, name),
                name,
            ) < 0:
                raise PPOValidationError(
                    f"{name} cannot be negative"
                )

        if (
            self.minimum_log_std
            >= self.maximum_log_std
        ):
            raise PPOValidationError(
                "Invalid log standard deviation bounds"
            )

        if self.device not in ("cpu", "cuda"):
            raise PPOValidationError(
                "Supported devices: cpu, cuda"
            )

        if (
            self.device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise PPOValidationError(
                "CUDA is unavailable"
            )

        if self.action_type == PPOActionType.CONTINUOUS:
            if (
                self.action_low is None
                or self.action_high is None
            ):
                raise PPOValidationError(
                    "Continuous policies require action bounds"
                )

            low = validate_vector(
                self.action_low,
                self.action_dimension,
                "action_low",
            )

            high = validate_vector(
                self.action_high,
                self.action_dimension,
                "action_high",
            )

            if np.any(low >= high):
                raise PPOValidationError(
                    "Each action lower bound must be "
                    "less than its upper bound"
                )

        elif (
            self.action_low is not None
            or self.action_high is not None
        ):
            raise PPOValidationError(
                "Discrete policies do not use action bounds"
            )

        if self.simulation_only is not True:
            raise PPOValidationError(
                "Physical training is unsupported"
            )

        if self.human_approval_required is not True:
            raise PPOValidationError(
                "Human approval cannot be disabled"
            )


# ============================================================
# NEURAL NETWORK
# ============================================================

def build_mlp(
    input_dimension: int,
    output_dimension: int,
    hidden_dimensions: tuple[int, ...],
) -> nn.Sequential:
    dimensions = (
        input_dimension,
        *hidden_dimensions,
        output_dimension,
    )

    layers: list[nn.Module] = []

    for index in range(len(dimensions) - 1):
        layers.append(
            nn.Linear(
                dimensions[index],
                dimensions[index + 1],
            )
        )

        if index < len(dimensions) - 2:
            layers.append(nn.Tanh())

    return nn.Sequential(*layers)


class PPOActorCritic(nn.Module):

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()

        self.config = config

        self.actor = build_mlp(
            config.observation_dimension,
            config.action_dimension,
            config.hidden_dimensions,
        )

        self.critic = build_mlp(
            config.observation_dimension,
            1,
            config.hidden_dimensions,
        )

        if config.action_type == PPOActionType.CONTINUOUS:
            self.log_std = nn.Parameter(
                torch.zeros(config.action_dimension)
            )

            low = torch.tensor(
                config.action_low,
                dtype=torch.float32,
            )

            high = torch.tensor(
                config.action_high,
                dtype=torch.float32,
            )

            self.register_buffer(
                "action_scale",
                (high - low) / 2.0,
            )

            self.register_buffer(
                "action_bias",
                (high + low) / 2.0,
            )

    def value(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.critic(
            observations
        ).squeeze(-1)

    def _distribution(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> Categorical | Normal:
        actor_output = self.actor(observations)

        if self.config.action_type == PPOActionType.DISCRETE:
            if action_masks is not None:
                if action_masks.shape != actor_output.shape:
                    raise PPOValidationError(
                        "Action mask shape mismatch"
                    )

                if not action_masks.any(dim=-1).all():
                    raise PPOValidationError(
                        "Every observation must allow an action"
                    )

                actor_output = actor_output.masked_fill(
                    ~action_masks,
                    float("-inf"),
                )

            return Categorical(logits=actor_output)

        if action_masks is not None:
            raise PPOValidationError(
                "Continuous action masking unsupported"
            )

        log_std = torch.clamp(
            self.log_std,
            self.config.minimum_log_std,
            self.config.maximum_log_std,
        )

        std = log_std.exp().expand_as(actor_output)

        return Normal(actor_output, std)

    def sample_action(
        self,
        observations: torch.Tensor,
        *,
        deterministic: bool = False,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        distribution = self._distribution(
            observations,
            action_masks,
        )

        values = self.value(observations)

        if self.config.action_type == PPOActionType.DISCRETE:
            if deterministic:
                actions = torch.argmax(
                    distribution.logits,
                    dim=-1,
                )
            else:
                actions = distribution.sample()

            log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy()

            return actions, log_probs, entropy, values

        if deterministic:
            raw_actions = distribution.mean
        else:
            raw_actions = distribution.sample()

        squashed = torch.tanh(raw_actions)

        actions = (
            self.action_bias
            + self.action_scale * squashed
        )

        log_probs = (
            distribution.log_prob(raw_actions)
            - torch.log(
                self.action_scale
                * (1.0 - squashed.square())
                + 1e-6
            )
        ).sum(dim=-1)

        # Base Gaussian entropy is an approximation
        # when actions are transformed by tanh.
        entropy = distribution.entropy().sum(dim=-1)

        return actions, log_probs, entropy, values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        distribution = self._distribution(
            observations,
            action_masks,
        )

        values = self.value(observations)

        if self.config.action_type == PPOActionType.DISCRETE:
            log_probs = distribution.log_prob(
                actions.long()
            )

            entropy = distribution.entropy()

            return log_probs, entropy, values

        normalized_actions = (
            actions - self.action_bias
        ) / self.action_scale

        normalized_actions = torch.clamp(
            normalized_actions,
            -1.0 + 1e-6,
            1.0 - 1e-6,
        )

        raw_actions = torch.atanh(
            normalized_actions
        )

        log_probs = (
            distribution.log_prob(raw_actions)
            - torch.log(
                self.action_scale
                * (1.0 - normalized_actions.square())
                + 1e-6
            )
        ).sum(dim=-1)

        entropy = distribution.entropy().sum(dim=-1)

        return log_probs, entropy, values


# ============================================================
# ROLLOUT STEP
# ============================================================

@dataclass(frozen=True)
class PPORolloutStep:
    environment_id: str

    observation: tuple[float, ...]

    action: int | tuple[float, ...]

    reward: float

    log_probability: float
    value_estimate: float

    terminated: bool
    truncated: bool

    next_value_estimate: float

    action_mask: tuple[bool, ...] | None = None

    policy_version: int = 0

    def __post_init__(self) -> None:
        validate_id(
            self.environment_id,
            "environment_id",
        )

        for name in (
            "reward",
            "log_probability",
            "value_estimate",
            "next_value_estimate",
        ):
            finite_float(
                getattr(self, name),
                name,
            )

        if self.terminated and self.truncated:
            raise PPOValidationError(
                "Step cannot be terminated and truncated"
            )

        positive_int(
            self.policy_version,
            "policy_version",
            minimum=0,
        )


# ============================================================
# ROLLOUT BUFFER
# ============================================================

class PPORolloutBuffer:
    """
    Stores complete transitions for one PPO update.

    Each transition includes the critic estimate for
    the next state, allowing correct GAE computation
    for terminated and time-limit-truncated episodes.

    Rollouts must be collected with the same policy
    version that is used for the PPO update.
    """

    def __init__(
        self,
        config: PPOConfig,
    ) -> None:
        self.config = config
        self.steps: list[PPORolloutStep] = []

    def add(
        self,
        step: PPORolloutStep,
    ) -> None:
        if len(self.steps) >= self.config.rollout_steps:
            raise PPOConflict(
                "Rollout buffer is full"
            )

        validate_vector(
            step.observation,
            self.config.observation_dimension,
            "observation",
        )

        if self.config.action_type == PPOActionType.DISCRETE:
            if (
                isinstance(step.action, bool)
                or not isinstance(step.action, int)
                or not 0 <= step.action < self.config.action_dimension
            ):
                raise PPOValidationError(
                    "Invalid discrete action"
                )

            if step.action_mask is not None:
                if (
                    len(step.action_mask)
                    != self.config.action_dimension
                    or not all(
                        isinstance(value, bool)
                        for value in step.action_mask
                    )
                    or not any(step.action_mask)
                    or not step.action_mask[step.action]
                ):
                    raise PPOValidationError(
                        "Invalid action mask"
                    )

        else:
            action = validate_vector(
                step.action,
                self.config.action_dimension,
                "action",
            )

            low = np.asarray(
                self.config.action_low,
                dtype=np.float32,
            )

            high = np.asarray(
                self.config.action_high,
                dtype=np.float32,
            )

            if np.any(action < low) or np.any(action > high):
                raise PPOValidationError(
                    "Continuous action outside bounds"
                )

            if step.action_mask is not None:
                raise PPOValidationError(
                    "Continuous masks unsupported"
                )

        if step.terminated and abs(step.next_value_estimate) > 1e-8:
            raise PPOValidationError(
                "Terminated transitions require "
                "zero next-state value"
            )

        self.steps.append(step)

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)

    def compute_gae(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.steps:
            raise PPOValidationError(
                "Rollout buffer is empty"
            )

        advantages = np.zeros(
            len(self.steps),
            dtype=np.float32,
        )

        returns = np.zeros(
            len(self.steps),
            dtype=np.float32,
        )

        # Compute GAE separately for each environment.
        # Time-limit truncation bootstraps the critic,
        # but does not connect to the next episode.

        environment_indices: dict[str, list[int]] = {}

        for index, step in enumerate(self.steps):
            environment_indices.setdefault(
                step.environment_id,
                [],
            ).append(index)

        for indices in environment_indices.values():
            gae = 0.0

            for position in reversed(range(len(indices))):
                index = indices[position]
                step = self.steps[index]

                bootstrap_mask = (
                    0.0 if step.terminated else 1.0
                )

                delta = (
                    step.reward
                    + self.config.gamma
                    * bootstrap_mask
                    * step.next_value_estimate
                    - step.value_estimate
                )

                # GAE cannot cross an episode boundary.
                # Both terminated and truncated episodes
                # reset the recursive advantage term.

                continuation = (
                    0.0
                    if step.terminated or step.truncated
                    else 1.0
                )

                gae = (
                    delta
                    + self.config.gamma
                    * self.config.gae_lambda
                    * continuation
                    * gae
                )

                advantages[index] = gae

                returns[index] = (
                    gae + step.value_estimate
                )

        return advantages, returns

    def as_tensors(
        self,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        advantages, returns = self.compute_gae()

        observations = np.asarray(
            [
                step.observation
                for step in self.steps
            ],
            dtype=np.float32,
        )

        if self.config.action_type == PPOActionType.DISCRETE:
            actions = torch.tensor(
                [step.action for step in self.steps],
                dtype=torch.long,
                device=device,
            )

            masks = torch.tensor(
                [
                    step.action_mask
                    if step.action_mask is not None
                    else (
                        True,
                    ) * self.config.action_dimension
                    for step in self.steps
                ],
                dtype=torch.bool,
                device=device,
            )

        else:
            actions = torch.tensor(
                [step.action for step in self.steps],
                dtype=torch.float32,
                device=device,
            )

            masks = None

        result = {
            "observations": torch.tensor(
                observations,
                dtype=torch.float32,
                device=device,
            ),
            "actions": actions,
            "old_log_probs": torch.tensor(
                [
                    step.log_probability
                    for step in self.steps
                ],
                dtype=torch.float32,
                device=device,
            ),
            "old_values": torch.tensor(
                [
                    step.value_estimate
                    for step in self.steps
                ],
                dtype=torch.float32,
                device=device,
            ),
            "advantages": torch.tensor(
                advantages,
                dtype=torch.float32,
                device=device,
            ),
            "returns": torch.tensor(
                returns,
                dtype=torch.float32,
                device=device,
            ),
        }

        if masks is not None:
            result["action_masks"] = masks

        return result


# ============================================================
# TRAINING METRICS
# ============================================================

@dataclass
class PPOTrainingMetrics:
    update_number: int

    policy_loss: float
    value_loss: float
    entropy: float

    total_loss: float

    approximate_kl: float
    clip_fraction: float

    explained_variance: float

    gradient_norm: float

    learning_rate: float

    samples_processed: int
    optimization_epochs: int

    early_stopped: bool

    created_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


# ============================================================
# TRAINING SESSION
# ============================================================

@dataclass
class PPOTrainingSession:
    session_id: str
    tenant_id: str
    robot_id: str
    policy_id: str

    name: str
    config: PPOConfig

    status: PPOTrainingStatus = (
        PPOTrainingStatus.CREATED
    )

    policy_version: int = 0
    update_count: int = 0
    total_environment_steps: int = 0

    metrics_history: list[PPOTrainingMetrics] = field(
        default_factory=list
    )

    evaluation_reference: str | None = None
    reviewer_id: str | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "session_id": self.session_id,
                "tenant_id": self.tenant_id,
                "robot_id": self.robot_id,
                "policy_id": self.policy_id,
                "name": self.name,
                "status": self.status,
                "policy_version": self.policy_version,
                "update_count": self.update_count,
                "total_environment_steps": (
                    self.total_environment_steps
                ),
                "metrics_history": [
                    item.to_dict()
                    for item in self.metrics_history
                ],
                "evaluation_reference": (
                    self.evaluation_reference
                ),
                "reviewer_id": self.reviewer_id,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "simulation_only": True,
                "physical_deployment_approved": False,
            }
        )


# ============================================================
# SIMULATION ENVIRONMENT INTERFACE
# ============================================================

class SimulationEnvironment(Protocol):
    """
    Gymnasium-style simulation interface.

    reset() -> (observation, info)

    step(action) ->
        (observation, reward, terminated, truncated, info)

    The implementation must be simulation-only.
    """

    def reset(
        self,
    ) -> tuple[
        Sequence[float],
        Mapping[str, Any],
    ]:
        ...

    def step(
        self,
        action: Any,
    ) -> tuple[
        Sequence[float],
        float,
        bool,
        bool,
        Mapping[str, Any],
    ]:
        ...


# ============================================================
# PPO TRAINER
# ============================================================

class RoboticsPPOTrainer:

    def __init__(self) -> None:
        self._sessions: dict[
            tuple[str, str],
            PPOTrainingSession,
        ] = {}

        self._models: dict[
            tuple[str, str],
            PPOActorCritic,
        ] = {}

        self._optimizers: dict[
            tuple[str, str],
            torch.optim.Optimizer,
        ] = {}

        self._buffers: dict[
            tuple[str, str],
            PPORolloutBuffer,
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
    # CREATE SESSION
    # --------------------------------------------------------

    def create_session(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        policy_id: str,
        session_id: str,
        name: str,
        config: PPOConfig,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PPOPermissionDenied(
                "Authorized training session required"
            )

        for field_name, value in (
            ("tenant_id", tenant_id),
            ("robot_id", robot_id),
            ("policy_id", policy_id),
            ("session_id", session_id),
            ("actor_id", actor_id),
        ):
            validate_id(value, field_name)

        if not isinstance(name, str) or not name.strip():
            raise PPOValidationError(
                "Session name cannot be empty"
            )

        if not isinstance(config, PPOConfig):
            raise PPOValidationError(
                "Invalid PPO configuration"
            )

        key = (tenant_id, session_id)

        with self._lock:
            if key in self._sessions:
                raise PPOConflict(
                    "Training session already exists"
                )

            torch.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)

            model = PPOActorCritic(config).to(
                config.device
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                eps=1e-5,
            )

            session = PPOTrainingSession(
                session_id=session_id,
                tenant_id=tenant_id,
                robot_id=robot_id,
                policy_id=policy_id,
                name=name.strip(),
                config=config,
            )

            self._sessions[key] = session
            self._models[key] = model
            self._optimizers[key] = optimizer
            self._buffers[key] = PPORolloutBuffer(
                config
            )

            self._log(
                tenant_id=tenant_id,
                action="ppo_session_created",
                entity_id=session_id,
                actor_id=actor_id,
                details={
                    "policy_id": policy_id,
                    "action_type": config.action_type.value,
                },
            )

            return session.to_dict()

    # --------------------------------------------------------
    # GET SESSION
    # --------------------------------------------------------

    def _get_session(
        self,
        tenant_id: str,
        session_id: str,
    ) -> PPOTrainingSession:
        validate_id(tenant_id, "tenant_id")
        validate_id(session_id, "session_id")

        session = self._sessions.get(
            (tenant_id, session_id)
        )

        if session is None:
            raise PPONotFound(
                "PPO training session not found"
            )

        return session

    def get_session(
        self,
        *,
        tenant_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get_session(
                tenant_id,
                session_id,
            ).to_dict()

    # --------------------------------------------------------
    # SELECT ACTION
    # --------------------------------------------------------

    def select_action(
        self,
        *,
        tenant_id: str,
        session_id: str,
        observation: Sequence[float],
        deterministic: bool = False,
        action_mask: Sequence[bool] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._get_session(
                tenant_id,
                session_id,
            )

            if session.status == PPOTrainingStatus.ARCHIVED:
                raise PPOConflict(
                    "Archived session cannot select actions"
                )

            config = session.config

            vector = validate_vector(
                observation,
                config.observation_dimension,
                "observation",
            )

            mask_tensor = None

            if action_mask is not None:
                if config.action_type != PPOActionType.DISCRETE:
                    raise PPOValidationError(
                        "Continuous masks unsupported"
                    )

                if (
                    len(action_mask) != config.action_dimension
                    or not all(
                        isinstance(value, bool)
                        for value in action_mask
                    )
                    or not any(action_mask)
                ):
                    raise PPOValidationError(
                        "Invalid action mask"
                    )

                mask_tensor = torch.tensor(
                    [list(action_mask)],
                    dtype=torch.bool,
                    device=config.device,
                )

            observation_tensor = torch.tensor(
                vector,
                dtype=torch.float32,
                device=config.device,
            ).unsqueeze(0)

            model = self._models[
                (tenant_id, session_id)
            ]

            model.eval()

            with torch.no_grad():
                (
                    actions,
                    log_probs,
                    entropy,
                    values,
                ) = model.sample_action(
                    observation_tensor,
                    deterministic=deterministic,
                    action_masks=mask_tensor,
                )

            if config.action_type == PPOActionType.DISCRETE:
                action = int(actions.item())
            else:
                action = tuple(
                    float(value)
                    for value in actions[0].cpu().tolist()
                )

            return {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "policy_id": session.policy_id,
                "policy_version": session.policy_version,
                "action": action,
                "log_probability": float(
                    log_probs.item()
                ),
                "entropy": float(
                    entropy.item()
                ),
                "value_estimate": float(
                    values.item()
                ),
                "deterministic": deterministic,
                "simulation_only": True,
                "physical_execution_authorized": False,
            }

    # --------------------------------------------------------
    # VALUE ESTIMATION
    # --------------------------------------------------------

    def estimate_value(
        self,
        *,
        tenant_id: str,
        session_id: str,
        observation: Sequence[float],
    ) -> float:
        with self._lock:
            session = self._get_session(
                tenant_id,
                session_id,
            )

            vector = validate_vector(
                observation,
                session.config.observation_dimension,
                "observation",
            )

            tensor = torch.tensor(
                vector,
                dtype=torch.float32,
                device=session.config.device,
            ).unsqueeze(0)

            model = self._models[
                (tenant_id, session_id)
            ]

            with torch.no_grad():
                value = model.value(tensor)

            return float(value.item())

    # --------------------------------------------------------
    # ADD ROLLOUT STEP
    # --------------------------------------------------------

    def add_rollout_step(
        self,
        *,
        tenant_id: str,
        session_id: str,
        step: PPORolloutStep,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PPOPermissionDenied(
                "Authorized rollout ingestion required"
            )

        if not isinstance(step, PPORolloutStep):
            raise PPOValidationError(
                "Invalid rollout step"
            )

        with self._lock:
            session = self._get_session(
                tenant_id,
                session_id,
            )

            if session.status in (
                PPOTrainingStatus.ARCHIVED,
                PPOTrainingStatus.APPROVED_FOR_SIMULATION,
            ):
                raise PPOConflict(
                    "Cannot modify approved or archived session"
                )

            if step.policy_version != session.policy_version:
                raise PPOConflict(
                    "Stale rollout policy version"
                )

            buffer = self._buffers[
                (tenant_id, session_id)
            ]

            buffer.add(step)

            session.total_environment_steps += 1
            session.status = PPOTrainingStatus.COLLECTING
            session.updated_at = utc_now()

            if len(buffer) >= session.config.rollout_steps:
                session.status = PPOTrainingStatus.READY

            return {
                "session_id": session_id,
                "rollout_size": len(buffer),
                "rollout_capacity": (
                    session.config.rollout_steps
                ),
                "ready": (
                    session.status == PPOTrainingStatus.READY
                ),
            }

    # --------------------------------------------------------
    # COLLECT SIMULATION ROLLOUT
    # --------------------------------------------------------

    def collect_rollout(
        self,
        *,
        tenant_id: str,
        session_id: str,
        environments: Mapping[
            str,
            SimulationEnvironment,
        ],
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        """
        Collect on-policy transitions from simulation
        environments implementing the Gymnasium API.

        Environments must be trusted simulation adapters.
        No physical hardware interface is supported.
        """

        if authorized is not True:
            raise PPOPermissionDenied(
                "Authorized simulation collection required"
            )

        if not environments:
            raise PPOValidationError(
                "At least one environment is required"
            )

        with self._lock:
            session = self._get_session(
                tenant_id,
                session_id,
            )

            if session.status in (
                PPOTrainingStatus.ARCHIVED,
                PPOTrainingStatus.APPROVED_FOR_SIMULATION,
            ):
                raise PPOConflict(
                    "Session cannot collect rollouts"
                )

            key = (tenant_id, session_id)
            buffer = self._buffers[key]

            if len(buffer) > 0:
                raise PPOConflict(
                    "Clear or train existing rollout first"
                )

            observations: dict[str, Sequence[float]] = {}

            for environment_id, environment in environments.items():
                validate_id(
                    environment_id,
                    "environment_id",
                )

                observation, _ = environment.reset()

                validate_vector(
                    observation,
                    session.config.observation_dimension,
                    "initial observation",
                )

                observations[environment_id] = observation

            collected = 0
            completed_episodes = 0
            episode_rewards = defaultdict(float)

            while collected < session.config.rollout_steps:
                for environment_id, environment in environments.items():
                    if collected >= session.config.rollout_steps:
                        break

                    observation = observations[
                        environment_id
                    ]

                    selected = self.select_action(
                        tenant_id=tenant_id,
                        session_id=session_id,
                        observation=observation,
                        deterministic=False,
                    )

                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = environment.step(
                        selected["action"]
                    )

                    validate_vector(
                        next_observation,
                        session.config.observation_dimension,
                        "next observation",
                    )

                    finite_float(
                        reward,
                        "reward",
                    )

                    if not isinstance(terminated, bool):
                        raise PPOValidationError(
                            "terminated must be boolean"
                        )

                    if not isinstance(truncated, bool):
                        raise PPOValidationError(
                            "truncated must be boolean"
                        )

                    if terminated and truncated:
                        raise PPOValidationError(
                            "Transition cannot be both "
                            "terminated and truncated"
                        )

                    next_value = (
                        0.0
                        if terminated
                        else self.estimate_value(
                            tenant_id=tenant_id,
                            session_id=session_id,
                            observation=next_observation,
                        )
                    )

                    step = PPORolloutStep(
                        environment_id=environment_id,
                        observation=tuple(
                            float(value)
                            for value in observation
                        ),
                        action=selected["action"],
                        reward=float(reward),
                        log_probability=selected[
                            "log_probability"
                        ],
                        value_estimate=selected[
                            "value_estimate"
                        ],
                        next_value_estimate=next_value,
                        terminated=terminated,
                        truncated=truncated,
                        policy_version=(
                            session.policy_version
                        ),
                    )

                    buffer.add(step)

                    collected += 1
                    session.total_environment_steps += 1

                    episode_rewards[
                        environment_id
                    ] += float(reward)

                    if terminated or truncated:
                        completed_episodes += 1

                        reset_observation, _ = (
                            environment.reset()
                        )

                        observations[
                            environment_id
                        ] = reset_observation

                        episode_rewards[
                            environment_id
                        ] = 0.0

                    else:
                        observations[
                            environment_id
                        ] = next_observation

            session.status = PPOTrainingStatus.READY
            session.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                action="ppo_rollout_collected",
                entity_id=session_id,
                actor_id=actor_id,
                details={
                    "steps": collected,
                    "completed_episodes": completed_episodes,
                    "environment_count": len(environments),
                },
            )

            return {
                "session_id": session_id,
                "steps_collected": collected,
                "completed_episodes": completed_episodes,
                "environment_count": len(environments),
                "policy_version": session.policy_version,
                "ready_for_update": True,
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # PPO UPDATE
    # --------------------------------------------------------

    def update_policy(
        self,
        *,
        tenant_id: str,
        session_id: str,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PPOPermissionDenied(
                "Authorized PPO update required"
            )

        with self._lock:
            session = self._get_session(
                tenant_id,
                session_id,
            )

            if session.status in (
                PPOTrainingStatus.ARCHIVED,
                PPOTrainingStatus.APPROVED_FOR_SIMULATION,
            ):
                raise PPOConflict(
                    "Cannot train approved or archived session"
                )

            key = (tenant_id, session_id)

            buffer = self._buffers[key]

            if len(buffer) == 0:
                raise PPOValidationError(
                    "Rollout buffer is empty"
                )

            if any(
                step.policy_version != session.policy_version
                for step in buffer.steps
            ):
                raise PPOConflict(
                    "Rollout contains stale policy versions"
                )

            config = session.config
            device = torch.device(config.device)

            tensors = buffer.as_tensors(device)

            advantages = tensors["advantages"]

            if (
                config.normalize_advantages
                and len(advantages) > 1
            ):
                tensors["advantages"] = (
                    advantages - advantages.mean()
                ) / (
                    advantages.std(unbiased=False)
                    + 1e-8
                )

            model = self._models[key]
            optimizer = self._optimizers[key]

            model.train()
            session.status = PPOTrainingStatus.TRAINING

            sample_count = len(buffer)

            policy_losses = []
            value_losses = []
            entropy_values = []
            total_losses = []
            kl_values = []
            clip_fractions = []
            gradient_norms = []

            early_stopped = False
            completed_epochs = 0

            for epoch in range(config.update_epochs):
                indices = torch.randperm(
                    sample_count,
                    device=device,
                )

                epoch_kls = []

                for start in range(
                    0,
                    sample_count,
                    config.minibatch_size,
                ):
                    batch_indices = indices[
                        start:
                        start + config.minibatch_size
                    ]

                    observations = tensors[
                        "observations"
                    ][batch_indices]

                    actions = tensors[
                        "actions"
                    ][batch_indices]

                    old_log_probs = tensors[
                        "old_log_probs"
                    ][batch_indices]

                    old_values = tensors[
                        "old_values"
                    ][batch_indices]

                    batch_advantages = tensors[
                        "advantages"
                    ][batch_indices]

                    returns = tensors[
                        "returns"
                    ][batch_indices]

                    masks = (
                        tensors["action_masks"][
                            batch_indices
                        ]
                        if "action_masks" in tensors
                        else None
                    )

                    (
                        new_log_probs,
                        entropy,
                        new_values,
                    ) = model.evaluate_actions(
                        observations,
                        actions,
                        masks,
                    )

                    log_ratio = (
                        new_log_probs - old_log_probs
                    )

                    ratio = torch.exp(log_ratio)

                    unclipped_objective = (
                        ratio * batch_advantages
                    )

                    clipped_objective = (
                        torch.clamp(
                            ratio,
                            1.0 - config.clip_epsilon,
                            1.0 + config.clip_epsilon,
                        )
                        * batch_advantages
                    )

                    policy_loss = -torch.min(
                        unclipped_objective,
                        clipped_objective,
                    ).mean()

                    clipped_values = (
                        old_values
                        + torch.clamp(
                            new_values - old_values,
                            -config.value_clip_epsilon,
                            config.value_clip_epsilon,
                        )
                    )

                    value_loss_unclipped = (
                        new_values - returns
                    ).square()

                    value_loss_clipped = (
                        clipped_values - returns
                    ).square()

                    value_loss = (
                        0.5
                        * torch.max(
                            value_loss_unclipped,
                            value_loss_clipped,
                        ).mean()
                    )

                    entropy_mean = entropy.mean()

                    total_loss = (
                        policy_loss
                        + config.value_loss_coefficient
                        * value_loss
                        - config.entropy_coefficient
                        * entropy_mean
                    )

                    if not torch.isfinite(total_loss):
                        raise PPOError(
                            "Non-finite PPO training loss"
                        )

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    total_loss.backward()

                    gradient_norm = (
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config.max_gradient_norm,
                        )
                    )

                    if not torch.isfinite(gradient_norm):
                        raise PPOError(
                            "Non-finite gradient norm"
                        )

                    optimizer.step()

                    with torch.no_grad():
                        approximate_kl = (
                            (
                                ratio - 1.0
                            )
                            - log_ratio
                        ).mean()

                        clip_fraction = (
                            (
                                torch.abs(
                                    ratio - 1.0
                                )
                                > config.clip_epsilon
                            )
                            .float()
                            .mean()
                        )

                    policy_losses.append(
                        float(policy_loss.item())
                    )

                    value_losses.append(
                        float(value_loss.item())
                    )

                    entropy_values.append(
                        float(entropy_mean.item())
                    )

                    total_losses.append(
                        float(total_loss.item())
                    )

                    kl_value = float(
                        approximate_kl.item()
                    )

                    kl_values.append(kl_value)
                    epoch_kls.append(kl_value)

                    clip_fractions.append(
                        float(clip_fraction.item())
                    )

                    gradient_norms.append(
                        float(gradient_norm.item())
                    )

                completed_epochs += 1

                if (
                    epoch_kls
                    and np.mean(epoch_kls)
                    > config.target_kl
                ):
                    early_stopped = True
                    break

            model.eval()

            with torch.no_grad():
                predicted_values = model.value(
                    tensors["observations"]
                )

                targets = tensors["returns"]

                target_variance = torch.var(
                    targets,
                    unbiased=False,
                )

                if target_variance.item() > 1e-8:
                    explained_variance = (
                        1.0
                        - torch.var(
                            targets - predicted_values,
                            unbiased=False,
                        )
                        / target_variance
                    ).item()

                else:
                    explained_variance = 0.0

            metrics = PPOTrainingMetrics(
                update_number=session.update_count + 1,
                policy_loss=float(
                    np.mean(policy_losses)
                ),
                value_loss=float(
                    np.mean(value_losses)
                ),
                entropy=float(
                    np.mean(entropy_values)
                ),
                total_loss=float(
                    np.mean(total_losses)
                ),
                approximate_kl=float(
                    np.mean(kl_values)
                ),
                clip_fraction=float(
                    np.mean(clip_fractions)
                ),
                explained_variance=float(
                    explained_variance
                ),
                gradient_norm=float(
                    np.mean(gradient_norms)
                ),
                learning_rate=float(
                    optimizer.param_groups[0]["lr"]
                ),
                samples_processed=sample_count,
                optimization_epochs=completed_epochs,
                early_stopped=early_stopped,
            )

            session.update_count += 1
            session.policy_version += 1

            session.metrics_history.append(
                metrics
            )

            session.status = PPOTrainingStatus.TRAINED
            session.updated_at = utc_now()

            session.evaluation_reference = None
            session.reviewer_id = None

            buffer.clear()

            self._log(
                tenant_id=tenant_id,
                action="ppo_policy_updated",
                entity_id=session_id,
                actor_id=actor_id,
                details={
                    "update_number": session.update_count,
                    "policy_version": session.policy_version,
                    "policy_loss": metrics.policy_loss,
                    "value_loss": metrics.value_loss,
                    "approximate_kl": (
                        metrics.approximate_kl
                    ),
                },
            )

            return metrics.to_dict()

    # --------------------------------------------------------
    # LEARNING RATE SCHEDULING
    # --------------------------------------------------------

    def set_learning_rate(
        self,
        *,
        tenant_id: str,
        session_id: str,
        learning_rate: float,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PPOPermissionDenied(
                "Authorized learning-rate update required"
            )

        value = finite_float(
            learning_rate,
            "learning_rate",
        )

        if value <= 0:
            raise PPOValidationError(
                "Learning rate must be positive"
            )

        with self._lock:
            session = self._get_session(
                tenant_id,
                session_id,
            )

            if session.status in (
                PPOTrainingStatus.ARCHIVED,
                PPOTrainingStatus.APPROVED_FOR_SIMULATION,
            ):
                raise PPOConflict(
                    "Cannot modify approved or archived session"
                )

            optimizer = self._optimizers[
                (tenant_id, session_id)
            ]

            for group in optimizer.param_groups:
                group["lr"] = value

            return {
                "session_id": session_id,
                "learning_rate": value,
            }

    # --------------------------------------------------------
    # MARK EVALUATED
    # --------------------------------------------------------

    def mark_evaluated(
        self,
        *,
        tenant_id: str,
        session_id: str,
        evaluation_id: str,
        passed: bool,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PPOPermissionDenied(
                "Authorized evaluation update required"
            )

        validate_id(
            evaluation_id,
            "evaluation_id",
        )

        with self._lock:
            session = self._get_session(
                tenant_id,
                session_id,
            )

            if session.status != PPOTrainingStatus.TRAINED:
                raise PPOConflict(
                    "Session must be trained before evaluation"
                )

            if passed is not True:
                return {
                    "session_id": session_id,
                    "evaluation_passed": False,
                    "status": session.status.value,
                }

            session.status = PPOTrainingStatus.EVALUATED

            session.evaluation_reference = evaluation_id
            session.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                action="ppo_policy_evaluated",
                entity_id=session_id,
                details={
                    "evaluation_id": evaluation_id,
                },
            )

            return session.to_dict()

    # --------------------------------------------------------
    # APPROVE FOR SIMULATION
    # --------------------------------------------------------

    def approve_for_simulation(
        self,
        *,
        tenant_id: str,
        session_id: str,
        reviewer_id: str,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PPOPermissionDenied(
                "Human approval required"
            )

        validate_id(
            reviewer_id,
            "reviewer_id",
        )

        with self._lock:
            session = self._get_session(
                tenant_id,
                session_id,
            )

            if session.status != PPOTrainingStatus.EVALUATED:
                raise PPOConflict(
                    "Session must pass evaluation first"
                )

            session.status = (
                PPOTrainingStatus.APPROVED_FOR_SIMULATION
            )

            session.reviewer_id = reviewer_id
            session.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                action="ppo_approved_for_simulation",
                entity_id=session_id,
                actor_id=reviewer_id,
            )

            return session.to_dict()

    # --------------------------------------------------------
    # TRAINING SUMMARY
    # --------------------------------------------------------

    def training_summary(
        self,
        *,
        tenant_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._get_session(
                tenant_id,
                session_id,
            )

            model = self._models[
                (tenant_id, session_id)
            ]

            parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
            )

            buffer = self._buffers[
                (tenant_id, session_id)
            ]

            latest_metrics = (
                session.metrics_history[-1].to_dict()
                if session.metrics_history
                else None
            )

            return {
                "session_id": session_id,
                "tenant_id": tenant_id,
                "policy_id": session.policy_id,
                "status": session.status.value,
                "action_type": (
                    session.config.action_type.value
                ),
                "total_parameters": parameter_count,
                "policy_version": session.policy_version,
                "update_count": session.update_count,
                "total_environment_steps": (
                    session.total_environment_steps
                ),
                "rollout_buffer_size": len(buffer),
                "latest_metrics": latest_metrics,
                "simulation_only": True,
                "physical_deployment_approved": False,
            }

    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    def save_checkpoint(
        self,
        *,
        tenant_id: str,
        session_id: str,
        directory: str | Path,
        authorized: bool = False,
    ) -> Path:
        if authorized is not True:
            raise PPOPermissionDenied(
                "Authorized checkpoint export required"
            )

        with self._lock:
            session = self._get_session(
                tenant_id,
                session_id,
            )

            key = (tenant_id, session_id)

            output_directory = Path(directory)

            output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            output = output_directory / (
                f"{tenant_id}_{session_id}_ppo.pt"
            )

            checkpoint = {
                "format_version": 1,
                "tenant_id": tenant_id,
                "session_id": session_id,
                "robot_id": session.robot_id,
                "policy_id": session.policy_id,
                "config": json_safe(
                    asdict(session.config)
                ),
                "model_state_dict": (
                    self._models[key].state_dict()
                ),
                "optimizer_state_dict": (
                    self._optimizers[key].state_dict()
                ),
                "policy_version": session.policy_version,
                "update_count": session.update_count,
                "total_environment_steps": (
                    session.total_environment_steps
                ),
                "metrics_history": [
                    item.to_dict()
                    for item in session.metrics_history
                ],
                "simulation_only": True,
            }

            with tempfile.NamedTemporaryFile(
                dir=output_directory,
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(
                    temporary.name
                )

            try:
                torch.save(
                    checkpoint,
                    temporary_path,
                )

                os.replace(
                    temporary_path,
                    output,
                )

            finally:
                temporary_path.unlink(
                    missing_ok=True
                )

            self._log(
                tenant_id=tenant_id,
                action="ppo_checkpoint_saved",
                entity_id=session_id,
                details={
                    "path": str(output),
                },
            )

            return output

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    def load_checkpoint(
        self,
        *,
        tenant_id: str,
        session_id: str,
        checkpoint_path: str | Path,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PPOPermissionDenied(
                "Authorized checkpoint loading required"
            )

        path = Path(checkpoint_path)

        if not path.is_file():
            raise PPONotFound(
                "Checkpoint not found"
            )

        with self._lock:
            session = self._get_session(
                tenant_id,
                session_id,
            )

            if session.status in (
                PPOTrainingStatus.ARCHIVED,
                PPOTrainingStatus.APPROVED_FOR_SIMULATION,
            ):
                raise PPOConflict(
                    "Cannot overwrite approved or archived session"
                )

            key = (tenant_id, session_id)

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            if (
                checkpoint.get("tenant_id") != tenant_id
                or checkpoint.get("session_id") != session_id
                or checkpoint.get("robot_id")
                != session.robot_id
                or checkpoint.get("policy_id")
                != session.policy_id
            ):
                raise PPOPermissionDenied(
                    "Checkpoint identity mismatch"
                )

            saved_config = checkpoint.get("config", {})

            current_config = json_safe(
                asdict(session.config)
            )

            for field_name in (
                "observation_dimension",
                "action_dimension",
                "action_type",
                "hidden_dimensions",
                "action_low",
                "action_high",
            ):
                if saved_config.get(field_name) != (
                    current_config.get(field_name)
                ):
                    raise PPOValidationError(
                        f"Checkpoint mismatch: {field_name}"
                    )

            candidate = copy.deepcopy(
                self._models[key]
            )

            candidate.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )

            for parameter in candidate.parameters():
                if not torch.isfinite(parameter).all():
                    raise PPOValidationError(
                        "Checkpoint contains non-finite weights"
                    )

            optimizer = torch.optim.Adam(
                candidate.parameters(),
                lr=session.config.learning_rate,
                eps=1e-5,
            )

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

            candidate = candidate.to(
                session.config.device
            )

            candidate.eval()

            self._models[key] = candidate
            self._optimizers[key] = optimizer

            session.policy_version = int(
                checkpoint["policy_version"]
            )

            session.update_count = int(
                checkpoint["update_count"]
            )

            session.total_environment_steps = int(
                checkpoint["total_environment_steps"]
            )

            session.metrics_history = [
                PPOTrainingMetrics(
                    **{
                        **item,
                        "created_at": datetime.fromisoformat(
                            item["created_at"]
                        ),
                    }
                )
                for item in checkpoint.get(
                    "metrics_history",
                    []
                )
            ]

            session.status = PPOTrainingStatus.TRAINED
            session.evaluation_reference = None
            session.reviewer_id = None
            session.updated_at = utc_now()

            self._buffers[key].clear()

            self._log(
                tenant_id=tenant_id,
                action="ppo_checkpoint_loaded",
                entity_id=session_id,
                details={
                    "path": str(path),
                },
            )

            return session.to_dict()

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

        positive_int(limit, "limit")

        with self._lock:
            return [
                copy.deepcopy(event)
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
        validate_id(tenant_id, "tenant_id")

        output = Path(path)

        with self._lock:
            sessions = [
                session.to_dict()
                for (tenant, _), session
                in self._sessions.items()
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
            "training_sessions": sessions,
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
# SYNTHETIC SIMULATION ENVIRONMENT
# ============================================================

class SyntheticNavigationEnvironment:
    """
    Simple 2D navigation environment for PPO testing.

    Observation:
        [robot_x, robot_y, goal_x, goal_y]

    Actions:
        0 = up
        1 = down
        2 = left
        3 = right

    This environment is synthetic and does not model
    real robot dynamics or safety constraints.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        maximum_steps: int = 100,
    ) -> None:
        self.rng = np.random.default_rng(seed)

        self.maximum_steps = maximum_steps

        self.position = np.zeros(
            2,
            dtype=np.float32,
        )

        self.goal = np.zeros(
            2,
            dtype=np.float32,
        )

        self.step_count = 0

        self.action_deltas = np.asarray(
            [
                [0.0, 1.0],
                [0.0, -1.0],
                [-1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self.position,
                self.goal,
            ]
        ).astype(np.float32)

    def reset(
        self,
    ) -> tuple[
        np.ndarray,
        dict[str, Any],
    ]:
        self.position = self.rng.uniform(
            -5.0,
            5.0,
            size=2,
        ).astype(np.float32)

        self.goal = self.rng.uniform(
            -5.0,
            5.0,
            size=2,
        ).astype(np.float32)

        self.step_count = 0

        return self._observation(), {}

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        if (
            isinstance(action, bool)
            or not isinstance(action, int)
            or not 0 <= action < 4
        ):
            raise PPOValidationError(
                "Invalid navigation action"
            )

        before_distance = float(
            np.linalg.norm(
                self.goal - self.position
            )
        )

        self.position += (
            self.action_deltas[action] * 0.5
        )

        self.step_count += 1

        after_distance = float(
            np.linalg.norm(
                self.goal - self.position
            )
        )

        reward = (
            before_distance
            - after_distance
            - 0.01
        )

        terminated = (
            after_distance < 0.5
        )

        truncated = (
            self.step_count >= self.maximum_steps
            and not terminated
        )

        if terminated:
            reward += 10.0

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            {
                "distance_to_goal": after_distance,
                "success": terminated,
                "synthetic": True,
            },
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    config = PPOConfig(
        observation_dimension=4,
        action_dimension=4,
        action_type=PPOActionType.DISCRETE,
        hidden_dimensions=(128, 128),
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        update_epochs=5,
        minibatch_size=64,
        rollout_steps=512,
        target_kl=0.03,
        device="cpu",
    )

    trainer = RoboticsPPOTrainer()

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"
    policy_id = "warehouse_navigation_ppo_v1"
    session_id = "ppo_training_session_001"

    print("\nCREATE PPO TRAINING SESSION")

    session = trainer.create_session(
        tenant_id=tenant_id,
        robot_id=robot_id,
        policy_id=policy_id,
        session_id=session_id,
        name="Warehouse Navigation PPO Training",
        config=config,
        authorized=True,
        actor_id="training_admin",
    )

    print(
        json.dumps(
            {
                "session_id": session["session_id"],
                "status": session["status"],
            },
            indent=2,
        )
    )

    environments = {
        "simulation_env_01": (
            SyntheticNavigationEnvironment(seed=42)
        ),
        "simulation_env_02": (
            SyntheticNavigationEnvironment(seed=43)
        ),
        "simulation_env_03": (
            SyntheticNavigationEnvironment(seed=44)
        ),
        "simulation_env_04": (
            SyntheticNavigationEnvironment(seed=45)
        ),
    }

    for update in range(3):
        print(
            f"\nCOLLECT ROLLOUT {update + 1}"
        )

        collection = trainer.collect_rollout(
            tenant_id=tenant_id,
            session_id=session_id,
            environments=environments,
            authorized=True,
            actor_id="training_admin",
        )

        print(
            json.dumps(
                collection,
                indent=2,
            )
        )

        print(
            f"\nPPO UPDATE {update + 1}"
        )

        metrics = trainer.update_policy(
            tenant_id=tenant_id,
            session_id=session_id,
            authorized=True,
            actor_id="training_admin",
        )

        print(
            json.dumps(
                metrics,
                indent=2,
            )
        )

    print("\nSELECT SIMULATION ACTION")

    prediction = trainer.select_action(
        tenant_id=tenant_id,
        session_id=session_id,
        observation=(
            1.0,
            1.0,
            4.0,
            4.0,
        ),
        deterministic=True,
    )

    print(
        json.dumps(
            prediction,
            indent=2,
        )
    )

    print("\nTRAINING SUMMARY")

    print(
        json.dumps(
            trainer.training_summary(
                tenant_id=tenant_id,
                session_id=session_id,
            ),
            indent=2,
        )
    )

    print("\nSAVE CHECKPOINT")

    checkpoint = trainer.save_checkpoint(
        tenant_id=tenant_id,
        session_id=session_id,
        directory="data/robotics/checkpoints",
        authorized=True,
    )

    print(checkpoint)

    print("\nEXPORT REPORT")

    output = trainer.export_json(
        "data/robotics/ppo_training_report.json",
        tenant_id=tenant_id,
    )

    print(output)


if __name__ == "__main__":
    main()
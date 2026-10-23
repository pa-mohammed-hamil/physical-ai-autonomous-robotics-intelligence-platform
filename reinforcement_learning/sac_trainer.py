"""Sac Trainer module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/sac_trainer.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Soft Actor-Critic (SAC) Trainer

Python 3.11+
Dependencies:
    pip install numpy torch

Features:
    - Continuous-action Soft Actor-Critic
    - Squashed Gaussian stochastic actor
    - Twin Q-value critics
    - Target critic networks
    - Entropy temperature auto-tuning
    - Reparameterization trick
    - Replay buffer with uniform sampling
    - Bellman target calculation
    - Polyak target updates
    - Observation validation
    - Action scaling and clipping
    - Gradient clipping
    - Training metrics
    - Policy evaluation
    - Checkpoint save/load
    - Reproducible random seeds
    - Simulation-only execution

Notes:
    SAC is an off-policy reinforcement learning
    algorithm for continuous action spaces.

    PPO is on-policy, while SAC can reuse transitions
    from a replay buffer.

Safety:
    Training and inference are simulation-only.
    This module does not interface with actuators,
    ROS2 command topics, or physical robots.
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
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F


# ============================================================
# EXCEPTIONS
# ============================================================

class SACError(Exception):
    pass


class SACValidationError(SACError):
    pass


class SACTrainingError(SACError):
    pass


class SACCheckpointError(SACError):
    pass


class SACPermissionDenied(SACError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def validate_identifier(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            char.isalnum() or char in "._-"
            for char in value
        )
    ):
        raise SACValidationError(
            f"Invalid {name}"
        )

    return value


def numeric_array(
    value: Any,
    dimension: int,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(
            value,
            dtype=np.float32,
        )
    except (TypeError, ValueError) as exc:
        raise SACValidationError(
            f"Invalid {name}"
        ) from exc

    if array.shape != (dimension,):
        raise SACValidationError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.isfinite(array).all():
        raise SACValidationError(
            f"{name} contains non-finite values"
        )

    return array


def finite_float(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, (bool, str))
        or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        )
    ):
        raise SACValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise SACValidationError(
            f"{name} must be finite"
        )

    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

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

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mlp(
    input_dimension: int,
    hidden_dimensions: Sequence[int],
    output_dimension: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []

    previous = input_dimension

    for dimension in hidden_dimensions:
        layers.extend(
            [
                nn.Linear(previous, dimension),
                nn.ReLU(),
            ]
        )

        previous = dimension

    layers.append(
        nn.Linear(previous, output_dimension)
    )

    return nn.Sequential(*layers)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SACConfig:
    observation_dimension: int
    action_dimension: int

    action_low: tuple[float, ...]
    action_high: tuple[float, ...]

    hidden_dimensions: tuple[int, ...] = (
        256,
        256,
    )

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.005

    batch_size: int = 256
    replay_capacity: int = 100_000

    learning_starts: int = 1_000
    gradient_steps: int = 1

    automatic_entropy_tuning: bool = True

    initial_alpha: float = 0.2
    target_entropy: float | None = None

    log_std_min: float = -20.0
    log_std_max: float = 2.0

    gradient_clip_norm: float = 10.0

    random_seed: int = 42
    device: str = "auto"

    bootstrap_on_truncation: bool = True

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "observation_dimension",
            "action_dimension",
            "batch_size",
            "replay_capacity",
            "learning_starts",
            "gradient_steps",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise SACValidationError(
                    f"{name} must be a positive integer"
                )

        if (
            not self.hidden_dimensions
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in self.hidden_dimensions
            )
        ):
            raise SACValidationError(
                "Invalid hidden dimensions"
            )

        low = numeric_array(
            self.action_low,
            self.action_dimension,
            "action_low",
        )

        high = numeric_array(
            self.action_high,
            self.action_dimension,
            "action_high",
        )

        if np.any(low >= high):
            raise SACValidationError(
                "Every action_low must be below action_high"
            )

        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "alpha_learning_rate",
            "initial_alpha",
            "gradient_clip_norm",
        ):
            if finite_float(
                getattr(self, name),
                name,
            ) <= 0:
                raise SACValidationError(
                    f"{name} must be positive"
                )

        gamma = finite_float(
            self.gamma,
            "gamma",
        )

        tau = finite_float(
            self.tau,
            "tau",
        )

        if not 0 <= gamma <= 1:
            raise SACValidationError(
                "gamma must be in [0, 1]"
            )

        if not 0 < tau <= 1:
            raise SACValidationError(
                "tau must be in (0, 1]"
            )

        if self.log_std_min >= self.log_std_max:
            raise SACValidationError(
                "Invalid log standard deviation bounds"
            )

        if self.target_entropy is not None:
            finite_float(
                self.target_entropy,
                "target_entropy",
            )

        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise SACValidationError(
                "Invalid random seed"
            )

        if self.device not in (
            "auto",
            "cpu",
            "cuda",
            "mps",
        ):
            raise SACValidationError(
                "Unsupported device"
            )

        if self.simulation_only is not True:
            raise SACPermissionDenied(
                "Physical robot control is unsupported"
            )


# ============================================================
# REPLAY TRANSITION
# ============================================================

@dataclass(frozen=True)
class SACTransition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray

    terminated: bool
    truncated: bool

    transition_id: str = field(
        default_factory=new_id
    )

    episode_id: str = "simulation_episode"

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "episode_id": self.episode_id,
            "observation": self.observation.tolist(),
            "action": self.action.tolist(),
            "reward": self.reward,
            "next_observation": (
                self.next_observation.tolist()
            ),
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


# ============================================================
# REPLAY BUFFER
# ============================================================

class SACReplayBuffer:

    def __init__(
        self,
        config: SACConfig,
    ) -> None:
        self.config = config

        self.capacity = config.replay_capacity

        self._observations = np.zeros(
            (
                self.capacity,
                config.observation_dimension,
            ),
            dtype=np.float32,
        )

        self._next_observations = np.zeros_like(
            self._observations
        )

        self._actions = np.zeros(
            (
                self.capacity,
                config.action_dimension,
            ),
            dtype=np.float32,
        )

        self._rewards = np.zeros(
            (self.capacity, 1),
            dtype=np.float32,
        )

        self._terminated = np.zeros(
            (self.capacity, 1),
            dtype=np.float32,
        )

        self._truncated = np.zeros(
            (self.capacity, 1),
            dtype=np.float32,
        )

        self._position = 0
        self._size = 0

        self._rng = np.random.default_rng(
            config.random_seed
        )

        self._lock = threading.RLock()

    def add(
        self,
        *,
        observation: Sequence[float],
        action: Sequence[float],
        reward: float,
        next_observation: Sequence[float],
        terminated: bool,
        truncated: bool,
    ) -> None:
        obs = numeric_array(
            observation,
            self.config.observation_dimension,
            "observation",
        )

        next_obs = numeric_array(
            next_observation,
            self.config.observation_dimension,
            "next_observation",
        )

        act = numeric_array(
            action,
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

        if np.any(act < low) or np.any(act > high):
            raise SACValidationError(
                "Action outside configured bounds"
            )

        reward_value = finite_float(
            reward,
            "reward",
        )

        if not isinstance(terminated, bool):
            raise SACValidationError(
                "terminated must be boolean"
            )

        if not isinstance(truncated, bool):
            raise SACValidationError(
                "truncated must be boolean"
            )

        if terminated and truncated:
            raise SACValidationError(
                "Transition cannot be both "
                "terminated and truncated"
            )

        with self._lock:
            index = self._position

            self._observations[index] = obs
            self._actions[index] = act
            self._rewards[index, 0] = reward_value

            self._next_observations[index] = next_obs

            self._terminated[index, 0] = float(
                terminated
            )

            self._truncated[index, 0] = float(
                truncated
            )

            self._position = (
                self._position + 1
            ) % self.capacity

            self._size = min(
                self._size + 1,
                self.capacity,
            )

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, Tensor]:
        with self._lock:
            if self._size < batch_size:
                raise SACTrainingError(
                    "Insufficient replay transitions"
                )

            indices = self._rng.choice(
                self._size,
                size=batch_size,
                replace=False,
            )

            arrays = {
                "observations": (
                    self._observations[indices].copy()
                ),
                "actions": (
                    self._actions[indices].copy()
                ),
                "rewards": (
                    self._rewards[indices].copy()
                ),
                "next_observations": (
                    self._next_observations[indices].copy()
                ),
                "terminated": (
                    self._terminated[indices].copy()
                ),
                "truncated": (
                    self._truncated[indices].copy()
                ),
            }

        return {
            key: torch.as_tensor(
                value,
                dtype=torch.float32,
                device=device,
            )
            for key, value in arrays.items()
        }

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "observations": (
                    self._observations.copy()
                ),
                "actions": self._actions.copy(),
                "rewards": self._rewards.copy(),
                "next_observations": (
                    self._next_observations.copy()
                ),
                "terminated": self._terminated.copy(),
                "truncated": self._truncated.copy(),
                "position": self._position,
                "size": self._size,
                "rng_state": copy.deepcopy(
                    self._rng.bit_generator.state
                ),
            }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        with self._lock:
            expected_shapes = {
                "observations": (
                    self._observations.shape
                ),
                "actions": self._actions.shape,
                "rewards": self._rewards.shape,
                "next_observations": (
                    self._next_observations.shape
                ),
                "terminated": self._terminated.shape,
                "truncated": self._truncated.shape,
            }

            restored = {}

            for key, shape in expected_shapes.items():
                array = np.asarray(
                    state[key],
                    dtype=np.float32,
                )

                if (
                    array.shape != shape
                    or not np.isfinite(array).all()
                ):
                    raise SACCheckpointError(
                        f"Invalid replay checkpoint: {key}"
                    )

                restored[key] = array.copy()

            size = int(state["size"])
            position = int(state["position"])

            if not 0 <= size <= self.capacity:
                raise SACCheckpointError(
                    "Invalid replay size"
                )

            if not 0 <= position < self.capacity:
                raise SACCheckpointError(
                    "Invalid replay position"
                )

            rng = np.random.default_rng()
            rng.bit_generator.state = copy.deepcopy(
                state["rng_state"]
            )

            self._observations = restored[
                "observations"
            ]

            self._actions = restored["actions"]
            self._rewards = restored["rewards"]

            self._next_observations = restored[
                "next_observations"
            ]

            self._terminated = restored["terminated"]
            self._truncated = restored["truncated"]

            self._size = size
            self._position = position
            self._rng = rng

    def __len__(self) -> int:
        with self._lock:
            return self._size


# ============================================================
# SQUASHED GAUSSIAN ACTOR
# ============================================================

class SquashedGaussianActor(nn.Module):

    def __init__(
        self,
        config: SACConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.backbone = build_mlp(
            config.observation_dimension,
            config.hidden_dimensions,
            config.hidden_dimensions[-1],
        )

        self.mean_layer = nn.Linear(
            config.hidden_dimensions[-1],
            config.action_dimension,
        )

        self.log_std_layer = nn.Linear(
            config.hidden_dimensions[-1],
            config.action_dimension,
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

    def forward(
        self,
        observations: Tensor,
    ) -> tuple[Tensor, Tensor]:
        features = self.backbone(
            observations
        )

        mean = self.mean_layer(
            features
        )

        log_std = self.log_std_layer(
            features
        )

        log_std = torch.clamp(
            log_std,
            self.config.log_std_min,
            self.config.log_std_max,
        )

        return mean, log_std

    def sample(
        self,
        observations: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self.forward(
            observations
        )

        std = log_std.exp()

        distribution = Normal(
            mean,
            std,
        )

        pre_tanh = distribution.rsample()

        squashed = torch.tanh(
            pre_tanh
        )

        action = (
            squashed * self.action_scale
            + self.action_bias
        )

        # Numerically stable tanh log-Jacobian.
        log_det_tanh = (
            2.0
            * (
                math.log(2.0)
                - pre_tanh
                - F.softplus(-2.0 * pre_tanh)
            )
        )

        log_probability = (
            distribution.log_prob(pre_tanh)
            - log_det_tanh
            - torch.log(self.action_scale)
        ).sum(
            dim=-1,
            keepdim=True,
        )

        deterministic_action = (
            torch.tanh(mean) * self.action_scale
            + self.action_bias
        )

        return (
            action,
            log_probability,
            deterministic_action,
        )


# ============================================================
# Q NETWORK
# ============================================================

class QNetwork(nn.Module):

    def __init__(
        self,
        config: SACConfig,
    ) -> None:
        super().__init__()

        self.network = build_mlp(
            (
                config.observation_dimension
                + config.action_dimension
            ),
            config.hidden_dimensions,
            1,
        )

    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
    ) -> Tensor:
        combined = torch.cat(
            [observations, actions],
            dim=-1,
        )

        return self.network(
            combined
        )


# ============================================================
# TWIN CRITIC
# ============================================================

class TwinQCritic(nn.Module):

    def __init__(
        self,
        config: SACConfig,
    ) -> None:
        super().__init__()

        self.q1 = QNetwork(config)
        self.q2 = QNetwork(config)

    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return (
            self.q1(observations, actions),
            self.q2(observations, actions),
        )


# ============================================================
# TRAINING METRICS
# ============================================================

@dataclass(frozen=True)
class SACTrainingMetrics:
    update_step: int

    critic_loss: float
    actor_loss: float
    alpha_loss: float

    alpha: float
    mean_q1: float
    mean_q2: float

    mean_log_probability: float
    target_q_mean: float

    replay_size: int

    timestamp: str = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# SIMULATION ENVIRONMENT PROTOCOL
# ============================================================

class SimulationEnvironment(Protocol):
    """
    Expected simulation interface:

    reset(seed=None) -> observation, info

    step(action) ->
        observation,
        reward,
        terminated,
        truncated,
        info
    """

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ...

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        ...


# ============================================================
# SAC TRAINER
# ============================================================

class RoboticsSACTrainer:

    def __init__(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        policy_id: str,
        config: SACConfig,
    ) -> None:
        self.tenant_id = validate_identifier(
            tenant_id,
            "tenant_id",
        )

        self.robot_id = validate_identifier(
            robot_id,
            "robot_id",
        )

        self.policy_id = validate_identifier(
            policy_id,
            "policy_id",
        )

        self.config = config

        set_random_seeds(
            config.random_seed
        )

        if config.device == "auto":
            device_name = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            device_name = config.device

        self.device = torch.device(
            device_name
        )

        self.actor = SquashedGaussianActor(
            config
        ).to(self.device)

        self.critic = TwinQCritic(
            config
        ).to(self.device)

        self.target_critic = copy.deepcopy(
            self.critic
        ).to(self.device)

        self.target_critic.eval()

        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_learning_rate,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_learning_rate,
        )

        self.log_alpha = torch.tensor(
            math.log(config.initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=(
                config.automatic_entropy_tuning
            ),
        )

        self.alpha_optimizer = (
            torch.optim.Adam(
                [self.log_alpha],
                lr=config.alpha_learning_rate,
            )
            if config.automatic_entropy_tuning
            else None
        )

        self.target_entropy = (
            config.target_entropy
            if config.target_entropy is not None
            else -float(config.action_dimension)
        )

        self.replay_buffer = SACReplayBuffer(
            config
        )

        self.environment_steps = 0
        self.update_steps = 0

        self.training_history: deque[
            SACTrainingMetrics
        ] = deque(maxlen=10_000)

        self.audit_events: deque[
            dict[str, Any]
        ] = deque(maxlen=20_000)

        self._lock = threading.RLock()

        self.created_at = utc_now()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        action: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.audit_events.append(
            {
                "event_id": new_id(),
                "timestamp": utc_now(),
                "tenant_id": self.tenant_id,
                "robot_id": self.robot_id,
                "policy_id": self.policy_id,
                "action": action,
                "details": json_safe(
                    dict(details or {})
                ),
            }
        )

    # --------------------------------------------------------
    # ALPHA
    # --------------------------------------------------------

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    # --------------------------------------------------------
    # SELECT ACTION
    # --------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        observation: Sequence[float],
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        obs = numeric_array(
            observation,
            self.config.observation_dimension,
            "observation",
        )

        tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action, _, mean_action = (
            self.actor.sample(tensor)
        )

        selected = (
            mean_action
            if deterministic
            else action
        )

        return selected.squeeze(0).cpu().numpy().copy()

    # --------------------------------------------------------
    # RANDOM EXPLORATION ACTION
    # --------------------------------------------------------

    def random_action(
        self,
    ) -> np.ndarray:
        low = np.asarray(
            self.config.action_low,
            dtype=np.float32,
        )

        high = np.asarray(
            self.config.action_high,
            dtype=np.float32,
        )

        return self.replay_buffer._rng.uniform(
            low,
            high,
        ).astype(np.float32)

    # --------------------------------------------------------
    # STORE TRANSITION
    # --------------------------------------------------------

    def store_transition(
        self,
        *,
        observation: Sequence[float],
        action: Sequence[float],
        reward: float,
        next_observation: Sequence[float],
        terminated: bool,
        truncated: bool,
    ) -> None:
        with self._lock:
            self.replay_buffer.add(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
            )

            self.environment_steps += 1

    # --------------------------------------------------------
    # SOFT UPDATE
    # --------------------------------------------------------

    @torch.no_grad()
    def _soft_update_targets(self) -> None:
        tau = self.config.tau

        for target, source in zip(
            self.target_critic.parameters(),
            self.critic.parameters(),
        ):
            target.mul_(1.0 - tau)
            target.add_(
                source,
                alpha=tau,
            )

    # --------------------------------------------------------
    # COMPUTE CRITIC TARGET
    # --------------------------------------------------------

    @torch.no_grad()
    def _critic_target(
        self,
        batch: Mapping[str, Tensor],
    ) -> Tensor:
        next_actions, next_log_probs, _ = (
            self.actor.sample(
                batch["next_observations"]
            )
        )

        target_q1, target_q2 = (
            self.target_critic(
                batch["next_observations"],
                next_actions,
            )
        )

        min_target_q = torch.minimum(
            target_q1,
            target_q2,
        )

        soft_value = (
            min_target_q
            - self.alpha.detach()
            * next_log_probs
        )

        if self.config.bootstrap_on_truncation:
            done = batch["terminated"]
        else:
            done = torch.maximum(
                batch["terminated"],
                batch["truncated"],
            )

        target = (
            batch["rewards"]
            + self.config.gamma
            * (1.0 - done)
            * soft_value
        )

        return target

    # --------------------------------------------------------
    # TRAIN ONE GRADIENT STEP
    # --------------------------------------------------------

    def train_step(
        self,
    ) -> SACTrainingMetrics:
        with self._lock:
            minimum_samples = max(
                self.config.batch_size,
                self.config.learning_starts,
            )

            if len(self.replay_buffer) < minimum_samples:
                raise SACTrainingError(
                    "Replay buffer has insufficient samples"
                )

            batch = self.replay_buffer.sample(
                self.config.batch_size,
                self.device,
            )

            # -----------------------------------------------
            # CRITIC UPDATE
            # -----------------------------------------------

            target_q = self._critic_target(
                batch
            )

            current_q1, current_q2 = (
                self.critic(
                    batch["observations"],
                    batch["actions"],
                )
            )

            critic_loss = (
                F.mse_loss(
                    current_q1,
                    target_q,
                )
                + F.mse_loss(
                    current_q2,
                    target_q,
                )
            )

            if not torch.isfinite(critic_loss):
                raise SACTrainingError(
                    "Non-finite critic loss"
                )

            self.critic_optimizer.zero_grad(
                set_to_none=True
            )

            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.config.gradient_clip_norm,
            )

            self.critic_optimizer.step()

            # -----------------------------------------------
            # ACTOR UPDATE
            # -----------------------------------------------

            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)

            try:
                actions, log_probs, _ = (
                    self.actor.sample(
                        batch["observations"]
                    )
                )

                q1_pi, q2_pi = self.critic(
                    batch["observations"],
                    actions,
                )

                min_q_pi = torch.minimum(
                    q1_pi,
                    q2_pi,
                )

                actor_loss = (
                    self.alpha.detach() * log_probs
                    - min_q_pi
                ).mean()

                if not torch.isfinite(actor_loss):
                    raise SACTrainingError(
                        "Non-finite actor loss"
                    )

                self.actor_optimizer.zero_grad(
                    set_to_none=True
                )

                actor_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.config.gradient_clip_norm,
                )

                self.actor_optimizer.step()

            finally:
                for parameter in self.critic.parameters():
                    parameter.requires_grad_(True)

            # -----------------------------------------------
            # ENTROPY TEMPERATURE UPDATE
            # -----------------------------------------------

            alpha_loss_value = 0.0

            if self.alpha_optimizer is not None:
                alpha_loss = -(
                    self.log_alpha
                    * (
                        log_probs.detach()
                        + self.target_entropy
                    )
                ).mean()

                if not torch.isfinite(alpha_loss):
                    raise SACTrainingError(
                        "Non-finite alpha loss"
                    )

                self.alpha_optimizer.zero_grad(
                    set_to_none=True
                )

                alpha_loss.backward()

                self.alpha_optimizer.step()

                with torch.no_grad():
                    self.log_alpha.clamp_(
                        min=-20.0,
                        max=5.0,
                    )

                alpha_loss_value = float(
                    alpha_loss.detach().item()
                )

            # -----------------------------------------------
            # TARGET UPDATE
            # -----------------------------------------------

            self._soft_update_targets()

            self.update_steps += 1

            metrics = SACTrainingMetrics(
                update_step=self.update_steps,
                critic_loss=float(
                    critic_loss.detach().item()
                ),
                actor_loss=float(
                    actor_loss.detach().item()
                ),
                alpha_loss=alpha_loss_value,
                alpha=float(
                    self.alpha.detach().item()
                ),
                mean_q1=float(
                    current_q1.detach().mean().item()
                ),
                mean_q2=float(
                    current_q2.detach().mean().item()
                ),
                mean_log_probability=float(
                    log_probs.detach().mean().item()
                ),
                target_q_mean=float(
                    target_q.detach().mean().item()
                ),
                replay_size=len(self.replay_buffer),
            )

            self.training_history.append(
                metrics
            )

            return metrics

    # --------------------------------------------------------
    # TRAIN MULTIPLE STEPS
    # --------------------------------------------------------

    def train(
        self,
        gradient_steps: int | None = None,
    ) -> list[dict[str, Any]]:
        steps = (
            gradient_steps
            if gradient_steps is not None
            else self.config.gradient_steps
        )

        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < 1
        ):
            raise SACValidationError(
                "gradient_steps must be positive"
            )

        results = []

        for _ in range(steps):
            metrics = self.train_step()
            results.append(metrics.to_dict())

        return results

    # --------------------------------------------------------
    # COLLECT SIMULATION EXPERIENCE
    # --------------------------------------------------------

    def collect_episode(
        self,
        environment: SimulationEnvironment,
        *,
        episode_seed: int | None = None,
        maximum_steps: int = 1_000,
        train_during_collection: bool = True,
    ) -> dict[str, Any]:
        if maximum_steps < 1:
            raise SACValidationError(
                "maximum_steps must be positive"
            )

        observation, _ = environment.reset(
            seed=episode_seed
        )

        observation = numeric_array(
            observation,
            self.config.observation_dimension,
            "initial_observation",
        )

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        for step_index in range(maximum_steps):
            if (
                self.environment_steps
                < self.config.learning_starts
            ):
                action = self.random_action()
            else:
                action = self.select_action(
                    observation
                )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = environment.step(action)

            next_observation = numeric_array(
                next_observation,
                self.config.observation_dimension,
                "next_observation",
            )

            reward = finite_float(
                reward,
                "reward",
            )

            if (
                step_index == maximum_steps - 1
                and not terminated
                and not truncated
            ):
                truncated = True

            self.store_transition(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
            )

            observation = next_observation

            total_reward += reward
            steps += 1

            if (
                train_during_collection
                and len(self.replay_buffer)
                >= max(
                    self.config.learning_starts,
                    self.config.batch_size,
                )
            ):
                self.train(
                    self.config.gradient_steps
                )

            if terminated or truncated:
                break

        summary = {
            "episode_id": new_id(),
            "steps": steps,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "replay_size": len(self.replay_buffer),
            "environment_steps": self.environment_steps,
            "update_steps": self.update_steps,
            "simulation_only": True,
        }

        self._audit(
            "simulation_episode_collected",
            summary,
        )

        return summary

    # --------------------------------------------------------
    # EVALUATE POLICY
    # --------------------------------------------------------

    def evaluate(
        self,
        environment_factory: Callable[
            [],
            SimulationEnvironment,
        ],
        *,
        episodes: int = 5,
        maximum_steps: int = 1_000,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        if episodes < 1:
            raise SACValidationError(
                "episodes must be positive"
            )

        returns = []
        lengths = []

        terminated_count = 0

        for episode_index in range(episodes):
            environment = environment_factory()

            observation, _ = environment.reset(
                seed=(
                    self.config.random_seed
                    + 100_000
                    + episode_index
                )
            )

            total_reward = 0.0
            steps = 0

            for _ in range(maximum_steps):
                action = self.select_action(
                    observation,
                    deterministic=deterministic,
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = environment.step(action)

                total_reward += finite_float(
                    reward,
                    "evaluation_reward",
                )

                steps += 1

                if terminated:
                    terminated_count += 1

                if terminated or truncated:
                    break

            returns.append(total_reward)
            lengths.append(steps)

        return {
            "policy_id": self.policy_id,
            "episodes": episodes,
            "mean_return": float(
                np.mean(returns)
            ),
            "std_return": float(
                np.std(returns)
            ),
            "minimum_return": float(
                np.min(returns)
            ),
            "maximum_return": float(
                np.max(returns)
            ),
            "mean_episode_length": float(
                np.mean(lengths)
            ),
            "termination_rate": (
                terminated_count / episodes
            ),
            "deterministic": deterministic,
            "simulation_only": True,
            "physical_deployment_approved": False,
        }

    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        include_replay_buffer: bool = True,
        authorized: bool = False,
    ) -> Path:
        if authorized is not True:
            raise SACPermissionDenied(
                "Authorized checkpoint export required"
            )

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            checkpoint = {
                "format_version": 1,
                "tenant_id": self.tenant_id,
                "robot_id": self.robot_id,
                "policy_id": self.policy_id,
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": (
                    self.target_critic.state_dict()
                ),
                "actor_optimizer": (
                    self.actor_optimizer.state_dict()
                ),
                "critic_optimizer": (
                    self.critic_optimizer.state_dict()
                ),
                "log_alpha": (
                    self.log_alpha.detach().cpu()
                ),
                "alpha_optimizer": (
                    self.alpha_optimizer.state_dict()
                    if self.alpha_optimizer is not None
                    else None
                ),
                "environment_steps": (
                    self.environment_steps
                ),
                "update_steps": self.update_steps,
                "training_history": [
                    item.to_dict()
                    for item in self.training_history
                ],
                "replay_buffer": (
                    self.replay_buffer.state_dict()
                    if include_replay_buffer
                    else None
                ),
                "created_at": self.created_at,
                "saved_at": utc_now(),
                "simulation_only": True,
            }

            with tempfile.NamedTemporaryFile(
                dir=output.parent,
                suffix=".pt",
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

            self._audit(
                "checkpoint_saved",
                {
                    "path": str(output),
                    "include_replay_buffer": (
                        include_replay_buffer
                    ),
                },
            )

        return output

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise SACPermissionDenied(
                "Authorized checkpoint import required"
            )

        path = Path(path)

        if not path.is_file():
            raise SACCheckpointError(
                "Checkpoint not found"
            )

        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=True,
            )
        except Exception as exc:
            raise SACCheckpointError(
                "Unable to load checkpoint"
            ) from exc

        if (
            checkpoint.get("tenant_id") != self.tenant_id
            or checkpoint.get("robot_id") != self.robot_id
            or checkpoint.get("policy_id") != self.policy_id
        ):
            raise SACPermissionDenied(
                "Checkpoint identity mismatch"
            )

        if checkpoint.get("config") != asdict(
            self.config
        ):
            raise SACCheckpointError(
                "Checkpoint configuration mismatch"
            )

        with self._lock:
            self.actor.load_state_dict(
                checkpoint["actor"]
            )

            self.critic.load_state_dict(
                checkpoint["critic"]
            )

            self.target_critic.load_state_dict(
                checkpoint["target_critic"]
            )

            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer"]
            )

            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer"]
            )

            with torch.no_grad():
                self.log_alpha.copy_(
                    checkpoint["log_alpha"].to(
                        self.device
                    )
                )

            if (
                self.alpha_optimizer is not None
                and checkpoint.get(
                    "alpha_optimizer"
                ) is not None
            ):
                self.alpha_optimizer.load_state_dict(
                    checkpoint["alpha_optimizer"]
                )

            self.environment_steps = int(
                checkpoint["environment_steps"]
            )

            self.update_steps = int(
                checkpoint["update_steps"]
            )

            self.training_history.clear()

            for item in checkpoint.get(
                "training_history",
                [],
            ):
                self.training_history.append(
                    SACTrainingMetrics(**item)
                )

            if checkpoint.get("replay_buffer") is not None:
                self.replay_buffer.load_state_dict(
                    checkpoint["replay_buffer"]
                )

            self._audit(
                "checkpoint_loaded",
                {
                    "path": str(path),
                },
            )

        return self.training_summary()

    # --------------------------------------------------------
    # TRAINING SUMMARY
    # --------------------------------------------------------

    def training_summary(self) -> dict[str, Any]:
        with self._lock:
            recent = list(
                self.training_history
            )[-100:]

            return {
                "tenant_id": self.tenant_id,
                "robot_id": self.robot_id,
                "policy_id": self.policy_id,
                "algorithm": "soft_actor_critic",
                "action_space": "continuous",
                "device": str(self.device),
                "environment_steps": (
                    self.environment_steps
                ),
                "gradient_updates": (
                    self.update_steps
                ),
                "replay_size": len(
                    self.replay_buffer
                ),
                "alpha": float(
                    self.alpha.detach().item()
                ),
                "mean_recent_actor_loss": (
                    float(
                        np.mean(
                            [
                                item.actor_loss
                                for item in recent
                            ]
                        )
                    )
                    if recent
                    else None
                ),
                "mean_recent_critic_loss": (
                    float(
                        np.mean(
                            [
                                item.critic_loss
                                for item in recent
                            ]
                        )
                    )
                    if recent
                    else None
                ),
                "created_at": self.created_at,
                "simulation_only": True,
                "physical_deployment_approved": False,
            }

    # --------------------------------------------------------
    # EXPORT JSON REPORT
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now(),
                "training_summary": (
                    self.training_summary()
                ),
                "recent_training_metrics": [
                    item.to_dict()
                    for item in list(
                        self.training_history
                    )[-1000:]
                ],
                "audit_events": list(
                    self.audit_events
                ),
            }

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
# SYNTHETIC NAVIGATION ENVIRONMENT
# ============================================================

class SyntheticNavigationEnvironment:
    """
    Lightweight 2D simulation for smoke testing.

    State:
        [x, y, goal_x - x, goal_y - y]

    Action:
        [delta_x, delta_y], bounded [-1, 1]

    This is not a physics simulator and cannot
    validate real-world robot safety.
    """

    def __init__(
        self,
        *,
        maximum_steps: int = 100,
        goal: tuple[float, float] = (3.0, 3.0),
    ) -> None:
        self.maximum_steps = maximum_steps
        self.goal = np.asarray(
            goal,
            dtype=np.float32,
        )

        self.position = np.zeros(
            2,
            dtype=np.float32,
        )

        self.steps = 0

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self.position,
                self.goal - self.position,
            ]
        ).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.position = np.zeros(
            2,
            dtype=np.float32,
        )

        self.steps = 0

        return self._observation(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        action = numeric_array(
            action,
            2,
            "action",
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        previous_distance = float(
            np.linalg.norm(
                self.goal - self.position
            )
        )

        self.position = (
            self.position + action * 0.2
        )

        self.steps += 1

        current_distance = float(
            np.linalg.norm(
                self.goal - self.position
            )
        )

        reached_goal = (
            current_distance <= 0.25
        )

        reward = (
            previous_distance
            - current_distance
            - 0.01
        )

        if reached_goal:
            reward += 10.0

        terminated = reached_goal

        truncated = (
            self.steps >= self.maximum_steps
            and not terminated
        )

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            {
                "distance_to_goal": (
                    current_distance
                ),
                "goal_reached": reached_goal,
                "synthetic": True,
            },
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    config = SACConfig(
        observation_dimension=4,
        action_dimension=2,
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
        hidden_dimensions=(128, 128),
        batch_size=64,
        replay_capacity=10_000,
        learning_starts=128,
        gradient_steps=1,
        random_seed=42,
        device="cpu",
    )

    trainer = RoboticsSACTrainer(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        policy_id="navigation_sac_v1",
        config=config,
    )

    environment = SyntheticNavigationEnvironment(
        maximum_steps=60
    )

    print("\nCOLLECT SIMULATION EPISODES")

    for episode in range(8):
        result = trainer.collect_episode(
            environment,
            episode_seed=42 + episode,
            maximum_steps=60,
            train_during_collection=True,
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    print("\nTRAINING SUMMARY")

    print(
        json.dumps(
            trainer.training_summary(),
            indent=2,
        )
    )

    print("\nPOLICY EVALUATION")

    evaluation = trainer.evaluate(
        lambda: SyntheticNavigationEnvironment(
            maximum_steps=60
        ),
        episodes=3,
        maximum_steps=60,
        deterministic=True,
    )

    print(
        json.dumps(
            evaluation,
            indent=2,
        )
    )

    print("\nSAVE CHECKPOINT")

    checkpoint_path = trainer.save_checkpoint(
        "data/robotics/checkpoints/sac_navigation.pt",
        include_replay_buffer=True,
        authorized=True,
    )

    print(checkpoint_path)

    print("\nEXPORT TRAINING REPORT")

    report_path = trainer.export_json(
        "data/robotics/sac_training_report.json"
    )

    print(report_path)


if __name__ == "__main__":
    main()
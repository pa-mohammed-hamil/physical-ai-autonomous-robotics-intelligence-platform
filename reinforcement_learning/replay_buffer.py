"""Replay Buffer module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/replay_buffer.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Experience Replay Buffer Engine

Python 3.11+
Dependencies:
    pip install numpy

Features:
    - Uniform Experience Replay
    - Prioritized Experience Replay (PER)
    - Proportional priority sampling
    - Importance Sampling (IS) weights
    - Priority updates from TD errors
    - N-step discounted returns
    - Multi-environment experience storage
    - Episode boundary handling
    - Discrete and continuous actions
    - Action masks
    - Terminal and truncated transitions
    - Batch sampling
    - Replay statistics
    - Dataset validation
    - Offline RL dataset ingestion
    - NumPy checkpoint persistence
    - Multi-tenant isolation
    - Thread-safe operations
    - Audit logging
    - JSON report export

Integrations:
    policy.py
    ppo_trainer.py
    offline_rl.py
    imitation_learning.py
    policy_evaluation.py
    observation_space.py
    action_space.py
    environment.py
    curriculum.py
    domain_randomization.py

Important:
    Replay buffers are designed primarily for
    off-policy algorithms such as DQN, DDPG, TD3,
    SAC, and offline reinforcement learning.

    PPO is an on-policy algorithm. Historical replay
    samples must not be reused for ordinary PPO
    policy-gradient updates.

Safety:
    Simulation and offline learning only.
    No physical robot control is implemented.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import threading
import uuid

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# ============================================================
# EXCEPTIONS
# ============================================================

class ReplayBufferError(Exception):
    pass


class ReplayValidationError(ReplayBufferError):
    pass


class ReplayBufferNotFound(ReplayBufferError):
    pass


class ReplayBufferConflict(ReplayBufferError):
    pass


class ReplayPermissionDenied(ReplayBufferError):
    pass


class ReplayBufferEmpty(ReplayBufferError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ReplayStrategy(str, Enum):
    UNIFORM = "uniform"
    PRIORITIZED = "prioritized"


class ReplayActionType(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class ReplayBufferStatus(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    FULL = "full"
    ARCHIVED = "archived"


class ReplaySource(str, Enum):
    SIMULATION = "simulation"
    OFFLINE_DATASET = "offline_dataset"
    SYNTHETIC = "synthetic"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


def validate_id(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            character.isalnum() or character in "._-"
            for character in value
        )
    ):
        raise ReplayValidationError(
            f"Invalid {name}"
        )

    return value


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
        raise ReplayValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise ReplayValidationError(
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
        raise ReplayValidationError(
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
        raise ReplayValidationError(
            f"Invalid {name}"
        ) from exc

    if array.shape != (dimension,):
        raise ReplayValidationError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.isfinite(array).all():
        raise ReplayValidationError(
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
        raise ReplayValidationError(
            "Cannot serialize non-finite values"
        )

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ReplayBufferConfig:
    observation_dimension: int
    action_dimension: int

    capacity: int = 100_000

    strategy: ReplayStrategy = ReplayStrategy.UNIFORM

    action_type: ReplayActionType = (
        ReplayActionType.DISCRETE
    )

    gamma: float = 0.99

    n_step: int = 1

    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_beta_increment: float = 0.0001

    priority_epsilon: float = 1e-6

    random_seed: int = 42

    allow_replacement_sampling: bool = False

    maximum_audit_events: int = 20_000

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "observation_dimension",
            "action_dimension",
            "capacity",
            "n_step",
            "maximum_audit_events",
        ):
            positive_int(
                getattr(self, name),
                name,
            )

        if not isinstance(self.strategy, ReplayStrategy):
            raise ReplayValidationError(
                "Invalid replay strategy"
            )

        if not isinstance(
            self.action_type,
            ReplayActionType,
        ):
            raise ReplayValidationError(
                "Invalid action type"
            )

        gamma = finite_float(
            self.gamma,
            "gamma",
        )

        if not 0.0 <= gamma <= 1.0:
            raise ReplayValidationError(
                "gamma must be between 0 and 1"
            )

        alpha = finite_float(
            self.priority_alpha,
            "priority_alpha",
        )

        beta = finite_float(
            self.priority_beta,
            "priority_beta",
        )

        if not 0.0 <= alpha <= 1.0:
            raise ReplayValidationError(
                "priority_alpha must be in [0, 1]"
            )

        if not 0.0 <= beta <= 1.0:
            raise ReplayValidationError(
                "priority_beta must be in [0, 1]"
            )

        if finite_float(
            self.priority_beta_increment,
            "priority_beta_increment",
        ) < 0:
            raise ReplayValidationError(
                "priority_beta_increment cannot be negative"
            )

        if finite_float(
            self.priority_epsilon,
            "priority_epsilon",
        ) <= 0:
            raise ReplayValidationError(
                "priority_epsilon must be positive"
            )

        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ReplayValidationError(
                "random_seed must be nonnegative"
            )

        if self.simulation_only is not True:
            raise ReplayValidationError(
                "Physical execution is unsupported"
            )


# ============================================================
# TRANSITION
# ============================================================

@dataclass(frozen=True)
class ReplayTransition:
    transition_id: str

    environment_id: str
    episode_id: str

    observation: tuple[float, ...]

    action: int | tuple[float, ...]

    reward: float

    next_observation: tuple[float, ...]

    terminated: bool
    truncated: bool

    source: ReplaySource = ReplaySource.SIMULATION

    action_mask: tuple[bool, ...] | None = None
    next_action_mask: tuple[bool, ...] | None = None

    discount: float | None = None
    n_steps: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(
            self.transition_id,
            "transition_id",
        )

        validate_id(
            self.environment_id,
            "environment_id",
        )

        validate_id(
            self.episode_id,
            "episode_id",
        )

        finite_float(
            self.reward,
            "reward",
        )

        if not isinstance(self.source, ReplaySource):
            raise ReplayValidationError(
                "Invalid replay source"
            )

        if not isinstance(self.terminated, bool):
            raise ReplayValidationError(
                "terminated must be boolean"
            )

        if not isinstance(self.truncated, bool):
            raise ReplayValidationError(
                "truncated must be boolean"
            )

        if self.terminated and self.truncated:
            raise ReplayValidationError(
                "Transition cannot be both "
                "terminated and truncated"
            )

        positive_int(
            self.n_steps,
            "n_steps",
        )

        if self.discount is not None:
            discount = finite_float(
                self.discount,
                "discount",
            )

            if not 0.0 <= discount <= 1.0:
                raise ReplayValidationError(
                    "discount must be in [0, 1]"
                )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


# ============================================================
# SAMPLED BATCH
# ============================================================

@dataclass(frozen=True)
class ReplaySampleBatch:
    buffer_id: str

    strategy: ReplayStrategy

    indices: np.ndarray

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray

    terminated: np.ndarray
    truncated: np.ndarray

    discounts: np.ndarray
    n_steps: np.ndarray

    importance_weights: np.ndarray
    sampling_probabilities: np.ndarray

    transition_ids: tuple[str, ...]

    action_masks: np.ndarray | None = None
    next_action_masks: np.ndarray | None = None

    sampled_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


# ============================================================
# N-STEP ACCUMULATOR
# ============================================================

class NStepAccumulator:
    """
    Accumulates sequential transitions separately
    for each simulation environment.

    Terminal transitions:
        stop bootstrapping.

    Truncated transitions:
        end the episode, but permit bootstrap
        from the final next observation.

    The caller must flush unfinished episodes
    before resetting an environment or ending
    a collection run.
    """

    def __init__(
        self,
        n_step: int,
        gamma: float,
    ) -> None:
        self.n_step = positive_int(
            n_step,
            "n_step",
        )

        self.gamma = finite_float(
            gamma,
            "gamma",
        )

        self._queues: dict[
            str,
            deque[ReplayTransition],
        ] = {}

    def _aggregate(
        self,
        queue: deque[ReplayTransition],
    ) -> ReplayTransition:
        if not queue:
            raise ReplayBufferEmpty(
                "N-step queue is empty"
            )

        first = queue[0]

        total_reward = 0.0
        discount_factor = 1.0

        last = first
        steps_used = 0

        for transition in list(queue)[:self.n_step]:
            total_reward += (
                discount_factor * transition.reward
            )

            discount_factor *= self.gamma

            steps_used += 1
            last = transition

            if transition.terminated or transition.truncated:
                break

        return ReplayTransition(
            transition_id=new_id(),
            environment_id=first.environment_id,
            episode_id=first.episode_id,
            observation=first.observation,
            action=first.action,
            reward=float(total_reward),
            next_observation=last.next_observation,
            terminated=last.terminated,
            truncated=last.truncated,
            source=first.source,
            action_mask=first.action_mask,
            next_action_mask=last.next_action_mask,
            discount=float(discount_factor),
            n_steps=steps_used,
            metadata={
                **copy.deepcopy(first.metadata),
                "n_step_aggregated": True,
                "origin_transition_id": (
                    first.transition_id
                ),
            },
        )

    def add(
        self,
        transition: ReplayTransition,
    ) -> list[ReplayTransition]:
        environment_id = transition.environment_id

        queue = self._queues.setdefault(
            environment_id,
            deque(),
        )

        if queue:
            previous = queue[-1]

            if previous.episode_id != transition.episode_id:
                raise ReplayBufferConflict(
                    "Episode changed without flushing "
                    "pending n-step transitions"
                )

            if previous.terminated or previous.truncated:
                raise ReplayBufferConflict(
                    "Cannot append after episode boundary"
                )

        queue.append(transition)

        results: list[ReplayTransition] = []

        if transition.terminated or transition.truncated:
            while queue:
                results.append(
                    self._aggregate(queue)
                )

                queue.popleft()

            self._queues.pop(
                environment_id,
                None,
            )

        elif len(queue) >= self.n_step:
            results.append(
                self._aggregate(queue)
            )

            queue.popleft()

        return results

    def flush(
        self,
        environment_id: str,
    ) -> list[ReplayTransition]:
        validate_id(
            environment_id,
            "environment_id",
        )

        queue = self._queues.pop(
            environment_id,
            deque(),
        )

        results: list[ReplayTransition] = []

        while queue:
            results.append(
                self._aggregate(queue)
            )

            queue.popleft()

        return results

    def flush_all(
        self,
    ) -> list[ReplayTransition]:
        results: list[ReplayTransition] = []

        for environment_id in list(self._queues):
            results.extend(
                self.flush(environment_id)
            )

        return results

    def pending_count(self) -> int:
        return sum(
            len(queue)
            for queue in self._queues.values()
        )

    def clear(self) -> None:
        self._queues.clear()


# ============================================================
# SEGMENT TREE
# ============================================================

class SumSegmentTree:
    """
    Binary sum tree for proportional prioritized replay.

    Leaf nodes store priority ** alpha.
    """

    def __init__(
        self,
        capacity: int,
    ) -> None:
        positive_int(
            capacity,
            "capacity",
        )

        tree_capacity = 1

        while tree_capacity < capacity:
            tree_capacity *= 2

        self.capacity = capacity
        self.tree_capacity = tree_capacity

        self.tree = np.zeros(
            2 * tree_capacity,
            dtype=np.float64,
        )

    def update(
        self,
        index: int,
        value: float,
    ) -> None:
        if not 0 <= index < self.capacity:
            raise ReplayValidationError(
                "Segment tree index out of range"
            )

        value = finite_float(
            value,
            "priority",
        )

        if value < 0:
            raise ReplayValidationError(
                "Priority cannot be negative"
            )

        position = (
            index + self.tree_capacity
        )

        self.tree[position] = value

        position //= 2

        while position >= 1:
            self.tree[position] = (
                self.tree[2 * position]
                + self.tree[2 * position + 1]
            )

            position //= 2

    def total(self) -> float:
        return float(self.tree[1])

    def get(self, index: int) -> float:
        if not 0 <= index < self.capacity:
            raise ReplayValidationError(
                "Segment tree index out of range"
            )

        return float(
            self.tree[
                index + self.tree_capacity
            ]
        )

    def find_prefixsum_index(
        self,
        mass: float,
    ) -> int:
        total = self.total()

        if total <= 0:
            raise ReplayBufferEmpty(
                "Priority tree has no mass"
            )

        mass = finite_float(
            mass,
            "mass",
        )

        mass = min(
            max(mass, 0.0),
            np.nextafter(total, 0.0),
        )

        position = 1

        while position < self.tree_capacity:
            left = position * 2

            if mass < self.tree[left]:
                position = left

            else:
                mass -= self.tree[left]
                position = left + 1

        index = position - self.tree_capacity

        if index >= self.capacity:
            raise ReplayBufferError(
                "Segment tree selected invalid index"
            )

        return index


# ============================================================
# REPLAY BUFFER
# ============================================================

class ExperienceReplayBuffer:

    def __init__(
        self,
        *,
        buffer_id: str,
        tenant_id: str,
        robot_id: str,
        config: ReplayBufferConfig,
    ) -> None:
        validate_id(
            buffer_id,
            "buffer_id",
        )

        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            robot_id,
            "robot_id",
        )

        self.buffer_id = buffer_id
        self.tenant_id = tenant_id
        self.robot_id = robot_id
        self.config = config

        self.status = ReplayBufferStatus.CREATED

        self._storage: list[
            ReplayTransition | None
        ] = [None] * config.capacity

        self._priorities = np.zeros(
            config.capacity,
            dtype=np.float64,
        )

        self._tree = SumSegmentTree(
            config.capacity
        )

        self._size = 0
        self._position = 0

        self._total_added = 0
        self._total_sampled = 0

        self._max_priority = 1.0
        self._beta = config.priority_beta

        self._rng = np.random.default_rng(
            config.random_seed
        )

        self._n_step_accumulator = (
            NStepAccumulator(
                n_step=config.n_step,
                gamma=config.gamma,
            )
        )

        self._lock = threading.RLock()

        self.created_at = utc_now()
        self.updated_at = utc_now()

    # --------------------------------------------------------
    # VALIDATE TRANSITION
    # --------------------------------------------------------

    def _validate_transition(
        self,
        transition: ReplayTransition,
    ) -> None:
        if not isinstance(
            transition,
            ReplayTransition,
        ):
            raise ReplayValidationError(
                "Invalid transition"
            )

        validate_vector(
            transition.observation,
            self.config.observation_dimension,
            "observation",
        )

        validate_vector(
            transition.next_observation,
            self.config.observation_dimension,
            "next_observation",
        )

        if (
            self.config.action_type
            == ReplayActionType.DISCRETE
        ):
            if (
                isinstance(transition.action, bool)
                or not isinstance(transition.action, int)
                or not (
                    0 <= transition.action
                    < self.config.action_dimension
                )
            ):
                raise ReplayValidationError(
                    "Invalid discrete action"
                )

            for mask_name in (
                "action_mask",
                "next_action_mask",
            ):
                mask = getattr(
                    transition,
                    mask_name,
                )

                if mask is None:
                    continue

                if (
                    len(mask) != self.config.action_dimension
                    or not all(
                        isinstance(value, bool)
                        for value in mask
                    )
                    or not any(mask)
                ):
                    raise ReplayValidationError(
                        f"Invalid {mask_name}"
                    )

            if (
                transition.action_mask is not None
                and not transition.action_mask[
                    transition.action
                ]
            ):
                raise ReplayValidationError(
                    "Selected action is masked"
                )

        else:
            validate_vector(
                transition.action,
                self.config.action_dimension,
                "continuous action",
            )

            if (
                transition.action_mask is not None
                or transition.next_action_mask is not None
            ):
                raise ReplayValidationError(
                    "Continuous action masks unsupported"
                )

        if (
            transition.discount is not None
            and transition.discount < 0
        ):
            raise ReplayValidationError(
                "Invalid transition discount"
            )

    # --------------------------------------------------------
    # INSERT
    # --------------------------------------------------------

    def _insert(
        self,
        transition: ReplayTransition,
        priority: float | None = None,
    ) -> int:
        self._validate_transition(
            transition
        )

        index = self._position

        self._storage[index] = transition

        if priority is None:
            priority_value = self._max_priority

        else:
            priority_value = (
                abs(
                    finite_float(
                        priority,
                        "priority",
                    )
                )
                + self.config.priority_epsilon
            )

        self._priorities[index] = priority_value

        self._tree.update(
            index,
            priority_value
            ** self.config.priority_alpha,
        )

        self._max_priority = max(
            self._max_priority,
            priority_value,
        )

        self._position = (
            self._position + 1
        ) % self.config.capacity

        self._size = min(
            self._size + 1,
            self.config.capacity,
        )

        self._total_added += 1

        self.status = (
            ReplayBufferStatus.FULL
            if self._size == self.config.capacity
            else ReplayBufferStatus.ACTIVE
        )

        self.updated_at = utc_now()

        return index

    # --------------------------------------------------------
    # ADD TRANSITION
    # --------------------------------------------------------

    def add(
        self,
        transition: ReplayTransition,
        *,
        priority: float | None = None,
        use_n_step: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            if self.status in (
                ReplayBufferStatus.PAUSED,
                ReplayBufferStatus.ARCHIVED,
            ):
                raise ReplayBufferConflict(
                    "Buffer is not writable"
                )

            self._validate_transition(
                transition
            )

            if (
                use_n_step
                and self.config.n_step > 1
            ):
                aggregated = (
                    self._n_step_accumulator.add(
                        transition
                    )
                )

            else:
                aggregated = [transition]

            indices = [
                self._insert(
                    item,
                    priority=priority,
                )
                for item in aggregated
            ]

            return {
                "buffer_id": self.buffer_id,
                "inserted": len(indices),
                "indices": indices,
                "size": self._size,
                "pending_n_step": (
                    self._n_step_accumulator.pending_count()
                ),
            }

    # --------------------------------------------------------
    # FLUSH N-STEP TRANSITIONS
    # --------------------------------------------------------

    def flush_n_step(
        self,
        environment_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self.status in (
                ReplayBufferStatus.PAUSED,
                ReplayBufferStatus.ARCHIVED,
            ):
                raise ReplayBufferConflict(
                    "Buffer is not writable"
                )

            if environment_id is None:
                transitions = (
                    self._n_step_accumulator.flush_all()
                )

            else:
                transitions = (
                    self._n_step_accumulator.flush(
                        environment_id
                    )
                )

            indices = [
                self._insert(transition)
                for transition in transitions
            ]

            return {
                "inserted": len(indices),
                "indices": indices,
                "pending_n_step": (
                    self._n_step_accumulator.pending_count()
                ),
            }

    # --------------------------------------------------------
    # UNIFORM SAMPLING
    # --------------------------------------------------------

    def _sample_uniform(
        self,
        batch_size: int,
        replace: bool,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        indices = self._rng.choice(
            self._size,
            size=batch_size,
            replace=replace,
        ).astype(np.int64)

        probabilities = np.full(
            batch_size,
            1.0 / self._size,
            dtype=np.float64,
        )

        weights = np.ones(
            batch_size,
            dtype=np.float32,
        )

        return indices, probabilities, weights

    # --------------------------------------------------------
    # PRIORITIZED SAMPLING
    # --------------------------------------------------------

    def _sample_prioritized(
        self,
        batch_size: int,
        replace: bool,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        total = self._tree.total()

        if total <= 0:
            raise ReplayBufferEmpty(
                "Priority sum is zero"
            )

        scaled_priorities = np.asarray(
            [
                self._tree.get(index)
                for index in range(self._size)
            ],
            dtype=np.float64,
        )

        if (
            not np.isfinite(scaled_priorities).all()
            or np.any(scaled_priorities < 0)
        ):
            raise ReplayBufferError(
                "Invalid priority distribution"
            )

        probabilities_all = (
            scaled_priorities / total
        )

        # Weighted sampling without replacement is
        # supported by NumPy's choice implementation.
        # IS weights below use the base proportional
        # distribution; for without-replacement
        # sampling these are approximate corrections.

        indices = self._rng.choice(
            self._size,
            size=batch_size,
            replace=replace,
            p=probabilities_all,
        ).astype(np.int64)

        probabilities = probabilities_all[
            indices
        ]

        weights = (
            self._size * probabilities
        ) ** (-self._beta)

        max_weight = float(
            np.max(weights)
        )

        if max_weight > 0:
            weights /= max_weight

        self._beta = min(
            1.0,
            self._beta
            + self.config.priority_beta_increment,
        )

        return (
            indices,
            probabilities,
            weights.astype(np.float32),
        )

    # --------------------------------------------------------
    # SAMPLE BATCH
    # --------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        *,
        replace: bool | None = None,
    ) -> ReplaySampleBatch:
        positive_int(
            batch_size,
            "batch_size",
        )

        with self._lock:
            if self._size == 0:
                raise ReplayBufferEmpty(
                    "Cannot sample empty replay buffer"
                )

            if replace is None:
                replace = (
                    self.config.allow_replacement_sampling
                )

            if not isinstance(replace, bool):
                raise ReplayValidationError(
                    "replace must be boolean"
                )

            if not replace and batch_size > self._size:
                raise ReplayValidationError(
                    "Batch size exceeds buffer size"
                )

            if (
                self.config.strategy
                == ReplayStrategy.PRIORITIZED
            ):
                (
                    indices,
                    probabilities,
                    weights,
                ) = self._sample_prioritized(
                    batch_size,
                    replace,
                )

            else:
                (
                    indices,
                    probabilities,
                    weights,
                ) = self._sample_uniform(
                    batch_size,
                    replace,
                )

            transitions = [
                self._storage[int(index)]
                for index in indices
            ]

            if any(
                transition is None
                for transition in transitions
            ):
                raise ReplayBufferError(
                    "Sampled uninitialized transition"
                )

            observations = np.asarray(
                [
                    item.observation
                    for item in transitions
                ],
                dtype=np.float32,
            )

            next_observations = np.asarray(
                [
                    item.next_observation
                    for item in transitions
                ],
                dtype=np.float32,
            )

            if (
                self.config.action_type
                == ReplayActionType.DISCRETE
            ):
                actions = np.asarray(
                    [
                        item.action
                        for item in transitions
                    ],
                    dtype=np.int64,
                )

                action_masks = np.asarray(
                    [
                        (
                            item.action_mask
                            if item.action_mask is not None
                            else (
                                True,
                            ) * self.config.action_dimension
                        )
                        for item in transitions
                    ],
                    dtype=np.bool_,
                )

                next_action_masks = np.asarray(
                    [
                        (
                            item.next_action_mask
                            if item.next_action_mask is not None
                            else (
                                True,
                            ) * self.config.action_dimension
                        )
                        for item in transitions
                    ],
                    dtype=np.bool_,
                )

            else:
                actions = np.asarray(
                    [
                        item.action
                        for item in transitions
                    ],
                    dtype=np.float32,
                )

                action_masks = None
                next_action_masks = None

            rewards = np.asarray(
                [
                    item.reward
                    for item in transitions
                ],
                dtype=np.float32,
            )

            terminated = np.asarray(
                [
                    item.terminated
                    for item in transitions
                ],
                dtype=np.bool_,
            )

            truncated = np.asarray(
                [
                    item.truncated
                    for item in transitions
                ],
                dtype=np.bool_,
            )

            discounts = np.asarray(
                [
                    (
                        item.discount
                        if item.discount is not None
                        else self.config.gamma
                        ** item.n_steps
                    )
                    for item in transitions
                ],
                dtype=np.float32,
            )

            n_steps = np.asarray(
                [
                    item.n_steps
                    for item in transitions
                ],
                dtype=np.int32,
            )

            self._total_sampled += batch_size

            return ReplaySampleBatch(
                buffer_id=self.buffer_id,
                strategy=self.config.strategy,
                indices=indices,
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                terminated=terminated,
                truncated=truncated,
                discounts=discounts,
                n_steps=n_steps,
                importance_weights=weights,
                sampling_probabilities=probabilities,
                transition_ids=tuple(
                    item.transition_id
                    for item in transitions
                ),
                action_masks=action_masks,
                next_action_masks=next_action_masks,
            )

    # --------------------------------------------------------
    # UPDATE PRIORITIES
    # --------------------------------------------------------

    def update_priorities(
        self,
        indices: Sequence[int],
        td_errors: Sequence[float],
    ) -> dict[str, Any]:
        if (
            self.config.strategy
            != ReplayStrategy.PRIORITIZED
        ):
            raise ReplayBufferConflict(
                "Priority updates require prioritized replay"
            )

        if len(indices) != len(td_errors):
            raise ReplayValidationError(
                "indices and td_errors length mismatch"
            )

        if not indices:
            raise ReplayValidationError(
                "Priority update cannot be empty"
            )

        with self._lock:
            for index, td_error in zip(
                indices,
                td_errors,
            ):
                if (
                    isinstance(index, bool)
                    or not isinstance(
                        index,
                        (int, np.integer),
                    )
                    or not 0 <= int(index) < self._size
                ):
                    raise ReplayValidationError(
                        "Invalid replay index"
                    )

                priority = (
                    abs(
                        finite_float(
                            td_error,
                            "td_error",
                        )
                    )
                    + self.config.priority_epsilon
                )

                self._priorities[int(index)] = priority

                self._tree.update(
                    int(index),
                    priority
                    ** self.config.priority_alpha,
                )

                self._max_priority = max(
                    self._max_priority,
                    priority,
                )

            self.updated_at = utc_now()

            return {
                "buffer_id": self.buffer_id,
                "updated": len(indices),
                "maximum_priority": self._max_priority,
                "beta": self._beta,
            }

    # --------------------------------------------------------
    # INGEST OFFLINE DATASET
    # --------------------------------------------------------

    def ingest_offline_dataset(
        self,
        transitions: Sequence[ReplayTransition],
    ) -> dict[str, Any]:
        with self._lock:
            if self.status in (
                ReplayBufferStatus.PAUSED,
                ReplayBufferStatus.ARCHIVED,
            ):
                raise ReplayBufferConflict(
                    "Buffer is not writable"
                )

            for transition in transitions:
                if (
                    transition.source
                    != ReplaySource.OFFLINE_DATASET
                ):
                    raise ReplayValidationError(
                        "Offline ingestion requires "
                        "OFFLINE_DATASET source"
                    )

                self._validate_transition(
                    transition
                )

            indices = [
                self._insert(
                    transition
                )
                for transition in transitions
            ]

            return {
                "buffer_id": self.buffer_id,
                "ingested": len(indices),
                "size": self._size,
            }

    # --------------------------------------------------------
    # GET TRANSITION
    # --------------------------------------------------------

    def get_transition(
        self,
        index: int,
    ) -> dict[str, Any]:
        with self._lock:
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < self._size
            ):
                raise ReplayValidationError(
                    "Invalid transition index"
                )

            transition = self._storage[index]

            if transition is None:
                raise ReplayBufferNotFound(
                    "Transition not found"
                )

            return transition.to_dict()

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        with self._lock:
            transitions = [
                item
                for item in self._storage[:self._size]
                if item is not None
            ]

            rewards = np.asarray(
                [
                    item.reward
                    for item in transitions
                ],
                dtype=np.float64,
            )

            terminated_count = sum(
                item.terminated
                for item in transitions
            )

            truncated_count = sum(
                item.truncated
                for item in transitions
            )

            source_counts = {
                source.value: sum(
                    item.source == source
                    for item in transitions
                )
                for source in ReplaySource
            }

            priorities = self._priorities[
                :self._size
            ]

            return {
                "buffer_id": self.buffer_id,
                "tenant_id": self.tenant_id,
                "robot_id": self.robot_id,
                "status": self.status.value,
                "strategy": self.config.strategy.value,
                "action_type": (
                    self.config.action_type.value
                ),
                "capacity": self.config.capacity,
                "size": self._size,
                "utilization": (
                    self._size / self.config.capacity
                ),
                "total_added": self._total_added,
                "total_sampled": self._total_sampled,
                "mean_reward": (
                    float(rewards.mean())
                    if len(rewards)
                    else 0.0
                ),
                "minimum_reward": (
                    float(rewards.min())
                    if len(rewards)
                    else 0.0
                ),
                "maximum_reward": (
                    float(rewards.max())
                    if len(rewards)
                    else 0.0
                ),
                "terminated_count": terminated_count,
                "truncated_count": truncated_count,
                "source_counts": source_counts,
                "priority_mean": (
                    float(priorities.mean())
                    if len(priorities)
                    else 0.0
                ),
                "priority_max": (
                    float(priorities.max())
                    if len(priorities)
                    else 0.0
                ),
                "importance_beta": self._beta,
                "pending_n_step": (
                    self._n_step_accumulator.pending_count()
                ),
                "created_at": (
                    self.created_at.isoformat()
                ),
                "updated_at": (
                    self.updated_at.isoformat()
                ),
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    def clear(self) -> None:
        with self._lock:
            if self.status == ReplayBufferStatus.ARCHIVED:
                raise ReplayBufferConflict(
                    "Archived buffer cannot be cleared"
                )

            self._storage = [
                None
            ] * self.config.capacity

            self._priorities.fill(0.0)

            self._tree = SumSegmentTree(
                self.config.capacity
            )

            self._size = 0
            self._position = 0

            self._max_priority = 1.0
            self._beta = self.config.priority_beta

            self._n_step_accumulator.clear()

            self.status = ReplayBufferStatus.CREATED
            self.updated_at = utc_now()

    # --------------------------------------------------------
    # CHECKPOINT STATE
    # --------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            if self._n_step_accumulator.pending_count():
                raise ReplayBufferConflict(
                    "Flush pending n-step transitions "
                    "before checkpointing"
                )

            return {
                "format_version": 1,
                "buffer_id": self.buffer_id,
                "tenant_id": self.tenant_id,
                "robot_id": self.robot_id,
                "config": json_safe(
                    asdict(self.config)
                ),
                "storage": [
                    (
                        item.to_dict()
                        if item is not None
                        else None
                    )
                    for item in self._storage
                ],
                "priorities": (
                    self._priorities.copy()
                ),
                "size": self._size,
                "position": self._position,
                "total_added": self._total_added,
                "total_sampled": self._total_sampled,
                "max_priority": self._max_priority,
                "beta": self._beta,
                "rng_state": copy.deepcopy(
                    self._rng.bit_generator.state
                ),
                "status": self.status.value,
                "created_at": (
                    self.created_at.isoformat()
                ),
                "updated_at": (
                    self.updated_at.isoformat()
                ),
            }

    # --------------------------------------------------------
    # RESTORE CHECKPOINT STATE
    # --------------------------------------------------------

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        with self._lock:
            if (
                state.get("buffer_id") != self.buffer_id
                or state.get("tenant_id") != self.tenant_id
                or state.get("robot_id") != self.robot_id
            ):
                raise ReplayPermissionDenied(
                    "Checkpoint identity mismatch"
                )

            if state.get("config") != json_safe(
                asdict(self.config)
            ):
                raise ReplayValidationError(
                    "Checkpoint configuration mismatch"
                )

            raw_storage = state["storage"]

            if len(raw_storage) != self.config.capacity:
                raise ReplayValidationError(
                    "Checkpoint capacity mismatch"
                )

            restored: list[
                ReplayTransition | None
            ] = []

            for item in raw_storage:
                if item is None:
                    restored.append(None)
                    continue

                restored_transition = ReplayTransition(
                    **{
                        **item,
                        "source": ReplaySource(
                            item["source"]
                        ),
                        "observation": tuple(
                            item["observation"]
                        ),
                        "next_observation": tuple(
                            item["next_observation"]
                        ),
                        "action": (
                            item["action"]
                            if self.config.action_type
                            == ReplayActionType.DISCRETE
                            else tuple(item["action"])
                        ),
                        "action_mask": (
                            tuple(item["action_mask"])
                            if item.get("action_mask")
                            is not None
                            else None
                        ),
                        "next_action_mask": (
                            tuple(item["next_action_mask"])
                            if item.get("next_action_mask")
                            is not None
                            else None
                        ),
                        "created_at": datetime.fromisoformat(
                            item["created_at"]
                        ),
                    }
                )

                self._validate_transition(
                    restored_transition
                )

                restored.append(
                    restored_transition
                )

            priorities = np.asarray(
                state["priorities"],
                dtype=np.float64,
            )

            if (
                priorities.shape
                != (self.config.capacity,)
                or not np.isfinite(priorities).all()
                or np.any(priorities < 0)
            ):
                raise ReplayValidationError(
                    "Invalid checkpoint priorities"
                )

            size = int(state["size"])
            position = int(state["position"])

            if not 0 <= size <= self.config.capacity:
                raise ReplayValidationError(
                    "Invalid checkpoint size"
                )

            if not 0 <= position < self.config.capacity:
                raise ReplayValidationError(
                    "Invalid checkpoint position"
                )

            if any(
                restored[index] is None
                for index in range(size)
            ):
                raise ReplayValidationError(
                    "Checkpoint contains missing active entries"
                )

            tree = SumSegmentTree(
                self.config.capacity
            )

            for index in range(size):
                tree.update(
                    index,
                    priorities[index]
                    ** self.config.priority_alpha,
                )

            restored_rng = np.random.default_rng()

            restored_rng.bit_generator.state = (
                copy.deepcopy(state["rng_state"])
            )

            self._storage = restored
            self._priorities = priorities
            self._tree = tree

            self._size = size
            self._position = position

            self._total_added = int(
                state["total_added"]
            )

            self._total_sampled = int(
                state["total_sampled"]
            )

            self._max_priority = finite_float(
                state["max_priority"],
                "max_priority",
            )

            self._beta = finite_float(
                state["beta"],
                "beta",
            )

            self._rng = restored_rng

            self.status = ReplayBufferStatus(
                state["status"]
            )

            self.created_at = datetime.fromisoformat(
                state["created_at"]
            )

            self.updated_at = datetime.fromisoformat(
                state["updated_at"]
            )

            self._n_step_accumulator.clear()

    def __len__(self) -> int:
        with self._lock:
            return self._size


# ============================================================
# MULTI-TENANT REPLAY BUFFER ENGINE
# ============================================================

class RoboticsReplayBufferEngine:

    def __init__(self) -> None:
        self._buffers: dict[
            tuple[str, str],
            ExperienceReplayBuffer,
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
    # CREATE BUFFER
    # --------------------------------------------------------

    def create_buffer(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        buffer_id: str,
        config: ReplayBufferConfig,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ReplayPermissionDenied(
                "Authorized buffer creation required"
            )

        validate_id(actor_id, "actor_id")

        key = (
            tenant_id,
            buffer_id,
        )

        with self._lock:
            if key in self._buffers:
                raise ReplayBufferConflict(
                    "Replay buffer already exists"
                )

            buffer = ExperienceReplayBuffer(
                buffer_id=buffer_id,
                tenant_id=tenant_id,
                robot_id=robot_id,
                config=config,
            )

            self._buffers[key] = buffer

            self._log(
                tenant_id=tenant_id,
                action="replay_buffer_created",
                entity_id=buffer_id,
                actor_id=actor_id,
                details={
                    "strategy": config.strategy.value,
                    "capacity": config.capacity,
                },
            )

            return buffer.statistics()

    # --------------------------------------------------------
    # GET BUFFER
    # --------------------------------------------------------

    def _get_buffer(
        self,
        tenant_id: str,
        buffer_id: str,
    ) -> ExperienceReplayBuffer:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            buffer_id,
            "buffer_id",
        )

        buffer = self._buffers.get(
            (tenant_id, buffer_id)
        )

        if buffer is None:
            raise ReplayBufferNotFound(
                "Replay buffer not found"
            )

        return buffer

    # --------------------------------------------------------
    # ADD EXPERIENCE
    # --------------------------------------------------------

    def add_experience(
        self,
        *,
        tenant_id: str,
        buffer_id: str,
        transition: ReplayTransition,
        priority: float | None = None,
        use_n_step: bool = True,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ReplayPermissionDenied(
                "Authorized experience ingestion required"
            )

        with self._lock:
            buffer = self._get_buffer(
                tenant_id,
                buffer_id,
            )

            result = buffer.add(
                transition,
                priority=priority,
                use_n_step=use_n_step,
            )

            self._log(
                tenant_id=tenant_id,
                action="experience_added",
                entity_id=buffer_id,
                actor_id=actor_id,
                details={
                    "transition_id": (
                        transition.transition_id
                    ),
                    "inserted": result["inserted"],
                },
            )

            return result

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    def sample_batch(
        self,
        *,
        tenant_id: str,
        buffer_id: str,
        batch_size: int,
        replace: bool | None = None,
    ) -> ReplaySampleBatch:
        with self._lock:
            buffer = self._get_buffer(
                tenant_id,
                buffer_id,
            )

            return buffer.sample(
                batch_size,
                replace=replace,
            )

    # --------------------------------------------------------
    # UPDATE PRIORITIES
    # --------------------------------------------------------

    def update_priorities(
        self,
        *,
        tenant_id: str,
        buffer_id: str,
        indices: Sequence[int],
        td_errors: Sequence[float],
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ReplayPermissionDenied(
                "Authorized priority update required"
            )

        with self._lock:
            buffer = self._get_buffer(
                tenant_id,
                buffer_id,
            )

            return buffer.update_priorities(
                indices,
                td_errors,
            )

    # --------------------------------------------------------
    # FLUSH N-STEP
    # --------------------------------------------------------

    def flush_n_step(
        self,
        *,
        tenant_id: str,
        buffer_id: str,
        environment_id: str | None = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ReplayPermissionDenied(
                "Authorized flush required"
            )

        with self._lock:
            buffer = self._get_buffer(
                tenant_id,
                buffer_id,
            )

            return buffer.flush_n_step(
                environment_id
            )

    # --------------------------------------------------------
    # BUFFER STATISTICS
    # --------------------------------------------------------

    def buffer_statistics(
        self,
        *,
        tenant_id: str,
        buffer_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get_buffer(
                tenant_id,
                buffer_id,
            ).statistics()

    # --------------------------------------------------------
    # LIST BUFFERS
    # --------------------------------------------------------

    def list_buffers(
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
                buffer.statistics()
                for (tenant, _), buffer
                in self._buffers.items()
                if tenant == tenant_id
            ]

    # --------------------------------------------------------
    # PAUSE BUFFER
    # --------------------------------------------------------

    def pause_buffer(
        self,
        *,
        tenant_id: str,
        buffer_id: str,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ReplayPermissionDenied(
                "Authorized pause required"
            )

        with self._lock:
            buffer = self._get_buffer(
                tenant_id,
                buffer_id,
            )

            if buffer.status == ReplayBufferStatus.ARCHIVED:
                raise ReplayBufferConflict(
                    "Archived buffer cannot be paused"
                )

            buffer.status = ReplayBufferStatus.PAUSED
            buffer.updated_at = utc_now()

            return buffer.statistics()

    # --------------------------------------------------------
    # RESUME BUFFER
    # --------------------------------------------------------

    def resume_buffer(
        self,
        *,
        tenant_id: str,
        buffer_id: str,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ReplayPermissionDenied(
                "Authorized resume required"
            )

        with self._lock:
            buffer = self._get_buffer(
                tenant_id,
                buffer_id,
            )

            if buffer.status != ReplayBufferStatus.PAUSED:
                raise ReplayBufferConflict(
                    "Buffer is not paused"
                )

            buffer.status = (
                ReplayBufferStatus.FULL
                if len(buffer) == buffer.config.capacity
                else ReplayBufferStatus.ACTIVE
            )

            buffer.updated_at = utc_now()

            return buffer.statistics()

    # --------------------------------------------------------
    # ARCHIVE BUFFER
    # --------------------------------------------------------

    def archive_buffer(
        self,
        *,
        tenant_id: str,
        buffer_id: str,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ReplayPermissionDenied(
                "Authorized archive required"
            )

        with self._lock:
            buffer = self._get_buffer(
                tenant_id,
                buffer_id,
            )

            buffer.status = ReplayBufferStatus.ARCHIVED
            buffer.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                action="replay_buffer_archived",
                entity_id=buffer_id,
            )

            return buffer.statistics()

    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    def save_checkpoint(
        self,
        *,
        tenant_id: str,
        buffer_id: str,
        directory: str | Path,
        authorized: bool = False,
    ) -> Path:
        if authorized is not True:
            raise ReplayPermissionDenied(
                "Authorized checkpoint export required"
            )

        with self._lock:
            buffer = self._get_buffer(
                tenant_id,
                buffer_id,
            )

            state = buffer.state_dict()

            directory = Path(directory)

            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            output = directory / (
                f"{tenant_id}_{buffer_id}_replay.npz"
            )

            with tempfile.NamedTemporaryFile(
                dir=directory,
                suffix=".npz",
                delete=False,
            ) as temporary:
                temporary_path = Path(
                    temporary.name
                )

            try:
                metadata = {
                    key: value
                    for key, value in state.items()
                    if key != "priorities"
                }

                np.savez_compressed(
                    temporary_path,
                    priorities=state["priorities"],
                    metadata=np.asarray(
                        json.dumps(
                            json_safe(metadata),
                            allow_nan=False,
                        )
                    ),
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
                action="replay_checkpoint_saved",
                entity_id=buffer_id,
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
        buffer_id: str,
        checkpoint_path: str | Path,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ReplayPermissionDenied(
                "Authorized checkpoint loading required"
            )

        path = Path(checkpoint_path)

        if not path.is_file():
            raise ReplayBufferNotFound(
                "Checkpoint not found"
            )

        with self._lock:
            buffer = self._get_buffer(
                tenant_id,
                buffer_id,
            )

            if buffer.status == ReplayBufferStatus.ARCHIVED:
                raise ReplayBufferConflict(
                    "Archived buffer cannot be restored"
                )

            with np.load(
                path,
                allow_pickle=False,
            ) as checkpoint:
                priorities = checkpoint[
                    "priorities"
                ].copy()

                metadata = json.loads(
                    str(checkpoint["metadata"].item())
                )

            state = {
                **metadata,
                "priorities": priorities,
            }

            buffer.load_state_dict(state)

            self._log(
                tenant_id=tenant_id,
                action="replay_checkpoint_loaded",
                entity_id=buffer_id,
                details={
                    "path": str(path),
                },
            )

            return buffer.statistics()

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

        positive_int(
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
            buffers = [
                buffer.statistics()
                for (tenant, _), buffer
                in self._buffers.items()
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
            "buffers": buffers,
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
# TRANSITION FACTORY
# ============================================================

def create_transition(
    *,
    environment_id: str,
    episode_id: str,
    observation: Sequence[float],
    action: int | Sequence[float],
    reward: float,
    next_observation: Sequence[float],
    terminated: bool = False,
    truncated: bool = False,
    source: ReplaySource = ReplaySource.SIMULATION,
    action_mask: Sequence[bool] | None = None,
    next_action_mask: Sequence[bool] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ReplayTransition:
    if isinstance(action, (int, np.integer)) and not isinstance(
        action,
        (bool, np.bool_),
    ):
        normalized_action: int | tuple[float, ...] = int(
            action
        )

    else:
        normalized_action = tuple(
            float(value)
            for value in action
        )

    return ReplayTransition(
        transition_id=new_id(),
        environment_id=environment_id,
        episode_id=episode_id,
        observation=tuple(
            float(value)
            for value in observation
        ),
        action=normalized_action,
        reward=finite_float(
            reward,
            "reward",
        ),
        next_observation=tuple(
            float(value)
            for value in next_observation
        ),
        terminated=terminated,
        truncated=truncated,
        source=source,
        action_mask=(
            tuple(action_mask)
            if action_mask is not None
            else None
        ),
        next_action_mask=(
            tuple(next_action_mask)
            if next_action_mask is not None
            else None
        ),
        metadata=dict(metadata or {}),
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    engine = RoboticsReplayBufferEngine()

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"
    buffer_id = "robot_navigation_replay_v1"

    config = ReplayBufferConfig(
        observation_dimension=4,
        action_dimension=4,
        capacity=10_000,
        strategy=ReplayStrategy.PRIORITIZED,
        action_type=ReplayActionType.DISCRETE,
        gamma=0.99,
        n_step=3,
        priority_alpha=0.6,
        priority_beta=0.4,
        random_seed=42,
    )

    print("\nCREATE REPLAY BUFFER")

    print(
        json.dumps(
            engine.create_buffer(
                tenant_id=tenant_id,
                robot_id=robot_id,
                buffer_id=buffer_id,
                config=config,
                authorized=True,
                actor_id="training_admin",
            ),
            indent=2,
        )
    )

    print("\nADD SIMULATION EXPERIENCES")

    rng = np.random.default_rng(42)

    for episode in range(10):
        observation = rng.uniform(
            -5.0,
            5.0,
            size=4,
        )

        for step_index in range(20):
            next_observation = (
                observation
                + rng.normal(
                    0.0,
                    0.2,
                    size=4,
                )
            )

            action = int(
                rng.integers(0, 4)
            )

            reward = float(
                rng.normal(0.0, 1.0)
            )

            terminated = (
                step_index == 19
            )

            transition = create_transition(
                environment_id="simulation_env_01",
                episode_id=f"episode_{episode:03d}",
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                source=ReplaySource.SIMULATION,
                metadata={
                    "step_index": step_index,
                    "synthetic": True,
                },
            )

            engine.add_experience(
                tenant_id=tenant_id,
                buffer_id=buffer_id,
                transition=transition,
                authorized=True,
                actor_id="simulation_collector",
            )

            observation = next_observation

    print("\nBUFFER STATISTICS")

    print(
        json.dumps(
            engine.buffer_statistics(
                tenant_id=tenant_id,
                buffer_id=buffer_id,
            ),
            indent=2,
        )
    )

    print("\nSAMPLE PRIORITIZED BATCH")

    batch = engine.sample_batch(
        tenant_id=tenant_id,
        buffer_id=buffer_id,
        batch_size=16,
    )

    print(
        json.dumps(
            {
                "batch_size": len(batch.indices),
                "indices": batch.indices.tolist(),
                "observation_shape": list(
                    batch.observations.shape
                ),
                "action_shape": list(
                    batch.actions.shape
                ),
                "rewards": batch.rewards.tolist(),
                "importance_weights": (
                    batch.importance_weights.tolist()
                ),
                "discounts": batch.discounts.tolist(),
                "n_steps": batch.n_steps.tolist(),
            },
            indent=2,
        )
    )

    print("\nUPDATE TD-ERROR PRIORITIES")

    td_errors = rng.normal(
        0.0,
        1.0,
        size=len(batch.indices),
    )

    print(
        json.dumps(
            engine.update_priorities(
                tenant_id=tenant_id,
                buffer_id=buffer_id,
                indices=batch.indices.tolist(),
                td_errors=td_errors.tolist(),
                authorized=True,
            ),
            indent=2,
        )
    )

    print("\nSAVE CHECKPOINT")

    checkpoint_path = engine.save_checkpoint(
        tenant_id=tenant_id,
        buffer_id=buffer_id,
        directory="data/robotics/checkpoints",
        authorized=True,
    )

    print(checkpoint_path)

    print("\nEXPORT REPORT")

    report_path = engine.export_json(
        "data/robotics/replay_buffer_report.json",
        tenant_id=tenant_id,
    )

    print(report_path)


if __name__ == "__main__":
    main()
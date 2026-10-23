"""Offline Rl module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/offline_rl.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Offline Reinforcement Learning Engine

Python 3.11+
Dependencies:
    pip install numpy torch

Features:
    - Offline reinforcement learning
    - Conservative Q-Learning (CQL)
    - Discrete action spaces
    - Offline experience replay
    - Expert and simulation datasets
    - Transition validation
    - Bellman target computation
    - Conservative Q-value regularization
    - Target network updates
    - Gradient clipping
    - Dataset normalization
    - Mini-batch training
    - Offline policy evaluation
    - Behavioral policy agreement
    - Model checkpointing
    - Model versioning
    - Human-reviewed simulation approval
    - Multi-tenant isolation
    - Audit logging
    - JSON report export

Integrations:
    environment.py
    observation_space.py
    action_space.py
    imitation_learning.py
    curriculum.py
    domain_randomization.py
    simulator.py
    simulation_adapter.py
    evaluation_service.py
    model_registry.py
    safety_monitor.py

Safety:
    Offline training and simulation only.

    No physical robot actuation is performed.

    Offline evaluation does not establish real-world
    safety or policy performance.

    Human approval is required before simulation
    promotion. Physical deployment is not supported.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# EXCEPTIONS
# ============================================================

class OfflineRLError(Exception):
    pass


class OfflineRLValidationError(OfflineRLError):
    pass


class OfflineRLNotFound(OfflineRLError):
    pass


class OfflineRLConflict(OfflineRLError):
    pass


class OfflineRLPermissionDenied(OfflineRLError):
    pass


# ============================================================
# ENUMS
# ============================================================

class OfflineRLAlgorithm(str, Enum):
    CQL = "conservative_q_learning"


class DatasetSource(str, Enum):
    SIMULATION = "simulation"
    EXPERT_DEMONSTRATION = "expert_demonstration"
    VERIFIED_REPLAY = "verified_replay"


class PolicyStatus(str, Enum):
    TRAINED = "trained"
    EVALUATED = "evaluated"
    APPROVED_FOR_SIMULATION = "approved_for_simulation"
    ARCHIVED = "archived"


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
        raise OfflineRLValidationError(
            f"Invalid {name}"
        )

    return value


def finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, str))
        or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        )
    ):
        raise OfflineRLValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise OfflineRLValidationError(
            f"{name} must be finite"
        )

    return result


def numeric_vector(
    value: Sequence[float],
    dimension: int,
    name: str,
) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise OfflineRLValidationError(
            f"{name} must be a numeric vector"
        )

    try:
        array = np.asarray(
            value,
            dtype=np.float32,
        )
    except (TypeError, ValueError) as exc:
        raise OfflineRLValidationError(
            f"Invalid {name}"
        ) from exc

    if array.shape != (dimension,):
        raise OfflineRLValidationError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.isfinite(array).all():
        raise OfflineRLValidationError(
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

    if isinstance(value, (tuple, list)):
        return [
            json_safe(item)
            for item in value
        ]

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class OfflineRLConfig:
    observation_dimension: int
    action_count: int

    hidden_dimensions: tuple[int, ...] = (256, 256)

    learning_rate: float = 0.0003
    discount_factor: float = 0.99

    batch_size: int = 128
    epochs: int = 100

    conservative_weight: float = 1.0

    target_update_interval: int = 100
    target_soft_update_tau: float = 0.005

    gradient_clip_norm: float = 10.0

    validation_fraction: float = 0.2

    minimum_transitions: int = 100
    maximum_transitions: int = 1_000_000

    maximum_audit_events: int = 20_000

    random_seed: int = 42

    normalize_observations: bool = True

    device: str = "cpu"

    simulation_only: bool = True
    human_approval_required: bool = True

    def __post_init__(self) -> None:
        for name in (
            "observation_dimension",
            "action_count",
            "batch_size",
            "epochs",
            "target_update_interval",
            "minimum_transitions",
            "maximum_transitions",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise OfflineRLValidationError(
                    f"{name} must be a positive integer"
                )

        if not self.hidden_dimensions:
            raise OfflineRLValidationError(
                "At least one hidden layer required"
            )

        for dimension in self.hidden_dimensions:
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or not 1 <= dimension <= 8192
            ):
                raise OfflineRLValidationError(
                    "Invalid hidden dimension"
                )

        for name in (
            "learning_rate",
            "conservative_weight",
            "gradient_clip_norm",
        ):
            if finite_number(
                getattr(self, name),
                name,
            ) <= 0:
                raise OfflineRLValidationError(
                    f"{name} must be positive"
                )

        for name in (
            "discount_factor",
            "validation_fraction",
            "target_soft_update_tau",
        ):
            value = finite_number(
                getattr(self, name),
                name,
            )

            if not 0 < value <= 1:
                raise OfflineRLValidationError(
                    f"{name} must be in (0, 1]"
                )

        if self.validation_fraction >= 1:
            raise OfflineRLValidationError(
                "Validation fraction must be below 1"
            )

        if self.maximum_transitions < self.minimum_transitions:
            raise OfflineRLValidationError(
                "Maximum transitions below minimum"
            )

        if self.device not in ("cpu", "cuda"):
            raise OfflineRLValidationError(
                "Supported devices: cpu and cuda"
            )

        if self.device == "cuda" and not torch.cuda.is_available():
            raise OfflineRLValidationError(
                "CUDA is unavailable"
            )

        if self.simulation_only is not True:
            raise OfflineRLValidationError(
                "Physical robot training is not supported"
            )

        if self.human_approval_required is not True:
            raise OfflineRLValidationError(
                "Human approval cannot be disabled"
            )


# ============================================================
# OFFLINE TRANSITION
# ============================================================

@dataclass(frozen=True)
class OfflineTransition:
    transition_id: str

    tenant_id: str
    robot_id: str
    dataset_id: str

    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]

    terminated: bool
    truncated: bool = False

    episode_id: str = ""
    timestep: int = 0

    source: DatasetSource = DatasetSource.SIMULATION

    safety_verified: bool = False

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "transition_id",
            "tenant_id",
            "robot_id",
            "dataset_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if (
            isinstance(self.action, bool)
            or not isinstance(self.action, int)
            or self.action < 0
        ):
            raise OfflineRLValidationError(
                "Action must be a nonnegative integer"
            )

        finite_number(self.reward, "reward")

        if not isinstance(self.terminated, bool):
            raise OfflineRLValidationError(
                "terminated must be boolean"
            )

        if not isinstance(self.truncated, bool):
            raise OfflineRLValidationError(
                "truncated must be boolean"
            )

        if self.terminated and self.truncated:
            raise OfflineRLValidationError(
                "Transition cannot be both terminated "
                "and truncated"
            )

        if not isinstance(self.source, DatasetSource):
            raise OfflineRLValidationError(
                "Invalid dataset source"
            )

        if not isinstance(self.safety_verified, bool):
            raise OfflineRLValidationError(
                "safety_verified must be boolean"
            )

        if self.episode_id:
            validate_id(self.episode_id, "episode_id")

        if (
            isinstance(self.timestep, bool)
            or not isinstance(self.timestep, int)
            or self.timestep < 0
        ):
            raise OfflineRLValidationError(
                "Invalid timestep"
            )

        try:
            json.dumps(
                json_safe(self.metadata),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise OfflineRLValidationError(
                "Metadata is not JSON-compatible"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "transition_id": self.transition_id,
                "tenant_id": self.tenant_id,
                "robot_id": self.robot_id,
                "dataset_id": self.dataset_id,
                "observation": self.observation,
                "action": self.action,
                "reward": self.reward,
                "next_observation": self.next_observation,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "episode_id": self.episode_id,
                "timestep": self.timestep,
                "source": self.source,
                "safety_verified": self.safety_verified,
                "metadata": self.metadata,
                "created_at": self.created_at,
            }
        )


# ============================================================
# OFFLINE DATASET
# ============================================================

@dataclass
class OfflineRLDataset:
    dataset_id: str
    tenant_id: str
    robot_id: str
    name: str

    transitions: list[OfflineTransition] = field(
        default_factory=list
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(
        self,
        *,
        include_transitions: bool = False,
    ) -> dict[str, Any]:
        result = {
            "dataset_id": self.dataset_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "name": self.name,
            "transition_count": len(self.transitions),
            "verified_count": sum(
                transition.safety_verified
                for transition in self.transitions
            ),
            "created_at": self.created_at.isoformat(),
        }

        if include_transitions:
            result["transitions"] = [
                transition.to_dict()
                for transition in self.transitions
            ]

        return result


# ============================================================
# OBSERVATION NORMALIZER
# ============================================================

@dataclass
class ObservationNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
    ) -> "ObservationNormalizer":
        mean = observations.mean(axis=0)

        std = observations.std(axis=0)

        std = np.where(
            std < 1e-6,
            1.0,
            std,
        )

        return cls(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
        )

    def transform(
        self,
        observations: np.ndarray,
    ) -> np.ndarray:
        return (
            (observations - self.mean) / self.std
        ).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


# ============================================================
# Q NETWORK
# ============================================================

class OfflineQNetwork(nn.Module):

    def __init__(
        self,
        observation_dimension: int,
        action_count: int,
        hidden_dimensions: tuple[int, ...],
    ) -> None:
        super().__init__()

        dimensions = (
            observation_dimension,
            *hidden_dimensions,
            action_count,
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
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(observations)


# ============================================================
# TRAINING RECORD
# ============================================================

@dataclass
class OfflineRLTrainingRecord:
    training_id: str

    tenant_id: str
    robot_id: str
    dataset_id: str
    model_id: str

    algorithm: OfflineRLAlgorithm

    epochs_completed: int

    training_losses: list[float]
    validation_losses: list[float]

    bellman_losses: list[float]
    conservative_losses: list[float]

    best_validation_loss: float

    training_sample_count: int
    validation_sample_count: int

    status: PolicyStatus = PolicyStatus.TRAINED

    reviewer_id: str | None = None
    checkpoint_path: str | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "training_id": self.training_id,
                "tenant_id": self.tenant_id,
                "robot_id": self.robot_id,
                "dataset_id": self.dataset_id,
                "model_id": self.model_id,
                "algorithm": self.algorithm,
                "epochs_completed": self.epochs_completed,
                "training_losses": self.training_losses,
                "validation_losses": self.validation_losses,
                "bellman_losses": self.bellman_losses,
                "conservative_losses": (
                    self.conservative_losses
                ),
                "best_validation_loss": (
                    self.best_validation_loss
                ),
                "training_sample_count": (
                    self.training_sample_count
                ),
                "validation_sample_count": (
                    self.validation_sample_count
                ),
                "status": self.status,
                "reviewer_id": self.reviewer_id,
                "checkpoint_path": self.checkpoint_path,
                "created_at": self.created_at,
                "simulation_only": True,
                "physical_deployment_approved": False,
            }
        )


# ============================================================
# OFFLINE RL ENGINE
# ============================================================

class RoboticsOfflineRLEngine:

    def __init__(
        self,
        config: OfflineRLConfig,
    ) -> None:
        self.config = config

        self._datasets: dict[
            tuple[str, str],
            OfflineRLDataset,
        ] = {}

        self._models: dict[
            tuple[str, str],
            OfflineQNetwork,
        ] = {}

        self._normalizers: dict[
            tuple[str, str],
            ObservationNormalizer,
        ] = {}

        self._records: dict[
            tuple[str, str],
            OfflineRLTrainingRecord,
        ] = {}

        self._audit: deque[
            dict[str, Any]
        ] = deque(
            maxlen=config.maximum_audit_events
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
    # CREATE DATASET
    # --------------------------------------------------------

    def create_dataset(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        dataset_id: str,
        name: str,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise OfflineRLPermissionDenied(
                "Authorized dataset creation required"
            )

        for field_name, value in (
            ("tenant_id", tenant_id),
            ("robot_id", robot_id),
            ("dataset_id", dataset_id),
            ("actor_id", actor_id),
        ):
            validate_id(value, field_name)

        if not isinstance(name, str) or not name.strip():
            raise OfflineRLValidationError(
                "Dataset name cannot be empty"
            )

        key = (tenant_id, dataset_id)

        with self._lock:
            if key in self._datasets:
                raise OfflineRLConflict(
                    "Dataset already exists"
                )

            self._datasets[key] = OfflineRLDataset(
                dataset_id=dataset_id,
                tenant_id=tenant_id,
                robot_id=robot_id,
                name=name.strip(),
            )

            self._log(
                tenant_id=tenant_id,
                action="offline_dataset_created",
                entity_id=dataset_id,
                actor_id=actor_id,
            )

            return self._datasets[key].to_dict()

    # --------------------------------------------------------
    # GET DATASET
    # --------------------------------------------------------

    def _get_dataset(
        self,
        tenant_id: str,
        dataset_id: str,
    ) -> OfflineRLDataset:
        validate_id(tenant_id, "tenant_id")
        validate_id(dataset_id, "dataset_id")

        dataset = self._datasets.get(
            (tenant_id, dataset_id)
        )

        if dataset is None:
            raise OfflineRLNotFound(
                "Offline dataset not found"
            )

        return dataset

    def get_dataset(
        self,
        *,
        tenant_id: str,
        dataset_id: str,
        include_transitions: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get_dataset(
                tenant_id,
                dataset_id,
            ).to_dict(
                include_transitions=include_transitions
            )

    # --------------------------------------------------------
    # ADD TRANSITION
    # --------------------------------------------------------

    def add_transition(
        self,
        transition: OfflineTransition,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise OfflineRLPermissionDenied(
                "Authorized transition ingestion required"
            )

        if not isinstance(
            transition,
            OfflineTransition,
        ):
            raise OfflineRLValidationError(
                "Invalid transition"
            )

        with self._lock:
            dataset = self._get_dataset(
                transition.tenant_id,
                transition.dataset_id,
            )

            if dataset.robot_id != transition.robot_id:
                raise OfflineRLConflict(
                    "Robot ID mismatch"
                )

            numeric_vector(
                transition.observation,
                self.config.observation_dimension,
                "observation",
            )

            numeric_vector(
                transition.next_observation,
                self.config.observation_dimension,
                "next_observation",
            )

            if not 0 <= transition.action < self.config.action_count:
                raise OfflineRLValidationError(
                    "Action index out of range"
                )

            if not transition.safety_verified:
                raise OfflineRLPermissionDenied(
                    "Transition requires simulation "
                    "safety verification"
                )

            if (
                len(dataset.transitions)
                >= self.config.maximum_transitions
            ):
                raise OfflineRLConflict(
                    "Dataset capacity reached"
                )

            if any(
                item.transition_id == transition.transition_id
                for item in dataset.transitions
            ):
                raise OfflineRLConflict(
                    "Duplicate transition ID"
                )

            dataset.transitions.append(transition)

            self._log(
                tenant_id=transition.tenant_id,
                action="offline_transition_added",
                entity_id=transition.transition_id,
                actor_id=actor_id,
                details={
                    "dataset_id": transition.dataset_id,
                    "source": transition.source.value,
                },
            )

            return transition.to_dict()

    # --------------------------------------------------------
    # ADD BATCH
    # --------------------------------------------------------

    def add_transitions(
        self,
        transitions: Sequence[OfflineTransition],
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise OfflineRLPermissionDenied(
                "Authorized ingestion required"
            )

        added_ids = []

        for transition in transitions:
            result = self.add_transition(
                transition,
                authorized=True,
                actor_id=actor_id,
            )

            added_ids.append(
                result["transition_id"]
            )

        return {
            "added_count": len(added_ids),
            "transition_ids": added_ids,
        }

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    def _prepare_data(
        self,
        dataset: OfflineRLDataset,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if (
            len(dataset.transitions)
            < self.config.minimum_transitions
        ):
            raise OfflineRLValidationError(
                "Insufficient offline transitions"
            )

        observations = np.asarray(
            [
                transition.observation
                for transition in dataset.transitions
            ],
            dtype=np.float32,
        )

        actions = np.asarray(
            [
                transition.action
                for transition in dataset.transitions
            ],
            dtype=np.int64,
        )

        rewards = np.asarray(
            [
                transition.reward
                for transition in dataset.transitions
            ],
            dtype=np.float32,
        )

        next_observations = np.asarray(
            [
                transition.next_observation
                for transition in dataset.transitions
            ],
            dtype=np.float32,
        )

        terminated = np.asarray(
            [
                float(transition.terminated)
                for transition in dataset.transitions
            ],
            dtype=np.float32,
        )

        return (
            observations,
            actions,
            rewards,
            next_observations,
            terminated,
        )

    # --------------------------------------------------------
    # CQL LOSS
    # --------------------------------------------------------

    def _cql_loss(
        self,
        policy: OfflineQNetwork,
        target_policy: OfflineQNetwork,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        terminated: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        q_values = policy(observations)

        selected_q = q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = target_policy(
                next_observations
            )

            max_next_q = next_q_values.max(
                dim=1
            ).values

            targets = (
                rewards
                + self.config.discount_factor
                * (1.0 - terminated)
                * max_next_q
            )

        bellman_loss = nn.functional.mse_loss(
            selected_q,
            targets,
        )

        conservative_loss = (
            torch.logsumexp(
                q_values,
                dim=1,
            )
            - selected_q
        ).mean()

        total_loss = (
            bellman_loss
            + self.config.conservative_weight
            * conservative_loss
        )

        return (
            total_loss,
            bellman_loss,
            conservative_loss,
        )

    # --------------------------------------------------------
    # SOFT TARGET UPDATE
    # --------------------------------------------------------

    @staticmethod
    def _soft_update(
        target_policy: OfflineQNetwork,
        policy: OfflineQNetwork,
        tau: float,
    ) -> None:
        with torch.no_grad():
            for target_parameter, parameter in zip(
                target_policy.parameters(),
                policy.parameters(),
            ):
                target_parameter.mul_(
                    1.0 - tau
                )

                target_parameter.add_(
                    parameter,
                    alpha=tau,
                )

    # --------------------------------------------------------
    # TRAIN CONSERVATIVE Q-LEARNING
    # --------------------------------------------------------

    def train_cql(
        self,
        *,
        tenant_id: str,
        dataset_id: str,
        model_id: str,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise OfflineRLPermissionDenied(
                "Authorized offline training required"
            )

        validate_id(model_id, "model_id")
        validate_id(actor_id, "actor_id")

        with self._lock:
            dataset = self._get_dataset(
                tenant_id,
                dataset_id,
            )

            key = (tenant_id, model_id)

            if key in self._models:
                raise OfflineRLConflict(
                    "Model ID already exists"
                )

            (
                observations,
                actions,
                rewards,
                next_observations,
                terminated,
            ) = self._prepare_data(dataset)

            seed = self.config.random_seed

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            sample_count = len(observations)

            indices = np.random.default_rng(
                seed
            ).permutation(sample_count)

            validation_count = max(
                1,
                int(
                    sample_count
                    * self.config.validation_fraction
                ),
            )

            train_indices = indices[
                validation_count:
            ]

            validation_indices = indices[
                :validation_count
            ]

            if len(train_indices) == 0:
                raise OfflineRLValidationError(
                    "Training split is empty"
                )

            if self.config.normalize_observations:
                normalizer = ObservationNormalizer.fit(
                    observations[train_indices]
                )

            else:
                normalizer = ObservationNormalizer(
                    mean=np.zeros(
                        self.config.observation_dimension,
                        dtype=np.float32,
                    ),
                    std=np.ones(
                        self.config.observation_dimension,
                        dtype=np.float32,
                    ),
                )

            observations = normalizer.transform(
                observations
            )

            next_observations = normalizer.transform(
                next_observations
            )

            device = torch.device(
                self.config.device
            )

            policy = OfflineQNetwork(
                self.config.observation_dimension,
                self.config.action_count,
                self.config.hidden_dimensions,
            ).to(device)

            target_policy = copy.deepcopy(
                policy
            ).to(device)

            target_policy.eval()

            optimizer = torch.optim.Adam(
                policy.parameters(),
                lr=self.config.learning_rate,
            )

            train_dataset = TensorDataset(
                torch.from_numpy(
                    observations[train_indices]
                ),
                torch.from_numpy(
                    actions[train_indices]
                ),
                torch.from_numpy(
                    rewards[train_indices]
                ),
                torch.from_numpy(
                    next_observations[train_indices]
                ),
                torch.from_numpy(
                    terminated[train_indices]
                ),
            )

            loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(
                    seed
                ),
            )

            validation_tensors = [
                torch.from_numpy(array[
                    validation_indices
                ]).to(device)
                for array in (
                    observations,
                    actions,
                    rewards,
                    next_observations,
                    terminated,
                )
            ]

            training_losses = []
            validation_losses = []

            bellman_losses = []
            conservative_losses = []

            best_validation_loss = float("inf")
            best_weights = None

            update_step = 0

            for epoch in range(self.config.epochs):
                policy.train()

                epoch_total = 0.0
                epoch_bellman = 0.0
                epoch_conservative = 0.0

                processed = 0

                for batch in loader:
                    (
                        batch_observations,
                        batch_actions,
                        batch_rewards,
                        batch_next_observations,
                        batch_terminated,
                    ) = [
                        item.to(device)
                        for item in batch
                    ]

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    (
                        total_loss,
                        bellman_loss,
                        conservative_loss,
                    ) = self._cql_loss(
                        policy,
                        target_policy,
                        batch_observations,
                        batch_actions,
                        batch_rewards,
                        batch_next_observations,
                        batch_terminated,
                    )

                    if not torch.isfinite(total_loss):
                        raise OfflineRLError(
                            "Non-finite CQL loss"
                        )

                    total_loss.backward()

                    nn.utils.clip_grad_norm_(
                        policy.parameters(),
                        max_norm=(
                            self.config.gradient_clip_norm
                        ),
                    )

                    optimizer.step()

                    update_step += 1

                    if (
                        update_step
                        % self.config.target_update_interval
                        == 0
                    ):
                        self._soft_update(
                            target_policy,
                            policy,
                            self.config.target_soft_update_tau,
                        )

                    batch_size = batch_observations.shape[0]

                    epoch_total += (
                        float(total_loss.item())
                        * batch_size
                    )

                    epoch_bellman += (
                        float(bellman_loss.item())
                        * batch_size
                    )

                    epoch_conservative += (
                        float(conservative_loss.item())
                        * batch_size
                    )

                    processed += batch_size

                policy.eval()

                with torch.no_grad():
                    (
                        validation_loss,
                        _,
                        _,
                    ) = self._cql_loss(
                        policy,
                        target_policy,
                        *validation_tensors,
                    )

                    validation_value = float(
                        validation_loss.item()
                    )

                if not math.isfinite(
                    validation_value
                ):
                    raise OfflineRLError(
                        "Non-finite validation loss"
                    )

                training_losses.append(
                    round(
                        epoch_total / processed,
                        8,
                    )
                )

                validation_losses.append(
                    round(
                        validation_value,
                        8,
                    )
                )

                bellman_losses.append(
                    round(
                        epoch_bellman / processed,
                        8,
                    )
                )

                conservative_losses.append(
                    round(
                        epoch_conservative / processed,
                        8,
                    )
                )

                if validation_value < best_validation_loss:
                    best_validation_loss = (
                        validation_value
                    )

                    best_weights = copy.deepcopy(
                        policy.state_dict()
                    )

            if best_weights is None:
                raise OfflineRLError(
                    "Training produced no valid checkpoint"
                )

            policy.load_state_dict(
                best_weights
            )

            policy.eval()

            record = OfflineRLTrainingRecord(
                training_id=new_id(),
                tenant_id=tenant_id,
                robot_id=dataset.robot_id,
                dataset_id=dataset_id,
                model_id=model_id,
                algorithm=OfflineRLAlgorithm.CQL,
                epochs_completed=len(
                    training_losses
                ),
                training_losses=training_losses,
                validation_losses=validation_losses,
                bellman_losses=bellman_losses,
                conservative_losses=conservative_losses,
                best_validation_loss=round(
                    best_validation_loss,
                    8,
                ),
                training_sample_count=len(
                    train_indices
                ),
                validation_sample_count=len(
                    validation_indices
                ),
            )

            self._models[key] = policy
            self._normalizers[key] = normalizer
            self._records[key] = record

            self._log(
                tenant_id=tenant_id,
                action="offline_cql_model_trained",
                entity_id=model_id,
                actor_id=actor_id,
                details={
                    "dataset_id": dataset_id,
                    "epochs": record.epochs_completed,
                    "validation_loss": (
                        record.best_validation_loss
                    ),
                },
            )

            return record.to_dict()

    # --------------------------------------------------------
    # PREDICT ACTION
    # --------------------------------------------------------

    def predict_action(
        self,
        *,
        tenant_id: str,
        model_id: str,
        observation: Sequence[float],
        allowed_actions: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            key = (tenant_id, model_id)

            policy = self._models.get(key)

            if policy is None:
                raise OfflineRLNotFound(
                    "Offline RL model not found"
                )

            vector = numeric_vector(
                observation,
                self.config.observation_dimension,
                "observation",
            )

            normalizer = self._normalizers[key]

            normalized = normalizer.transform(
                vector
            )

            tensor = torch.from_numpy(
                normalized
            ).unsqueeze(0).to(
                self.config.device
            )

            policy.eval()

            with torch.no_grad():
                q_values = policy(
                    tensor
                ).cpu().numpy()[0]

            if allowed_actions is None:
                valid_actions = list(
                    range(self.config.action_count)
                )

            else:
                valid_actions = list(
                    allowed_actions
                )

                if not valid_actions:
                    raise OfflineRLValidationError(
                        "Allowed actions cannot be empty"
                    )

                for action in valid_actions:
                    if (
                        isinstance(action, bool)
                        or not isinstance(action, int)
                        or not 0 <= action < self.config.action_count
                    ):
                        raise OfflineRLValidationError(
                            "Invalid allowed action"
                        )

            selected_action = max(
                valid_actions,
                key=lambda action: float(
                    q_values[action]
                ),
            )

            return {
                "tenant_id": tenant_id,
                "model_id": model_id,
                "selected_action": int(
                    selected_action
                ),
                "q_values": q_values.tolist(),
                "selected_q_value": float(
                    q_values[selected_action]
                ),
                "allowed_actions": valid_actions,
                "simulation_only": True,
                "physical_execution_authorized": False,
            }

    # --------------------------------------------------------
    # OFFLINE POLICY EVALUATION
    # --------------------------------------------------------

    def evaluate_policy(
        self,
        *,
        tenant_id: str,
        model_id: str,
        dataset_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            key = (tenant_id, model_id)

            policy = self._models.get(key)

            if policy is None:
                raise OfflineRLNotFound(
                    "Model not found"
                )

            dataset = self._get_dataset(
                tenant_id,
                dataset_id,
            )

            record = self._records[key]

            if dataset.robot_id != record.robot_id:
                raise OfflineRLConflict(
                    "Dataset robot does not match model"
                )

            (
                observations,
                actions,
                rewards,
                _,
                _,
            ) = self._prepare_data(
                dataset
            )

            normalized = self._normalizers[
                key
            ].transform(
                observations
            )

            tensor = torch.from_numpy(
                normalized
            ).to(self.config.device)

            policy.eval()

            with torch.no_grad():
                q_values = policy(
                    tensor
                ).cpu().numpy()

            predicted_actions = np.argmax(
                q_values,
                axis=1,
            )

            agreement = float(
                np.mean(
                    predicted_actions == actions
                )
            )

            selected_q = q_values[
                np.arange(len(actions)),
                actions,
            ]

            metrics = {
                "behavior_action_agreement": agreement,
                "mean_dataset_reward": float(
                    np.mean(rewards)
                ),
                "mean_logged_action_q": float(
                    np.mean(selected_q)
                ),
                "mean_greedy_action_q": float(
                    np.mean(
                        np.max(
                            q_values,
                            axis=1,
                        )
                    )
                ),
            }

            record.status = PolicyStatus.EVALUATED

            self._log(
                tenant_id=tenant_id,
                action="offline_policy_evaluated",
                entity_id=model_id,
                details={
                    "dataset_id": dataset_id,
                    "metrics": metrics,
                },
            )

            return {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "sample_count": len(actions),
                "metrics": metrics,
                "evaluation_limitations": (
                    "Action agreement, logged rewards, "
                    "and Q estimates do not constitute "
                    "unbiased off-policy performance "
                    "evaluation. Validate on independent "
                    "episodes in a simulator."
                ),
                "physical_deployment_approved": False,
            }

    # --------------------------------------------------------
    # APPROVE FOR SIMULATION
    # --------------------------------------------------------

    def approve_for_simulation(
        self,
        *,
        tenant_id: str,
        model_id: str,
        reviewer_id: str,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise OfflineRLPermissionDenied(
                "Human approval required"
            )

        validate_id(reviewer_id, "reviewer_id")

        with self._lock:
            record = self._records.get(
                (tenant_id, model_id)
            )

            if record is None:
                raise OfflineRLNotFound(
                    "Training record not found"
                )

            if record.status != PolicyStatus.EVALUATED:
                raise OfflineRLConflict(
                    "Policy must be evaluated first"
                )

            record.status = (
                PolicyStatus.APPROVED_FOR_SIMULATION
            )

            record.reviewer_id = reviewer_id

            self._log(
                tenant_id=tenant_id,
                action="policy_approved_for_simulation",
                entity_id=model_id,
                actor_id=reviewer_id,
            )

            return {
                "model_id": model_id,
                "status": record.status.value,
                "reviewer_id": reviewer_id,
                "simulation_approved": True,
                "physical_deployment_approved": False,
            }

    # --------------------------------------------------------
    # GET TRAINING RECORD
    # --------------------------------------------------------

    def get_training_record(
        self,
        *,
        tenant_id: str,
        model_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(
                (tenant_id, model_id)
            )

            if record is None:
                raise OfflineRLNotFound(
                    "Training record not found"
                )

            return record.to_dict()

    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    def save_checkpoint(
        self,
        *,
        tenant_id: str,
        model_id: str,
        directory: str | Path,
        authorized: bool = False,
    ) -> Path:
        if authorized is not True:
            raise OfflineRLPermissionDenied(
                "Authorized checkpoint export required"
            )

        with self._lock:
            key = (tenant_id, model_id)

            policy = self._models.get(key)

            if policy is None:
                raise OfflineRLNotFound(
                    "Model not found"
                )

            record = self._records[key]
            normalizer = self._normalizers[key]

            output_directory = Path(
                directory
            )

            output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            output = output_directory / (
                f"{tenant_id}_{model_id}_offline_rl.pt"
            )

            checkpoint = {
                "format_version": 1,
                "tenant_id": tenant_id,
                "model_id": model_id,
                "robot_id": record.robot_id,
                "dataset_id": record.dataset_id,
                "algorithm": OfflineRLAlgorithm.CQL.value,
                "observation_dimension": (
                    self.config.observation_dimension
                ),
                "action_count": self.config.action_count,
                "hidden_dimensions": (
                    self.config.hidden_dimensions
                ),
                "model_state_dict": (
                    policy.state_dict()
                ),
                "normalizer_mean": torch.from_numpy(
                    normalizer.mean.copy()
                ),
                "normalizer_std": torch.from_numpy(
                    normalizer.std.copy()
                ),
                "training_record": record.to_dict(),
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

            record.checkpoint_path = str(output)

            self._log(
                tenant_id=tenant_id,
                action="offline_checkpoint_saved",
                entity_id=model_id,
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
        model_id: str,
        checkpoint_path: str | Path,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise OfflineRLPermissionDenied(
                "Authorized checkpoint loading required"
            )

        validate_id(tenant_id, "tenant_id")
        validate_id(model_id, "model_id")

        path = Path(checkpoint_path)

        if not path.is_file():
            raise OfflineRLNotFound(
                "Checkpoint not found"
            )

        with self._lock:
            key = (tenant_id, model_id)

            if key in self._models:
                raise OfflineRLConflict(
                    "Model ID already exists"
                )

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            if not isinstance(checkpoint, dict):
                raise OfflineRLValidationError(
                    "Invalid checkpoint"
                )

            if (
                checkpoint.get("tenant_id") != tenant_id
                or checkpoint.get("model_id") != model_id
            ):
                raise OfflineRLPermissionDenied(
                    "Checkpoint identity mismatch"
                )

            if (
                checkpoint.get("observation_dimension")
                != self.config.observation_dimension
                or checkpoint.get("action_count")
                != self.config.action_count
                or tuple(
                    checkpoint.get("hidden_dimensions", ())
                ) != self.config.hidden_dimensions
                or checkpoint.get("algorithm")
                != OfflineRLAlgorithm.CQL.value
            ):
                raise OfflineRLValidationError(
                    "Checkpoint architecture mismatch"
                )

            policy = OfflineQNetwork(
                self.config.observation_dimension,
                self.config.action_count,
                self.config.hidden_dimensions,
            )

            policy.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )

            policy = policy.to(
                self.config.device
            )

            policy.eval()

            mean = checkpoint[
                "normalizer_mean"
            ].cpu().numpy().astype(np.float32)

            std = checkpoint[
                "normalizer_std"
            ].cpu().numpy().astype(np.float32)

            expected_shape = (
                self.config.observation_dimension,
            )

            if (
                mean.shape != expected_shape
                or std.shape != expected_shape
                or not np.isfinite(mean).all()
                or not np.isfinite(std).all()
                or np.any(std <= 0)
            ):
                raise OfflineRLValidationError(
                    "Invalid checkpoint normalization"
                )

            normalizer = ObservationNormalizer(
                mean=mean,
                std=std,
            )

            saved = checkpoint[
                "training_record"
            ]

            record = OfflineRLTrainingRecord(
                training_id=saved["training_id"],
                tenant_id=tenant_id,
                robot_id=saved["robot_id"],
                dataset_id=saved["dataset_id"],
                model_id=model_id,
                algorithm=OfflineRLAlgorithm(
                    saved["algorithm"]
                ),
                epochs_completed=saved[
                    "epochs_completed"
                ],
                training_losses=list(
                    saved["training_losses"]
                ),
                validation_losses=list(
                    saved["validation_losses"]
                ),
                bellman_losses=list(
                    saved["bellman_losses"]
                ),
                conservative_losses=list(
                    saved["conservative_losses"]
                ),
                best_validation_loss=saved[
                    "best_validation_loss"
                ],
                training_sample_count=saved[
                    "training_sample_count"
                ],
                validation_sample_count=saved[
                    "validation_sample_count"
                ],
                status=PolicyStatus(
                    saved["status"]
                ),
                reviewer_id=saved.get(
                    "reviewer_id"
                ),
                checkpoint_path=str(path),
            )

            self._models[key] = policy
            self._normalizers[key] = normalizer
            self._records[key] = record

            self._log(
                tenant_id=tenant_id,
                action="offline_checkpoint_loaded",
                entity_id=model_id,
                details={
                    "path": str(path),
                },
            )

            return record.to_dict()

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
            datasets = [
                dataset.to_dict()
                for (tenant, _), dataset
                in self._datasets.items()
                if tenant == tenant_id
            ]

            records = [
                record.to_dict()
                for (tenant, _), record
                in self._records.items()
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
            "datasets": datasets,
            "training_records": records,
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
# SYNTHETIC OFFLINE DATASET
# ============================================================

def create_synthetic_offline_transitions(
    *,
    tenant_id: str,
    robot_id: str,
    dataset_id: str,
    count: int = 1_000,
    seed: int = 42,
) -> list[OfflineTransition]:
    """
    Synthetic 2D navigation task.

    Observation:
        [robot_x, robot_y, goal_x, goal_y]

    Discrete actions:
        0 = move up
        1 = move down
        2 = move left
        3 = move right

    Transitions are generated by a simple
    simulation behavior policy.
    """

    if count < 1:
        raise OfflineRLValidationError(
            "count must be positive"
        )

    rng = np.random.default_rng(
        seed
    )

    action_deltas = np.asarray(
        [
            [0.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    transitions = []

    for index in range(count):
        position = rng.uniform(
            -10.0,
            10.0,
            size=2,
        )

        goal = rng.uniform(
            -10.0,
            10.0,
            size=2,
        )

        before_distance = float(
            np.linalg.norm(
                goal - position
            )
        )

        action = int(
            rng.integers(
                0,
                4,
            )
        )

        next_position = (
            position
            + action_deltas[action]
        )

        after_distance = float(
            np.linalg.norm(
                goal - next_position
            )
        )

        reward = (
            before_distance
            - after_distance
            - 0.01
        )

        terminated = (
            after_distance < 0.75
        )

        observation = (
            float(position[0]),
            float(position[1]),
            float(goal[0]),
            float(goal[1]),
        )

        next_observation = (
            float(next_position[0]),
            float(next_position[1]),
            float(goal[0]),
            float(goal[1]),
        )

        transitions.append(
            OfflineTransition(
                transition_id=(
                    f"transition_{index + 1:07d}"
                ),
                tenant_id=tenant_id,
                robot_id=robot_id,
                dataset_id=dataset_id,
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                source=DatasetSource.SIMULATION,
                safety_verified=True,
                episode_id=(
                    f"synthetic_episode_{index:07d}"
                ),
                timestep=0,
                metadata={
                    "synthetic": True,
                    "simulation_only": True,
                },
            )
        )

    return transitions


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    config = OfflineRLConfig(
        observation_dimension=4,
        action_count=4,
        hidden_dimensions=(128, 128),
        learning_rate=0.0003,
        discount_factor=0.99,
        batch_size=128,
        epochs=30,
        conservative_weight=1.0,
        target_update_interval=50,
        minimum_transitions=100,
        device="cpu",
    )

    engine = RoboticsOfflineRLEngine(
        config
    )

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"
    dataset_id = "offline_navigation_dataset"
    model_id = "navigation_cql_policy_v1"

    print("\nCREATE OFFLINE DATASET")

    dataset = engine.create_dataset(
        tenant_id=tenant_id,
        robot_id=robot_id,
        dataset_id=dataset_id,
        name="Synthetic Offline Navigation Dataset",
        authorized=True,
        actor_id="training_admin",
    )

    print(
        json.dumps(
            dataset,
            indent=2,
        )
    )

    print("\nGENERATE OFFLINE TRANSITIONS")

    transitions = (
        create_synthetic_offline_transitions(
            tenant_id=tenant_id,
            robot_id=robot_id,
            dataset_id=dataset_id,
            count=1_000,
        )
    )

    ingestion = engine.add_transitions(
        transitions,
        authorized=True,
        actor_id="training_admin",
    )

    print(
        json.dumps(
            {
                "added_count": ingestion["added_count"],
            },
            indent=2,
        )
    )

    print("\nTRAIN CONSERVATIVE Q-LEARNING")

    training = engine.train_cql(
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        model_id=model_id,
        authorized=True,
        actor_id="training_admin",
    )

    print(
        json.dumps(
            {
                "model_id": training["model_id"],
                "algorithm": training["algorithm"],
                "epochs_completed": (
                    training["epochs_completed"]
                ),
                "best_validation_loss": (
                    training["best_validation_loss"]
                ),
            },
            indent=2,
        )
    )

    print("\nPREDICT SIMULATION ACTION")

    prediction = engine.predict_action(
        tenant_id=tenant_id,
        model_id=model_id,
        observation=(1.0, 1.0, 8.0, 8.0),
    )

    print(
        json.dumps(
            prediction,
            indent=2,
        )
    )

    print("\nEVALUATE OFFLINE POLICY")

    evaluation = engine.evaluate_policy(
        tenant_id=tenant_id,
        model_id=model_id,
        dataset_id=dataset_id,
    )

    print(
        json.dumps(
            evaluation,
            indent=2,
        )
    )

    print("\nAPPROVE FOR SIMULATION")

    approval = engine.approve_for_simulation(
        tenant_id=tenant_id,
        model_id=model_id,
        reviewer_id="simulation_reviewer",
        authorized=True,
    )

    print(
        json.dumps(
            approval,
            indent=2,
        )
    )

    print("\nSAVE CHECKPOINT")

    checkpoint = engine.save_checkpoint(
        tenant_id=tenant_id,
        model_id=model_id,
        directory="data/robotics/checkpoints",
        authorized=True,
    )

    print(checkpoint)

    print("\nEXPORT REPORT")

    output = engine.export_json(
        "data/robotics/offline_rl_report.json",
        tenant_id=tenant_id,
    )

    print(output)


if __name__ == "__main__":
    main()
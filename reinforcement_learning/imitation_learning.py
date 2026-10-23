"""Imitation Learning module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/imitation_learning.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Imitation Learning Engine

Python 3.11+
Dependencies:
    pip install numpy torch

Features:
    - Behavioral cloning (BC)
    - Expert demonstration collection and validation
    - Offline imitation learning
    - Continuous and discrete action spaces
    - MLP-based imitation policy
    - Dataset normalization
    - Training and validation splits
    - Mini-batch training
    - Early stopping
    - Gradient clipping
    - Training metrics
    - Policy evaluation
    - Expert-policy action comparison
    - DAgger-style dataset aggregation
    - Checkpoint save/load
    - Reproducible training
    - Multi-tenant isolation
    - Model version tracking
    - Human-reviewed model promotion
    - Audit logging
    - JSON report export

Integrations:
    action_space.py
    checkpoints.py
    curriculum.py
    domain_randomization.py
    environment.py
    simulator.py
    simulation_adapter.py
    robot_state.py
    safety_monitor.py
    evaluation_service.py
    model_registry.py

Safety:
    Simulation and offline training only.

    The engine does not connect to physical robots,
    execute hardware commands, or authorize deployment.

    DAgger aggregation requires externally supplied expert
    labels. It never assumes that its own predictions are
    expert demonstrations.

    Model predictions are research outputs and must not
    bypass robot command validation or safety interlocks.
"""

from __future__ import annotations

import copy
import hashlib
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

class ImitationLearningError(Exception):
    pass


class ImitationValidationError(ImitationLearningError):
    pass


class ImitationNotFound(ImitationLearningError):
    pass


class ImitationConflict(ImitationLearningError):
    pass


class ImitationPermissionDenied(ImitationLearningError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ActionSpaceType(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class LearningAlgorithm(str, Enum):
    BEHAVIORAL_CLONING = "behavioral_cloning"
    DAGGER = "dagger"


class ModelStatus(str, Enum):
    DRAFT = "draft"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    APPROVED_FOR_SIMULATION = "approved_for_simulation"
    ARCHIVED = "archived"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"


class DemonstrationSource(str, Enum):
    HUMAN_EXPERT = "human_expert"
    VERIFIED_EXPERT_POLICY = "verified_expert_policy"
    SIMULATION_EXPERT = "simulation_expert"


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
        raise ImitationValidationError(
            f"Invalid {name}"
        )

    return value


def validate_integer(
    value: int,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 10_000_000,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ImitationValidationError(
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
        raise ImitationValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise ImitationValidationError(
            f"{name} must be finite"
        )

    return result


def probability(value: Any, name: str) -> float:
    result = finite_number(value, name)

    if not 0 <= result <= 1:
        raise ImitationValidationError(
            f"{name} must be between 0 and 1"
        )

    return result


def numeric_vector(
    values: Sequence[Any],
    dimension: int,
    name: str,
) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ImitationValidationError(
            f"{name} must be a numeric sequence"
        )

    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ImitationValidationError(
            f"Invalid {name}"
        ) from exc

    if array.shape != (dimension,):
        raise ImitationValidationError(
            f"{name} must have shape ({dimension},)"
        )

    if array.dtype.kind not in "iuf":
        raise ImitationValidationError(
            f"{name} must contain numeric values"
        )

    array = array.astype(np.float32)

    if not np.isfinite(array).all():
        raise ImitationValidationError(
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

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, float) and not math.isfinite(value):
        raise ImitationValidationError(
            "Non-finite JSON value"
        )

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ImitationLearningConfig:
    observation_dimension: int
    action_dimension: int

    action_space_type: ActionSpaceType = (
        ActionSpaceType.CONTINUOUS
    )

    hidden_dimensions: tuple[int, ...] = (256, 256)

    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100

    validation_fraction: float = 0.2
    weight_decay: float = 0.00001
    gradient_clip_norm: float = 1.0

    early_stopping_patience: int = 15

    random_seed: int = 42

    normalize_observations: bool = True

    minimum_demonstrations: int = 20
    maximum_demonstrations: int = 1_000_000

    maximum_audit_events: int = 20_000

    device: str = "cpu"

    simulation_only: bool = True
    human_approval_required: bool = True

    def __post_init__(self) -> None:
        for name in (
            "observation_dimension",
            "action_dimension",
            "batch_size",
            "epochs",
            "early_stopping_patience",
            "minimum_demonstrations",
            "maximum_demonstrations",
            "maximum_audit_events",
        ):
            validate_integer(
                getattr(self, name),
                name,
                minimum=1,
            )

        if not isinstance(
            self.action_space_type,
            ActionSpaceType,
        ):
            raise ImitationValidationError(
                "Invalid action space type"
            )

        if not self.hidden_dimensions:
            raise ImitationValidationError(
                "At least one hidden layer is required"
            )

        for dimension in self.hidden_dimensions:
            validate_integer(
                dimension,
                "hidden dimension",
                minimum=1,
                maximum=8192,
            )

        if finite_number(
            self.learning_rate,
            "learning_rate",
        ) <= 0:
            raise ImitationValidationError(
                "Learning rate must be positive"
            )

        if finite_number(
            self.gradient_clip_norm,
            "gradient_clip_norm",
        ) <= 0:
            raise ImitationValidationError(
                "Gradient clipping norm must be positive"
            )

        if finite_number(
            self.weight_decay,
            "weight_decay",
        ) < 0:
            raise ImitationValidationError(
                "Weight decay cannot be negative"
            )

        fraction = probability(
            self.validation_fraction,
            "validation_fraction",
        )

        if not 0 < fraction < 1:
            raise ImitationValidationError(
                "Validation fraction must be between 0 and 1"
            )

        validate_integer(
            self.random_seed,
            "random_seed",
            maximum=2**32 - 1,
        )

        if (
            self.maximum_demonstrations
            < self.minimum_demonstrations
        ):
            raise ImitationValidationError(
                "Maximum demonstrations below minimum"
            )

        if self.device not in ("cpu", "cuda"):
            raise ImitationValidationError(
                "Supported devices: cpu, cuda"
            )

        if self.device == "cuda" and not torch.cuda.is_available():
            raise ImitationValidationError(
                "CUDA is not available"
            )

        if self.simulation_only is not True:
            raise ImitationValidationError(
                "Only simulation/offline training is supported"
            )

        if self.human_approval_required is not True:
            raise ImitationValidationError(
                "Human approval cannot be disabled"
            )


# ============================================================
# DEMONSTRATION
# ============================================================

@dataclass(frozen=True)
class ExpertDemonstration:
    demonstration_id: str

    tenant_id: str
    robot_id: str
    dataset_id: str

    observation: tuple[float, ...]

    expert_action: tuple[float, ...] | int

    source: DemonstrationSource

    episode_id: str = ""
    timestep: int = 0

    reward: float | None = None

    success: bool | None = None

    safety_verified: bool = False

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "demonstration_id",
            "tenant_id",
            "robot_id",
            "dataset_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if not isinstance(
            self.source,
            DemonstrationSource,
        ):
            raise ImitationValidationError(
                "Invalid demonstration source"
            )

        if not isinstance(self.observation, tuple):
            raise ImitationValidationError(
                "Observation must be a tuple"
            )

        for value in self.observation:
            finite_number(value, "observation value")

        if isinstance(self.expert_action, tuple):
            for value in self.expert_action:
                finite_number(value, "expert action")
        elif isinstance(self.expert_action, bool) or not isinstance(
            self.expert_action,
            int,
        ):
            raise ImitationValidationError(
                "Expert action must be a tuple or integer"
            )

        validate_integer(
            self.timestep,
            "timestep",
        )

        if self.episode_id:
            validate_id(self.episode_id, "episode_id")

        if self.reward is not None:
            finite_number(self.reward, "reward")

        if self.success is not None and not isinstance(
            self.success,
            bool,
        ):
            raise ImitationValidationError(
                "success must be boolean"
            )

        if not isinstance(self.safety_verified, bool):
            raise ImitationValidationError(
                "safety_verified must be boolean"
            )

        try:
            json.dumps(
                self.metadata,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ImitationValidationError(
                "Invalid metadata"
            ) from exc

        if (
            self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ImitationValidationError(
                "created_at must be timezone-aware"
            )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "demonstration_id": self.demonstration_id,
                "tenant_id": self.tenant_id,
                "robot_id": self.robot_id,
                "dataset_id": self.dataset_id,
                "observation": self.observation,
                "expert_action": self.expert_action,
                "source": self.source,
                "episode_id": self.episode_id,
                "timestep": self.timestep,
                "reward": self.reward,
                "success": self.success,
                "safety_verified": self.safety_verified,
                "metadata": self.metadata,
                "created_at": self.created_at,
            }
        )


# ============================================================
# DATASET
# ============================================================

@dataclass
class DemonstrationDataset:
    dataset_id: str
    tenant_id: str
    robot_id: str

    name: str

    demonstrations: list[ExpertDemonstration] = field(
        default_factory=list
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(
        self,
        *,
        include_demonstrations: bool = False,
    ) -> dict[str, Any]:
        result = {
            "dataset_id": self.dataset_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "name": self.name,
            "demonstration_count": len(
                self.demonstrations
            ),
            "verified_count": sum(
                item.safety_verified
                for item in self.demonstrations
            ),
            "created_at": self.created_at.isoformat(),
        }

        if include_demonstrations:
            result["demonstrations"] = [
                item.to_dict()
                for item in self.demonstrations
            ]

        return result


# ============================================================
# POLICY NETWORK
# ============================================================

class ImitationPolicyNetwork(nn.Module):

    def __init__(
        self,
        observation_dimension: int,
        action_dimension: int,
        hidden_dimensions: tuple[int, ...],
    ) -> None:
        super().__init__()

        dimensions = (
            observation_dimension,
            *hidden_dimensions,
            action_dimension,
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
# NORMALIZATION
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
# TRAINING RECORD
# ============================================================

@dataclass
class TrainingRecord:
    training_id: str

    tenant_id: str
    robot_id: str
    dataset_id: str
    model_id: str

    algorithm: LearningAlgorithm

    epochs_completed: int

    training_losses: list[float]
    validation_losses: list[float]

    best_validation_loss: float

    training_sample_count: int
    validation_sample_count: int

    status: ModelStatus = ModelStatus.TRAINED

    reviewer_id: str | None = None

    checkpoint_path: str | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_id": self.training_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "algorithm": self.algorithm.value,
            "epochs_completed": self.epochs_completed,
            "training_losses": list(
                self.training_losses
            ),
            "validation_losses": list(
                self.validation_losses
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
            "status": self.status.value,
            "reviewer_id": self.reviewer_id,
            "checkpoint_path": self.checkpoint_path,
            "created_at": self.created_at.isoformat(),
            "simulation_only": True,
            "physical_deployment_approved": False,
        }


# ============================================================
# IMITATION LEARNING ENGINE
# ============================================================

class RoboticsImitationLearningEngine:

    def __init__(
        self,
        config: ImitationLearningConfig,
    ) -> None:
        self.config = config

        self._datasets: dict[
            tuple[str, str],
            DemonstrationDataset,
        ] = {}

        self._models: dict[
            tuple[str, str],
            ImitationPolicyNetwork,
        ] = {}

        self._normalizers: dict[
            tuple[str, str],
            ObservationNormalizer,
        ] = {}

        self._training_records: dict[
            tuple[str, str],
            TrainingRecord,
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
            raise ImitationPermissionDenied(
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
            raise ImitationValidationError(
                "Dataset name cannot be empty"
            )

        key = (tenant_id, dataset_id)

        with self._lock:
            if key in self._datasets:
                raise ImitationConflict(
                    "Dataset already exists"
                )

            dataset = DemonstrationDataset(
                dataset_id=dataset_id,
                tenant_id=tenant_id,
                robot_id=robot_id,
                name=name.strip(),
            )

            self._datasets[key] = dataset

            self._log(
                tenant_id=tenant_id,
                action="dataset_created",
                entity_id=dataset_id,
                actor_id=actor_id,
            )

            return dataset.to_dict()

    # --------------------------------------------------------
    # GET DATASET
    # --------------------------------------------------------

    def _get_dataset(
        self,
        tenant_id: str,
        dataset_id: str,
    ) -> DemonstrationDataset:
        validate_id(tenant_id, "tenant_id")
        validate_id(dataset_id, "dataset_id")

        dataset = self._datasets.get(
            (tenant_id, dataset_id)
        )

        if dataset is None:
            raise ImitationNotFound(
                "Dataset not found"
            )

        return dataset

    def get_dataset(
        self,
        *,
        tenant_id: str,
        dataset_id: str,
        include_demonstrations: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get_dataset(
                tenant_id,
                dataset_id,
            ).to_dict(
                include_demonstrations=(
                    include_demonstrations
                )
            )

    # --------------------------------------------------------
    # ADD EXPERT DEMONSTRATION
    # --------------------------------------------------------

    def add_demonstration(
        self,
        demonstration: ExpertDemonstration,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ImitationPermissionDenied(
                "Authorized demonstration ingestion required"
            )

        if not isinstance(
            demonstration,
            ExpertDemonstration,
        ):
            raise ImitationValidationError(
                "Invalid demonstration"
            )

        validate_id(actor_id, "actor_id")

        with self._lock:
            dataset = self._get_dataset(
                demonstration.tenant_id,
                demonstration.dataset_id,
            )

            if dataset.robot_id != demonstration.robot_id:
                raise ImitationConflict(
                    "Robot ID does not match dataset"
                )

            numeric_vector(
                demonstration.observation,
                self.config.observation_dimension,
                "observation",
            )

            if (
                self.config.action_space_type
                == ActionSpaceType.CONTINUOUS
            ):
                numeric_vector(
                    demonstration.expert_action,
                    self.config.action_dimension,
                    "expert_action",
                )

            else:
                action = demonstration.expert_action

                if (
                    isinstance(action, bool)
                    or not isinstance(action, int)
                    or not 0 <= action < self.config.action_dimension
                ):
                    raise ImitationValidationError(
                        "Discrete action index out of range"
                    )

            if not demonstration.safety_verified:
                raise ImitationPermissionDenied(
                    "Expert demonstration requires "
                    "verified simulation safety review"
                )

            if (
                len(dataset.demonstrations)
                >= self.config.maximum_demonstrations
            ):
                raise ImitationConflict(
                    "Dataset capacity reached"
                )

            if any(
                item.demonstration_id
                == demonstration.demonstration_id
                for item in dataset.demonstrations
            ):
                raise ImitationConflict(
                    "Duplicate demonstration ID"
                )

            dataset.demonstrations.append(
                demonstration
            )

            self._log(
                tenant_id=demonstration.tenant_id,
                action="demonstration_added",
                entity_id=demonstration.demonstration_id,
                actor_id=actor_id,
                details={
                    "dataset_id": demonstration.dataset_id,
                    "source": demonstration.source.value,
                },
            )

            return demonstration.to_dict()

    # --------------------------------------------------------
    # BATCH INGESTION
    # --------------------------------------------------------

    def add_demonstrations(
        self,
        demonstrations: Sequence[ExpertDemonstration],
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ImitationPermissionDenied(
                "Authorized ingestion required"
            )

        added = []

        for demonstration in demonstrations:
            result = self.add_demonstration(
                demonstration,
                authorized=True,
                actor_id=actor_id,
            )

            added.append(result["demonstration_id"])

        return {
            "added_count": len(added),
            "demonstration_ids": added,
        }

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    def _prepare_data(
        self,
        dataset: DemonstrationDataset,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        demonstrations = dataset.demonstrations

        if (
            len(demonstrations)
            < self.config.minimum_demonstrations
        ):
            raise ImitationValidationError(
                "Insufficient expert demonstrations"
            )

        observations = np.asarray(
            [
                item.observation
                for item in demonstrations
            ],
            dtype=np.float32,
        )

        if (
            self.config.action_space_type
            == ActionSpaceType.CONTINUOUS
        ):
            actions = np.asarray(
                [
                    item.expert_action
                    for item in demonstrations
                ],
                dtype=np.float32,
            )

        else:
            actions = np.asarray(
                [
                    item.expert_action
                    for item in demonstrations
                ],
                dtype=np.int64,
            )

        return observations, actions

    # --------------------------------------------------------
    # TRAIN BEHAVIORAL CLONING
    # --------------------------------------------------------

    def train_behavioral_cloning(
        self,
        *,
        tenant_id: str,
        dataset_id: str,
        model_id: str,
        authorized: bool = False,
        actor_id: str = "system",
        algorithm: LearningAlgorithm = (
            LearningAlgorithm.BEHAVIORAL_CLONING
        ),
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ImitationPermissionDenied(
                "Authorized training required"
            )

        validate_id(model_id, "model_id")
        validate_id(actor_id, "actor_id")

        if not isinstance(
            algorithm,
            LearningAlgorithm,
        ):
            raise ImitationValidationError(
                "Invalid learning algorithm"
            )

        with self._lock:
            dataset = self._get_dataset(
                tenant_id,
                dataset_id,
            )

            key = (tenant_id, model_id)

            if key in self._models:
                raise ImitationConflict(
                    "Model ID already exists"
                )

            observations, actions = self._prepare_data(
                dataset
            )

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
                raise ImitationValidationError(
                    "Training split is empty"
                )

            x_train = observations[
                train_indices
            ]

            x_validation = observations[
                validation_indices
            ]

            y_train = actions[
                train_indices
            ]

            y_validation = actions[
                validation_indices
            ]

            if self.config.normalize_observations:
                normalizer = ObservationNormalizer.fit(
                    x_train
                )

                x_train = normalizer.transform(
                    x_train
                )

                x_validation = normalizer.transform(
                    x_validation
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

            device = torch.device(
                self.config.device
            )

            policy = ImitationPolicyNetwork(
                observation_dimension=(
                    self.config.observation_dimension
                ),
                action_dimension=(
                    self.config.action_dimension
                ),
                hidden_dimensions=(
                    self.config.hidden_dimensions
                ),
            ).to(device)

            x_train_tensor = torch.from_numpy(
                x_train
            )

            x_validation_tensor = torch.from_numpy(
                x_validation
            ).to(device)

            if (
                self.config.action_space_type
                == ActionSpaceType.CONTINUOUS
            ):
                y_train_tensor = torch.from_numpy(
                    y_train
                )

                y_validation_tensor = torch.from_numpy(
                    y_validation
                ).to(device)

                criterion: nn.Module = nn.MSELoss()

            else:
                y_train_tensor = torch.from_numpy(
                    y_train
                ).long()

                y_validation_tensor = torch.from_numpy(
                    y_validation
                ).long().to(device)

                criterion = nn.CrossEntropyLoss()

            loader = DataLoader(
                TensorDataset(
                    x_train_tensor,
                    y_train_tensor,
                ),
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(
                    seed
                ),
            )

            optimizer = torch.optim.AdamW(
                policy.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            training_losses = []
            validation_losses = []

            best_validation_loss = float("inf")
            best_weights = None

            patience_counter = 0

            for epoch in range(self.config.epochs):
                policy.train()

                epoch_loss = 0.0
                processed_samples = 0

                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    predictions = policy(
                        batch_x
                    )

                    loss = criterion(
                        predictions,
                        batch_y,
                    )

                    if not torch.isfinite(loss):
                        raise ImitationLearningError(
                            "Non-finite training loss"
                        )

                    loss.backward()

                    nn.utils.clip_grad_norm_(
                        policy.parameters(),
                        max_norm=(
                            self.config.gradient_clip_norm
                        ),
                    )

                    optimizer.step()

                    batch_size = batch_x.shape[0]

                    epoch_loss += (
                        float(loss.item())
                        * batch_size
                    )

                    processed_samples += batch_size

                mean_training_loss = (
                    epoch_loss / processed_samples
                )

                policy.eval()

                with torch.no_grad():
                    validation_predictions = policy(
                        x_validation_tensor
                    )

                    validation_loss = criterion(
                        validation_predictions,
                        y_validation_tensor,
                    )

                    mean_validation_loss = float(
                        validation_loss.item()
                    )

                if not math.isfinite(
                    mean_validation_loss
                ):
                    raise ImitationLearningError(
                        "Non-finite validation loss"
                    )

                training_losses.append(
                    round(mean_training_loss, 8)
                )

                validation_losses.append(
                    round(mean_validation_loss, 8)
                )

                if (
                    mean_validation_loss
                    < best_validation_loss - 1e-8
                ):
                    best_validation_loss = (
                        mean_validation_loss
                    )

                    best_weights = copy.deepcopy(
                        policy.state_dict()
                    )

                    patience_counter = 0

                else:
                    patience_counter += 1

                if (
                    patience_counter
                    >= self.config.early_stopping_patience
                ):
                    break

            if best_weights is None:
                raise ImitationLearningError(
                    "No valid model checkpoint produced"
                )

            policy.load_state_dict(
                best_weights
            )

            policy.eval()

            training_id = new_id()

            record = TrainingRecord(
                training_id=training_id,
                tenant_id=tenant_id,
                robot_id=dataset.robot_id,
                dataset_id=dataset_id,
                model_id=model_id,
                algorithm=algorithm,
                epochs_completed=len(
                    training_losses
                ),
                training_losses=training_losses,
                validation_losses=validation_losses,
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
            self._training_records[key] = record

            self._log(
                tenant_id=tenant_id,
                action="model_trained",
                entity_id=model_id,
                actor_id=actor_id,
                details={
                    "dataset_id": dataset_id,
                    "algorithm": algorithm.value,
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
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(model_id, "model_id")

        with self._lock:
            key = (tenant_id, model_id)

            policy = self._models.get(key)

            if policy is None:
                raise ImitationNotFound(
                    "Model not found"
                )

            normalizer = self._normalizers[key]

            vector = numeric_vector(
                observation,
                self.config.observation_dimension,
                "observation",
            )

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
                output = policy(
                    tensor
                ).cpu().numpy()[0]

            if (
                self.config.action_space_type
                == ActionSpaceType.CONTINUOUS
            ):
                action: Any = output.tolist()
                confidence = None

            else:
                shifted = output - np.max(output)

                probabilities = np.exp(
                    shifted
                )

                probabilities /= probabilities.sum()

                action = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    np.max(probabilities)
                )

            return {
                "model_id": model_id,
                "tenant_id": tenant_id,
                "predicted_action": action,
                "confidence": confidence,
                "simulation_only": True,
                "physical_execution_authorized": False,
            }

    # --------------------------------------------------------
    # EVALUATE MODEL
    # --------------------------------------------------------

    def evaluate_model(
        self,
        *,
        tenant_id: str,
        model_id: str,
        dataset_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            key = (tenant_id, model_id)

            if key not in self._models:
                raise ImitationNotFound(
                    "Model not found"
                )

            dataset = self._get_dataset(
                tenant_id,
                dataset_id,
            )

            record = self._training_records[key]

            if dataset.robot_id != record.robot_id:
                raise ImitationConflict(
                    "Dataset robot does not match model"
                )

            observations, actions = self._prepare_data(
                dataset
            )

            normalizer = self._normalizers[key]

            normalized = normalizer.transform(
                observations
            )

            policy = self._models[key]
            policy.eval()

            tensor = torch.from_numpy(
                normalized
            ).to(self.config.device)

            with torch.no_grad():
                outputs = policy(
                    tensor
                ).cpu().numpy()

            if (
                self.config.action_space_type
                == ActionSpaceType.CONTINUOUS
            ):
                mse = float(
                    np.mean(
                        (outputs - actions) ** 2
                    )
                )

                mae = float(
                    np.mean(
                        np.abs(outputs - actions)
                    )
                )

                metrics = {
                    "mean_squared_error": mse,
                    "mean_absolute_error": mae,
                }

            else:
                predictions = np.argmax(
                    outputs,
                    axis=1,
                )

                accuracy = float(
                    np.mean(
                        predictions == actions
                    )
                )

                metrics = {
                    "action_accuracy": accuracy,
                }

            record.status = ModelStatus.EVALUATED

            self._log(
                tenant_id=tenant_id,
                action="model_evaluated",
                entity_id=model_id,
                details={
                    "dataset_id": dataset_id,
                    "sample_count": len(observations),
                    "metrics": metrics,
                },
            )

            return {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "sample_count": len(observations),
                "metrics": metrics,
                "evaluation_note": (
                    "Metrics describe action imitation "
                    "on the supplied dataset. "
                    "Use a separate held-out dataset "
                    "for unbiased generalization evaluation."
                ),
                "physical_deployment_approved": False,
            }

    # --------------------------------------------------------
    # DAgger DATA AGGREGATION
    # --------------------------------------------------------

    def aggregate_dagger_demonstrations(
        self,
        *,
        tenant_id: str,
        dataset_id: str,
        demonstrations: Sequence[ExpertDemonstration],
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        """
        Append externally labeled DAgger demonstrations.

        Typical workflow:
            1. Run the learner inside a simulator.
            2. Record visited observations.
            3. Request actions from a verified expert.
            4. Review safety and label quality.
            5. Append expert-labeled observations.
            6. Train a new policy version.

        This method does not execute the learner
        or query an external expert automatically.
        """

        if authorized is not True:
            raise ImitationPermissionDenied(
                "Authorized DAgger aggregation required"
            )

        with self._lock:
            dataset = self._get_dataset(
                tenant_id,
                dataset_id,
            )

            for demonstration in demonstrations:
                if demonstration.dataset_id != dataset_id:
                    raise ImitationConflict(
                        "Demonstration dataset mismatch"
                    )

                if demonstration.tenant_id != tenant_id:
                    raise ImitationPermissionDenied(
                        "Cross-tenant demonstrations forbidden"
                    )

                if demonstration.robot_id != dataset.robot_id:
                    raise ImitationConflict(
                        "Robot mismatch"
                    )

            result = self.add_demonstrations(
                demonstrations,
                authorized=True,
                actor_id=actor_id,
            )

            self._log(
                tenant_id=tenant_id,
                action="dagger_dataset_aggregated",
                entity_id=dataset_id,
                actor_id=actor_id,
                details={
                    "added_count": result["added_count"],
                },
            )

            return {
                "dataset_id": dataset_id,
                "algorithm": LearningAlgorithm.DAGGER.value,
                "added_demonstrations": (
                    result["added_count"]
                ),
                "total_demonstrations": len(
                    dataset.demonstrations
                ),
                "expert_labels_required": True,
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # HUMAN REVIEW
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
            raise ImitationPermissionDenied(
                "Authorized human approval required"
            )

        validate_id(reviewer_id, "reviewer_id")

        with self._lock:
            key = (tenant_id, model_id)

            record = self._training_records.get(key)

            if record is None:
                raise ImitationNotFound(
                    "Training record not found"
                )

            if record.status != ModelStatus.EVALUATED:
                raise ImitationConflict(
                    "Model must be evaluated before approval"
                )

            record.status = (
                ModelStatus.APPROVED_FOR_SIMULATION
            )

            record.reviewer_id = reviewer_id

            self._log(
                tenant_id=tenant_id,
                action="model_approved_for_simulation",
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
            record = self._training_records.get(
                (tenant_id, model_id)
            )

            if record is None:
                raise ImitationNotFound(
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
            raise ImitationPermissionDenied(
                "Authorized checkpoint export required"
            )

        with self._lock:
            key = (tenant_id, model_id)

            policy = self._models.get(key)

            if policy is None:
                raise ImitationNotFound(
                    "Model not found"
                )

            record = self._training_records[key]
            normalizer = self._normalizers[key]

            output_directory = Path(directory)
            output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            output = output_directory / (
                f"{tenant_id}_{model_id}.pt"
            )

            checkpoint = {
                "format_version": 1,
                "tenant_id": tenant_id,
                "model_id": model_id,
                "robot_id": record.robot_id,
                "dataset_id": record.dataset_id,
                "action_space_type": (
                    self.config.action_space_type.value
                ),
                "observation_dimension": (
                    self.config.observation_dimension
                ),
                "action_dimension": (
                    self.config.action_dimension
                ),
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

            record.checkpoint_path = str(
                output
            )

            self._log(
                tenant_id=tenant_id,
                action="checkpoint_saved",
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
        """
        Load a trusted, locally produced checkpoint.

        Never load checkpoints from untrusted sources.
        weights_only=True reduces unsafe deserialization
        exposure but does not validate model provenance.
        """

        if authorized is not True:
            raise ImitationPermissionDenied(
                "Authorized checkpoint loading required"
            )

        validate_id(tenant_id, "tenant_id")
        validate_id(model_id, "model_id")

        path = Path(checkpoint_path)

        if not path.is_file():
            raise ImitationNotFound(
                "Checkpoint file not found"
            )

        with self._lock:
            key = (tenant_id, model_id)

            if key in self._models:
                raise ImitationConflict(
                    "Model ID already exists"
                )

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            if not isinstance(checkpoint, dict):
                raise ImitationValidationError(
                    "Invalid checkpoint"
                )

            if (
                checkpoint.get("tenant_id") != tenant_id
                or checkpoint.get("model_id") != model_id
            ):
                raise ImitationPermissionDenied(
                    "Checkpoint identity mismatch"
                )

            if (
                checkpoint.get("observation_dimension")
                != self.config.observation_dimension
                or checkpoint.get("action_dimension")
                != self.config.action_dimension
                or checkpoint.get("action_space_type")
                != self.config.action_space_type.value
                or tuple(
                    checkpoint.get("hidden_dimensions", ())
                ) != self.config.hidden_dimensions
            ):
                raise ImitationValidationError(
                    "Checkpoint architecture mismatch"
                )

            policy = ImitationPolicyNetwork(
                self.config.observation_dimension,
                self.config.action_dimension,
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

            if (
                mean.shape
                != (self.config.observation_dimension,)
                or std.shape
                != (self.config.observation_dimension,)
                or not np.isfinite(mean).all()
                or not np.isfinite(std).all()
                or np.any(std <= 0)
            ):
                raise ImitationValidationError(
                    "Invalid checkpoint normalization"
                )

            normalizer = ObservationNormalizer(
                mean=mean,
                std=std,
            )

            saved_record = checkpoint[
                "training_record"
            ]

            record = TrainingRecord(
                training_id=saved_record["training_id"],
                tenant_id=tenant_id,
                robot_id=saved_record["robot_id"],
                dataset_id=saved_record["dataset_id"],
                model_id=model_id,
                algorithm=LearningAlgorithm(
                    saved_record["algorithm"]
                ),
                epochs_completed=saved_record[
                    "epochs_completed"
                ],
                training_losses=list(
                    saved_record["training_losses"]
                ),
                validation_losses=list(
                    saved_record["validation_losses"]
                ),
                best_validation_loss=saved_record[
                    "best_validation_loss"
                ],
                training_sample_count=saved_record[
                    "training_sample_count"
                ],
                validation_sample_count=saved_record[
                    "validation_sample_count"
                ],
                status=ModelStatus(
                    saved_record["status"]
                ),
                reviewer_id=saved_record.get(
                    "reviewer_id"
                ),
                checkpoint_path=str(path),
            )

            self._models[key] = policy
            self._normalizers[key] = normalizer
            self._training_records[key] = record

            self._log(
                tenant_id=tenant_id,
                action="checkpoint_loaded",
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

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
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
        validate_id(tenant_id, "tenant_id")

        output = Path(path)

        with self._lock:
            datasets = [
                dataset.to_dict()
                for (tenant, _), dataset
                in self._datasets.items()
                if tenant == tenant_id
            ]

            training_records = [
                record.to_dict()
                for (tenant, _), record
                in self._training_records.items()
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
            "training_records": training_records,
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
# SYNTHETIC EXPERT DEMONSTRATIONS
# ============================================================

def create_synthetic_navigation_demonstrations(
    *,
    tenant_id: str,
    robot_id: str,
    dataset_id: str,
    count: int = 500,
    seed: int = 42,
) -> list[ExpertDemonstration]:
    """
    Generate simple expert-labeled navigation examples.

    Observation:
        [robot_x, robot_y, goal_x, goal_y]

    Expert action:
        [normalized_velocity_x, normalized_velocity_y]

    These are synthetic simulation examples, not
    validated physical robot demonstrations.
    """

    validate_integer(
        count,
        "count",
        minimum=1,
    )

    rng = np.random.default_rng(
        seed
    )

    demonstrations = []

    for index in range(count):
        robot_position = rng.uniform(
            -10,
            10,
            size=2,
        )

        goal_position = rng.uniform(
            -10,
            10,
            size=2,
        )

        direction = (
            goal_position - robot_position
        )

        norm = np.linalg.norm(
            direction
        )

        if norm > 1e-8:
            action = direction / norm
        else:
            action = np.zeros(2)

        observation = (
            float(robot_position[0]),
            float(robot_position[1]),
            float(goal_position[0]),
            float(goal_position[1]),
        )

        expert_action = (
            float(action[0]),
            float(action[1]),
        )

        demonstrations.append(
            ExpertDemonstration(
                demonstration_id=(
                    f"demo_{index + 1:06d}"
                ),
                tenant_id=tenant_id,
                robot_id=robot_id,
                dataset_id=dataset_id,
                observation=observation,
                expert_action=expert_action,
                source=(
                    DemonstrationSource.SIMULATION_EXPERT
                ),
                episode_id=(
                    f"synthetic_episode_{index // 20:04d}"
                ),
                timestep=index % 20,
                reward=1.0,
                success=True,
                safety_verified=True,
                metadata={
                    "synthetic": True,
                    "simulation_only": True,
                },
            )
        )

    return demonstrations


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    config = ImitationLearningConfig(
        observation_dimension=4,
        action_dimension=2,
        action_space_type=(
            ActionSpaceType.CONTINUOUS
        ),
        hidden_dimensions=(128, 128),
        learning_rate=0.001,
        batch_size=64,
        epochs=50,
        validation_fraction=0.2,
        early_stopping_patience=10,
        minimum_demonstrations=20,
        device="cpu",
    )

    engine = RoboticsImitationLearningEngine(
        config
    )

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"
    dataset_id = "navigation_expert_dataset"
    model_id = "navigation_bc_policy_v1"

    print("\nCREATE DATASET")

    print(
        json.dumps(
            engine.create_dataset(
                tenant_id=tenant_id,
                robot_id=robot_id,
                dataset_id=dataset_id,
                name="Synthetic Navigation Expert Dataset",
                authorized=True,
                actor_id="training_admin",
            ),
            indent=2,
        )
    )

    print("\nGENERATE EXPERT DEMONSTRATIONS")

    demonstrations = (
        create_synthetic_navigation_demonstrations(
            tenant_id=tenant_id,
            robot_id=robot_id,
            dataset_id=dataset_id,
            count=500,
        )
    )

    ingestion = engine.add_demonstrations(
        demonstrations,
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

    print("\nTRAIN BEHAVIORAL CLONING")

    training = engine.train_behavioral_cloning(
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
                "epochs_completed": (
                    training["epochs_completed"]
                ),
                "best_validation_loss": (
                    training["best_validation_loss"]
                ),
                "status": training["status"],
            },
            indent=2,
        )
    )

    print("\nPREDICT ACTION")

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

    print("\nEVALUATE MODEL")

    evaluation = engine.evaluate_model(
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

    print("\nHUMAN REVIEW")

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
        "data/robotics/imitation_learning_report.json",
        tenant_id=tenant_id,
    )

    print(output)


if __name__ == "__main__":
    main()
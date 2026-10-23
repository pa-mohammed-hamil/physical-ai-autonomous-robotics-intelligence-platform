"""
Physical AI & Autonomous Robotics Intelligence Platform
File: ml/training/train_world_model.py

Neural World Model Training Pipeline

Features:
- Robot state-transition prediction
- Action-conditioned dynamics learning
- Latent state representation
- GRU-based temporal dynamics
- Future robot-state forecasting
- Multi-step trajectory prediction
- Reward and episode-termination prediction
- Episode-aware dataset splitting
- State/action normalization
- Sequence-based training
- Mixed-precision training
- Gradient clipping
- Early stopping
- Model checkpointing
- MAE, RMSE, and termination metrics
- Offline trajectory evaluation
- Simulation-only imagination rollouts
- Training metadata and model checksums

Dataset CSV columns:
    episode_id
    timestep
    state_1, state_2, ...
    action_1, action_2, ...
    next_state_1, next_state_2, ...
    reward
    done

Install:
    pip install torch numpy pandas scikit-learn

Train:
    python -m ml.training.train_world_model train \
        --data data/world_model/transitions.csv \
        --state-columns state_1 state_2 \
        --action-columns action_1 action_2 \
        --next-state-columns next_state_1 next_state_2 \
        --epochs 50 \
        --output artifacts/world_model

Evaluate:
    python -m ml.training.train_world_model evaluate \
        --checkpoint artifacts/world_model/best_model.pt \
        --data data/world_model/transitions.csv \
        --trusted-checkpoint

Safety:
    Offline and simulation use only.
    No physical robot control is implemented.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import tempfile
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import GroupShuffleSplit


LOGGER = logging.getLogger("train_world_model")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCHEMA_VERSION = "1.0.0"


# ============================================================
# Exceptions
# ============================================================

class WorldModelError(Exception):
    pass


class WorldModelDatasetError(WorldModelError):
    pass


class WorldModelTrainingError(WorldModelError):
    pass


class WorldModelArtifactError(WorldModelError):
    pass


# ============================================================
# Configuration
# ============================================================

@dataclass
class WorldModelConfig:
    data_path: str

    output_directory: str = "artifacts/world_model"

    episode_column: str = "episode_id"
    timestep_column: str = "timestep"

    state_columns: list[str] = field(default_factory=list)
    action_columns: list[str] = field(default_factory=list)
    next_state_columns: list[str] = field(default_factory=list)

    reward_column: str = "reward"
    done_column: str = "done"

    sequence_length: int = 16
    sequence_stride: int = 1

    latent_dimension: int = 128
    hidden_dimension: int = 256
    gru_layers: int = 2
    dropout: float = 0.1

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 10

    validation_size: float = 0.15
    test_size: float = 0.15

    state_loss_weight: float = 1.0
    reward_loss_weight: float = 0.2
    done_loss_weight: float = 0.2

    random_seed: int = 42
    num_workers: int = 0

    device: str = "auto"
    mixed_precision: bool = True

    max_rollout_steps: int = 100

    def validate(self) -> None:
        if not self.data_path:
            raise ValueError("Dataset path is required")

        if not self.state_columns:
            raise ValueError("State columns are required")

        if not self.action_columns:
            raise ValueError("Action columns are required")

        if len(self.state_columns) != len(self.next_state_columns):
            raise ValueError(
                "State and next-state dimensions must match"
            )

        if len(set(
            self.state_columns
            + self.action_columns
            + self.next_state_columns
        )) != (
            len(self.state_columns)
            + len(self.action_columns)
            + len(self.next_state_columns)
        ):
            raise ValueError(
                "State, action, and next-state columns must be distinct"
            )

        if self.sequence_length < 2:
            raise ValueError(
                "Sequence length must be at least 2"
            )

        if self.sequence_stride < 1:
            raise ValueError(
                "Sequence stride must be positive"
            )

        if self.latent_dimension < 1:
            raise ValueError(
                "Latent dimension must be positive"
            )

        if self.hidden_dimension < 1:
            raise ValueError(
                "Hidden dimension must be positive"
            )

        if self.gru_layers < 1:
            raise ValueError(
                "GRU layer count must be positive"
            )

        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError(
                "Epochs and batch size must be positive"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "Learning rate must be positive"
            )

        if not (
            0 < self.validation_size < 1
            and 0 < self.test_size < 1
            and self.validation_size + self.test_size < 1
        ):
            raise ValueError(
                "Invalid validation/test fractions"
            )

        if min(
            self.state_loss_weight,
            self.reward_loss_weight,
            self.done_loss_weight,
        ) < 0:
            raise ValueError(
                "Loss weights cannot be negative"
            )

        if self.state_loss_weight == 0:
            raise ValueError(
                "State prediction loss must be enabled"
            )

        if self.max_rollout_steps < 1:
            raise ValueError(
                "Maximum rollout steps must be positive"
            )


# ============================================================
# Utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise WorldModelError("CUDA is unavailable")

        if device.type == "mps" and not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise WorldModelError("MPS is unavailable")

        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def save_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                allow_nan=False,
            )

            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)

        os.replace(temporary, path)

    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


# ============================================================
# Normalization
# ============================================================

class ArrayNormalizer:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)

        if values.ndim != 2 or not np.isfinite(values).all():
            raise WorldModelDatasetError(
                "Normalizer requires finite 2D data"
            )

        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0)

        self.std = np.where(
            self.std < 1e-6,
            1.0,
            self.std,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise WorldModelError(
                "Normalizer has not been fitted"
            )

        return np.asarray(
            (values - self.mean) / self.std,
            dtype=np.float32,
        )

    def inverse_transform(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise WorldModelError(
                "Normalizer has not been fitted"
            )

        return values * self.std + self.mean

    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.std is None:
            raise WorldModelError(
                "Normalizer has not been fitted"
            )

        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ArrayNormalizer":
        normalizer = cls()

        normalizer.mean = np.asarray(
            payload["mean"],
            dtype=np.float64,
        )

        normalizer.std = np.asarray(
            payload["std"],
            dtype=np.float64,
        )

        return normalizer


# ============================================================
# Dataset Loading
# ============================================================

class WorldModelDatasetLoader:
    def __init__(self, config: WorldModelConfig) -> None:
        self.config = config

    def load(self) -> pd.DataFrame:
        path = Path(self.config.data_path)

        if not path.is_file():
            raise WorldModelDatasetError(
                f"Dataset not found: {path}"
            )

        if path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(path)

        elif path.suffix.lower() in {".parquet", ".pq"}:
            dataframe = pd.read_parquet(path)

        else:
            raise WorldModelDatasetError(
                "Supported formats: CSV and Parquet"
            )

        if dataframe.empty:
            raise WorldModelDatasetError(
                "Dataset is empty"
            )

        required = [
            self.config.episode_column,
            self.config.timestep_column,
            *self.config.state_columns,
            *self.config.action_columns,
            *self.config.next_state_columns,
            self.config.reward_column,
            self.config.done_column,
        ]

        if len(set(required)) != len(required):
            raise WorldModelDatasetError(
                "Duplicate configured columns"
            )

        missing = [
            name for name in required
            if name not in dataframe.columns
        ]

        if missing:
            raise WorldModelDatasetError(
                f"Missing dataset columns: {missing}"
            )

        if dataframe[required].isna().any().any():
            raise WorldModelDatasetError(
                "Required dataset columns contain null values"
            )

        numeric_columns = [
            *self.config.state_columns,
            *self.config.action_columns,
            *self.config.next_state_columns,
            self.config.reward_column,
            self.config.timestep_column,
        ]

        for name in numeric_columns:
            dataframe[name] = pd.to_numeric(
                dataframe[name],
                errors="coerce",
            )

            if not np.isfinite(
                dataframe[name].to_numpy(dtype=np.float64)
            ).all():
                raise WorldModelDatasetError(
                    f"Invalid numeric data in {name}"
                )

        done = dataframe[
            self.config.done_column
        ]

        if done.dtype == bool:
            dataframe[self.config.done_column] = done.astype(
                np.float32
            )

        else:
            done_numeric = pd.to_numeric(
                done,
                errors="coerce",
            )

            if not done_numeric.isin([0, 1]).all():
                raise WorldModelDatasetError(
                    "Done column must contain 0/1 or booleans"
                )

            dataframe[self.config.done_column] = (
                done_numeric.astype(np.float32)
            )

        dataframe = dataframe.sort_values(
            [
                self.config.episode_column,
                self.config.timestep_column,
            ],
            kind="stable",
        ).reset_index(drop=True)

        if dataframe.duplicated(
            [
                self.config.episode_column,
                self.config.timestep_column,
            ]
        ).any():
            raise WorldModelDatasetError(
                "Duplicate episode/timestep combinations"
            )

        LOGGER.info(
            "Loaded %d transitions from %d episodes",
            len(dataframe),
            dataframe[self.config.episode_column].nunique(),
        )

        return dataframe


# ============================================================
# Episode-Aware Split
# ============================================================

@dataclass
class WorldModelSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_world_model_dataset(
    dataframe: pd.DataFrame,
    config: WorldModelConfig,
) -> WorldModelSplits:
    groups = dataframe[
        config.episode_column
    ].astype(str).to_numpy()

    if len(np.unique(groups)) < 3:
        raise WorldModelDatasetError(
            "At least three episodes are required"
        )

    temporary_fraction = (
        config.validation_size + config.test_size
    )

    first = GroupShuffleSplit(
        n_splits=1,
        test_size=temporary_fraction,
        random_state=config.random_seed,
    )

    train_idx, temporary_idx = next(
        first.split(dataframe, groups=groups)
    )

    temporary = dataframe.iloc[
        temporary_idx
    ].reset_index(drop=True)

    temporary_groups = temporary[
        config.episode_column
    ].astype(str).to_numpy()

    if len(np.unique(temporary_groups)) < 2:
        raise WorldModelDatasetError(
            "Insufficient episodes for validation/test split"
        )

    second = GroupShuffleSplit(
        n_splits=1,
        test_size=(
            config.test_size / temporary_fraction
        ),
        random_state=config.random_seed + 1,
    )

    validation_idx, test_idx = next(
        second.split(
            temporary,
            groups=temporary_groups,
        )
    )

    return WorldModelSplits(
        train=dataframe.iloc[
            train_idx
        ].reset_index(drop=True),
        validation=temporary.iloc[
            validation_idx
        ].reset_index(drop=True),
        test=temporary.iloc[
            test_idx
        ].reset_index(drop=True),
    )


# ============================================================
# Sequence Dataset
# ============================================================

class RobotWorldModelDataset(Dataset):
    """
    Each item contains a contiguous trajectory segment.

    The model receives:
        states:      [T, state_dim]
        actions:     [T, action_dim]

    Targets:
        next_states: [T, state_dim]
        rewards:     [T]
        dones:       [T]

    Windows never cross episode boundaries or continue
    past a terminal transition.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        config: WorldModelConfig,
        state_normalizer: ArrayNormalizer,
        action_normalizer: ArrayNormalizer,
        reward_normalizer: ArrayNormalizer,
    ) -> None:
        self.config = config

        self.samples: list[
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]
        ] = []

        for _, episode in dataframe.groupby(
            config.episode_column,
            sort=False,
        ):
            episode = episode.sort_values(
                config.timestep_column
            )

            states = episode[
                config.state_columns
            ].to_numpy(dtype=np.float32)

            actions = episode[
                config.action_columns
            ].to_numpy(dtype=np.float32)

            next_states = episode[
                config.next_state_columns
            ].to_numpy(dtype=np.float32)

            rewards = episode[
                config.reward_column
            ].to_numpy(dtype=np.float32)

            dones = episode[
                config.done_column
            ].to_numpy(dtype=np.float32)

            states = state_normalizer.transform(
                states
            )

            next_states = state_normalizer.transform(
                next_states
            )

            actions = action_normalizer.transform(
                actions
            )

            rewards = reward_normalizer.transform(
                rewards.reshape(-1, 1)
            ).reshape(-1)

            length = config.sequence_length

            if len(states) < length:
                continue

            for start in range(
                0,
                len(states) - length + 1,
                config.sequence_stride,
            ):
                end = start + length

                # Earlier terminal transitions invalidate
                # the remainder of a sequence.
                if np.any(
                    dones[start:end - 1] > 0.5
                ):
                    continue

                self.samples.append(
                    (
                        states[start:end],
                        actions[start:end],
                        next_states[start:end],
                        rewards[start:end],
                        dones[start:end],
                    )
                )

        if not self.samples:
            raise WorldModelDatasetError(
                "No valid trajectory sequences generated"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        states, actions, next_states, rewards, dones = (
            self.samples[index]
        )

        return {
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "next_states": torch.from_numpy(next_states),
            "rewards": torch.from_numpy(rewards),
            "dones": torch.from_numpy(dones),
        }


# ============================================================
# Neural World Model
# ============================================================

class NeuralRobotWorldModel(nn.Module):
    """
    Action-conditioned latent recurrent world model.

    At each timestep:
        latent_state = encoder(state)
        fused = concat(latent_state, action)
        hidden = GRU(fused)
        predicted_next_state = decoder(hidden)
        predicted_reward = reward_head(hidden)
        predicted_done = sigmoid(done_head(hidden))

    This is a learned dynamics approximation, not a
    certified physics simulator.
    """

    def __init__(
        self,
        *,
        state_dimension: int,
        action_dimension: int,
        latent_dimension: int = 128,
        hidden_dimension: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.state_dimension = state_dimension
        self.action_dimension = action_dimension

        self.latent_dimension = latent_dimension
        self.hidden_dimension = hidden_dimension
        self.gru_layers = gru_layers

        self.state_encoder = nn.Sequential(
            nn.Linear(
                state_dimension,
                hidden_dimension,
            ),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
            nn.Linear(
                hidden_dimension,
                latent_dimension,
            ),
        )

        self.dynamics = nn.GRU(
            input_size=(
                latent_dimension + action_dimension
            ),
            hidden_size=hidden_dimension,
            num_layers=gru_layers,
            batch_first=True,
            dropout=(
                dropout if gru_layers > 1 else 0.0
            ),
        )

        self.state_decoder = nn.Sequential(
            nn.Linear(
                hidden_dimension,
                hidden_dimension,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dimension,
                state_dimension,
            ),
        )

        self.reward_head = nn.Sequential(
            nn.Linear(
                hidden_dimension,
                hidden_dimension // 2 or 1,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dimension // 2 or 1,
                1,
            ),
        )

        self.done_head = nn.Linear(
            hidden_dimension,
            1,
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if states.ndim != 3 or actions.ndim != 3:
            raise ValueError(
                "States/actions must have [B, T, D] shape"
            )

        if states.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "State/action batch and time dimensions differ"
            )

        latent = self.state_encoder(
            states
        )

        fused = torch.cat(
            [latent, actions],
            dim=-1,
        )

        dynamics_output, next_hidden = (
            self.dynamics(
                fused,
                hidden,
            )
        )

        # Residual dynamics prediction.
        delta = self.state_decoder(
            dynamics_output
        )

        next_states = states + delta

        rewards = self.reward_head(
            dynamics_output
        ).squeeze(-1)

        done_logits = self.done_head(
            dynamics_output
        ).squeeze(-1)

        return {
            "next_states": next_states,
            "rewards": rewards,
            "done_logits": done_logits,
            "hidden": next_hidden,
            "latent_states": latent,
        }


# ============================================================
# Loss
# ============================================================

class WorldModelLoss(nn.Module):
    def __init__(
        self,
        config: WorldModelConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.state_loss = nn.SmoothL1Loss()
        self.reward_loss = nn.SmoothL1Loss()

        self.done_loss = (
            nn.BCEWithLogitsLoss()
        )

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        state_loss = self.state_loss(
            prediction["next_states"],
            batch["next_states"],
        )

        reward_loss = self.reward_loss(
            prediction["rewards"],
            batch["rewards"],
        )

        done_loss = self.done_loss(
            prediction["done_logits"],
            batch["dones"],
        )

        total = (
            self.config.state_loss_weight * state_loss
            + self.config.reward_loss_weight * reward_loss
            + self.config.done_loss_weight * done_loss
        )

        return {
            "total": total,
            "state": state_loss,
            "reward": reward_loss,
            "done": done_loss,
        }


# ============================================================
# Evaluation
# ============================================================

def calculate_world_model_metrics(
    actual_states: np.ndarray,
    predicted_states: np.ndarray,
    actual_rewards: np.ndarray,
    predicted_rewards: np.ndarray,
    actual_dones: np.ndarray,
    predicted_done_probabilities: np.ndarray,
) -> dict[str, Any]:
    if actual_states.shape != predicted_states.shape:
        raise WorldModelError(
            "State prediction shape mismatch"
        )

    state_error = (
        predicted_states - actual_states
    )

    reward_error = (
        predicted_rewards - actual_rewards
    )

    done_predictions = (
        predicted_done_probabilities >= 0.5
    )

    done_targets = (
        actual_dones >= 0.5
    )

    true_positive = int(
        np.sum(done_predictions & done_targets)
    )

    false_positive = int(
        np.sum(done_predictions & ~done_targets)
    )

    false_negative = int(
        np.sum(~done_predictions & done_targets)
    )

    precision = (
        true_positive
        / max(true_positive + false_positive, 1)
    )

    recall = (
        true_positive
        / max(true_positive + false_negative, 1)
    )

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    return {
        "state_mae": float(
            np.mean(np.abs(state_error))
        ),
        "state_rmse": float(
            np.sqrt(np.mean(state_error ** 2))
        ),
        "reward_mae": float(
            np.mean(np.abs(reward_error))
        ),
        "reward_rmse": float(
            np.sqrt(np.mean(reward_error ** 2))
        ),
        "done_accuracy": float(
            np.mean(
                done_predictions == done_targets
            )
        ),
        "done_precision": float(precision),
        "done_recall": float(recall),
        "done_f1": float(f1),
        "transition_count": int(
            actual_dones.size
        ),
    }


# ============================================================
# Training Result
# ============================================================

@dataclass
class WorldModelTrainingResult:
    run_id: str
    status: str

    train_sequences: int
    validation_sequences: int
    test_sequences: int

    epochs_completed: int
    best_validation_loss: float

    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]

    artifact_paths: dict[str, str]

    started_at: str
    completed_at: str
    duration_seconds: float

    simulation_only: bool = True


# ============================================================
# Training Pipeline
# ============================================================

class WorldModelTrainingPipeline:
    def __init__(
        self,
        config: WorldModelConfig,
    ) -> None:
        config.validate()

        self.config = config
        self.run_id = str(uuid.uuid4())

        self.device = resolve_device(
            config.device
        )

        self.output_directory = Path(
            config.output_directory
        ).resolve()

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.state_normalizer = ArrayNormalizer()
        self.action_normalizer = ArrayNormalizer()
        self.reward_normalizer = ArrayNormalizer()

        self.model: NeuralRobotWorldModel | None = None

    def _prepare(
        self,
    ) -> tuple[
        RobotWorldModelDataset,
        RobotWorldModelDataset,
        RobotWorldModelDataset,
    ]:
        dataframe = WorldModelDatasetLoader(
            self.config
        ).load()

        splits = split_world_model_dataset(
            dataframe,
            self.config,
        )

        # Include both current and next states when fitting
        # the state scaler, using training episodes only.
        training_states = np.concatenate(
            [
                splits.train[
                    self.config.state_columns
                ].to_numpy(dtype=np.float32),
                splits.train[
                    self.config.next_state_columns
                ].to_numpy(dtype=np.float32),
            ],
            axis=0,
        )

        training_actions = splits.train[
            self.config.action_columns
        ].to_numpy(dtype=np.float32)

        training_rewards = splits.train[
            self.config.reward_column
        ].to_numpy(
            dtype=np.float32
        ).reshape(-1, 1)

        self.state_normalizer.fit(
            training_states
        )

        self.action_normalizer.fit(
            training_actions
        )

        self.reward_normalizer.fit(
            training_rewards
        )

        def build(
            frame: pd.DataFrame,
        ) -> RobotWorldModelDataset:
            return RobotWorldModelDataset(
                frame,
                self.config,
                self.state_normalizer,
                self.action_normalizer,
                self.reward_normalizer,
            )

        return (
            build(splits.train),
            build(splits.validation),
            build(splits.test),
        )

    def _build_model(self) -> NeuralRobotWorldModel:
        return NeuralRobotWorldModel(
            state_dimension=len(
                self.config.state_columns
            ),
            action_dimension=len(
                self.config.action_columns
            ),
            latent_dimension=(
                self.config.latent_dimension
            ),
            hidden_dimension=(
                self.config.hidden_dimension
            ),
            gru_layers=self.config.gru_layers,
            dropout=self.config.dropout,
        ).to(self.device)

    def _loader(
        self,
        dataset: RobotWorldModelDataset,
        *,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=(
                self.device.type == "cuda"
            ),
        )

    def _move_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            name: value.to(
                self.device,
                non_blocking=True,
            )
            for name, value in batch.items()
        }

    def _evaluate(
        self,
        loader: DataLoader,
    ) -> tuple[float, dict[str, Any]]:
        assert self.model is not None

        self.model.eval()

        criterion = WorldModelLoss(
            self.config
        )

        total_loss = 0.0
        total_samples = 0

        actual_states = []
        predicted_states = []

        actual_rewards = []
        predicted_rewards = []

        actual_dones = []
        predicted_dones = []

        with torch.no_grad():
            for raw_batch in loader:
                batch = self._move_batch(
                    raw_batch
                )

                prediction = self.model(
                    batch["states"],
                    batch["actions"],
                )

                losses = criterion(
                    prediction,
                    batch,
                )

                batch_size = len(
                    batch["states"]
                )

                total_loss += (
                    losses["total"].item()
                    * batch_size
                )

                total_samples += batch_size

                actual_states.append(
                    batch["next_states"].cpu().numpy()
                )

                predicted_states.append(
                    prediction["next_states"].cpu().numpy()
                )

                actual_rewards.append(
                    batch["rewards"].cpu().numpy()
                )

                predicted_rewards.append(
                    prediction["rewards"].cpu().numpy()
                )

                actual_dones.append(
                    batch["dones"].cpu().numpy()
                )

                predicted_dones.append(
                    torch.sigmoid(
                        prediction["done_logits"]
                    ).cpu().numpy()
                )

        if total_samples == 0:
            raise WorldModelError(
                "Evaluation loader is empty"
            )

        actual_states_array = np.concatenate(
            actual_states,
            axis=0,
        )

        predicted_states_array = np.concatenate(
            predicted_states,
            axis=0,
        )

        actual_rewards_array = np.concatenate(
            actual_rewards,
            axis=0,
        )

        predicted_rewards_array = np.concatenate(
            predicted_rewards,
            axis=0,
        )

        actual_states_array = (
            self.state_normalizer.inverse_transform(
                actual_states_array
            )
        )

        predicted_states_array = (
            self.state_normalizer.inverse_transform(
                predicted_states_array
            )
        )

        actual_rewards_array = (
            self.reward_normalizer.inverse_transform(
                actual_rewards_array[..., None]
            )[..., 0]
        )

        predicted_rewards_array = (
            self.reward_normalizer.inverse_transform(
                predicted_rewards_array[..., None]
            )[..., 0]
        )

        metrics = calculate_world_model_metrics(
            actual_states_array,
            predicted_states_array,
            actual_rewards_array,
            predicted_rewards_array,
            np.concatenate(
                actual_dones,
                axis=0,
            ),
            np.concatenate(
                predicted_dones,
                axis=0,
            ),
        )

        return (
            total_loss / total_samples,
            metrics,
        )

    def _save_checkpoint(
        self,
        path: Path,
    ) -> None:
        assert self.model is not None

        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "model_state_dict": (
                self.model.state_dict()
            ),
            "model_config": {
                "state_dimension": len(
                    self.config.state_columns
                ),
                "action_dimension": len(
                    self.config.action_columns
                ),
                "latent_dimension": (
                    self.config.latent_dimension
                ),
                "hidden_dimension": (
                    self.config.hidden_dimension
                ),
                "gru_layers": (
                    self.config.gru_layers
                ),
                "dropout": (
                    self.config.dropout
                ),
            },
            "state_normalizer": (
                self.state_normalizer.to_dict()
            ),
            "action_normalizer": (
                self.action_normalizer.to_dict()
            ),
            "reward_normalizer": (
                self.reward_normalizer.to_dict()
            ),
            "state_columns": (
                self.config.state_columns
            ),
            "action_columns": (
                self.config.action_columns
            ),
            "next_state_columns": (
                self.config.next_state_columns
            ),
            "sequence_length": (
                self.config.sequence_length
            ),
            "max_rollout_steps": (
                self.config.max_rollout_steps
            ),
            "simulation_only": True,
        }

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            checkpoint,
            path,
        )

    def train(self) -> WorldModelTrainingResult:
        started_at = utc_now()
        started = time.perf_counter()

        set_random_seed(
            self.config.random_seed
        )

        train_dataset, validation_dataset, test_dataset = (
            self._prepare()
        )

        train_loader = self._loader(
            train_dataset,
            shuffle=True,
        )

        validation_loader = self._loader(
            validation_dataset,
            shuffle=False,
        )

        test_loader = self._loader(
            test_dataset,
            shuffle=False,
        )

        self.model = self._build_model()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs,
            )
        )

        criterion = WorldModelLoss(
            self.config
        )

        use_amp = (
            self.config.mixed_precision
            and self.device.type == "cuda"
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_amp,
        )

        best_validation_loss = float("inf")
        best_state = None

        patience_counter = 0

        history = {
            "training_loss": [],
            "validation_loss": [],
        }

        for epoch in range(self.config.epochs):
            self.model.train()

            total_loss = 0.0
            total_samples = 0

            for raw_batch in train_loader:
                batch = self._move_batch(
                    raw_batch
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=use_amp,
                ):
                    prediction = self.model(
                        batch["states"],
                        batch["actions"],
                    )

                    losses = criterion(
                        prediction,
                        batch,
                    )

                    loss = losses["total"]

                scaler.scale(loss).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                batch_size = len(
                    batch["states"]
                )

                total_loss += (
                    loss.item() * batch_size
                )

                total_samples += batch_size

            training_loss = (
                total_loss
                / max(total_samples, 1)
            )

            validation_loss, _ = self._evaluate(
                validation_loader
            )

            scheduler.step()

            history["training_loss"].append(
                training_loss
            )

            history["validation_loss"].append(
                validation_loss
            )

            LOGGER.info(
                "Epoch %d/%d | train=%.6f | val=%.6f",
                epoch + 1,
                self.config.epochs,
                training_loss,
                validation_loss,
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss

                best_state = copy.deepcopy(
                    self.model.state_dict()
                )

                patience_counter = 0

                self._save_checkpoint(
                    self.output_directory
                    / "best_model.pt"
                )

            else:
                patience_counter += 1

            if (
                patience_counter
                >= self.config.early_stopping_patience
            ):
                LOGGER.info(
                    "Early stopping at epoch %d",
                    epoch + 1,
                )
                break

        if best_state is None:
            raise WorldModelTrainingError(
                "No valid checkpoint produced"
            )

        self.model.load_state_dict(
            best_state
        )

        _, validation_metrics = self._evaluate(
            validation_loader
        )

        _, test_metrics = self._evaluate(
            test_loader
        )

        checkpoint_path = (
            self.output_directory
            / "best_model.pt"
        )

        metadata_path = (
            self.output_directory
            / "training_metadata.json"
        )

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "model_name": "neural-robot-world-model",
            "created_at": utc_now(),
            "configuration": asdict(
                self.config
            ),
            "epochs_completed": len(
                history["training_loss"]
            ),
            "best_validation_loss": (
                best_validation_loss
            ),
            "history": history,
            "validation_metrics": (
                validation_metrics
            ),
            "test_metrics": test_metrics,
            "artifact": {
                "filename": checkpoint_path.name,
                "sha256": sha256_file(
                    checkpoint_path
                ),
                "size_bytes": (
                    checkpoint_path.stat().st_size
                ),
            },
            "safety": {
                "simulation_only": True,
                "physical_execution_authorized": False,
                "safety_interlock_bypass_authorized": False,
                "emergency_stop_reset_authorized": False,
            },
        }

        save_json_atomic(
            metadata_path,
            metadata,
        )

        result = WorldModelTrainingResult(
            run_id=self.run_id,
            status="completed",
            train_sequences=len(train_dataset),
            validation_sequences=len(validation_dataset),
            test_sequences=len(test_dataset),
            epochs_completed=len(
                history["training_loss"]
            ),
            best_validation_loss=(
                best_validation_loss
            ),
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            artifact_paths={
                "checkpoint": str(checkpoint_path),
                "metadata": str(metadata_path),
            },
            started_at=started_at,
            completed_at=utc_now(),
            duration_seconds=round(
                time.perf_counter() - started,
                4,
            ),
        )

        save_json_atomic(
            self.output_directory
            / "training_result.json",
            asdict(result),
        )

        LOGGER.info(
            "World model trained | state MAE=%.6f | "
            "reward MAE=%.6f",
            test_metrics["state_mae"],
            test_metrics["reward_mae"],
        )

        return result


# ============================================================
# Simulation-Only World Model Inference
# ============================================================

class TrainedRobotWorldModel:
    """
    Load a locally trusted checkpoint and perform
    offline action-conditioned state prediction.

    Predicted trajectories must not be treated as
    validated physical safety envelopes.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        trusted_checkpoint: bool = False,
        expected_sha256: str | None = None,
    ) -> None:
        if not trusted_checkpoint:
            raise WorldModelArtifactError(
                "Explicit trusted_checkpoint=True required"
            )

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.is_file():
            raise WorldModelArtifactError(
                "Checkpoint not found"
            )

        if expected_sha256 is not None:
            if sha256_file(checkpoint_path) != expected_sha256:
                raise WorldModelArtifactError(
                    "Checkpoint checksum mismatch"
                )

        self.device = resolve_device(
            device
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint.get("simulation_only") is not True:
            raise WorldModelArtifactError(
                "Checkpoint lacks simulation-only restriction"
            )

        self.state_columns = list(
            checkpoint["state_columns"]
        )

        self.action_columns = list(
            checkpoint["action_columns"]
        )

        self.sequence_length = int(
            checkpoint["sequence_length"]
        )

        self.max_rollout_steps = int(
            checkpoint["max_rollout_steps"]
        )

        self.state_normalizer = (
            ArrayNormalizer.from_dict(
                checkpoint["state_normalizer"]
            )
        )

        self.action_normalizer = (
            ArrayNormalizer.from_dict(
                checkpoint["action_normalizer"]
            )
        )

        self.reward_normalizer = (
            ArrayNormalizer.from_dict(
                checkpoint["reward_normalizer"]
            )
        )

        self.model = NeuralRobotWorldModel(
            **checkpoint["model_config"]
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def predict_transition(
        self,
        *,
        state: Mapping[str, float],
        action: Mapping[str, float],
    ) -> dict[str, Any]:
        """
        One-step prediction with a fresh recurrent state.

        For history-aware inference, use predict_trajectory
        or implement a session-scoped recurrent state.
        """
        state_array = np.asarray(
            [
                float(state[name])
                for name in self.state_columns
            ],
            dtype=np.float32,
        )

        action_array = np.asarray(
            [
                float(action[name])
                for name in self.action_columns
            ],
            dtype=np.float32,
        )

        if not (
            np.isfinite(state_array).all()
            and np.isfinite(action_array).all()
        ):
            raise WorldModelError(
                "State/action values must be finite"
            )

        normalized_state = (
            self.state_normalizer.transform(
                state_array.reshape(1, -1)
            )
        )

        normalized_action = (
            self.action_normalizer.transform(
                action_array.reshape(1, -1)
            )
        )

        states = torch.tensor(
            normalized_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        actions = torch.tensor(
            normalized_action,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(
                states,
                actions,
            )

        next_state = (
            self.state_normalizer.inverse_transform(
                prediction["next_states"]
                .cpu()
                .numpy()[0]
            )[0]
        )

        reward = (
            self.reward_normalizer.inverse_transform(
                prediction["rewards"]
                .cpu()
                .numpy()
                .reshape(-1, 1)
            )[0, 0]
        )

        done_probability = float(
            torch.sigmoid(
                prediction["done_logits"]
            ).cpu().item()
        )

        return {
            "predicted_next_state": {
                name: float(value)
                for name, value in zip(
                    self.state_columns,
                    next_state,
                )
            },
            "predicted_reward": float(reward),
            "termination_probability": done_probability,
            "simulation_only": True,
            "physical_execution_authorized": False,
            "created_at": utc_now(),
        }

    def imagine_rollout(
        self,
        *,
        initial_state: Mapping[str, float],
        action_sequence: Sequence[Mapping[str, float]],
        termination_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Generate a simulated trajectory using planned
        actions. No actions are sent to any robot.

        The recurrent hidden state is preserved between
        simulated transitions.
        """
        if not 0 <= termination_threshold <= 1:
            raise ValueError(
                "Invalid termination threshold"
            )

        if len(action_sequence) > self.max_rollout_steps:
            raise WorldModelError(
                "Rollout exceeds maximum length"
            )

        state = np.asarray(
            [
                float(initial_state[name])
                for name in self.state_columns
            ],
            dtype=np.float32,
        )

        if not np.isfinite(state).all():
            raise WorldModelError(
                "Initial state must be finite"
            )

        normalized_state = (
            self.state_normalizer.transform(
                state.reshape(1, -1)
            )
        )

        hidden = None
        trajectory = []

        self.model.eval()

        with torch.no_grad():
            for step, action in enumerate(
                action_sequence
            ):
                action_values = np.asarray(
                    [
                        float(action[name])
                        for name in self.action_columns
                    ],
                    dtype=np.float32,
                )

                if not np.isfinite(
                    action_values
                ).all():
                    raise WorldModelError(
                        "Rollout action contains invalid values"
                    )

                normalized_action = (
                    self.action_normalizer.transform(
                        action_values.reshape(1, -1)
                    )
                )

                state_tensor = torch.tensor(
                    normalized_state,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                action_tensor = torch.tensor(
                    normalized_action,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                prediction = self.model(
                    state_tensor,
                    action_tensor,
                    hidden,
                )

                hidden = prediction[
                    "hidden"
                ]

                normalized_state = (
                    prediction["next_states"]
                    .cpu()
                    .numpy()[0]
                )

                state = (
                    self.state_normalizer.inverse_transform(
                        normalized_state
                    )[0]
                )

                reward = (
                    self.reward_normalizer.inverse_transform(
                        prediction["rewards"]
                        .cpu()
                        .numpy()
                        .reshape(-1, 1)
                    )[0, 0]
                )

                done_probability = float(
                    torch.sigmoid(
                        prediction["done_logits"]
                    ).cpu().item()
                )

                trajectory.append(
                    {
                        "step": step,
                        "state": {
                            name: float(value)
                            for name, value in zip(
                                self.state_columns,
                                state,
                            )
                        },
                        "predicted_reward": float(reward),
                        "termination_probability": (
                            done_probability
                        ),
                    }
                )

                if done_probability >= termination_threshold:
                    break

        return {
            "trajectory": trajectory,
            "steps_generated": len(trajectory),
            "simulation_only": True,
            "physical_execution_authorized": False,
            "warning": (
                "Learned dynamics can diverge from real "
                "robot behavior over multi-step rollouts."
            ),
        }


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a neural robot world model"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser("train")

    train_parser.add_argument(
        "--data",
        required=True,
    )

    train_parser.add_argument(
        "--state-columns",
        nargs="+",
        required=True,
    )

    train_parser.add_argument(
        "--action-columns",
        nargs="+",
        required=True,
    )

    train_parser.add_argument(
        "--next-state-columns",
        nargs="+",
        required=True,
    )

    train_parser.add_argument(
        "--output",
        default="artifacts/world_model",
    )

    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    train_parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
    )

    train_parser.add_argument(
        "--latent-dimension",
        type=int,
        default=128,
    )

    train_parser.add_argument(
        "--hidden-dimension",
        type=int,
        default=256,
    )

    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    train_parser.add_argument(
        "--device",
        default="auto",
    )

    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate"
    )

    evaluate_parser.add_argument(
        "--checkpoint",
        required=True,
    )

    evaluate_parser.add_argument(
        "--data",
        required=True,
    )

    evaluate_parser.add_argument(
        "--device",
        default="auto",
    )

    evaluate_parser.add_argument(
        "--trusted-checkpoint",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        config = WorldModelConfig(
            data_path=args.data,
            output_directory=args.output,
            state_columns=args.state_columns,
            action_columns=args.action_columns,
            next_state_columns=args.next_state_columns,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            latent_dimension=args.latent_dimension,
            hidden_dimension=args.hidden_dimension,
            learning_rate=args.learning_rate,
            device=args.device,
            random_seed=args.seed,
        )

        result = WorldModelTrainingPipeline(
            config
        ).train()

        print(
            json.dumps(
                asdict(result),
                indent=2,
                allow_nan=False,
            )
        )

        return

    if args.command == "evaluate":
        if not args.trusted_checkpoint:
            raise WorldModelArtifactError(
                "Use --trusted-checkpoint for trusted artifacts"
            )

        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        config = WorldModelConfig(
            data_path=args.data,
            state_columns=checkpoint[
                "state_columns"
            ],
            action_columns=checkpoint[
                "action_columns"
            ],
            next_state_columns=checkpoint[
                "next_state_columns"
            ],
            sequence_length=checkpoint[
                "sequence_length"
            ],
        )

        dataframe = WorldModelDatasetLoader(
            config
        ).load()

        state_normalizer = ArrayNormalizer.from_dict(
            checkpoint["state_normalizer"]
        )

        action_normalizer = ArrayNormalizer.from_dict(
            checkpoint["action_normalizer"]
        )

        reward_normalizer = ArrayNormalizer.from_dict(
            checkpoint["reward_normalizer"]
        )

        dataset = RobotWorldModelDataset(
            dataframe,
            config,
            state_normalizer,
            action_normalizer,
            reward_normalizer,
        )

        device = resolve_device(
            args.device
        )

        model = NeuralRobotWorldModel(
            **checkpoint["model_config"]
        ).to(device)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
        )

        actual_states = []
        predicted_states = []

        actual_rewards = []
        predicted_rewards = []

        actual_dones = []
        predicted_dones = []

        with torch.no_grad():
            for batch in loader:
                states = batch[
                    "states"
                ].to(device)

                actions = batch[
                    "actions"
                ].to(device)

                prediction = model(
                    states,
                    actions,
                )

                actual_states.append(
                    batch["next_states"].numpy()
                )

                predicted_states.append(
                    prediction["next_states"]
                    .cpu()
                    .numpy()
                )

                actual_rewards.append(
                    batch["rewards"].numpy()
                )

                predicted_rewards.append(
                    prediction["rewards"]
                    .cpu()
                    .numpy()
                )

                actual_dones.append(
                    batch["dones"].numpy()
                )

                predicted_dones.append(
                    torch.sigmoid(
                        prediction["done_logits"]
                    ).cpu().numpy()
                )

        true_states = state_normalizer.inverse_transform(
            np.concatenate(actual_states, axis=0)
        )

        forecast_states = state_normalizer.inverse_transform(
            np.concatenate(predicted_states, axis=0)
        )

        true_rewards = reward_normalizer.inverse_transform(
            np.concatenate(
                actual_rewards,
                axis=0,
            )[..., None]
        )[..., 0]

        forecast_rewards = reward_normalizer.inverse_transform(
            np.concatenate(
                predicted_rewards,
                axis=0,
            )[..., None]
        )[..., 0]

        metrics = calculate_world_model_metrics(
            true_states,
            forecast_states,
            true_rewards,
            forecast_rewards,
            np.concatenate(
                actual_dones,
                axis=0,
            ),
            np.concatenate(
                predicted_dones,
                axis=0,
            ),
        )

        print(
            json.dumps(
                {
                    "metrics": metrics,
                    "evaluation_scope": "provided_dataset",
                    "simulation_only": True,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
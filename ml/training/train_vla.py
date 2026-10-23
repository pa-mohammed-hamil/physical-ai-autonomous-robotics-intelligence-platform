"""
Physical AI & Autonomous Robotics Intelligence Platform
File: ml/training/train_vla.py

Vision-Language-Action (VLA) Model Training Pipeline

Features:
- Vision-language-action imitation learning
- Camera image + natural-language instruction inputs
- Robot state and proprioception inputs
- Continuous action-vector prediction
- Transformer-based multimodal fusion
- Frozen or fine-tuned vision encoder
- Text tokenization and embeddings
- Dataset validation and preprocessing
- Episode-aware train/validation/test splitting
- Image augmentation
- Action normalization
- Mixed-precision training
- Gradient clipping and accumulation
- Learning-rate scheduling
- Early stopping
- Model checkpointing
- MAE, RMSE and action accuracy evaluation
- Training metadata and SHA-256 checksums
- Simulation-only action prediction
- CLI training and evaluation

Dataset CSV example:
    episode_id,image_path,instruction,joint_1,joint_2,action_1,action_2
    ep001,images/001.jpg,move toward object,0.1,0.2,0.03,0.04
    ep001,images/002.jpg,move toward object,0.2,0.3,0.04,0.05
    ep002,images/003.jpg,move away,0.3,0.4,-0.02,-0.01

Install:
    pip install torch torchvision pandas numpy scikit-learn
    pip install pillow

Training:
    python -m ml.training.train_vla train \
        --data data/vla/demonstrations.csv \
        --image-root data/vla \
        --state-columns joint_1 joint_2 \
        --action-columns action_1 action_2 \
        --epochs 50 \
        --output artifacts/vla

Evaluation:
    python -m ml.training.train_vla evaluate \
        --checkpoint artifacts/vla/best_model.pt \
        --data data/vla/demonstrations.csv \
        --image-root data/vla

Safety:
    This model predicts actions for offline analysis
    and simulation only. It never executes commands
    on physical robots, overrides safety controllers,
    or resets emergency stops.
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
import re
import tempfile
import time
import uuid

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from PIL import Image

from sklearn.model_selection import GroupShuffleSplit

from torchvision import transforms
from torchvision.models import resnet18


LOGGER = logging.getLogger("ml.training.train_vla")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCHEMA_VERSION = "1.0.0"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]

MAX_DATASET_ROWS = 2_000_000
MAX_VOCAB_SIZE = 50_000
MAX_ACTION_DIMENSION = 128
MAX_STATE_DIMENSION = 512


# ============================================================
# Exceptions
# ============================================================

class VLATrainingError(Exception):
    pass


class VLADatasetError(VLATrainingError):
    pass


class VLAModelError(VLATrainingError):
    pass


class VLAEvaluationError(VLATrainingError):
    pass


class VLAArtifactError(VLATrainingError):
    pass


# ============================================================
# Enumerations
# ============================================================

class TrainingStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class VLAModelArchitecture(str, Enum):
    TRANSFORMER = "transformer"


class ActionRepresentation(str, Enum):
    CONTINUOUS = "continuous"


class ExecutionEnvironment(str, Enum):
    OFFLINE = "offline"
    SIMULATION = "simulation"


# ============================================================
# Configuration
# ============================================================

@dataclass
class VLATrainingConfig:
    data_path: str

    image_root: str = ""

    output_directory: str = "artifacts/vla"

    image_column: str = "image_path"
    instruction_column: str = "instruction"
    episode_column: str = "episode_id"

    state_columns: list[str] = field(
        default_factory=list
    )

    action_columns: list[str] = field(
        default_factory=list
    )

    image_size: int = 224

    max_text_length: int = 48
    vocabulary_size: int = 20_000
    minimum_token_frequency: int = 1

    embedding_dimension: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 4

    feedforward_dimension: int = 512
    dropout: float = 0.1

    freeze_vision_encoder: bool = False

    epochs: int = 50
    batch_size: int = 32

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    early_stopping_patience: int = 10

    validation_size: float = 0.15
    test_size: float = 0.15

    random_seed: int = 42

    num_workers: int = 0
    device: str = "auto"

    mixed_precision: bool = True
    image_augmentation: bool = True

    action_tolerance: float = 0.1

    execution_environment: ExecutionEnvironment = (
        ExecutionEnvironment.SIMULATION
    )

    experiment_name: str = "robot_vla_imitation_learning"

    def validate(self) -> None:
        if not self.data_path:
            raise VLADatasetError(
                "Training dataset path is required"
            )

        if not self.action_columns:
            raise VLADatasetError(
                "At least one action column is required"
            )

        if len(self.action_columns) > MAX_ACTION_DIMENSION:
            raise VLADatasetError(
                "Action dimension exceeds limit"
            )

        if len(self.state_columns) > MAX_STATE_DIMENSION:
            raise VLADatasetError(
                "State dimension exceeds limit"
            )

        if set(self.state_columns) & set(self.action_columns):
            raise VLADatasetError(
                "State and action columns overlap"
            )

        if self.image_size < 32:
            raise ValueError(
                "Image size must be at least 32"
            )

        if self.max_text_length < 4:
            raise ValueError(
                "Maximum text length must be at least 4"
            )

        if not 0 < self.vocabulary_size <= MAX_VOCAB_SIZE:
            raise ValueError(
                "Invalid vocabulary size"
            )

        if self.embedding_dimension < 16:
            raise ValueError(
                "Embedding dimension is too small"
            )

        if (
            self.embedding_dimension
            % self.transformer_heads != 0
        ):
            raise ValueError(
                "Embedding dimension must be divisible "
                "by transformer head count"
            )

        if not (
            0 < self.validation_size < 1
            and 0 < self.test_size < 1
            and self.validation_size + self.test_size < 1
        ):
            raise ValueError(
                "Invalid dataset split fractions"
            )

        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError(
                "Epochs and batch size must be positive"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "Learning rate must be positive"
            )

        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "Gradient accumulation steps must be positive"
            )

        if self.action_tolerance <= 0:
            raise ValueError(
                "Action tolerance must be positive"
            )

        if (
            self.execution_environment
            not in {
                ExecutionEnvironment.OFFLINE,
                ExecutionEnvironment.SIMULATION,
            }
        ):
            raise ValueError(
                "Only offline and simulation environments "
                "are supported"
            )


# ============================================================
# Utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_run_id() -> str:
    return str(uuid.uuid4())


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)

        if (
            device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise VLAModelError(
                "CUDA is unavailable"
            )

        if (
            device.type == "mps"
            and not (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
        ):
            raise VLAModelError(
                "MPS is unavailable"
            )

        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def calculate_sha256(path: str | Path) -> str:
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

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

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

            temporary_path = Path(stream.name)

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def safe_image_path(
    image_root: Path,
    relative_path: str,
) -> Path:
    image_root = image_root.resolve()

    path = (
        image_root / relative_path
    ).resolve()

    if not path.is_relative_to(image_root):
        raise VLADatasetError(
            "Image path escapes dataset root"
        )

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise VLADatasetError(
            f"Unsupported image extension: {path.suffix}"
        )

    if not path.is_file():
        raise VLADatasetError(
            f"Image not found: {path}"
        )

    return path


# ============================================================
# Text Tokenizer
# ============================================================

class VLATokenizer:
    def __init__(
        self,
        *,
        vocabulary_size: int = 20_000,
        minimum_frequency: int = 1,
        max_length: int = 48,
    ) -> None:
        self.vocabulary_size = vocabulary_size
        self.minimum_frequency = minimum_frequency
        self.max_length = max_length

        self.token_to_id = {
            token: index
            for index, token in enumerate(
                SPECIAL_TOKENS
            )
        }

        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }

        self.fitted = False

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(
            r"[a-z0-9]+|[^\w\s]",
            str(text).lower(),
        )

    def fit(
        self,
        texts: Sequence[str],
    ) -> None:
        counter = Counter()

        for text in texts:
            counter.update(
                self.tokenize(text)
            )

        available = (
            self.vocabulary_size
            - len(SPECIAL_TOKENS)
        )

        for token, frequency in counter.most_common():
            if frequency < self.minimum_frequency:
                continue

            if len(self.token_to_id) >= self.vocabulary_size:
                break

            index = len(self.token_to_id)

            self.token_to_id[token] = index
            self.id_to_token[index] = token

        self.fitted = True

    def encode(
        self,
        text: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise VLADatasetError(
                "Tokenizer has not been fitted"
            )

        tokens = self.tokenize(text)

        token_ids = [
            self.token_to_id["<bos>"]
        ]

        token_ids.extend(
            self.token_to_id.get(
                token,
                self.token_to_id["<unk>"],
            )
            for token in tokens[
                :self.max_length - 2
            ]
        )

        token_ids.append(
            self.token_to_id["<eos>"]
        )

        attention_mask = [
            1
        ] * len(token_ids)

        padding_length = (
            self.max_length
            - len(token_ids)
        )

        token_ids.extend(
            [self.token_to_id["<pad>"]]
            * padding_length
        )

        attention_mask.extend(
            [0] * padding_length
        )

        return (
            np.asarray(
                token_ids,
                dtype=np.int64,
            ),
            np.asarray(
                attention_mask,
                dtype=np.int64,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_to_id": self.token_to_id,
            "max_length": self.max_length,
            "vocabulary_size": self.vocabulary_size,
            "minimum_frequency": self.minimum_frequency,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "VLATokenizer":
        tokenizer = cls(
            vocabulary_size=int(
                data["vocabulary_size"]
            ),
            minimum_frequency=int(
                data["minimum_frequency"]
            ),
            max_length=int(
                data["max_length"]
            ),
        )

        tokenizer.token_to_id = {
            str(key): int(value)
            for key, value in data[
                "token_to_id"
            ].items()
        }

        tokenizer.id_to_token = {
            value: key
            for key, value in tokenizer.token_to_id.items()
        }

        tokenizer.fitted = True

        return tokenizer


# ============================================================
# Action Normalizer
# ============================================================

class ActionNormalizer:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(
        self,
        actions: np.ndarray,
    ) -> None:
        self.mean = np.mean(
            actions,
            axis=0,
        )

        self.std = np.std(
            actions,
            axis=0,
        )

        self.std = np.where(
            self.std < 1e-6,
            1.0,
            self.std,
        )

    def transform(
        self,
        actions: np.ndarray,
    ) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise VLADatasetError(
                "Action normalizer is not fitted"
            )

        return (
            (actions - self.mean)
            / self.std
        ).astype(np.float32)

    def inverse_transform(
        self,
        normalized: np.ndarray,
    ) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise VLADatasetError(
                "Action normalizer is not fitted"
            )

        return (
            normalized * self.std
            + self.mean
        )

    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.std is None:
            raise VLADatasetError(
                "Action normalizer is not fitted"
            )

        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ActionNormalizer":
        normalizer = cls()

        normalizer.mean = np.asarray(
            data["mean"],
            dtype=np.float64,
        )

        normalizer.std = np.asarray(
            data["std"],
            dtype=np.float64,
        )

        return normalizer


# ============================================================
# Dataset Loading
# ============================================================

class VLADatasetLoader:
    def __init__(
        self,
        config: VLATrainingConfig,
    ) -> None:
        self.config = config

    def load(self) -> pd.DataFrame:
        path = Path(
            self.config.data_path
        )

        if not path.is_file():
            raise VLADatasetError(
                f"Dataset not found: {path}"
            )

        if path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(path)

        elif path.suffix.lower() in {".parquet", ".pq"}:
            dataframe = pd.read_parquet(path)

        else:
            raise VLADatasetError(
                "Supported dataset formats: CSV, Parquet"
            )

        if dataframe.empty:
            raise VLADatasetError(
                "Dataset is empty"
            )

        if len(dataframe) > MAX_DATASET_ROWS:
            raise VLADatasetError(
                "Dataset exceeds maximum row count"
            )

        required_columns = [
            self.config.image_column,
            self.config.instruction_column,
            self.config.episode_column,
            *self.config.state_columns,
            *self.config.action_columns,
        ]

        missing = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise VLADatasetError(
                f"Missing dataset columns: {missing}"
            )

        if dataframe[
            required_columns
        ].isna().any().any():
            raise VLADatasetError(
                "Required dataset fields contain missing values"
            )

        numeric_columns = (
            self.config.state_columns
            + self.config.action_columns
        )

        for column in numeric_columns:
            values = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            if not np.isfinite(
                values.to_numpy(
                    dtype=np.float64
                )
            ).all():
                raise VLADatasetError(
                    f"Invalid numeric values in {column}"
                )

            dataframe[column] = values

        image_root = (
            Path(self.config.image_root)
            if self.config.image_root
            else path.parent
        )

        for image_path in dataframe[
            self.config.image_column
        ].unique():
            safe_image_path(
                image_root,
                str(image_path),
            )

        LOGGER.info(
            "Loaded %d VLA demonstrations",
            len(dataframe),
        )

        return dataframe


# ============================================================
# Episode-Aware Splitting
# ============================================================

@dataclass
class VLASplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_vla_dataset(
    dataframe: pd.DataFrame,
    config: VLATrainingConfig,
) -> VLASplits:
    groups = dataframe[
        config.episode_column
    ].astype(str).to_numpy()

    if len(np.unique(groups)) < 3:
        raise VLADatasetError(
            "At least three episodes are required"
        )

    temporary_fraction = (
        config.validation_size
        + config.test_size
    )

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=temporary_fraction,
        random_state=config.random_seed,
    )

    train_indices, temporary_indices = next(
        first_split.split(
            dataframe,
            groups=groups,
        )
    )

    temporary = dataframe.iloc[
        temporary_indices
    ].reset_index(drop=True)

    temporary_groups = temporary[
        config.episode_column
    ].astype(str).to_numpy()

    if len(np.unique(temporary_groups)) < 2:
        raise VLADatasetError(
            "Insufficient episodes for validation/test split"
        )

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=(
            config.test_size
            / temporary_fraction
        ),
        random_state=config.random_seed + 1,
    )

    validation_indices, test_indices = next(
        second_split.split(
            temporary,
            groups=temporary_groups,
        )
    )

    return VLASplits(
        train=dataframe.iloc[
            train_indices
        ].reset_index(drop=True),
        validation=temporary.iloc[
            validation_indices
        ].reset_index(drop=True),
        test=temporary.iloc[
            test_indices
        ].reset_index(drop=True),
    )


# ============================================================
# Image Transforms
# ============================================================

def build_image_transform(
    image_size: int,
    *,
    augment: bool,
):
    operations = [
        transforms.Resize(
            (image_size, image_size)
        )
    ]

    if augment:
        operations.extend(
            [
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                ),
                transforms.RandomRotation(
                    degrees=5
                ),
            ]
        )

    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    return transforms.Compose(
        operations
    )


# ============================================================
# PyTorch Dataset
# ============================================================

class RobotVLADataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        config: VLATrainingConfig,
        tokenizer: VLATokenizer,
        action_normalizer: ActionNormalizer,
        *,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        training: bool,
    ) -> None:
        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.config = config
        self.tokenizer = tokenizer
        self.action_normalizer = (
            action_normalizer
        )

        self.image_root = (
            Path(config.image_root)
            if config.image_root
            else Path(config.data_path).parent
        ).resolve()

        self.transform = build_image_transform(
            config.image_size,
            augment=(
                training
                and config.image_augmentation
            ),
        )

        self.state_mean = state_mean
        self.state_std = state_std

    def __len__(self) -> int:
        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[
            index
        ]

        image_path = safe_image_path(
            self.image_root,
            str(
                row[
                    self.config.image_column
                ]
            ),
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            image_tensor = self.transform(
                image
            )

        token_ids, attention_mask = (
            self.tokenizer.encode(
                str(
                    row[
                        self.config.instruction_column
                    ]
                )
            )
        )

        if self.config.state_columns:
            state = row[
                self.config.state_columns
            ].to_numpy(
                dtype=np.float32
            )

            state = (
                (state - self.state_mean)
                / self.state_std
            )

        else:
            state = np.zeros(
                0,
                dtype=np.float32,
            )

        action = row[
            self.config.action_columns
        ].to_numpy(
            dtype=np.float32
        )

        normalized_action = (
            self.action_normalizer.transform(
                action
            )
        )

        return {
            "image": image_tensor,
            "input_ids": torch.tensor(
                token_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.bool,
            ),
            "robot_state": torch.tensor(
                state,
                dtype=torch.float32,
            ),
            "action": torch.tensor(
                normalized_action,
                dtype=torch.float32,
            ),
        }


# ============================================================
# Vision Encoder
# ============================================================

class VisionEncoder(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        *,
        freeze: bool,
    ) -> None:
        super().__init__()

        # Random initialization avoids an implicit
        # network download and supports offline training.
        backbone = resnet18(
            weights=None
        )

        feature_dimension = (
            backbone.fc.in_features
        )

        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.projection = nn.Linear(
            feature_dimension,
            embedding_dimension,
        )

        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

            # With randomly initialized weights, freezing
            # the backbone is usually not useful.
            LOGGER.warning(
                "Vision backbone is randomly initialized "
                "and frozen; consider using a trained encoder"
            )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(
            images
        )

        return self.projection(
            features
        )


# ============================================================
# Language Encoder
# ============================================================

class LanguageEncoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        max_length: int,
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimension,
            padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            max_length,
            embedding_dimension,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = (
            input_ids.shape
        )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0).expand(
            batch_size,
            sequence_length,
        )

        return (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )


# ============================================================
# VLA Transformer
# ============================================================

class VisionLanguageActionTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int,
        max_text_length: int,
        state_dimension: int,
        action_dimension: int,
        embedding_dimension: int,
        transformer_heads: int,
        transformer_layers: int,
        feedforward_dimension: int,
        dropout: float,
        freeze_vision_encoder: bool,
    ) -> None:
        super().__init__()

        self.vision_encoder = VisionEncoder(
            embedding_dimension,
            freeze=freeze_vision_encoder,
        )

        self.language_encoder = LanguageEncoder(
            vocabulary_size,
            embedding_dimension,
            max_text_length,
        )

        self.state_dimension = (
            state_dimension
        )

        if state_dimension > 0:
            self.state_encoder = nn.Sequential(
                nn.Linear(
                    state_dimension,
                    embedding_dimension,
                ),
                nn.LayerNorm(
                    embedding_dimension
                ),
                nn.GELU(),
            )

        else:
            self.state_encoder = None

        transformer_layer = (
            nn.TransformerEncoderLayer(
                d_model=embedding_dimension,
                nhead=transformer_heads,
                dim_feedforward=feedforward_dimension,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )

        self.action_head = nn.Sequential(
            nn.LayerNorm(
                embedding_dimension
            ),
            nn.Linear(
                embedding_dimension,
                feedforward_dimension,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                feedforward_dimension,
                action_dimension,
            ),
        )

        self.action_query = nn.Parameter(
            torch.randn(
                1,
                1,
                embedding_dimension,
            ) * 0.02
        )

    def forward(
        self,
        *,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        robot_state: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = images.shape[0]

        vision_token = self.vision_encoder(
            images
        ).unsqueeze(1)

        language_tokens = (
            self.language_encoder(
                input_ids
            )
        )

        tokens = [
            self.action_query.expand(
                batch_size,
                -1,
                -1,
            ),
            vision_token,
        ]

        prefix_mask = [
            torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
                device=images.device,
            ),
            torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
                device=images.device,
            ),
        ]

        if self.state_encoder is not None:
            state_token = self.state_encoder(
                robot_state
            ).unsqueeze(1)

            tokens.append(
                state_token
            )

            prefix_mask.append(
                torch.zeros(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=images.device,
                )
            )

        tokens.append(
            language_tokens
        )

        padding_mask = torch.cat(
            [
                *prefix_mask,
                ~attention_mask,
            ],
            dim=1,
        )

        sequence = torch.cat(
            tokens,
            dim=1,
        )

        encoded = self.transformer(
            sequence,
            src_key_padding_mask=padding_mask,
        )

        action_representation = (
            encoded[:, 0, :]
        )

        return self.action_head(
            action_representation
        )


# ============================================================
# Evaluation
# ============================================================

def evaluate_action_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, Any]:
    if actual.shape != predicted.shape:
        raise VLAEvaluationError(
            "Action prediction shape mismatch"
        )

    if actual.size == 0:
        raise VLAEvaluationError(
            "No predictions available"
        )

    error = predicted - actual

    absolute_error = np.abs(
        error
    )

    squared_error = error ** 2

    per_dimension_mae = np.mean(
        absolute_error,
        axis=0,
    )

    return {
        "sample_count": int(
            actual.shape[0]
        ),
        "action_dimension": int(
            actual.shape[1]
        ),
        "mae": float(
            np.mean(absolute_error)
        ),
        "rmse": float(
            np.sqrt(
                np.mean(squared_error)
            )
        ),
        "max_absolute_error": float(
            np.max(absolute_error)
        ),
        "per_dimension_mae": (
            per_dimension_mae.tolist()
        ),
        "elementwise_tolerance_accuracy": float(
            np.mean(
                absolute_error <= tolerance
            )
        ),
        "full_action_tolerance_accuracy": float(
            np.mean(
                np.all(
                    absolute_error <= tolerance,
                    axis=1,
                )
            )
        ),
    }


# ============================================================
# Training Result
# ============================================================

@dataclass
class VLATrainingResult:
    run_id: str
    status: str

    architecture: str

    training_samples: int
    validation_samples: int
    test_samples: int

    vocabulary_size: int
    state_dimension: int
    action_dimension: int

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

class VLATrainingPipeline:
    def __init__(
        self,
        config: VLATrainingConfig,
    ) -> None:
        config.validate()

        self.config = config

        self.run_id = generate_run_id()

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

        self.tokenizer = VLATokenizer(
            vocabulary_size=config.vocabulary_size,
            minimum_frequency=(
                config.minimum_token_frequency
            ),
            max_length=config.max_text_length,
        )

        self.action_normalizer = (
            ActionNormalizer()
        )

        self.state_mean: np.ndarray | None = None
        self.state_std: np.ndarray | None = None

        self.model: (
            VisionLanguageActionTransformer | None
        ) = None

    def _prepare_datasets(
        self,
    ) -> tuple[
        RobotVLADataset,
        RobotVLADataset,
        RobotVLADataset,
    ]:
        dataframe = VLADatasetLoader(
            self.config
        ).load()

        splits = split_vla_dataset(
            dataframe,
            self.config,
        )

        self.tokenizer.fit(
            splits.train[
                self.config.instruction_column
            ].astype(str).tolist()
        )

        training_actions = splits.train[
            self.config.action_columns
        ].to_numpy(
            dtype=np.float32
        )

        self.action_normalizer.fit(
            training_actions
        )

        if self.config.state_columns:
            training_states = splits.train[
                self.config.state_columns
            ].to_numpy(
                dtype=np.float32
            )

            self.state_mean = np.mean(
                training_states,
                axis=0,
            )

            self.state_std = np.std(
                training_states,
                axis=0,
            )

            self.state_std = np.where(
                self.state_std < 1e-6,
                1.0,
                self.state_std,
            )

        else:
            self.state_mean = np.zeros(
                0,
                dtype=np.float32,
            )

            self.state_std = np.ones(
                0,
                dtype=np.float32,
            )

        def build(
            frame: pd.DataFrame,
            training: bool,
        ) -> RobotVLADataset:
            return RobotVLADataset(
                frame,
                self.config,
                self.tokenizer,
                self.action_normalizer,
                state_mean=self.state_mean,
                state_std=self.state_std,
                training=training,
            )

        return (
            build(splits.train, True),
            build(splits.validation, False),
            build(splits.test, False),
        )

    def _build_model(
        self,
    ) -> VisionLanguageActionTransformer:
        return VisionLanguageActionTransformer(
            vocabulary_size=len(
                self.tokenizer.token_to_id
            ),
            max_text_length=(
                self.config.max_text_length
            ),
            state_dimension=len(
                self.config.state_columns
            ),
            action_dimension=len(
                self.config.action_columns
            ),
            embedding_dimension=(
                self.config.embedding_dimension
            ),
            transformer_heads=(
                self.config.transformer_heads
            ),
            transformer_layers=(
                self.config.transformer_layers
            ),
            feedforward_dimension=(
                self.config.feedforward_dimension
            ),
            dropout=self.config.dropout,
            freeze_vision_encoder=(
                self.config.freeze_vision_encoder
            ),
        ).to(self.device)

    def _create_loader(
        self,
        dataset: RobotVLADataset,
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

    def _forward_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        assert self.model is not None

        images = batch["image"].to(
            self.device
        )

        input_ids = batch["input_ids"].to(
            self.device
        )

        attention_mask = batch[
            "attention_mask"
        ].to(self.device)

        robot_state = batch[
            "robot_state"
        ].to(self.device)

        target = batch["action"].to(
            self.device
        )

        prediction = self.model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            robot_state=robot_state,
        )

        return prediction, target

    def _evaluate_loader(
        self,
        loader: DataLoader,
    ) -> tuple[
        float,
        np.ndarray,
        np.ndarray,
    ]:
        assert self.model is not None

        self.model.eval()

        criterion = nn.SmoothL1Loss()

        total_loss = 0.0
        total_samples = 0

        predictions = []
        targets = []

        with torch.no_grad():
            for batch in loader:
                prediction, target = (
                    self._forward_batch(
                        batch
                    )
                )

                loss = criterion(
                    prediction,
                    target,
                )

                total_loss += (
                    loss.item()
                    * len(target)
                )

                total_samples += len(target)

                predictions.append(
                    prediction.cpu().numpy()
                )

                targets.append(
                    target.cpu().numpy()
                )

        if total_samples == 0:
            raise VLAEvaluationError(
                "Evaluation loader is empty"
            )

        return (
            total_loss / total_samples,
            np.concatenate(
                predictions,
                axis=0,
            ),
            np.concatenate(
                targets,
                axis=0,
            ),
        )

    def _save_checkpoint(
        self,
        path: Path,
    ) -> None:
        assert self.model is not None
        assert self.state_mean is not None
        assert self.state_std is not None

        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "model_state_dict": (
                self.model.state_dict()
            ),
            "model_config": {
                "vocabulary_size": len(
                    self.tokenizer.token_to_id
                ),
                "max_text_length": (
                    self.config.max_text_length
                ),
                "state_dimension": len(
                    self.config.state_columns
                ),
                "action_dimension": len(
                    self.config.action_columns
                ),
                "embedding_dimension": (
                    self.config.embedding_dimension
                ),
                "transformer_heads": (
                    self.config.transformer_heads
                ),
                "transformer_layers": (
                    self.config.transformer_layers
                ),
                "feedforward_dimension": (
                    self.config.feedforward_dimension
                ),
                "dropout": self.config.dropout,
                "freeze_vision_encoder": (
                    self.config.freeze_vision_encoder
                ),
            },
            "tokenizer": self.tokenizer.to_dict(),
            "action_normalizer": (
                self.action_normalizer.to_dict()
            ),
            "state_mean": (
                self.state_mean.tolist()
            ),
            "state_std": (
                self.state_std.tolist()
            ),
            "state_columns": (
                self.config.state_columns
            ),
            "action_columns": (
                self.config.action_columns
            ),
            "image_size": (
                self.config.image_size
            ),
            "execution_environment": (
                self.config.execution_environment.value
            ),
        }

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            checkpoint,
            path,
        )

    def train(
        self,
    ) -> VLATrainingResult:
        started_at = utc_now()
        started = time.perf_counter()

        set_random_seed(
            self.config.random_seed
        )

        train_dataset, validation_dataset, test_dataset = (
            self._prepare_datasets()
        )

        train_loader = self._create_loader(
            train_dataset,
            shuffle=True,
        )

        validation_loader = self._create_loader(
            validation_dataset,
            shuffle=False,
        )

        test_loader = self._create_loader(
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

        criterion = nn.SmoothL1Loss()

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

        for epoch in range(
            self.config.epochs
        ):
            self.model.train()

            optimizer.zero_grad(
                set_to_none=True
            )

            total_loss = 0.0
            total_samples = 0

            accumulation = (
                self.config.gradient_accumulation_steps
            )

            for step, batch in enumerate(
                train_loader
            ):
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=use_amp,
                ):
                    prediction, target = (
                        self._forward_batch(
                            batch
                        )
                    )

                    raw_loss = criterion(
                        prediction,
                        target,
                    )

                    loss = (
                        raw_loss
                        / accumulation
                    )

                scaler.scale(
                    loss
                ).backward()

                should_step = (
                    (step + 1) % accumulation == 0
                    or (step + 1) == len(train_loader)
                )

                if should_step:
                    scaler.unscale_(
                        optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=(
                            self.config.gradient_clip_norm
                        ),
                    )

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                total_loss += (
                    raw_loss.item()
                    * len(target)
                )

                total_samples += len(target)

            training_loss = (
                total_loss
                / max(total_samples, 1)
            )

            validation_loss, _, _ = (
                self._evaluate_loader(
                    validation_loader
                )
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
                best_validation_loss = (
                    validation_loss
                )

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
            raise VLAModelError(
                "Training did not produce a valid checkpoint"
            )

        self.model.load_state_dict(
            best_state
        )

        _, validation_predictions, validation_targets = (
            self._evaluate_loader(
                validation_loader
            )
        )

        _, test_predictions, test_targets = (
            self._evaluate_loader(
                test_loader
            )
        )

        validation_predictions = (
            self.action_normalizer.inverse_transform(
                validation_predictions
            )
        )

        validation_targets = (
            self.action_normalizer.inverse_transform(
                validation_targets
            )
        )

        test_predictions = (
            self.action_normalizer.inverse_transform(
                test_predictions
            )
        )

        test_targets = (
            self.action_normalizer.inverse_transform(
                test_targets
            )
        )

        validation_metrics = (
            evaluate_action_predictions(
                validation_targets,
                validation_predictions,
                tolerance=self.config.action_tolerance,
            )
        )

        test_metrics = (
            evaluate_action_predictions(
                test_targets,
                test_predictions,
                tolerance=self.config.action_tolerance,
            )
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
            "model_name": (
                "robot-vision-language-action-transformer"
            ),
            "architecture": (
                VLAModelArchitecture.TRANSFORMER.value
            ),
            "created_at": utc_now(),
            "configuration": {
                **asdict(self.config),
                "execution_environment": (
                    self.config.execution_environment.value
                ),
            },
            "vocabulary_size": len(
                self.tokenizer.token_to_id
            ),
            "epochs_completed": len(
                history["training_loss"]
            ),
            "best_validation_loss": (
                best_validation_loss
            ),
            "training_history": history,
            "validation_metrics": (
                validation_metrics
            ),
            "test_metrics": test_metrics,
            "artifact": {
                "filename": (
                    checkpoint_path.name
                ),
                "sha256": calculate_sha256(
                    checkpoint_path
                ),
                "size_bytes": (
                    checkpoint_path.stat().st_size
                ),
            },
            "safety": {
                "simulation_only": True,
                "physical_execution_authorized": False,
                "emergency_stop_reset_authorized": False,
                "safety_interlock_bypass_authorized": False,
                "human_approval_required_for_real_world_use": True,
            },
        }

        save_json_atomic(
            metadata_path,
            metadata,
        )

        result = VLATrainingResult(
            run_id=self.run_id,
            status=TrainingStatus.COMPLETED.value,
            architecture=(
                VLAModelArchitecture.TRANSFORMER.value
            ),
            training_samples=len(
                train_dataset
            ),
            validation_samples=len(
                validation_dataset
            ),
            test_samples=len(
                test_dataset
            ),
            vocabulary_size=len(
                self.tokenizer.token_to_id
            ),
            state_dimension=len(
                self.config.state_columns
            ),
            action_dimension=len(
                self.config.action_columns
            ),
            epochs_completed=len(
                history["training_loss"]
            ),
            best_validation_loss=(
                best_validation_loss
            ),
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            artifact_paths={
                "checkpoint": str(
                    checkpoint_path
                ),
                "metadata": str(
                    metadata_path
                ),
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
            "VLA training completed | MAE=%.6f | "
            "RMSE=%.6f | duration=%.2fs",
            test_metrics["mae"],
            test_metrics["rmse"],
            result.duration_seconds,
        )

        return result


# ============================================================
# Simulation-Only Inference
# ============================================================

class TrainedVLAPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        trusted_checkpoint: bool = False,
    ) -> None:
        if not trusted_checkpoint:
            raise VLAArtifactError(
                "Explicit trusted_checkpoint=True required"
            )

        self.device = resolve_device(
            device
        )

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.is_file():
            raise VLAArtifactError(
                "VLA checkpoint not found"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if (
            checkpoint.get("execution_environment")
            not in {
                "offline",
                "simulation",
            }
        ):
            raise VLAArtifactError(
                "Unsupported execution environment"
            )

        self.tokenizer = VLATokenizer.from_dict(
            checkpoint["tokenizer"]
        )

        self.action_normalizer = (
            ActionNormalizer.from_dict(
                checkpoint["action_normalizer"]
            )
        )

        self.state_mean = np.asarray(
            checkpoint["state_mean"],
            dtype=np.float32,
        )

        self.state_std = np.asarray(
            checkpoint["state_std"],
            dtype=np.float32,
        )

        self.state_columns = list(
            checkpoint["state_columns"]
        )

        self.action_columns = list(
            checkpoint["action_columns"]
        )

        self.image_size = int(
            checkpoint["image_size"]
        )

        self.model = VisionLanguageActionTransformer(
            **checkpoint["model_config"]
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.transform = build_image_transform(
            self.image_size,
            augment=False,
        )

    def predict(
        self,
        *,
        image_path: str | Path,
        instruction: str,
        robot_state: Mapping[str, float],
        environment: ExecutionEnvironment = (
            ExecutionEnvironment.SIMULATION
        ),
    ) -> dict[str, Any]:
        if environment not in {
            ExecutionEnvironment.SIMULATION,
            ExecutionEnvironment.OFFLINE,
        }:
            raise VLAModelError(
                "Physical robot execution is not supported"
            )

        with Image.open(
            image_path
        ) as image:
            image_tensor = self.transform(
                image.convert("RGB")
            ).unsqueeze(0).to(
                self.device
            )

        token_ids, attention_mask = (
            self.tokenizer.encode(
                instruction
            )
        )

        state_values = np.asarray(
            [
                float(
                    robot_state[name]
                )
                for name in self.state_columns
            ],
            dtype=np.float32,
        )

        if not np.isfinite(
            state_values
        ).all():
            raise VLAModelError(
                "Robot state contains invalid values"
            )

        normalized_state = (
            (state_values - self.state_mean)
            / self.state_std
        )

        with torch.no_grad():
            prediction = self.model(
                images=image_tensor,
                input_ids=torch.tensor(
                    token_ids,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0),
                attention_mask=torch.tensor(
                    attention_mask,
                    dtype=torch.bool,
                    device=self.device,
                ).unsqueeze(0),
                robot_state=torch.tensor(
                    normalized_state,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0),
            )

        normalized_action = (
            prediction.cpu().numpy()
        )

        action = (
            self.action_normalizer.inverse_transform(
                normalized_action
            )[0]
        )

        return {
            "predicted_action": {
                name: float(value)
                for name, value in zip(
                    self.action_columns,
                    action,
                )
            },
            "instruction": instruction,
            "environment": environment.value,
            "simulation_only": True,
            "physical_execution_authorized": False,
            "created_at": utc_now(),
        }


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train Vision-Language-Action models"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser(
        "train"
    )

    train_parser.add_argument(
        "--data",
        required=True,
    )

    train_parser.add_argument(
        "--image-root",
        default="",
    )

    train_parser.add_argument(
        "--state-columns",
        nargs="*",
        default=[],
    )

    train_parser.add_argument(
        "--action-columns",
        nargs="+",
        required=True,
    )

    train_parser.add_argument(
        "--output",
        default="artifacts/vla",
    )

    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    train_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    train_parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=256,
    )

    train_parser.add_argument(
        "--transformer-layers",
        type=int,
        default=4,
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
        "--image-root",
        default="",
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
    parser = build_parser()

    args = parser.parse_args()

    if args.command == "train":
        config = VLATrainingConfig(
            data_path=args.data,
            image_root=args.image_root,
            state_columns=args.state_columns,
            action_columns=args.action_columns,
            output_directory=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            embedding_dimension=args.embedding_dimension,
            transformer_layers=args.transformer_layers,
            device=args.device,
            random_seed=args.seed,
        )

        pipeline = VLATrainingPipeline(
            config
        )

        result = pipeline.train()

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
            raise VLAArtifactError(
                "Use --trusted-checkpoint only for "
                "checkpoints from a trusted source"
            )

        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        configuration = (
            checkpoint["model_config"]
        )

        config = VLATrainingConfig(
            data_path=args.data,
            image_root=args.image_root,
            state_columns=checkpoint[
                "state_columns"
            ],
            action_columns=checkpoint[
                "action_columns"
            ],
            image_size=checkpoint[
                "image_size"
            ],
            max_text_length=configuration[
                "max_text_length"
            ],
        )

        dataframe = VLADatasetLoader(
            config
        ).load()

        tokenizer = VLATokenizer.from_dict(
            checkpoint["tokenizer"]
        )

        normalizer = ActionNormalizer.from_dict(
            checkpoint["action_normalizer"]
        )

        dataset = RobotVLADataset(
            dataframe,
            config,
            tokenizer,
            normalizer,
            state_mean=np.asarray(
                checkpoint["state_mean"],
                dtype=np.float32,
            ),
            state_std=np.asarray(
                checkpoint["state_std"],
                dtype=np.float32,
            ),
            training=False,
        )

        device = resolve_device(
            args.device
        )

        model = VisionLanguageActionTransformer(
            **configuration
        ).to(device)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
        )

        predictions = []
        targets = []

        with torch.no_grad():
            for batch in loader:
                output = model(
                    images=batch[
                        "image"
                    ].to(device),
                    input_ids=batch[
                        "input_ids"
                    ].to(device),
                    attention_mask=batch[
                        "attention_mask"
                    ].to(device),
                    robot_state=batch[
                        "robot_state"
                    ].to(device),
                )

                predictions.append(
                    output.cpu().numpy()
                )

                targets.append(
                    batch[
                        "action"
                    ].numpy()
                )

        predicted = normalizer.inverse_transform(
            np.concatenate(
                predictions,
                axis=0,
            )
        )

        actual = normalizer.inverse_transform(
            np.concatenate(
                targets,
                axis=0,
            )
        )

        metrics = evaluate_action_predictions(
            actual,
            predicted,
            tolerance=config.action_tolerance,
        )

        print(
            json.dumps(
                {
                    "metrics": metrics,
                    "evaluation_scope": (
                        "provided_dataset"
                    ),
                    "simulation_only": True,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
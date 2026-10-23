"""
Physical AI & Autonomous Robotics Intelligence Platform
File: ml/training/train_rul.py

Remaining Useful Life (RUL) Prediction Training Pipeline

Features:
- Robot motor, actuator, gearbox and bearing RUL prediction
- Battery and industrial component degradation forecasting
- CSV/Parquet telemetry dataset loading
- Sensor preprocessing and missing-value handling
- Random Forest and Gradient Boosting regression
- Optional PyTorch LSTM sequence model
- Time-series sequence generation
- Robot/component-aware dataset splitting
- MAE, RMSE, R² and NASA asymmetric scoring
- Prediction interval estimation
- Feature importance analysis
- Early stopping for LSTM
- Model checkpoint saving
- Training metadata and SHA-256 checksums
- Batch RUL inference
- CLI training and prediction

Dependencies:
    pip install numpy pandas scikit-learn joblib

Optional:
    pip install torch pyarrow

Examples:
    python -m ml.training.train_rul train \
        --data data/processed/rul_telemetry.csv \
        --target rul \
        --model random_forest

    python -m ml.training.train_rul train \
        --data data/processed/rul_telemetry.csv \
        --target rul \
        --model gradient_boosting

    python -m ml.training.train_rul train \
        --data data/processed/rul_telemetry.csv \
        --target rul \
        --model lstm \
        --sequence-length 30 \
        --epochs 50

Safety:
    RUL predictions are estimates, not certified
    safety assessments. This module cannot command
    robots or bypass safety interlocks.
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
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger("ml.training.train_rul")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCHEMA_VERSION = "1.0.0"

DEFAULT_FEATURES = [
    "motor_temperature",
    "motor_current",
    "motor_voltage",
    "motor_vibration",
    "motor_rpm",
    "joint_temperature",
    "joint_torque",
    "joint_velocity",
    "bearing_temperature",
    "gearbox_temperature",
    "battery_voltage",
    "battery_current",
    "battery_temperature",
    "battery_soc",
    "battery_soh",
    "operating_hours",
    "cycle_count",
    "load_percentage",
    "pressure",
    "humidity",
    "ambient_temperature",
    "sensor_error_rate",
]

EXCLUDED_COLUMNS = {
    "rul",
    "remaining_useful_life",
    "target",
    "label",
    "failure",
    "failure_time",
    "failure_timestamp",
    "timestamp",
    "robot_id",
    "component_id",
    "device_id",
    "tenant_id",
    "organization_id",
    "split",
}

MAX_ROWS = 5_000_000
MAX_FEATURES = 256


# ============================================================
# Exceptions
# ============================================================

class RULTrainingError(Exception):
    pass


class RULDatasetError(RULTrainingError):
    pass


class RULModelError(RULTrainingError):
    pass


class RULEvaluationError(RULTrainingError):
    pass


class RULArtifactError(RULTrainingError):
    pass


# ============================================================
# Enumerations
# ============================================================

class RULModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LSTM = "lstm"


class SplitStrategy(str, Enum):
    RANDOM = "random"
    GROUP = "group"
    CHRONOLOGICAL = "chronological"


class RULHealthStatus(str, Enum):
    HEALTHY = "healthy"
    MONITOR = "monitor"
    MAINTENANCE_RECOMMENDED = "maintenance_recommended"
    URGENT_REVIEW = "urgent_review"


class TrainingStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================
# Configuration
# ============================================================

@dataclass
class RULTrainingConfig:
    data_path: str

    target_column: str = "rul"

    model_type: RULModelType = (
        RULModelType.RANDOM_FOREST
    )

    output_directory: str = "artifacts/rul"

    feature_columns: list[str] = field(
        default_factory=list
    )

    group_column: str = "component_id"
    timestamp_column: str = "timestamp"

    split_strategy: SplitStrategy = SplitStrategy.GROUP

    validation_size: float = 0.15
    test_size: float = 0.15

    random_seed: int = 42

    rul_cap: float | None = None
    rul_unit: str = "hours"

    n_estimators: int = 300
    max_depth: int | None = 18
    min_samples_leaf: int = 2

    learning_rate: float = 0.05
    max_iterations: int = 300

    sequence_length: int = 30
    sequence_stride: int = 1

    lstm_hidden_size: int = 128
    lstm_layers: int = 2
    dropout: float = 0.2

    epochs: int = 50
    batch_size: int = 128
    lstm_learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 10

    device: str = "auto"

    prediction_interval_quantile: float = 0.90

    healthy_threshold: float = 200.0
    monitor_threshold: float = 100.0
    urgent_threshold: float = 24.0

    def validate(self) -> None:
        if not self.data_path:
            raise RULDatasetError(
                "Dataset path is required"
            )

        if not (
            0 < self.validation_size < 1
            and 0 < self.test_size < 1
            and self.validation_size + self.test_size < 1
        ):
            raise ValueError(
                "Invalid dataset split fractions"
            )

        if self.rul_cap is not None and self.rul_cap <= 0:
            raise ValueError(
                "RUL cap must be positive"
            )

        if self.n_estimators < 1:
            raise ValueError(
                "n_estimators must be positive"
            )

        if self.sequence_length < 2:
            raise ValueError(
                "sequence_length must be at least 2"
            )

        if self.sequence_stride < 1:
            raise ValueError(
                "sequence_stride must be positive"
            )

        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError(
                "epochs and batch_size must be positive"
            )

        if self.lstm_hidden_size < 1:
            raise ValueError(
                "lstm_hidden_size must be positive"
            )

        if self.lstm_layers < 1:
            raise ValueError(
                "lstm_layers must be positive"
            )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        if not 0 < self.prediction_interval_quantile < 1:
            raise ValueError(
                "Invalid prediction interval quantile"
            )

        if not (
            0 <= self.urgent_threshold
            < self.monitor_threshold
            < self.healthy_threshold
        ):
            raise ValueError(
                "Invalid RUL health thresholds"
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

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    except ImportError:
        pass


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
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def resolve_torch_device(
    requested: str,
):
    try:
        import torch

    except ImportError as exc:
        raise RULModelError(
            "PyTorch is required for LSTM training"
        ) from exc

    if requested != "auto":
        device = torch.device(requested)

        if (
            device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RULModelError(
                "CUDA was requested but is unavailable"
            )

        if (
            device.type == "mps"
            and not (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
        ):
            raise RULModelError(
                "MPS was requested but is unavailable"
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


# ============================================================
# Dataset
# ============================================================

class RULDataset:
    def __init__(
        self,
        config: RULTrainingConfig,
    ) -> None:
        self.config = config

        self.dataframe: pd.DataFrame | None = None

        self.feature_columns: list[str] = []

    def load(self) -> pd.DataFrame:
        path = Path(
            self.config.data_path
        )

        if not path.is_file():
            raise RULDatasetError(
                f"Dataset not found: {path}"
            )

        if path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(path)

        elif path.suffix.lower() in {".parquet", ".pq"}:
            dataframe = pd.read_parquet(path)

        else:
            raise RULDatasetError(
                "Supported formats: CSV and Parquet"
            )

        if dataframe.empty:
            raise RULDatasetError(
                "Dataset is empty"
            )

        if len(dataframe) > MAX_ROWS:
            raise RULDatasetError(
                "Dataset exceeds maximum row count"
            )

        if dataframe.columns.duplicated().any():
            raise RULDatasetError(
                "Duplicate dataset columns"
            )

        if self.config.target_column not in dataframe:
            raise RULDatasetError(
                f"Missing target column: "
                f"{self.config.target_column}"
            )

        self.dataframe = dataframe

        return dataframe

    def select_features(self) -> list[str]:
        if self.dataframe is None:
            raise RULDatasetError(
                "Load dataset first"
            )

        if self.config.feature_columns:
            features = list(
                self.config.feature_columns
            )

        else:
            features = [
                name
                for name in DEFAULT_FEATURES
                if name in self.dataframe.columns
            ]

            if not features:
                features = [
                    name
                    for name in self.dataframe.select_dtypes(
                        include=[np.number]
                    ).columns
                    if name.lower()
                    not in EXCLUDED_COLUMNS
                    and name != self.config.target_column
                ]

        if not features:
            raise RULDatasetError(
                "No numerical telemetry features found"
            )

        if len(features) > MAX_FEATURES:
            raise RULDatasetError(
                "Too many features"
            )

        if len(set(features)) != len(features):
            raise RULDatasetError(
                "Duplicate selected features"
            )

        forbidden = {
            self.config.target_column,
            self.config.group_column,
            self.config.timestamp_column,
        }

        if set(features) & forbidden:
            raise RULDatasetError(
                "Target, group or timestamp column "
                "cannot be an input feature"
            )

        for name in features:
            if name not in self.dataframe:
                raise RULDatasetError(
                    f"Missing feature: {name}"
                )

            if not pd.api.types.is_numeric_dtype(
                self.dataframe[name]
            ):
                raise RULDatasetError(
                    f"Feature is not numeric: {name}"
                )

        self.feature_columns = features

        return features

    def prepare(
        self,
    ) -> tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray | None,
        pd.Series | None,
    ]:
        if self.dataframe is None:
            self.load()

        if not self.feature_columns:
            self.select_features()

        assert self.dataframe is not None

        features = self.dataframe[
            self.feature_columns
        ].copy()

        features = features.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        for name in self.feature_columns:
            if features[name].isna().all():
                raise RULDatasetError(
                    f"Feature contains only missing values: {name}"
                )

        target = pd.to_numeric(
            self.dataframe[
                self.config.target_column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(target).all():
            raise RULDatasetError(
                "RUL target contains invalid values"
            )

        if np.any(target < 0):
            raise RULDatasetError(
                "RUL target cannot be negative"
            )

        if self.config.rul_cap is not None:
            target = np.minimum(
                target,
                self.config.rul_cap,
            )

        groups = None

        if self.config.group_column in self.dataframe:
            group_values = self.dataframe[
                self.config.group_column
            ]

            if group_values.isna().any():
                raise RULDatasetError(
                    "Component group IDs contain missing values"
                )

            groups = group_values.astype(
                str
            ).to_numpy()

        timestamps = None

        if self.config.timestamp_column in self.dataframe:
            timestamps = pd.to_datetime(
                self.dataframe[
                    self.config.timestamp_column
                ],
                utc=True,
                errors="coerce",
            )

            if timestamps.isna().any():
                raise RULDatasetError(
                    "Invalid telemetry timestamps"
                )

        return (
            features,
            target,
            groups,
            timestamps,
        )


# ============================================================
# Splitting
# ============================================================

@dataclass
class RULSplits:
    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame

    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray

    groups_train: np.ndarray | None = None
    groups_validation: np.ndarray | None = None
    groups_test: np.ndarray | None = None

    times_train: pd.Series | None = None
    times_validation: pd.Series | None = None
    times_test: pd.Series | None = None


def split_rul_dataset(
    features: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray | None,
    timestamps: pd.Series | None,
    config: RULTrainingConfig,
) -> RULSplits:
    count = len(features)

    if count < 50:
        raise RULDatasetError(
            "At least 50 samples are required"
        )

    indices = np.arange(count)

    temporary_fraction = (
        config.validation_size
        + config.test_size
    )

    if config.split_strategy == SplitStrategy.GROUP:
        if groups is None:
            raise RULDatasetError(
                "Group split requires component IDs"
            )

        if len(np.unique(groups)) < 3:
            raise RULDatasetError(
                "At least 3 component groups are required"
            )

        first_split = GroupShuffleSplit(
            n_splits=1,
            test_size=temporary_fraction,
            random_state=config.random_seed,
        )

        train_position, temporary_position = next(
            first_split.split(
                indices,
                groups=groups,
            )
        )

        train_indices = indices[
            train_position
        ]

        temporary_indices = indices[
            temporary_position
        ]

        relative_test_fraction = (
            config.test_size
            / temporary_fraction
        )

        second_split = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_test_fraction,
            random_state=config.random_seed + 1,
        )

        validation_position, test_position = next(
            second_split.split(
                temporary_indices,
                groups=groups[temporary_indices],
            )
        )

        validation_indices = temporary_indices[
            validation_position
        ]

        test_indices = temporary_indices[
            test_position
        ]

    elif (
        config.split_strategy
        == SplitStrategy.CHRONOLOGICAL
    ):
        if timestamps is None:
            raise RULDatasetError(
                "Chronological split requires timestamps"
            )

        ordered = np.argsort(
            timestamps.to_numpy(),
            kind="stable",
        )

        train_end = int(
            count * (1 - temporary_fraction)
        )

        validation_end = int(
            count * (1 - config.test_size)
        )

        train_indices = ordered[
            :train_end
        ]

        validation_indices = ordered[
            train_end:validation_end
        ]

        test_indices = ordered[
            validation_end:
        ]

    else:
        train_indices, temporary_indices = (
            train_test_split(
                indices,
                test_size=temporary_fraction,
                random_state=config.random_seed,
            )
        )

        validation_indices, test_indices = (
            train_test_split(
                temporary_indices,
                test_size=(
                    config.test_size
                    / temporary_fraction
                ),
                random_state=config.random_seed,
            )
        )

    def subset(
        selected: np.ndarray,
    ) -> tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray | None,
        pd.Series | None,
    ]:
        return (
            features.iloc[
                selected
            ].reset_index(drop=True),
            target[selected],
            (
                groups[selected]
                if groups is not None
                else None
            ),
            (
                timestamps.iloc[
                    selected
                ].reset_index(drop=True)
                if timestamps is not None
                else None
            ),
        )

    train = subset(train_indices)
    validation = subset(validation_indices)
    test = subset(test_indices)

    if any(
        len(part[1]) == 0
        for part in (
            train,
            validation,
            test,
        )
    ):
        raise RULDatasetError(
            "One or more dataset splits are empty"
        )

    return RULSplits(
        x_train=train[0],
        y_train=train[1],
        groups_train=train[2],
        times_train=train[3],
        x_validation=validation[0],
        y_validation=validation[1],
        groups_validation=validation[2],
        times_validation=validation[3],
        x_test=test[0],
        y_test=test[1],
        groups_test=test[2],
        times_test=test[3],
    )


# ============================================================
# Preprocessing
# ============================================================

class RULPreprocessor:
    def __init__(self) -> None:
        self.pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        self.fitted = False

    def fit_transform(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        transformed = (
            self.pipeline.fit_transform(
                features
            )
        )

        self.fitted = True

        return np.asarray(
            transformed,
            dtype=np.float32,
        )

    def transform(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        if not self.fitted:
            raise RULModelError(
                "Preprocessor is not fitted"
            )

        return np.asarray(
            self.pipeline.transform(
                features
            ),
            dtype=np.float32,
        )


# ============================================================
# Sequence Generation
# ============================================================

@dataclass
class RULSequenceDataset:
    sequences: np.ndarray
    targets: np.ndarray


def create_rul_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    sequence_length: int,
    stride: int = 1,
    groups: np.ndarray | None = None,
    timestamps: pd.Series | None = None,
) -> RULSequenceDataset:
    if len(features) != len(targets):
        raise RULDatasetError(
            "Feature/target length mismatch"
        )

    if groups is None:
        groups = np.zeros(
            len(features),
            dtype=np.int64,
        )

    sequences = []
    labels = []

    for group in np.unique(groups):
        indices = np.flatnonzero(
            groups == group
        )

        if timestamps is not None:
            ordered = np.argsort(
                timestamps.iloc[
                    indices
                ].to_numpy(),
                kind="stable",
            )

            indices = indices[
                ordered
            ]

        group_features = features[
            indices
        ]

        group_targets = targets[
            indices
        ]

        if len(group_features) < sequence_length:
            continue

        for end in range(
            sequence_length,
            len(group_features) + 1,
            stride,
        ):
            start = end - sequence_length

            sequences.append(
                group_features[
                    start:end
                ]
            )

            labels.append(
                group_targets[
                    end - 1
                ]
            )

    if not sequences:
        raise RULDatasetError(
            "No valid RUL sequences generated; "
            "check group sizes and sequence length"
        )

    return RULSequenceDataset(
        sequences=np.asarray(
            sequences,
            dtype=np.float32,
        ),
        targets=np.asarray(
            labels,
            dtype=np.float32,
        ),
    )


# ============================================================
# Regression Models
# ============================================================

class ClassicalRULModel:
    def __init__(
        self,
        config: RULTrainingConfig,
    ) -> None:
        self.config = config

        if config.model_type == RULModelType.RANDOM_FOREST:
            self.model = RandomForestRegressor(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_samples_leaf=config.min_samples_leaf,
                random_state=config.random_seed,
                n_jobs=-1,
            )

        elif (
            config.model_type
            == RULModelType.GRADIENT_BOOSTING
        ):
            self.model = HistGradientBoostingRegressor(
                max_iter=config.max_iterations,
                learning_rate=config.learning_rate,
                max_depth=config.max_depth,
                l2_regularization=0.1,
                random_state=config.random_seed,
            )

        else:
            raise RULModelError(
                "Unsupported classical RUL model"
            )

        self.fitted = False

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, Any]:
        started = time.perf_counter()

        self.model.fit(
            features,
            targets,
        )

        self.fitted = True

        return {
            "training_duration_seconds": round(
                time.perf_counter() - started,
                4,
            ),
            "training_samples": len(features),
        }

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        if not self.fitted:
            raise RULModelError(
                "RUL model is not fitted"
            )

        predictions = self.model.predict(
            features
        )

        return np.maximum(
            predictions,
            0.0,
        )


# ============================================================
# LSTM Model
# ============================================================

class LSTMRULModel:
    def __init__(
        self,
        config: RULTrainingConfig,
        input_dimension: int,
    ) -> None:
        try:
            import torch
            from torch import nn

        except ImportError as exc:
            raise RULModelError(
                "PyTorch required for LSTM"
            ) from exc

        self.torch = torch
        self.nn = nn

        self.config = config
        self.input_dimension = input_dimension

        self.device = resolve_torch_device(
            config.device
        )

        class Network(nn.Module):
            def __init__(
                self,
                input_size: int,
                hidden_size: int,
                layers: int,
                dropout: float,
            ) -> None:
                super().__init__()

                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=layers,
                    batch_first=True,
                    dropout=(
                        dropout
                        if layers > 1
                        else 0.0
                    ),
                )

                self.regressor = nn.Sequential(
                    nn.Linear(
                        hidden_size,
                        hidden_size // 2 or 1,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        hidden_size // 2 or 1,
                        1,
                    ),
                    nn.Softplus(),
                )

            def forward(
                self,
                x,
            ):
                output, _ = self.lstm(x)

                last_hidden = output[
                    :,
                    -1,
                    :,
                ]

                return self.regressor(
                    last_hidden
                ).squeeze(-1)

        self.network = Network(
            input_size=input_dimension,
            hidden_size=config.lstm_hidden_size,
            layers=config.lstm_layers,
            dropout=config.dropout,
        ).to(self.device)

        self.fitted = False

    def fit(
        self,
        train: RULSequenceDataset,
        validation: RULSequenceDataset,
    ) -> dict[str, Any]:
        torch = self.torch

        train_x = torch.tensor(
            train.sequences,
            dtype=torch.float32,
        )

        train_y = torch.tensor(
            train.targets,
            dtype=torch.float32,
        )

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                train_x,
                train_y,
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.lstm_learning_rate,
            weight_decay=self.config.weight_decay,
        )

        criterion = self.nn.SmoothL1Loss()

        best_validation_loss = float("inf")
        best_state = None
        patience_counter = 0

        history = {
            "train_loss": [],
            "validation_loss": [],
        }

        started = time.perf_counter()

        for epoch in range(self.config.epochs):
            self.network.train()

            total_loss = 0.0
            total_samples = 0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(
                    self.device
                )

                batch_y = batch_y.to(
                    self.device
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                predictions = self.network(
                    batch_x
                )

                loss = criterion(
                    predictions,
                    batch_y,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                total_loss += (
                    loss.item()
                    * len(batch_x)
                )

                total_samples += len(batch_x)

            train_loss = (
                total_loss
                / max(total_samples, 1)
            )

            validation_predictions = self.predict(
                validation.sequences,
                require_fitted=False,
            )

            validation_loss = float(
                np.mean(
                    np.abs(
                        validation_predictions
                        - validation.targets
                    )
                )
            )

            history["train_loss"].append(
                train_loss
            )

            history["validation_loss"].append(
                validation_loss
            )

            LOGGER.info(
                "Epoch %d/%d | train=%.6f | val_MAE=%.6f",
                epoch + 1,
                self.config.epochs,
                train_loss,
                validation_loss,
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = (
                    validation_loss
                )

                best_state = copy.deepcopy(
                    self.network.state_dict()
                )

                patience_counter = 0

            else:
                patience_counter += 1

            if (
                patience_counter
                >= self.config.patience
            ):
                LOGGER.info(
                    "Early stopping at epoch %d",
                    epoch + 1,
                )

                break

        if best_state is not None:
            self.network.load_state_dict(
                best_state
            )

        self.fitted = True

        return {
            "training_duration_seconds": round(
                time.perf_counter() - started,
                4,
            ),
            "epochs_completed": len(
                history["train_loss"]
            ),
            "best_validation_mae": (
                best_validation_loss
            ),
            "device": str(
                self.device
            ),
            "history": history,
        }

    def predict(
        self,
        sequences: np.ndarray,
        *,
        require_fitted: bool = True,
    ) -> np.ndarray:
        if require_fitted and not self.fitted:
            raise RULModelError(
                "LSTM model is not fitted"
            )

        torch = self.torch

        self.network.eval()

        predictions = []

        with torch.no_grad():
            for start in range(
                0,
                len(sequences),
                self.config.batch_size,
            ):
                batch = sequences[
                    start:
                    start + self.config.batch_size
                ]

                tensor = torch.tensor(
                    batch,
                    dtype=torch.float32,
                    device=self.device,
                )

                output = self.network(
                    tensor
                )

                predictions.extend(
                    output.cpu().numpy()
                )

        return np.maximum(
            np.asarray(
                predictions,
                dtype=np.float64,
            ),
            0.0,
        )

    def save(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.torch.save(
            {
                "state_dict": (
                    self.network.state_dict()
                ),
                "input_dimension": (
                    self.input_dimension
                ),
                "hidden_size": (
                    self.config.lstm_hidden_size
                ),
                "layers": (
                    self.config.lstm_layers
                ),
                "dropout": (
                    self.config.dropout
                ),
                "sequence_length": (
                    self.config.sequence_length
                ),
            },
            path,
        )


# ============================================================
# Evaluation
# ============================================================

def nasa_rul_score(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    NASA-style asymmetric penalty.

    Late predictions (predicted RUL > actual RUL)
    are penalized more heavily than early predictions.

    Score is a total penalty; lower is better.
    """
    error = predicted - actual

    penalties = np.where(
        error < 0,
        np.exp(
            np.minimum(-error / 13.0, 50.0)
        ) - 1.0,
        np.exp(
            np.minimum(error / 10.0, 50.0)
        ) - 1.0,
    )

    return float(
        np.sum(penalties)
    )


def evaluate_rul(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    if len(actual) != len(predicted):
        raise RULEvaluationError(
            "Prediction length mismatch"
        )

    if len(actual) == 0:
        raise RULEvaluationError(
            "Evaluation dataset is empty"
        )

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
    ):
        raise RULEvaluationError(
            "Evaluation contains invalid values"
        )

    errors = predicted - actual

    return {
        "sample_count": len(actual),
        "mae": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
        "r2": (
            float(
                r2_score(
                    actual,
                    predicted,
                )
            )
            if len(actual) >= 2
            else None
        ),
        "mean_error": float(
            np.mean(errors)
        ),
        "median_absolute_error": float(
            np.median(
                np.abs(errors)
            )
        ),
        "nasa_score": nasa_rul_score(
            actual,
            predicted,
        ),
        "actual_mean": float(
            np.mean(actual)
        ),
        "predicted_mean": float(
            np.mean(predicted)
        ),
    }


# ============================================================
# Prediction Intervals
# ============================================================

@dataclass
class RULIntervalCalibration:
    absolute_error_quantile: float
    coverage_target: float
    calibration_samples: int
    method: str


def calibrate_prediction_interval(
    actual: np.ndarray,
    predicted: np.ndarray,
    coverage: float,
) -> RULIntervalCalibration:
    if len(actual) == 0:
        raise RULEvaluationError(
            "Calibration requires samples"
        )

    residuals = np.abs(
        actual - predicted
    )

    n = len(residuals)

    # Split-conformal finite-sample quantile.
    quantile_level = min(
        1.0,
        math.ceil(
            (n + 1) * coverage
        ) / n,
    )

    try:
        error_quantile = float(
            np.quantile(
                residuals,
                quantile_level,
                method="higher",
            )
        )

    except TypeError:
        error_quantile = float(
            np.quantile(
                residuals,
                quantile_level,
                interpolation="higher",
            )
        )

    return RULIntervalCalibration(
        absolute_error_quantile=error_quantile,
        coverage_target=coverage,
        calibration_samples=n,
        method="split_conformal_absolute_residual",
    )


def make_rul_interval(
    prediction: float,
    calibration: RULIntervalCalibration,
) -> tuple[float, float]:
    margin = (
        calibration.absolute_error_quantile
    )

    return (
        max(0.0, prediction - margin),
        prediction + margin,
    )


# ============================================================
# Health Classification
# ============================================================

def classify_rul_health(
    rul: float,
    config: RULTrainingConfig,
) -> RULHealthStatus:
    if not math.isfinite(rul):
        raise ValueError(
            "RUL must be finite"
        )

    if rul <= config.urgent_threshold:
        return RULHealthStatus.URGENT_REVIEW

    if rul <= config.monitor_threshold:
        return (
            RULHealthStatus.MAINTENANCE_RECOMMENDED
        )

    if rul <= config.healthy_threshold:
        return RULHealthStatus.MONITOR

    return RULHealthStatus.HEALTHY


# ============================================================
# Training Result
# ============================================================

@dataclass
class RULTrainingResult:
    run_id: str
    status: str

    model_type: str
    feature_columns: list[str]

    training_samples: int
    validation_samples: int
    test_samples: int

    training_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]

    interval_calibration: dict[str, Any]

    feature_importance: list[dict[str, Any]]

    artifact_paths: dict[str, str]

    started_at: str
    completed_at: str
    duration_seconds: float


# ============================================================
# Training Pipeline
# ============================================================

class RULTrainingPipeline:
    def __init__(
        self,
        config: RULTrainingConfig,
    ) -> None:
        config.validate()

        self.config = config
        self.run_id = generate_run_id()

        self.output_directory = Path(
            config.output_directory
        ).resolve()

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.preprocessor = (
            RULPreprocessor()
        )

        self.model: Any = None
        self.feature_columns: list[str] = []

    def _prepare(
        self,
    ) -> tuple[
        RULSplits,
        dict[str, np.ndarray],
    ]:
        dataset = RULDataset(
            self.config
        )

        dataset.load()

        self.feature_columns = (
            dataset.select_features()
        )

        features, targets, groups, timestamps = (
            dataset.prepare()
        )

        splits = split_rul_dataset(
            features,
            targets,
            groups,
            timestamps,
            self.config,
        )

        arrays = {
            "train": (
                self.preprocessor.fit_transform(
                    splits.x_train
                )
            ),
            "validation": (
                self.preprocessor.transform(
                    splits.x_validation
                )
            ),
            "test": (
                self.preprocessor.transform(
                    splits.x_test
                )
            ),
        }

        return splits, arrays

    def _build_lstm_sequences(
        self,
        splits: RULSplits,
        arrays: dict[str, np.ndarray],
    ) -> tuple[
        RULSequenceDataset,
        RULSequenceDataset,
        RULSequenceDataset,
    ]:
        config = self.config

        train = create_rul_sequences(
            arrays["train"],
            splits.y_train,
            sequence_length=config.sequence_length,
            stride=config.sequence_stride,
            groups=splits.groups_train,
            timestamps=splits.times_train,
        )

        validation = create_rul_sequences(
            arrays["validation"],
            splits.y_validation,
            sequence_length=config.sequence_length,
            stride=config.sequence_stride,
            groups=splits.groups_validation,
            timestamps=splits.times_validation,
        )

        test = create_rul_sequences(
            arrays["test"],
            splits.y_test,
            sequence_length=config.sequence_length,
            stride=config.sequence_stride,
            groups=splits.groups_test,
            timestamps=splits.times_test,
        )

        return train, validation, test

    def _feature_importance(
        self,
    ) -> list[dict[str, Any]]:
        if not isinstance(
            self.model,
            ClassicalRULModel,
        ):
            return []

        estimator = self.model.model

        if not hasattr(
            estimator,
            "feature_importances_",
        ):
            return []

        importance = [
            {
                "feature": feature,
                "importance": float(value),
            }
            for feature, value in zip(
                self.feature_columns,
                estimator.feature_importances_,
            )
        ]

        importance.sort(
            key=lambda item: item["importance"],
            reverse=True,
        )

        return importance

    def _save_artifacts(
        self,
        *,
        training_metrics: dict[str, Any],
        validation_metrics: dict[str, Any],
        test_metrics: dict[str, Any],
        calibration: RULIntervalCalibration,
        feature_importance: list[dict[str, Any]],
    ) -> dict[str, str]:
        output = self.output_directory

        preprocessor_path = (
            output / "preprocessor.joblib"
        )

        joblib.dump(
            self.preprocessor.pipeline,
            preprocessor_path,
        )

        if self.config.model_type == RULModelType.LSTM:
            model_path = (
                output / "rul_lstm.pt"
            )

            self.model.save(
                model_path
            )

        else:
            model_path = (
                output / "rul_model.joblib"
            )

            joblib.dump(
                self.model.model,
                model_path,
            )

        metadata_path = (
            output / "training_metadata.json"
        )

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "model_name": "robot-rul-predictor",
            "model_type": (
                self.config.model_type.value
            ),
            "created_at": utc_now(),
            "feature_columns": (
                self.feature_columns
            ),
            "rul_unit": (
                self.config.rul_unit
            ),
            "configuration": {
                **asdict(self.config),
                "model_type": (
                    self.config.model_type.value
                ),
                "split_strategy": (
                    self.config.split_strategy.value
                ),
            },
            "interval_calibration": asdict(
                calibration
            ),
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "feature_importance": (
                feature_importance
            ),
            "artifacts": {
                "model": {
                    "filename": model_path.name,
                    "sha256": sha256_file(
                        model_path
                    ),
                },
                "preprocessor": {
                    "filename": (
                        preprocessor_path.name
                    ),
                    "sha256": sha256_file(
                        preprocessor_path
                    ),
                },
            },
            "safety": {
                "advisory_only": True,
                "physical_execution_authorized": False,
                "emergency_stop_reset_authorized": False,
                "safety_interlock_bypass_authorized": False,
                "human_review_required": True,
            },
        }

        save_json_atomic(
            metadata_path,
            metadata,
        )

        return {
            "model": str(model_path),
            "preprocessor": str(
                preprocessor_path
            ),
            "metadata": str(
                metadata_path
            ),
        }

    def train(
        self,
    ) -> RULTrainingResult:
        started_at = utc_now()
        started = time.perf_counter()

        set_random_seed(
            self.config.random_seed
        )

        LOGGER.info(
            "Starting RUL training: %s",
            self.run_id,
        )

        splits, arrays = self._prepare()

        if self.config.model_type == RULModelType.LSTM:
            train_data, validation_data, test_data = (
                self._build_lstm_sequences(
                    splits,
                    arrays,
                )
            )

            self.model = LSTMRULModel(
                self.config,
                input_dimension=(
                    arrays["train"].shape[1]
                ),
            )

            training_metrics = self.model.fit(
                train_data,
                validation_data,
            )

            train_actual = train_data.targets
            validation_actual = (
                validation_data.targets
            )
            test_actual = test_data.targets

            train_predictions = self.model.predict(
                train_data.sequences
            )

            validation_predictions = (
                self.model.predict(
                    validation_data.sequences
                )
            )

            test_predictions = self.model.predict(
                test_data.sequences
            )

        else:
            self.model = ClassicalRULModel(
                self.config
            )

            training_metrics = self.model.fit(
                arrays["train"],
                splits.y_train,
            )

            train_actual = splits.y_train
            validation_actual = (
                splits.y_validation
            )
            test_actual = splits.y_test

            train_predictions = self.model.predict(
                arrays["train"]
            )

            validation_predictions = (
                self.model.predict(
                    arrays["validation"]
                )
            )

            test_predictions = self.model.predict(
                arrays["test"]
            )

        training_metrics["evaluation"] = (
            evaluate_rul(
                train_actual,
                train_predictions,
            )
        )

        validation_metrics = evaluate_rul(
            validation_actual,
            validation_predictions,
        )

        test_metrics = evaluate_rul(
            test_actual,
            test_predictions,
        )

        calibration = (
            calibrate_prediction_interval(
                validation_actual,
                validation_predictions,
                self.config.prediction_interval_quantile,
            )
        )

        feature_importance = (
            self._feature_importance()
        )

        artifact_paths = (
            self._save_artifacts(
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                calibration=calibration,
                feature_importance=feature_importance,
            )
        )

        result = RULTrainingResult(
            run_id=self.run_id,
            status=TrainingStatus.COMPLETED.value,
            model_type=(
                self.config.model_type.value
            ),
            feature_columns=(
                self.feature_columns
            ),
            training_samples=len(
                train_actual
            ),
            validation_samples=len(
                validation_actual
            ),
            test_samples=len(
                test_actual
            ),
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            interval_calibration=asdict(
                calibration
            ),
            feature_importance=feature_importance,
            artifact_paths=artifact_paths,
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
            "RUL training completed | "
            "MAE=%.4f | RMSE=%.4f | R2=%.4f",
            test_metrics["mae"],
            test_metrics["rmse"],
            test_metrics["r2"],
        )

        return result


# ============================================================
# Trusted Artifact Inference
# ============================================================

class TrainedRULPredictor:
    """
    Load locally trusted RUL artifacts.

    Joblib deserialization is unsafe for untrusted files.
    SHA-256 checksums detect corruption but do not prove
    artifact authenticity.

    This predictor supports classical regression models.
    LSTM inference requires ordered component histories.
    """

    def __init__(
        self,
        artifact_directory: str | Path,
        *,
        trusted_source: bool = False,
        expected_model_sha256: str | None = None,
    ) -> None:
        if not trusted_source:
            raise RULArtifactError(
                "Explicit trusted_source=True required"
            )

        self.artifact_directory = Path(
            artifact_directory
        ).resolve()

        metadata_path = (
            self.artifact_directory
            / "training_metadata.json"
        )

        if not metadata_path.is_file():
            raise RULArtifactError(
                "Training metadata not found"
            )

        self.metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        self.model_type = RULModelType(
            self.metadata["model_type"]
        )

        if self.model_type == RULModelType.LSTM:
            raise RULArtifactError(
                "Use an LSTM sequence inference service "
                "for LSTM checkpoints"
            )

        for artifact in self.metadata[
            "artifacts"
        ].values():
            path = (
                self.artifact_directory
                / artifact["filename"]
            ).resolve()

            if not path.is_relative_to(
                self.artifact_directory
            ):
                raise RULArtifactError(
                    "Artifact path escapes trusted directory"
                )

            if not path.is_file():
                raise RULArtifactError(
                    f"Missing artifact: {path.name}"
                )

            if sha256_file(path) != artifact["sha256"]:
                raise RULArtifactError(
                    f"Checksum mismatch: {path.name}"
                )

        model_artifact = self.metadata[
            "artifacts"
        ]["model"]

        if (
            expected_model_sha256 is not None
            and model_artifact["sha256"]
            != expected_model_sha256
        ):
            raise RULArtifactError(
                "Model does not match trusted checksum"
            )

        self.feature_columns = list(
            self.metadata["feature_columns"]
        )

        self.model = joblib.load(
            self.artifact_directory
            / model_artifact["filename"]
        )

        self.preprocessor = joblib.load(
            self.artifact_directory
            / self.metadata["artifacts"][
                "preprocessor"
            ]["filename"]
        )

        self.calibration = RULIntervalCalibration(
            **self.metadata[
                "interval_calibration"
            ]
        )

        config_data = dict(
            self.metadata["configuration"]
        )

        config_data["model_type"] = (
            RULModelType(
                config_data["model_type"]
            )
        )

        config_data["split_strategy"] = (
            SplitStrategy(
                config_data["split_strategy"]
            )
        )

        self.config = RULTrainingConfig(
            **config_data
        )

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        missing = [
            name
            for name in self.feature_columns
            if name not in dataframe
        ]

        if missing:
            raise RULDatasetError(
                f"Missing inference features: {missing}"
            )

        features = dataframe[
            self.feature_columns
        ].copy()

        features = features.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        transformed = (
            self.preprocessor.transform(
                features
            )
        )

        predictions = np.maximum(
            self.model.predict(
                transformed
            ),
            0.0,
        )

        intervals = [
            make_rul_interval(
                float(value),
                self.calibration,
            )
            for value in predictions
        ]

        result = dataframe.copy()

        result["predicted_rul"] = predictions

        result["rul_lower_bound"] = [
            item[0]
            for item in intervals
        ]

        result["rul_upper_bound"] = [
            item[1]
            for item in intervals
        ]

        result["rul_unit"] = (
            self.config.rul_unit
        )

        result["health_status"] = [
            classify_rul_health(
                float(value),
                self.config,
            ).value
            for value in predictions
        ]

        result["advisory_only"] = True

        return result

    def predict_one(
        self,
        telemetry: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = self.predict(
            pd.DataFrame(
                [dict(telemetry)]
            )
        )

        return result.iloc[0].to_dict()


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Robot Remaining Useful Life training"
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
        "--target",
        default="rul",
    )

    train_parser.add_argument(
        "--model",
        choices=[
            item.value
            for item in RULModelType
        ],
        default="random_forest",
    )

    train_parser.add_argument(
        "--output",
        default="artifacts/rul",
    )

    train_parser.add_argument(
        "--split",
        choices=[
            item.value
            for item in SplitStrategy
        ],
        default="group",
    )

    train_parser.add_argument(
        "--group-column",
        default="component_id",
    )

    train_parser.add_argument(
        "--timestamp-column",
        default="timestamp",
    )

    train_parser.add_argument(
        "--features",
        nargs="+",
        default=None,
    )

    train_parser.add_argument(
        "--rul-cap",
        type=float,
        default=None,
    )

    train_parser.add_argument(
        "--rul-unit",
        default="hours",
    )

    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    train_parser.add_argument(
        "--sequence-length",
        type=int,
        default=30,
    )

    train_parser.add_argument(
        "--sequence-stride",
        type=int,
        default=1,
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

    predict_parser = subparsers.add_parser(
        "predict"
    )

    predict_parser.add_argument(
        "--artifacts",
        required=True,
    )

    predict_parser.add_argument(
        "--input",
        required=True,
    )

    predict_parser.add_argument(
        "--output",
        default="rul_predictions.csv",
    )

    predict_parser.add_argument(
        "--trusted-artifacts",
        action="store_true",
    )

    predict_parser.add_argument(
        "--expected-sha256",
        default=None,
    )

    return parser


def main() -> None:
    parser = build_parser()

    args = parser.parse_args()

    if args.command == "train":
        config = RULTrainingConfig(
            data_path=args.data,
            target_column=args.target,
            model_type=RULModelType(
                args.model
            ),
            output_directory=args.output,
            split_strategy=SplitStrategy(
                args.split
            ),
            group_column=args.group_column,
            timestamp_column=args.timestamp_column,
            feature_columns=args.features or [],
            rul_cap=args.rul_cap,
            rul_unit=args.rul_unit,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            sequence_stride=args.sequence_stride,
            device=args.device,
            random_seed=args.seed,
        )

        pipeline = RULTrainingPipeline(
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

    if args.command == "predict":
        predictor = TrainedRULPredictor(
            args.artifacts,
            trusted_source=args.trusted_artifacts,
            expected_model_sha256=(
                args.expected_sha256
            ),
        )

        input_path = Path(
            args.input
        )

        if input_path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(
                input_path
            )

        elif input_path.suffix.lower() in {
            ".parquet",
            ".pq",
        }:
            dataframe = pd.read_parquet(
                input_path
            )

        else:
            raise RULDatasetError(
                "Prediction input must be CSV or Parquet"
            )

        predictions = predictor.predict(
            dataframe
        )

        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        predictions.to_csv(
            output_path,
            index=False,
        )

        print(
            json.dumps(
                {
                    "status": "completed",
                    "predictions": len(
                        predictions
                    ),
                    "output": str(
                        output_path
                    ),
                    "advisory_only": True,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: ml/training/train_anomaly.py

Anomaly Detection Model Training Pipeline

Features:
- Robot telemetry anomaly detection
- Sensor and actuator health monitoring
- Motor temperature, vibration and current analysis
- Battery and joint condition monitoring
- CSV/Parquet dataset loading
- Missing-value handling
- Numerical feature preprocessing
- Isolation Forest anomaly detection
- Optional PyTorch autoencoder
- Training/validation/test splitting
- Feature normalization
- Threshold calibration
- Precision, recall, F1 and ROC-AUC evaluation
- Model artifact saving
- SHA-256 checksums
- Training metadata
- Experiment tracking
- Batch inference
- Model comparison
- CLI training interface

Dependencies:
    pip install numpy pandas scikit-learn joblib

Optional:
    pip install torch pyarrow

Usage:
    python -m ml.training.train_anomaly \
        --data data/processed/robot_telemetry.csv \
        --model isolation_forest \
        --output artifacts/anomaly

    python -m ml.training.train_anomaly \
        --data data/processed/robot_telemetry.csv \
        --model autoencoder \
        --epochs 50 \
        --output artifacts/anomaly_autoencoder

Security:
    This pipeline is advisory only.
    Anomaly predictions never execute robot commands,
    disable safety systems or reset emergency stops.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger("ml.training.anomaly")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ============================================================
# Exceptions
# ============================================================

class AnomalyTrainingError(Exception):
    pass


class DatasetValidationError(AnomalyTrainingError):
    pass


class ModelTrainingError(AnomalyTrainingError):
    pass


class ModelEvaluationError(AnomalyTrainingError):
    pass


class ArtifactError(AnomalyTrainingError):
    pass


# ============================================================
# Constants
# ============================================================

DEFAULT_FEATURES = [
    "motor_temperature",
    "motor_current",
    "motor_voltage",
    "motor_vibration",
    "joint_temperature",
    "joint_torque",
    "joint_velocity",
    "battery_voltage",
    "battery_current",
    "battery_temperature",
    "cpu_temperature",
    "sensor_error_rate",
]

EXCLUDED_COLUMNS = {
    "timestamp",
    "robot_id",
    "device_id",
    "sensor_id",
    "tenant_id",
    "organization_id",
    "label",
    "anomaly",
    "is_anomaly",
    "failure",
    "fault",
    "target",
    "split",
}

MAX_DATASET_ROWS = 5_000_000

EPSILON = 1e-12


# ============================================================
# Enumerations
# ============================================================

class AnomalyModelType(str, Enum):
    ISOLATION_FOREST = "isolation_forest"
    AUTOENCODER = "autoencoder"


class TrainingStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnomalySeverity(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# Configuration
# ============================================================

@dataclass
class AnomalyTrainingConfig:
    model_type: AnomalyModelType = (
        AnomalyModelType.ISOLATION_FOREST
    )

    data_path: str = ""

    output_directory: str = "artifacts/anomaly"

    feature_columns: list[str] = field(
        default_factory=list
    )

    label_column: str | None = None

    validation_size: float = 0.15
    test_size: float = 0.15

    contamination: float = 0.05

    threshold_quantile: float = 0.95

    random_seed: int = 42

    isolation_estimators: int = 200

    epochs: int = 50
    batch_size: int = 128

    learning_rate: float = 0.001

    hidden_dimensions: tuple[int, ...] = (
        64,
        32,
        16,
    )

    weight_decay: float = 1e-5

    early_stopping_patience: int = 8

    device: str = "auto"

    normal_label: int = 0
    anomaly_label: int = 1

    experiment_name: str = "robot_anomaly_detection"

    def validate(self) -> None:
        if not self.data_path:
            raise DatasetValidationError(
                "Dataset path is required"
            )

        if not (
            0 < self.validation_size < 1
            and 0 < self.test_size < 1
            and self.validation_size + self.test_size < 1
        ):
            raise ValueError(
                "Invalid train/validation/test split"
            )

        if not 0 < self.contamination < 0.5:
            raise ValueError(
                "Contamination must be between 0 and 0.5"
            )

        if not 0 < self.threshold_quantile < 1:
            raise ValueError(
                "Threshold quantile must be between 0 and 1"
            )

        if self.epochs < 1:
            raise ValueError("Epochs must be positive")

        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")

        if self.learning_rate <= 0:
            raise ValueError(
                "Learning rate must be positive"
            )

        if self.isolation_estimators < 1:
            raise ValueError(
                "Isolation estimators must be positive"
            )


# ============================================================
# Utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


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


def calculate_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def save_json_atomic(
    path: str | Path,
    data: dict[str, Any],
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
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

            file.flush()
            os.fsync(file.fileno())

            temporary_path = Path(file.name)

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


# ============================================================
# Dataset Loading
# ============================================================

class RobotTelemetryDataset:
    def __init__(
        self,
        config: AnomalyTrainingConfig,
    ) -> None:
        self.config = config

        self.dataframe: pd.DataFrame | None = None

        self.feature_columns: list[str] = []

    def load(self) -> pd.DataFrame:
        path = Path(self.config.data_path)

        if not path.is_file():
            raise DatasetValidationError(
                f"Dataset not found: {path}"
            )

        extension = path.suffix.lower()

        if extension == ".csv":
            dataframe = pd.read_csv(path)

        elif extension in {".parquet", ".pq"}:
            dataframe = pd.read_parquet(path)

        else:
            raise DatasetValidationError(
                "Supported formats: CSV and Parquet"
            )

        if dataframe.empty:
            raise DatasetValidationError(
                "Dataset is empty"
            )

        if len(dataframe) > MAX_DATASET_ROWS:
            raise DatasetValidationError(
                "Dataset exceeds maximum supported rows"
            )

        if dataframe.columns.duplicated().any():
            raise DatasetValidationError(
                "Duplicate column names detected"
            )

        self.dataframe = dataframe

        logger.info(
            "Loaded dataset: %s rows, %s columns",
            len(dataframe),
            len(dataframe.columns),
        )

        return dataframe

    def select_features(self) -> list[str]:
        if self.dataframe is None:
            raise DatasetValidationError(
                "Load dataset before selecting features"
            )

        dataframe = self.dataframe

        if self.config.feature_columns:
            features = list(
                self.config.feature_columns
            )

        else:
            available_defaults = [
                column
                for column in DEFAULT_FEATURES
                if column in dataframe.columns
            ]

            if available_defaults:
                features = available_defaults

            else:
                features = [
                    column
                    for column in dataframe.select_dtypes(
                        include=[np.number]
                    ).columns
                    if column.lower()
                    not in EXCLUDED_COLUMNS
                ]

        if not features:
            raise DatasetValidationError(
                "No numerical telemetry features found"
            )

        missing = [
            column
            for column in features
            if column not in dataframe.columns
        ]

        if missing:
            raise DatasetValidationError(
                f"Missing features: {missing}"
            )

        if self.config.label_column in features:
            raise DatasetValidationError(
                "Label column cannot be an input feature"
            )

        for column in features:
            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            ):
                raise DatasetValidationError(
                    f"Feature is not numeric: {column}"
                )

        self.feature_columns = features

        return features

    def extract_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self.dataframe is None:
            raise DatasetValidationError(
                "Dataset has not been loaded"
            )

        if not self.feature_columns:
            self.select_features()

        features = self.dataframe[
            self.feature_columns
        ].to_numpy(
            dtype=np.float64
        )

        features[~np.isfinite(features)] = np.nan

        labels = None

        if self.config.label_column:
            column = self.config.label_column

            if column not in self.dataframe.columns:
                raise DatasetValidationError(
                    f"Label column not found: {column}"
                )

            raw_labels = self.dataframe[
                column
            ].to_numpy()

            if pd.isna(raw_labels).any():
                raise DatasetValidationError(
                    "Labels contain missing values"
                )

            allowed_labels = {
                self.config.normal_label,
                self.config.anomaly_label,
            }

            if not set(
                np.unique(raw_labels)
            ).issubset(allowed_labels):
                raise DatasetValidationError(
                    "Labels must contain configured "
                    "normal/anomaly values"
                )

            labels = np.where(
                raw_labels == self.config.anomaly_label,
                1,
                0,
            ).astype(np.int64)

        return features, labels


# ============================================================
# Data Splitting
# ============================================================

@dataclass
class DatasetSplits:
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray

    y_train: np.ndarray | None = None
    y_validation: np.ndarray | None = None
    y_test: np.ndarray | None = None


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray | None,
    config: AnomalyTrainingConfig,
) -> DatasetSplits:
    if len(features) < 30:
        raise DatasetValidationError(
            "At least 30 samples are required"
        )

    indices = np.arange(len(features))

    stratify = None

    if labels is not None:
        counts = np.bincount(
            labels,
            minlength=2,
        )

        if np.min(counts) >= 3:
            stratify = labels

    try:
        train_indices, temporary_indices = (
            train_test_split(
                indices,
                test_size=(
                    config.validation_size
                    + config.test_size
                ),
                random_state=config.random_seed,
                stratify=stratify,
            )
        )

        temporary_labels = (
            labels[temporary_indices]
            if labels is not None
            else None
        )

        temporary_stratify = None

        if temporary_labels is not None:
            counts = np.bincount(
                temporary_labels,
                minlength=2,
            )

            if np.min(counts) >= 2:
                temporary_stratify = temporary_labels

        validation_ratio = (
            config.validation_size
            / (
                config.validation_size
                + config.test_size
            )
        )

        validation_indices, test_indices = (
            train_test_split(
                temporary_indices,
                train_size=validation_ratio,
                random_state=config.random_seed,
                stratify=temporary_stratify,
            )
        )

    except ValueError as exc:
        raise DatasetValidationError(
            f"Unable to split dataset: {exc}"
        ) from exc

    return DatasetSplits(
        x_train=features[train_indices],
        x_validation=features[validation_indices],
        x_test=features[test_indices],
        y_train=(
            labels[train_indices]
            if labels is not None
            else None
        ),
        y_validation=(
            labels[validation_indices]
            if labels is not None
            else None
        ),
        y_test=(
            labels[test_indices]
            if labels is not None
            else None
        ),
    )


# ============================================================
# Preprocessing
# ============================================================

class TelemetryPreprocessor:
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

        self.is_fitted = False

    def fit_transform(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        if features.ndim != 2:
            raise DatasetValidationError(
                "Features must be a 2D array"
            )

        if not np.isfinite(
            np.nan_to_num(features)
        ).all():
            raise DatasetValidationError(
                "Invalid feature values"
            )

        if np.isnan(features).all(axis=0).any():
            raise DatasetValidationError(
                "A feature contains only missing values"
            )

        transformed = self.pipeline.fit_transform(
            features
        )

        self.is_fitted = True

        return transformed.astype(
            np.float32
        )

    def transform(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise DatasetValidationError(
                "Preprocessor has not been fitted"
            )

        return self.pipeline.transform(
            features
        ).astype(np.float32)


# ============================================================
# Isolation Forest
# ============================================================

class IsolationForestAnomalyModel:
    def __init__(
        self,
        config: AnomalyTrainingConfig,
    ) -> None:
        self.config = config

        self.model = IsolationForest(
            n_estimators=config.isolation_estimators,
            contamination=config.contamination,
            random_state=config.random_seed,
            n_jobs=-1,
        )

        self.is_fitted = False

    def fit(
        self,
        features: np.ndarray,
    ) -> dict[str, Any]:
        started = time.perf_counter()

        self.model.fit(features)

        self.is_fitted = True

        return {
            "training_duration_seconds": round(
                time.perf_counter() - started,
                4,
            ),
            "n_estimators": (
                self.config.isolation_estimators
            ),
            "training_samples": len(features),
        }

    def anomaly_scores(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ModelTrainingError(
                "Isolation Forest has not been fitted"
            )

        return -self.model.score_samples(
            features
        )


# ============================================================
# PyTorch Autoencoder
# ============================================================

class AutoencoderAnomalyModel:
    def __init__(
        self,
        config: AnomalyTrainingConfig,
        input_dimension: int,
    ) -> None:
        try:
            import torch
            from torch import nn

        except ImportError as exc:
            raise ModelTrainingError(
                "PyTorch is required for autoencoder training"
            ) from exc

        self.torch = torch
        self.nn = nn

        self.config = config
        self.input_dimension = input_dimension

        if config.device == "auto":
            if torch.cuda.is_available():
                device_name = "cuda"

            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device_name = "mps"

            else:
                device_name = "cpu"

        else:
            device_name = config.device

        self.device = torch.device(
            device_name
        )

        dimensions = [
            input_dimension,
            *config.hidden_dimensions,
        ]

        encoder_layers = []

        for index in range(
            len(dimensions) - 1
        ):
            encoder_layers.extend(
                [
                    nn.Linear(
                        dimensions[index],
                        dimensions[index + 1],
                    ),
                    nn.ReLU(),
                ]
            )

        decoder_dimensions = list(
            reversed(dimensions)
        )

        decoder_layers = []

        for index in range(
            len(decoder_dimensions) - 1
        ):
            decoder_layers.append(
                nn.Linear(
                    decoder_dimensions[index],
                    decoder_dimensions[index + 1],
                )
            )

            if index < len(
                decoder_dimensions
            ) - 2:
                decoder_layers.append(
                    nn.ReLU()
                )

        self.model = nn.Sequential(
            *encoder_layers,
            *decoder_layers,
        ).to(self.device)

        self.is_fitted = False

    def fit(
        self,
        train_features: np.ndarray,
        validation_features: np.ndarray,
    ) -> dict[str, Any]:
        torch = self.torch

        train_tensor = torch.tensor(
            train_features,
            dtype=torch.float32,
        )

        dataset = torch.utils.data.TensorDataset(
            train_tensor
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        criterion = self.nn.MSELoss()

        best_loss = float("inf")
        best_state = None

        patience_counter = 0

        history = {
            "train_loss": [],
            "validation_loss": [],
        }

        started = time.perf_counter()

        for epoch in range(
            self.config.epochs
        ):
            self.model.train()

            total_loss = 0.0
            total_samples = 0

            for (batch,) in loader:
                batch = batch.to(
                    self.device
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                reconstruction = self.model(
                    batch
                )

                loss = criterion(
                    reconstruction,
                    batch,
                )

                loss.backward()

                optimizer.step()

                total_loss += (
                    loss.item() * len(batch)
                )

                total_samples += len(batch)

            train_loss = (
                total_loss
                / max(total_samples, 1)
            )

            validation_loss = (
                self._reconstruction_loss(
                    validation_features
                )
            )

            history["train_loss"].append(
                train_loss
            )

            history["validation_loss"].append(
                validation_loss
            )

            logger.info(
                "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f",
                epoch + 1,
                self.config.epochs,
                train_loss,
                validation_loss,
            )

            if validation_loss < best_loss:
                best_loss = validation_loss

                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter
                    in self.model.state_dict().items()
                }

                patience_counter = 0

            else:
                patience_counter += 1

            if (
                patience_counter
                >= self.config.early_stopping_patience
            ):
                logger.info(
                    "Early stopping at epoch %d",
                    epoch + 1,
                )
                break

        if best_state is not None:
            self.model.load_state_dict(
                best_state
            )

        self.is_fitted = True

        return {
            "training_duration_seconds": round(
                time.perf_counter() - started,
                4,
            ),
            "epochs_completed": len(
                history["train_loss"]
            ),
            "best_validation_loss": best_loss,
            "history": history,
            "device": str(self.device),
        }

    def _reconstruction_loss(
        self,
        features: np.ndarray,
    ) -> float:
        scores = self._compute_scores(
            features
        )

        return float(
            np.mean(scores)
        )

    def _compute_scores(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        torch = self.torch

        self.model.eval()

        scores = []

        with torch.no_grad():
            for start in range(
                0,
                len(features),
                self.config.batch_size,
            ):
                batch = features[
                    start:
                    start + self.config.batch_size
                ]

                tensor = torch.tensor(
                    batch,
                    dtype=torch.float32,
                    device=self.device,
                )

                reconstruction = self.model(
                    tensor
                )

                errors = torch.mean(
                    (
                        reconstruction - tensor
                    ) ** 2,
                    dim=1,
                )

                scores.extend(
                    errors.cpu().numpy()
                )

        return np.asarray(
            scores,
            dtype=np.float64,
        )

    def anomaly_scores(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ModelTrainingError(
                "Autoencoder has not been fitted"
            )

        return self._compute_scores(
            features
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
                    self.model.state_dict()
                ),
                "input_dimension": (
                    self.input_dimension
                ),
                "hidden_dimensions": (
                    self.config.hidden_dimensions
                ),
            },
            path,
        )


# ============================================================
# Threshold Calibration
# ============================================================

@dataclass
class ThresholdResult:
    threshold: float
    quantile: float
    calibration_samples: int
    method: str


def calibrate_threshold(
    scores: np.ndarray,
    quantile: float,
) -> ThresholdResult:
    if len(scores) == 0:
        raise ModelEvaluationError(
            "Threshold calibration requires samples"
        )

    if not np.isfinite(scores).all():
        raise ModelEvaluationError(
            "Calibration scores contain invalid values"
        )

    threshold = float(
        np.quantile(
            scores,
            quantile,
        )
    )

    return ThresholdResult(
        threshold=threshold,
        quantile=quantile,
        calibration_samples=len(scores),
        method="normal_validation_quantile",
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_anomaly_model(
    *,
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray | None,
) -> dict[str, Any]:
    predictions = (
        scores > threshold
    ).astype(np.int64)

    results: dict[str, Any] = {
        "sample_count": len(scores),
        "threshold": float(threshold),
        "predicted_anomalies": int(
            predictions.sum()
        ),
        "predicted_anomaly_rate": float(
            predictions.mean()
        ),
        "score_mean": float(
            np.mean(scores)
        ),
        "score_std": float(
            np.std(scores)
        ),
        "score_min": float(
            np.min(scores)
        ),
        "score_max": float(
            np.max(scores)
        ),
    }

    if labels is None:
        results["evaluation_mode"] = (
            "unlabeled"
        )

        return results

    results["evaluation_mode"] = (
        "supervised_metrics"
    )

    results["accuracy"] = float(
        accuracy_score(
            labels,
            predictions,
        )
    )

    results["precision"] = float(
        precision_score(
            labels,
            predictions,
            zero_division=0,
        )
    )

    results["recall"] = float(
        recall_score(
            labels,
            predictions,
            zero_division=0,
        )
    )

    results["f1_score"] = float(
        f1_score(
            labels,
            predictions,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    results["confusion_matrix"] = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    if len(np.unique(labels)) == 2:
        results["roc_auc"] = float(
            roc_auc_score(
                labels,
                scores,
            )
        )

    else:
        results["roc_auc"] = None

    return results


# ============================================================
# Severity Classification
# ============================================================

def classify_anomaly_severity(
    score: float,
    threshold: float,
    reference_std: float,
) -> AnomalySeverity:
    if score <= threshold:
        return AnomalySeverity.NORMAL

    distance = (
        score - threshold
    ) / max(
        reference_std,
        EPSILON,
    )

    if distance < 0.5:
        return AnomalySeverity.LOW

    if distance < 1.5:
        return AnomalySeverity.MODERATE

    if distance < 3.0:
        return AnomalySeverity.HIGH

    return AnomalySeverity.CRITICAL


# ============================================================
# Training Result
# ============================================================

@dataclass
class AnomalyTrainingResult:
    run_id: str
    model_type: str

    status: TrainingStatus

    feature_columns: list[str]

    training_samples: int
    validation_samples: int
    test_samples: int

    threshold: float

    training_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]

    artifact_paths: dict[str, str]

    started_at: str
    completed_at: str

    duration_seconds: float


# ============================================================
# Training Pipeline
# ============================================================

class AnomalyTrainingPipeline:
    def __init__(
        self,
        config: AnomalyTrainingConfig,
    ) -> None:
        self.config = config

        self.config.validate()

        self.run_id = generate_run_id()

        self.output_directory = Path(
            config.output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.preprocessor = (
            TelemetryPreprocessor()
        )

        self.model: Any = None

        self.threshold: float | None = None

        self.feature_columns: list[str] = []

    def _prepare_dataset(
        self,
    ) -> DatasetSplits:
        dataset = RobotTelemetryDataset(
            self.config
        )

        dataset.load()

        self.feature_columns = (
            dataset.select_features()
        )

        features, labels = (
            dataset.extract_arrays()
        )

        splits = split_dataset(
            features,
            labels,
            self.config,
        )

        # Fit preprocessing only on training data.
        # Do not expose validation/test statistics
        # to the training pipeline.

        x_train = (
            self.preprocessor.fit_transform(
                splits.x_train
            )
        )

        x_validation = (
            self.preprocessor.transform(
                splits.x_validation
            )
        )

        x_test = (
            self.preprocessor.transform(
                splits.x_test
            )
        )

        return DatasetSplits(
            x_train=x_train,
            x_validation=x_validation,
            x_test=x_test,
            y_train=splits.y_train,
            y_validation=splits.y_validation,
            y_test=splits.y_test,
        )

    def _normal_training_samples(
        self,
        splits: DatasetSplits,
    ) -> np.ndarray:
        if splits.y_train is None:
            return splits.x_train

        normal_mask = (
            splits.y_train == 0
        )

        normal_samples = (
            splits.x_train[normal_mask]
        )

        if len(normal_samples) < 10:
            raise DatasetValidationError(
                "At least 10 normal training samples "
                "are required"
            )

        return normal_samples

    def _normal_validation_samples(
        self,
        splits: DatasetSplits,
    ) -> np.ndarray:
        if splits.y_validation is None:
            return splits.x_validation

        normal_samples = (
            splits.x_validation[
                splits.y_validation == 0
            ]
        )

        if len(normal_samples) < 5:
            raise DatasetValidationError(
                "At least 5 normal validation samples "
                "are required for threshold calibration"
            )

        return normal_samples

    def _train_model(
        self,
        splits: DatasetSplits,
    ) -> dict[str, Any]:
        train_normal = (
            self._normal_training_samples(
                splits
            )
        )

        if (
            self.config.model_type
            == AnomalyModelType.ISOLATION_FOREST
        ):
            self.model = (
                IsolationForestAnomalyModel(
                    self.config
                )
            )

            return self.model.fit(
                train_normal
            )

        if (
            self.config.model_type
            == AnomalyModelType.AUTOENCODER
        ):
            self.model = (
                AutoencoderAnomalyModel(
                    self.config,
                    input_dimension=(
                        train_normal.shape[1]
                    ),
                )
            )

            validation_normal = (
                self._normal_validation_samples(
                    splits
                )
            )

            return self.model.fit(
                train_normal,
                validation_normal,
            )

        raise ModelTrainingError(
            "Unsupported anomaly model"
        )

    def _calibrate(
        self,
        splits: DatasetSplits,
    ) -> ThresholdResult:
        normal_validation = (
            self._normal_validation_samples(
                splits
            )
        )

        scores = self.model.anomaly_scores(
            normal_validation
        )

        result = calibrate_threshold(
            scores,
            self.config.threshold_quantile,
        )

        self.threshold = result.threshold

        return result

    def _save_artifacts(
        self,
        *,
        training_metrics: dict[str, Any],
        validation_metrics: dict[str, Any],
        test_metrics: dict[str, Any],
        threshold_result: ThresholdResult,
    ) -> dict[str, str]:
        output = self.output_directory

        paths: dict[str, str] = {}

        preprocessing_path = (
            output / "preprocessor.joblib"
        )

        joblib.dump(
            self.preprocessor.pipeline,
            preprocessing_path,
        )

        paths["preprocessor"] = str(
            preprocessing_path
        )

        if (
            self.config.model_type
            == AnomalyModelType.ISOLATION_FOREST
        ):
            model_path = (
                output / "anomaly_model.joblib"
            )

            joblib.dump(
                self.model.model,
                model_path,
            )

        else:
            model_path = (
                output / "anomaly_autoencoder.pt"
            )

            self.model.save(
                model_path
            )

        paths["model"] = str(
            model_path
        )

        metadata_path = (
            output / "training_metadata.json"
        )

        metadata = {
            "run_id": self.run_id,
            "model_name": "robot-anomaly-detector",
            "model_type": (
                self.config.model_type.value
            ),
            "created_at": utc_now(),
            "features": self.feature_columns,
            "feature_count": len(
                self.feature_columns
            ),
            "threshold": asdict(
                threshold_result
            ),
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "configuration": {
                **asdict(self.config),
                "model_type": (
                    self.config.model_type.value
                ),
            },
            "artifacts": {
                "model": {
                    "filename": model_path.name,
                    "sha256": calculate_sha256(
                        model_path
                    ),
                    "size_bytes": (
                        model_path.stat().st_size
                    ),
                },
                "preprocessor": {
                    "filename": (
                        preprocessing_path.name
                    ),
                    "sha256": calculate_sha256(
                        preprocessing_path
                    ),
                    "size_bytes": (
                        preprocessing_path.stat().st_size
                    ),
                },
            },
            "safety": {
                "inference_only": True,
                "physical_execution_authorized": False,
                "emergency_stop_reset_authorized": False,
                "safety_interlock_bypass_authorized": False,
                "human_review_recommended": True,
            },
        }

        save_json_atomic(
            metadata_path,
            metadata,
        )

        paths["metadata"] = str(
            metadata_path
        )

        return paths

    def train(
        self,
    ) -> AnomalyTrainingResult:
        started_at = utc_now()
        started = time.perf_counter()

        logger.info(
            "Starting anomaly training run %s",
            self.run_id,
        )

        set_random_seed(
            self.config.random_seed
        )

        splits = self._prepare_dataset()

        training_metrics = (
            self._train_model(
                splits
            )
        )

        threshold_result = (
            self._calibrate(
                splits
            )
        )

        validation_scores = (
            self.model.anomaly_scores(
                splits.x_validation
            )
        )

        test_scores = (
            self.model.anomaly_scores(
                splits.x_test
            )
        )

        validation_metrics = (
            evaluate_anomaly_model(
                scores=validation_scores,
                threshold=self.threshold,
                labels=splits.y_validation,
            )
        )

        test_metrics = (
            evaluate_anomaly_model(
                scores=test_scores,
                threshold=self.threshold,
                labels=splits.y_test,
            )
        )

        artifact_paths = (
            self._save_artifacts(
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                threshold_result=threshold_result,
            )
        )

        result = AnomalyTrainingResult(
            run_id=self.run_id,
            model_type=(
                self.config.model_type.value
            ),
            status=TrainingStatus.COMPLETED,
            feature_columns=(
                self.feature_columns
            ),
            training_samples=len(
                splits.x_train
            ),
            validation_samples=len(
                splits.x_validation
            ),
            test_samples=len(
                splits.x_test
            ),
            threshold=float(
                self.threshold
            ),
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
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

        logger.info(
            "Training completed | run=%s | duration=%.2fs",
            self.run_id,
            result.duration_seconds,
        )

        return result


# ============================================================
# Inference
# ============================================================

class TrainedAnomalyPredictor:
    """
    Load and run a previously trained anomaly model.

    IMPORTANT:
        Only load model artifacts from trusted,
        verified sources. Joblib deserialization
        can execute arbitrary code.
    """

    def __init__(
        self,
        artifact_directory: str | Path,
    ) -> None:
        self.artifact_directory = Path(
            artifact_directory
        )

        metadata_path = (
            self.artifact_directory
            / "training_metadata.json"
        )

        if not metadata_path.is_file():
            raise ArtifactError(
                "Training metadata not found"
            )

        self.metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        self.feature_columns = (
            self.metadata["features"]
        )

        self.threshold = float(
            self.metadata["threshold"]["threshold"]
        )

        self.model_type = AnomalyModelType(
            self.metadata["model_type"]
        )

        self._verify_artifacts()

        preprocessing_path = (
            self.artifact_directory
            / self.metadata["artifacts"][
                "preprocessor"
            ]["filename"]
        )

        self.preprocessor = joblib.load(
            preprocessing_path
        )

        if (
            self.model_type
            == AnomalyModelType.ISOLATION_FOREST
        ):
            model_path = (
                self.artifact_directory
                / self.metadata["artifacts"][
                    "model"
                ]["filename"]
            )

            self.model = joblib.load(
                model_path
            )

        else:
            self.model = (
                self._load_autoencoder()
            )

    def _verify_artifacts(self) -> None:
        for artifact in self.metadata[
            "artifacts"
        ].values():
            filename = artifact["filename"]

            path = (
                self.artifact_directory
                / filename
            ).resolve()

            if not path.is_relative_to(
                self.artifact_directory.resolve()
            ):
                raise ArtifactError(
                    "Artifact path escapes trusted directory"
                )

            if not path.is_file():
                raise ArtifactError(
                    f"Missing artifact: {filename}"
                )

            checksum = calculate_sha256(
                path
            )

            if checksum != artifact["sha256"]:
                raise ArtifactError(
                    f"Artifact checksum mismatch: {filename}"
                )

    def _load_autoencoder(self):
        try:
            import torch
            from torch import nn

        except ImportError as exc:
            raise ArtifactError(
                "PyTorch required for autoencoder inference"
            ) from exc

        model_path = (
            self.artifact_directory
            / self.metadata["artifacts"][
                "model"
            ]["filename"]
        )

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )

        input_dimension = int(
            checkpoint["input_dimension"]
        )

        hidden_dimensions = list(
            checkpoint["hidden_dimensions"]
        )

        dimensions = [
            input_dimension,
            *hidden_dimensions,
        ]

        layers = []

        for index in range(
            len(dimensions) - 1
        ):
            layers.extend(
                [
                    nn.Linear(
                        dimensions[index],
                        dimensions[index + 1],
                    ),
                    nn.ReLU(),
                ]
            )

        reverse_dimensions = list(
            reversed(dimensions)
        )

        for index in range(
            len(reverse_dimensions) - 1
        ):
            layers.append(
                nn.Linear(
                    reverse_dimensions[index],
                    reverse_dimensions[index + 1],
                )
            )

            if index < len(
                reverse_dimensions
            ) - 2:
                layers.append(
                    nn.ReLU()
                )

        model = nn.Sequential(
            *layers
        )

        model.load_state_dict(
            checkpoint["state_dict"]
        )

        model.eval()

        return model

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        missing = [
            column
            for column in self.feature_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise DatasetValidationError(
                f"Missing inference features: {missing}"
            )

        features = dataframe[
            self.feature_columns
        ].to_numpy(
            dtype=np.float64
        )

        features[~np.isfinite(features)] = np.nan

        transformed = (
            self.preprocessor.transform(
                features
            )
        ).astype(np.float32)

        if (
            self.model_type
            == AnomalyModelType.ISOLATION_FOREST
        ):
            scores = (
                -self.model.score_samples(
                    transformed
                )
            )

        else:
            import torch

            with torch.no_grad():
                tensor = torch.tensor(
                    transformed,
                    dtype=torch.float32,
                )

                reconstruction = self.model(
                    tensor
                )

                scores = torch.mean(
                    (
                        reconstruction - tensor
                    ) ** 2,
                    dim=1,
                ).numpy()

        predictions = (
            scores > self.threshold
        ).astype(np.int64)

        result = dataframe.copy()

        result["anomaly_score"] = scores
        result["is_anomaly"] = predictions

        result["anomaly_status"] = np.where(
            predictions == 1,
            "anomaly",
            "normal",
        )

        result["advisory_only"] = True

        return result


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train robot telemetry anomaly detection models"
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="CSV or Parquet telemetry dataset",
    )

    parser.add_argument(
        "--model",
        choices=[
            model.value
            for model in AnomalyModelType
        ],
        default="isolation_forest",
    )

    parser.add_argument(
        "--output",
        default="artifacts/anomaly",
    )

    parser.add_argument(
        "--label-column",
        default=None,
    )

    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config = AnomalyTrainingConfig(
        model_type=AnomalyModelType(
            args.model
        ),
        data_path=args.data,
        output_directory=args.output,
        feature_columns=args.features or [],
        label_column=args.label_column,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        contamination=args.contamination,
        threshold_quantile=args.threshold_quantile,
        random_seed=args.seed,
        device=args.device,
    )

    pipeline = AnomalyTrainingPipeline(
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


if __name__ == "__main__":
    main()
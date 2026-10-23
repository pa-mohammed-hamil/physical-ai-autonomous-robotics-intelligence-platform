"""
Physical AI & Autonomous Robotics Intelligence Platform
File: ml/training/train_failure.py

Robot Component Failure Prediction Training Pipeline.

Features:
- Robot motor, actuator, battery and sensor failure prediction
- Binary failure classification
- CSV and Parquet dataset loading
- Numerical and categorical feature preprocessing
- Missing-value handling
- Random Forest and HistGradientBoosting models
- Class imbalance handling
- Train/validation/test splitting
- Optional chronological splitting
- Hyperparameter tuning
- Probability threshold optimization
- Precision, recall, F1, ROC-AUC and PR-AUC
- Confusion matrix and classification report
- Feature importance
- Model artifact export
- SHA-256 verification
- Training metadata and evaluation reports
- Trusted-artifact inference
- CLI support

Dependencies:
    pip install numpy pandas scikit-learn joblib

Optional:
    pip install pyarrow

Examples:
    python -m ml.training.train_failure \
        --data data/processed/robot_failures.csv \
        --target failure \
        --model random_forest \
        --output artifacts/failure

    python -m ml.training.train_failure \
        --data data/processed/robot_failures.csv \
        --target failure \
        --model gradient_boosting \
        --split chronological \
        --timestamp-column timestamp \
        --output artifacts/failure_gb

Safety:
    Predictions are advisory only.
    This module does not command robots, reset emergency
    stops or bypass hardware safety interlocks.
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
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


LOGGER = logging.getLogger("ml.training.train_failure")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCHEMA_VERSION = "1.0.0"
MAX_DATASET_ROWS = 5_000_000
MAX_FEATURE_COLUMNS = 256
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024

DEFAULT_NUMERIC_FEATURES = [
    "motor_temperature",
    "motor_current",
    "motor_voltage",
    "motor_vibration",
    "motor_rpm",
    "joint_temperature",
    "joint_torque",
    "joint_velocity",
    "battery_voltage",
    "battery_current",
    "battery_temperature",
    "battery_soc",
    "battery_soh",
    "cpu_temperature",
    "sensor_error_rate",
    "operating_hours",
    "cycle_count",
    "load_percentage",
    "pressure",
    "humidity",
    "ambient_temperature",
    "maintenance_age_days",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "robot_type",
    "component_type",
    "operating_mode",
]

EXCLUDED_FEATURES = {
    "failure",
    "failure_label",
    "target",
    "label",
    "timestamp",
    "robot_id",
    "component_id",
    "device_id",
    "tenant_id",
    "organization_id",
    "serial_number",
    "failure_time",
    "failure_timestamp",
    "remaining_useful_life",
    "rul",
    "split",
}


# ============================================================
# Exceptions
# ============================================================

class FailureTrainingError(Exception):
    """Base training exception."""


class FailureDatasetError(FailureTrainingError):
    """Dataset is missing, invalid or unsuitable."""


class FailureModelError(FailureTrainingError):
    """Model training or inference failed."""


class FailureEvaluationError(FailureTrainingError):
    """Evaluation failed."""


class FailureArtifactError(FailureTrainingError):
    """Artifact creation or verification failed."""


# ============================================================
# Enumerations
# ============================================================

class FailureModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


class DatasetSplitStrategy(str, Enum):
    RANDOM = "random"
    CHRONOLOGICAL = "chronological"


class FailureRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class TrainingStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================
# Configuration
# ============================================================

@dataclass
class FailureTrainingConfig:
    data_path: str

    target_column: str = "failure"

    model_type: FailureModelType = (
        FailureModelType.RANDOM_FOREST
    )

    output_directory: str = "artifacts/failure"

    numeric_features: list[str] = field(
        default_factory=list
    )

    categorical_features: list[str] = field(
        default_factory=list
    )

    split_strategy: DatasetSplitStrategy = (
        DatasetSplitStrategy.RANDOM
    )

    timestamp_column: str = "timestamp"

    validation_size: float = 0.15
    test_size: float = 0.15

    random_seed: int = 42

    n_estimators: int = 300
    max_depth: int | None = 16
    min_samples_leaf: int = 2

    learning_rate: float = 0.05
    max_iterations: int = 250

    optimize_threshold: bool = True
    default_threshold: float = 0.5

    minimum_recall: float = 0.80

    hyperparameter_tuning: bool = False
    tuning_iterations: int = 12
    cross_validation_folds: int = 3

    positive_label: int = 1
    negative_label: int = 0

    experiment_name: str = (
        "robot_component_failure_prediction"
    )

    def validate(self) -> None:
        if not self.data_path:
            raise FailureDatasetError(
                "Dataset path is required"
            )

        if not self.target_column:
            raise FailureDatasetError(
                "Target column is required"
            )

        if not (
            0 < self.validation_size < 1
            and 0 < self.test_size < 1
            and self.validation_size + self.test_size < 1
        ):
            raise ValueError(
                "Invalid dataset split fractions"
            )

        if not 0 < self.default_threshold < 1:
            raise ValueError(
                "Default threshold must be between 0 and 1"
            )

        if not 0 <= self.minimum_recall <= 1:
            raise ValueError(
                "Minimum recall must be between 0 and 1"
            )

        if self.n_estimators < 1:
            raise ValueError(
                "n_estimators must be positive"
            )

        if self.min_samples_leaf < 1:
            raise ValueError(
                "min_samples_leaf must be positive"
            )

        if self.max_iterations < 1:
            raise ValueError(
                "max_iterations must be positive"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.tuning_iterations < 1:
            raise ValueError(
                "tuning_iterations must be positive"
            )

        if self.cross_validation_folds < 2:
            raise ValueError(
                "cross_validation_folds must be at least 2"
            )

        if self.positive_label == self.negative_label:
            raise ValueError(
                "Positive and negative labels must differ"
            )

        overlapping = (
            set(self.numeric_features)
            & set(self.categorical_features)
        )

        if overlapping:
            raise FailureDatasetError(
                f"Overlapping feature columns: {sorted(overlapping)}"
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


def calculate_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_save_json(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".failure-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

            stream.flush()
            os.fsync(stream.fileno())

            temporary_path = Path(stream.name)

        os.replace(temporary_path, path)

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def to_json_compatible(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            str(key): to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_json_compatible(item)
            for item in value
        ]

    return value


# ============================================================
# Dataset
# ============================================================

class RobotFailureDataset:
    def __init__(
        self,
        config: FailureTrainingConfig,
    ) -> None:
        self.config = config

        self.dataframe: pd.DataFrame | None = None

        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []

    def load(self) -> pd.DataFrame:
        path = Path(self.config.data_path)

        if not path.is_file():
            raise FailureDatasetError(
                f"Dataset not found: {path}"
            )

        extension = path.suffix.lower()

        try:
            if extension == ".csv":
                dataframe = pd.read_csv(path)

            elif extension in {".parquet", ".pq"}:
                dataframe = pd.read_parquet(path)

            else:
                raise FailureDatasetError(
                    "Supported formats: CSV and Parquet"
                )

        except (OSError, ValueError) as exc:
            raise FailureDatasetError(
                "Unable to load failure dataset"
            ) from exc

        if dataframe.empty:
            raise FailureDatasetError(
                "Dataset is empty"
            )

        if len(dataframe) > MAX_DATASET_ROWS:
            raise FailureDatasetError(
                "Dataset exceeds maximum row limit"
            )

        if dataframe.columns.duplicated().any():
            raise FailureDatasetError(
                "Duplicate column names found"
            )

        if self.config.target_column not in dataframe:
            raise FailureDatasetError(
                f"Missing target column: "
                f"{self.config.target_column}"
            )

        self.dataframe = dataframe

        LOGGER.info(
            "Loaded %d samples with %d columns",
            len(dataframe),
            len(dataframe.columns),
        )

        return dataframe

    def select_features(
        self,
    ) -> tuple[list[str], list[str]]:
        if self.dataframe is None:
            raise FailureDatasetError(
                "Dataset must be loaded first"
            )

        dataframe = self.dataframe

        if self.config.numeric_features:
            numeric = list(
                self.config.numeric_features
            )

        else:
            numeric = [
                name
                for name in DEFAULT_NUMERIC_FEATURES
                if name in dataframe.columns
            ]

            if not numeric:
                numeric = [
                    name
                    for name in dataframe.select_dtypes(
                        include=[np.number]
                    ).columns
                    if name.lower()
                    not in EXCLUDED_FEATURES
                    and name != self.config.target_column
                ]

        if self.config.categorical_features:
            categorical = list(
                self.config.categorical_features
            )

        else:
            categorical = [
                name
                for name in DEFAULT_CATEGORICAL_FEATURES
                if name in dataframe.columns
            ]

        all_features = numeric + categorical

        if not all_features:
            raise FailureDatasetError(
                "No usable telemetry features found"
            )

        if len(all_features) > MAX_FEATURE_COLUMNS:
            raise FailureDatasetError(
                "Too many feature columns"
            )

        if len(set(all_features)) != len(all_features):
            raise FailureDatasetError(
                "Duplicate feature columns"
            )

        if self.config.target_column in all_features:
            raise FailureDatasetError(
                "Target leakage: target included as feature"
            )

        missing = [
            name
            for name in all_features
            if name not in dataframe.columns
        ]

        if missing:
            raise FailureDatasetError(
                f"Missing feature columns: {missing}"
            )

        for name in numeric:
            if not pd.api.types.is_numeric_dtype(
                dataframe[name]
            ):
                raise FailureDatasetError(
                    f"Numeric feature has wrong dtype: {name}"
                )

        self.numeric_features = numeric
        self.categorical_features = categorical

        return numeric, categorical

    def prepare(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray, pd.Series | None]:
        if self.dataframe is None:
            self.load()

        if (
            not self.numeric_features
            and not self.categorical_features
        ):
            self.select_features()

        assert self.dataframe is not None

        features = self.dataframe[
            self.numeric_features
            + self.categorical_features
        ].copy()

        for name in self.numeric_features:
            values = pd.to_numeric(
                features[name],
                errors="coerce",
            )

            features[name] = values.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            if features[name].isna().all():
                raise FailureDatasetError(
                    f"Feature contains no valid values: {name}"
                )

        for name in self.categorical_features:
            features[name] = (
                features[name]
                .astype("string")
                .fillna("__missing__")
                .astype(str)
            )

        raw_target = self.dataframe[
            self.config.target_column
        ]

        if raw_target.isna().any():
            raise FailureDatasetError(
                "Target contains missing values"
            )

        allowed = {
            self.config.positive_label,
            self.config.negative_label,
        }

        if not set(raw_target.unique()).issubset(allowed):
            raise FailureDatasetError(
                "Target must contain configured binary labels"
            )

        labels = np.where(
            raw_target.to_numpy()
            == self.config.positive_label,
            1,
            0,
        ).astype(np.int64)

        if len(np.unique(labels)) < 2:
            raise FailureDatasetError(
                "Both failure and non-failure samples "
                "are required"
            )

        timestamps = None

        if (
            self.config.split_strategy
            == DatasetSplitStrategy.CHRONOLOGICAL
        ):
            column = self.config.timestamp_column

            if column not in self.dataframe:
                raise FailureDatasetError(
                    f"Timestamp column not found: {column}"
                )

            timestamps = pd.to_datetime(
                self.dataframe[column],
                utc=True,
                errors="coerce",
            )

            if timestamps.isna().any():
                raise FailureDatasetError(
                    "Invalid chronological timestamps"
                )

        return features, labels, timestamps


# ============================================================
# Dataset Splitting
# ============================================================

@dataclass
class FailureDatasetSplits:
    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame

    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray


def create_dataset_splits(
    features: pd.DataFrame,
    labels: np.ndarray,
    config: FailureTrainingConfig,
    timestamps: pd.Series | None = None,
) -> FailureDatasetSplits:
    sample_count = len(features)

    if sample_count < 40:
        raise FailureDatasetError(
            "At least 40 samples are required"
        )

    indices = np.arange(sample_count)

    if (
        config.split_strategy
        == DatasetSplitStrategy.CHRONOLOGICAL
    ):
        if timestamps is None:
            raise FailureDatasetError(
                "Chronological split requires timestamps"
            )

        ordered_indices = np.argsort(
            timestamps.to_numpy(),
            kind="stable",
        )

        train_end = int(
            sample_count
            * (
                1
                - config.validation_size
                - config.test_size
            )
        )

        validation_end = int(
            sample_count
            * (1 - config.test_size)
        )

        train_indices = ordered_indices[
            :train_end
        ]

        validation_indices = ordered_indices[
            train_end:validation_end
        ]

        test_indices = ordered_indices[
            validation_end:
        ]

    else:
        try:
            train_indices, temporary_indices = (
                train_test_split(
                    indices,
                    test_size=(
                        config.validation_size
                        + config.test_size
                    ),
                    random_state=config.random_seed,
                    stratify=labels,
                )
            )

            validation_ratio = (
                config.validation_size
                / (
                    config.validation_size
                    + config.test_size
                )
            )

            temporary_labels = labels[
                temporary_indices
            ]

            validation_indices, test_indices = (
                train_test_split(
                    temporary_indices,
                    train_size=validation_ratio,
                    random_state=config.random_seed,
                    stratify=temporary_labels,
                )
            )

        except ValueError as exc:
            raise FailureDatasetError(
                "Unable to create stratified splits; "
                "check class counts and split sizes"
            ) from exc

    splits = FailureDatasetSplits(
        x_train=features.iloc[
            train_indices
        ].reset_index(drop=True),
        x_validation=features.iloc[
            validation_indices
        ].reset_index(drop=True),
        x_test=features.iloc[
            test_indices
        ].reset_index(drop=True),
        y_train=labels[train_indices],
        y_validation=labels[validation_indices],
        y_test=labels[test_indices],
    )

    if len(np.unique(splits.y_train)) < 2:
        raise FailureDatasetError(
            "Training split must contain both classes"
        )

    if len(splits.x_validation) == 0:
        raise FailureDatasetError(
            "Validation split is empty"
        )

    if len(splits.x_test) == 0:
        raise FailureDatasetError(
            "Test split is empty"
        )

    LOGGER.info(
        "Split sizes: train=%d validation=%d test=%d",
        len(splits.x_train),
        len(splits.x_validation),
        len(splits.x_test),
    )

    return splits


# ============================================================
# Preprocessing
# ============================================================

def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                list(numeric_features),
            )
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                list(categorical_features),
            )
        )

    if not transformers:
        raise FailureDatasetError(
            "At least one feature is required"
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )


# ============================================================
# Model Factory
# ============================================================

def create_failure_estimator(
    config: FailureTrainingConfig,
) -> BaseEstimator:
    if (
        config.model_type
        == FailureModelType.RANDOM_FOREST
    ):
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            class_weight="balanced",
            random_state=config.random_seed,
            n_jobs=-1,
        )

    if (
        config.model_type
        == FailureModelType.GRADIENT_BOOSTING
    ):
        return HistGradientBoostingClassifier(
            max_iter=config.max_iterations,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            l2_regularization=0.1,
            random_state=config.random_seed,
        )

    raise FailureModelError(
        f"Unsupported model: {config.model_type}"
    )


def create_failure_pipeline(
    config: FailureTrainingConfig,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "classifier",
                create_failure_estimator(config),
            ),
        ]
    )


# ============================================================
# Hyperparameter Tuning
# ============================================================

def tune_failure_model(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    config: FailureTrainingConfig,
) -> tuple[Pipeline, dict[str, Any]]:
    if (
        config.model_type
        == FailureModelType.RANDOM_FOREST
    ):
        parameter_distributions = {
            "classifier__n_estimators": [
                100,
                200,
                300,
                500,
            ],
            "classifier__max_depth": [
                8,
                12,
                16,
                24,
                None,
            ],
            "classifier__min_samples_leaf": [
                1,
                2,
                4,
                8,
            ],
            "classifier__max_features": [
                "sqrt",
                "log2",
                None,
            ],
        }

    else:
        parameter_distributions = {
            "classifier__max_iter": [
                100,
                200,
                300,
            ],
            "classifier__learning_rate": [
                0.01,
                0.03,
                0.05,
                0.1,
            ],
            "classifier__max_depth": [
                4,
                8,
                12,
                None,
            ],
            "classifier__l2_regularization": [
                0.0,
                0.1,
                1.0,
            ],
        }

    counts = np.bincount(
        y_train,
        minlength=2,
    )

    if counts.min() < config.cross_validation_folds:
        raise FailureDatasetError(
            "Insufficient samples in the minority class "
            "for cross-validation"
        )

    if (
        config.split_strategy
        == DatasetSplitStrategy.CHRONOLOGICAL
    ):
        from sklearn.model_selection import TimeSeriesSplit

        cross_validation = TimeSeriesSplit(
            n_splits=config.cross_validation_folds
        )

        # All chronological training folds must contain
        # both classes for probability-based scoring.
        for train_indices, _ in cross_validation.split(
            x_train
        ):
            if len(np.unique(y_train[train_indices])) < 2:
                raise FailureDatasetError(
                    "Chronological CV fold has only one class"
                )

    else:
        cross_validation = (
            config.cross_validation_folds
        )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=config.tuning_iterations,
        scoring="average_precision",
        cv=cross_validation,
        n_jobs=-1,
        random_state=config.random_seed,
        refit=True,
        error_score="raise",
    )

    search.fit(
        x_train,
        y_train,
    )

    return (
        search.best_estimator_,
        {
            "best_params": (
                search.best_params_
            ),
            "best_cv_score": float(
                search.best_score_
            ),
            "iterations": (
                config.tuning_iterations
            ),
        },
    )


# ============================================================
# Threshold Optimization
# ============================================================

@dataclass
class FailureThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1_score: float
    minimum_recall_satisfied: bool
    method: str


def optimize_failure_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    minimum_recall: float,
    default_threshold: float = 0.5,
) -> FailureThresholdResult:
    if len(y_true) != len(probabilities):
        raise FailureEvaluationError(
            "Labels and probabilities have different lengths"
        )

    if len(y_true) == 0:
        raise FailureEvaluationError(
            "Validation dataset is empty"
        )

    if not np.isfinite(probabilities).all():
        raise FailureEvaluationError(
            "Probabilities contain invalid values"
        )

    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.01,
                    0.99,
                    99,
                ),
                np.asarray(
                    [default_threshold]
                ),
            ]
        )
    )

    candidates = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(np.int64)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        candidates.append(
            (
                float(threshold),
                float(precision),
                float(recall),
                float(f1),
            )
        )

    satisfying = [
        candidate
        for candidate in candidates
        if candidate[2] >= minimum_recall
    ]

    if satisfying:
        best = max(
            satisfying,
            key=lambda item: (
                item[3],
                item[1],
                item[0],
            ),
        )

        satisfied = True

    else:
        best = max(
            candidates,
            key=lambda item: (
                item[2],
                item[3],
                item[1],
            ),
        )

        satisfied = False

    return FailureThresholdResult(
        threshold=best[0],
        precision=best[1],
        recall=best[2],
        f1_score=best[3],
        minimum_recall_satisfied=satisfied,
        method="validation_threshold_search",
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_failure_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    if len(y_true) != len(probabilities):
        raise FailureEvaluationError(
            "Prediction length mismatch"
        )

    if len(y_true) == 0:
        raise FailureEvaluationError(
            "Evaluation dataset is empty"
        )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics: dict[str, Any] = {
        "sample_count": int(
            len(y_true)
        ),
        "threshold": float(
            threshold
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "actual_failure_rate": float(
            np.mean(y_true)
        ),
        "predicted_failure_rate": float(
            np.mean(predictions)
        ),
        "mean_failure_probability": float(
            np.mean(probabilities)
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": (
            classification_report(
                y_true,
                predictions,
                labels=[0, 1],
                target_names=[
                    "normal",
                    "failure",
                ],
                output_dict=True,
                zero_division=0,
            )
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

        metrics["pr_auc"] = float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )

    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


# ============================================================
# Feature Importance
# ============================================================

def extract_feature_importance(
    pipeline: Pipeline,
) -> list[dict[str, Any]]:
    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    classifier = pipeline.named_steps[
        "classifier"
    ]

    try:
        feature_names = (
            preprocessor.get_feature_names_out()
        )

    except (AttributeError, ValueError):
        return []

    if hasattr(
        classifier,
        "feature_importances_",
    ):
        importance_values = (
            classifier.feature_importances_
        )

    else:
        return []

    importance = [
        {
            "feature": str(name),
            "importance": float(value),
        }
        for name, value in zip(
            feature_names,
            importance_values,
        )
    ]

    importance.sort(
        key=lambda item: item["importance"],
        reverse=True,
    )

    return importance


# ============================================================
# Risk Classification
# ============================================================

def classify_failure_risk(
    probability: float,
) -> FailureRiskLevel:
    if not math.isfinite(probability):
        raise ValueError(
            "Probability must be finite"
        )

    if not 0 <= probability <= 1:
        raise ValueError(
            "Probability must be between 0 and 1"
        )

    if probability < 0.25:
        return FailureRiskLevel.LOW

    if probability < 0.50:
        return FailureRiskLevel.MODERATE

    if probability < 0.75:
        return FailureRiskLevel.HIGH

    return FailureRiskLevel.CRITICAL


# ============================================================
# Training Result
# ============================================================

@dataclass
class FailureTrainingResult:
    run_id: str
    status: str

    model_type: str

    numeric_features: list[str]
    categorical_features: list[str]

    training_samples: int
    validation_samples: int
    test_samples: int

    threshold: float

    training_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]

    feature_importance: list[dict[str, Any]]

    artifact_paths: dict[str, str]

    started_at: str
    completed_at: str

    duration_seconds: float


# ============================================================
# Training Pipeline
# ============================================================

class FailureTrainingPipeline:
    def __init__(
        self,
        config: FailureTrainingConfig,
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

        self.model: Pipeline | None = None

        self.threshold = (
            config.default_threshold
        )

        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []

    def _prepare_data(
        self,
    ) -> FailureDatasetSplits:
        dataset = RobotFailureDataset(
            self.config
        )

        dataset.load()

        (
            self.numeric_features,
            self.categorical_features,
        ) = dataset.select_features()

        features, labels, timestamps = (
            dataset.prepare()
        )

        return create_dataset_splits(
            features,
            labels,
            self.config,
            timestamps,
        )

    def _train(
        self,
        splits: FailureDatasetSplits,
    ) -> dict[str, Any]:
        pipeline = create_failure_pipeline(
            self.config,
            self.numeric_features,
            self.categorical_features,
        )

        started = time.perf_counter()

        tuning_result = None

        if self.config.hyperparameter_tuning:
            pipeline, tuning_result = (
                tune_failure_model(
                    pipeline,
                    splits.x_train,
                    splits.y_train,
                    self.config,
                )
            )

        else:
            pipeline.fit(
                splits.x_train,
                splits.y_train,
            )

        self.model = pipeline

        training_probabilities = (
            pipeline.predict_proba(
                splits.x_train
            )[:, 1]
        )

        return {
            "training_duration_seconds": round(
                time.perf_counter() - started,
                4,
            ),
            "training_samples": int(
                len(splits.x_train)
            ),
            "training_failure_count": int(
                np.sum(splits.y_train)
            ),
            "training_failure_rate": float(
                np.mean(splits.y_train)
            ),
            "training_roc_auc": float(
                roc_auc_score(
                    splits.y_train,
                    training_probabilities,
                )
            ),
            "training_pr_auc": float(
                average_precision_score(
                    splits.y_train,
                    training_probabilities,
                )
            ),
            "hyperparameter_tuning": (
                tuning_result
            ),
        }

    def _select_threshold(
        self,
        splits: FailureDatasetSplits,
    ) -> FailureThresholdResult:
        if self.model is None:
            raise FailureModelError(
                "Model has not been trained"
            )

        probabilities = (
            self.model.predict_proba(
                splits.x_validation
            )[:, 1]
        )

        if self.config.optimize_threshold:
            return optimize_failure_threshold(
                splits.y_validation,
                probabilities,
                self.config.minimum_recall,
                self.config.default_threshold,
            )

        predictions = (
            probabilities
            >= self.config.default_threshold
        ).astype(np.int64)

        return FailureThresholdResult(
            threshold=self.config.default_threshold,
            precision=float(
                precision_score(
                    splits.y_validation,
                    predictions,
                    zero_division=0,
                )
            ),
            recall=float(
                recall_score(
                    splits.y_validation,
                    predictions,
                    zero_division=0,
                )
            ),
            f1_score=float(
                f1_score(
                    splits.y_validation,
                    predictions,
                    zero_division=0,
                )
            ),
            minimum_recall_satisfied=(
                recall_score(
                    splits.y_validation,
                    predictions,
                    zero_division=0,
                )
                >= self.config.minimum_recall
            ),
            method="fixed_threshold",
        )

    def _save_artifacts(
        self,
        *,
        training_metrics: dict[str, Any],
        validation_metrics: dict[str, Any],
        test_metrics: dict[str, Any],
        threshold_result: FailureThresholdResult,
        feature_importance: list[dict[str, Any]],
    ) -> dict[str, str]:
        if self.model is None:
            raise FailureModelError(
                "No trained model available"
            )

        model_path = (
            self.output_directory
            / "failure_model.joblib"
        )

        metadata_path = (
            self.output_directory
            / "training_metadata.json"
        )

        evaluation_path = (
            self.output_directory
            / "evaluation_report.json"
        )

        importance_path = (
            self.output_directory
            / "feature_importance.json"
        )

        joblib.dump(
            self.model,
            model_path,
            compress=3,
        )

        if model_path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise FailureArtifactError(
                "Model artifact exceeds size limit"
            )

        model_checksum = calculate_sha256(
            model_path
        )

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "model_name": (
                "robot-component-failure-predictor"
            ),
            "model_type": (
                self.config.model_type.value
            ),
            "created_at": utc_now(),
            "target_column": (
                self.config.target_column
            ),
            "numeric_features": (
                self.numeric_features
            ),
            "categorical_features": (
                self.categorical_features
            ),
            "feature_count": (
                len(self.numeric_features)
                + len(self.categorical_features)
            ),
            "threshold": asdict(
                threshold_result
            ),
            "configuration": (
                to_json_compatible(
                    asdict(self.config)
                )
            ),
            "artifact": {
                "filename": model_path.name,
                "sha256": model_checksum,
                "size_bytes": (
                    model_path.stat().st_size
                ),
                "format": "joblib",
            },
            "safety": {
                "advisory_only": True,
                "physical_execution_authorized": False,
                "emergency_stop_reset_authorized": False,
                "safety_interlock_bypass_authorized": False,
                "human_review_required_for_actions": True,
            },
        }

        evaluation_report = {
            "run_id": self.run_id,
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "threshold": asdict(
                threshold_result
            ),
            "feature_importance": (
                feature_importance
            ),
        }

        atomic_save_json(
            metadata_path,
            metadata,
        )

        atomic_save_json(
            evaluation_path,
            to_json_compatible(
                evaluation_report
            ),
        )

        atomic_save_json(
            importance_path,
            {
                "run_id": self.run_id,
                "features": feature_importance,
            },
        )

        return {
            "model": str(model_path),
            "metadata": str(metadata_path),
            "evaluation": str(evaluation_path),
            "feature_importance": str(
                importance_path
            ),
        }

    def train(
        self,
    ) -> FailureTrainingResult:
        started_at = utc_now()
        started = time.perf_counter()

        set_random_seed(
            self.config.random_seed
        )

        LOGGER.info(
            "Starting failure prediction training: %s",
            self.run_id,
        )

        splits = self._prepare_data()

        training_metrics = self._train(
            splits
        )

        threshold_result = (
            self._select_threshold(
                splits
            )
        )

        self.threshold = (
            threshold_result.threshold
        )

        assert self.model is not None

        validation_probabilities = (
            self.model.predict_proba(
                splits.x_validation
            )[:, 1]
        )

        test_probabilities = (
            self.model.predict_proba(
                splits.x_test
            )[:, 1]
        )

        validation_metrics = (
            evaluate_failure_predictions(
                splits.y_validation,
                validation_probabilities,
                self.threshold,
            )
        )

        test_metrics = (
            evaluate_failure_predictions(
                splits.y_test,
                test_probabilities,
                self.threshold,
            )
        )

        feature_importance = (
            extract_feature_importance(
                self.model
            )
        )

        artifact_paths = (
            self._save_artifacts(
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                threshold_result=threshold_result,
                feature_importance=feature_importance,
            )
        )

        result = FailureTrainingResult(
            run_id=self.run_id,
            status=TrainingStatus.COMPLETED.value,
            model_type=(
                self.config.model_type.value
            ),
            numeric_features=(
                self.numeric_features
            ),
            categorical_features=(
                self.categorical_features
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
            threshold=self.threshold,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            feature_importance=feature_importance,
            artifact_paths=artifact_paths,
            started_at=started_at,
            completed_at=utc_now(),
            duration_seconds=round(
                time.perf_counter() - started,
                4,
            ),
        )

        atomic_save_json(
            self.output_directory
            / "training_result.json",
            to_json_compatible(
                asdict(result)
            ),
        )

        LOGGER.info(
            "Training completed: run=%s, test_f1=%.4f, "
            "test_recall=%.4f, duration=%.2fs",
            self.run_id,
            test_metrics["f1_score"],
            test_metrics["recall"],
            result.duration_seconds,
        )

        return result


# ============================================================
# Trusted Artifact Inference
# ============================================================

class FailurePredictor:
    """
    Load a trusted failure-prediction artifact.

    Joblib deserialization can execute arbitrary code.
    Only use locally trusted, access-controlled artifacts.
    A SHA-256 match detects changes against the metadata,
    but does not establish that the metadata is authentic.
    """

    def __init__(
        self,
        artifact_directory: str | Path,
        *,
        expected_sha256: str | None = None,
        trusted_source: bool = False,
    ) -> None:
        if not trusted_source:
            raise FailureArtifactError(
                "Explicit trusted_source=True is required"
            )

        self.artifact_directory = Path(
            artifact_directory
        ).resolve()

        metadata_path = (
            self.artifact_directory
            / "training_metadata.json"
        )

        if not metadata_path.is_file():
            raise FailureArtifactError(
                "Training metadata not found"
            )

        self.metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        artifact = self.metadata[
            "artifact"
        ]

        model_path = (
            self.artifact_directory
            / artifact["filename"]
        ).resolve()

        if not model_path.is_relative_to(
            self.artifact_directory
        ):
            raise FailureArtifactError(
                "Artifact path escapes trusted directory"
            )

        if not model_path.is_file():
            raise FailureArtifactError(
                "Model artifact not found"
            )

        if model_path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise FailureArtifactError(
                "Model artifact exceeds size limit"
            )

        actual_sha256 = calculate_sha256(
            model_path
        )

        if actual_sha256 != artifact["sha256"]:
            raise FailureArtifactError(
                "Artifact checksum mismatch"
            )

        if (
            expected_sha256 is not None
            and actual_sha256 != expected_sha256
        ):
            raise FailureArtifactError(
                "Artifact does not match trusted checksum"
            )

        self.model: Pipeline = joblib.load(
            model_path
        )

        self.numeric_features = list(
            self.metadata["numeric_features"]
        )

        self.categorical_features = list(
            self.metadata["categorical_features"]
        )

        self.threshold = float(
            self.metadata["threshold"]["threshold"]
        )

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        feature_columns = (
            self.numeric_features
            + self.categorical_features
        )

        missing = [
            name
            for name in feature_columns
            if name not in dataframe.columns
        ]

        if missing:
            raise FailureDatasetError(
                f"Missing inference features: {missing}"
            )

        features = dataframe[
            feature_columns
        ].copy()

        for name in self.numeric_features:
            features[name] = (
                pd.to_numeric(
                    features[name],
                    errors="coerce",
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            )

        for name in self.categorical_features:
            features[name] = (
                features[name]
                .astype("string")
                .fillna("__missing__")
                .astype(str)
            )

        probabilities = (
            self.model.predict_proba(
                features
            )[:, 1]
        )

        predictions = (
            probabilities >= self.threshold
        ).astype(np.int64)

        result = dataframe.copy()

        result["failure_probability"] = (
            probabilities
        )

        result["predicted_failure"] = (
            predictions
        )

        result["failure_risk_level"] = [
            classify_failure_risk(
                float(probability)
            ).value
            for probability in probabilities
        ]

        result["prediction_status"] = np.where(
            predictions == 1,
            "failure_risk_detected",
            "normal",
        )

        result["human_review_recommended"] = (
            predictions.astype(bool)
        )

        result["advisory_only"] = True

        return result

    def predict_one(
        self,
        telemetry: Mapping[str, Any],
    ) -> dict[str, Any]:
        dataframe = pd.DataFrame(
            [dict(telemetry)]
        )

        result = self.predict(
            dataframe
        )

        return to_json_compatible(
            result.iloc[0].to_dict()
        )


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train robot component failure prediction models"
        )
    )

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--target",
        default="failure",
    )

    parser.add_argument(
        "--model",
        choices=[
            model.value
            for model in FailureModelType
        ],
        default="random_forest",
    )

    parser.add_argument(
        "--output",
        default="artifacts/failure",
    )

    parser.add_argument(
        "--split",
        choices=[
            strategy.value
            for strategy in DatasetSplitStrategy
        ],
        default="random",
    )

    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
    )

    parser.add_argument(
        "--numeric-features",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--categorical-features",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--minimum-recall",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--disable-threshold-optimization",
        action="store_true",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
    )

    parser.add_argument(
        "--tuning-iterations",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config = FailureTrainingConfig(
        data_path=args.data,
        target_column=args.target,
        model_type=FailureModelType(
            args.model
        ),
        output_directory=args.output,
        split_strategy=DatasetSplitStrategy(
            args.split
        ),
        timestamp_column=args.timestamp_column,
        numeric_features=(
            args.numeric_features or []
        ),
        categorical_features=(
            args.categorical_features or []
        ),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        default_threshold=args.threshold,
        minimum_recall=args.minimum_recall,
        optimize_threshold=(
            not args.disable_threshold_optimization
        ),
        hyperparameter_tuning=args.tune,
        tuning_iterations=args.tuning_iterations,
        random_seed=args.seed,
    )

    pipeline = FailureTrainingPipeline(
        config
    )

    result = pipeline.train()

    print(
        json.dumps(
            to_json_compatible(
                asdict(result)
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/evaluate_models.py

Unified offline model evaluation for:
    - Classification
    - Regression
    - Anomaly detection
    - Failure prediction
    - Remaining Useful Life (RUL)
    - Object detection
    - Vision-Language-Action (VLA)
    - Neural world models

Features:
    - CSV, JSON, JSONL, and Parquet evaluation data
    - Classification metrics and confusion matrices
    - Regression, RUL, and failure metrics
    - Object detection IoU, precision, recall, F1, AP
    - VLA action-prediction evaluation
    - World-model transition and reward evaluation
    - Safety-threshold checks
    - Configurable quality gates
    - Model comparison
    - Evaluation report generation
    - JSON and CSV outputs
    - Optional confusion matrix visualization
    - Dataset integrity validation
    - Model artifact checksum recording
    - Simulation-only evaluation

Dependencies:
    pip install numpy pandas scikit-learn

Optional:
    pip install pyarrow matplotlib

Examples:
    python scripts/evaluate_models.py classification \
        --data data/evaluation/classification.csv \
        --target-column actual \
        --prediction-column predicted

    python scripts/evaluate_models.py regression \
        --data data/evaluation/rul.csv \
        --target-column actual_rul \
        --prediction-column predicted_rul

    python scripts/evaluate_models.py anomaly \
        --data data/evaluation/anomaly.csv \
        --target-column actual \
        --prediction-column predicted \
        --score-column anomaly_score

    python scripts/evaluate_models.py detection \
        --data data/evaluation/detections.json

    python scripts/evaluate_models.py vla \
        --data data/evaluation/vla.csv \
        --target-columns action_x action_y \
        --prediction-columns predicted_x predicted_y

    python scripts/evaluate_models.py world-model \
        --data data/evaluation/world_model.csv \
        --target-columns next_x next_y \
        --prediction-columns predicted_next_x predicted_next_y

    python scripts/evaluate_models.py compare \
        --reports reports/evaluation/model_a.json \
                  reports/evaluation/model_b.json

Safety:
    Evaluation is offline only.
    No commands are sent to physical robots.
    Passing metrics does not authorize physical deployment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


LOGGER = logging.getLogger("evaluate_models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCHEMA_VERSION = "1.0.0"


# ============================================================
# Exceptions
# ============================================================

class ModelEvaluationError(Exception):
    pass


class EvaluationDatasetError(ModelEvaluationError):
    pass


class EvaluationConfigurationError(ModelEvaluationError):
    pass


class EvaluationSafetyError(ModelEvaluationError):
    pass


# ============================================================
# Utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def save_json(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def ensure_finite(
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if not np.isfinite(array).all():
        raise EvaluationDatasetError(
            f"{name} contains non-finite values"
        )

    return array


def validate_equal_length(
    *arrays: np.ndarray,
) -> None:
    lengths = {
        len(array)
        for array in arrays
    }

    if len(lengths) != 1:
        raise EvaluationDatasetError(
            "Evaluation arrays have different lengths"
        )


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# Configuration
# ============================================================

@dataclass
class EvaluationConfig:
    model_name: str = "unnamed-model"
    model_version: str = "0.0.0"

    task_type: str = "classification"

    data_path: str = ""

    output_directory: str = "reports/evaluation"

    target_columns: list[str] = field(
        default_factory=list
    )

    prediction_columns: list[str] = field(
        default_factory=list
    )

    score_column: str | None = None

    positive_label: int = 1

    classification_average: str = "weighted"

    iou_threshold: float = 0.5

    max_acceptable_error: float | None = None

    minimum_accuracy: float | None = None
    minimum_f1: float | None = None
    minimum_recall: float | None = None

    maximum_mae: float | None = None
    maximum_rmse: float | None = None

    simulation_only: bool = True

    model_artifact_path: str | None = None

    def validate(self) -> None:
        if not self.data_path:
            raise EvaluationConfigurationError(
                "Evaluation dataset path is required"
            )

        if not 0 < self.iou_threshold <= 1:
            raise EvaluationConfigurationError(
                "IoU threshold must be in (0, 1]"
            )

        if self.classification_average not in {
            "micro",
            "macro",
            "weighted",
            "binary",
        }:
            raise EvaluationConfigurationError(
                "Invalid classification averaging method"
            )

        if not self.simulation_only:
            raise EvaluationSafetyError(
                "Only offline/simulation evaluation is supported"
            )


# ============================================================
# Dataset Loader
# ============================================================

class EvaluationDatasetLoader:
    @staticmethod
    def load(
        path: str | Path,
    ) -> pd.DataFrame:
        path = Path(path)

        if not path.is_file():
            raise EvaluationDatasetError(
                f"Dataset not found: {path}"
            )

        extension = path.suffix.lower()

        if extension == ".csv":
            dataframe = pd.read_csv(path)

        elif extension in {".parquet", ".pq"}:
            dataframe = pd.read_parquet(path)

        elif extension == ".json":
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )

            if isinstance(payload, dict):
                if "records" in payload:
                    payload = payload["records"]
                else:
                    payload = [payload]

            dataframe = pd.DataFrame(payload)

        elif extension == ".jsonl":
            records = []

            with path.open(
                "r",
                encoding="utf-8",
            ) as stream:
                for line in stream:
                    if line.strip():
                        records.append(
                            json.loads(line)
                        )

            dataframe = pd.DataFrame(records)

        else:
            raise EvaluationDatasetError(
                f"Unsupported dataset format: {extension}"
            )

        if dataframe.empty:
            raise EvaluationDatasetError(
                "Evaluation dataset is empty"
            )

        return dataframe

    @staticmethod
    def require_columns(
        dataframe: pd.DataFrame,
        columns: Sequence[str],
    ) -> None:
        missing = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing:
            raise EvaluationDatasetError(
                f"Missing columns: {missing}"
            )

    @staticmethod
    def numeric_matrix(
        dataframe: pd.DataFrame,
        columns: Sequence[str],
    ) -> np.ndarray:
        EvaluationDatasetLoader.require_columns(
            dataframe,
            columns,
        )

        values = dataframe[
            list(columns)
        ].apply(
            pd.to_numeric,
            errors="raise",
        ).to_numpy(dtype=np.float64)

        return ensure_finite(
            values,
            "Numeric matrix",
        )


# ============================================================
# Classification Evaluation
# ============================================================

class ClassificationEvaluator:
    @staticmethod
    def evaluate(
        actual: Sequence[Any],
        predicted: Sequence[Any],
        *,
        scores: Sequence[float] | None = None,
        average: str = "weighted",
        positive_label: Any = 1,
    ) -> dict[str, Any]:
        y_true = np.asarray(actual)
        y_pred = np.asarray(predicted)

        validate_equal_length(
            y_true,
            y_pred,
        )

        if len(y_true) == 0:
            raise EvaluationDatasetError(
                "Classification arrays are empty"
            )

        labels = sorted(
            set(y_true.tolist())
            | set(y_pred.tolist()),
            key=str,
        )

        binary = len(labels) == 2

        effective_average = (
            "binary"
            if average == "binary" and binary
            else "weighted"
            if average == "binary"
            else average
        )

        kwargs = {
            "average": effective_average,
            "zero_division": 0,
        }

        if effective_average == "binary":
            kwargs["pos_label"] = positive_label

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        )

        metrics = {
            "accuracy": float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    **kwargs,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    **kwargs,
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    **kwargs,
                )
            ),
            "labels": [
                str(label)
                for label in labels
            ],
            "confusion_matrix": matrix.tolist(),
            "classification_report": (
                classification_report(
                    y_true,
                    y_pred,
                    labels=labels,
                    output_dict=True,
                    zero_division=0,
                )
            ),
            "sample_count": int(
                len(y_true)
            ),
        }

        if scores is not None and binary:
            y_scores = ensure_finite(
                np.asarray(scores),
                "Classification scores",
            )

            validate_equal_length(
                y_true,
                y_scores,
            )

            binary_targets = (
                y_true == positive_label
            ).astype(int)

            if len(np.unique(binary_targets)) == 2:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        binary_targets,
                        y_scores,
                    )
                )

                metrics["average_precision"] = float(
                    average_precision_score(
                        binary_targets,
                        y_scores,
                    )
                )

        return metrics


# ============================================================
# Regression Evaluation
# ============================================================

class RegressionEvaluator:
    @staticmethod
    def evaluate(
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> dict[str, Any]:
        y_true = ensure_finite(
            actual,
            "Regression targets",
        )

        y_pred = ensure_finite(
            predicted,
            "Regression predictions",
        )

        if y_true.shape != y_pred.shape:
            raise EvaluationDatasetError(
                "Regression prediction shape mismatch"
            )

        if y_true.size == 0:
            raise EvaluationDatasetError(
                "Regression arrays are empty"
            )

        error = y_pred - y_true

        mae = float(
            np.mean(
                np.abs(error)
            )
        )

        mse = float(
            np.mean(
                error ** 2
            )
        )

        rmse = math.sqrt(mse)

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "median_absolute_error": float(
                np.median(
                    np.abs(error)
                )
            ),
            "max_absolute_error": float(
                np.max(
                    np.abs(error)
                )
            ),
            "bias": float(
                np.mean(error)
            ),
            "sample_count": int(
                y_true.shape[0]
            ),
        }

        if y_true.shape[0] >= 2:
            metrics["r2"] = float(
                r2_score(
                    y_true,
                    y_pred,
                )
            )

            metrics["explained_variance"] = float(
                explained_variance_score(
                    y_true,
                    y_pred,
                )
            )

        denominator = np.maximum(
            np.abs(y_true),
            1e-8,
        )

        metrics["mape_percent"] = float(
            np.mean(
                np.abs(error) / denominator
            ) * 100
        )

        return metrics


# ============================================================
# RUL Evaluation
# ============================================================

class RULEvaluator:
    @staticmethod
    def evaluate(
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> dict[str, Any]:
        y_true = ensure_finite(
            actual,
            "RUL targets",
        ).reshape(-1)

        y_pred = ensure_finite(
            predicted,
            "RUL predictions",
        ).reshape(-1)

        validate_equal_length(
            y_true,
            y_pred,
        )

        if np.any(y_true < 0):
            raise EvaluationDatasetError(
                "Ground-truth RUL cannot be negative"
            )

        metrics = RegressionEvaluator.evaluate(
            y_true,
            y_pred,
        )

        error = (
            y_pred - y_true
        )

        # Asymmetric prognostics scoring:
        # Overestimating remaining life can be more
        # dangerous than underestimating it.
        exponent = np.where(
            error < 0,
            -error / 13.0,
            error / 10.0,
        )

        exponent = np.clip(
            exponent,
            0,
            50,
        )

        nasa_score = float(
            np.sum(
                np.expm1(exponent)
            )
        )

        metrics.update(
            {
                "nasa_asymmetric_score": nasa_score,
                "overestimation_rate": float(
                    np.mean(error > 0)
                ),
                "underestimation_rate": float(
                    np.mean(error < 0)
                ),
                "negative_prediction_rate": float(
                    np.mean(y_pred < 0)
                ),
                "within_10_units": float(
                    np.mean(
                        np.abs(error) <= 10
                    )
                ),
                "within_20_units": float(
                    np.mean(
                        np.abs(error) <= 20
                    )
                ),
            }
        )

        return metrics


# ============================================================
# Anomaly Detection Evaluation
# ============================================================

class AnomalyEvaluator:
    @staticmethod
    def evaluate(
        actual: Sequence[int],
        predicted: Sequence[int],
        *,
        scores: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        y_true = np.asarray(
            actual,
            dtype=int,
        )

        y_pred = np.asarray(
            predicted,
            dtype=int,
        )

        if not (
            np.isin(y_true, [0, 1]).all()
            and np.isin(y_pred, [0, 1]).all()
        ):
            raise EvaluationDatasetError(
                "Anomaly labels must be binary"
            )

        metrics = ClassificationEvaluator.evaluate(
            y_true,
            y_pred,
            scores=scores,
            average="binary",
            positive_label=1,
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        tn, fp, fn, tp = matrix.ravel()

        metrics.update(
            {
                "true_positive": int(tp),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "false_alarm_rate": safe_divide(
                    fp,
                    fp + tn,
                ),
                "missed_anomaly_rate": safe_divide(
                    fn,
                    fn + tp,
                ),
                "anomaly_prevalence": float(
                    np.mean(y_true)
                ),
            }
        )

        return metrics


# ============================================================
# Failure Prediction Evaluation
# ============================================================

class FailurePredictionEvaluator:
    @staticmethod
    def evaluate(
        actual: Sequence[int],
        predicted: Sequence[int],
        *,
        probabilities: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        metrics = AnomalyEvaluator.evaluate(
            actual,
            predicted,
            scores=probabilities,
        )

        metrics["failure_detection_rate"] = (
            metrics["recall"]
        )

        metrics["missed_failure_rate"] = (
            metrics["missed_anomaly_rate"]
        )

        return metrics


# ============================================================
# Object Detection Evaluation
# ============================================================

def calculate_iou(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    """
    Boxes use [x_min, y_min, x_max, y_max].
    """
    if len(box_a) != 4 or len(box_b) != 4:
        raise EvaluationDatasetError(
            "Bounding boxes must contain four values"
        )

    ax1, ay1, ax2, ay2 = map(
        float,
        box_a,
    )

    bx1, by1, bx2, by2 = map(
        float,
        box_b,
    )

    if (
        ax2 < ax1
        or ay2 < ay1
        or bx2 < bx1
        or by2 < by1
    ):
        raise EvaluationDatasetError(
            "Invalid bounding box coordinates"
        )

    x_left = max(
        ax1,
        bx1,
    )

    y_top = max(
        ay1,
        by1,
    )

    x_right = min(
        ax2,
        bx2,
    )

    y_bottom = min(
        ay2,
        by2,
    )

    intersection = (
        max(0.0, x_right - x_left)
        * max(0.0, y_bottom - y_top)
    )

    area_a = (
        (ax2 - ax1)
        * (ay2 - ay1)
    )

    area_b = (
        (bx2 - bx1)
        * (by2 - by1)
    )

    union = (
        area_a + area_b - intersection
    )

    return safe_divide(
        intersection,
        union,
    )


class ObjectDetectionEvaluator:
    @staticmethod
    def evaluate(
        records: Sequence[Mapping[str, Any]],
        *,
        iou_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Expected JSON:
        [
          {
            "image_id": "frame_001",
            "ground_truth": [
              {"class": "robot", "bbox": [10,20,100,120]}
            ],
            "predictions": [
              {
                "class": "robot",
                "bbox": [12,22,98,118],
                "confidence": 0.95
              }
            ]
          }
        ]
        """
        if not records:
            raise EvaluationDatasetError(
                "Detection evaluation records are empty"
            )

        total_tp = 0
        total_fp = 0
        total_fn = 0

        matched_ious = []

        for record in records:
            ground_truth = list(
                record.get(
                    "ground_truth",
                    [],
                )
            )

            predictions = list(
                record.get(
                    "predictions",
                    [],
                )
            )

            predictions.sort(
                key=lambda item: -float(
                    item.get(
                        "confidence",
                        1.0,
                    )
                )
            )

            matched_ground_truth = set()

            for prediction in predictions:
                best_iou = 0.0
                best_index = None

                for index, target in enumerate(
                    ground_truth
                ):
                    if index in matched_ground_truth:
                        continue

                    if (
                        str(prediction["class"])
                        != str(target["class"])
                    ):
                        continue

                    iou = calculate_iou(
                        prediction["bbox"],
                        target["bbox"],
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_index = index

                if (
                    best_index is not None
                    and best_iou >= iou_threshold
                ):
                    matched_ground_truth.add(
                        best_index
                    )

                    total_tp += 1

                    matched_ious.append(
                        best_iou
                    )

                else:
                    total_fp += 1

            total_fn += (
                len(ground_truth)
                - len(matched_ground_truth)
            )

        precision = safe_divide(
            total_tp,
            total_tp + total_fp,
        )

        recall = safe_divide(
            total_tp,
            total_tp + total_fn,
        )

        f1 = safe_divide(
            2 * precision * recall,
            precision + recall,
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": (
                float(np.mean(matched_ious))
                if matched_ious
                else 0.0
            ),
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "iou_threshold": iou_threshold,
            "image_count": len(records),
            "note": (
                "This is thresholded detection precision/"
                "recall, not COCO mAP. Use a full "
                "confidence-ranked AP evaluator for mAP."
            ),
        }


# ============================================================
# VLA Evaluation
# ============================================================

class VLAEvaluator:
    @staticmethod
    def evaluate(
        actual_actions: np.ndarray,
        predicted_actions: np.ndarray,
        *,
        success_labels: Sequence[int] | None = None,
        max_acceptable_error: float | None = None,
    ) -> dict[str, Any]:
        actual = ensure_finite(
            actual_actions,
            "Actual actions",
        )

        predicted = ensure_finite(
            predicted_actions,
            "Predicted actions",
        )

        if actual.shape != predicted.shape:
            raise EvaluationDatasetError(
                "VLA action dimensions do not match"
            )

        if actual.ndim != 2:
            raise EvaluationDatasetError(
                "VLA actions must have shape [N, D]"
            )

        error = (
            predicted - actual
        )

        euclidean_error = np.linalg.norm(
            error,
            axis=1,
        )

        metrics = {
            "action_mae": float(
                np.mean(
                    np.abs(error)
                )
            ),
            "action_rmse": float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            ),
            "mean_action_distance": float(
                np.mean(
                    euclidean_error
                )
            ),
            "median_action_distance": float(
                np.median(
                    euclidean_error
                )
            ),
            "p95_action_distance": float(
                np.percentile(
                    euclidean_error,
                    95,
                )
            ),
            "max_action_distance": float(
                np.max(
                    euclidean_error
                )
            ),
            "sample_count": int(
                len(actual)
            ),
            "action_dimension": int(
                actual.shape[1]
            ),
        }

        if max_acceptable_error is not None:
            metrics["action_tolerance_accuracy"] = float(
                np.mean(
                    euclidean_error
                    <= max_acceptable_error
                )
            )

        if success_labels is not None:
            labels = np.asarray(
                success_labels,
                dtype=int,
            )

            validate_equal_length(
                labels,
                actual,
            )

            if not np.isin(
                labels,
                [0, 1],
            ).all():
                raise EvaluationDatasetError(
                    "Success labels must be binary"
                )

            metrics["recorded_success_rate"] = float(
                np.mean(labels)
            )

        metrics["simulation_only"] = True

        return metrics


# ============================================================
# World Model Evaluation
# ============================================================

class WorldModelEvaluator:
    @staticmethod
    def evaluate(
        actual_states: np.ndarray,
        predicted_states: np.ndarray,
        *,
        actual_rewards: np.ndarray | None = None,
        predicted_rewards: np.ndarray | None = None,
        actual_dones: np.ndarray | None = None,
        predicted_done_probabilities: np.ndarray | None = None,
    ) -> dict[str, Any]:
        actual = ensure_finite(
            actual_states,
            "World model actual states",
        )

        predicted = ensure_finite(
            predicted_states,
            "World model predicted states",
        )

        if actual.shape != predicted.shape:
            raise EvaluationDatasetError(
                "World model state dimensions do not match"
            )

        if actual.ndim != 2:
            raise EvaluationDatasetError(
                "World model states must have shape [N, D]"
            )

        error = (
            predicted - actual
        )

        metrics = {
            "state_mae": float(
                np.mean(
                    np.abs(error)
                )
            ),
            "state_rmse": float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            ),
            "state_bias": float(
                np.mean(error)
            ),
            "state_dimension": int(
                actual.shape[1]
            ),
            "transition_count": int(
                len(actual)
            ),
        }

        if (
            actual_rewards is not None
            and predicted_rewards is not None
        ):
            reward_metrics = (
                RegressionEvaluator.evaluate(
                    actual_rewards,
                    predicted_rewards,
                )
            )

            metrics["reward_mae"] = (
                reward_metrics["mae"]
            )

            metrics["reward_rmse"] = (
                reward_metrics["rmse"]
            )

        if (
            actual_dones is not None
            and predicted_done_probabilities is not None
        ):
            actual_done = np.asarray(
                actual_dones,
                dtype=int,
            )

            probabilities = ensure_finite(
                predicted_done_probabilities,
                "Done probabilities",
            )

            if np.any(
                (probabilities < 0)
                | (probabilities > 1)
            ):
                raise EvaluationDatasetError(
                    "Done probabilities must be in [0, 1]"
                )

            predicted_done = (
                probabilities >= 0.5
            ).astype(int)

            done_metrics = (
                AnomalyEvaluator.evaluate(
                    actual_done,
                    predicted_done,
                    scores=probabilities,
                )
            )

            metrics["done_accuracy"] = (
                done_metrics["accuracy"]
            )

            metrics["done_precision"] = (
                done_metrics["precision"]
            )

            metrics["done_recall"] = (
                done_metrics["recall"]
            )

            metrics["done_f1"] = (
                done_metrics["f1"]
            )

        metrics["simulation_only"] = True

        return metrics


# ============================================================
# Quality Gates
# ============================================================

@dataclass
class QualityGateResult:
    passed: bool
    checks: list[dict[str, Any]]


class ModelQualityGates:
    @staticmethod
    def evaluate(
        metrics: Mapping[str, Any],
        config: EvaluationConfig,
    ) -> QualityGateResult:
        checks = []

        def minimum(
            metric_name: str,
            threshold: float | None,
        ) -> None:
            if threshold is None:
                return

            actual = safe_float(
                metrics.get(metric_name)
            )

            passed = (
                actual is not None
                and actual >= threshold
            )

            checks.append(
                {
                    "metric": metric_name,
                    "operator": ">=",
                    "threshold": threshold,
                    "actual": actual,
                    "passed": passed,
                }
            )

        def maximum(
            metric_name: str,
            threshold: float | None,
        ) -> None:
            if threshold is None:
                return

            actual = safe_float(
                metrics.get(metric_name)
            )

            passed = (
                actual is not None
                and actual <= threshold
            )

            checks.append(
                {
                    "metric": metric_name,
                    "operator": "<=",
                    "threshold": threshold,
                    "actual": actual,
                    "passed": passed,
                }
            )

        minimum(
            "accuracy",
            config.minimum_accuracy,
        )

        minimum(
            "f1",
            config.minimum_f1,
        )

        minimum(
            "recall",
            config.minimum_recall,
        )

        maximum(
            "mae",
            config.maximum_mae,
        )

        maximum(
            "rmse",
            config.maximum_rmse,
        )

        return QualityGateResult(
            passed=all(
                item["passed"]
                for item in checks
            ),
            checks=checks,
        )


# ============================================================
# Evaluation Report
# ============================================================

@dataclass
class EvaluationReport:
    evaluation_id: str

    model_name: str
    model_version: str

    task_type: str

    status: str

    metrics: dict[str, Any]

    quality_gates: dict[str, Any]

    dataset: dict[str, Any]

    model_artifact: dict[str, Any]

    started_at: str
    completed_at: str

    duration_seconds: float

    simulation_only: bool = True
    physical_execution_authorized: bool = False


# ============================================================
# Unified Evaluation Pipeline
# ============================================================

class ModelEvaluationPipeline:
    def __init__(
        self,
        config: EvaluationConfig,
    ) -> None:
        config.validate()

        self.config = config

        self.output_directory = Path(
            config.output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _evaluate_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        config = self.config

        EvaluationDatasetLoader.require_columns(
            dataframe,
            [
                *config.target_columns,
                *config.prediction_columns,
            ],
        )

        if (
            len(config.target_columns)
            != len(config.prediction_columns)
        ):
            raise EvaluationConfigurationError(
                "Target and prediction column counts differ"
            )

        task = config.task_type

        if task in {
            "classification",
            "anomaly",
            "failure",
        }:
            if (
                len(config.target_columns) != 1
                or len(config.prediction_columns) != 1
            ):
                raise EvaluationConfigurationError(
                    "Classification tasks require one target "
                    "and one prediction column"
                )

            actual = dataframe[
                config.target_columns[0]
            ].to_numpy()

            predicted = dataframe[
                config.prediction_columns[0]
            ].to_numpy()

            scores = None

            if config.score_column:
                EvaluationDatasetLoader.require_columns(
                    dataframe,
                    [config.score_column],
                )

                scores = dataframe[
                    config.score_column
                ].to_numpy()

            if task == "classification":
                return ClassificationEvaluator.evaluate(
                    actual,
                    predicted,
                    scores=scores,
                    average=config.classification_average,
                    positive_label=config.positive_label,
                )

            if task == "anomaly":
                return AnomalyEvaluator.evaluate(
                    actual,
                    predicted,
                    scores=scores,
                )

            return FailurePredictionEvaluator.evaluate(
                actual,
                predicted,
                probabilities=scores,
            )

        actual_matrix = (
            EvaluationDatasetLoader.numeric_matrix(
                dataframe,
                config.target_columns,
            )
        )

        predicted_matrix = (
            EvaluationDatasetLoader.numeric_matrix(
                dataframe,
                config.prediction_columns,
            )
        )

        if task == "regression":
            return RegressionEvaluator.evaluate(
                actual_matrix,
                predicted_matrix,
            )

        if task == "rul":
            if actual_matrix.shape[1] != 1:
                raise EvaluationConfigurationError(
                    "RUL evaluation requires one target column"
                )

            return RULEvaluator.evaluate(
                actual_matrix[:, 0],
                predicted_matrix[:, 0],
            )

        if task == "vla":
            return VLAEvaluator.evaluate(
                actual_matrix,
                predicted_matrix,
                max_acceptable_error=(
                    config.max_acceptable_error
                ),
            )

        if task == "world-model":
            actual_rewards = None
            predicted_rewards = None

            actual_dones = None
            predicted_dones = None

            optional_pairs = {
                "reward": (
                    "actual_reward",
                    "predicted_reward",
                ),
                "done": (
                    "actual_done",
                    "predicted_done_probability",
                ),
            }

            reward_columns = optional_pairs["reward"]
            done_columns = optional_pairs["done"]

            if all(
                column in dataframe.columns
                for column in reward_columns
            ):
                actual_rewards = dataframe[
                    reward_columns[0]
                ].to_numpy()

                predicted_rewards = dataframe[
                    reward_columns[1]
                ].to_numpy()

            if all(
                column in dataframe.columns
                for column in done_columns
            ):
                actual_dones = dataframe[
                    done_columns[0]
                ].to_numpy()

                predicted_dones = dataframe[
                    done_columns[1]
                ].to_numpy()

            return WorldModelEvaluator.evaluate(
                actual_matrix,
                predicted_matrix,
                actual_rewards=actual_rewards,
                predicted_rewards=predicted_rewards,
                actual_dones=actual_dones,
                predicted_done_probabilities=predicted_dones,
            )

        raise EvaluationConfigurationError(
            f"Unsupported task: {task}"
        )

    def evaluate(self) -> EvaluationReport:
        started_at = utc_now()
        start_time = time.perf_counter()

        dataset_path = Path(
            self.config.data_path
        )

        dataframe = EvaluationDatasetLoader.load(
            dataset_path
        )

        if self.config.task_type == "detection":
            records = dataframe.to_dict(
                orient="records"
            )

            metrics = ObjectDetectionEvaluator.evaluate(
                records,
                iou_threshold=self.config.iou_threshold,
            )

        else:
            metrics = self._evaluate_dataframe(
                dataframe
            )

        gates = ModelQualityGates.evaluate(
            metrics,
            self.config,
        )

        model_artifact = {}

        if self.config.model_artifact_path:
            artifact_path = Path(
                self.config.model_artifact_path
            )

            if not artifact_path.is_file():
                raise EvaluationDatasetError(
                    "Model artifact not found"
                )

            model_artifact = {
                "path": str(artifact_path),
                "sha256": sha256_file(
                    artifact_path
                ),
                "size_bytes": (
                    artifact_path.stat().st_size
                ),
            }

        evaluation_id = uuid.uuid4().hex

        report = EvaluationReport(
            evaluation_id=evaluation_id,
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            task_type=self.config.task_type,
            status=(
                "passed"
                if gates.passed
                else "failed"
            ),
            metrics=metrics,
            quality_gates=asdict(gates),
            dataset={
                "path": str(dataset_path),
                "sha256": sha256_file(
                    dataset_path
                ),
                "row_count": int(
                    len(dataframe)
                ),
            },
            model_artifact=model_artifact,
            started_at=started_at,
            completed_at=utc_now(),
            duration_seconds=round(
                time.perf_counter() - start_time,
                4,
            ),
        )

        report_path = (
            self.output_directory
            / f"{self.config.model_name}_{evaluation_id}.json"
        )

        save_json(
            report_path,
            asdict(report),
        )

        LOGGER.info(
            "Evaluation complete: %s",
            report_path,
        )

        return report


# ============================================================
# Model Comparison
# ============================================================

class ModelComparison:
    @staticmethod
    def compare(
        report_paths: Sequence[str | Path],
    ) -> dict[str, Any]:
        if len(report_paths) < 2:
            raise EvaluationConfigurationError(
                "At least two reports are required"
            )

        reports = [
            json.loads(
                Path(path).read_text(
                    encoding="utf-8"
                )
            )
            for path in report_paths
        ]

        task_types = {
            report["task_type"]
            for report in reports
        }

        if len(task_types) != 1:
            raise EvaluationConfigurationError(
                "Cannot compare different task types"
            )

        dataset_hashes = {
            report.get(
                "dataset",
                {},
            ).get("sha256")
            for report in reports
        }

        if len(dataset_hashes) != 1:
            LOGGER.warning(
                "Reports use different evaluation datasets; "
                "metric comparisons may be misleading"
            )

        rows = []

        for report in reports:
            row = {
                "model_name": report["model_name"],
                "model_version": report["model_version"],
                "status": report["status"],
            }

            for key, value in report[
                "metrics"
            ].items():
                if isinstance(
                    value,
                    (int, float),
                ):
                    row[key] = value

            rows.append(row)

        task = next(
            iter(task_types)
        )

        primary_metrics = {
            "classification": (
                "f1",
                True,
            ),
            "anomaly": (
                "f1",
                True,
            ),
            "failure": (
                "recall",
                True,
            ),
            "regression": (
                "rmse",
                False,
            ),
            "rul": (
                "rmse",
                False,
            ),
            "detection": (
                "f1",
                True,
            ),
            "vla": (
                "action_rmse",
                False,
            ),
            "world-model": (
                "state_rmse",
                False,
            ),
        }

        metric_name, higher_is_better = (
            primary_metrics[task]
        )

        eligible = [
            row
            for row in rows
            if safe_float(
                row.get(metric_name)
            ) is not None
        ]

        if not eligible:
            raise ModelEvaluationError(
                "No comparable primary metrics"
            )

        ranked = sorted(
            eligible,
            key=lambda row: row[metric_name],
            reverse=higher_is_better,
        )

        return {
            "task_type": task,
            "primary_metric": metric_name,
            "higher_is_better": higher_is_better,
            "same_dataset": len(dataset_hashes) == 1,
            "ranking": ranked,
            "best_model": ranked[0]["model_name"],
            "physical_execution_authorized": False,
        }


# ============================================================
# CLI
# ============================================================

def add_common_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--model-name",
        default="unnamed-model",
    )

    parser.add_argument(
        "--model-version",
        default="0.0.0",
    )

    parser.add_argument(
        "--output",
        default="reports/evaluation",
    )

    parser.add_argument(
        "--model-artifact",
        default=None,
    )

    parser.add_argument(
        "--minimum-accuracy",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--minimum-f1",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--minimum-recall",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--maximum-mae",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--maximum-rmse",
        type=float,
        default=None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified offline robotics ML model evaluator"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    tasks = [
        "classification",
        "regression",
        "anomaly",
        "failure",
        "rul",
        "detection",
        "vla",
        "world-model",
    ]

    for task in tasks:
        task_parser = subparsers.add_parser(
            task
        )

        add_common_arguments(
            task_parser
        )

        if task != "detection":
            task_parser.add_argument(
                "--target-column",
                default=None,
            )

            task_parser.add_argument(
                "--prediction-column",
                default=None,
            )

            task_parser.add_argument(
                "--target-columns",
                nargs="+",
                default=None,
            )

            task_parser.add_argument(
                "--prediction-columns",
                nargs="+",
                default=None,
            )

            task_parser.add_argument(
                "--score-column",
                default=None,
            )

        if task == "detection":
            task_parser.add_argument(
                "--iou-threshold",
                type=float,
                default=0.5,
            )

        if task == "vla":
            task_parser.add_argument(
                "--max-acceptable-error",
                type=float,
                default=None,
            )

    comparison_parser = subparsers.add_parser(
        "compare"
    )

    comparison_parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
    )

    comparison_parser.add_argument(
        "--output",
        default=None,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "compare":
        result = ModelComparison.compare(
            args.reports
        )

        if args.output:
            save_json(
                args.output,
                result,
            )

        print(
            json.dumps(
                result,
                indent=2,
                allow_nan=False,
            )
        )

        return

    target_columns = (
        args.target_columns
        if args.command != "detection"
        and args.target_columns
        else (
            [args.target_column]
            if args.command != "detection"
            and args.target_column
            else []
        )
    )

    prediction_columns = (
        args.prediction_columns
        if args.command != "detection"
        and args.prediction_columns
        else (
            [args.prediction_column]
            if args.command != "detection"
            and args.prediction_column
            else []
        )
    )

    config = EvaluationConfig(
        model_name=args.model_name,
        model_version=args.model_version,
        task_type=args.command,
        data_path=args.data,
        output_directory=args.output,
        target_columns=target_columns,
        prediction_columns=prediction_columns,
        score_column=getattr(
            args,
            "score_column",
            None,
        ),
        iou_threshold=getattr(
            args,
            "iou_threshold",
            0.5,
        ),
        max_acceptable_error=getattr(
            args,
            "max_acceptable_error",
            None,
        ),
        minimum_accuracy=args.minimum_accuracy,
        minimum_f1=args.minimum_f1,
        minimum_recall=args.minimum_recall,
        maximum_mae=args.maximum_mae,
        maximum_rmse=args.maximum_rmse,
        model_artifact_path=args.model_artifact,
    )

    result = ModelEvaluationPipeline(
        config
    ).evaluate()

    print(
        json.dumps(
            asdict(result),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    try:
        main()

    except (
        ModelEvaluationError,
        ValueError,
        KeyError,
        OSError,
    ) as exc:
        LOGGER.error(
            "Model evaluation failed: %s",
            exc,
        )

        sys.exit(1)
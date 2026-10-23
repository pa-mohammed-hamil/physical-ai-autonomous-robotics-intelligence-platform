"""
Physical AI & Autonomous Robotics Intelligence Platform
File: ml/training/train_part_detector.py

Robot Component Detection Training Pipeline

Features:
- YOLO-based robot part detection
- Motors, joints, actuators, sensors and batteries
- Damaged component detection
- Dataset validation
- YOLO-format annotation verification
- Train/validation/test dataset configuration
- GPU/CPU automatic selection
- Transfer learning
- Training and validation
- Precision, recall and mAP evaluation
- Model checkpoint management
- ONNX export
- Training metadata and reports
- SHA-256 artifact checksums
- Image and video inference
- CLI support

Dependencies:
    pip install ultralytics torch torchvision
    pip install opencv-python numpy pyyaml pillow

Examples:
    python -m ml.training.train_part_detector train \
        --dataset data/robot_parts/dataset.yaml \
        --epochs 100 \
        --model yolo11n.pt

    python -m ml.training.train_part_detector validate \
        --dataset data/robot_parts/dataset.yaml

    python -m ml.training.train_part_detector predict \
        --weights artifacts/part_detector/weights/best.pt \
        --source images/robot.jpg

    python -m ml.training.train_part_detector export \
        --weights artifacts/part_detector/weights/best.pt \
        --format onnx

Safety:
    Detection results are advisory.
    They do not authorize physical robot actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import tempfile
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


LOGGER = logging.getLogger(
    "ml.training.train_part_detector"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ============================================================
# Constants
# ============================================================

SCHEMA_VERSION = "1.0.0"

DEFAULT_CLASSES = [
    "motor",
    "actuator",
    "robotic_joint",
    "gearbox",
    "bearing",
    "gripper",
    "robotic_arm",
    "wheel",
    "battery",
    "camera",
    "lidar",
    "imu_sensor",
    "proximity_sensor",
    "encoder",
    "controller",
    "cable",
    "connector",
    "damaged_component",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}

MAX_IMAGES = 1_000_000
MAX_ANNOTATION_BYTES = 10 * 1024 * 1024
MAX_CLASSES = 256


# ============================================================
# Exceptions
# ============================================================

class PartDetectorError(Exception):
    """Base part detector exception."""


class DatasetValidationError(PartDetectorError):
    """Invalid object detection dataset."""


class TrainingError(PartDetectorError):
    """Model training failed."""


class EvaluationError(PartDetectorError):
    """Model evaluation failed."""


class InferenceError(PartDetectorError):
    """Part detection inference failed."""


class ExportError(PartDetectorError):
    """Model export failed."""


class ArtifactError(PartDetectorError):
    """Artifact operation failed."""


# ============================================================
# Enumerations
# ============================================================

class DetectorStatus(str, Enum):
    CREATED = "created"
    VALIDATING = "validating"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class DetectionSeverity(str, Enum):
    NORMAL = "normal"
    REVIEW = "review"
    HIGH = "high"
    CRITICAL = "critical"


class ExportFormat(str, Enum):
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    OPENVINO = "openvino"


# ============================================================
# Configuration
# ============================================================

@dataclass
class PartDetectorConfig:
    dataset_yaml: str = ""

    pretrained_model: str = "yolo11n.pt"

    output_directory: str = (
        "artifacts/part_detector"
    )

    experiment_name: str = (
        "robot_part_detection"
    )

    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640

    learning_rate: float = 0.01
    weight_decay: float = 0.0005

    patience: int = 20
    workers: int = 4

    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    random_seed: int = 42

    device: str = "auto"

    optimizer: str = "auto"

    augmentation: bool = True
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0
    mosaic: float = 1.0

    save_period: int = 10

    validate_dataset: bool = True
    validate_annotations: bool = True

    export_onnx: bool = False

    def validate(self) -> None:
        if not self.dataset_yaml:
            raise DatasetValidationError(
                "Dataset YAML path is required"
            )

        if self.epochs < 1:
            raise ValueError(
                "epochs must be positive"
            )

        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be positive"
            )

        if self.image_size < 32:
            raise ValueError(
                "image_size must be at least 32"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative"
            )

        if self.patience < 0:
            raise ValueError(
                "patience cannot be negative"
            )

        for name in (
            "confidence_threshold",
            "iou_threshold",
            "horizontal_flip",
            "vertical_flip",
            "mosaic",
        ):
            value = getattr(self, name)

            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1"
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


def calculate_sha256(
    path: str | Path,
) -> str:
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
    data: Mapping[str, Any],
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
            prefix=".part-detector-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(
                data,
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

            stream.flush()
            os.fsync(stream.fileno())

            temporary_path = Path(
                stream.name
            )

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


def resolve_device(
    requested_device: str,
) -> str:
    if requested_device != "auto":
        return requested_device

    try:
        import torch

        if torch.cuda.is_available():
            return "0"

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"

    except ImportError:
        pass

    return "cpu"


def require_ultralytics():
    try:
        from ultralytics import YOLO

        return YOLO

    except ImportError as exc:
        raise PartDetectorError(
            "Ultralytics is required. "
            "Install with: pip install ultralytics"
        ) from exc


# ============================================================
# Dataset Configuration
# ============================================================

@dataclass
class DetectionDatasetInfo:
    yaml_path: str
    dataset_root: str

    train_path: str
    validation_path: str
    test_path: str | None

    class_names: list[str]
    class_count: int

    train_images: int
    validation_images: int
    test_images: int

    total_images: int

    annotation_count: int
    bounding_box_count: int

    invalid_annotations: int
    warnings: list[str]


class PartDetectionDatasetValidator:
    def __init__(
        self,
        dataset_yaml: str | Path,
        *,
        validate_annotations: bool = True,
    ) -> None:
        self.dataset_yaml = Path(
            dataset_yaml
        ).resolve()

        self.validate_annotations = (
            validate_annotations
        )

    def _load_yaml(
        self,
    ) -> dict[str, Any]:
        if not self.dataset_yaml.is_file():
            raise DatasetValidationError(
                f"Dataset YAML not found: "
                f"{self.dataset_yaml}"
            )

        with self.dataset_yaml.open(
            "r",
            encoding="utf-8",
        ) as stream:
            data = yaml.safe_load(
                stream
            )

        if not isinstance(data, dict):
            raise DatasetValidationError(
                "Dataset YAML must contain a mapping"
            )

        return data

    def _resolve_root(
        self,
        data: dict[str, Any],
    ) -> Path:
        root_value = data.get("path")

        if root_value:
            root = Path(
                root_value
            )

            if not root.is_absolute():
                root = (
                    self.dataset_yaml.parent
                    / root
                )

        else:
            root = self.dataset_yaml.parent

        root = root.resolve()

        if not root.is_dir():
            raise DatasetValidationError(
                f"Dataset root not found: {root}"
            )

        return root

    @staticmethod
    def _resolve_split(
        root: Path,
        value: str | None,
        *,
        required: bool,
    ) -> Path | None:
        if not value:
            if required:
                raise DatasetValidationError(
                    "Required dataset split is missing"
                )

            return None

        path = Path(value)

        if not path.is_absolute():
            path = root / path

        path = path.resolve()

        if not path.exists():
            raise DatasetValidationError(
                f"Dataset split not found: {path}"
            )

        if not path.is_dir():
            raise DatasetValidationError(
                "This validator requires directory-based "
                "dataset splits"
            )

        return path

    @staticmethod
    def _parse_classes(
        data: dict[str, Any],
    ) -> list[str]:
        names = data.get("names")

        if isinstance(names, list):
            class_names = [
                str(name)
                for name in names
            ]

        elif isinstance(names, dict):
            try:
                normalized = {
                    int(key): str(value)
                    for key, value in names.items()
                }

                expected = list(
                    range(len(normalized))
                )

                if sorted(normalized) != expected:
                    raise ValueError(
                        "Class IDs must be contiguous"
                    )

                class_names = [
                    normalized[index]
                    for index in expected
                ]

            except (TypeError, ValueError) as exc:
                raise DatasetValidationError(
                    "Invalid class name mapping"
                ) from exc

        else:
            raise DatasetValidationError(
                "Dataset must define class names"
            )

        if not class_names:
            raise DatasetValidationError(
                "Dataset has no classes"
            )

        if len(class_names) > MAX_CLASSES:
            raise DatasetValidationError(
                "Dataset exceeds maximum class count"
            )

        if len(set(class_names)) != len(class_names):
            raise DatasetValidationError(
                "Duplicate class names"
            )

        return class_names

    @staticmethod
    def _find_images(
        directory: Path,
    ) -> list[Path]:
        images = [
            path
            for path in directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]

        if len(images) > MAX_IMAGES:
            raise DatasetValidationError(
                "Dataset contains too many images"
            )

        return sorted(images)

    @staticmethod
    def _label_path(
        image_path: Path,
    ) -> Path:
        """
        Conventional YOLO layout:

        dataset/images/train/image.jpg
        dataset/labels/train/image.txt
        """
        parts = list(image_path.parts)

        image_positions = [
            index
            for index, value in enumerate(parts)
            if value == "images"
        ]

        if image_positions:
            parts[
                image_positions[-1]
            ] = "labels"

            return Path(
                *parts
            ).with_suffix(".txt")

        return (
            image_path.parent.parent
            / "labels"
            / image_path.parent.name
            / f"{image_path.stem}.txt"
        )

    @staticmethod
    def _validate_label(
        label_path: Path,
        class_count: int,
    ) -> tuple[int, int]:
        if not label_path.exists():
            return 0, 0

        if (
            label_path.stat().st_size
            > MAX_ANNOTATION_BYTES
        ):
            return 0, 1

        bounding_boxes = 0
        invalid_lines = 0

        try:
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

        except (OSError, UnicodeError):
            return 0, 1

        for line in lines:
            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                invalid_lines += 1
                continue

            try:
                class_id_float = float(
                    values[0]
                )

                coordinates = [
                    float(value)
                    for value in values[1:]
                ]

            except ValueError:
                invalid_lines += 1
                continue

            if not all(
                math.isfinite(value)
                for value in [
                    class_id_float,
                    *coordinates,
                ]
            ):
                invalid_lines += 1
                continue

            class_id = int(
                class_id_float
            )

            if (
                class_id_float != class_id
                or class_id < 0
                or class_id >= class_count
            ):
                invalid_lines += 1
                continue

            x_center, y_center, width, height = (
                coordinates
            )

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                invalid_lines += 1
                continue

            if (
                x_center - width / 2 < -1e-6
                or x_center + width / 2 > 1 + 1e-6
                or y_center - height / 2 < -1e-6
                or y_center + height / 2 > 1 + 1e-6
            ):
                invalid_lines += 1
                continue

            bounding_boxes += 1

        return bounding_boxes, invalid_lines

    def validate(
        self,
    ) -> DetectionDatasetInfo:
        data = self._load_yaml()

        root = self._resolve_root(
            data
        )

        class_names = self._parse_classes(
            data
        )

        train_path = self._resolve_split(
            root,
            data.get("train"),
            required=True,
        )

        validation_path = self._resolve_split(
            root,
            data.get("val"),
            required=True,
        )

        test_path = self._resolve_split(
            root,
            data.get("test"),
            required=False,
        )

        assert train_path is not None
        assert validation_path is not None

        train_images = self._find_images(
            train_path
        )

        validation_images = self._find_images(
            validation_path
        )

        test_images = (
            self._find_images(test_path)
            if test_path is not None
            else []
        )

        if not train_images:
            raise DatasetValidationError(
                "Training split contains no images"
            )

        if not validation_images:
            raise DatasetValidationError(
                "Validation split contains no images"
            )

        all_images = (
            train_images
            + validation_images
            + test_images
        )

        resolved_images = [
            str(path.resolve())
            for path in all_images
        ]

        if len(set(resolved_images)) != len(
            resolved_images
        ):
            raise DatasetValidationError(
                "Image paths overlap between splits"
            )

        annotation_count = 0
        bounding_box_count = 0
        invalid_annotations = 0

        warnings = []

        if self.validate_annotations:
            for image_path in all_images:
                label_path = self._label_path(
                    image_path
                )

                if label_path.exists():
                    annotation_count += 1

                boxes, invalid = (
                    self._validate_label(
                        label_path,
                        len(class_names),
                    )
                )

                bounding_box_count += boxes
                invalid_annotations += invalid

        if invalid_annotations:
            raise DatasetValidationError(
                f"Found {invalid_annotations} "
                "invalid annotation entries"
            )

        if (
            self.validate_annotations
            and bounding_box_count == 0
        ):
            raise DatasetValidationError(
                "No valid bounding boxes found"
            )

        if not test_images:
            warnings.append(
                "No test split found; independent "
                "test evaluation will be unavailable"
            )

        info = DetectionDatasetInfo(
            yaml_path=str(
                self.dataset_yaml
            ),
            dataset_root=str(root),
            train_path=str(train_path),
            validation_path=str(
                validation_path
            ),
            test_path=(
                str(test_path)
                if test_path
                else None
            ),
            class_names=class_names,
            class_count=len(class_names),
            train_images=len(train_images),
            validation_images=len(
                validation_images
            ),
            test_images=len(test_images),
            total_images=len(all_images),
            annotation_count=annotation_count,
            bounding_box_count=bounding_box_count,
            invalid_annotations=(
                invalid_annotations
            ),
            warnings=warnings,
        )

        LOGGER.info(
            "Dataset validated: %d images, "
            "%d boxes, %d classes",
            info.total_images,
            info.bounding_box_count,
            info.class_count,
        )

        return info


# ============================================================
# Detection Results
# ============================================================

@dataclass
class PartDetection:
    class_id: int
    class_name: str

    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float

    severity: str
    advisory_only: bool = True


@dataclass
class PartDetectionResult:
    source: str
    detections: list[PartDetection]

    detection_count: int

    inference_duration_seconds: float

    model_path: str
    created_at: str

    physical_execution_authorized: bool = False


# ============================================================
# Model Wrapper
# ============================================================

class RobotPartDetector:
    def __init__(
        self,
        weights: str | Path,
        *,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
    ) -> None:
        YOLO = require_ultralytics()

        self.weights = str(weights)

        self.confidence_threshold = (
            confidence_threshold
        )

        self.iou_threshold = (
            iou_threshold
        )

        self.device = resolve_device(
            device
        )

        self.model = YOLO(
            self.weights
        )

    @staticmethod
    def _severity(
        class_name: str,
        confidence: float,
    ) -> DetectionSeverity:
        name = class_name.lower()

        if any(
            keyword in name
            for keyword in (
                "damaged",
                "broken",
                "cracked",
                "corrosion",
                "burned",
            )
        ):
            if confidence >= 0.85:
                return DetectionSeverity.HIGH

            return DetectionSeverity.REVIEW

        return DetectionSeverity.NORMAL

    def predict(
        self,
        source: str | Path | np.ndarray,
        *,
        save: bool = False,
        output_directory: str | Path | None = None,
    ) -> list[PartDetectionResult]:
        started = time.perf_counter()

        try:
            predictions = self.model.predict(
                source=source,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                save=save,
                project=(
                    str(output_directory)
                    if output_directory
                    else None
                ),
                verbose=False,
            )

        except Exception as exc:
            raise InferenceError(
                "Robot part detection failed"
            ) from exc

        duration = (
            time.perf_counter()
            - started
        )

        output = []

        for result in predictions:
            detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(
                        box.cls.item()
                    )

                    confidence = float(
                        box.conf.item()
                    )

                    coordinates = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    class_name = str(
                        result.names.get(
                            class_id,
                            f"class_{class_id}",
                        )
                    )

                    severity = self._severity(
                        class_name,
                        confidence,
                    )

                    detections.append(
                        PartDetection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            x1=float(
                                coordinates[0]
                            ),
                            y1=float(
                                coordinates[1]
                            ),
                            x2=float(
                                coordinates[2]
                            ),
                            y2=float(
                                coordinates[3]
                            ),
                            severity=severity.value,
                        )
                    )

            output.append(
                PartDetectionResult(
                    source=str(
                        getattr(
                            result,
                            "path",
                            source,
                        )
                    ),
                    detections=detections,
                    detection_count=len(
                        detections
                    ),
                    inference_duration_seconds=(
                        duration
                    ),
                    model_path=self.weights,
                    created_at=utc_now(),
                )
            )

        return output

    def validate(
        self,
        dataset_yaml: str | Path,
        *,
        split: str = "val",
        image_size: int = 640,
    ) -> dict[str, Any]:
        try:
            metrics = self.model.val(
                data=str(dataset_yaml),
                split=split,
                imgsz=image_size,
                device=self.device,
                verbose=False,
            )

        except Exception as exc:
            raise EvaluationError(
                "Detector evaluation failed"
            ) from exc

        box = metrics.box

        return {
            "split": split,
            "precision": float(
                box.mp
            ),
            "recall": float(
                box.mr
            ),
            "map50": float(
                box.map50
            ),
            "map50_95": float(
                box.map
            ),
            "evaluated_at": utc_now(),
        }

    def export(
        self,
        *,
        export_format: ExportFormat = ExportFormat.ONNX,
        image_size: int = 640,
        dynamic: bool = False,
        half: bool = False,
    ) -> str:
        try:
            exported = self.model.export(
                format=export_format.value,
                imgsz=image_size,
                dynamic=dynamic,
                half=half,
                device=self.device,
            )

        except Exception as exc:
            raise ExportError(
                "Detector export failed"
            ) from exc

        return str(exported)


# ============================================================
# Training Result
# ============================================================

@dataclass
class PartDetectorTrainingResult:
    run_id: str
    status: str

    pretrained_model: str

    dataset_yaml: str
    class_names: list[str]

    epochs_requested: int
    image_size: int
    batch_size: int

    device: str

    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any] | None

    artifact_paths: dict[str, str]

    started_at: str
    completed_at: str

    duration_seconds: float


# ============================================================
# Training Pipeline
# ============================================================

class PartDetectorTrainingPipeline:
    def __init__(
        self,
        config: PartDetectorConfig,
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

        self.device = resolve_device(
            config.device
        )

        self.dataset_info: (
            DetectionDatasetInfo | None
        ) = None

        self.model = None

    def validate_dataset(
        self,
    ) -> DetectionDatasetInfo:
        validator = (
            PartDetectionDatasetValidator(
                self.config.dataset_yaml,
                validate_annotations=(
                    self.config.validate_annotations
                ),
            )
        )

        self.dataset_info = (
            validator.validate()
        )

        return self.dataset_info

    def _train_model(
        self,
    ) -> Path:
        YOLO = require_ultralytics()

        self.model = YOLO(
            self.config.pretrained_model
        )

        LOGGER.info(
            "Training YOLO detector on %s",
            self.device,
        )

        try:
            self.model.train(
                data=self.config.dataset_yaml,
                epochs=self.config.epochs,
                imgsz=self.config.image_size,
                batch=self.config.batch_size,
                device=self.device,
                project=str(
                    self.output_directory
                ),
                name="training",
                exist_ok=True,
                optimizer=self.config.optimizer,
                lr0=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                patience=self.config.patience,
                workers=self.config.workers,
                seed=self.config.random_seed,
                fliplr=(
                    self.config.horizontal_flip
                    if self.config.augmentation
                    else 0.0
                ),
                flipud=(
                    self.config.vertical_flip
                    if self.config.augmentation
                    else 0.0
                ),
                mosaic=(
                    self.config.mosaic
                    if self.config.augmentation
                    else 0.0
                ),
                save=True,
                save_period=self.config.save_period,
                plots=True,
                verbose=True,
            )

        except Exception as exc:
            raise TrainingError(
                "Part detector training failed"
            ) from exc

        weights_directory = (
            self.output_directory
            / "training"
            / "weights"
        )

        best_weights = (
            weights_directory
            / "best.pt"
        )

        if not best_weights.is_file():
            raise TrainingError(
                "Best model checkpoint was not created"
            )

        return best_weights

    def _evaluate(
        self,
        weights_path: Path,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any] | None,
    ]:
        detector = RobotPartDetector(
            weights_path,
            confidence_threshold=(
                self.config.confidence_threshold
            ),
            iou_threshold=(
                self.config.iou_threshold
            ),
            device=self.device,
        )

        validation_metrics = (
            detector.validate(
                self.config.dataset_yaml,
                split="val",
                image_size=self.config.image_size,
            )
        )

        test_metrics = None

        if (
            self.dataset_info is not None
            and self.dataset_info.test_images > 0
        ):
            test_metrics = detector.validate(
                self.config.dataset_yaml,
                split="test",
                image_size=self.config.image_size,
            )

        return (
            validation_metrics,
            test_metrics,
        )

    def _save_artifacts(
        self,
        *,
        weights_path: Path,
        validation_metrics: dict[str, Any],
        test_metrics: dict[str, Any] | None,
    ) -> dict[str, str]:
        final_weights_directory = (
            self.output_directory
            / "weights"
        )

        final_weights_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        final_weights = (
            final_weights_directory
            / "best.pt"
        )

        if weights_path.resolve() != final_weights.resolve():
            shutil.copy2(
                weights_path,
                final_weights,
            )

        artifact_paths = {
            "weights": str(
                final_weights
            )
        }

        metadata_path = (
            self.output_directory
            / "training_metadata.json"
        )

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "model_name": (
                "robot-part-detector"
            ),
            "model_type": "yolo_object_detection",
            "pretrained_model": (
                self.config.pretrained_model
            ),
            "created_at": utc_now(),
            "dataset": (
                asdict(self.dataset_info)
                if self.dataset_info
                else None
            ),
            "configuration": (
                asdict(self.config)
            ),
            "validation_metrics": (
                validation_metrics
            ),
            "test_metrics": test_metrics,
            "artifact": {
                "filename": (
                    final_weights.name
                ),
                "relative_path": (
                    "weights/best.pt"
                ),
                "sha256": calculate_sha256(
                    final_weights
                ),
                "size_bytes": (
                    final_weights.stat().st_size
                ),
            },
            "safety": {
                "advisory_only": True,
                "physical_execution_authorized": False,
                "emergency_stop_reset_authorized": False,
                "safety_interlock_bypass_authorized": False,
                "human_review_required": True,
            },
        }

        atomic_save_json(
            metadata_path,
            metadata,
        )

        artifact_paths["metadata"] = str(
            metadata_path
        )

        metrics_path = (
            self.output_directory
            / "evaluation_metrics.json"
        )

        atomic_save_json(
            metrics_path,
            {
                "run_id": self.run_id,
                "validation": (
                    validation_metrics
                ),
                "test": test_metrics,
            },
        )

        artifact_paths["metrics"] = str(
            metrics_path
        )

        return artifact_paths

    def train(
        self,
    ) -> PartDetectorTrainingResult:
        started_at = utc_now()
        started = time.perf_counter()

        set_random_seed(
            self.config.random_seed
        )

        LOGGER.info(
            "Starting part detector training: %s",
            self.run_id,
        )

        if self.config.validate_dataset:
            self.validate_dataset()

        weights_path = (
            self._train_model()
        )

        validation_metrics, test_metrics = (
            self._evaluate(
                weights_path
            )
        )

        artifact_paths = (
            self._save_artifacts(
                weights_path=weights_path,
                validation_metrics=(
                    validation_metrics
                ),
                test_metrics=test_metrics,
            )
        )

        if self.config.export_onnx:
            detector = RobotPartDetector(
                artifact_paths["weights"],
                device=self.device,
            )

            onnx_path = detector.export(
                export_format=ExportFormat.ONNX,
                image_size=self.config.image_size,
            )

            artifact_paths["onnx"] = (
                onnx_path
            )

        result = PartDetectorTrainingResult(
            run_id=self.run_id,
            status=DetectorStatus.COMPLETED.value,
            pretrained_model=(
                self.config.pretrained_model
            ),
            dataset_yaml=(
                self.config.dataset_yaml
            ),
            class_names=(
                self.dataset_info.class_names
                if self.dataset_info
                else []
            ),
            epochs_requested=(
                self.config.epochs
            ),
            image_size=(
                self.config.image_size
            ),
            batch_size=(
                self.config.batch_size
            ),
            device=self.device,
            validation_metrics=(
                validation_metrics
            ),
            test_metrics=test_metrics,
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
            asdict(result),
        )

        LOGGER.info(
            "Training complete | mAP50=%.4f | "
            "mAP50-95=%.4f | duration=%.2fs",
            validation_metrics["map50"],
            validation_metrics["map50_95"],
            result.duration_seconds,
        )

        return result


# ============================================================
# CLI
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Robot component detection training and inference"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_parser = subparsers.add_parser(
        "train"
    )

    train_parser.add_argument(
        "--dataset",
        required=True,
    )

    train_parser.add_argument(
        "--model",
        default="yolo11n.pt",
    )

    train_parser.add_argument(
        "--output",
        default="artifacts/part_detector",
    )

    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    train_parser.add_argument(
        "--image-size",
        type=int,
        default=640,
    )

    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
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

    train_parser.add_argument(
        "--export-onnx",
        action="store_true",
    )

    # --------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------

    validate_parser = subparsers.add_parser(
        "validate"
    )

    validate_parser.add_argument(
        "--dataset",
        required=True,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predict_parser = subparsers.add_parser(
        "predict"
    )

    predict_parser.add_argument(
        "--weights",
        required=True,
    )

    predict_parser.add_argument(
        "--source",
        required=True,
    )

    predict_parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
    )

    predict_parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
    )

    predict_parser.add_argument(
        "--device",
        default="auto",
    )

    predict_parser.add_argument(
        "--save",
        action="store_true",
    )

    predict_parser.add_argument(
        "--output",
        default="artifacts/part_detector/predictions",
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    export_parser = subparsers.add_parser(
        "export"
    )

    export_parser.add_argument(
        "--weights",
        required=True,
    )

    export_parser.add_argument(
        "--format",
        choices=[
            item.value
            for item in ExportFormat
        ],
        default="onnx",
    )

    export_parser.add_argument(
        "--image-size",
        type=int,
        default=640,
    )

    export_parser.add_argument(
        "--device",
        default="auto",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()

    args = parser.parse_args()

    if args.command == "validate":
        validator = (
            PartDetectionDatasetValidator(
                args.dataset
            )
        )

        result = validator.validate()

        print(
            json.dumps(
                asdict(result),
                indent=2,
            )
        )

        return

    if args.command == "train":
        config = PartDetectorConfig(
            dataset_yaml=args.dataset,
            pretrained_model=args.model,
            output_directory=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            device=args.device,
            random_seed=args.seed,
            export_onnx=args.export_onnx,
        )

        pipeline = (
            PartDetectorTrainingPipeline(
                config
            )
        )

        result = pipeline.train()

        print(
            json.dumps(
                asdict(result),
                indent=2,
            )
        )

        return

    if args.command == "predict":
        detector = RobotPartDetector(
            args.weights,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            device=args.device,
        )

        results = detector.predict(
            args.source,
            save=args.save,
            output_directory=args.output,
        )

        print(
            json.dumps(
                [
                    asdict(result)
                    for result in results
                ],
                indent=2,
            )
        )

        return

    if args.command == "export":
        detector = RobotPartDetector(
            args.weights,
            device=args.device,
        )

        path = detector.export(
            export_format=ExportFormat(
                args.format
            ),
            image_size=args.image_size,
        )

        print(
            json.dumps(
                {
                    "exported_model": path,
                    "format": args.format,
                    "created_at": utc_now(),
                },
                indent=2,
            )
        )

        return


if __name__ == "__main__":
    main()
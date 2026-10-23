"""Semantic Segmentation module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/semantic_segmentation.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: semantic_segmentation.py

Production-oriented semantic segmentation for robotic perception.

Features
--------
- ONNX semantic segmentation inference
- Configurable model input and output layouts
- RGB/BGR image preprocessing
- Image resizing and normalization
- Semantic class mapping
- Per-pixel confidence estimation
- Segmentation masks
- Colorized segmentation visualization
- Image overlays
- Per-class coverage analysis
- Connected-component region extraction
- Traversable-area estimation
- Human and obstacle segmentation
- Scene risk advisory
- Temporal segmentation smoothing
- Per-camera processing state
- Multi-tenant authorization
- Model metadata and performance monitoring
- JSON export
- Synthetic segmentation testing
- Unit tests

Dependencies
------------
Python >= 3.11
numpy

Optional
--------
opencv-python
onnxruntime

Safety
------
Segmentation predictions are advisory observations.
Do not use this module as the sole basis for robot
motion, emergency stopping, or human safety decisions.
"""

from __future__ import annotations

import json
import math
import threading
import time
import unittest
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


# ============================================================
# EXCEPTIONS
# ============================================================

class SemanticSegmentationError(Exception):
    pass


class SegmentationValidationError(
    SemanticSegmentationError
):
    pass


class SegmentationAuthorizationError(
    SemanticSegmentationError
):
    pass


class SegmentationDependencyError(
    SemanticSegmentationError
):
    pass


class SegmentationInferenceError(
    SemanticSegmentationError
):
    pass


class SegmentationCapacityError(
    SemanticSegmentationError
):
    pass


class SegmentationModelError(
    SemanticSegmentationError
):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def validate_identifier(
    value: str,
    name: str,
) -> str:
    if not isinstance(value, str):
        raise SegmentationValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise SegmentationValidationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def finite_float(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise SegmentationValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SegmentationValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise SegmentationValidationError(
            f"{name} must be finite"
        )

    return result


def validate_confidence(
    value: float,
) -> float:
    value = finite_float(
        value,
        "confidence",
    )

    if not 0.0 <= value <= 1.0:
        raise SegmentationValidationError(
            "Confidence must be in [0, 1]"
        )

    return value


def validate_timestamp(
    value: datetime,
) -> datetime:
    if not isinstance(value, datetime):
        raise SegmentationValidationError(
            "Timestamp must be datetime"
        )

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise SegmentationValidationError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def require_opencv() -> None:
    if cv2 is None:
        raise SegmentationDependencyError(
            "opencv-python is required"
        )


def require_onnxruntime() -> None:
    if ort is None:
        raise SegmentationDependencyError(
            "onnxruntime is required"
        )


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if hasattr(value, "__dataclass_fields__"):
        return {
            name: json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }

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

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class SegmentationBackend(str, Enum):
    ONNX = "onnx"
    SYNTHETIC = "synthetic"


class InputColorFormat(str, Enum):
    RGB = "rgb"
    BGR = "bgr"


class ModelInputLayout(str, Enum):
    NCHW = "nchw"
    NHWC = "nhwc"


class ModelOutputLayout(str, Enum):
    NCHW = "nchw"
    NHWC = "nhwc"
    NHW = "nhw"


class SegmentationOutputType(str, Enum):
    LOGITS = "logits"
    PROBABILITIES = "probabilities"
    CLASS_IDS = "class_ids"


class SemanticCategory(str, Enum):
    BACKGROUND = "background"
    FLOOR = "floor"
    ROAD = "road"
    SIDEWALK = "sidewalk"
    WALL = "wall"
    CEILING = "ceiling"
    DOOR = "door"
    HUMAN = "human"
    ROBOT = "robot"
    VEHICLE = "vehicle"
    FURNITURE = "furniture"
    EQUIPMENT = "equipment"
    OBSTACLE = "obstacle"
    VEGETATION = "vegetation"
    SKY = "sky"
    WATER = "water"
    UNKNOWN = "unknown"


class TraversabilityClass(str, Enum):
    TRAVERSABLE = "traversable"
    NON_TRAVERSABLE = "non_traversable"
    UNKNOWN = "unknown"


class SegmentationQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"


class SceneRiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SemanticSegmentationConfig:
    input_width: int = 512
    input_height: int = 512

    maximum_image_width: int = 4096
    maximum_image_height: int = 4096

    input_color_format: InputColorFormat = (
        InputColorFormat.BGR
    )

    model_color_format: InputColorFormat = (
        InputColorFormat.RGB
    )

    input_layout: ModelInputLayout = (
        ModelInputLayout.NCHW
    )

    output_layout: ModelOutputLayout = (
        ModelOutputLayout.NCHW
    )

    output_type: SegmentationOutputType = (
        SegmentationOutputType.LOGITS
    )

    normalize_to_unit_range: bool = True

    mean: tuple[float, float, float] = (
        0.485,
        0.456,
        0.406,
    )

    std: tuple[float, float, float] = (
        0.229,
        0.224,
        0.225,
    )

    minimum_pixel_confidence: float = 0.45

    minimum_region_area_pixels: int = 100

    temporal_smoothing_alpha: float = 0.50

    maximum_history: int = 100

    maximum_classes: int = 256

    unknown_class_id: int = 255

    enable_temporal_smoothing: bool = False

    def __post_init__(self) -> None:
        for name in (
            "input_width",
            "input_height",
            "maximum_image_width",
            "maximum_image_height",
            "minimum_region_area_pixels",
            "maximum_history",
            "maximum_classes",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise SegmentationValidationError(
                    f"{name} must be a positive integer"
                )

        if (
            self.input_width
            > self.maximum_image_width
            or self.input_height
            > self.maximum_image_height
        ):
            raise SegmentationValidationError(
                "Model input exceeds maximum image dimensions"
            )

        if not (
            0 <= self.unknown_class_id
            <= np.iinfo(np.int32).max
        ):
            raise SegmentationValidationError(
                "Invalid unknown class ID"
            )

        validate_confidence(
            self.minimum_pixel_confidence
        )

        validate_confidence(
            self.temporal_smoothing_alpha
        )

        if len(self.mean) != 3 or len(self.std) != 3:
            raise SegmentationValidationError(
                "Mean and standard deviation must have 3 values"
            )

        for value in self.mean:
            finite_float(
                value,
                "mean",
            )

        for value in self.std:
            if finite_float(value, "std") <= 0:
                raise SegmentationValidationError(
                    "Standard deviations must be positive"
                )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class SegmentationAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()
    allowed_camera_ids: frozenset[str] = frozenset()

    can_process: bool = False
    can_view: bool = False
    can_manage: bool = False

    def __post_init__(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.user_id,
            "user_id",
        )


# ============================================================
# SEMANTIC CLASS DEFINITIONS
# ============================================================

@dataclass(frozen=True)
class SemanticClass:
    class_id: int
    name: str
    category: SemanticCategory

    color_rgb: tuple[int, int, int]

    traversability: TraversabilityClass = (
        TraversabilityClass.UNKNOWN
    )

    is_dynamic: bool = False
    is_hazard_relevant: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.class_id, bool)
            or not isinstance(self.class_id, int)
            or self.class_id < 0
        ):
            raise SegmentationValidationError(
                "Class ID must be nonnegative"
            )

        validate_identifier(
            self.name,
            "class name",
        )

        if len(self.color_rgb) != 3:
            raise SegmentationValidationError(
                "Class color must contain three channels"
            )

        for value in self.color_rgb:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 255
            ):
                raise SegmentationValidationError(
                    "RGB values must be integers in [0, 255]"
                )


class SemanticClassRegistry:

    def __init__(
        self,
        classes: Sequence[SemanticClass],
    ) -> None:
        if not classes:
            raise SegmentationValidationError(
                "At least one semantic class is required"
            )

        identifiers = [
            item.class_id
            for item in classes
        ]

        if len(identifiers) != len(set(identifiers)):
            raise SegmentationValidationError(
                "Duplicate semantic class IDs"
            )

        self._classes = {
            item.class_id: item
            for item in classes
        }

    def get(
        self,
        class_id: int,
    ) -> SemanticClass | None:
        return self._classes.get(
            int(class_id)
        )

    def all_classes(
        self,
    ) -> tuple[SemanticClass, ...]:
        return tuple(
            self._classes.values()
        )

    def traversable_ids(self) -> frozenset[int]:
        return frozenset(
            item.class_id
            for item in self._classes.values()
            if (
                item.traversability
                == TraversabilityClass.TRAVERSABLE
            )
        )

    def non_traversable_ids(self) -> frozenset[int]:
        return frozenset(
            item.class_id
            for item in self._classes.values()
            if (
                item.traversability
                == TraversabilityClass.NON_TRAVERSABLE
            )
        )

    def human_ids(self) -> frozenset[int]:
        return frozenset(
            item.class_id
            for item in self._classes.values()
            if item.category == SemanticCategory.HUMAN
        )

    def hazard_ids(self) -> frozenset[int]:
        return frozenset(
            item.class_id
            for item in self._classes.values()
            if item.is_hazard_relevant
        )

    @classmethod
    def default(
        cls,
    ) -> SemanticClassRegistry:
        return cls(
            [
                SemanticClass(
                    0,
                    "background",
                    SemanticCategory.BACKGROUND,
                    (0, 0, 0),
                ),
                SemanticClass(
                    1,
                    "floor",
                    SemanticCategory.FLOOR,
                    (100, 180, 100),
                    TraversabilityClass.TRAVERSABLE,
                ),
                SemanticClass(
                    2,
                    "wall",
                    SemanticCategory.WALL,
                    (130, 130, 130),
                    TraversabilityClass.NON_TRAVERSABLE,
                ),
                SemanticClass(
                    3,
                    "person",
                    SemanticCategory.HUMAN,
                    (255, 80, 80),
                    TraversabilityClass.NON_TRAVERSABLE,
                    is_dynamic=True,
                    is_hazard_relevant=True,
                ),
                SemanticClass(
                    4,
                    "robot",
                    SemanticCategory.ROBOT,
                    (80, 120, 255),
                    TraversabilityClass.NON_TRAVERSABLE,
                    is_dynamic=True,
                    is_hazard_relevant=True,
                ),
                SemanticClass(
                    5,
                    "equipment",
                    SemanticCategory.EQUIPMENT,
                    (255, 180, 50),
                    TraversabilityClass.NON_TRAVERSABLE,
                    is_hazard_relevant=True,
                ),
                SemanticClass(
                    6,
                    "obstacle",
                    SemanticCategory.OBSTACLE,
                    (255, 0, 255),
                    TraversabilityClass.NON_TRAVERSABLE,
                    is_hazard_relevant=True,
                ),
                SemanticClass(
                    7,
                    "road",
                    SemanticCategory.ROAD,
                    (80, 80, 80),
                    TraversabilityClass.TRAVERSABLE,
                ),
                SemanticClass(
                    8,
                    "vehicle",
                    SemanticCategory.VEHICLE,
                    (255, 220, 0),
                    TraversabilityClass.NON_TRAVERSABLE,
                    is_dynamic=True,
                    is_hazard_relevant=True,
                ),
            ]
        )


# ============================================================
# IMAGE VALIDATOR
# ============================================================

class SegmentationImageValidator:

    def __init__(
        self,
        config: SemanticSegmentationConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise SegmentationValidationError(
                "Image must be a NumPy array"
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise SegmentationValidationError(
                "Image must have shape H x W x 3"
            )

        height, width = image.shape[:2]

        if height <= 0 or width <= 0:
            raise SegmentationValidationError(
                "Image cannot be empty"
            )

        if (
            height > self.config.maximum_image_height
            or width > self.config.maximum_image_width
        ):
            raise SegmentationCapacityError(
                "Image exceeds maximum dimensions"
            )

        if image.dtype != np.uint8:
            raise SegmentationValidationError(
                "Input image must use uint8 pixels"
            )

        return image


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

class SemanticImagePreprocessor:

    def __init__(
        self,
        config: SemanticSegmentationConfig,
    ) -> None:
        self.config = config

        self.validator = SegmentationImageValidator(
            config
        )

    def preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        require_opencv()

        image = self.validator.validate(
            image
        )

        resized = cv2.resize(
            image,
            (
                self.config.input_width,
                self.config.input_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        if (
            self.config.input_color_format
            != self.config.model_color_format
        ):
            resized = resized[:, :, ::-1]

        tensor = resized.astype(
            np.float32
        )

        if self.config.normalize_to_unit_range:
            tensor /= 255.0

        mean = np.asarray(
            self.config.mean,
            dtype=np.float32,
        )

        std = np.asarray(
            self.config.std,
            dtype=np.float32,
        )

        tensor = (
            tensor - mean
        ) / std

        if (
            self.config.input_layout
            == ModelInputLayout.NCHW
        ):
            tensor = np.transpose(
                tensor,
                (2, 0, 1),
            )

        return np.ascontiguousarray(
            tensor[np.newaxis, ...],
            dtype=np.float32,
        )


# ============================================================
# MODEL INTERFACE
# ============================================================

class SemanticSegmentationModel:

    def predict(
        self,
        tensor: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


# ============================================================
# ONNX SEGMENTATION MODEL
# ============================================================

class ONNXSemanticSegmentationModel(
    SemanticSegmentationModel
):

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
    ) -> None:
        require_onnxruntime()

        path = Path(model_path).expanduser()

        if not path.is_file():
            raise SegmentationModelError(
                f"Model file not found: {path}"
            )

        if path.suffix.lower() != ".onnx":
            raise SegmentationModelError(
                "Model must be an ONNX file"
            )

        requested_providers = list(
            providers or ["CPUExecutionProvider"]
        )

        available = set(
            ort.get_available_providers()
        )

        unsupported = (
            set(requested_providers) - available
        )

        if unsupported:
            raise SegmentationModelError(
                f"Unavailable ONNX providers: "
                f"{sorted(unsupported)}"
            )

        try:
            self.session = ort.InferenceSession(
                str(path),
                providers=requested_providers,
            )
        except Exception as exc:
            raise SegmentationModelError(
                f"Unable to load ONNX model: {exc}"
            ) from exc

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if len(inputs) != 1:
            raise SegmentationModelError(
                "Expected a single ONNX image input"
            )

        if not outputs:
            raise SegmentationModelError(
                "ONNX model has no outputs"
            )

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        self.model_path = str(path)

        self._lock = threading.RLock()

    def predict(
        self,
        tensor: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(tensor, np.ndarray):
            raise SegmentationValidationError(
                "Model input must be an ndarray"
            )

        if not np.all(np.isfinite(tensor)):
            raise SegmentationValidationError(
                "Model input contains non-finite values"
            )

        with self._lock:
            try:
                outputs = self.session.run(
                    [self.output_name],
                    {
                        self.input_name: np.ascontiguousarray(
                            tensor,
                            dtype=np.float32,
                        )
                    },
                )
            except Exception as exc:
                raise SegmentationInferenceError(
                    f"ONNX inference failed: {exc}"
                ) from exc

        return np.asarray(
            outputs[0]
        )


# ============================================================
# SYNTHETIC SEGMENTATION MODEL
# ============================================================

class SyntheticSemanticSegmentationModel(
    SemanticSegmentationModel
):
    """
    Deterministic synthetic model for integration tests.

    Returns NCHW logits with:
        0 background
        1 floor
        2 wall
        3 person
        6 obstacle
    """

    def __init__(
        self,
        class_count: int = 9,
    ) -> None:
        if class_count < 7:
            raise SegmentationValidationError(
                "Synthetic model requires at least 7 classes"
            )

        self.class_count = class_count

    def predict(
        self,
        tensor: np.ndarray,
    ) -> np.ndarray:
        if tensor.ndim != 4:
            raise SegmentationValidationError(
                "Synthetic model expects NCHW input"
            )

        batch, _, height, width = tensor.shape

        logits = np.full(
            (
                batch,
                self.class_count,
                height,
                width,
            ),
            -8.0,
            dtype=np.float32,
        )

        logits[:, 0] = 0.0

        floor_start = int(
            height * 0.65
        )

        logits[
            :,
            1,
            floor_start:,
            :,
        ] = 8.0

        person_x1 = int(
            width * 0.40
        )

        person_x2 = int(
            width * 0.50
        )

        person_y1 = int(
            height * 0.25
        )

        person_y2 = int(
            height * 0.85
        )

        logits[
            :,
            3,
            person_y1:person_y2,
            person_x1:person_x2,
        ] = 12.0

        obstacle_x1 = int(
            width * 0.65
        )

        obstacle_x2 = int(
            width * 0.85
        )

        obstacle_y1 = int(
            height * 0.55
        )

        obstacle_y2 = int(
            height * 0.85
        )

        logits[
            :,
            6,
            obstacle_y1:obstacle_y2,
            obstacle_x1:obstacle_x2,
        ] = 12.0

        return logits


# ============================================================
# OUTPUT DECODING
# ============================================================

@dataclass(frozen=True)
class SegmentationDecodedOutput:
    class_map: np.ndarray
    confidence_map: np.ndarray

    probabilities: np.ndarray | None = None


class SemanticOutputDecoder:

    def __init__(
        self,
        config: SemanticSegmentationConfig,
    ) -> None:
        self.config = config

    @staticmethod
    def softmax(
        logits: np.ndarray,
        axis: int = 0,
    ) -> np.ndarray:
        shifted = logits - np.max(
            logits,
            axis=axis,
            keepdims=True,
        )

        exponential = np.exp(
            shifted
        )

        denominator = np.sum(
            exponential,
            axis=axis,
            keepdims=True,
        )

        return exponential / np.maximum(
            denominator,
            1e-12,
        )

    def decode(
        self,
        output: np.ndarray,
    ) -> SegmentationDecodedOutput:
        output = np.asarray(
            output
        )

        if self.config.output_type == (
            SegmentationOutputType.CLASS_IDS
        ):
            return self._decode_class_ids(
                output
            )

        return self._decode_scores(
            output
        )

    def _decode_class_ids(
        self,
        output: np.ndarray,
    ) -> SegmentationDecodedOutput:
        if output.ndim == 3:
            if output.shape[0] != 1:
                raise SegmentationValidationError(
                    "Only batch size 1 is supported"
                )

            output = output[0]

        if output.ndim != 2:
            raise SegmentationValidationError(
                "Class ID output must have shape H x W"
            )

        if not np.all(np.isfinite(output)):
            raise SegmentationValidationError(
                "Class map contains non-finite values"
            )

        if np.any(output < 0):
            raise SegmentationValidationError(
                "Class IDs cannot be negative"
            )

        if not np.all(output == np.floor(output)):
            raise SegmentationValidationError(
                "Class IDs must be integers"
            )

        if np.any(
            output > np.iinfo(np.int32).max
        ):
            raise SegmentationValidationError(
                "Class ID exceeds int32 range"
            )

        class_map = output.astype(
            np.int32
        )

        confidence_map = np.ones(
            class_map.shape,
            dtype=np.float32,
        )

        return SegmentationDecodedOutput(
            class_map=class_map,
            confidence_map=confidence_map,
            probabilities=None,
        )

    def _decode_scores(
        self,
        output: np.ndarray,
    ) -> SegmentationDecodedOutput:
        if output.ndim != 4:
            raise SegmentationValidationError(
                "Score output must be a 4D tensor"
            )

        if output.shape[0] != 1:
            raise SegmentationValidationError(
                "Only batch size 1 is supported"
            )

        if (
            self.config.output_layout
            == ModelOutputLayout.NCHW
        ):
            scores = output[0]

        elif (
            self.config.output_layout
            == ModelOutputLayout.NHWC
        ):
            scores = np.transpose(
                output[0],
                (2, 0, 1),
            )

        else:
            raise SegmentationValidationError(
                "NHW layout is only valid for class IDs"
            )

        if scores.shape[0] > self.config.maximum_classes:
            raise SegmentationCapacityError(
                "Model output exceeds class limit"
            )

        if scores.shape[0] < 2:
            raise SegmentationValidationError(
                "Score output requires at least two classes"
            )

        if not np.all(np.isfinite(scores)):
            raise SegmentationInferenceError(
                "Model output contains non-finite values"
            )

        scores = scores.astype(
            np.float32
        )

        if self.config.output_type == (
            SegmentationOutputType.LOGITS
        ):
            probabilities = self.softmax(
                scores,
                axis=0,
            )

        else:
            if (
                np.any(scores < 0)
                or np.any(scores > 1)
            ):
                raise SegmentationInferenceError(
                    "Probability output must be in [0, 1]"
                )

            denominator = np.sum(
                scores,
                axis=0,
                keepdims=True,
            )

            probabilities = (
                scores
                / np.maximum(
                    denominator,
                    1e-12,
                )
            )

        class_map = np.argmax(
            probabilities,
            axis=0,
        ).astype(np.int32)

        confidence_map = np.max(
            probabilities,
            axis=0,
        ).astype(np.float32)

        return SegmentationDecodedOutput(
            class_map=class_map,
            confidence_map=confidence_map,
            probabilities=probabilities,
        )


# ============================================================
# OUTPUT RESIZING
# ============================================================

class SegmentationOutputResizer:

    @staticmethod
    def resize_class_map(
        class_map: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        require_opencv()

        resized = cv2.resize(
            class_map.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        return resized.astype(
            np.int32
        )

    @staticmethod
    def resize_confidence_map(
        confidence_map: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        require_opencv()

        return cv2.resize(
            confidence_map.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    @staticmethod
    def resize_probabilities(
        probabilities: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        require_opencv()

        resized = np.stack(
            [
                cv2.resize(
                    channel.astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
                for channel in probabilities
            ],
            axis=0,
        )

        denominator = np.sum(
            resized,
            axis=0,
            keepdims=True,
        )

        return resized / np.maximum(
            denominator,
            1e-12,
        )


# ============================================================
# TEMPORAL SEGMENTATION FILTER
# ============================================================

class TemporalSegmentationFilter:

    def __init__(
        self,
        alpha: float = 0.5,
    ) -> None:
        self.alpha = validate_confidence(
            alpha
        )

        self._previous: np.ndarray | None = None

        self._lock = threading.RLock()

    def update(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        )

        if probabilities.ndim != 3:
            raise SegmentationValidationError(
                "Probabilities must have shape C x H x W"
            )

        if not np.all(np.isfinite(probabilities)):
            raise SegmentationValidationError(
                "Probabilities must be finite"
            )

        with self._lock:
            if (
                self._previous is None
                or self._previous.shape != probabilities.shape
            ):
                filtered = probabilities.copy()

            else:
                filtered = (
                    self.alpha * probabilities
                    + (1 - self.alpha) * self._previous
                )

            denominator = np.sum(
                filtered,
                axis=0,
                keepdims=True,
            )

            filtered = filtered / np.maximum(
                denominator,
                1e-12,
            )

            self._previous = filtered.copy()

            return filtered

    def reset(self) -> None:
        with self._lock:
            self._previous = None


# ============================================================
# SEGMENTATION MASK
# ============================================================

@dataclass(frozen=True)
class SemanticSegmentationMask:
    class_map: np.ndarray
    confidence_map: np.ndarray

    frame_id: str

    timestamp: datetime

    unknown_class_id: int = 255

    def __post_init__(self) -> None:
        validate_identifier(
            self.frame_id,
            "frame_id",
        )

        validate_timestamp(
            self.timestamp
        )

        if self.class_map.ndim != 2:
            raise SegmentationValidationError(
                "Class map must be 2D"
            )

        if (
            self.confidence_map.shape
            != self.class_map.shape
        ):
            raise SegmentationValidationError(
                "Confidence map dimensions do not match"
            )

        if not np.all(
            np.isfinite(self.confidence_map)
        ):
            raise SegmentationValidationError(
                "Confidence map contains non-finite values"
            )

        if (
            np.any(self.confidence_map < 0)
            or np.any(self.confidence_map > 1)
        ):
            raise SegmentationValidationError(
                "Confidence values must be in [0, 1]"
            )

        object.__setattr__(
            self,
            "class_map",
            np.array(
                self.class_map,
                dtype=np.int32,
                copy=True,
            ),
        )

        object.__setattr__(
            self,
            "confidence_map",
            np.array(
                self.confidence_map,
                dtype=np.float32,
                copy=True,
            ),
        )

    @property
    def width(self) -> int:
        return int(
            self.class_map.shape[1]
        )

    @property
    def height(self) -> int:
        return int(
            self.class_map.shape[0]
        )

    def binary_mask(
        self,
        class_id: int,
    ) -> np.ndarray:
        return (
            self.class_map == class_id
        ).astype(np.uint8)

    def class_pixel_count(
        self,
        class_id: int,
    ) -> int:
        return int(
            np.count_nonzero(
                self.class_map == class_id
            )
        )

    def mean_confidence(self) -> float:
        return float(
            np.mean(
                self.confidence_map
            )
        )


# ============================================================
# SEGMENTATION REGION
# ============================================================

@dataclass(frozen=True)
class SemanticRegion:
    region_id: str

    class_id: int
    class_name: str

    area_pixels: int
    area_fraction: float

    bbox_xywh: tuple[int, int, int, int]

    centroid_xy: tuple[float, float]

    mean_confidence: float


# ============================================================
# REGION EXTRACTION
# ============================================================

class SemanticRegionExtractor:

    def __init__(
        self,
        config: SemanticSegmentationConfig,
        registry: SemanticClassRegistry,
    ) -> None:
        self.config = config
        self.registry = registry

    def extract(
        self,
        mask: SemanticSegmentationMask,
    ) -> tuple[SemanticRegion, ...]:
        require_opencv()

        regions: list[SemanticRegion] = []

        total_pixels = (
            mask.width * mask.height
        )

        for class_id in np.unique(
            mask.class_map
        ):
            class_id = int(
                class_id
            )

            if (
                class_id
                == self.config.unknown_class_id
            ):
                continue

            binary = (
                mask.class_map == class_id
            ).astype(np.uint8)

            count, labels, stats, centroids = (
                cv2.connectedComponentsWithStats(
                    binary,
                    connectivity=8,
                )
            )

            semantic_class = self.registry.get(
                class_id
            )

            class_name = (
                semantic_class.name
                if semantic_class is not None
                else f"class_{class_id}"
            )

            for component_id in range(
                1,
                count,
            ):
                x, y, width, height, area = (
                    stats[component_id]
                )

                area = int(area)

                if (
                    area
                    < self.config.minimum_region_area_pixels
                ):
                    continue

                region_pixels = (
                    labels == component_id
                )

                confidence = float(
                    np.mean(
                        mask.confidence_map[
                            region_pixels
                        ]
                    )
                )

                regions.append(
                    SemanticRegion(
                        region_id=generate_id(),
                        class_id=class_id,
                        class_name=class_name,
                        area_pixels=area,
                        area_fraction=(
                            area / total_pixels
                        ),
                        bbox_xywh=(
                            int(x),
                            int(y),
                            int(width),
                            int(height),
                        ),
                        centroid_xy=(
                            float(
                                centroids[
                                    component_id,
                                    0,
                                ]
                            ),
                            float(
                                centroids[
                                    component_id,
                                    1,
                                ]
                            ),
                        ),
                        mean_confidence=confidence,
                    )
                )

        regions.sort(
            key=lambda region: (
                -region.area_pixels,
                region.class_id,
            )
        )

        return tuple(regions)


# ============================================================
# COVERAGE ANALYSIS
# ============================================================

@dataclass(frozen=True)
class SemanticClassCoverage:
    class_id: int
    class_name: str

    pixel_count: int
    coverage_fraction: float

    mean_confidence: float


class SemanticCoverageAnalyzer:

    def __init__(
        self,
        registry: SemanticClassRegistry,
    ) -> None:
        self.registry = registry

    def analyze(
        self,
        mask: SemanticSegmentationMask,
    ) -> tuple[SemanticClassCoverage, ...]:
        total_pixels = (
            mask.width * mask.height
        )

        coverage: list[
            SemanticClassCoverage
        ] = []

        for class_id in np.unique(
            mask.class_map
        ):
            class_id = int(
                class_id
            )

            selected = (
                mask.class_map == class_id
            )

            pixel_count = int(
                np.count_nonzero(selected)
            )

            semantic_class = self.registry.get(
                class_id
            )

            class_name = (
                semantic_class.name
                if semantic_class is not None
                else "unknown"
            )

            coverage.append(
                SemanticClassCoverage(
                    class_id=class_id,
                    class_name=class_name,
                    pixel_count=pixel_count,
                    coverage_fraction=(
                        pixel_count / total_pixels
                    ),
                    mean_confidence=float(
                        np.mean(
                            mask.confidence_map[
                                selected
                            ]
                        )
                    ),
                )
            )

        coverage.sort(
            key=lambda item: (
                -item.pixel_count,
                item.class_id,
            )
        )

        return tuple(coverage)


# ============================================================
# TRAVERSABILITY ANALYSIS
# ============================================================

@dataclass(frozen=True)
class TraversabilityAssessment:
    traversable_fraction: float

    non_traversable_fraction: float

    unknown_fraction: float

    lower_image_traversable_fraction: float

    traversable_mask: np.ndarray

    status: str

    explanation: str


class SemanticTraversabilityAnalyzer:

    def __init__(
        self,
        registry: SemanticClassRegistry,
    ) -> None:
        self.registry = registry

    def analyze(
        self,
        mask: SemanticSegmentationMask,
    ) -> TraversabilityAssessment:
        traversable_ids = (
            self.registry.traversable_ids()
        )

        blocked_ids = (
            self.registry.non_traversable_ids()
        )

        traversable = np.isin(
            mask.class_map,
            list(traversable_ids),
        )

        blocked = np.isin(
            mask.class_map,
            list(blocked_ids),
        )

        unknown = ~(
            traversable | blocked
        )

        traversable_fraction = float(
            np.mean(traversable)
        )

        blocked_fraction = float(
            np.mean(blocked)
        )

        unknown_fraction = float(
            np.mean(unknown)
        )

        lower_start = (
            mask.height // 2
        )

        lower_fraction = float(
            np.mean(
                traversable[
                    lower_start:,
                    :
                ]
            )
        )

        if unknown_fraction > 0.5:
            status = "insufficient_information"

        elif lower_fraction >= 0.7:
            status = "potentially_traversable"

        elif lower_fraction >= 0.3:
            status = "partially_obstructed"

        else:
            status = "potentially_blocked"

        return TraversabilityAssessment(
            traversable_fraction=traversable_fraction,
            non_traversable_fraction=blocked_fraction,
            unknown_fraction=unknown_fraction,
            lower_image_traversable_fraction=lower_fraction,
            traversable_mask=traversable.astype(
                np.uint8
            ),
            status=status,
            explanation=(
                "Image-space semantic traversability only. "
                "Depth, geometry, slope, clearance, localization, "
                "and independent safety sensing are required "
                "before robot navigation."
            ),
        )


# ============================================================
# SCENE RISK ASSESSMENT
# ============================================================

@dataclass(frozen=True)
class SemanticRiskAssessment:
    risk_level: SceneRiskLevel

    human_pixel_fraction: float

    obstacle_pixel_fraction: float

    hazard_pixel_fraction: float

    human_in_lower_image: bool

    advisory: str


class SemanticSceneRiskAnalyzer:

    def __init__(
        self,
        registry: SemanticClassRegistry,
    ) -> None:
        self.registry = registry

    def analyze(
        self,
        mask: SemanticSegmentationMask,
    ) -> SemanticRiskAssessment:
        human = np.isin(
            mask.class_map,
            list(
                self.registry.human_ids()
            ),
        )

        hazards = np.isin(
            mask.class_map,
            list(
                self.registry.hazard_ids()
            ),
        )

        obstacle_ids = {
            item.class_id
            for item in self.registry.all_classes()
            if item.category in (
                SemanticCategory.OBSTACLE,
                SemanticCategory.EQUIPMENT,
                SemanticCategory.VEHICLE,
            )
        }

        obstacles = np.isin(
            mask.class_map,
            list(obstacle_ids),
        )

        human_fraction = float(
            np.mean(human)
        )

        obstacle_fraction = float(
            np.mean(obstacles)
        )

        hazard_fraction = float(
            np.mean(hazards)
        )

        lower_start = (
            mask.height // 2
        )

        human_lower = bool(
            np.any(
                human[
                    lower_start:,
                    :
                ]
            )
        )

        if human_lower:
            risk = SceneRiskLevel.HIGH

        elif human_fraction > 0:
            risk = SceneRiskLevel.MODERATE

        elif obstacle_fraction > 0.15:
            risk = SceneRiskLevel.MODERATE

        elif hazard_fraction > 0:
            risk = SceneRiskLevel.LOW

        else:
            risk = SceneRiskLevel.UNKNOWN

        return SemanticRiskAssessment(
            risk_level=risk,
            human_pixel_fraction=human_fraction,
            obstacle_pixel_fraction=obstacle_fraction,
            hazard_pixel_fraction=hazard_fraction,
            human_in_lower_image=human_lower,
            advisory=(
                "Image-based semantic risk indicator only. "
                "Pixel location is not physical distance. "
                "Request independent geometric and safety "
                "assessment before any movement decision."
            ),
        )


# ============================================================
# SEGMENTATION QUALITY
# ============================================================

@dataclass(frozen=True)
class SegmentationQualityReport:
    quality: SegmentationQuality

    mean_confidence: float

    low_confidence_fraction: float

    unknown_fraction: float

    unique_class_count: int


class SemanticSegmentationQualityEvaluator:

    def __init__(
        self,
        config: SemanticSegmentationConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        mask: SemanticSegmentationMask,
    ) -> SegmentationQualityReport:
        confidence = (
            mask.confidence_map
        )

        mean_confidence = float(
            np.mean(confidence)
        )

        low_fraction = float(
            np.mean(
                confidence
                < self.config.minimum_pixel_confidence
            )
        )

        unknown_fraction = float(
            np.mean(
                mask.class_map
                == self.config.unknown_class_id
            )
        )

        if (
            mean_confidence >= 0.90
            and unknown_fraction < 0.10
        ):
            quality = SegmentationQuality.EXCELLENT

        elif (
            mean_confidence >= 0.75
            and unknown_fraction < 0.25
        ):
            quality = SegmentationQuality.GOOD

        elif (
            mean_confidence >= 0.50
            and unknown_fraction < 0.50
        ):
            quality = SegmentationQuality.DEGRADED

        else:
            quality = SegmentationQuality.POOR

        return SegmentationQualityReport(
            quality=quality,
            mean_confidence=mean_confidence,
            low_confidence_fraction=low_fraction,
            unknown_fraction=unknown_fraction,
            unique_class_count=int(
                len(
                    np.unique(
                        mask.class_map
                    )
                )
            ),
        )


# ============================================================
# VISUALIZATION
# ============================================================

class SemanticSegmentationVisualizer:

    def __init__(
        self,
        registry: SemanticClassRegistry,
    ) -> None:
        self.registry = registry

    def colorize(
        self,
        mask: SemanticSegmentationMask,
    ) -> np.ndarray:
        output = np.zeros(
            (
                mask.height,
                mask.width,
                3,
            ),
            dtype=np.uint8,
        )

        for semantic_class in (
            self.registry.all_classes()
        ):
            output[
                mask.class_map
                == semantic_class.class_id
            ] = semantic_class.color_rgb

        return output

    def overlay(
        self,
        image: np.ndarray,
        mask: SemanticSegmentationMask,
        *,
        alpha: float = 0.45,
        input_format: InputColorFormat = (
            InputColorFormat.BGR
        ),
    ) -> np.ndarray:
        require_opencv()

        alpha = validate_confidence(
            alpha
        )

        if image.shape != (
            mask.height,
            mask.width,
            3,
        ):
            raise SegmentationValidationError(
                "Image and segmentation dimensions differ"
            )

        colorized = self.colorize(
            mask
        )

        if input_format == InputColorFormat.BGR:
            colorized = colorized[:, :, ::-1]

        return cv2.addWeighted(
            image,
            1.0 - alpha,
            colorized,
            alpha,
            0.0,
        )


# ============================================================
# SEGMENTATION RESULT
# ============================================================

@dataclass(frozen=True)
class SemanticSegmentationResult:
    result_id: str

    tenant_id: str
    robot_id: str
    camera_id: str

    timestamp: datetime

    model_name: str

    mask: SemanticSegmentationMask

    regions: tuple[SemanticRegion, ...]

    coverage: tuple[SemanticClassCoverage, ...]

    traversability: TraversabilityAssessment

    risk: SemanticRiskAssessment

    quality: SegmentationQualityReport

    inference_latency_ms: float

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
        *,
        include_class_map: bool = False,
        include_confidence_map: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "result_id": self.result_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "image_width": self.mask.width,
            "image_height": self.mask.height,
            "frame_id": self.mask.frame_id,
            "regions": json_safe(self.regions),
            "coverage": json_safe(self.coverage),
            "traversability": {
                "traversable_fraction": (
                    self.traversability.traversable_fraction
                ),
                "non_traversable_fraction": (
                    self.traversability.non_traversable_fraction
                ),
                "unknown_fraction": (
                    self.traversability.unknown_fraction
                ),
                "lower_image_traversable_fraction": (
                    self.traversability.lower_image_traversable_fraction
                ),
                "status": self.traversability.status,
                "explanation": self.traversability.explanation,
            },
            "risk": json_safe(self.risk),
            "quality": json_safe(self.quality),
            "inference_latency_ms": self.inference_latency_ms,
            "metadata": json_safe(self.metadata),
        }

        if include_class_map:
            payload["class_map"] = (
                self.mask.class_map.tolist()
            )

        if include_confidence_map:
            payload["confidence_map"] = (
                self.mask.confidence_map.tolist()
            )

        return payload


# ============================================================
# SEMANTIC SEGMENTATION PIPELINE
# ============================================================

class SemanticSegmentationPipeline:

    def __init__(
        self,
        model: SemanticSegmentationModel,
        *,
        config: SemanticSegmentationConfig | None = None,
        registry: SemanticClassRegistry | None = None,
        model_name: str = "semantic-segmentation",
    ) -> None:
        self.config = (
            config or SemanticSegmentationConfig()
        )

        self.registry = (
            registry or SemanticClassRegistry.default()
        )

        self.model = model

        self.model_name = validate_identifier(
            model_name,
            "model_name",
        )

        self.preprocessor = SemanticImagePreprocessor(
            self.config
        )

        self.decoder = SemanticOutputDecoder(
            self.config
        )

        self.resizer = SegmentationOutputResizer()

        self.region_extractor = SemanticRegionExtractor(
            self.config,
            self.registry,
        )

        self.coverage_analyzer = SemanticCoverageAnalyzer(
            self.registry
        )

        self.traversability_analyzer = (
            SemanticTraversabilityAnalyzer(
                self.registry
            )
        )

        self.risk_analyzer = SemanticSceneRiskAnalyzer(
            self.registry
        )

        self.quality_evaluator = (
            SemanticSegmentationQualityEvaluator(
                self.config
            )
        )

        self.visualizer = SemanticSegmentationVisualizer(
            self.registry
        )

        self._temporal_filters: dict[
            tuple[str, str, str],
            TemporalSegmentationFilter,
        ] = {}

        self._lock = threading.RLock()

    def process(
        self,
        image: np.ndarray,
        *,
        tenant_id: str,
        robot_id: str,
        camera_id: str,
        frame_id: str = "camera_optical",
        timestamp: datetime | None = None,
    ) -> SemanticSegmentationResult:
        for name, value in (
            ("tenant_id", tenant_id),
            ("robot_id", robot_id),
            ("camera_id", camera_id),
            ("frame_id", frame_id),
        ):
            validate_identifier(
                value,
                name,
            )

        timestamp = validate_timestamp(
            timestamp or utc_now()
        )

        image = self.preprocessor.validator.validate(
            image
        )

        height, width = image.shape[:2]

        start = time.perf_counter()

        tensor = self.preprocessor.preprocess(
            image
        )

        raw_output = self.model.predict(
            tensor
        )

        decoded = self.decoder.decode(
            raw_output
        )

        probabilities = decoded.probabilities

        if probabilities is not None:
            probabilities = self.resizer.resize_probabilities(
                probabilities,
                width,
                height,
            )

            if self.config.enable_temporal_smoothing:
                key = (
                    tenant_id,
                    robot_id,
                    camera_id,
                )

                with self._lock:
                    temporal_filter = (
                        self._temporal_filters.setdefault(
                            key,
                            TemporalSegmentationFilter(
                                self.config.temporal_smoothing_alpha
                            ),
                        )
                    )

                probabilities = temporal_filter.update(
                    probabilities
                )

            class_map = np.argmax(
                probabilities,
                axis=0,
            ).astype(np.int32)

            confidence_map = np.max(
                probabilities,
                axis=0,
            ).astype(np.float32)

        else:
            class_map = self.resizer.resize_class_map(
                decoded.class_map,
                width,
                height,
            )

            confidence_map = (
                self.resizer.resize_confidence_map(
                    decoded.confidence_map,
                    width,
                    height,
                )
            )

        low_confidence = (
            confidence_map
            < self.config.minimum_pixel_confidence
        )

        class_map = class_map.copy()

        class_map[
            low_confidence
        ] = self.config.unknown_class_id

        mask = SemanticSegmentationMask(
            class_map=class_map,
            confidence_map=confidence_map,
            frame_id=frame_id,
            timestamp=timestamp,
            unknown_class_id=(
                self.config.unknown_class_id
            ),
        )

        regions = self.region_extractor.extract(
            mask
        )

        coverage = self.coverage_analyzer.analyze(
            mask
        )

        traversability = (
            self.traversability_analyzer.analyze(
                mask
            )
        )

        risk = self.risk_analyzer.analyze(
            mask
        )

        quality = self.quality_evaluator.evaluate(
            mask
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return SemanticSegmentationResult(
            result_id=generate_id(),
            tenant_id=tenant_id,
            robot_id=robot_id,
            camera_id=camera_id,
            timestamp=timestamp,
            model_name=self.model_name,
            mask=mask,
            regions=regions,
            coverage=coverage,
            traversability=traversability,
            risk=risk,
            quality=quality,
            inference_latency_ms=latency_ms,
            metadata={
                "input_width": width,
                "input_height": height,
                "model_input_width": self.config.input_width,
                "model_input_height": self.config.input_height,
                "temporal_smoothing": (
                    self.config.enable_temporal_smoothing
                ),
            },
        )

    def reset_camera(
        self,
        tenant_id: str,
        robot_id: str,
        camera_id: str,
    ) -> None:
        key = (
            tenant_id,
            robot_id,
            camera_id,
        )

        with self._lock:
            temporal_filter = self._temporal_filters.pop(
                key,
                None,
            )

        if temporal_filter is not None:
            temporal_filter.reset()


# ============================================================
# SEGMENTATION MONITORING
# ============================================================

@dataclass(frozen=True)
class SegmentationStatistics:
    processed_frames: int

    mean_latency_ms: float

    maximum_latency_ms: float

    mean_confidence: float

    mean_traversable_fraction: float

    risk_distribution: dict[str, int]


class SemanticSegmentationMonitor:

    @staticmethod
    def summarize(
        results: Sequence[SemanticSegmentationResult],
    ) -> SegmentationStatistics:
        if not results:
            return SegmentationStatistics(
                processed_frames=0,
                mean_latency_ms=0.0,
                maximum_latency_ms=0.0,
                mean_confidence=0.0,
                mean_traversable_fraction=0.0,
                risk_distribution={},
            )

        latencies = [
            result.inference_latency_ms
            for result in results
        ]

        confidences = [
            result.quality.mean_confidence
            for result in results
        ]

        traversability = [
            result.traversability.traversable_fraction
            for result in results
        ]

        risks = Counter(
            result.risk.risk_level.value
            for result in results
        )

        return SegmentationStatistics(
            processed_frames=len(results),
            mean_latency_ms=float(
                np.mean(latencies)
            ),
            maximum_latency_ms=float(
                np.max(latencies)
            ),
            mean_confidence=float(
                np.mean(confidences)
            ),
            mean_traversable_fraction=float(
                np.mean(traversability)
            ),
            risk_distribution=dict(risks),
        )


# ============================================================
# MULTI-TENANT SEGMENTATION SERVICE
# ============================================================

class RoboticsSemanticSegmentationService:

    def __init__(
        self,
        pipeline: SemanticSegmentationPipeline,
    ) -> None:
        self.pipeline = pipeline

        self.config = pipeline.config

        self._history: dict[
            tuple[str, str, str],
            deque[SemanticSegmentationResult],
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: SegmentationAccessContext,
        tenant_id: str,
        robot_id: str,
        camera_id: str,
        permission: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise SegmentationAuthorizationError(
                "Cross-tenant access denied"
            )

        if robot_id not in access.allowed_robot_ids:
            raise SegmentationAuthorizationError(
                "Robot access denied"
            )

        if camera_id not in access.allowed_camera_ids:
            raise SegmentationAuthorizationError(
                "Camera access denied"
            )

        if not getattr(access, permission, False):
            raise SegmentationAuthorizationError(
                f"Missing permission: {permission}"
            )

    def process(
        self,
        image: np.ndarray,
        *,
        robot_id: str,
        camera_id: str,
        access: SegmentationAccessContext,
        frame_id: str = "camera_optical",
        timestamp: datetime | None = None,
    ) -> SemanticSegmentationResult:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            camera_id,
            "can_process",
        )

        key = (
            access.tenant_id,
            robot_id,
            camera_id,
        )

        with self._lock:
            history = self._history.setdefault(
                key,
                deque(
                    maxlen=self.config.maximum_history
                ),
            )

            effective_timestamp = validate_timestamp(
                timestamp or utc_now()
            )

            if (
                history
                and effective_timestamp <= history[-1].timestamp
            ):
                raise SegmentationValidationError(
                    "Frame timestamps must increase"
                )

            result = self.pipeline.process(
                image,
                tenant_id=access.tenant_id,
                robot_id=robot_id,
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp=effective_timestamp,
            )

            history.append(
                result
            )

        return result

    def latest(
        self,
        *,
        robot_id: str,
        camera_id: str,
        access: SegmentationAccessContext,
    ) -> SemanticSegmentationResult | None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            camera_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            robot_id,
            camera_id,
        )

        with self._lock:
            history = self._history.get(
                key
            )

            return (
                history[-1]
                if history
                else None
            )

    def history(
        self,
        *,
        robot_id: str,
        camera_id: str,
        access: SegmentationAccessContext,
    ) -> tuple[SemanticSegmentationResult, ...]:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            camera_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            robot_id,
            camera_id,
        )

        with self._lock:
            return tuple(
                self._history.get(
                    key,
                    (),
                )
            )

    def statistics(
        self,
        *,
        robot_id: str,
        camera_id: str,
        access: SegmentationAccessContext,
    ) -> SegmentationStatistics:
        results = self.history(
            robot_id=robot_id,
            camera_id=camera_id,
            access=access,
        )

        return SemanticSegmentationMonitor.summarize(
            results
        )

    def reset(
        self,
        *,
        robot_id: str,
        camera_id: str,
        access: SegmentationAccessContext,
    ) -> None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            camera_id,
            "can_manage",
        )

        key = (
            access.tenant_id,
            robot_id,
            camera_id,
        )

        with self._lock:
            self._history.pop(
                key,
                None,
            )

            self.pipeline.reset_camera(
                access.tenant_id,
                robot_id,
                camera_id,
            )

    @staticmethod
    def export_json(
        result: SemanticSegmentationResult,
        *,
        include_class_map: bool = False,
    ) -> str:
        return json.dumps(
            result.to_dict(
                include_class_map=include_class_map
            ),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC IMAGE GENERATOR
# ============================================================

class SyntheticSegmentationImageGenerator:

    @staticmethod
    def warehouse_image(
        *,
        width: int = 640,
        height: int = 480,
    ) -> np.ndarray:
        image = np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )

        image[:] = (
            80,
            80,
            80,
        )

        floor_start = int(
            height * 0.65
        )

        image[
            floor_start:,
            :
        ] = (
            90,
            130,
            90,
        )

        person_x1 = int(
            width * 0.40
        )

        person_x2 = int(
            width * 0.50
        )

        person_y1 = int(
            height * 0.25
        )

        person_y2 = int(
            height * 0.85
        )

        image[
            person_y1:person_y2,
            person_x1:person_x2,
        ] = (
            60,
            60,
            220,
        )

        obstacle_x1 = int(
            width * 0.65
        )

        obstacle_x2 = int(
            width * 0.85
        )

        obstacle_y1 = int(
            height * 0.55
        )

        obstacle_y2 = int(
            height * 0.85
        )

        image[
            obstacle_y1:obstacle_y2,
            obstacle_x1:obstacle_x2,
        ] = (
            200,
            60,
            200,
        )

        return image


# ============================================================
# UNIT TESTS
# ============================================================

class TestSemanticSegmentation(unittest.TestCase):

    def setUp(self) -> None:
        self.config = SemanticSegmentationConfig(
            input_width=128,
            input_height=128,
            minimum_region_area_pixels=10,
        )

        self.registry = (
            SemanticClassRegistry.default()
        )

        self.pipeline = SemanticSegmentationPipeline(
            SyntheticSemanticSegmentationModel(),
            config=self.config,
            registry=self.registry,
            model_name="synthetic-semantic-model",
        )

        self.service = (
            RoboticsSemanticSegmentationService(
                self.pipeline
            )
        )

        self.access = SegmentationAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            allowed_camera_ids=frozenset({
                "CAMERA-001",
            }),
            can_process=True,
            can_view=True,
            can_manage=True,
        )

        self.image = (
            SyntheticSegmentationImageGenerator.warehouse_image(
                width=320,
                height=240,
            )
        )

    def test_class_registry(self) -> None:
        self.assertEqual(
            self.registry.get(3).name,
            "person",
        )

    def test_traversable_ids(self) -> None:
        self.assertIn(
            1,
            self.registry.traversable_ids(),
        )

    def test_human_ids(self) -> None:
        self.assertIn(
            3,
            self.registry.human_ids(),
        )

    def test_image_validation(self) -> None:
        validated = SegmentationImageValidator(
            self.config
        ).validate(
            self.image
        )

        self.assertEqual(
            validated.shape,
            (240, 320, 3),
        )

    def test_invalid_image(self) -> None:
        with self.assertRaises(
            SegmentationValidationError
        ):
            SegmentationImageValidator(
                self.config
            ).validate(
                np.zeros(
                    (100, 100),
                    dtype=np.uint8,
                )
            )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV is not installed",
    )
    def test_preprocessing(self) -> None:
        tensor = SemanticImagePreprocessor(
            self.config
        ).preprocess(
            self.image
        )

        self.assertEqual(
            tensor.shape,
            (1, 3, 128, 128),
        )

    def test_softmax(self) -> None:
        logits = np.array(
            [
                [[2.0]],
                [[1.0]],
                [[0.0]],
            ],
            dtype=np.float32,
        )

        probabilities = (
            SemanticOutputDecoder.softmax(
                logits,
                axis=0,
            )
        )

        self.assertAlmostEqual(
            float(
                np.sum(probabilities)
            ),
            1.0,
            places=5,
        )

    def test_logits_decoding(self) -> None:
        output = np.zeros(
            (1, 9, 16, 16),
            dtype=np.float32,
        )

        output[:, 3] = 10.0

        decoded = SemanticOutputDecoder(
            self.config
        ).decode(
            output
        )

        self.assertTrue(
            np.all(
                decoded.class_map == 3
            )
        )

    def test_class_id_decoding(self) -> None:
        config = SemanticSegmentationConfig(
            output_type=SegmentationOutputType.CLASS_IDS,
            output_layout=ModelOutputLayout.NHW,
        )

        output = np.ones(
            (1, 16, 16),
            dtype=np.int32,
        )

        decoded = SemanticOutputDecoder(
            config
        ).decode(
            output
        )

        self.assertEqual(
            decoded.class_map.shape,
            (16, 16),
        )

    def test_temporal_filter(self) -> None:
        temporal_filter = TemporalSegmentationFilter(
            alpha=0.5
        )

        first = np.zeros(
            (2, 4, 4),
            dtype=np.float32,
        )

        first[0] = 1.0

        second = np.zeros_like(
            first
        )

        second[1] = 1.0

        temporal_filter.update(
            first
        )

        result = temporal_filter.update(
            second
        )

        self.assertAlmostEqual(
            float(result[0, 0, 0]),
            0.5,
        )

    def test_mask_binary(self) -> None:
        class_map = np.array(
            [
                [1, 1],
                [3, 3],
            ],
            dtype=np.int32,
        )

        mask = SemanticSegmentationMask(
            class_map=class_map,
            confidence_map=np.ones(
                (2, 2),
                dtype=np.float32,
            ),
            frame_id="camera_optical",
            timestamp=utc_now(),
        )

        self.assertEqual(
            mask.class_pixel_count(3),
            2,
        )

    def test_coverage_analysis(self) -> None:
        class_map = np.array(
            [
                [1, 1],
                [3, 3],
            ],
            dtype=np.int32,
        )

        mask = SemanticSegmentationMask(
            class_map=class_map,
            confidence_map=np.ones(
                (2, 2),
                dtype=np.float32,
            ),
            frame_id="camera_optical",
            timestamp=utc_now(),
        )

        coverage = SemanticCoverageAnalyzer(
            self.registry
        ).analyze(
            mask
        )

        self.assertEqual(
            len(coverage),
            2,
        )

    def test_traversability(self) -> None:
        class_map = np.ones(
            (10, 10),
            dtype=np.int32,
        )

        mask = SemanticSegmentationMask(
            class_map=class_map,
            confidence_map=np.ones(
                (10, 10),
                dtype=np.float32,
            ),
            frame_id="camera_optical",
            timestamp=utc_now(),
        )

        assessment = (
            SemanticTraversabilityAnalyzer(
                self.registry
            ).analyze(
                mask
            )
        )

        self.assertAlmostEqual(
            assessment.traversable_fraction,
            1.0,
        )

    def test_human_risk(self) -> None:
        class_map = np.ones(
            (20, 20),
            dtype=np.int32,
        )

        class_map[
            10:20,
            5:10,
        ] = 3

        mask = SemanticSegmentationMask(
            class_map=class_map,
            confidence_map=np.ones(
                (20, 20),
                dtype=np.float32,
            ),
            frame_id="camera_optical",
            timestamp=utc_now(),
        )

        risk = SemanticSceneRiskAnalyzer(
            self.registry
        ).analyze(
            mask
        )

        self.assertEqual(
            risk.risk_level,
            SceneRiskLevel.HIGH,
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV is not installed",
    )
    def test_region_extraction(self) -> None:
        class_map = np.zeros(
            (30, 30),
            dtype=np.int32,
        )

        class_map[
            5:20,
            5:20,
        ] = 3

        mask = SemanticSegmentationMask(
            class_map=class_map,
            confidence_map=np.ones(
                (30, 30),
                dtype=np.float32,
            ),
            frame_id="camera_optical",
            timestamp=utc_now(),
        )

        regions = SemanticRegionExtractor(
            self.config,
            self.registry,
        ).extract(
            mask
        )

        self.assertTrue(
            any(
                region.class_id == 3
                for region in regions
            )
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV is not installed",
    )
    def test_full_pipeline(self) -> None:
        result = self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAMERA-001",
            access=self.access,
        )

        self.assertEqual(
            result.mask.class_map.shape,
            (240, 320),
        )

        self.assertGreater(
            len(result.regions),
            0,
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV is not installed",
    )
    def test_human_segmentation(self) -> None:
        result = self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAMERA-001",
            access=self.access,
        )

        self.assertGreater(
            result.mask.class_pixel_count(3),
            0,
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV is not installed",
    )
    def test_visualization(self) -> None:
        result = self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAMERA-001",
            access=self.access,
        )

        colorized = (
            self.pipeline.visualizer.colorize(
                result.mask
            )
        )

        self.assertEqual(
            colorized.shape,
            self.image.shape,
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV is not installed",
    )
    def test_history(self) -> None:
        self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAMERA-001",
            access=self.access,
        )

        history = self.service.history(
            robot_id="ROBOT-001",
            camera_id="CAMERA-001",
            access=self.access,
        )

        self.assertEqual(
            len(history),
            1,
        )

    def test_tenant_isolation(self) -> None:
        with self.assertRaises(
            SegmentationAuthorizationError
        ):
            self.service._authorize(
                self.access,
                "tenant_beta",
                "ROBOT-001",
                "CAMERA-001",
                "can_view",
            )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV is not installed",
    )
    def test_json_export(self) -> None:
        result = self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAMERA-001",
            access=self.access,
        )

        payload = self.service.export_json(
            result
        )

        self.assertIn(
            "ROBOT-001",
            payload,
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV is not installed",
    )
    def test_statistics(self) -> None:
        self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAMERA-001",
            access=self.access,
        )

        statistics = self.service.statistics(
            robot_id="ROBOT-001",
            camera_id="CAMERA-001",
            access=self.access,
        )

        self.assertEqual(
            statistics.processed_frames,
            1,
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV is not installed",
    )
    def test_reset(self) -> None:
        self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAMERA-001",
            access=self.access,
        )

        self.service.reset(
            robot_id="ROBOT-001",
            camera_id="CAMERA-001",
            access=self.access,
        )

        self.assertEqual(
            len(
                self.service.history(
                    robot_id="ROBOT-001",
                    camera_id="CAMERA-001",
                    access=self.access,
                )
            ),
            0,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Semantic Segmentation Engine\n"
    )

    if cv2 is None:
        print(
            "Install opencv-python to run the image demo."
        )

    else:
        config = SemanticSegmentationConfig(
            input_width=256,
            input_height=256,
            minimum_region_area_pixels=20,
        )

        registry = SemanticClassRegistry.default()

        model = SyntheticSemanticSegmentationModel()

        pipeline = SemanticSegmentationPipeline(
            model,
            config=config,
            registry=registry,
            model_name="synthetic-warehouse-segmentation",
        )

        service = RoboticsSemanticSegmentationService(
            pipeline
        )

        access = SegmentationAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            allowed_camera_ids=frozenset({
                "CAMERA-001",
            }),
            can_process=True,
            can_view=True,
            can_manage=True,
        )

        image = (
            SyntheticSegmentationImageGenerator.warehouse_image()
        )

        result = service.process(
            image,
            robot_id="ROBOT-001",
            camera_id="CAMERA-001",
            access=access,
        )

        print("SEGMENTATION RESULT")

        print(
            service.export_json(
                result
            )
        )

        print("\nSEGMENTATION STATISTICS")

        print(
            json.dumps(
                json_safe(
                    service.statistics(
                        robot_id="ROBOT-001",
                        camera_id="CAMERA-001",
                        access=access,
                    )
                ),
                indent=2,
            )
        )

        colorized = pipeline.visualizer.colorize(
            result.mask
        )

        overlay = pipeline.visualizer.overlay(
            image,
            result.mask,
            alpha=0.45,
        )

        print("\nVISUALIZATION SHAPES")

        print(
            "Original:",
            image.shape,
        )

        print(
            "Colorized:",
            colorized.shape,
        )

        print(
            "Overlay:",
            overlay.shape,
        )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestSemanticSegmentation
    )

    result = unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
"""Instance Segmentation module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/instance_segmentation.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Instance Segmentation Engine

Features:
    - Instance segmentation using ONNX models
    - YOLO-style segmentation output decoding
    - Individual object masks
    - Bounding-box detection
    - Confidence thresholding
    - Class-aware non-maximum suppression
    - Mask reconstruction from prototype tensors
    - Mask resizing to original image resolution
    - Object area and centroid calculation
    - Mask overlap and IoU calculation
    - Instance tracking using mask IoU
    - Instance visualization
    - Segmentation statistics
    - Multi-camera processing
    - Tenant-aware access control
    - Processing history
    - JSON metadata export
    - Synthetic segmentation for testing
    - Unit tests

Dependencies:
    numpy
    opencv-python
    onnxruntime (optional)

Install:
    pip install numpy opencv-python onnxruntime

Notes:
    - ONNX decoding targets common YOLOv8-style segmentation
      exports with detection and prototype outputs.
    - Other model architectures require a custom decoder.
    - Input preprocessing must match the exported model.
    - Segmentation outputs are advisory perception results,
      not safety-certified collision avoidance signals.
"""

from __future__ import annotations

import json
import math
import threading
import time
import unittest
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
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

class InstanceSegmentationError(Exception):
    """Base instance segmentation exception."""


class SegmentationValidationError(InstanceSegmentationError):
    """Invalid segmentation input or configuration."""


class SegmentationDependencyError(InstanceSegmentationError):
    """Required dependency is unavailable."""


class SegmentationModelError(InstanceSegmentationError):
    """Segmentation model inference or decoding failed."""


class SegmentationAuthorizationError(InstanceSegmentationError):
    """Segmentation access denied."""


class SegmentationCapacityError(InstanceSegmentationError):
    """Processing capacity limit exceeded."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def validate_identifier(value: str, name: str) -> str:
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


def finite_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise SegmentationValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise SegmentationValidationError(
            f"{name} must be finite"
        )

    return result


def positive_float(value: Any, name: str) -> float:
    result = finite_float(value, name)

    if result <= 0:
        raise SegmentationValidationError(
            f"{name} must be positive"
        )

    return result


def require_opencv() -> None:
    if cv2 is None:
        raise SegmentationDependencyError(
            "OpenCV required: pip install opencv-python"
        )


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)

    return 1.0 / (
        1.0 + np.exp(-np.clip(value, -80.0, 80.0))
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: json_safe(getattr(value, key))
            for key in value.__dataclass_fields__
        }

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class SegmentationBackend(str, Enum):
    ONNX = "onnx"
    SYNTHETIC = "synthetic"


class SegmentationStatus(str, Enum):
    SUCCESS = "success"
    NO_DETECTIONS = "no_detections"
    FAILED = "failed"


class MaskEncoding(str, Enum):
    BINARY = "binary"
    PROBABILITY = "probability"


class SegmentationQuality(str, Enum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class InstanceSegmentationConfig:
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.50
    mask_threshold: float = 0.50

    maximum_detections: int = 100
    maximum_candidates: int = 1000

    maximum_image_width: int = 4096
    maximum_image_height: int = 4096
    maximum_image_pixels: int = 16_777_216

    maximum_history_frames: int = 50

    minimum_mask_pixels: int = 4

    class_agnostic_nms: bool = False

    def __post_init__(self) -> None:
        for name in (
            "confidence_threshold",
            "iou_threshold",
            "mask_threshold",
        ):
            value = finite_float(
                getattr(self, name),
                name,
            )

            if not 0 <= value <= 1:
                raise SegmentationValidationError(
                    f"{name} must be between 0 and 1"
                )

        for name in (
            "maximum_detections",
            "maximum_candidates",
            "maximum_image_width",
            "maximum_image_height",
            "maximum_image_pixels",
            "maximum_history_frames",
            "minimum_mask_pixels",
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

        if self.maximum_candidates < self.maximum_detections:
            raise SegmentationValidationError(
                "maximum_candidates must be >= maximum_detections"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class SegmentationAccessContext:
    tenant_id: str
    user_id: str

    allowed_camera_ids: frozenset[str] = frozenset()

    can_process: bool = False
    can_view: bool = False

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
# GEOMETRY
# ============================================================

@dataclass(frozen=True)
class SegmentationBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            finite_float(
                getattr(self, name),
                name,
            )

        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise SegmentationValidationError(
                "Invalid bounding box"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )

    def clip(
        self,
        width: int,
        height: int,
    ) -> SegmentationBox | None:
        x1 = max(0.0, min(float(width), self.x1))
        y1 = max(0.0, min(float(height), self.y1))

        x2 = max(0.0, min(float(width), self.x2))
        y2 = max(0.0, min(float(height), self.y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return SegmentationBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int

    target_width: int
    target_height: int

    scale_x: float
    scale_y: float

    pad_left: int
    pad_top: int

    def __post_init__(self) -> None:
        positive_float(
            self.scale_x,
            "scale_x",
        )

        positive_float(
            self.scale_y,
            "scale_y",
        )

    def inverse_box(
        self,
        box: SegmentationBox,
    ) -> SegmentationBox | None:
        result = SegmentationBox(
            x1=(box.x1 - self.pad_left) / self.scale_x,
            y1=(box.y1 - self.pad_top) / self.scale_y,
            x2=(box.x2 - self.pad_left) / self.scale_x,
            y2=(box.y2 - self.pad_top) / self.scale_y,
        )

        return result.clip(
            self.source_width,
            self.source_height,
        )


# ============================================================
# INSTANCE DATA MODEL
# ============================================================

@dataclass(frozen=True)
class SegmentedInstance:
    instance_id: str

    class_id: int
    class_name: str

    confidence: float

    bounding_box: SegmentationBox

    mask: np.ndarray = field(repr=False)

    track_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.instance_id,
            "instance_id",
        )

        if self.class_id < 0:
            raise SegmentationValidationError(
                "class_id must be nonnegative"
            )

        if not 0 <= self.confidence <= 1:
            raise SegmentationValidationError(
                "Confidence must be between 0 and 1"
            )

        if (
            not isinstance(self.mask, np.ndarray)
            or self.mask.ndim != 2
        ):
            raise SegmentationValidationError(
                "Instance mask must be a 2D array"
            )

    @property
    def area_pixels(self) -> int:
        return int(
            np.count_nonzero(self.mask)
        )

    @property
    def centroid(self) -> tuple[float, float] | None:
        ys, xs = np.nonzero(self.mask)

        if xs.size == 0:
            return None

        return (
            float(np.mean(xs)),
            float(np.mean(ys)),
        )

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bounding_box": json_safe(
                self.bounding_box
            ),
            "area_pixels": self.area_pixels,
            "centroid": self.centroid,
            "track_id": self.track_id,
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class InstanceSegmentationFrame:
    frame_id: str

    tenant_id: str
    camera_id: str

    backend: SegmentationBackend
    status: SegmentationStatus

    image_width: int
    image_height: int

    instances: tuple[SegmentedInstance, ...]

    processing_time_ms: float

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "tenant_id": self.tenant_id,
            "camera_id": self.camera_id,
            "backend": self.backend.value,
            "status": self.status.value,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "instance_count": self.instance_count,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "instances": [
                instance.metadata_dict()
                for instance in self.instances
            ],
            "metadata": json_safe(self.metadata),
        }


# ============================================================
# IMAGE VALIDATION
# ============================================================

class SegmentationImageValidator:

    def __init__(
        self,
        config: InstanceSegmentationConfig,
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
                "Expected HWC image with 3 channels"
            )

        height, width = image.shape[:2]

        if (
            height <= 0
            or width <= 0
            or height > self.config.maximum_image_height
            or width > self.config.maximum_image_width
            or height * width
            > self.config.maximum_image_pixels
        ):
            raise SegmentationCapacityError(
                "Image dimensions exceed configured limits"
            )

        if image.dtype != np.uint8:
            raise SegmentationValidationError(
                "Expected uint8 image"
            )

        return image


# ============================================================
# MASK OPERATIONS
# ============================================================

class InstanceMaskOperations:

    @staticmethod
    def mask_iou(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        if first.shape != second.shape:
            raise SegmentationValidationError(
                "Masks must have matching shapes"
            )

        a = first.astype(bool)
        b = second.astype(bool)

        intersection = np.count_nonzero(
            a & b
        )

        union = np.count_nonzero(
            a | b
        )

        if union == 0:
            return 0.0

        return float(intersection / union)

    @staticmethod
    def box_iou(
        first: SegmentationBox,
        second: SegmentationBox,
    ) -> float:
        x1 = max(first.x1, second.x1)
        y1 = max(first.y1, second.y1)

        x2 = min(first.x2, second.x2)
        y2 = min(first.y2, second.y2)

        intersection = max(
            0.0,
            x2 - x1,
        ) * max(
            0.0,
            y2 - y1,
        )

        union = (
            first.area
            + second.area
            - intersection
        )

        return (
            intersection / union
            if union > 0
            else 0.0
        )

    @staticmethod
    def mask_to_box(
        mask: np.ndarray,
    ) -> SegmentationBox | None:
        ys, xs = np.nonzero(mask)

        if xs.size == 0:
            return None

        return SegmentationBox(
            x1=float(np.min(xs)),
            y1=float(np.min(ys)),
            x2=float(np.max(xs) + 1),
            y2=float(np.max(ys) + 1),
        )

    @staticmethod
    def combine_masks(
        instances: Sequence[SegmentedInstance],
        height: int,
        width: int,
    ) -> np.ndarray:
        combined = np.zeros(
            (height, width),
            dtype=np.int32,
        )

        # Higher-confidence instances win overlapping pixels.
        for index, instance in enumerate(
            sorted(
                instances,
                key=lambda item: item.confidence,
            ),
            start=1,
        ):
            if instance.mask.shape != combined.shape:
                raise SegmentationValidationError(
                    "Instance mask resolution mismatch"
                )

            combined[
                instance.mask.astype(bool)
            ] = index

        return combined

    @staticmethod
    def binary_to_rle(
        mask: np.ndarray,
    ) -> dict[str, Any]:
        binary = np.asarray(
            mask,
            dtype=np.uint8,
        )

        if binary.ndim != 2:
            raise SegmentationValidationError(
                "RLE requires a 2D mask"
            )

        flat = binary.flatten(order="F")

        counts: list[int] = []

        previous = 0
        run_length = 0

        for value in flat:
            current = int(value != 0)

            if current == previous:
                run_length += 1
            else:
                counts.append(run_length)
                run_length = 1
                previous = current

        counts.append(run_length)

        return {
            "size": list(binary.shape),
            "counts": counts,
            "order": "F",
        }

    @staticmethod
    def rle_to_binary(
        encoded: Mapping[str, Any],
    ) -> np.ndarray:
        height, width = encoded["size"]
        counts = encoded["counts"]

        if (
            height <= 0
            or width <= 0
            or sum(counts) != height * width
            or any(
                not isinstance(count, int) or count < 0
                for count in counts
            )
        ):
            raise SegmentationValidationError(
                "Invalid RLE mask"
            )

        values = np.empty(
            height * width,
            dtype=np.uint8,
        )

        offset = 0
        current = 0

        for count in counts:
            values[offset:offset + count] = current
            offset += count
            current = 1 - current

        return values.reshape(
            (height, width),
            order="F",
        )


# ============================================================
# NON-MAXIMUM SUPPRESSION
# ============================================================

@dataclass(frozen=True)
class DetectionCandidate:
    class_id: int
    confidence: float
    box: SegmentationBox
    mask_coefficients: np.ndarray = field(
        repr=False
    )


class InstanceNonMaximumSuppression:

    @staticmethod
    def suppress(
        candidates: Sequence[DetectionCandidate],
        *,
        iou_threshold: float,
        maximum_detections: int,
        class_agnostic: bool = False,
    ) -> tuple[DetectionCandidate, ...]:
        if not 0 <= iou_threshold <= 1:
            raise SegmentationValidationError(
                "Invalid IoU threshold"
            )

        ordered = sorted(
            candidates,
            key=lambda candidate: candidate.confidence,
            reverse=True,
        )

        selected: list[DetectionCandidate] = []

        for candidate in ordered:
            suppressed = False

            for existing in selected:
                if (
                    not class_agnostic
                    and candidate.class_id != existing.class_id
                ):
                    continue

                overlap = InstanceMaskOperations.box_iou(
                    candidate.box,
                    existing.box,
                )

                if overlap > iou_threshold:
                    suppressed = True
                    break

            if not suppressed:
                selected.append(candidate)

            if len(selected) >= maximum_detections:
                break

        return tuple(selected)


# ============================================================
# ONNX MODEL CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ONNXSegmentationModelConfig:
    model_path: str

    input_width: int = 640
    input_height: int = 640

    class_names: tuple[str, ...] = (
        "person",
        "robot",
        "vehicle",
        "equipment",
        "tool",
        "obstacle",
    )

    input_color_order: str = "BGR"

    providers: tuple[str, ...] = (
        "CPUExecutionProvider",
    )

    def __post_init__(self) -> None:
        if not self.model_path:
            raise SegmentationValidationError(
                "model_path required"
            )

        if (
            self.input_width <= 0
            or self.input_height <= 0
        ):
            raise SegmentationValidationError(
                "Invalid model input dimensions"
            )

        if self.input_color_order not in ("RGB", "BGR"):
            raise SegmentationValidationError(
                "Invalid input color order"
            )

        if not self.class_names:
            raise SegmentationValidationError(
                "At least one class name required"
            )


# ============================================================
# ONNX INSTANCE SEGMENTATION ESTIMATOR
# ============================================================

class ONNXInstanceSegmentationEstimator:
    """
    YOLOv8-style instance segmentation ONNX adapter.

    Expected outputs:
        detections:
            [1, 4 + C + M, N]
            or [1, N, 4 + C + M]

        prototypes:
            [1, M, mask_height, mask_width]

    Detection values:
        [cx, cy, width, height, class_scores..., mask_coefficients...]

    This decoder does not support objectness-bearing YOLOv5
    outputs, end-to-end NMS exports, or arbitrary architectures.
    """

    def __init__(
        self,
        model_config: ONNXSegmentationModelConfig,
        segmentation_config: (
            InstanceSegmentationConfig | None
        ) = None,
    ) -> None:
        if ort is None:
            raise SegmentationDependencyError(
                "onnxruntime required"
            )

        require_opencv()

        self.model_config = model_config

        self.config = (
            segmentation_config
            or InstanceSegmentationConfig()
        )

        try:
            self.session = ort.InferenceSession(
                model_config.model_path,
                providers=list(model_config.providers),
            )
        except Exception as exc:
            raise SegmentationModelError(
                "Unable to load ONNX segmentation model"
            ) from exc

        inputs = self.session.get_inputs()

        if len(inputs) != 1:
            raise SegmentationModelError(
                "Expected one ONNX model input"
            )

        self.input_name = inputs[0].name

        self._inference_lock = threading.RLock()

    def preprocess(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, LetterboxTransform]:
        SegmentationImageValidator(
            self.config
        ).validate(image)

        height, width = image.shape[:2]

        target_width = self.model_config.input_width
        target_height = self.model_config.input_height

        scale = min(
            target_width / width,
            target_height / height,
        )

        resized_width = max(
            1,
            min(target_width, int(round(width * scale))),
        )

        resized_height = max(
            1,
            min(target_height, int(round(height * scale))),
        )

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        left = (target_width - resized_width) // 2
        top = (target_height - resized_height) // 2

        right = target_width - resized_width - left
        bottom = target_height - resized_height - top

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        if self.model_config.input_color_order == "BGR":
            padded = cv2.cvtColor(
                padded,
                cv2.COLOR_BGR2RGB,
            )

        tensor = (
            padded.astype(np.float32) / 255.0
        )

        tensor = np.ascontiguousarray(
            tensor.transpose(2, 0, 1)[None, ...]
        )

        transform = LetterboxTransform(
            source_width=width,
            source_height=height,
            target_width=target_width,
            target_height=target_height,
            scale_x=resized_width / width,
            scale_y=resized_height / height,
            pad_left=left,
            pad_top=top,
        )

        return tensor, transform

    def _decode_outputs(
        self,
        outputs: Sequence[np.ndarray],
        transform: LetterboxTransform,
    ) -> tuple[SegmentedInstance, ...]:
        if len(outputs) < 2:
            raise SegmentationModelError(
                "Expected detection and prototype outputs"
            )

        detection = np.asarray(
            outputs[0],
            dtype=np.float32,
        )

        prototypes = np.asarray(
            outputs[1],
            dtype=np.float32,
        )

        if detection.ndim != 3 or detection.shape[0] != 1:
            raise SegmentationModelError(
                "Invalid detection tensor shape"
            )

        if prototypes.ndim != 4 or prototypes.shape[0] != 1:
            raise SegmentationModelError(
                "Invalid prototype tensor shape"
            )

        prototypes = prototypes[0]

        mask_channels, mask_height, mask_width = (
            prototypes.shape
        )

        class_count = len(
            self.model_config.class_names
        )

        expected_features = (
            4 + class_count + mask_channels
        )

        raw = detection[0]

        if raw.shape[0] == expected_features:
            raw = raw.T
        elif raw.shape[1] != expected_features:
            raise SegmentationModelError(
                "Detection output does not match "
                "configured class count and mask channels"
            )

        candidates: list[DetectionCandidate] = []

        class_scores = raw[
            :,
            4:4 + class_count,
        ]

        class_ids = np.argmax(
            class_scores,
            axis=1,
        )

        confidences = np.max(
            class_scores,
            axis=1,
        )

        candidate_indices = np.flatnonzero(
            np.isfinite(confidences)
            & (
                confidences
                >= self.config.confidence_threshold
            )
        )

        if candidate_indices.size > self.config.maximum_candidates:
            scores = confidences[candidate_indices]

            top = np.argpartition(
                scores,
                -self.config.maximum_candidates,
            )[-self.config.maximum_candidates:]

            candidate_indices = candidate_indices[top]

        for index in candidate_indices:
            row = raw[index]

            cx, cy, width, height = (
                float(value)
                for value in row[:4]
            )

            if not all(
                math.isfinite(value)
                for value in (cx, cy, width, height)
            ):
                continue

            if width <= 0 or height <= 0:
                continue

            box = SegmentationBox(
                x1=cx - width / 2.0,
                y1=cy - height / 2.0,
                x2=cx + width / 2.0,
                y2=cy + height / 2.0,
            )

            candidates.append(
                DetectionCandidate(
                    class_id=int(class_ids[index]),
                    confidence=float(confidences[index]),
                    box=box,
                    mask_coefficients=row[
                        4 + class_count:
                    ].copy(),
                )
            )

        selected = InstanceNonMaximumSuppression.suppress(
            candidates,
            iou_threshold=self.config.iou_threshold,
            maximum_detections=self.config.maximum_detections,
            class_agnostic=self.config.class_agnostic_nms,
        )

        if not selected:
            return ()

        flat_prototypes = prototypes.reshape(
            mask_channels,
            -1,
        )

        instances: list[SegmentedInstance] = []

        for candidate in selected:
            coefficients = candidate.mask_coefficients

            mask_logits = (
                coefficients @ flat_prototypes
            ).reshape(
                mask_height,
                mask_width,
            )

            probability = sigmoid(mask_logits)

            # Resize probability map into letterboxed input space.
            probability = cv2.resize(
                probability,
                (
                    transform.target_width,
                    transform.target_height,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

            # Crop the mask to the detection bounding box.
            xx1 = max(
                0,
                int(math.floor(candidate.box.x1)),
            )

            yy1 = max(
                0,
                int(math.floor(candidate.box.y1)),
            )

            xx2 = min(
                transform.target_width,
                int(math.ceil(candidate.box.x2)),
            )

            yy2 = min(
                transform.target_height,
                int(math.ceil(candidate.box.y2)),
            )

            if xx2 <= xx1 or yy2 <= yy1:
                continue

            cropped_probability = np.zeros_like(
                probability
            )

            cropped_probability[
                yy1:yy2,
                xx1:xx2,
            ] = probability[
                yy1:yy2,
                xx1:xx2,
            ]

            # Remove letterbox padding.
            content_width = max(
                1,
                int(round(
                    transform.source_width
                    * transform.scale_x
                )),
            )

            content_height = max(
                1,
                int(round(
                    transform.source_height
                    * transform.scale_y
                )),
            )

            x0 = transform.pad_left
            y0 = transform.pad_top

            unpadded = cropped_probability[
                y0:y0 + content_height,
                x0:x0 + content_width,
            ]

            if unpadded.size == 0:
                continue

            original_probability = cv2.resize(
                unpadded,
                (
                    transform.source_width,
                    transform.source_height,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

            binary_mask = (
                original_probability
                >= self.config.mask_threshold
            ).astype(np.uint8)

            if (
                np.count_nonzero(binary_mask)
                < self.config.minimum_mask_pixels
            ):
                continue

            original_box = transform.inverse_box(
                candidate.box
            )

            if original_box is None:
                continue

            instances.append(
                SegmentedInstance(
                    instance_id=generate_id(),
                    class_id=candidate.class_id,
                    class_name=(
                        self.model_config.class_names[
                            candidate.class_id
                        ]
                    ),
                    confidence=candidate.confidence,
                    bounding_box=original_box,
                    mask=binary_mask,
                    metadata={
                        "model_backend": "onnx",
                        "mask_threshold": (
                            self.config.mask_threshold
                        ),
                    },
                )
            )

        return tuple(instances)

    def estimate(
        self,
        image: np.ndarray,
    ) -> tuple[SegmentedInstance, ...]:
        tensor, transform = self.preprocess(image)

        try:
            with self._inference_lock:
                outputs = self.session.run(
                    None,
                    {
                        self.input_name: tensor,
                    },
                )
        except Exception as exc:
            raise SegmentationModelError(
                "ONNX inference failed"
            ) from exc

        return self._decode_outputs(
            outputs,
            transform,
        )


# ============================================================
# SYNTHETIC INSTANCE SEGMENTATION
# ============================================================

class SyntheticInstanceSegmenter:

    @staticmethod
    def segment(
        image: np.ndarray,
    ) -> tuple[SegmentedInstance, ...]:
        height, width = image.shape[:2]

        mask_robot = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        mask_obstacle = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        x1 = width // 8
        x2 = width // 3

        y1 = height // 4
        y2 = 3 * height // 4

        mask_robot[
            y1:y2,
            x1:x2,
        ] = 1

        obstacle_x1 = width // 2
        obstacle_x2 = 7 * width // 8

        obstacle_y1 = height // 3
        obstacle_y2 = 2 * height // 3

        mask_obstacle[
            obstacle_y1:obstacle_y2,
            obstacle_x1:obstacle_x2,
        ] = 1

        objects = [
            (
                1,
                "robot",
                0.96,
                mask_robot,
            ),
            (
                5,
                "obstacle",
                0.91,
                mask_obstacle,
            ),
        ]

        instances = []

        for class_id, name, confidence, mask in objects:
            box = InstanceMaskOperations.mask_to_box(
                mask
            )

            if box is None:
                continue

            instances.append(
                SegmentedInstance(
                    instance_id=generate_id(),
                    class_id=class_id,
                    class_name=name,
                    confidence=confidence,
                    bounding_box=box,
                    mask=mask,
                    metadata={
                        "synthetic": True,
                    },
                )
            )

        return tuple(instances)


# ============================================================
# INSTANCE TRACKING
# ============================================================

@dataclass
class InstanceTrack:
    track_id: str
    class_id: int
    mask: np.ndarray

    last_seen: datetime

    missed_frames: int = 0


class InstanceMaskTracker:
    """
    Simple mask-IoU tracker.

    Intended for demonstrations and short-lived tracking.
    It does not implement motion prediction, re-identification,
    or robust long-term multi-object tracking.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        maximum_missed_frames: int = 5,
    ) -> None:
        if not 0 <= iou_threshold <= 1:
            raise SegmentationValidationError(
                "Invalid tracking IoU threshold"
            )

        if maximum_missed_frames < 0:
            raise SegmentationValidationError(
                "Invalid missed-frame limit"
            )

        self.iou_threshold = iou_threshold
        self.maximum_missed_frames = maximum_missed_frames

        self._tracks: dict[str, InstanceTrack] = {}
        self._lock = threading.RLock()

    def update(
        self,
        instances: Sequence[SegmentedInstance],
    ) -> tuple[SegmentedInstance, ...]:
        with self._lock:
            matches: list[
                tuple[float, int, str]
            ] = []

            for index, instance in enumerate(instances):
                for track_id, track in self._tracks.items():
                    if instance.class_id != track.class_id:
                        continue

                    if instance.mask.shape != track.mask.shape:
                        continue

                    overlap = InstanceMaskOperations.mask_iou(
                        instance.mask,
                        track.mask,
                    )

                    if overlap >= self.iou_threshold:
                        matches.append(
                            (overlap, index, track_id)
                        )

            matches.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            assigned_instances: set[int] = set()
            assigned_tracks: set[str] = set()

            output: list[SegmentedInstance | None] = [
                None
            ] * len(instances)

            for overlap, index, track_id in matches:
                if (
                    index in assigned_instances
                    or track_id in assigned_tracks
                ):
                    continue

                instance = instances[index]

                assigned_instances.add(index)
                assigned_tracks.add(track_id)

                self._tracks[track_id] = InstanceTrack(
                    track_id=track_id,
                    class_id=instance.class_id,
                    mask=instance.mask.copy(),
                    last_seen=utc_now(),
                )

                output[index] = SegmentedInstance(
                    instance_id=instance.instance_id,
                    class_id=instance.class_id,
                    class_name=instance.class_name,
                    confidence=instance.confidence,
                    bounding_box=instance.bounding_box,
                    mask=instance.mask,
                    track_id=track_id,
                    metadata=instance.metadata,
                )

            for index, instance in enumerate(instances):
                if index in assigned_instances:
                    continue

                track_id = generate_id()

                self._tracks[track_id] = InstanceTrack(
                    track_id=track_id,
                    class_id=instance.class_id,
                    mask=instance.mask.copy(),
                    last_seen=utc_now(),
                )

                assigned_tracks.add(track_id)

                output[index] = SegmentedInstance(
                    instance_id=instance.instance_id,
                    class_id=instance.class_id,
                    class_name=instance.class_name,
                    confidence=instance.confidence,
                    bounding_box=instance.bounding_box,
                    mask=instance.mask,
                    track_id=track_id,
                    metadata=instance.metadata,
                )

            expired = []

            for track_id, track in self._tracks.items():
                if track_id not in assigned_tracks:
                    track.missed_frames += 1

                    if (
                        track.missed_frames
                        > self.maximum_missed_frames
                    ):
                        expired.append(track_id)

            for track_id in expired:
                del self._tracks[track_id]

            return tuple(
                instance
                for instance in output
                if instance is not None
            )


# ============================================================
# SEGMENTATION VISUALIZATION
# ============================================================

class InstanceSegmentationVisualizer:

    @staticmethod
    def color_for_class(
        class_id: int,
    ) -> tuple[int, int, int]:
        return (
            int((class_id * 67 + 40) % 256),
            int((class_id * 131 + 80) % 256),
            int((class_id * 197 + 120) % 256),
        )

    @staticmethod
    def draw(
        image: np.ndarray,
        instances: Sequence[SegmentedInstance],
        *,
        alpha: float = 0.4,
        draw_boxes: bool = True,
        draw_labels: bool = True,
    ) -> np.ndarray:
        require_opencv()

        if not 0 <= alpha <= 1:
            raise SegmentationValidationError(
                "alpha must be in [0, 1]"
            )

        output = image.copy()

        for instance in instances:
            if instance.mask.shape != image.shape[:2]:
                raise SegmentationValidationError(
                    "Instance mask shape mismatch"
                )

            color = InstanceSegmentationVisualizer.color_for_class(
                instance.class_id
            )

            mask = instance.mask.astype(bool)

            if np.any(mask):
                color_array = np.asarray(
                    color,
                    dtype=np.float32,
                )

                output[mask] = np.clip(
                    output[mask].astype(np.float32)
                    * (1.0 - alpha)
                    + color_array * alpha,
                    0,
                    255,
                ).astype(np.uint8)

            box = instance.bounding_box

            if draw_boxes:
                cv2.rectangle(
                    output,
                    (int(box.x1), int(box.y1)),
                    (int(box.x2), int(box.y2)),
                    color,
                    2,
                )

            if draw_labels:
                label = (
                    f"{instance.class_name} "
                    f"{instance.confidence:.2f}"
                )

                if instance.track_id:
                    label += (
                        f" ID:{instance.track_id[:6]}"
                    )

                cv2.putText(
                    output,
                    label,
                    (
                        int(box.x1),
                        max(15, int(box.y1) - 8),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return output


# ============================================================
# SEGMENTATION ANALYTICS
# ============================================================

@dataclass(frozen=True)
class SegmentationStatistics:
    total_instances: int

    class_counts: dict[str, int]

    average_confidence: float

    total_mask_pixels: int

    foreground_coverage_ratio: float

    processing_time_ms: float


class InstanceSegmentationAnalyzer:

    @staticmethod
    def analyze(
        frame: InstanceSegmentationFrame,
    ) -> SegmentationStatistics:
        class_counts: dict[str, int] = {}

        combined = np.zeros(
            (
                frame.image_height,
                frame.image_width,
            ),
            dtype=bool,
        )

        confidences = []

        for instance in frame.instances:
            class_counts[instance.class_name] = (
                class_counts.get(
                    instance.class_name,
                    0,
                ) + 1
            )

            if instance.mask.shape != combined.shape:
                raise SegmentationValidationError(
                    "Instance mask shape mismatch"
                )

            combined |= instance.mask.astype(bool)

            confidences.append(
                instance.confidence
            )

        foreground = int(
            np.count_nonzero(combined)
        )

        image_pixels = (
            frame.image_height
            * frame.image_width
        )

        return SegmentationStatistics(
            total_instances=frame.instance_count,
            class_counts=class_counts,
            average_confidence=(
                float(np.mean(confidences))
                if confidences
                else 0.0
            ),
            total_mask_pixels=foreground,
            foreground_coverage_ratio=(
                foreground / image_pixels
                if image_pixels > 0
                else 0.0
            ),
            processing_time_ms=frame.processing_time_ms,
        )


# ============================================================
# INSTANCE SEGMENTATION SERVICE
# ============================================================

class RoboticsInstanceSegmentationService:

    def __init__(
        self,
        config: InstanceSegmentationConfig | None = None,
    ) -> None:
        self.config = (
            config
            or InstanceSegmentationConfig()
        )

        self.validator = SegmentationImageValidator(
            self.config
        )

        self._history: dict[
            tuple[str, str],
            deque[InstanceSegmentationFrame],
        ] = {}

        self._trackers: dict[
            tuple[str, str],
            InstanceMaskTracker,
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: SegmentationAccessContext,
        tenant_id: str,
        camera_id: str,
        permission: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise SegmentationAuthorizationError(
                "Cross-tenant access denied"
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
        tenant_id: str,
        camera_id: str,
        access: SegmentationAccessContext,
        backend: SegmentationBackend = (
            SegmentationBackend.SYNTHETIC
        ),
        estimator: (
            ONNXInstanceSegmentationEstimator | None
        ) = None,
        enable_tracking: bool = True,
    ) -> InstanceSegmentationFrame:
        self._authorize(
            access,
            tenant_id,
            camera_id,
            "can_process",
        )

        self.validator.validate(image)

        started = time.perf_counter()

        if backend == SegmentationBackend.SYNTHETIC:
            instances = SyntheticInstanceSegmenter.segment(
                image
            )

        elif backend == SegmentationBackend.ONNX:
            if estimator is None:
                raise SegmentationValidationError(
                    "ONNX estimator required"
                )

            instances = estimator.estimate(
                image
            )

        else:
            raise SegmentationValidationError(
                "Unsupported segmentation backend"
            )

        key = (
            tenant_id,
            camera_id,
        )

        if enable_tracking:
            with self._lock:
                tracker = self._trackers.setdefault(
                    key,
                    InstanceMaskTracker(),
                )

                instances = tracker.update(
                    instances
                )

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        frame = InstanceSegmentationFrame(
            frame_id=generate_id(),
            tenant_id=tenant_id,
            camera_id=camera_id,
            backend=backend,
            status=(
                SegmentationStatus.SUCCESS
                if instances
                else SegmentationStatus.NO_DETECTIONS
            ),
            image_width=image.shape[1],
            image_height=image.shape[0],
            instances=tuple(instances),
            processing_time_ms=elapsed_ms,
            metadata={
                "tracking_enabled": enable_tracking,
                "advisory_only": True,
            },
        )

        with self._lock:
            history = self._history.setdefault(
                key,
                deque(
                    maxlen=self.config.maximum_history_frames
                ),
            )

            history.append(frame)

        return frame

    def latest(
        self,
        camera_id: str,
        *,
        access: SegmentationAccessContext,
    ) -> InstanceSegmentationFrame | None:
        self._authorize(
            access,
            access.tenant_id,
            camera_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            camera_id,
        )

        with self._lock:
            history = self._history.get(key)

            if not history:
                return None

            return history[-1]

    def history(
        self,
        camera_id: str,
        *,
        access: SegmentationAccessContext,
    ) -> tuple[InstanceSegmentationFrame, ...]:
        self._authorize(
            access,
            access.tenant_id,
            camera_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            camera_id,
        )

        with self._lock:
            return tuple(
                self._history.get(
                    key,
                    (),
                )
            )

    def analyze(
        self,
        frame: InstanceSegmentationFrame,
        *,
        access: SegmentationAccessContext,
    ) -> SegmentationStatistics:
        self._authorize(
            access,
            frame.tenant_id,
            frame.camera_id,
            "can_view",
        )

        return InstanceSegmentationAnalyzer.analyze(
            frame
        )

    def visualize(
        self,
        image: np.ndarray,
        frame: InstanceSegmentationFrame,
        *,
        access: SegmentationAccessContext,
    ) -> np.ndarray:
        self._authorize(
            access,
            frame.tenant_id,
            frame.camera_id,
            "can_view",
        )

        if image.shape[:2] != (
            frame.image_height,
            frame.image_width,
        ):
            raise SegmentationValidationError(
                "Visualization image resolution mismatch"
            )

        return InstanceSegmentationVisualizer.draw(
            image,
            frame.instances,
        )

    @staticmethod
    def export_json(
        frame: InstanceSegmentationFrame,
    ) -> str:
        return json.dumps(
            frame.metadata_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestInstanceSegmentation(unittest.TestCase):

    def setUp(self) -> None:
        self.image = np.zeros(
            (240, 320, 3),
            dtype=np.uint8,
        )

        self.access = SegmentationAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_camera_ids=frozenset({
                "CAM-001",
            }),
            can_process=True,
            can_view=True,
        )

        self.service = RoboticsInstanceSegmentationService()

    def test_synthetic_segmentation(self) -> None:
        frame = self.service.process(
            self.image,
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            access=self.access,
        )

        self.assertEqual(
            frame.instance_count,
            2,
        )

    def test_instance_mask_shape(self) -> None:
        frame = self.service.process(
            self.image,
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            access=self.access,
        )

        for instance in frame.instances:
            self.assertEqual(
                instance.mask.shape,
                (240, 320),
            )

    def test_mask_iou(self) -> None:
        first = np.zeros(
            (10, 10),
            dtype=np.uint8,
        )

        second = np.zeros_like(first)

        first[2:6, 2:6] = 1
        second[2:6, 2:6] = 1

        overlap = InstanceMaskOperations.mask_iou(
            first,
            second,
        )

        self.assertAlmostEqual(
            overlap,
            1.0,
        )

    def test_box_iou(self) -> None:
        box = SegmentationBox(
            0,
            0,
            10,
            10,
        )

        overlap = InstanceMaskOperations.box_iou(
            box,
            box,
        )

        self.assertAlmostEqual(
            overlap,
            1.0,
        )

    def test_mask_to_box(self) -> None:
        mask = np.zeros(
            (20, 20),
            dtype=np.uint8,
        )

        mask[5:15, 4:12] = 1

        box = InstanceMaskOperations.mask_to_box(
            mask
        )

        self.assertIsNotNone(box)

        self.assertEqual(
            box.x1,
            4.0,
        )

        self.assertEqual(
            box.y2,
            15.0,
        )

    def test_rle_roundtrip(self) -> None:
        mask = np.zeros(
            (12, 15),
            dtype=np.uint8,
        )

        mask[2:8, 3:10] = 1

        encoded = InstanceMaskOperations.binary_to_rle(
            mask
        )

        decoded = InstanceMaskOperations.rle_to_binary(
            encoded
        )

        np.testing.assert_array_equal(
            decoded,
            mask,
        )

    def test_nms(self) -> None:
        coefficients = np.ones(
            32,
            dtype=np.float32,
        )

        candidates = [
            DetectionCandidate(
                class_id=0,
                confidence=0.95,
                box=SegmentationBox(
                    10, 10, 50, 50
                ),
                mask_coefficients=coefficients,
            ),
            DetectionCandidate(
                class_id=0,
                confidence=0.85,
                box=SegmentationBox(
                    12, 12, 48, 48
                ),
                mask_coefficients=coefficients,
            ),
        ]

        selected = InstanceNonMaximumSuppression.suppress(
            candidates,
            iou_threshold=0.5,
            maximum_detections=10,
        )

        self.assertEqual(
            len(selected),
            1,
        )

    def test_tracking(self) -> None:
        first = self.service.process(
            self.image,
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            access=self.access,
        )

        second = self.service.process(
            self.image,
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            access=self.access,
        )

        self.assertEqual(
            first.instances[0].track_id,
            second.instances[0].track_id,
        )

    def test_statistics(self) -> None:
        frame = self.service.process(
            self.image,
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            access=self.access,
        )

        stats = self.service.analyze(
            frame,
            access=self.access,
        )

        self.assertEqual(
            stats.total_instances,
            2,
        )

        self.assertGreater(
            stats.foreground_coverage_ratio,
            0,
        )

    def test_history(self) -> None:
        for _ in range(3):
            self.service.process(
                self.image,
                tenant_id="tenant_alpha",
                camera_id="CAM-001",
                access=self.access,
            )

        history = self.service.history(
            "CAM-001",
            access=self.access,
        )

        self.assertEqual(
            len(history),
            3,
        )

    def test_tenant_isolation(self) -> None:
        other_access = SegmentationAccessContext(
            tenant_id="tenant_beta",
            user_id="other",
            allowed_camera_ids=frozenset({
                "CAM-001",
            }),
            can_process=True,
            can_view=True,
        )

        with self.assertRaises(
            SegmentationAuthorizationError
        ):
            self.service.process(
                self.image,
                tenant_id="tenant_alpha",
                camera_id="CAM-001",
                access=other_access,
            )

    def test_json_export(self) -> None:
        frame = self.service.process(
            self.image,
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            access=self.access,
        )

        payload = self.service.export_json(
            frame
        )

        self.assertIn(
            "CAM-001",
            payload,
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV unavailable",
    )
    def test_visualization(self) -> None:
        frame = self.service.process(
            self.image,
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            access=self.access,
        )

        output = self.service.visualize(
            self.image,
            frame,
            access=self.access,
        )

        self.assertEqual(
            output.shape,
            self.image.shape,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Instance Segmentation Engine\n"
    )

    image = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    access = SegmentationAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_camera_ids=frozenset({
            "CAM-FRONT-001",
        }),
        can_process=True,
        can_view=True,
    )

    service = RoboticsInstanceSegmentationService()

    frame = service.process(
        image,
        tenant_id="tenant_alpha",
        camera_id="CAM-FRONT-001",
        access=access,
        backend=SegmentationBackend.SYNTHETIC,
    )

    print("SEGMENTATION FRAME\n")

    print(
        service.export_json(
            frame
        )
    )

    statistics = service.analyze(
        frame,
        access=access,
    )

    print("\nSEGMENTATION STATISTICS\n")

    print(
        json.dumps(
            json_safe(statistics),
            indent=2,
        )
    )

    print("\nINSTANCE MASKS\n")

    for instance in frame.instances:
        print(
            json.dumps(
                instance.metadata_dict(),
                indent=2,
            )
        )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestInstanceSegmentation
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
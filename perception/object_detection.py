"""Object Detection module.

TODO: implement and test; no real robot control is enabled.
"""
"""Placeholder module: voice_activity_detection.

TODO: Implement, test, and document this component.
"""
# backend/app/robotics/perception/object_detection.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Object Detection Engine

Features:
    - Real-time image object detection
    - ONNX YOLO-style model inference
    - RGB/BGR image preprocessing
    - Letterbox image resizing
    - Bounding-box decoding
    - Confidence filtering
    - Class-aware non-maximum suppression
    - Object distance estimation using depth maps
    - Camera-coordinate 3D localization
    - Basic multi-object tracking
    - Human, vehicle, robot, tool and obstacle detection
    - Advisory proximity-risk classification
    - Multi-camera support
    - Tenant and camera authorization
    - Detection history and audit events
    - JSON export
    - Synthetic detection backend
    - Unit tests

Dependencies:
    numpy
    opencv-python
    onnxruntime (optional)

Install:
    pip install numpy opencv-python onnxruntime

Model compatibility:
    Supports common YOLOv8/YOLO11-style ONNX detection
    tensors shaped [1, 4+C, N] or [1, N, 4+C].

    Models using objectness scores, built-in NMS, or
    different output layouts require a custom decoder.

Safety:
    Detection and risk outputs are advisory only.
    They must not directly authorize physical robot motion.
"""

from __future__ import annotations

import json
import math
import threading
import time
import unittest
import uuid

from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

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

class ObjectDetectionError(Exception):
    """Base object detection exception."""


class DetectionValidationError(ObjectDetectionError):
    """Invalid detection input."""


class DetectionAuthorizationError(ObjectDetectionError):
    """Access denied."""


class DetectionModelError(ObjectDetectionError):
    """Model inference or decoding failed."""


class DetectionCapacityError(ObjectDetectionError):
    """Processing capacity exceeded."""


class DetectionDependencyError(ObjectDetectionError):
    """Required dependency unavailable."""


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
        raise DetectionValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise DetectionValidationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def finite_number(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.number))
    ):
        raise DetectionValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise DetectionValidationError(
            f"{name} must be finite"
        )

    return result


def require_opencv() -> None:
    if cv2 is None:
        raise DetectionDependencyError(
            "OpenCV required: pip install opencv-python"
        )


def to_json_safe(value: Any) -> Any:
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
            str(key): to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [to_json_safe(item) for item in value]

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: to_json_safe(getattr(value, key))
            for key in value.__dataclass_fields__
        }

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class DetectionBackend(str, Enum):
    ONNX = "onnx"
    SYNTHETIC = "synthetic"


class DetectionStatus(str, Enum):
    SUCCESS = "success"
    NO_DETECTIONS = "no_detections"
    DEGRADED = "degraded"


class ObjectCategory(str, Enum):
    PERSON = "person"
    ROBOT = "robot"
    VEHICLE = "vehicle"
    TOOL = "tool"
    EQUIPMENT = "equipment"
    OBSTACLE = "obstacle"
    ANIMAL = "animal"
    OTHER = "other"


class ProximityRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ObjectDetectionConfig:
    confidence_threshold: float = 0.35
    nms_iou_threshold: float = 0.50

    maximum_detections: int = 100
    maximum_candidates: int = 1000

    maximum_image_width: int = 4096
    maximum_image_height: int = 4096
    maximum_image_pixels: int = 16_777_216

    maximum_history_frames: int = 50

    tracking_iou_threshold: float = 0.30
    maximum_missed_frames: int = 5

    critical_distance_m: float = 0.5
    high_risk_distance_m: float = 1.5
    moderate_risk_distance_m: float = 3.0

    class_agnostic_nms: bool = False

    def __post_init__(self) -> None:
        for name in (
            "confidence_threshold",
            "nms_iou_threshold",
            "tracking_iou_threshold",
        ):
            value = finite_number(
                getattr(self, name),
                name,
            )

            if not 0 <= value <= 1:
                raise DetectionValidationError(
                    f"{name} must be in [0, 1]"
                )

        for name in (
            "maximum_detections",
            "maximum_candidates",
            "maximum_image_width",
            "maximum_image_height",
            "maximum_image_pixels",
            "maximum_history_frames",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise DetectionValidationError(
                    f"{name} must be a positive integer"
                )

        if self.maximum_missed_frames < 0:
            raise DetectionValidationError(
                "maximum_missed_frames cannot be negative"
            )

        if self.maximum_candidates < self.maximum_detections:
            raise DetectionValidationError(
                "maximum_candidates must be >= maximum_detections"
            )

        if not (
            0 < self.critical_distance_m
            < self.high_risk_distance_m
            < self.moderate_risk_distance_m
        ):
            raise DetectionValidationError(
                "Distance thresholds must be strictly increasing"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class DetectionAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()
    allowed_camera_ids: frozenset[str] = frozenset()

    can_process: bool = False
    can_view: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.user_id, "user_id")


# ============================================================
# BOUNDING BOX
# ============================================================

@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            finite_number(getattr(self, name), name)

        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise DetectionValidationError(
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
    ) -> BoundingBox | None:
        x1 = max(0.0, min(float(width), self.x1))
        y1 = max(0.0, min(float(height), self.y1))
        x2 = max(0.0, min(float(width), self.x2))
        y2 = max(0.0, min(float(height), self.y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return BoundingBox(x1, y1, x2, y2)

    def iou(self, other: BoundingBox) -> float:
        intersection_width = max(
            0.0,
            min(self.x2, other.x2)
            - max(self.x1, other.x1),
        )

        intersection_height = max(
            0.0,
            min(self.y2, other.y2)
            - max(self.y1, other.y1),
        )

        intersection = (
            intersection_width * intersection_height
        )

        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


# ============================================================
# CAMERA INTRINSICS
# ============================================================

@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if finite_number(self.fx, "fx") <= 0:
            raise DetectionValidationError("fx must be positive")

        if finite_number(self.fy, "fy") <= 0:
            raise DetectionValidationError("fy must be positive")

        finite_number(self.cx, "cx")
        finite_number(self.cy, "cy")

    def project_to_3d(
        self,
        u: float,
        v: float,
        depth_m: float,
    ) -> tuple[float, float, float]:
        depth_m = finite_number(
            depth_m,
            "depth_m",
        )

        if depth_m <= 0:
            raise DetectionValidationError(
                "Depth must be positive"
            )

        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy

        return (
            float(x),
            float(y),
            float(depth_m),
        )


# ============================================================
# DETECTED OBJECT
# ============================================================

@dataclass(frozen=True)
class DetectedObject:
    detection_id: str

    class_id: int
    class_name: str

    confidence: float
    bounding_box: BoundingBox

    category: ObjectCategory = ObjectCategory.OTHER

    track_id: str | None = None

    distance_m: float | None = None

    position_3d: tuple[float, float, float] | None = None

    proximity_risk: ProximityRisk = ProximityRisk.UNKNOWN

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.detection_id,
            "detection_id",
        )

        if self.class_id < 0:
            raise DetectionValidationError(
                "class_id must be nonnegative"
            )

        if not 0 <= self.confidence <= 1:
            raise DetectionValidationError(
                "Confidence must be in [0, 1]"
            )

        if self.distance_m is not None:
            if finite_number(
                self.distance_m,
                "distance_m",
            ) <= 0:
                raise DetectionValidationError(
                    "Distance must be positive"
                )

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe(self)


# ============================================================
# DETECTION FRAME
# ============================================================

@dataclass(frozen=True)
class ObjectDetectionFrame:
    frame_id: str

    tenant_id: str
    robot_id: str
    camera_id: str

    backend: DetectionBackend
    status: DetectionStatus

    image_width: int
    image_height: int

    objects: tuple[DetectedObject, ...]

    processing_time_ms: float

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def object_count(self) -> int:
        return len(self.objects)

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe(self)


# ============================================================
# IMAGE VALIDATION
# ============================================================

class DetectionImageValidator:

    def __init__(
        self,
        config: ObjectDetectionConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise DetectionValidationError(
                "Image must be a NumPy array"
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise DetectionValidationError(
                "Expected HWC image with three channels"
            )

        if image.dtype != np.uint8:
            raise DetectionValidationError(
                "Expected uint8 image"
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
            raise DetectionCapacityError(
                "Image exceeds configured size limits"
            )

        return image


# ============================================================
# NON-MAXIMUM SUPPRESSION
# ============================================================

class DetectionNMS:

    @staticmethod
    def suppress(
        detections: Sequence[DetectedObject],
        *,
        iou_threshold: float,
        maximum_detections: int,
        class_agnostic: bool = False,
    ) -> tuple[DetectedObject, ...]:
        if not 0 <= iou_threshold <= 1:
            raise DetectionValidationError(
                "Invalid IoU threshold"
            )

        ordered = sorted(
            detections,
            key=lambda item: item.confidence,
            reverse=True,
        )

        selected: list[DetectedObject] = []

        for detection in ordered:
            duplicate = False

            for existing in selected:
                if (
                    not class_agnostic
                    and detection.class_id != existing.class_id
                ):
                    continue

                if (
                    detection.bounding_box.iou(
                        existing.bounding_box
                    ) > iou_threshold
                ):
                    duplicate = True
                    break

            if not duplicate:
                selected.append(detection)

            if len(selected) >= maximum_detections:
                break

        return tuple(selected)


# ============================================================
# OBJECT CATEGORY CLASSIFIER
# ============================================================

class ObjectCategoryClassifier:

    CATEGORY_MAP = {
        "person": ObjectCategory.PERSON,
        "human": ObjectCategory.PERSON,
        "worker": ObjectCategory.PERSON,
        "robot": ObjectCategory.ROBOT,
        "robot arm": ObjectCategory.ROBOT,
        "mobile robot": ObjectCategory.ROBOT,
        "car": ObjectCategory.VEHICLE,
        "truck": ObjectCategory.VEHICLE,
        "bus": ObjectCategory.VEHICLE,
        "forklift": ObjectCategory.VEHICLE,
        "vehicle": ObjectCategory.VEHICLE,
        "tool": ObjectCategory.TOOL,
        "hammer": ObjectCategory.TOOL,
        "wrench": ObjectCategory.TOOL,
        "drill": ObjectCategory.TOOL,
        "machine": ObjectCategory.EQUIPMENT,
        "equipment": ObjectCategory.EQUIPMENT,
        "motor": ObjectCategory.EQUIPMENT,
        "obstacle": ObjectCategory.OBSTACLE,
        "barrier": ObjectCategory.OBSTACLE,
        "animal": ObjectCategory.ANIMAL,
        "dog": ObjectCategory.ANIMAL,
        "cat": ObjectCategory.ANIMAL,
    }

    @classmethod
    def classify(
        cls,
        class_name: str,
    ) -> ObjectCategory:
        return cls.CATEGORY_MAP.get(
            class_name.strip().lower(),
            ObjectCategory.OTHER,
        )


# ============================================================
# LETTERBOX TRANSFORM
# ============================================================

@dataclass(frozen=True)
class LetterboxTransform:
    original_width: int
    original_height: int

    input_width: int
    input_height: int

    scale_x: float
    scale_y: float

    pad_left: int
    pad_top: int

    def restore_box(
        self,
        box: BoundingBox,
    ) -> BoundingBox | None:
        restored = BoundingBox(
            x1=(box.x1 - self.pad_left) / self.scale_x,
            y1=(box.y1 - self.pad_top) / self.scale_y,
            x2=(box.x2 - self.pad_left) / self.scale_x,
            y2=(box.y2 - self.pad_top) / self.scale_y,
        )

        return restored.clip(
            self.original_width,
            self.original_height,
        )


# ============================================================
# ONNX MODEL CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ONNXObjectDetectionModelConfig:
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
        if not self.model_path.strip():
            raise DetectionValidationError(
                "model_path required"
            )

        if self.input_width <= 0 or self.input_height <= 0:
            raise DetectionValidationError(
                "Invalid model input dimensions"
            )

        if not self.class_names:
            raise DetectionValidationError(
                "At least one class required"
            )

        if self.input_color_order not in ("RGB", "BGR"):
            raise DetectionValidationError(
                "input_color_order must be RGB or BGR"
            )


# ============================================================
# DETECTION BACKEND PROTOCOL
# ============================================================

class ObjectDetectorBackend(Protocol):

    def detect(
        self,
        image: np.ndarray,
    ) -> tuple[DetectedObject, ...]:
        ...


# ============================================================
# ONNX OBJECT DETECTOR
# ============================================================

class ONNXObjectDetector:
    """
    YOLOv8/YOLO11-style ONNX detection adapter.

    Expected output:
        [1, 4 + classes, candidates]

    or:
        [1, candidates, 4 + classes]

    Each candidate:
        cx, cy, width, height, class_scores...

    Scores must already be probabilities in [0, 1].
    """

    def __init__(
        self,
        model_config: ONNXObjectDetectionModelConfig,
        detection_config: ObjectDetectionConfig | None = None,
    ) -> None:
        if ort is None:
            raise DetectionDependencyError(
                "onnxruntime required"
            )

        require_opencv()

        self.model_config = model_config

        self.config = (
            detection_config
            or ObjectDetectionConfig()
        )

        if not Path(model_config.model_path).is_file():
            raise DetectionModelError(
                "ONNX model file does not exist"
            )

        try:
            self.session = ort.InferenceSession(
                model_config.model_path,
                providers=list(model_config.providers),
            )
        except Exception as exc:
            raise DetectionModelError(
                "Unable to load ONNX model"
            ) from exc

        inputs = self.session.get_inputs()

        if len(inputs) != 1:
            raise DetectionModelError(
                "Expected exactly one model input"
            )

        self.input_name = inputs[0].name

        self._lock = threading.RLock()

    def preprocess(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, LetterboxTransform]:
        DetectionImageValidator(
            self.config
        ).validate(image)

        original_height, original_width = image.shape[:2]

        target_width = self.model_config.input_width
        target_height = self.model_config.input_height

        scale = min(
            target_width / original_width,
            target_height / original_height,
        )

        resized_width = max(
            1,
            min(
                target_width,
                round(original_width * scale),
            ),
        )

        resized_height = max(
            1,
            min(
                target_height,
                round(original_height * scale),
            ),
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
            original_width=original_width,
            original_height=original_height,
            input_width=target_width,
            input_height=target_height,
            scale_x=resized_width / original_width,
            scale_y=resized_height / original_height,
            pad_left=left,
            pad_top=top,
        )

        return tensor, transform

    def decode(
        self,
        output: np.ndarray,
        transform: LetterboxTransform,
    ) -> tuple[DetectedObject, ...]:
        raw = np.asarray(
            output,
            dtype=np.float32,
        )

        if raw.ndim != 3 or raw.shape[0] != 1:
            raise DetectionModelError(
                "Unsupported ONNX detection tensor shape"
            )

        class_count = len(
            self.model_config.class_names
        )

        expected_features = 4 + class_count

        raw = raw[0]

        if raw.shape[0] == expected_features:
            raw = raw.T
        elif raw.shape[1] != expected_features:
            raise DetectionModelError(
                "Model output does not match class count"
            )

        class_scores = raw[:, 4:]

        if not np.all(np.isfinite(class_scores)):
            raise DetectionModelError(
                "Non-finite model class scores"
            )

        if np.any(class_scores < 0) or np.any(class_scores > 1):
            raise DetectionModelError(
                "Expected class probabilities in [0, 1]"
            )

        class_ids = np.argmax(
            class_scores,
            axis=1,
        )

        confidences = np.max(
            class_scores,
            axis=1,
        )

        indices = np.flatnonzero(
            confidences >= self.config.confidence_threshold
        )

        if indices.size > self.config.maximum_candidates:
            candidate_scores = confidences[indices]

            selected_indices = np.argpartition(
                candidate_scores,
                -self.config.maximum_candidates,
            )[-self.config.maximum_candidates:]

            indices = indices[selected_indices]

        detections: list[DetectedObject] = []

        for index in indices:
            cx, cy, width, height = (
                float(value)
                for value in raw[index, :4]
            )

            if not all(
                math.isfinite(value)
                for value in (cx, cy, width, height)
            ):
                continue

            if width <= 0 or height <= 0:
                continue

            model_box = BoundingBox(
                x1=cx - width / 2,
                y1=cy - height / 2,
                x2=cx + width / 2,
                y2=cy + height / 2,
            )

            original_box = transform.restore_box(
                model_box
            )

            if original_box is None:
                continue

            class_id = int(class_ids[index])

            class_name = (
                self.model_config.class_names[class_id]
            )

            detections.append(
                DetectedObject(
                    detection_id=generate_id(),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(confidences[index]),
                    bounding_box=original_box,
                    category=ObjectCategoryClassifier.classify(
                        class_name
                    ),
                    metadata={
                        "backend": "onnx",
                    },
                )
            )

        return DetectionNMS.suppress(
            detections,
            iou_threshold=self.config.nms_iou_threshold,
            maximum_detections=self.config.maximum_detections,
            class_agnostic=self.config.class_agnostic_nms,
        )

    def detect(
        self,
        image: np.ndarray,
    ) -> tuple[DetectedObject, ...]:
        tensor, transform = self.preprocess(
            image
        )

        try:
            with self._lock:
                outputs = self.session.run(
                    None,
                    {self.input_name: tensor},
                )
        except Exception as exc:
            raise DetectionModelError(
                "ONNX inference failed"
            ) from exc

        if not outputs:
            raise DetectionModelError(
                "Model returned no outputs"
            )

        return self.decode(
            outputs[0],
            transform,
        )


# ============================================================
# SYNTHETIC OBJECT DETECTOR
# ============================================================

class SyntheticObjectDetector:
    """
    Deterministic demonstration detector.

    Bounding boxes are generated from image dimensions.
    No actual visual recognition is performed.
    """

    def detect(
        self,
        image: np.ndarray,
    ) -> tuple[DetectedObject, ...]:
        height, width = image.shape[:2]

        specifications = (
            (
                0,
                "person",
                0.96,
                (0.10, 0.15, 0.30, 0.85),
            ),
            (
                1,
                "robot",
                0.93,
                (0.42, 0.25, 0.70, 0.90),
            ),
            (
                5,
                "obstacle",
                0.89,
                (0.75, 0.40, 0.95, 0.85),
            ),
        )

        detections = []

        for class_id, name, confidence, coords in specifications:
            box = BoundingBox(
                x1=coords[0] * width,
                y1=coords[1] * height,
                x2=coords[2] * width,
                y2=coords[3] * height,
            )

            detections.append(
                DetectedObject(
                    detection_id=generate_id(),
                    class_id=class_id,
                    class_name=name,
                    confidence=confidence,
                    bounding_box=box,
                    category=ObjectCategoryClassifier.classify(
                        name
                    ),
                    metadata={
                        "synthetic": True,
                    },
                )
            )

        return tuple(detections)


# ============================================================
# DEPTH-BASED DISTANCE ESTIMATION
# ============================================================

class ObjectDistanceEstimator:

    @staticmethod
    def estimate(
        detections: Sequence[DetectedObject],
        depth_map: np.ndarray,
        *,
        intrinsics: CameraIntrinsics | None = None,
        depth_scale_m: float = 1.0,
    ) -> tuple[DetectedObject, ...]:
        if not isinstance(depth_map, np.ndarray):
            raise DetectionValidationError(
                "depth_map must be a NumPy array"
            )

        if depth_map.ndim != 2:
            raise DetectionValidationError(
                "depth_map must be two-dimensional"
            )

        if not np.issubdtype(
            depth_map.dtype,
            np.number,
        ):
            raise DetectionValidationError(
                "depth_map must be numeric"
            )

        if finite_number(
            depth_scale_m,
            "depth_scale_m",
        ) <= 0:
            raise DetectionValidationError(
                "depth_scale_m must be positive"
            )

        height, width = depth_map.shape

        output = []

        for detection in detections:
            box = detection.bounding_box.clip(
                width,
                height,
            )

            if box is None:
                output.append(detection)
                continue

            x1 = max(0, int(math.floor(box.x1)))
            y1 = max(0, int(math.floor(box.y1)))
            x2 = min(width, int(math.ceil(box.x2)))
            y2 = min(height, int(math.ceil(box.y2)))

            region = depth_map[
                y1:y2,
                x1:x2,
            ].astype(np.float64)

            valid = region[
                np.isfinite(region)
                & (region > 0)
            ]

            if valid.size == 0:
                output.append(detection)
                continue

            distance = float(
                np.median(valid) * depth_scale_m
            )

            position = None

            if intrinsics is not None:
                center_x, center_y = (
                    detection.bounding_box.center
                )

                position = intrinsics.project_to_3d(
                    center_x,
                    center_y,
                    distance,
                )

            output.append(
                replace(
                    detection,
                    distance_m=distance,
                    position_3d=position,
                    metadata={
                        **detection.metadata,
                        "depth_method": "bbox_median",
                        "valid_depth_pixels": int(
                            valid.size
                        ),
                        "distance_reference": (
                            "camera_optical_axis"
                        ),
                    },
                )
            )

        return tuple(output)


# ============================================================
# ADVISORY PROXIMITY ASSESSMENT
# ============================================================

class ObjectProximityAssessor:

    def __init__(
        self,
        config: ObjectDetectionConfig,
    ) -> None:
        self.config = config

    def assess(
        self,
        detections: Sequence[DetectedObject],
    ) -> tuple[DetectedObject, ...]:
        results = []

        for detection in detections:
            distance = detection.distance_m

            if distance is None:
                risk = ProximityRisk.UNKNOWN

            elif distance < self.config.critical_distance_m:
                risk = ProximityRisk.CRITICAL

            elif distance < self.config.high_risk_distance_m:
                risk = ProximityRisk.HIGH

            elif distance < self.config.moderate_risk_distance_m:
                risk = ProximityRisk.MODERATE

            else:
                risk = ProximityRisk.LOW

            results.append(
                replace(
                    detection,
                    proximity_risk=risk,
                )
            )

        return tuple(results)


# ============================================================
# MULTI-OBJECT TRACKING
# ============================================================

@dataclass
class ObjectTrack:
    track_id: str
    class_id: int
    bounding_box: BoundingBox

    last_seen: datetime = field(
        default_factory=utc_now
    )

    missed_frames: int = 0


class ObjectTracker:
    """
    Simple IoU-based tracking.

    Suitable for demonstrations and short sequences.
    Does not provide motion prediction or re-identification.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        maximum_missed_frames: int = 5,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.maximum_missed_frames = maximum_missed_frames

        self._tracks: dict[str, ObjectTrack] = {}

        self._lock = threading.RLock()

    def update(
        self,
        detections: Sequence[DetectedObject],
    ) -> tuple[DetectedObject, ...]:
        with self._lock:
            matches = []

            for index, detection in enumerate(detections):
                for track_id, track in self._tracks.items():
                    if detection.class_id != track.class_id:
                        continue

                    overlap = detection.bounding_box.iou(
                        track.bounding_box
                    )

                    if overlap >= self.iou_threshold:
                        matches.append(
                            (overlap, index, track_id)
                        )

            matches.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            used_detections = set()
            used_tracks = set()

            output: list[DetectedObject | None] = [
                None
            ] * len(detections)

            for _, index, track_id in matches:
                if (
                    index in used_detections
                    or track_id in used_tracks
                ):
                    continue

                detection = detections[index]

                used_detections.add(index)
                used_tracks.add(track_id)

                self._tracks[track_id] = ObjectTrack(
                    track_id=track_id,
                    class_id=detection.class_id,
                    bounding_box=detection.bounding_box,
                )

                output[index] = replace(
                    detection,
                    track_id=track_id,
                )

            for index, detection in enumerate(detections):
                if index in used_detections:
                    continue

                track_id = generate_id()

                self._tracks[track_id] = ObjectTrack(
                    track_id=track_id,
                    class_id=detection.class_id,
                    bounding_box=detection.bounding_box,
                )

                used_tracks.add(track_id)

                output[index] = replace(
                    detection,
                    track_id=track_id,
                )

            expired = []

            for track_id, track in self._tracks.items():
                if track_id not in used_tracks:
                    track.missed_frames += 1

                    if (
                        track.missed_frames
                        > self.maximum_missed_frames
                    ):
                        expired.append(track_id)

            for track_id in expired:
                del self._tracks[track_id]

            return tuple(
                detection
                for detection in output
                if detection is not None
            )


# ============================================================
# DETECTION ANALYTICS
# ============================================================

@dataclass(frozen=True)
class ObjectDetectionStatistics:
    total_objects: int

    class_counts: dict[str, int]
    category_counts: dict[str, int]

    average_confidence: float

    closest_object_distance_m: float | None

    critical_risk_objects: int
    high_risk_objects: int

    processing_time_ms: float


class ObjectDetectionAnalyzer:

    @staticmethod
    def analyze(
        frame: ObjectDetectionFrame,
    ) -> ObjectDetectionStatistics:
        class_counts: dict[str, int] = {}
        category_counts: dict[str, int] = {}

        distances = []
        confidences = []

        critical_count = 0
        high_count = 0

        for detection in frame.objects:
            class_counts[detection.class_name] = (
                class_counts.get(
                    detection.class_name,
                    0,
                ) + 1
            )

            category = detection.category.value

            category_counts[category] = (
                category_counts.get(
                    category,
                    0,
                ) + 1
            )

            confidences.append(
                detection.confidence
            )

            if detection.distance_m is not None:
                distances.append(
                    detection.distance_m
                )

            if detection.proximity_risk == ProximityRisk.CRITICAL:
                critical_count += 1

            if detection.proximity_risk == ProximityRisk.HIGH:
                high_count += 1

        return ObjectDetectionStatistics(
            total_objects=len(frame.objects),
            class_counts=class_counts,
            category_counts=category_counts,
            average_confidence=(
                float(np.mean(confidences))
                if confidences
                else 0.0
            ),
            closest_object_distance_m=(
                min(distances)
                if distances
                else None
            ),
            critical_risk_objects=critical_count,
            high_risk_objects=high_count,
            processing_time_ms=frame.processing_time_ms,
        )


# ============================================================
# VISUALIZATION
# ============================================================

class ObjectDetectionVisualizer:

    @staticmethod
    def draw(
        image: np.ndarray,
        detections: Sequence[DetectedObject],
    ) -> np.ndarray:
        require_opencv()

        output = image.copy()

        category_colors = {
            ObjectCategory.PERSON: (0, 0, 255),
            ObjectCategory.ROBOT: (255, 128, 0),
            ObjectCategory.VEHICLE: (0, 255, 255),
            ObjectCategory.TOOL: (255, 0, 255),
            ObjectCategory.EQUIPMENT: (255, 255, 0),
            ObjectCategory.OBSTACLE: (0, 165, 255),
            ObjectCategory.ANIMAL: (128, 0, 255),
            ObjectCategory.OTHER: (0, 255, 0),
        }

        for detection in detections:
            box = detection.bounding_box

            color = category_colors.get(
                detection.category,
                (0, 255, 0),
            )

            cv2.rectangle(
                output,
                (int(box.x1), int(box.y1)),
                (int(box.x2), int(box.y2)),
                color,
                2,
            )

            label = (
                f"{detection.class_name} "
                f"{detection.confidence:.2f}"
            )

            if detection.distance_m is not None:
                label += (
                    f" {detection.distance_m:.2f}m"
                )

            if detection.track_id:
                label += (
                    f" ID:{detection.track_id[:6]}"
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
# AUDIT LOGGING
# ============================================================

@dataclass(frozen=True)
class DetectionAuditEvent:
    event_id: str
    tenant_id: str
    user_id: str
    robot_id: str
    camera_id: str

    action: str
    frame_id: str

    timestamp: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# OBJECT DETECTION SERVICE
# ============================================================

class RoboticsObjectDetectionService:

    def __init__(
        self,
        config: ObjectDetectionConfig | None = None,
    ) -> None:
        self.config = (
            config
            or ObjectDetectionConfig()
        )

        self.validator = DetectionImageValidator(
            self.config
        )

        self.proximity_assessor = ObjectProximityAssessor(
            self.config
        )

        self.synthetic_detector = SyntheticObjectDetector()

        self._trackers: dict[
            tuple[str, str, str],
            ObjectTracker,
        ] = {}

        self._history: dict[
            tuple[str, str, str],
            deque[ObjectDetectionFrame],
        ] = {}

        self._audit: deque[DetectionAuditEvent] = deque(
            maxlen=1000
        )

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: DetectionAccessContext,
        tenant_id: str,
        robot_id: str,
        camera_id: str,
        permission: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise DetectionAuthorizationError(
                "Cross-tenant access denied"
            )

        if robot_id not in access.allowed_robot_ids:
            raise DetectionAuthorizationError(
                "Robot access denied"
            )

        if camera_id not in access.allowed_camera_ids:
            raise DetectionAuthorizationError(
                "Camera access denied"
            )

        if not getattr(access, permission, False):
            raise DetectionAuthorizationError(
                f"Missing permission: {permission}"
            )

    def process(
        self,
        image: np.ndarray,
        *,
        robot_id: str,
        camera_id: str,
        access: DetectionAccessContext,
        backend: DetectionBackend = DetectionBackend.SYNTHETIC,
        detector: ObjectDetectorBackend | None = None,
        depth_map: np.ndarray | None = None,
        intrinsics: CameraIntrinsics | None = None,
        depth_scale_m: float = 1.0,
        enable_tracking: bool = True,
    ) -> ObjectDetectionFrame:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            camera_id,
            "can_process",
        )

        self.validator.validate(image)

        started = time.perf_counter()

        if backend == DetectionBackend.SYNTHETIC:
            detections = self.synthetic_detector.detect(
                image
            )

        elif backend == DetectionBackend.ONNX:
            if detector is None:
                raise DetectionValidationError(
                    "ONNX detector required"
                )

            detections = detector.detect(
                image
            )

        else:
            raise DetectionValidationError(
                "Unsupported backend"
            )

        if len(detections) > self.config.maximum_detections:
            raise DetectionCapacityError(
                "Detector returned too many objects"
            )

        if depth_map is not None:
            if depth_map.shape != image.shape[:2]:
                raise DetectionValidationError(
                    "Depth map resolution must match image"
                )

            detections = ObjectDistanceEstimator.estimate(
                detections,
                depth_map,
                intrinsics=intrinsics,
                depth_scale_m=depth_scale_m,
            )

        detections = self.proximity_assessor.assess(
            detections
        )

        key = (
            access.tenant_id,
            robot_id,
            camera_id,
        )

        if enable_tracking:
            with self._lock:
                tracker = self._trackers.setdefault(
                    key,
                    ObjectTracker(
                        iou_threshold=(
                            self.config.tracking_iou_threshold
                        ),
                        maximum_missed_frames=(
                            self.config.maximum_missed_frames
                        ),
                    ),
                )

                detections = tracker.update(
                    detections
                )

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        frame = ObjectDetectionFrame(
            frame_id=generate_id(),
            tenant_id=access.tenant_id,
            robot_id=robot_id,
            camera_id=camera_id,
            backend=backend,
            status=(
                DetectionStatus.SUCCESS
                if detections
                else DetectionStatus.NO_DETECTIONS
            ),
            image_width=image.shape[1],
            image_height=image.shape[0],
            objects=tuple(detections),
            processing_time_ms=elapsed_ms,
            metadata={
                "tracking_enabled": enable_tracking,
                "depth_available": depth_map is not None,
                "advisory_only": True,
                "motion_authorization": False,
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

            self._audit.append(
                DetectionAuditEvent(
                    event_id=generate_id(),
                    tenant_id=access.tenant_id,
                    user_id=access.user_id,
                    robot_id=robot_id,
                    camera_id=camera_id,
                    action="object_detection.process",
                    frame_id=frame.frame_id,
                )
            )

        return frame

    def latest(
        self,
        *,
        robot_id: str,
        camera_id: str,
        access: DetectionAccessContext,
    ) -> ObjectDetectionFrame | None:
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
            history = self._history.get(key)

            return history[-1] if history else None

    def history(
        self,
        *,
        robot_id: str,
        camera_id: str,
        access: DetectionAccessContext,
    ) -> tuple[ObjectDetectionFrame, ...]:
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
                self._history.get(key, ())
            )

    def statistics(
        self,
        frame: ObjectDetectionFrame,
        *,
        access: DetectionAccessContext,
    ) -> ObjectDetectionStatistics:
        self._authorize(
            access,
            frame.tenant_id,
            frame.robot_id,
            frame.camera_id,
            "can_view",
        )

        return ObjectDetectionAnalyzer.analyze(
            frame
        )

    def visualize(
        self,
        image: np.ndarray,
        frame: ObjectDetectionFrame,
        *,
        access: DetectionAccessContext,
    ) -> np.ndarray:
        self._authorize(
            access,
            frame.tenant_id,
            frame.robot_id,
            frame.camera_id,
            "can_view",
        )

        if image.shape[:2] != (
            frame.image_height,
            frame.image_width,
        ):
            raise DetectionValidationError(
                "Image resolution mismatch"
            )

        return ObjectDetectionVisualizer.draw(
            image,
            frame.objects,
        )

    def audit_events(
        self,
        *,
        access: DetectionAccessContext,
    ) -> tuple[DetectionAuditEvent, ...]:
        if not access.can_view:
            raise DetectionAuthorizationError(
                "Audit viewing permission denied"
            )

        with self._lock:
            return tuple(
                event
                for event in self._audit
                if (
                    event.tenant_id == access.tenant_id
                    and event.robot_id in access.allowed_robot_ids
                    and event.camera_id in access.allowed_camera_ids
                )
            )

    @staticmethod
    def export_json(
        frame: ObjectDetectionFrame,
    ) -> str:
        return json.dumps(
            frame.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestObjectDetection(unittest.TestCase):

    def setUp(self) -> None:
        self.image = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        self.depth_map = np.full(
            (480, 640),
            2.5,
            dtype=np.float32,
        )

        self.access = DetectionAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            allowed_camera_ids=frozenset({
                "CAM-001",
            }),
            can_process=True,
            can_view=True,
        )

        self.service = RoboticsObjectDetectionService()

    def test_synthetic_detection(self) -> None:
        frame = self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
        )

        self.assertEqual(
            frame.object_count,
            3,
        )

    def test_bounding_box_iou(self) -> None:
        box = BoundingBox(
            10,
            10,
            100,
            100,
        )

        self.assertAlmostEqual(
            box.iou(box),
            1.0,
        )

    def test_category_classification(self) -> None:
        category = ObjectCategoryClassifier.classify(
            "person"
        )

        self.assertEqual(
            category,
            ObjectCategory.PERSON,
        )

    def test_depth_estimation(self) -> None:
        frame = self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
            depth_map=self.depth_map,
        )

        for detection in frame.objects:
            self.assertAlmostEqual(
                detection.distance_m,
                2.5,
            )

    def test_3d_localization(self) -> None:
        intrinsics = CameraIntrinsics(
            fx=600,
            fy=600,
            cx=320,
            cy=240,
        )

        frame = self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
            depth_map=self.depth_map,
            intrinsics=intrinsics,
        )

        self.assertIsNotNone(
            frame.objects[0].position_3d
        )

    def test_tracking(self) -> None:
        first = self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
        )

        second = self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
        )

        self.assertEqual(
            first.objects[0].track_id,
            second.objects[0].track_id,
        )

    def test_nms(self) -> None:
        detector = SyntheticObjectDetector()

        detections = detector.detect(
            self.image
        )

        duplicated = (
            detections[0],
            replace(
                detections[0],
                detection_id=generate_id(),
                confidence=0.80,
            ),
        )

        selected = DetectionNMS.suppress(
            duplicated,
            iou_threshold=0.5,
            maximum_detections=10,
        )

        self.assertEqual(
            len(selected),
            1,
        )

    def test_statistics(self) -> None:
        frame = self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
        )

        stats = self.service.statistics(
            frame,
            access=self.access,
        )

        self.assertEqual(
            stats.total_objects,
            3,
        )

    def test_history(self) -> None:
        for _ in range(3):
            self.service.process(
                self.image,
                robot_id="ROBOT-001",
                camera_id="CAM-001",
                access=self.access,
            )

        history = self.service.history(
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
        )

        self.assertEqual(
            len(history),
            3,
        )

    def test_tenant_isolation(self) -> None:
        other_access = DetectionAccessContext(
            tenant_id="tenant_beta",
            user_id="other_user",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            allowed_camera_ids=frozenset({
                "CAM-001",
            }),
            can_process=True,
            can_view=True,
        )

        with self.assertRaises(
            DetectionAuthorizationError
        ):
            self.service._authorize(
                other_access,
                "tenant_alpha",
                "ROBOT-001",
                "CAM-001",
                "can_process",
            )

    def test_json_export(self) -> None:
        frame = self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
        )

        payload = self.service.export_json(
            frame
        )

        self.assertIn(
            "ROBOT-001",
            payload,
        )

    def test_audit_events(self) -> None:
        self.service.process(
            self.image,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
        )

        events = self.service.audit_events(
            access=self.access,
        )

        self.assertEqual(
            len(events),
            1,
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV unavailable",
    )
    def test_visualization(self) -> None:
        frame = self.service.process(
            self.image,
            robot_id="ROBOT-001",
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
        "Object Detection and Perception Engine\n"
    )

    image = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    depth_map = np.full(
        (480, 640),
        2.5,
        dtype=np.float32,
    )

    intrinsics = CameraIntrinsics(
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
    )

    access = DetectionAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
        }),
        allowed_camera_ids=frozenset({
            "CAM-FRONT-001",
        }),
        can_process=True,
        can_view=True,
    )

    service = RoboticsObjectDetectionService()

    frame = service.process(
        image,
        robot_id="ROBOT-001",
        camera_id="CAM-FRONT-001",
        access=access,
        backend=DetectionBackend.SYNTHETIC,
        depth_map=depth_map,
        intrinsics=intrinsics,
    )

    print("DETECTION RESULT\n")

    print(
        service.export_json(frame)
    )

    stats = service.statistics(
        frame,
        access=access,
    )

    print("\nDETECTION STATISTICS\n")

    print(
        json.dumps(
            to_json_safe(stats),
            indent=2,
        )
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestObjectDetection
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
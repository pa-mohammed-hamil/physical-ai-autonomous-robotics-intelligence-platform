"""Part Detection module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/part_detection.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robotics Part Detection Engine

Python 3.11+

Dependencies:
    pip install numpy opencv-python

Features:
    - Robot component and spare-part detection
    - Reference-image template matching
    - Multiple part detection
    - Multi-scale detection
    - Bounding-box extraction
    - Non-maximum suppression
    - Component identification
    - Missing-part candidate detection
    - Part inventory comparison
    - Detection confidence scoring
    - Image quality assessment
    - Detection overlay generation
    - Multi-tenant isolation
    - Batch image processing
    - Thread-safe detection history
    - JSON report export

Important:
    This implementation uses OpenCV template matching,
    not a trained object detection model.

    Template matching is sensitive to rotation,
    lighting, occlusion, viewpoint and scale.

    A missing detection does not prove that a part
    is absent. Detection scores are not calibrated
    probabilities.

    This module is for inspection decision support.
    It does not control physical robots.
"""

from __future__ import annotations

import json
import math
import re
import threading
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


# ============================================================
# EXCEPTIONS
# ============================================================

class PartDetectionError(Exception):
    pass


class PartValidationError(PartDetectionError):
    pass


class PartNotFound(PartDetectionError):
    pass


class DetectionNotFound(PartDetectionError):
    pass


class DetectionConflict(PartDetectionError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RobotPartCategory(str, Enum):
    MOTOR = "motor"
    GEARBOX = "gearbox"
    BEARING = "bearing"
    JOINT = "joint"
    ACTUATOR = "actuator"
    SENSOR = "sensor"
    CAMERA = "camera"
    LIDAR = "lidar"
    BATTERY = "battery"
    CONTROLLER = "controller"
    CIRCUIT_BOARD = "circuit_board"
    WHEEL = "wheel"
    GRIPPER = "gripper"
    ARM = "arm"
    CONNECTOR = "connector"
    CABLE = "cable"
    FASTENER = "fastener"
    FRAME = "frame"
    OTHER = "other"


class DetectionStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


class PartPresence(str, Enum):
    DETECTED = "detected"
    NOT_DETECTED = "not_detected"
    UNCERTAIN = "uncertain"


class ImageQuality(str, Enum):
    ACCEPTABLE = "acceptable"
    BLURRY = "blurry"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    LOW_CONTRAST = "low_contrast"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}",
            value,
        ) is None
    ):
        raise PartValidationError(
            f"Invalid {name}"
        )

    return value


def validate_text(
    value: str,
    name: str,
    maximum_length: int = 512,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum_length
    ):
        raise PartValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def finite(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise PartValidationError(
            f"{name} must be finite"
        )

    return float(value)


def unit_interval(
    value: Any,
    name: str,
) -> float:
    result = finite(value, name)

    if not 0 <= result <= 1:
        raise PartValidationError(
            f"{name} must be between 0 and 1"
        )

    return result


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PartDetectionConfig:
    maximum_image_width: int = 4096
    maximum_image_height: int = 4096
    maximum_image_pixels: int = 16_777_216
    maximum_file_bytes: int = 20_000_000

    maximum_parts: int = 1000
    maximum_detections: int = 100000
    maximum_matches_per_part: int = 50
    maximum_audit_events: int = 20000

    matching_threshold: float = 0.80
    uncertain_threshold: float = 0.60
    nms_iou_threshold: float = 0.35

    minimum_template_width: int = 12
    minimum_template_height: int = 12

    blur_threshold: float = 70.0
    minimum_brightness: float = 35.0
    maximum_brightness: float = 225.0
    minimum_contrast: float = 20.0

    scale_factors: tuple[float, ...] = (
        0.75,
        0.90,
        1.00,
        1.10,
        1.25,
    )

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_image_width",
            "maximum_image_height",
            "maximum_image_pixels",
            "maximum_file_bytes",
            "maximum_parts",
            "maximum_detections",
            "maximum_matches_per_part",
            "maximum_audit_events",
            "minimum_template_width",
            "minimum_template_height",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise PartValidationError(
                    f"{name} must be a positive integer"
                )

        unit_interval(
            self.matching_threshold,
            "matching_threshold",
        )

        unit_interval(
            self.uncertain_threshold,
            "uncertain_threshold",
        )

        unit_interval(
            self.nms_iou_threshold,
            "nms_iou_threshold",
        )

        if (
            self.uncertain_threshold
            >= self.matching_threshold
        ):
            raise PartValidationError(
                "Uncertain threshold must be lower "
                "than matching threshold"
            )

        if not (
            0 <= self.minimum_brightness
            < self.maximum_brightness
            <= 255
        ):
            raise PartValidationError(
                "Invalid brightness thresholds"
            )

        if (
            not isinstance(self.scale_factors, tuple)
            or not self.scale_factors
        ):
            raise PartValidationError(
                "At least one scale factor is required"
            )

        for scale in self.scale_factors:
            if not 0.1 <= finite(
                scale,
                "scale",
            ) <= 4.0:
                raise PartValidationError(
                    "Scale factor outside allowed range"
                )

        if self.simulation_only is not True:
            raise PartValidationError(
                "Physical robot control is unsupported"
            )


# ============================================================
# PART DEFINITION
# ============================================================

@dataclass(frozen=True)
class RobotPartDefinition:
    tenant_id: str
    part_id: str

    part_name: str
    category: RobotPartCategory

    manufacturer: str = ""
    model_number: str = ""

    safety_critical: bool = False

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.part_id, "part_id")

        validate_text(
            self.part_name,
            "part_name",
            256,
        )

        if not isinstance(
            self.category,
            RobotPartCategory,
        ):
            raise PartValidationError(
                "Invalid part category"
            )

        for name in (
            "manufacturer",
            "model_number",
        ):
            value = getattr(self, name)

            if value:
                validate_text(
                    value,
                    name,
                    256,
                )

        if not isinstance(
            self.safety_critical,
            bool,
        ):
            raise PartValidationError(
                "safety_critical must be boolean"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "part_id": self.part_id,
            "part_name": self.part_name,
            "category": self.category.value,
            "manufacturer": self.manufacturer,
            "model_number": self.model_number,
            "safety_critical": self.safety_critical,
        }


# ============================================================
# BOUNDING BOX
# ============================================================

@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


# ============================================================
# DETECTED PART
# ============================================================

@dataclass(frozen=True)
class DetectedRobotPart:
    detection_id: str

    part_id: str
    part_name: str
    category: RobotPartCategory

    bounding_box: BoundingBox

    matching_score: float
    scale_factor: float

    safety_critical: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "part_id": self.part_id,
            "part_name": self.part_name,
            "category": self.category.value,
            "bounding_box": (
                self.bounding_box.to_dict()
            ),
            "matching_score": round(
                self.matching_score,
                4,
            ),
            "scale_factor": round(
                self.scale_factor,
                4,
            ),
            "safety_critical": self.safety_critical,
            "method": "template_matching",
        }


# ============================================================
# IMAGE QUALITY
# ============================================================

@dataclass(frozen=True)
class PartImageQuality:
    width: int
    height: int

    brightness: float
    contrast: float
    blur_variance: float

    flags: tuple[ImageQuality, ...]
    acceptable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "brightness": round(
                self.brightness,
                3,
            ),
            "contrast": round(
                self.contrast,
                3,
            ),
            "blur_variance": round(
                self.blur_variance,
                3,
            ),
            "flags": [
                flag.value
                for flag in self.flags
            ],
            "acceptable": self.acceptable,
        }


# ============================================================
# DETECTION REQUEST
# ============================================================

@dataclass(frozen=True)
class PartDetectionRequest:
    tenant_id: str
    robot_id: str

    image_reference: str

    expected_part_ids: tuple[str, ...] = ()

    inspection_reference: str = ""

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        validate_text(
            self.image_reference,
            "image_reference",
            1024,
        )

        if not isinstance(
            self.expected_part_ids,
            tuple,
        ):
            raise PartValidationError(
                "Expected part IDs must be a tuple"
            )

        for part_id in self.expected_part_ids:
            validate_id(part_id, "part_id")

        if (
            len(set(self.expected_part_ids))
            != len(self.expected_part_ids)
        ):
            raise PartValidationError(
                "Duplicate expected part ID"
            )

        if self.inspection_reference:
            validate_id(
                self.inspection_reference,
                "inspection_reference",
            )


# ============================================================
# PART PRESENCE RESULT
# ============================================================

@dataclass(frozen=True)
class PartPresenceResult:
    part_id: str
    part_name: str

    presence: PartPresence

    detected_count: int
    best_matching_score: float

    safety_critical: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "part_name": self.part_name,
            "presence": self.presence.value,
            "detected_count": self.detected_count,
            "best_matching_score": round(
                self.best_matching_score,
                4,
            ),
            "safety_critical": self.safety_critical,
        }


# ============================================================
# DETECTION RECORD
# ============================================================

@dataclass
class PartDetectionRecord:
    inspection_id: str
    request: PartDetectionRequest

    image_quality: PartImageQuality

    detections: list[DetectedRobotPart]

    presence_results: list[PartPresenceResult]

    human_review_required: bool

    status: DetectionStatus = (
        DetectionStatus.PENDING_REVIEW
    )

    review_notes: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    reviewed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "inspection_id": self.inspection_id,
            "tenant_id": self.request.tenant_id,
            "robot_id": self.request.robot_id,
            "image_reference": (
                self.request.image_reference
            ),
            "inspection_reference": (
                self.request.inspection_reference
            ),
            "image_quality": (
                self.image_quality.to_dict()
            ),
            "detection_count": len(
                self.detections
            ),
            "detections": [
                detection.to_dict()
                for detection in self.detections
            ],
            "presence_results": [
                result.to_dict()
                for result in self.presence_results
            ],
            "human_review_required": (
                self.human_review_required
            ),
            "status": self.status.value,
            "review_notes": self.review_notes,
            "created_at": (
                self.created_at.isoformat()
            ),
            "reviewed_at": (
                self.reviewed_at.isoformat()
                if self.reviewed_at
                else None
            ),
            "trained_model": False,
            "simulation_only": True,
        }


# ============================================================
# PART DETECTION ENGINE
# ============================================================

class RoboticsPartDetectionEngine:

    def __init__(
        self,
        config: PartDetectionConfig | None = None,
    ) -> None:
        self.config = (
            config or PartDetectionConfig()
        )

        self._parts: dict[
            tuple[str, str],
            RobotPartDefinition,
        ] = {}

        self._templates: dict[
            tuple[str, str],
            np.ndarray,
        ] = {}

        self._records: dict[
            str,
            PartDetectionRecord,
        ] = {}

        self._audit_events: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # IMAGE VALIDATION
    # --------------------------------------------------------

    def validate_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(
            image,
            np.ndarray,
        ):
            raise PartValidationError(
                "Image must be a NumPy array"
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise PartValidationError(
                "Expected uint8 BGR image (H,W,3)"
            )

        height, width = image.shape[:2]

        if (
            height <= 0
            or width <= 0
            or width > self.config.maximum_image_width
            or height > self.config.maximum_image_height
            or height * width
            > self.config.maximum_image_pixels
        ):
            raise PartValidationError(
                "Invalid image dimensions"
            )

        return np.ascontiguousarray(image)

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    def load_image(
        self,
        path: str | Path,
    ) -> np.ndarray:
        path = Path(path)

        if not path.is_file():
            raise PartValidationError(
                "Image file not found"
            )

        if (
            path.stat().st_size
            > self.config.maximum_file_bytes
        ):
            raise PartValidationError(
                "Image file exceeds size limit"
            )

        encoded = np.fromfile(
            str(path),
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise PartValidationError(
                "Unable to decode image"
            )

        return self.validate_image(image)

    # --------------------------------------------------------
    # REGISTER PART
    # --------------------------------------------------------

    def register_part(
        self,
        part: RobotPartDefinition,
        template_image: np.ndarray,
    ) -> dict[str, Any]:
        if not isinstance(
            part,
            RobotPartDefinition,
        ):
            raise PartValidationError(
                "Invalid part definition"
            )

        template = self.validate_image(
            template_image
        )

        height, width = template.shape[:2]

        if (
            width < self.config.minimum_template_width
            or height
            < self.config.minimum_template_height
        ):
            raise PartValidationError(
                "Template image is too small"
            )

        gray = cv2.cvtColor(
            template,
            cv2.COLOR_BGR2GRAY,
        )

        if float(np.std(gray)) < 5.0:
            raise PartValidationError(
                "Template has insufficient visual texture"
            )

        key = (
            part.tenant_id,
            part.part_id,
        )

        with self._lock:
            if key in self._parts:
                raise DetectionConflict(
                    "Part already registered"
                )

            if (
                len(self._parts)
                >= self.config.maximum_parts
            ):
                raise PartDetectionError(
                    "Part registry capacity reached"
                )

            self._parts[key] = part

            self._templates[key] = (
                gray.copy()
            )

            return part.to_dict()

    # --------------------------------------------------------
    # REGISTER PART FROM FILE
    # --------------------------------------------------------

    def register_part_from_file(
        self,
        part: RobotPartDefinition,
        template_path: str | Path,
    ) -> dict[str, Any]:
        image = self.load_image(
            template_path
        )

        return self.register_part(
            part,
            image,
        )

    # --------------------------------------------------------
    # GET PART
    # --------------------------------------------------------

    def get_part(
        self,
        *,
        tenant_id: str,
        part_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(part_id, "part_id")

        with self._lock:
            part = self._parts.get(
                (tenant_id, part_id)
            )

            if part is None:
                raise PartNotFound(
                    "Part not found"
                )

            return part.to_dict()

    # --------------------------------------------------------
    # LIST PARTS
    # --------------------------------------------------------

    def list_parts(
        self,
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        with self._lock:
            return [
                part.to_dict()
                for (tenant, _), part
                in self._parts.items()
                if tenant == tenant_id
            ]

    # --------------------------------------------------------
    # IMAGE QUALITY
    # --------------------------------------------------------

    def assess_image_quality(
        self,
        image: np.ndarray,
    ) -> PartImageQuality:
        image = self.validate_image(
            image
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        brightness = float(
            np.mean(gray)
        )

        contrast = float(
            np.std(gray)
        )

        blur_variance = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        flags = []

        if (
            blur_variance
            < self.config.blur_threshold
        ):
            flags.append(
                ImageQuality.BLURRY
            )

        if (
            brightness
            < self.config.minimum_brightness
        ):
            flags.append(
                ImageQuality.TOO_DARK
            )

        if (
            brightness
            > self.config.maximum_brightness
        ):
            flags.append(
                ImageQuality.TOO_BRIGHT
            )

        if (
            contrast
            < self.config.minimum_contrast
        ):
            flags.append(
                ImageQuality.LOW_CONTRAST
            )

        if not flags:
            flags.append(
                ImageQuality.ACCEPTABLE
            )

        return PartImageQuality(
            width=image.shape[1],
            height=image.shape[0],
            brightness=brightness,
            contrast=contrast,
            blur_variance=blur_variance,
            flags=tuple(flags),
            acceptable=(
                flags == [ImageQuality.ACCEPTABLE]
            ),
        )

    # --------------------------------------------------------
    # IOU
    # --------------------------------------------------------

    @staticmethod
    def intersection_over_union(
        first: BoundingBox,
        second: BoundingBox,
    ) -> float:
        x1 = max(first.x, second.x)
        y1 = max(first.y, second.y)

        x2 = min(
            first.x + first.width,
            second.x + second.width,
        )

        y2 = min(
            first.y + first.height,
            second.y + second.height,
        )

        intersection = (
            max(0, x2 - x1)
            * max(0, y2 - y1)
        )

        union = (
            first.area
            + second.area
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    # --------------------------------------------------------
    # NON-MAXIMUM SUPPRESSION
    # --------------------------------------------------------

    def non_maximum_suppression(
        self,
        detections: Sequence[DetectedRobotPart],
    ) -> list[DetectedRobotPart]:
        ordered = sorted(
            detections,
            key=lambda item: item.matching_score,
            reverse=True,
        )

        selected = []

        for candidate in ordered:
            duplicate = any(
                candidate.part_id == existing.part_id
                and self.intersection_over_union(
                    candidate.bounding_box,
                    existing.bounding_box,
                )
                > self.config.nms_iou_threshold
                for existing in selected
            )

            if not duplicate:
                selected.append(candidate)

        return selected

    # --------------------------------------------------------
    # MATCH TEMPLATE
    # --------------------------------------------------------

    def _match_template(
        self,
        image_gray: np.ndarray,
        part: RobotPartDefinition,
        template_gray: np.ndarray,
    ) -> tuple[
        list[DetectedRobotPart],
        float,
    ]:
        image_height, image_width = (
            image_gray.shape
        )

        candidates = []
        best_score = 0.0

        for scale in self.config.scale_factors:
            width = max(
                1,
                round(
                    template_gray.shape[1]
                    * scale
                ),
            )

            height = max(
                1,
                round(
                    template_gray.shape[0]
                    * scale
                ),
            )

            if (
                width > image_width
                or height > image_height
                or width
                < self.config.minimum_template_width
                or height
                < self.config.minimum_template_height
            ):
                continue

            interpolation = (
                cv2.INTER_AREA
                if scale < 1
                else cv2.INTER_LINEAR
            )

            resized = cv2.resize(
                template_gray,
                (width, height),
                interpolation=interpolation,
            )

            if float(np.std(resized)) < 5.0:
                continue

            scores = cv2.matchTemplate(
                image_gray,
                resized,
                cv2.TM_CCOEFF_NORMED,
            )

            scores = np.nan_to_num(
                scores,
                nan=-1.0,
                posinf=-1.0,
                neginf=-1.0,
            )

            _, maximum, _, _ = (
                cv2.minMaxLoc(scores)
            )

            best_score = max(
                best_score,
                float(maximum),
            )

            # Local maxima reduce duplicate candidates
            # before non-maximum suppression.
            neighborhood = cv2.dilate(
                scores,
                np.ones(
                    (3, 3),
                    dtype=np.uint8,
                ),
            )

            peak_mask = (
                (scores >= self.config.matching_threshold)
                & (scores >= neighborhood - 1e-6)
            )

            ys, xs = np.where(
                peak_mask
            )

            ranked = sorted(
                zip(xs.tolist(), ys.tolist()),
                key=lambda point: float(
                    scores[point[1], point[0]]
                ),
                reverse=True,
            )

            # Bound intermediate candidates per scale.
            limit = (
                self.config.maximum_matches_per_part
                * 4
            )

            for x, y in ranked[:limit]:
                score = float(
                    scores[y, x]
                )

                candidates.append(
                    DetectedRobotPart(
                        detection_id=str(uuid.uuid4()),
                        part_id=part.part_id,
                        part_name=part.part_name,
                        category=part.category,
                        bounding_box=BoundingBox(
                            x=x,
                            y=y,
                            width=width,
                            height=height,
                        ),
                        matching_score=score,
                        scale_factor=scale,
                        safety_critical=(
                            part.safety_critical
                        ),
                    )
                )

        selected = self.non_maximum_suppression(
            candidates
        )

        return (
            selected[
                :self.config.maximum_matches_per_part
            ],
            best_score,
        )

    # --------------------------------------------------------
    # DETECT PARTS
    # --------------------------------------------------------

    def detect(
        self,
        request: PartDetectionRequest,
        image: np.ndarray,
    ) -> tuple[
        dict[str, Any],
        np.ndarray,
    ]:
        if not isinstance(
            request,
            PartDetectionRequest,
        ):
            raise PartValidationError(
                "Invalid detection request"
            )

        image = self.validate_image(
            image
        )

        quality = self.assess_image_quality(
            image
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        with self._lock:
            if (
                len(self._records)
                >= self.config.maximum_detections
            ):
                raise PartDetectionError(
                    "Detection history capacity reached"
                )

            if request.expected_part_ids:
                requested_ids = (
                    request.expected_part_ids
                )
            else:
                requested_ids = tuple(
                    part_id
                    for tenant_id, part_id
                    in self._parts
                    if tenant_id == request.tenant_id
                )

            part_templates = []

            for part_id in requested_ids:
                key = (
                    request.tenant_id,
                    part_id,
                )

                part = self._parts.get(
                    key
                )

                template = self._templates.get(
                    key
                )

                if part is None or template is None:
                    raise PartNotFound(
                        f"Part not registered: {part_id}"
                    )

                part_templates.append(
                    (
                        part,
                        template.copy(),
                    )
                )

        detections = []
        presence_results = []

        for part, template in part_templates:
            matches, best_score = (
                self._match_template(
                    gray,
                    part,
                    template,
                )
            )

            detections.extend(
                matches
            )

            if not quality.acceptable:
                presence = PartPresence.UNCERTAIN

            elif matches:
                presence = PartPresence.DETECTED

            elif (
                best_score
                >= self.config.uncertain_threshold
            ):
                presence = PartPresence.UNCERTAIN

            else:
                presence = PartPresence.NOT_DETECTED

            presence_results.append(
                PartPresenceResult(
                    part_id=part.part_id,
                    part_name=part.part_name,
                    presence=presence,
                    detected_count=len(matches),
                    best_matching_score=best_score,
                    safety_critical=(
                        part.safety_critical
                    ),
                )
            )

        detections.sort(
            key=lambda item: item.matching_score,
            reverse=True,
        )

        human_review_required = (
            not quality.acceptable
            or any(
                result.presence
                != PartPresence.DETECTED
                for result in presence_results
            )
            or any(
                result.safety_critical
                for result in presence_results
            )
        )

        record = PartDetectionRecord(
            inspection_id=str(uuid.uuid4()),
            request=request,
            image_quality=quality,
            detections=detections,
            presence_results=presence_results,
            human_review_required=(
                human_review_required
            ),
        )

        overlay = self.create_overlay(
            image,
            detections,
        )

        with self._lock:
            if (
                len(self._records)
                >= self.config.maximum_detections
            ):
                raise PartDetectionError(
                    "Detection history capacity reached"
                )

            self._records[
                record.inspection_id
            ] = record

            self._audit_events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "timestamp": utc_now().isoformat(),
                    "tenant_id": request.tenant_id,
                    "robot_id": request.robot_id,
                    "inspection_id": (
                        record.inspection_id
                    ),
                    "event": "part_detection_created",
                }
            )

        return (
            record.to_dict(),
            overlay,
        )

    # --------------------------------------------------------
    # CREATE OVERLAY
    # --------------------------------------------------------

    def create_overlay(
        self,
        image: np.ndarray,
        detections: Sequence[DetectedRobotPart],
    ) -> np.ndarray:
        overlay = self.validate_image(
            image
        ).copy()

        for detection in detections:
            box = detection.bounding_box

            color = (
                (0, 165, 255)
                if detection.safety_critical
                else (0, 255, 0)
            )

            cv2.rectangle(
                overlay,
                (box.x, box.y),
                (
                    box.x + box.width,
                    box.y + box.height,
                ),
                color,
                2,
            )

            label = (
                f"{detection.part_name}: "
                f"{detection.matching_score:.2f}"
            )

            cv2.putText(
                overlay,
                label,
                (
                    box.x,
                    max(15, box.y - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        return overlay

    # --------------------------------------------------------
    # DETECT FROM FILE
    # --------------------------------------------------------

    def detect_file(
        self,
        request: PartDetectionRequest,
        image_path: str | Path,
        output_directory: str | Path,
    ) -> dict[str, Any]:
        image = self.load_image(
            image_path
        )

        result, overlay = self.detect(
            request,
            image,
        )

        output = (
            Path(output_directory)
            / result["inspection_id"]
        )

        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        overlay_path = (
            output / "part_detection_overlay.png"
        )

        report_path = (
            output / "part_detection_report.json"
        )

        if not cv2.imwrite(
            str(overlay_path),
            overlay,
        ):
            raise PartDetectionError(
                "Unable to save detection overlay"
            )

        report_path.write_text(
            json.dumps(
                result,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return {
            "inspection": result,
            "files": {
                "overlay": str(overlay_path),
                "report": str(report_path),
            },
        }

    # --------------------------------------------------------
    # BATCH DETECTION
    # --------------------------------------------------------

    def detect_batch(
        self,
        requests_and_paths: Sequence[
            tuple[
                PartDetectionRequest,
                str | Path,
            ]
        ],
        output_directory: str | Path,
    ) -> list[dict[str, Any]]:
        results = []

        for request, image_path in requests_and_paths:
            results.append(
                self.detect_file(
                    request,
                    image_path,
                    output_directory,
                )
            )

        return results

    # --------------------------------------------------------
    # GET DETECTION
    # --------------------------------------------------------

    def get_detection(
        self,
        *,
        tenant_id: str,
        inspection_id: str,
    ) -> dict[str, Any]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            inspection_id,
            "inspection_id",
        )

        with self._lock:
            record = self._records.get(
                inspection_id
            )

            if (
                record is None
                or record.request.tenant_id
                != tenant_id
            ):
                raise DetectionNotFound(
                    "Detection record not found"
                )

            return record.to_dict()

    # --------------------------------------------------------
    # REVIEW DETECTION
    # --------------------------------------------------------

    def review_detection(
        self,
        *,
        tenant_id: str,
        inspection_id: str,
        approved: bool,
        notes: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise DetectionConflict(
                "Review authorization required"
            )

        if not isinstance(approved, bool):
            raise PartValidationError(
                "approved must be boolean"
            )

        notes = validate_text(
            notes,
            "notes",
            2048,
        )

        with self._lock:
            record = self._records.get(
                inspection_id
            )

            if (
                record is None
                or record.request.tenant_id
                != tenant_id
            ):
                raise DetectionNotFound(
                    "Detection record not found"
                )

            if (
                record.status
                != DetectionStatus.PENDING_REVIEW
            ):
                raise DetectionConflict(
                    "Detection already reviewed"
                )

            record.status = (
                DetectionStatus.REVIEWED
                if approved
                else DetectionStatus.REJECTED
            )

            record.review_notes = notes
            record.reviewed_at = utc_now()

            self._audit_events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "timestamp": utc_now().isoformat(),
                    "tenant_id": tenant_id,
                    "robot_id": record.request.robot_id,
                    "inspection_id": inspection_id,
                    "event": (
                        "part_detection_reviewed"
                        if approved
                        else "part_detection_rejected"
                    ),
                }
            )

            return record.to_dict()

    # --------------------------------------------------------
    # LIST DETECTIONS
    # --------------------------------------------------------

    def list_detections(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        if robot_id is not None:
            validate_id(
                robot_id,
                "robot_id",
            )

        with self._lock:
            return [
                record.to_dict()
                for record in self._records.values()
                if (
                    record.request.tenant_id
                    == tenant_id
                    and (
                        robot_id is None
                        or record.request.robot_id
                        == robot_id
                    )
                )
            ]

    # --------------------------------------------------------
    # ROBOT PART SUMMARY
    # --------------------------------------------------------

    def robot_part_summary(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        records = self.list_detections(
            tenant_id=tenant_id,
            robot_id=robot_id,
        )

        presence_counts = Counter()

        for record in records:
            for part in record["presence_results"]:
                presence_counts[
                    part["presence"]
                ] += 1

        return {
            "tenant_id": tenant_id,
            "robot_id": robot_id,
            "inspection_count": len(records),
            "presence_counts": dict(
                presence_counts
            ),
            "pending_reviews": sum(
                record["status"]
                == DetectionStatus.PENDING_REVIEW.value
                for record in records
            ),
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
    ) -> Path:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "registered_parts": self.list_parts(
                    tenant_id=tenant_id
                ),
                "detections": self.list_detections(
                    tenant_id=tenant_id
                ),
                "audit_events": [
                    event
                    for event in self._audit_events
                    if event["tenant_id"] == tenant_id
                ],
                "simulation_only": True,
            }

        output.write_text(
            json.dumps(
                payload,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    engine = RoboticsPartDetectionEngine()

    # Create a synthetic robot part template.
    template = np.full(
        (80, 100, 3),
        210,
        dtype=np.uint8,
    )

    cv2.rectangle(
        template,
        (10, 10),
        (90, 70),
        (60, 60, 60),
        -1,
    )

    cv2.circle(
        template,
        (50, 40),
        18,
        (220, 220, 220),
        -1,
    )

    cv2.line(
        template,
        (15, 15),
        (85, 65),
        (0, 0, 0),
        2,
    )

    part = RobotPartDefinition(
        tenant_id="tenant_alpha",
        part_id="motor_001",
        part_name="Robot Joint Motor",
        category=RobotPartCategory.MOTOR,
        manufacturer="Demo Robotics",
        model_number="MTR-100",
        safety_critical=True,
    )

    engine.register_part(
        part,
        template,
    )

    # Synthetic inspection image.
    image = np.full(
        (480, 640, 3),
        180,
        dtype=np.uint8,
    )

    image[120:200, 150:250] = template
    image[280:360, 400:500] = template

    request = PartDetectionRequest(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        image_reference="synthetic_robot_image",
        expected_part_ids=("motor_001",),
        inspection_reference="inspection_001",
    )

    result, overlay = engine.detect(
        request,
        image,
    )

    output = Path(
        "data/robotics/part_detection"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output / "input.png"),
        image,
    ):
        raise PartDetectionError(
            "Failed to save input image"
        )

    if not cv2.imwrite(
        str(output / "template.png"),
        template,
    ):
        raise PartDetectionError(
            "Failed to save template image"
        )

    if not cv2.imwrite(
        str(output / "overlay.png"),
        overlay,
    ):
        raise PartDetectionError(
            "Failed to save overlay"
        )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    report = engine.export_json(
        output / "part_detection_report.json",
        tenant_id="tenant_alpha",
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()
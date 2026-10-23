"""Image Inspection module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/image_inspection.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robotics Image Inspection Engine

Python 3.11+

Dependencies:
    pip install numpy opencv-python

Features:
    - Robot component image inspection
    - Image upload/path processing
    - Image quality validation
    - Blur and brightness analysis
    - Contrast and exposure analysis
    - Crack-like candidate detection
    - Corrosion-color candidate detection
    - Surface edge/anomaly candidates
    - Damage candidate segmentation
    - Region bounding boxes
    - Damage area percentage
    - Heuristic severity prioritization
    - Human inspection recommendations
    - Inspection image overlays
    - JSON report generation
    - Multi-tenant inspection records
    - Batch image inspection
    - Thread-safe history
    - Runnable synthetic demonstration

Important:
    This module uses classical computer vision heuristics.
    It does not contain a trained damage detection model.
    Detected regions may be normal component features.
    Scores are not calibrated probabilities.
    Human verification is required.
    No physical robot control is performed.
"""

from __future__ import annotations

import json
import math
import re
import threading
import uuid

from collections import deque
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

class ImageInspectionError(Exception):
    pass


class ImageValidationError(ImageInspectionError):
    pass


class InspectionNotFound(ImageInspectionError):
    pass


class InspectionConflict(ImageInspectionError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ImageQuality(str, Enum):
    ACCEPTABLE = "acceptable"
    BLURRY = "blurry"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    LOW_CONTRAST = "low_contrast"


class InspectionStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


class CandidateType(str, Enum):
    CRACK_LIKE = "crack_like"
    CORROSION_COLOR = "corrosion_color"
    EDGE_ANOMALY = "edge_anomaly"


class InspectionPriority(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}",
            value,
        )
    ):
        raise ImageValidationError(
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
        raise ImageValidationError(
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
        raise ImageValidationError(
            f"{name} must be finite"
        )
    return float(value)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ImageInspectionConfig:
    maximum_image_width: int = 4096
    maximum_image_height: int = 4096
    maximum_image_pixels: int = 16_777_216
    maximum_file_bytes: int = 20_000_000

    maximum_inspections: int = 100000
    maximum_regions: int = 300
    maximum_audit_events: int = 20000

    blur_threshold: float = 70.0
    minimum_brightness: float = 35.0
    maximum_brightness: float = 225.0
    minimum_contrast: float = 20.0

    gaussian_kernel_size: int = 5
    blackhat_kernel_size: int = 15
    blackhat_threshold: int = 30

    canny_low: int = 60
    canny_high: int = 160

    morphology_kernel_size: int = 3
    minimum_region_area: int = 30

    low_area_threshold: float = 2.0
    high_area_threshold: float = 10.0

    simulation_only: bool = True

    def __post_init__(self) -> None:
        positive_fields = (
            "maximum_image_width",
            "maximum_image_height",
            "maximum_image_pixels",
            "maximum_file_bytes",
            "maximum_inspections",
            "maximum_regions",
            "maximum_audit_events",
            "gaussian_kernel_size",
            "blackhat_kernel_size",
            "morphology_kernel_size",
            "minimum_region_area",
        )

        for name in positive_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ImageValidationError(
                    f"{name} must be a positive integer"
                )

        for name in (
            "gaussian_kernel_size",
            "blackhat_kernel_size",
            "morphology_kernel_size",
        ):
            if getattr(self, name) % 2 == 0:
                raise ImageValidationError(
                    f"{name} must be odd"
                )

        for name in (
            "blur_threshold",
            "minimum_brightness",
            "maximum_brightness",
            "minimum_contrast",
            "low_area_threshold",
            "high_area_threshold",
        ):
            finite(getattr(self, name), name)

        if not (
            0 <= self.minimum_brightness
            < self.maximum_brightness
            <= 255
        ):
            raise ImageValidationError(
                "Invalid brightness thresholds"
            )

        if not (
            0 <= self.low_area_threshold
            < self.high_area_threshold
            <= 100
        ):
            raise ImageValidationError(
                "Invalid area thresholds"
            )

        if not (
            0 <= self.canny_low
            < self.canny_high
            <= 255
        ):
            raise ImageValidationError(
                "Invalid Canny thresholds"
            )

        if not (
            0 <= self.blackhat_threshold <= 255
        ):
            raise ImageValidationError(
                "Invalid blackhat threshold"
            )

        if self.simulation_only is not True:
            raise ImageValidationError(
                "Physical robot operation is unsupported"
            )


# ============================================================
# INSPECTION REQUEST
# ============================================================

@dataclass(frozen=True)
class ImageInspectionRequest:
    tenant_id: str
    robot_id: str
    component_id: str

    image_reference: str
    component_name: str

    safety_critical: bool = False

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.component_id, "component_id")

        validate_text(
            self.image_reference,
            "image_reference",
            1024,
        )

        validate_text(
            self.component_name,
            "component_name",
            256,
        )

        if not isinstance(self.safety_critical, bool):
            raise ImageValidationError(
                "safety_critical must be boolean"
            )


# ============================================================
# IMAGE QUALITY REPORT
# ============================================================

@dataclass(frozen=True)
class ImageQualityReport:
    width: int
    height: int

    brightness: float
    contrast: float
    blur_variance: float

    quality_flags: tuple[ImageQuality, ...]
    acceptable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "brightness": round(
                self.brightness, 3
            ),
            "contrast": round(
                self.contrast, 3
            ),
            "blur_variance": round(
                self.blur_variance, 3
            ),
            "quality_flags": [
                flag.value
                for flag in self.quality_flags
            ],
            "acceptable": self.acceptable,
        }


# ============================================================
# REGION
# ============================================================

@dataclass(frozen=True)
class InspectionRegion:
    region_id: str
    candidate_type: CandidateType

    x: int
    y: int
    width: int
    height: int

    area_pixels: int
    area_percentage: float

    centroid_x: float
    centroid_y: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "candidate_type": (
                self.candidate_type.value
            ),
            "bounding_box": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "area_pixels": self.area_pixels,
            "area_percentage": round(
                self.area_percentage,
                4,
            ),
            "centroid": {
                "x": round(self.centroid_x, 2),
                "y": round(self.centroid_y, 2),
            },
        }


# ============================================================
# INSPECTION RESULT
# ============================================================

@dataclass
class ImageInspectionResult:
    inspection_id: str
    request: ImageInspectionRequest

    quality: ImageQualityReport

    candidate_area_pixels: int
    candidate_area_percentage: float

    regions: list[InspectionRegion]

    priority: InspectionPriority

    human_review_required: bool

    status: InspectionStatus = (
        InspectionStatus.PENDING_REVIEW
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
            "component_id": self.request.component_id,
            "component_name": (
                self.request.component_name
            ),
            "image_reference": (
                self.request.image_reference
            ),
            "safety_critical": (
                self.request.safety_critical
            ),
            "image_quality": self.quality.to_dict(),
            "candidate_area_pixels": (
                self.candidate_area_pixels
            ),
            "candidate_area_percentage": round(
                self.candidate_area_percentage,
                4,
            ),
            "candidate_region_count": len(
                self.regions
            ),
            "regions": [
                region.to_dict()
                for region in self.regions
            ],
            "priority": self.priority.value,
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
            "method": "classical_cv_heuristics",
            "trained_model": False,
            "simulation_only": True,
        }


# ============================================================
# IMAGE INSPECTION ENGINE
# ============================================================

class RoboticsImageInspectionEngine:

    def __init__(
        self,
        config: ImageInspectionConfig | None = None,
    ) -> None:
        self.config = (
            config or ImageInspectionConfig()
        )

        self._inspections: dict[
            str,
            ImageInspectionResult,
        ] = {}

        self._audit: deque[
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
        if not isinstance(image, np.ndarray):
            raise ImageValidationError(
                "Image must be a NumPy array"
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise ImageValidationError(
                "Expected uint8 BGR image (H,W,3)"
            )

        height, width = image.shape[:2]

        if (
            height == 0
            or width == 0
            or width > self.config.maximum_image_width
            or height > self.config.maximum_image_height
            or width * height
            > self.config.maximum_image_pixels
        ):
            raise ImageValidationError(
                "Invalid image dimensions"
            )

        return np.ascontiguousarray(image)

    # --------------------------------------------------------
    # IMAGE LOADING
    # --------------------------------------------------------

    def load_image(
        self,
        image_path: str | Path,
    ) -> np.ndarray:
        path = Path(image_path)

        if not path.is_file():
            raise ImageValidationError(
                "Image file not found"
            )

        if (
            path.stat().st_size
            > self.config.maximum_file_bytes
        ):
            raise ImageValidationError(
                "Image file too large"
            )

        data = np.fromfile(
            str(path),
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            data,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ImageValidationError(
                "Cannot decode image"
            )

        return self.validate_image(image)

    # --------------------------------------------------------
    # IMAGE QUALITY
    # --------------------------------------------------------

    def assess_image_quality(
        self,
        image: np.ndarray,
    ) -> ImageQualityReport:
        image = self.validate_image(image)

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

        laplacian = cv2.Laplacian(
            gray,
            cv2.CV_64F,
        )

        blur_variance = float(
            laplacian.var()
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

        return ImageQualityReport(
            width=image.shape[1],
            height=image.shape[0],
            brightness=brightness,
            contrast=contrast,
            blur_variance=blur_variance,
            quality_flags=tuple(flags),
            acceptable=(
                flags == [ImageQuality.ACCEPTABLE]
            ),
        )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    def preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        image = self.validate_image(image)

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        kernel = (
            self.config.gaussian_kernel_size
        )

        return cv2.GaussianBlur(
            gray,
            (kernel, kernel),
            0,
        )

    # --------------------------------------------------------
    # CRACK-LIKE CANDIDATES
    # --------------------------------------------------------

    def detect_crack_candidates(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        gray = self.preprocess(image)

        kernel_size = (
            self.config.blackhat_kernel_size
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size),
        )

        blackhat = cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            kernel,
        )

        _, mask = cv2.threshold(
            blackhat,
            self.config.blackhat_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        return mask

    # --------------------------------------------------------
    # CORROSION-COLOR CANDIDATES
    # --------------------------------------------------------

    def detect_corrosion_candidates(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        image = self.validate_image(image)

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        lower = np.array(
            [5, 45, 25],
            dtype=np.uint8,
        )

        upper = np.array(
            [30, 255, 245],
            dtype=np.uint8,
        )

        return cv2.inRange(
            hsv,
            lower,
            upper,
        )

    # --------------------------------------------------------
    # EDGE CANDIDATES
    # --------------------------------------------------------

    def detect_edge_candidates(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        gray = self.preprocess(image)

        return cv2.Canny(
            gray,
            self.config.canny_low,
            self.config.canny_high,
        )

    # --------------------------------------------------------
    # MASK CLEANUP
    # --------------------------------------------------------

    def clean_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        if (
            not isinstance(mask, np.ndarray)
            or mask.ndim != 2
            or mask.dtype != np.uint8
        ):
            raise ImageValidationError(
                "Invalid mask"
            )

        size = (
            self.config.morphology_kernel_size
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size),
        )

        cleaned = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        return np.where(
            cleaned > 0,
            255,
            0,
        ).astype(np.uint8)

    # --------------------------------------------------------
    # REGION EXTRACTION
    # --------------------------------------------------------

    def extract_regions(
        self,
        mask: np.ndarray,
        candidate_type: CandidateType,
    ) -> list[InspectionRegion]:
        binary = np.where(
            mask > 0,
            1,
            0,
        ).astype(np.uint8)

        height, width = binary.shape

        total_pixels = height * width

        count, _, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
        )

        candidates = []

        for index in range(1, count):
            area = int(
                stats[index, cv2.CC_STAT_AREA]
            )

            if (
                area
                >= self.config.minimum_region_area
            ):
                candidates.append(
                    (area, index)
                )

        candidates.sort(
            reverse=True
        )

        regions = []

        for area, index in candidates[
            :self.config.maximum_regions
        ]:
            x = int(
                stats[index, cv2.CC_STAT_LEFT]
            )

            y = int(
                stats[index, cv2.CC_STAT_TOP]
            )

            width_box = int(
                stats[index, cv2.CC_STAT_WIDTH]
            )

            height_box = int(
                stats[index, cv2.CC_STAT_HEIGHT]
            )

            cx, cy = centroids[index]

            regions.append(
                InspectionRegion(
                    region_id=str(uuid.uuid4()),
                    candidate_type=candidate_type,
                    x=x,
                    y=y,
                    width=width_box,
                    height=height_box,
                    area_pixels=area,
                    area_percentage=(
                        area / total_pixels * 100
                    ),
                    centroid_x=float(cx),
                    centroid_y=float(cy),
                )
            )

        return regions

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    def classify_priority(
        self,
        *,
        area_percentage: float,
        quality_acceptable: bool,
        safety_critical: bool,
    ) -> InspectionPriority:
        if not quality_acceptable:
            return InspectionPriority.UNKNOWN

        if (
            area_percentage
            >= self.config.high_area_threshold
        ):
            return InspectionPriority.HIGH

        if (
            area_percentage
            >= self.config.low_area_threshold
        ):
            return InspectionPriority.MODERATE

        if safety_critical and area_percentage > 0:
            return InspectionPriority.MODERATE

        return InspectionPriority.LOW

    # --------------------------------------------------------
    # INSPECT IMAGE
    # --------------------------------------------------------

    def inspect(
        self,
        request: ImageInspectionRequest,
        image: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        if not isinstance(
            request,
            ImageInspectionRequest,
        ):
            raise ImageValidationError(
                "Invalid inspection request"
            )

        image = self.validate_image(image)

        quality = self.assess_image_quality(
            image
        )

        crack_mask = self.clean_mask(
            self.detect_crack_candidates(
                image
            )
        )

        corrosion_mask = self.clean_mask(
            self.detect_corrosion_candidates(
                image
            )
        )

        edge_mask = self.clean_mask(
            self.detect_edge_candidates(
                image
            )
        )

        combined = cv2.bitwise_or(
            crack_mask,
            corrosion_mask,
        )

        combined = cv2.bitwise_or(
            combined,
            edge_mask,
        )

        combined = self.clean_mask(
            combined
        )

        regions = self.extract_regions(
            combined,
            CandidateType.EDGE_ANOMALY,
        )

        total_pixels = (
            image.shape[0] * image.shape[1]
        )

        candidate_pixels = int(
            cv2.countNonZero(combined)
        )

        candidate_percentage = (
            candidate_pixels
            / total_pixels
            * 100
        )

        priority = self.classify_priority(
            area_percentage=(
                candidate_percentage
            ),
            quality_acceptable=(
                quality.acceptable
            ),
            safety_critical=(
                request.safety_critical
            ),
        )

        human_review_required = (
            request.safety_critical
            or not quality.acceptable
            or candidate_pixels > 0
        )

        result = ImageInspectionResult(
            inspection_id=str(uuid.uuid4()),
            request=request,
            quality=quality,
            candidate_area_pixels=(
                candidate_pixels
            ),
            candidate_area_percentage=(
                candidate_percentage
            ),
            regions=regions,
            priority=priority,
            human_review_required=(
                human_review_required
            ),
        )

        with self._lock:
            if (
                len(self._inspections)
                >= self.config.maximum_inspections
            ):
                raise ImageInspectionError(
                    "Inspection capacity reached"
                )

            self._inspections[
                result.inspection_id
            ] = result

            self._audit.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "timestamp": utc_now().isoformat(),
                    "tenant_id": request.tenant_id,
                    "robot_id": request.robot_id,
                    "inspection_id": (
                        result.inspection_id
                    ),
                    "event": "inspection_created",
                }
            )

        overlay = self.create_overlay(
            image,
            combined,
            regions,
        )

        return (
            result.to_dict(),
            combined,
            overlay,
        )

    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    def create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        regions: Sequence[InspectionRegion],
    ) -> np.ndarray:
        image = self.validate_image(image)

        if mask.shape != image.shape[:2]:
            raise ImageValidationError(
                "Mask/image dimension mismatch"
            )

        colored = image.copy()

        colored[mask > 0] = (
            0, 0, 255
        )

        overlay = cv2.addWeighted(
            image,
            0.60,
            colored,
            0.40,
            0,
        )

        for region in regions:
            cv2.rectangle(
                overlay,
                (region.x, region.y),
                (
                    region.x + region.width,
                    region.y + region.height,
                ),
                (0, 255, 255),
                2,
            )

        return overlay

    # --------------------------------------------------------
    # INSPECT FILE
    # --------------------------------------------------------

    def inspect_file(
        self,
        request: ImageInspectionRequest,
        image_path: str | Path,
        output_directory: str | Path,
    ) -> dict[str, Any]:
        image = self.load_image(
            image_path
        )

        result, mask, overlay = self.inspect(
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

        mask_path = output / "candidate_mask.png"
        overlay_path = output / "inspection_overlay.png"
        report_path = output / "inspection_report.json"

        if not cv2.imwrite(
            str(mask_path),
            mask,
        ):
            raise ImageInspectionError(
                "Unable to save candidate mask"
            )

        if not cv2.imwrite(
            str(overlay_path),
            overlay,
        ):
            raise ImageInspectionError(
                "Unable to save overlay"
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
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "report": str(report_path),
            },
        }

    # --------------------------------------------------------
    # BATCH INSPECTION
    # --------------------------------------------------------

    def inspect_batch(
        self,
        requests_and_paths: Sequence[
            tuple[
                ImageInspectionRequest,
                str | Path,
            ]
        ],
        output_directory: str | Path,
    ) -> list[dict[str, Any]]:
        results = []

        for request, image_path in requests_and_paths:
            results.append(
                self.inspect_file(
                    request,
                    image_path,
                    output_directory,
                )
            )

        return results

    # --------------------------------------------------------
    # GET INSPECTION
    # --------------------------------------------------------

    def get_inspection(
        self,
        *,
        tenant_id: str,
        inspection_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(inspection_id, "inspection_id")

        with self._lock:
            result = self._inspections.get(
                inspection_id
            )

            if (
                result is None
                or result.request.tenant_id
                != tenant_id
            ):
                raise InspectionNotFound(
                    "Inspection not found"
                )

            return result.to_dict()

    # --------------------------------------------------------
    # REVIEW
    # --------------------------------------------------------

    def review_inspection(
        self,
        *,
        tenant_id: str,
        inspection_id: str,
        approved: bool,
        notes: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise InspectionConflict(
                "Authorization required"
            )

        if not isinstance(approved, bool):
            raise ImageValidationError(
                "approved must be boolean"
            )

        validate_text(
            notes,
            "notes",
            2048,
        )

        with self._lock:
            result = self._inspections.get(
                inspection_id
            )

            if (
                result is None
                or result.request.tenant_id
                != tenant_id
            ):
                raise InspectionNotFound(
                    "Inspection not found"
                )

            if (
                result.status
                != InspectionStatus.PENDING_REVIEW
            ):
                raise InspectionConflict(
                    "Inspection already reviewed"
                )

            result.status = (
                InspectionStatus.REVIEWED
                if approved
                else InspectionStatus.REJECTED
            )

            result.review_notes = notes
            result.reviewed_at = utc_now()

            self._audit.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "timestamp": utc_now().isoformat(),
                    "tenant_id": tenant_id,
                    "robot_id": result.request.robot_id,
                    "inspection_id": inspection_id,
                    "event": (
                        "inspection_reviewed"
                        if approved
                        else "inspection_rejected"
                    ),
                }
            )

            return result.to_dict()

    # --------------------------------------------------------
    # LIST INSPECTIONS
    # --------------------------------------------------------

    def list_inspections(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        with self._lock:
            return [
                result.to_dict()
                for result in self._inspections.values()
                if (
                    result.request.tenant_id
                    == tenant_id
                    and (
                        robot_id is None
                        or result.request.robot_id
                        == robot_id
                    )
                )
            ]

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
    ) -> Path:
        validate_id(tenant_id, "tenant_id")

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "inspections": self.list_inspections(
                    tenant_id=tenant_id
                ),
                "audit_events": [
                    event
                    for event in self._audit
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
    engine = RoboticsImageInspectionEngine()

    image = np.full(
        (512, 768, 3),
        190,
        dtype=np.uint8,
    )

    cv2.line(
        image,
        (100, 120),
        (440, 340),
        (20, 20, 20),
        4,
    )

    cv2.circle(
        image,
        (580, 300),
        65,
        (35, 85, 155),
        -1,
    )

    request = ImageInspectionRequest(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        component_id="arm_joint_001",
        image_reference="synthetic_image_001",
        component_name="Robot Arm Joint",
        safety_critical=True,
    )

    result, mask, overlay = engine.inspect(
        request,
        image,
    )

    output = Path(
        "data/robotics/image_inspection"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output / "synthetic_input.png"),
        image,
    ):
        raise ImageInspectionError(
            "Failed to save input image"
        )

    if not cv2.imwrite(
        str(output / "candidate_mask.png"),
        mask,
    ):
        raise ImageInspectionError(
            "Failed to save mask"
        )

    if not cv2.imwrite(
        str(output / "inspection_overlay.png"),
        overlay,
    ):
        raise ImageInspectionError(
            "Failed to save overlay"
        )

    report = engine.export_json(
        output / "inspection_report.json",
        tenant_id="tenant_alpha",
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()
"""Part Identification module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/part_identification.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robot Part Identification Engine

Python 3.11+

Dependencies:
    pip install numpy opencv-python

Features:
    - Robot component identification
    - Reference-image matching
    - ORB feature extraction
    - Multi-scale template matching fallback
    - Visual similarity scoring
    - Part category identification
    - Manufacturer and model metadata
    - Serial number and part-number lookup
    - OCR integration through external text input
    - Unknown part detection
    - Top-K identification candidates
    - Part compatibility metadata
    - Repair and replacement information
    - Part inventory information
    - Human verification workflow
    - Multi-tenant part catalog
    - Thread-safe history
    - JSON report export

Limitations:
    - Uses ORB and classical computer vision.
    - No pretrained neural network is included.
    - Similarity scores are not calibrated probabilities.
    - OCR text must be supplied by an external OCR system.
    - Compatibility must be confirmed against manufacturer data.
    - Results are advisory and do not control physical robots.
"""

from __future__ import annotations

import json
import math
import re
import threading
import uuid

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

class PartIdentificationError(Exception):
    pass


class PartValidationError(PartIdentificationError):
    pass


class PartCatalogError(PartIdentificationError):
    pass


class IdentificationNotFound(PartIdentificationError):
    pass


class IdentificationConflict(PartIdentificationError):
    pass


class IdentificationPermissionDenied(PartIdentificationError):
    pass


# ============================================================
# ENUMS
# ============================================================

class PartCategory(str, Enum):
    MOTOR = "motor"
    GEARBOX = "gearbox"
    BEARING = "bearing"
    ACTUATOR = "actuator"
    JOINT = "joint"
    SENSOR = "sensor"
    CAMERA = "camera"
    LIDAR = "lidar"
    BATTERY = "battery"
    CONTROLLER = "controller"
    CIRCUIT_BOARD = "circuit_board"
    WHEEL = "wheel"
    GRIPPER = "gripper"
    ROBOT_ARM = "robot_arm"
    CONNECTOR = "connector"
    CABLE = "cable"
    FASTENER = "fastener"
    FRAME = "frame"
    POWER_SUPPLY = "power_supply"
    OTHER = "other"


class IdentificationMethod(str, Enum):
    ORB_FEATURES = "orb_features"
    TEMPLATE_MATCHING = "template_matching"
    OCR_METADATA = "ocr_metadata"
    MULTIMODAL_HEURISTIC = "multimodal_heuristic"


class IdentificationStatus(str, Enum):
    IDENTIFIED = "identified"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class PartCondition(str, Enum):
    NEW = "new"
    GOOD = "good"
    USED = "used"
    DAMAGED = "damaged"
    UNKNOWN = "unknown"


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
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise PartValidationError(
            f"{name} must be text"
        )

    cleaned = value.strip()

    if (
        len(cleaned) > maximum_length
        or (not cleaned and not allow_empty)
    ):
        raise PartValidationError(
            f"Invalid {name}"
        )

    return cleaned


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


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def normalize_identifier(
    value: str,
) -> str:
    return re.sub(
        r"[^A-Z0-9]",
        "",
        value.upper(),
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PartIdentificationConfig:
    maximum_image_width: int = 4096
    maximum_image_height: int = 4096
    maximum_image_pixels: int = 16_777_216
    maximum_file_bytes: int = 20_000_000

    maximum_catalog_parts: int = 10000
    maximum_reference_images_per_part: int = 20
    maximum_identifications: int = 100000

    orb_features: int = 1500
    orb_ratio_threshold: float = 0.75
    minimum_orb_matches: int = 8

    identification_threshold: float = 0.75
    uncertain_threshold: float = 0.45

    visual_weight: float = 0.70
    metadata_weight: float = 0.30

    default_top_k: int = 5

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_image_width",
            "maximum_image_height",
            "maximum_image_pixels",
            "maximum_file_bytes",
            "maximum_catalog_parts",
            "maximum_reference_images_per_part",
            "maximum_identifications",
            "orb_features",
            "minimum_orb_matches",
            "default_top_k",
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

        for name in (
            "orb_ratio_threshold",
            "identification_threshold",
            "uncertain_threshold",
            "visual_weight",
            "metadata_weight",
        ):
            value = finite(
                getattr(self, name),
                name,
            )

            if not 0 <= value <= 1:
                raise PartValidationError(
                    f"{name} must be between 0 and 1"
                )

        if (
            self.uncertain_threshold
            >= self.identification_threshold
        ):
            raise PartValidationError(
                "Uncertain threshold must be lower "
                "than identification threshold"
            )

        if not math.isclose(
            self.visual_weight + self.metadata_weight,
            1.0,
            abs_tol=1e-8,
        ):
            raise PartValidationError(
                "Visual and metadata weights must sum to 1"
            )

        if self.simulation_only is not True:
            raise PartValidationError(
                "Only simulation mode is supported"
            )


# ============================================================
# PART CATALOG MODEL
# ============================================================

@dataclass(frozen=True)
class RobotPart:
    tenant_id: str
    part_id: str

    part_name: str
    category: PartCategory

    manufacturer: str = ""
    model_number: str = ""
    part_number: str = ""

    description: str = ""

    compatible_robot_models: tuple[str, ...] = ()

    replacement_part_ids: tuple[str, ...] = ()

    estimated_replacement_price: float = 0.0
    currency: str = "INR"

    stock_quantity: int = 0

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
            PartCategory,
        ):
            raise PartValidationError(
                "Invalid part category"
            )

        for name in (
            "manufacturer",
            "model_number",
            "part_number",
            "description",
        ):
            validate_text(
                getattr(self, name),
                name,
                1024,
                allow_empty=True,
            )

        if (
            not isinstance(self.currency, str)
            or re.fullmatch(
                r"[A-Z]{3}",
                self.currency,
            ) is None
        ):
            raise PartValidationError(
                "Invalid currency"
            )

        if (
            isinstance(self.stock_quantity, bool)
            or not isinstance(self.stock_quantity, int)
            or self.stock_quantity < 0
        ):
            raise PartValidationError(
                "Invalid stock quantity"
            )

        if finite(
            self.estimated_replacement_price,
            "estimated_replacement_price",
        ) < 0:
            raise PartValidationError(
                "Replacement price cannot be negative"
            )

        if not isinstance(
            self.safety_critical,
            bool,
        ):
            raise PartValidationError(
                "safety_critical must be boolean"
            )

        for robot_model in self.compatible_robot_models:
            validate_text(
                robot_model,
                "compatible_robot_model",
                256,
            )

        for replacement_id in self.replacement_part_ids:
            validate_id(
                replacement_id,
                "replacement_part_id",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "part_id": self.part_id,
            "part_name": self.part_name,
            "category": self.category.value,
            "manufacturer": self.manufacturer,
            "model_number": self.model_number,
            "part_number": self.part_number,
            "description": self.description,
            "compatible_robot_models": list(
                self.compatible_robot_models
            ),
            "replacement_part_ids": list(
                self.replacement_part_ids
            ),
            "estimated_replacement_price": (
                self.estimated_replacement_price
            ),
            "currency": self.currency,
            "stock_quantity": self.stock_quantity,
            "safety_critical": self.safety_critical,
        }


# ============================================================
# REFERENCE IMAGE
# ============================================================

@dataclass
class PartReferenceImage:
    reference_id: str
    tenant_id: str
    part_id: str

    image: np.ndarray

    keypoints: Any
    descriptors: np.ndarray | None

    created_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# IDENTIFICATION REQUEST
# ============================================================

@dataclass(frozen=True)
class PartIdentificationRequest:
    tenant_id: str
    robot_id: str

    image_reference: str

    ocr_text: str = ""

    robot_model: str = ""

    expected_category: PartCategory | None = None

    top_k: int = 5

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        validate_text(
            self.image_reference,
            "image_reference",
            1024,
        )

        validate_text(
            self.ocr_text,
            "ocr_text",
            4096,
            allow_empty=True,
        )

        validate_text(
            self.robot_model,
            "robot_model",
            256,
            allow_empty=True,
        )

        if (
            self.expected_category is not None
            and not isinstance(
                self.expected_category,
                PartCategory,
            )
        ):
            raise PartValidationError(
                "Invalid expected category"
            )

        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k <= 0
            or self.top_k > 100
        ):
            raise PartValidationError(
                "top_k must be between 1 and 100"
            )


# ============================================================
# IDENTIFICATION CANDIDATE
# ============================================================

@dataclass(frozen=True)
class PartIdentificationCandidate:
    part_id: str
    part_name: str

    category: PartCategory

    manufacturer: str
    model_number: str
    part_number: str

    visual_score: float
    metadata_score: float
    combined_score: float

    method: IdentificationMethod

    compatible_with_robot: bool | None

    replacement_price: float
    currency: str
    stock_quantity: int

    safety_critical: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "part_name": self.part_name,
            "category": self.category.value,
            "manufacturer": self.manufacturer,
            "model_number": self.model_number,
            "part_number": self.part_number,
            "visual_score": round(
                self.visual_score,
                4,
            ),
            "metadata_score": round(
                self.metadata_score,
                4,
            ),
            "combined_score": round(
                self.combined_score,
                4,
            ),
            "method": self.method.value,
            "compatible_with_robot": (
                self.compatible_with_robot
            ),
            "replacement_price": (
                self.replacement_price
            ),
            "currency": self.currency,
            "stock_quantity": self.stock_quantity,
            "safety_critical": self.safety_critical,
        }


# ============================================================
# IDENTIFICATION RECORD
# ============================================================

@dataclass
class PartIdentificationRecord:
    identification_id: str

    request: PartIdentificationRequest

    status: IdentificationStatus

    candidates: list[PartIdentificationCandidate]

    best_part_id: str | None

    human_review_required: bool = True

    review_status: ReviewStatus = ReviewStatus.PENDING

    verified_part_id: str | None = None

    review_notes: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    reviewed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "identification_id": (
                self.identification_id
            ),
            "tenant_id": self.request.tenant_id,
            "robot_id": self.request.robot_id,
            "image_reference": (
                self.request.image_reference
            ),
            "ocr_text": self.request.ocr_text,
            "robot_model": self.request.robot_model,
            "status": self.status.value,
            "best_part_id": self.best_part_id,
            "candidate_count": len(
                self.candidates
            ),
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "human_review_required": (
                self.human_review_required
            ),
            "review_status": (
                self.review_status.value
            ),
            "verified_part_id": (
                self.verified_part_id
            ),
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
# PART IDENTIFICATION ENGINE
# ============================================================

class RoboticsPartIdentificationEngine:

    def __init__(
        self,
        config: PartIdentificationConfig | None = None,
    ) -> None:
        self.config = (
            config or PartIdentificationConfig()
        )

        self._parts: dict[
            tuple[str, str],
            RobotPart,
        ] = {}

        self._references: dict[
            tuple[str, str],
            list[PartReferenceImage],
        ] = {}

        self._identifications: dict[
            str,
            PartIdentificationRecord,
        ] = {}

        self._audit_events: list[
            dict[str, Any]
        ] = []

        self._lock = threading.RLock()

        self._orb = cv2.ORB_create(
            nfeatures=self.config.orb_features
        )

        self._matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False,
        )

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
                "Expected uint8 BGR image"
            )

        height, width = image.shape[:2]

        if (
            height <= 0
            or width <= 0
            or width > self.config.maximum_image_width
            or height > self.config.maximum_image_height
            or width * height
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
    # EXTRACT FEATURES
    # --------------------------------------------------------

    def extract_features(
        self,
        image: np.ndarray,
    ) -> tuple[Any, np.ndarray | None]:
        image = self.validate_image(
            image
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        keypoints, descriptors = (
            self._orb.detectAndCompute(
                gray,
                None,
            )
        )

        return keypoints, descriptors

    # --------------------------------------------------------
    # REGISTER PART
    # --------------------------------------------------------

    def register_part(
        self,
        part: RobotPart,
    ) -> dict[str, Any]:
        if not isinstance(part, RobotPart):
            raise PartValidationError(
                "Invalid part definition"
            )

        key = (
            part.tenant_id,
            part.part_id,
        )

        with self._lock:
            if key in self._parts:
                raise PartCatalogError(
                    "Part already registered"
                )

            if (
                len(self._parts)
                >= self.config.maximum_catalog_parts
            ):
                raise PartCatalogError(
                    "Catalog capacity reached"
                )

            self._parts[key] = part
            self._references[key] = []

            return part.to_dict()

    # --------------------------------------------------------
    # ADD REFERENCE IMAGE
    # --------------------------------------------------------

    def add_reference_image(
        self,
        *,
        tenant_id: str,
        part_id: str,
        image: np.ndarray,
    ) -> str:
        validate_id(tenant_id, "tenant_id")
        validate_id(part_id, "part_id")

        image = self.validate_image(
            image
        )

        keypoints, descriptors = (
            self.extract_features(image)
        )

        key = (
            tenant_id,
            part_id,
        )

        with self._lock:
            if key not in self._parts:
                raise PartCatalogError(
                    "Part not registered"
                )

            if (
                len(self._references[key])
                >= self.config.maximum_reference_images_per_part
            ):
                raise PartCatalogError(
                    "Reference image limit reached"
                )

            reference = PartReferenceImage(
                reference_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                part_id=part_id,
                image=image.copy(),
                keypoints=keypoints,
                descriptors=(
                    descriptors.copy()
                    if descriptors is not None
                    else None
                ),
            )

            self._references[key].append(
                reference
            )

            return reference.reference_id

    # --------------------------------------------------------
    # ADD REFERENCE FROM FILE
    # --------------------------------------------------------

    def add_reference_from_file(
        self,
        *,
        tenant_id: str,
        part_id: str,
        image_path: str | Path,
    ) -> str:
        image = self.load_image(
            image_path
        )

        return self.add_reference_image(
            tenant_id=tenant_id,
            part_id=part_id,
            image=image,
        )

    # --------------------------------------------------------
    # ORB SIMILARITY
    # --------------------------------------------------------

    def orb_similarity(
        self,
        query_descriptors: np.ndarray | None,
        reference_descriptors: np.ndarray | None,
    ) -> float:
        if (
            query_descriptors is None
            or reference_descriptors is None
            or len(query_descriptors) < 2
            or len(reference_descriptors) < 2
        ):
            return 0.0

        matches = self._matcher.knnMatch(
            query_descriptors,
            reference_descriptors,
            k=2,
        )

        good_matches = []

        for pair in matches:
            if len(pair) != 2:
                continue

            first, second = pair

            if (
                first.distance
                < self.config.orb_ratio_threshold
                * second.distance
            ):
                good_matches.append(first)

        if (
            len(good_matches)
            < self.config.minimum_orb_matches
        ):
            return 0.0

        denominator = max(
            1,
            min(
                len(query_descriptors),
                len(reference_descriptors),
            ),
        )

        return clamp(
            len(good_matches) / denominator
        )

    # --------------------------------------------------------
    # TEMPLATE SIMILARITY
    # --------------------------------------------------------

    @staticmethod
    def template_similarity(
        query_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> float:
        query_gray = cv2.cvtColor(
            query_image,
            cv2.COLOR_BGR2GRAY,
        )

        reference_gray = cv2.cvtColor(
            reference_image,
            cv2.COLOR_BGR2GRAY,
        )

        query_height, query_width = (
            query_gray.shape
        )

        reference_height, reference_width = (
            reference_gray.shape
        )

        if (
            reference_height > query_height
            or reference_width > query_width
        ):
            scale = min(
                query_width / reference_width,
                query_height / reference_height,
            )

            reference_gray = cv2.resize(
                reference_gray,
                (
                    max(
                        1,
                        int(reference_width * scale),
                    ),
                    max(
                        1,
                        int(reference_height * scale),
                    ),
                ),
                interpolation=cv2.INTER_AREA,
            )

        if float(
            np.std(reference_gray)
        ) < 5.0:
            return 0.0

        result = cv2.matchTemplate(
            query_gray,
            reference_gray,
            cv2.TM_CCOEFF_NORMED,
        )

        _, maximum, _, _ = cv2.minMaxLoc(
            result
        )

        return clamp(
            max(0.0, float(maximum))
        )

    # --------------------------------------------------------
    # METADATA SIMILARITY
    # --------------------------------------------------------

    @staticmethod
    def metadata_similarity(
        part: RobotPart,
        ocr_text: str,
    ) -> float:
        if not ocr_text.strip():
            return 0.0

        normalized_text = normalize_identifier(
            ocr_text
        )

        weighted_fields = (
            (part.part_number, 0.50),
            (part.model_number, 0.30),
            (part.manufacturer, 0.15),
            (part.part_name, 0.05),
        )

        score = 0.0
        available_weight = 0.0

        for value, weight in weighted_fields:
            if not value:
                continue

            normalized_value = (
                normalize_identifier(value)
            )

            if not normalized_value:
                continue

            available_weight += weight

            if (
                normalized_value
                in normalized_text
            ):
                score += weight

        if available_weight == 0:
            return 0.0

        return clamp(
            score / available_weight
        )

    # --------------------------------------------------------
    # ROBOT COMPATIBILITY
    # --------------------------------------------------------

    @staticmethod
    def check_compatibility(
        part: RobotPart,
        robot_model: str,
    ) -> bool | None:
        if not robot_model.strip():
            return None

        if not part.compatible_robot_models:
            return None

        normalized = normalize_identifier(
            robot_model
        )

        return any(
            normalize_identifier(model) == normalized
            for model in part.compatible_robot_models
        )

    # --------------------------------------------------------
    # IDENTIFY PART
    # --------------------------------------------------------

    def identify(
        self,
        request: PartIdentificationRequest,
        image: np.ndarray,
    ) -> dict[str, Any]:
        if not isinstance(
            request,
            PartIdentificationRequest,
        ):
            raise PartValidationError(
                "Invalid identification request"
            )

        image = self.validate_image(
            image
        )

        _, query_descriptors = (
            self.extract_features(image)
        )

        with self._lock:
            catalog = [
                (
                    part,
                    tuple(self._references[key]),
                )
                for key, part in self._parts.items()
                if (
                    key[0] == request.tenant_id
                    and (
                        request.expected_category is None
                        or part.category
                        == request.expected_category
                    )
                )
            ]

        candidates = []

        for part, references in catalog:
            best_orb = 0.0
            best_template = 0.0

            for reference in references:
                orb_score = self.orb_similarity(
                    query_descriptors,
                    reference.descriptors,
                )

                template_score = (
                    self.template_similarity(
                        image,
                        reference.image,
                    )
                )

                best_orb = max(
                    best_orb,
                    orb_score,
                )

                best_template = max(
                    best_template,
                    template_score,
                )

            visual_score = max(
                best_orb,
                best_template,
            )

            metadata_score = (
                self.metadata_similarity(
                    part,
                    request.ocr_text,
                )
            )

            has_visual = bool(references)
            has_metadata = bool(
                request.ocr_text.strip()
            )

            if has_visual and has_metadata:
                combined_score = (
                    visual_score
                    * self.config.visual_weight
                    + metadata_score
                    * self.config.metadata_weight
                )

                method = (
                    IdentificationMethod
                    .MULTIMODAL_HEURISTIC
                )

            elif has_visual:
                combined_score = visual_score

                method = (
                    IdentificationMethod.ORB_FEATURES
                    if best_orb >= best_template
                    else IdentificationMethod.TEMPLATE_MATCHING
                )

            elif has_metadata:
                combined_score = metadata_score

                method = (
                    IdentificationMethod.OCR_METADATA
                )

            else:
                continue

            if combined_score <= 0:
                continue

            candidates.append(
                PartIdentificationCandidate(
                    part_id=part.part_id,
                    part_name=part.part_name,
                    category=part.category,
                    manufacturer=part.manufacturer,
                    model_number=part.model_number,
                    part_number=part.part_number,
                    visual_score=visual_score,
                    metadata_score=metadata_score,
                    combined_score=clamp(
                        combined_score
                    ),
                    method=method,
                    compatible_with_robot=(
                        self.check_compatibility(
                            part,
                            request.robot_model,
                        )
                    ),
                    replacement_price=(
                        part.estimated_replacement_price
                    ),
                    currency=part.currency,
                    stock_quantity=part.stock_quantity,
                    safety_critical=part.safety_critical,
                )
            )

        candidates.sort(
            key=lambda candidate: (
                candidate.combined_score
            ),
            reverse=True,
        )

        candidates = candidates[
            :request.top_k
        ]

        best = (
            candidates[0]
            if candidates
            else None
        )

        if best is None:
            status = IdentificationStatus.UNKNOWN
            best_part_id = None

        elif (
            best.combined_score
            >= self.config.identification_threshold
        ):
            status = IdentificationStatus.IDENTIFIED
            best_part_id = best.part_id

        elif (
            best.combined_score
            >= self.config.uncertain_threshold
        ):
            status = IdentificationStatus.UNCERTAIN
            best_part_id = best.part_id

        else:
            status = IdentificationStatus.UNKNOWN
            best_part_id = None

        record = PartIdentificationRecord(
            identification_id=str(uuid.uuid4()),
            request=request,
            status=status,
            candidates=candidates,
            best_part_id=best_part_id,
        )

        with self._lock:
            if (
                len(self._identifications)
                >= self.config.maximum_identifications
            ):
                raise PartIdentificationError(
                    "Identification capacity reached"
                )

            self._identifications[
                record.identification_id
            ] = record

            self._audit_events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "timestamp": utc_now().isoformat(),
                    "tenant_id": request.tenant_id,
                    "robot_id": request.robot_id,
                    "identification_id": (
                        record.identification_id
                    ),
                    "event": "part_identification_created",
                }
            )

            return record.to_dict()

    # --------------------------------------------------------
    # IDENTIFY FROM FILE
    # --------------------------------------------------------

    def identify_file(
        self,
        request: PartIdentificationRequest,
        image_path: str | Path,
    ) -> dict[str, Any]:
        image = self.load_image(
            image_path
        )

        return self.identify(
            request,
            image,
        )

    # --------------------------------------------------------
    # BATCH IDENTIFICATION
    # --------------------------------------------------------

    def identify_batch(
        self,
        items: Sequence[
            tuple[
                PartIdentificationRequest,
                str | Path,
            ]
        ],
    ) -> list[dict[str, Any]]:
        return [
            self.identify_file(
                request,
                image_path,
            )
            for request, image_path in items
        ]

    # --------------------------------------------------------
    # GET IDENTIFICATION
    # --------------------------------------------------------

    def get_identification(
        self,
        *,
        tenant_id: str,
        identification_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(
            identification_id,
            "identification_id",
        )

        with self._lock:
            record = self._identifications.get(
                identification_id
            )

            if (
                record is None
                or record.request.tenant_id
                != tenant_id
            ):
                raise IdentificationNotFound(
                    "Identification not found"
                )

            return record.to_dict()

    # --------------------------------------------------------
    # REVIEW IDENTIFICATION
    # --------------------------------------------------------

    def review_identification(
        self,
        *,
        tenant_id: str,
        identification_id: str,
        verified_part_id: str | None,
        approved: bool,
        notes: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise IdentificationPermissionDenied(
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
            record = self._identifications.get(
                identification_id
            )

            if (
                record is None
                or record.request.tenant_id
                != tenant_id
            ):
                raise IdentificationNotFound(
                    "Identification not found"
                )

            if (
                record.review_status
                != ReviewStatus.PENDING
            ):
                raise IdentificationConflict(
                    "Identification already reviewed"
                )

            if approved:
                if verified_part_id is None:
                    raise PartValidationError(
                        "Verified part ID required"
                    )

                validate_id(
                    verified_part_id,
                    "verified_part_id",
                )

                if (
                    tenant_id,
                    verified_part_id,
                ) not in self._parts:
                    raise PartCatalogError(
                        "Verified part not in catalog"
                    )

                record.verified_part_id = (
                    verified_part_id
                )

                record.review_status = (
                    ReviewStatus.APPROVED
                )

            else:
                record.verified_part_id = None
                record.review_status = (
                    ReviewStatus.REJECTED
                )

            record.review_notes = notes
            record.reviewed_at = utc_now()

            self._audit_events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "timestamp": utc_now().isoformat(),
                    "tenant_id": tenant_id,
                    "robot_id": record.request.robot_id,
                    "identification_id": (
                        identification_id
                    ),
                    "event": (
                        "identification_approved"
                        if approved
                        else "identification_rejected"
                    ),
                }
            )

            return record.to_dict()

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
                (
                    tenant_id,
                    part_id,
                )
            )

            if part is None:
                raise PartCatalogError(
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
    # LIST IDENTIFICATIONS
    # --------------------------------------------------------

    def list_identifications(
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
                record.to_dict()
                for record in self._identifications.values()
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
                "catalog": self.list_parts(
                    tenant_id=tenant_id
                ),
                "identifications": (
                    self.list_identifications(
                        tenant_id=tenant_id
                    )
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
    engine = RoboticsPartIdentificationEngine()

    part = RobotPart(
        tenant_id="tenant_alpha",
        part_id="motor_001",
        part_name="Industrial Robot Joint Motor",
        category=PartCategory.MOTOR,
        manufacturer="Demo Robotics",
        model_number="MTR-100",
        part_number="DR-MTR-100",
        description="Robot joint actuator motor",
        compatible_robot_models=(
            "ROBOT-X100",
            "ROBOT-X200",
        ),
        replacement_part_ids=(
            "motor_002",
        ),
        estimated_replacement_price=8500.0,
        currency="INR",
        stock_quantity=5,
        safety_critical=True,
    )

    engine.register_part(part)

    template = np.full(
        (180, 240, 3),
        220,
        dtype=np.uint8,
    )

    cv2.rectangle(
        template,
        (25, 35),
        (215, 145),
        (55, 55, 55),
        -1,
    )

    cv2.circle(
        template,
        (120, 90),
        42,
        (190, 190, 190),
        -1,
    )

    cv2.putText(
        template,
        "MTR-100",
        (55, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
    )

    engine.add_reference_image(
        tenant_id="tenant_alpha",
        part_id="motor_001",
        image=template,
    )

    request = PartIdentificationRequest(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        image_reference="motor_image_001",
        ocr_text="Demo Robotics MTR-100 DR-MTR-100",
        robot_model="ROBOT-X100",
        expected_category=PartCategory.MOTOR,
        top_k=5,
    )

    result = engine.identify(
        request,
        template,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    if result["best_part_id"]:
        reviewed = engine.review_identification(
            tenant_id="tenant_alpha",
            identification_id=(
                result["identification_id"]
            ),
            verified_part_id=(
                result["best_part_id"]
            ),
            approved=True,
            notes="Synthetic demo identification verified",
            authorized=True,
        )

        print(
            "\nReviewed identification:"
        )

        print(
            json.dumps(
                reviewed,
                indent=2,
            )
        )

    report = engine.export_json(
        "data/robotics/part_identification_report.json",
        tenant_id="tenant_alpha",
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()
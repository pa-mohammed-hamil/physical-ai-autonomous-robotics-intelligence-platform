"""Image Preprocessing module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/image_preprocessing.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: image_preprocessing.py

Production-oriented image preprocessing for robotics perception.

Features
--------
- RGB, BGR, grayscale, BGRA and RGBA image handling
- Image validation and input size limits
- Image resizing and aspect-ratio-preserving letterboxing
- Center cropping and ROI cropping
- Brightness, contrast and gamma correction
- Histogram equalization and CLAHE
- Gaussian, median and bilateral denoising
- Sharpening and edge detection
- Image normalization and standardization
- Image tensor conversion (CHW / HWC / NCHW)
- Camera intrinsic scaling and cropping
- Lens distortion correction
- Depth-map preprocessing
- Segmentation mask resizing
- Bounding-box coordinate transformations
- Model-specific preprocessing profiles
- Batch preprocessing
- Image quality assessment
- Thread-safe preprocessing pipelines
- Tenant-aware processing service
- Audit events and processing statistics
- JSON metadata export
- Unit tests and demonstration

Dependencies
------------
Required:
    numpy

Optional:
    opencv-python

Installation:
    pip install numpy opencv-python

Safety
------
- Images are processed locally.
- Image bytes are not written to disk automatically.
- Input dimensions and memory consumption are bounded.
- Invalid depth measurements remain invalid.
- Calibration metadata is updated when geometric transforms occur.
- This module does not control physical robot actuators.
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
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ============================================================
# EXCEPTIONS
# ============================================================

class ImagePreprocessingError(Exception):
    """Base preprocessing exception."""


class ImageValidationError(ImagePreprocessingError):
    """Invalid image or image metadata."""


class ImageDependencyError(ImagePreprocessingError):
    """Required optional dependency unavailable."""


class ImageProcessingError(ImagePreprocessingError):
    """Image transformation failed."""


class ImageAuthorizationError(ImagePreprocessingError):
    """Image processing access denied."""


class ImageCalibrationError(ImagePreprocessingError):
    """Invalid camera calibration or geometry."""


class ImageCapacityError(ImagePreprocessingError):
    """Image exceeds configured processing limits."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def validate_identifier(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise ImageValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise ImageValidationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def finite_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ImageValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise ImageValidationError(
            f"{name} must be finite"
        )

    return result


def positive_float(value: Any, name: str) -> float:
    result = finite_float(value, name)

    if result <= 0:
        raise ImageValidationError(
            f"{name} must be positive"
        )

    return result


def require_opencv() -> None:
    if cv2 is None:
        raise ImageDependencyError(
            "This operation requires opencv-python"
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

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if hasattr(value, "__dataclass_fields__"):
        return {
            name: json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class ImageColorSpace(str, Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    RGBA = "rgba"
    BGRA = "bgra"


class ResizeMode(str, Enum):
    STRETCH = "stretch"
    LETTERBOX = "letterbox"
    CENTER_CROP = "center_crop"


class InterpolationMode(str, Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    AREA = "area"


class NormalizationMode(str, Enum):
    NONE = "none"
    ZERO_ONE = "zero_one"
    MINUS_ONE_ONE = "minus_one_one"
    IMAGENET = "imagenet"
    CUSTOM = "custom"


class TensorLayout(str, Enum):
    HWC = "hwc"
    CHW = "chw"
    NHWC = "nhwc"
    NCHW = "nchw"


class DenoisingMethod(str, Enum):
    NONE = "none"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"


class EdgeDetectionMethod(str, Enum):
    SOBEL = "sobel"
    CANNY = "canny"
    LAPLACIAN = "laplacian"


class ImageQualityLevel(str, Enum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ImagePreprocessingConfig:
    maximum_width: int = 8192
    maximum_height: int = 8192
    maximum_pixels: int = 33_554_432
    maximum_image_bytes: int = 256 * 1024 * 1024

    maximum_batch_size: int = 32
    maximum_history_events: int = 1000

    default_interpolation: InterpolationMode = (
        InterpolationMode.LINEAR
    )

    preserve_input: bool = True
    reject_nonfinite_images: bool = True

    minimum_brightness: float = 15.0
    maximum_brightness: float = 245.0
    minimum_contrast_std: float = 10.0
    minimum_sharpness_variance: float = 20.0

    def __post_init__(self) -> None:
        for name in (
            "maximum_width",
            "maximum_height",
            "maximum_pixels",
            "maximum_image_bytes",
            "maximum_batch_size",
            "maximum_history_events",
        ):
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
            "minimum_brightness",
            "maximum_brightness",
            "minimum_contrast_std",
            "minimum_sharpness_variance",
        ):
            finite_float(getattr(self, name), name)

        if self.minimum_brightness >= self.maximum_brightness:
            raise ImageValidationError(
                "Brightness thresholds are inconsistent"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class ImageAccessContext:
    tenant_id: str
    user_id: str

    allowed_camera_ids: frozenset[str] = frozenset()

    can_process: bool = False
    can_view: bool = False
    can_manage: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.user_id, "user_id")


# ============================================================
# GEOMETRY
# ============================================================

@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
        ):
            raise ImageValidationError(
                "Image dimensions must be positive integers"
            )


@dataclass(frozen=True)
class ImageROI:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            self.x < 0
            or self.y < 0
            or self.width <= 0
            or self.height <= 0
        ):
            raise ImageValidationError(
                "Invalid region of interest"
            )


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            finite_float(getattr(self, name), name)

        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ImageValidationError(
                "Bounding box has invalid coordinates"
            )


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int

    fx: float
    fy: float
    cx: float
    cy: float

    skew: float = 0.0

    def __post_init__(self) -> None:
        ImageSize(self.width, self.height)

        positive_float(self.fx, "fx")
        positive_float(self.fy, "fy")

        finite_float(self.cx, "cx")
        finite_float(self.cy, "cy")
        finite_float(self.skew, "skew")

    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, self.skew, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class GeometricTransform:
    """
    Affine mapping from original image coordinates to output coordinates.

        x_out = scale_x * x_in + offset_x
        y_out = scale_y * y_in + offset_y

    Coordinates follow a continuous pixel-center convention.
    """

    source_size: ImageSize
    target_size: ImageSize

    scale_x: float = 1.0
    scale_y: float = 1.0

    offset_x: float = 0.0
    offset_y: float = 0.0

    def __post_init__(self) -> None:
        positive_float(self.scale_x, "scale_x")
        positive_float(self.scale_y, "scale_y")
        finite_float(self.offset_x, "offset_x")
        finite_float(self.offset_y, "offset_y")

    def transform_point(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        return (
            self.scale_x * x + self.offset_x,
            self.scale_y * y + self.offset_y,
        )

    def inverse_point(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        return (
            (x - self.offset_x) / self.scale_x,
            (y - self.offset_y) / self.scale_y,
        )

    def transform_box(
        self,
        box: BoundingBox,
    ) -> BoundingBox:
        x1, y1 = self.transform_point(box.x1, box.y1)
        x2, y2 = self.transform_point(box.x2, box.y2)

        return BoundingBox(x1, y1, x2, y2)

    def inverse_box(
        self,
        box: BoundingBox,
    ) -> BoundingBox:
        x1, y1 = self.inverse_point(box.x1, box.y1)
        x2, y2 = self.inverse_point(box.x2, box.y2)

        return BoundingBox(x1, y1, x2, y2)

    def transform_intrinsics(
        self,
        intrinsics: CameraIntrinsics,
    ) -> CameraIntrinsics:
        if (
            intrinsics.width != self.source_size.width
            or intrinsics.height != self.source_size.height
        ):
            raise ImageCalibrationError(
                "Intrinsics do not match source image resolution"
            )

        return CameraIntrinsics(
            width=self.target_size.width,
            height=self.target_size.height,
            fx=intrinsics.fx * self.scale_x,
            fy=intrinsics.fy * self.scale_y,
            cx=intrinsics.cx * self.scale_x + self.offset_x,
            cy=intrinsics.cy * self.scale_y + self.offset_y,
            skew=intrinsics.skew * self.scale_x,
        )

    def compose(
        self,
        next_transform: GeometricTransform,
    ) -> GeometricTransform:
        if self.target_size != next_transform.source_size:
            raise ImageValidationError(
                "Cannot compose incompatible image transforms"
            )

        return GeometricTransform(
            source_size=self.source_size,
            target_size=next_transform.target_size,
            scale_x=self.scale_x * next_transform.scale_x,
            scale_y=self.scale_y * next_transform.scale_y,
            offset_x=(
                self.offset_x * next_transform.scale_x
                + next_transform.offset_x
            ),
            offset_y=(
                self.offset_y * next_transform.scale_y
                + next_transform.offset_y
            ),
        )


# ============================================================
# IMAGE VALIDATOR
# ============================================================

class ImageValidator:

    def __init__(
        self,
        config: ImagePreprocessingConfig | None = None,
    ) -> None:
        self.config = config or ImagePreprocessingConfig()

    def validate(
        self,
        image: np.ndarray,
        *,
        allow_float: bool = True,
    ) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise ImageValidationError(
                "Image must be a NumPy array"
            )

        if image.ndim not in (2, 3):
            raise ImageValidationError(
                "Image must have 2 or 3 dimensions"
            )

        height, width = image.shape[:2]

        if height <= 0 or width <= 0:
            raise ImageValidationError(
                "Image cannot be empty"
            )

        if (
            width > self.config.maximum_width
            or height > self.config.maximum_height
            or width * height > self.config.maximum_pixels
        ):
            raise ImageCapacityError(
                "Image dimensions exceed configured limits"
            )

        if image.nbytes > self.config.maximum_image_bytes:
            raise ImageCapacityError(
                "Image memory size exceeds configured limit"
            )

        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ImageValidationError(
                "Image must have 1, 3, or 4 channels"
            )

        if not (
            np.issubdtype(image.dtype, np.integer)
            or (
                allow_float
                and np.issubdtype(image.dtype, np.floating)
            )
        ):
            raise ImageValidationError(
                "Unsupported image data type"
            )

        if (
            self.config.reject_nonfinite_images
            and np.issubdtype(image.dtype, np.floating)
            and not np.all(np.isfinite(image))
        ):
            raise ImageValidationError(
                "Image contains NaN or infinity"
            )

        return image

    @staticmethod
    def validate_mask(mask: np.ndarray) -> np.ndarray:
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise ImageValidationError(
                "Segmentation mask must be a 2D array"
            )

        if mask.size == 0:
            raise ImageValidationError(
                "Segmentation mask cannot be empty"
            )

        return mask


# ============================================================
# IMAGE COLOR CONVERSION
# ============================================================

class ImageColorConverter:

    @staticmethod
    def convert(
        image: np.ndarray,
        source: ImageColorSpace,
        target: ImageColorSpace,
    ) -> np.ndarray:
        if source == target:
            return image.copy()

        require_opencv()

        conversions = {
            (ImageColorSpace.BGR, ImageColorSpace.RGB):
                cv2.COLOR_BGR2RGB,

            (ImageColorSpace.RGB, ImageColorSpace.BGR):
                cv2.COLOR_RGB2BGR,

            (ImageColorSpace.BGR, ImageColorSpace.GRAY):
                cv2.COLOR_BGR2GRAY,

            (ImageColorSpace.RGB, ImageColorSpace.GRAY):
                cv2.COLOR_RGB2GRAY,

            (ImageColorSpace.GRAY, ImageColorSpace.BGR):
                cv2.COLOR_GRAY2BGR,

            (ImageColorSpace.GRAY, ImageColorSpace.RGB):
                cv2.COLOR_GRAY2RGB,

            (ImageColorSpace.BGRA, ImageColorSpace.BGR):
                cv2.COLOR_BGRA2BGR,

            (ImageColorSpace.RGBA, ImageColorSpace.RGB):
                cv2.COLOR_RGBA2RGB,

            (ImageColorSpace.BGRA, ImageColorSpace.RGBA):
                cv2.COLOR_BGRA2RGBA,

            (ImageColorSpace.RGBA, ImageColorSpace.BGRA):
                cv2.COLOR_RGBA2BGRA,

            (ImageColorSpace.BGRA, ImageColorSpace.GRAY):
                cv2.COLOR_BGRA2GRAY,

            (ImageColorSpace.RGBA, ImageColorSpace.GRAY):
                cv2.COLOR_RGBA2GRAY,

            (ImageColorSpace.GRAY, ImageColorSpace.BGRA):
                cv2.COLOR_GRAY2BGRA,

            (ImageColorSpace.GRAY, ImageColorSpace.RGBA):
                cv2.COLOR_GRAY2RGBA,
        }

        code = conversions.get((source, target))

        if code is not None:
            return cv2.cvtColor(image, code)

        if source in (
            ImageColorSpace.BGRA,
            ImageColorSpace.RGBA,
        ):
            intermediate = (
                ImageColorSpace.BGR
                if source == ImageColorSpace.BGRA
                else ImageColorSpace.RGB
            )

            return ImageColorConverter.convert(
                ImageColorConverter.convert(
                    image,
                    source,
                    intermediate,
                ),
                intermediate,
                target,
            )

        if target in (
            ImageColorSpace.BGRA,
            ImageColorSpace.RGBA,
        ):
            intermediate = (
                ImageColorSpace.BGR
                if target == ImageColorSpace.BGRA
                else ImageColorSpace.RGB
            )

            return ImageColorConverter.convert(
                ImageColorConverter.convert(
                    image,
                    source,
                    intermediate,
                ),
                intermediate,
                target,
            )

        raise ImageValidationError(
            f"Unsupported conversion: {source} -> {target}"
        )


# ============================================================
# IMAGE RESIZING
# ============================================================

class ImageResizer:

    @staticmethod
    def _interpolation(
        mode: InterpolationMode,
    ) -> int:
        require_opencv()

        return {
            InterpolationMode.NEAREST: cv2.INTER_NEAREST,
            InterpolationMode.LINEAR: cv2.INTER_LINEAR,
            InterpolationMode.CUBIC: cv2.INTER_CUBIC,
            InterpolationMode.AREA: cv2.INTER_AREA,
        }[mode]

    @staticmethod
    def resize(
        image: np.ndarray,
        target: ImageSize,
        interpolation: InterpolationMode = (
            InterpolationMode.LINEAR
        ),
    ) -> tuple[np.ndarray, GeometricTransform]:
        require_opencv()

        source_height, source_width = image.shape[:2]

        output = cv2.resize(
            image,
            (target.width, target.height),
            interpolation=ImageResizer._interpolation(
                interpolation
            ),
        )

        scale_x = target.width / source_width
        scale_y = target.height / source_height

        transform = GeometricTransform(
            source_size=ImageSize(
                source_width,
                source_height,
            ),
            target_size=target,
            scale_x=scale_x,
            scale_y=scale_y,
            offset_x=(scale_x - 1.0) / 2.0,
            offset_y=(scale_y - 1.0) / 2.0,
        )

        return output, transform

    @staticmethod
    def letterbox(
        image: np.ndarray,
        target: ImageSize,
        *,
        padding_value: int = 114,
        interpolation: InterpolationMode = (
            InterpolationMode.LINEAR
        ),
    ) -> tuple[np.ndarray, GeometricTransform]:
        require_opencv()

        source_height, source_width = image.shape[:2]

        scale = min(
            target.width / source_width,
            target.height / source_height,
        )

        resized_width = max(
            1,
            min(
                target.width,
                int(round(source_width * scale)),
            ),
        )

        resized_height = max(
            1,
            min(
                target.height,
                int(round(source_height * scale)),
            ),
        )

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=ImageResizer._interpolation(
                interpolation
            ),
        )

        left = (target.width - resized_width) // 2
        top = (target.height - resized_height) // 2

        right = target.width - resized_width - left
        bottom = target.height - resized_height - top

        output = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=padding_value,
        )

        sx = resized_width / source_width
        sy = resized_height / source_height

        transform = GeometricTransform(
            source_size=ImageSize(
                source_width,
                source_height,
            ),
            target_size=target,
            scale_x=sx,
            scale_y=sy,
            offset_x=left + (sx - 1.0) / 2.0,
            offset_y=top + (sy - 1.0) / 2.0,
        )

        return output, transform

    @staticmethod
    def center_crop_resize(
        image: np.ndarray,
        target: ImageSize,
    ) -> tuple[np.ndarray, GeometricTransform]:
        require_opencv()

        source_height, source_width = image.shape[:2]

        source_ratio = source_width / source_height
        target_ratio = target.width / target.height

        if source_ratio > target_ratio:
            crop_height = source_height
            crop_width = max(
                1,
                int(round(source_height * target_ratio)),
            )
        else:
            crop_width = source_width
            crop_height = max(
                1,
                int(round(source_width / target_ratio)),
            )

        crop_x = (source_width - crop_width) // 2
        crop_y = (source_height - crop_height) // 2

        cropped = image[
            crop_y:crop_y + crop_height,
            crop_x:crop_x + crop_width,
        ]

        output = cv2.resize(
            cropped,
            (target.width, target.height),
        )

        sx = target.width / crop_width
        sy = target.height / crop_height

        transform = GeometricTransform(
            source_size=ImageSize(
                source_width,
                source_height,
            ),
            target_size=target,
            scale_x=sx,
            scale_y=sy,
            offset_x=(
                -crop_x * sx
                + (sx - 1.0) / 2.0
            ),
            offset_y=(
                -crop_y * sy
                + (sy - 1.0) / 2.0
            ),
        )

        return output, transform

    @staticmethod
    def crop_roi(
        image: np.ndarray,
        roi: ImageROI,
    ) -> tuple[np.ndarray, GeometricTransform]:
        height, width = image.shape[:2]

        if (
            roi.x + roi.width > width
            or roi.y + roi.height > height
        ):
            raise ImageValidationError(
                "ROI exceeds image boundaries"
            )

        output = image[
            roi.y:roi.y + roi.height,
            roi.x:roi.x + roi.width,
        ].copy()

        transform = GeometricTransform(
            source_size=ImageSize(width, height),
            target_size=ImageSize(
                roi.width,
                roi.height,
            ),
            offset_x=-float(roi.x),
            offset_y=-float(roi.y),
        )

        return output, transform


# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

class ImageEnhancer:

    @staticmethod
    def brightness_contrast(
        image: np.ndarray,
        *,
        brightness: float = 0.0,
        contrast: float = 1.0,
    ) -> np.ndarray:
        brightness = finite_float(
            brightness,
            "brightness",
        )

        contrast = positive_float(
            contrast,
            "contrast",
        )

        result = (
            image.astype(np.float32) * contrast
            + brightness
        )

        return np.clip(
            result,
            0,
            255,
        ).astype(np.uint8)

    @staticmethod
    def gamma_correction(
        image: np.ndarray,
        gamma: float = 1.0,
    ) -> np.ndarray:
        gamma = positive_float(
            gamma,
            "gamma",
        )

        normalized = (
            image.astype(np.float32) / 255.0
        )

        corrected = np.power(
            np.clip(normalized, 0, 1),
            gamma,
        )

        return np.clip(
            corrected * 255,
            0,
            255,
        ).astype(np.uint8)

    @staticmethod
    def histogram_equalization(
        image: np.ndarray,
        color_space: ImageColorSpace = (
            ImageColorSpace.BGR
        ),
    ) -> np.ndarray:
        require_opencv()

        if image.ndim == 2:
            return cv2.equalizeHist(
                image.astype(np.uint8)
            )

        if color_space == ImageColorSpace.RGB:
            conversion = cv2.COLOR_RGB2LAB
            reverse = cv2.COLOR_LAB2RGB
        elif color_space == ImageColorSpace.BGR:
            conversion = cv2.COLOR_BGR2LAB
            reverse = cv2.COLOR_LAB2BGR
        else:
            raise ImageValidationError(
                "Histogram equalization supports RGB/BGR"
            )

        lab = cv2.cvtColor(
            image,
            conversion,
        )

        lab[:, :, 0] = cv2.equalizeHist(
            lab[:, :, 0]
        )

        return cv2.cvtColor(
            lab,
            reverse,
        )

    @staticmethod
    def clahe(
        image: np.ndarray,
        *,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        color_space: ImageColorSpace = (
            ImageColorSpace.BGR
        ),
    ) -> np.ndarray:
        require_opencv()

        positive_float(
            clip_limit,
            "clip_limit",
        )

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

        if image.ndim == 2:
            return clahe.apply(
                image.astype(np.uint8)
            )

        if color_space == ImageColorSpace.RGB:
            forward = cv2.COLOR_RGB2LAB
            backward = cv2.COLOR_LAB2RGB
        elif color_space == ImageColorSpace.BGR:
            forward = cv2.COLOR_BGR2LAB
            backward = cv2.COLOR_LAB2BGR
        else:
            raise ImageValidationError(
                "CLAHE supports RGB/BGR images"
            )

        lab = cv2.cvtColor(
            image,
            forward,
        )

        lab[:, :, 0] = clahe.apply(
            lab[:, :, 0]
        )

        return cv2.cvtColor(
            lab,
            backward,
        )

    @staticmethod
    def denoise(
        image: np.ndarray,
        method: DenoisingMethod,
        kernel_size: int = 5,
    ) -> np.ndarray:
        if method == DenoisingMethod.NONE:
            return image.copy()

        require_opencv()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ImageValidationError(
                "Denoising kernel must be positive and odd"
            )

        if method == DenoisingMethod.GAUSSIAN:
            return cv2.GaussianBlur(
                image,
                (kernel_size, kernel_size),
                0,
            )

        if method == DenoisingMethod.MEDIAN:
            return cv2.medianBlur(
                image,
                kernel_size,
            )

        if method == DenoisingMethod.BILATERAL:
            return cv2.bilateralFilter(
                image,
                kernel_size,
                75,
                75,
            )

        raise ImageValidationError(
            "Unsupported denoising method"
        )

    @staticmethod
    def sharpen(
        image: np.ndarray,
        strength: float = 1.0,
    ) -> np.ndarray:
        require_opencv()

        strength = positive_float(
            strength,
            "strength",
        )

        blurred = cv2.GaussianBlur(
            image,
            (0, 0),
            1.0,
        )

        return cv2.addWeighted(
            image,
            1.0 + strength,
            blurred,
            -strength,
            0,
        )


# ============================================================
# EDGE DETECTION
# ============================================================

class ImageEdgeDetector:

    @staticmethod
    def detect(
        image: np.ndarray,
        method: EdgeDetectionMethod = (
            EdgeDetectionMethod.CANNY
        ),
        *,
        threshold1: float = 50.0,
        threshold2: float = 150.0,
    ) -> np.ndarray:
        require_opencv()

        if image.ndim == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY,
                )
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGRA2GRAY,
                )
            else:
                gray = image[:, :, 0]
        else:
            gray = image

        gray = np.clip(
            gray,
            0,
            255,
        ).astype(np.uint8)

        if method == EdgeDetectionMethod.CANNY:
            return cv2.Canny(
                gray,
                threshold1,
                threshold2,
            )

        if method == EdgeDetectionMethod.SOBEL:
            gx = cv2.Sobel(
                gray,
                cv2.CV_32F,
                1,
                0,
                ksize=3,
            )

            gy = cv2.Sobel(
                gray,
                cv2.CV_32F,
                0,
                1,
                ksize=3,
            )

            return cv2.magnitude(gx, gy)

        if method == EdgeDetectionMethod.LAPLACIAN:
            return cv2.Laplacian(
                gray,
                cv2.CV_32F,
            )

        raise ImageValidationError(
            "Unsupported edge detection method"
        )


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

class ImageNormalizer:

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    @staticmethod
    def normalize(
        image: np.ndarray,
        mode: NormalizationMode,
        *,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> np.ndarray:
        result = image.astype(np.float32)

        if mode == NormalizationMode.NONE:
            return result

        result /= 255.0

        if mode == NormalizationMode.ZERO_ONE:
            return result

        if mode == NormalizationMode.MINUS_ONE_ONE:
            return result * 2.0 - 1.0

        if mode == NormalizationMode.IMAGENET:
            mean = ImageNormalizer.IMAGENET_MEAN
            std = ImageNormalizer.IMAGENET_STD

        elif mode != NormalizationMode.CUSTOM:
            raise ImageValidationError(
                "Unsupported normalization mode"
            )

        if mean is None or std is None:
            raise ImageValidationError(
                "Custom normalization requires mean and std"
            )

        channels = (
            1 if result.ndim == 2
            else result.shape[2]
        )

        if len(mean) != channels or len(std) != channels:
            raise ImageValidationError(
                "Normalization channel count mismatch"
            )

        mean_array = np.asarray(
            mean,
            dtype=np.float32,
        )

        std_array = np.asarray(
            std,
            dtype=np.float32,
        )

        if (
            not np.all(np.isfinite(mean_array))
            or not np.all(np.isfinite(std_array))
            or np.any(std_array <= 0)
        ):
            raise ImageValidationError(
                "Invalid normalization statistics"
            )

        if result.ndim == 2:
            return (
                result - mean_array[0]
            ) / std_array[0]

        return (
            result - mean_array
        ) / std_array

    @staticmethod
    def to_tensor(
        image: np.ndarray,
        layout: TensorLayout = TensorLayout.NCHW,
    ) -> np.ndarray:
        array = np.asarray(
            image,
            dtype=np.float32,
        )

        if array.ndim == 2:
            array = array[:, :, None]

        if array.ndim != 3:
            raise ImageValidationError(
                "Expected HWC image"
            )

        if layout == TensorLayout.HWC:
            return np.ascontiguousarray(array)

        if layout == TensorLayout.CHW:
            return np.ascontiguousarray(
                array.transpose(2, 0, 1)
            )

        if layout == TensorLayout.NHWC:
            return np.ascontiguousarray(
                array[None, ...]
            )

        if layout == TensorLayout.NCHW:
            return np.ascontiguousarray(
                array.transpose(2, 0, 1)[None, ...]
            )

        raise ImageValidationError(
            "Unsupported tensor layout"
        )


# ============================================================
# CAMERA UNDISTORTION
# ============================================================

@dataclass(frozen=True)
class LensDistortion:
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def coefficients(self) -> np.ndarray:
        values = (
            self.k1,
            self.k2,
            self.p1,
            self.p2,
            self.k3,
        )

        if not all(math.isfinite(v) for v in values):
            raise ImageCalibrationError(
                "Distortion coefficients must be finite"
            )

        return np.asarray(
            values,
            dtype=np.float64,
        )


class ImageUndistorter:

    @staticmethod
    def undistort(
        image: np.ndarray,
        intrinsics: CameraIntrinsics,
        distortion: LensDistortion,
    ) -> np.ndarray:
        require_opencv()

        height, width = image.shape[:2]

        if (
            width != intrinsics.width
            or height != intrinsics.height
        ):
            raise ImageCalibrationError(
                "Image resolution does not match calibration"
            )

        return cv2.undistort(
            image,
            intrinsics.matrix(),
            distortion.coefficients(),
        )


# ============================================================
# DEPTH MAP PREPROCESSING
# ============================================================

class DepthImagePreprocessor:

    @staticmethod
    def sanitize(
        depth_map: np.ndarray,
        *,
        minimum_depth_m: float = 0.1,
        maximum_depth_m: float = 20.0,
    ) -> np.ndarray:
        minimum_depth_m = positive_float(
            minimum_depth_m,
            "minimum_depth_m",
        )

        maximum_depth_m = positive_float(
            maximum_depth_m,
            "maximum_depth_m",
        )

        if minimum_depth_m >= maximum_depth_m:
            raise ImageValidationError(
                "Invalid depth range"
            )

        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        ).copy()

        if depth.ndim != 2:
            raise ImageValidationError(
                "Depth map must be two-dimensional"
            )

        valid = (
            np.isfinite(depth)
            & (depth >= minimum_depth_m)
            & (depth <= maximum_depth_m)
        )

        depth[~valid] = np.nan

        return depth

    @staticmethod
    def resize(
        depth_map: np.ndarray,
        target: ImageSize,
    ) -> np.ndarray:
        require_opencv()

        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        )

        if depth.ndim != 2:
            raise ImageValidationError(
                "Depth map must be two-dimensional"
            )

        valid = (
            np.isfinite(depth)
            & (depth > 0)
        )

        values = np.where(
            valid,
            depth,
            0.0,
        ).astype(np.float32)

        resized_values = cv2.resize(
            values,
            (target.width, target.height),
            interpolation=cv2.INTER_LINEAR,
        )

        resized_weights = cv2.resize(
            valid.astype(np.float32),
            (target.width, target.height),
            interpolation=cv2.INTER_LINEAR,
        )

        result = np.full(
            (target.height, target.width),
            np.nan,
            dtype=np.float32,
        )

        supported = resized_weights >= 0.999

        result[supported] = (
            resized_values[supported]
            / resized_weights[supported]
        )

        return result

    @staticmethod
    def median_filter(
        depth_map: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        require_opencv()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ImageValidationError(
                "Median kernel must be positive and odd"
            )

        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        )

        valid = (
            np.isfinite(depth)
            & (depth > 0)
        )

        working = np.where(
            valid,
            depth,
            0.0,
        ).astype(np.float32)

        filtered = cv2.medianBlur(
            working,
            kernel_size,
        )

        filtered[~valid] = np.nan

        return filtered


# ============================================================
# SEGMENTATION MASK PREPROCESSING
# ============================================================

class SegmentationMaskPreprocessor:

    @staticmethod
    def resize(
        mask: np.ndarray,
        target: ImageSize,
    ) -> np.ndarray:
        require_opencv()

        ImageValidator.validate_mask(mask)

        return cv2.resize(
            mask,
            (target.width, target.height),
            interpolation=cv2.INTER_NEAREST,
        )

    @staticmethod
    def binary_mask(
        mask: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        threshold = finite_float(
            threshold,
            "threshold",
        )

        return (
            np.asarray(mask) >= threshold
        ).astype(np.uint8)


# ============================================================
# IMAGE QUALITY ASSESSMENT
# ============================================================

@dataclass(frozen=True)
class ImageQualityReport:
    quality: ImageQualityLevel

    brightness_mean: float
    contrast_std: float
    sharpness_variance: float

    dark_pixel_ratio: float
    bright_pixel_ratio: float

    warnings: tuple[str, ...]

    evaluated_at: datetime = field(
        default_factory=utc_now
    )


class ImageQualityEvaluator:

    def __init__(
        self,
        config: ImagePreprocessingConfig | None = None,
    ) -> None:
        self.config = config or ImagePreprocessingConfig()

    def evaluate(
        self,
        image: np.ndarray,
        color_space: ImageColorSpace = (
            ImageColorSpace.BGR
        ),
    ) -> ImageQualityReport:
        require_opencv()

        ImageValidator(self.config).validate(image)

        if image.ndim == 2:
            gray = image

        elif image.shape[2] == 1:
            gray = image[:, :, 0]

        else:
            gray = ImageColorConverter.convert(
                image,
                color_space,
                ImageColorSpace.GRAY,
            )

        gray = np.clip(
            gray,
            0,
            255,
        ).astype(np.uint8)

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

        sharpness = float(
            np.var(laplacian)
        )

        dark_ratio = float(
            np.mean(gray <= 15)
        )

        bright_ratio = float(
            np.mean(gray >= 240)
        )

        warnings = []

        if brightness < self.config.minimum_brightness:
            warnings.append(
                "Image appears underexposed"
            )

        if brightness > self.config.maximum_brightness:
            warnings.append(
                "Image appears overexposed"
            )

        if contrast < self.config.minimum_contrast_std:
            warnings.append(
                "Image has low contrast"
            )

        if (
            sharpness
            < self.config.minimum_sharpness_variance
        ):
            warnings.append(
                "Image may be blurry"
            )

        if len(warnings) == 0:
            quality = ImageQualityLevel.GOOD

        elif len(warnings) == 1:
            quality = ImageQualityLevel.ACCEPTABLE

        else:
            quality = ImageQualityLevel.POOR

        return ImageQualityReport(
            quality=quality,
            brightness_mean=brightness,
            contrast_std=contrast,
            sharpness_variance=sharpness,
            dark_pixel_ratio=dark_ratio,
            bright_pixel_ratio=bright_ratio,
            warnings=tuple(warnings),
        )


# ============================================================
# MODEL PREPROCESSING PROFILES
# ============================================================

@dataclass(frozen=True)
class ModelPreprocessingProfile:
    profile_name: str

    target_size: ImageSize

    input_color_space: ImageColorSpace = (
        ImageColorSpace.BGR
    )

    output_color_space: ImageColorSpace = (
        ImageColorSpace.RGB
    )

    resize_mode: ResizeMode = ResizeMode.LETTERBOX

    normalization: NormalizationMode = (
        NormalizationMode.ZERO_ONE
    )

    tensor_layout: TensorLayout = TensorLayout.NCHW

    padding_value: int = 114

    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        validate_identifier(
            self.profile_name,
            "profile_name",
        )

        if not 0 <= self.padding_value <= 255:
            raise ImageValidationError(
                "padding_value must be in [0, 255]"
            )


class ModelPreprocessingProfiles:

    @staticmethod
    def yolo(
        width: int = 640,
        height: int = 640,
    ) -> ModelPreprocessingProfile:
        return ModelPreprocessingProfile(
            profile_name="yolo_detection",
            target_size=ImageSize(width, height),
            resize_mode=ResizeMode.LETTERBOX,
            normalization=NormalizationMode.ZERO_ONE,
            tensor_layout=TensorLayout.NCHW,
        )

    @staticmethod
    def imagenet(
        width: int = 224,
        height: int = 224,
    ) -> ModelPreprocessingProfile:
        return ModelPreprocessingProfile(
            profile_name="imagenet_classification",
            target_size=ImageSize(width, height),
            resize_mode=ResizeMode.CENTER_CROP,
            normalization=NormalizationMode.IMAGENET,
            tensor_layout=TensorLayout.NCHW,
        )

    @staticmethod
    def segmentation(
        width: int = 512,
        height: int = 512,
    ) -> ModelPreprocessingProfile:
        return ModelPreprocessingProfile(
            profile_name="semantic_segmentation",
            target_size=ImageSize(width, height),
            resize_mode=ResizeMode.STRETCH,
            normalization=NormalizationMode.ZERO_ONE,
            tensor_layout=TensorLayout.NCHW,
        )

    @staticmethod
    def vision_transformer(
        width: int = 224,
        height: int = 224,
    ) -> ModelPreprocessingProfile:
        return ModelPreprocessingProfile(
            profile_name="vision_transformer",
            target_size=ImageSize(width, height),
            resize_mode=ResizeMode.CENTER_CROP,
            normalization=NormalizationMode.IMAGENET,
            tensor_layout=TensorLayout.NCHW,
        )


# ============================================================
# PREPROCESSING RESULT
# ============================================================

@dataclass(frozen=True)
class ImagePreprocessingResult:
    processing_id: str

    tenant_id: str
    camera_id: str

    profile_name: str

    original_size: ImageSize
    processed_size: ImageSize

    image: np.ndarray = field(repr=False)
    tensor: np.ndarray = field(repr=False)

    geometric_transform: GeometricTransform

    original_intrinsics: CameraIntrinsics | None = None
    processed_intrinsics: CameraIntrinsics | None = None

    processing_time_ms: float = 0.0

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "processing_id": self.processing_id,
            "tenant_id": self.tenant_id,
            "camera_id": self.camera_id,
            "profile_name": self.profile_name,
            "original_size": json_safe(
                self.original_size
            ),
            "processed_size": json_safe(
                self.processed_size
            ),
            "image_shape": list(self.image.shape),
            "tensor_shape": list(self.tensor.shape),
            "tensor_dtype": str(self.tensor.dtype),
            "geometric_transform": json_safe(
                self.geometric_transform
            ),
            "original_intrinsics": json_safe(
                self.original_intrinsics
            ),
            "processed_intrinsics": json_safe(
                self.processed_intrinsics
            ),
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": json_safe(self.metadata),
        }


# ============================================================
# IMAGE PREPROCESSING PIPELINE
# ============================================================

class ImagePreprocessingPipeline:

    def __init__(
        self,
        profile: ModelPreprocessingProfile,
        config: ImagePreprocessingConfig | None = None,
    ) -> None:
        self.profile = profile
        self.config = config or ImagePreprocessingConfig()

        self.validator = ImageValidator(self.config)

    def process(
        self,
        image: np.ndarray,
        *,
        tenant_id: str,
        camera_id: str,
        intrinsics: CameraIntrinsics | None = None,
    ) -> ImagePreprocessingResult:
        validate_identifier(
            tenant_id,
            "tenant_id",
        )

        validate_identifier(
            camera_id,
            "camera_id",
        )

        started = time.perf_counter()

        self.validator.validate(image)

        original_height, original_width = image.shape[:2]

        original_size = ImageSize(
            original_width,
            original_height,
        )

        if intrinsics is not None:
            if (
                intrinsics.width != original_width
                or intrinsics.height != original_height
            ):
                raise ImageCalibrationError(
                    "Camera intrinsics do not match image"
                )

        working = (
            image.copy()
            if self.config.preserve_input
            else image
        )

        working = ImageColorConverter.convert(
            working,
            self.profile.input_color_space,
            self.profile.output_color_space,
        )

        target = self.profile.target_size

        if self.profile.resize_mode == ResizeMode.STRETCH:
            processed, transform = ImageResizer.resize(
                working,
                target,
                self.config.default_interpolation,
            )

        elif self.profile.resize_mode == ResizeMode.LETTERBOX:
            processed, transform = ImageResizer.letterbox(
                working,
                target,
                padding_value=self.profile.padding_value,
                interpolation=self.config.default_interpolation,
            )

        elif self.profile.resize_mode == ResizeMode.CENTER_CROP:
            processed, transform = (
                ImageResizer.center_crop_resize(
                    working,
                    target,
                )
            )

        else:
            raise ImageValidationError(
                "Unsupported resize mode"
            )

        normalized = ImageNormalizer.normalize(
            processed,
            self.profile.normalization,
            mean=self.profile.mean,
            std=self.profile.std,
        )

        tensor = ImageNormalizer.to_tensor(
            normalized,
            self.profile.tensor_layout,
        )

        processed_intrinsics = (
            transform.transform_intrinsics(intrinsics)
            if intrinsics is not None
            else None
        )

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        return ImagePreprocessingResult(
            processing_id=generate_id(),
            tenant_id=tenant_id,
            camera_id=camera_id,
            profile_name=self.profile.profile_name,
            original_size=original_size,
            processed_size=target,
            image=processed,
            tensor=tensor,
            geometric_transform=transform,
            original_intrinsics=intrinsics,
            processed_intrinsics=processed_intrinsics,
            processing_time_ms=elapsed_ms,
            metadata={
                "resize_mode": self.profile.resize_mode.value,
                "normalization": (
                    self.profile.normalization.value
                ),
                "tensor_layout": (
                    self.profile.tensor_layout.value
                ),
                "output_color_space": (
                    self.profile.output_color_space.value
                ),
            },
        )

    def process_batch(
        self,
        images: Sequence[np.ndarray],
        *,
        tenant_id: str,
        camera_id: str,
        intrinsics: CameraIntrinsics | None = None,
    ) -> tuple[ImagePreprocessingResult, ...]:
        if len(images) > self.config.maximum_batch_size:
            raise ImageCapacityError(
                "Batch size exceeds configured limit"
            )

        return tuple(
            self.process(
                image,
                tenant_id=tenant_id,
                camera_id=camera_id,
                intrinsics=intrinsics,
            )
            for image in images
        )


# ============================================================
# PREPROCESSING STATISTICS
# ============================================================

@dataclass(frozen=True)
class ImageProcessingStatistics:
    total_images_processed: int
    total_processing_errors: int

    average_processing_time_ms: float
    maximum_processing_time_ms: float

    camera_count: int


# ============================================================
# TENANT-AWARE PREPROCESSING SERVICE
# ============================================================

class RoboticsImagePreprocessingService:

    def __init__(
        self,
        config: ImagePreprocessingConfig | None = None,
    ) -> None:
        self.config = config or ImagePreprocessingConfig()

        self._profiles: dict[
            tuple[str, str],
            ModelPreprocessingProfile,
        ] = {}

        self._events: deque[dict[str, Any]] = deque(
            maxlen=self.config.maximum_history_events
        )

        self._processing_times: deque[float] = deque(
            maxlen=self.config.maximum_history_events
        )

        self._processed_count = 0
        self._error_count = 0

        self._cameras: set[tuple[str, str]] = set()

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: ImageAccessContext,
        tenant_id: str,
        camera_id: str,
        permission: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise ImageAuthorizationError(
                "Cross-tenant access denied"
            )

        if camera_id not in access.allowed_camera_ids:
            raise ImageAuthorizationError(
                "Camera access denied"
            )

        if not getattr(access, permission, False):
            raise ImageAuthorizationError(
                f"Missing permission: {permission}"
            )

    def register_profile(
        self,
        profile: ModelPreprocessingProfile,
        *,
        access: ImageAccessContext,
    ) -> None:
        if not access.can_manage:
            raise ImageAuthorizationError(
                "Profile management permission required"
            )

        key = (
            access.tenant_id,
            profile.profile_name,
        )

        with self._lock:
            self._profiles[key] = profile

    def get_profile(
        self,
        profile_name: str,
        *,
        access: ImageAccessContext,
    ) -> ModelPreprocessingProfile:
        if not access.can_view:
            raise ImageAuthorizationError(
                "Profile view permission required"
            )

        key = (
            access.tenant_id,
            profile_name,
        )

        with self._lock:
            profile = self._profiles.get(key)

            if profile is None:
                raise ImageValidationError(
                    "Preprocessing profile not found"
                )

            return profile

    def process(
        self,
        image: np.ndarray,
        *,
        camera_id: str,
        profile_name: str,
        access: ImageAccessContext,
        intrinsics: CameraIntrinsics | None = None,
    ) -> ImagePreprocessingResult:
        self._authorize(
            access,
            access.tenant_id,
            camera_id,
            "can_process",
        )

        key = (
            access.tenant_id,
            profile_name,
        )

        with self._lock:
            profile = self._profiles.get(key)

        if profile is None:
            raise ImageValidationError(
                "Preprocessing profile not registered"
            )

        pipeline = ImagePreprocessingPipeline(
            profile,
            self.config,
        )

        try:
            result = pipeline.process(
                image,
                tenant_id=access.tenant_id,
                camera_id=camera_id,
                intrinsics=intrinsics,
            )

        except Exception:
            with self._lock:
                self._error_count += 1

            raise

        with self._lock:
            self._processed_count += 1

            self._processing_times.append(
                result.processing_time_ms
            )

            self._cameras.add(
                (
                    access.tenant_id,
                    camera_id,
                )
            )

            self._events.append({
                "event_id": generate_id(),
                "event_type": "image_preprocessed",
                "tenant_id": access.tenant_id,
                "camera_id": camera_id,
                "user_id": access.user_id,
                "profile_name": profile_name,
                "processing_id": result.processing_id,
                "processing_time_ms": (
                    result.processing_time_ms
                ),
                "timestamp": utc_now().isoformat(),
            })

        return result

    def statistics(
        self,
        *,
        access: ImageAccessContext,
    ) -> ImageProcessingStatistics:
        if not access.can_view:
            raise ImageAuthorizationError(
                "View permission required"
            )

        with self._lock:
            times = tuple(self._processing_times)

            camera_count = sum(
                1
                for tenant_id, camera_id in self._cameras
                if tenant_id == access.tenant_id
                and camera_id in access.allowed_camera_ids
            )

            return ImageProcessingStatistics(
                total_images_processed=self._processed_count,
                total_processing_errors=self._error_count,
                average_processing_time_ms=(
                    float(np.mean(times))
                    if times
                    else 0.0
                ),
                maximum_processing_time_ms=(
                    max(times)
                    if times
                    else 0.0
                ),
                camera_count=camera_count,
            )

    def audit_history(
        self,
        *,
        access: ImageAccessContext,
    ) -> tuple[dict[str, Any], ...]:
        if not access.can_view:
            raise ImageAuthorizationError(
                "View permission required"
            )

        with self._lock:
            return tuple(
                dict(event)
                for event in self._events
                if event["tenant_id"] == access.tenant_id
                and event["camera_id"]
                in access.allowed_camera_ids
            )

    @staticmethod
    def export_metadata_json(
        result: ImagePreprocessingResult,
    ) -> str:
        return json.dumps(
            result.metadata_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestImagePreprocessing(unittest.TestCase):

    def setUp(self) -> None:
        self.image = np.zeros(
            (120, 160, 3),
            dtype=np.uint8,
        )

        self.image[:, :, 0] = 80
        self.image[:, :, 1] = 120
        self.image[:, :, 2] = 200

        self.config = ImagePreprocessingConfig()

        self.access = ImageAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_camera_ids=frozenset({
                "CAM-001",
            }),
            can_process=True,
            can_view=True,
            can_manage=True,
        )

    def test_image_validation(self) -> None:
        validated = ImageValidator(
            self.config
        ).validate(self.image)

        self.assertEqual(
            validated.shape,
            (120, 160, 3),
        )

    def test_invalid_image(self) -> None:
        with self.assertRaises(ImageValidationError):
            ImageValidator().validate(
                np.zeros((0, 0, 3))
            )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV unavailable",
    )
    def test_color_conversion(self) -> None:
        rgb = ImageColorConverter.convert(
            self.image,
            ImageColorSpace.BGR,
            ImageColorSpace.RGB,
        )

        self.assertEqual(
            int(rgb[0, 0, 0]),
            200,
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV unavailable",
    )
    def test_resize(self) -> None:
        resized, transform = ImageResizer.resize(
            self.image,
            ImageSize(320, 240),
        )

        self.assertEqual(
            resized.shape,
            (240, 320, 3),
        )

        self.assertAlmostEqual(
            transform.scale_x,
            2.0,
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV unavailable",
    )
    def test_letterbox(self) -> None:
        output, transform = ImageResizer.letterbox(
            self.image,
            ImageSize(224, 224),
        )

        self.assertEqual(
            output.shape,
            (224, 224, 3),
        )

        self.assertGreater(
            transform.offset_y,
            0,
        )

    def test_bounding_box_roundtrip(self) -> None:
        transform = GeometricTransform(
            source_size=ImageSize(100, 100),
            target_size=ImageSize(200, 200),
            scale_x=2.0,
            scale_y=2.0,
            offset_x=0.5,
            offset_y=0.5,
        )

        original = BoundingBox(
            10,
            20,
            30,
            40,
        )

        transformed = transform.transform_box(
            original
        )

        recovered = transform.inverse_box(
            transformed
        )

        self.assertAlmostEqual(
            recovered.x1,
            original.x1,
        )

        self.assertAlmostEqual(
            recovered.y2,
            original.y2,
        )

    def test_normalization(self) -> None:
        normalized = ImageNormalizer.normalize(
            self.image,
            NormalizationMode.ZERO_ONE,
        )

        self.assertEqual(
            normalized.dtype,
            np.float32,
        )

        self.assertLessEqual(
            float(np.max(normalized)),
            1.0,
        )

    def test_tensor_conversion(self) -> None:
        tensor = ImageNormalizer.to_tensor(
            self.image,
            TensorLayout.NCHW,
        )

        self.assertEqual(
            tensor.shape,
            (1, 3, 120, 160),
        )

    def test_gamma_correction(self) -> None:
        output = ImageEnhancer.gamma_correction(
            self.image,
            gamma=1.5,
        )

        self.assertEqual(
            output.shape,
            self.image.shape,
        )

    def test_brightness_contrast(self) -> None:
        output = ImageEnhancer.brightness_contrast(
            self.image,
            brightness=10,
            contrast=1.2,
        )

        self.assertEqual(
            output.dtype,
            np.uint8,
        )

    def test_depth_sanitization(self) -> None:
        depth = np.array(
            [
                [1.0, 2.0],
                [0.0, np.inf],
            ],
            dtype=np.float32,
        )

        result = DepthImagePreprocessor.sanitize(
            depth
        )

        self.assertTrue(
            np.isnan(result[1, 0])
        )

        self.assertTrue(
            np.isnan(result[1, 1])
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV unavailable",
    )
    def test_segmentation_resize(self) -> None:
        mask = np.array(
            [
                [0, 1],
                [2, 3],
            ],
            dtype=np.uint8,
        )

        resized = SegmentationMaskPreprocessor.resize(
            mask,
            ImageSize(8, 8),
        )

        self.assertTrue(
            set(np.unique(resized)).issubset({
                0, 1, 2, 3,
            })
        )

    def test_camera_intrinsic_transform(self) -> None:
        intrinsics = CameraIntrinsics(
            width=160,
            height=120,
            fx=100,
            fy=100,
            cx=80,
            cy=60,
        )

        transform = GeometricTransform(
            source_size=ImageSize(160, 120),
            target_size=ImageSize(320, 240),
            scale_x=2,
            scale_y=2,
        )

        updated = transform.transform_intrinsics(
            intrinsics
        )

        self.assertAlmostEqual(
            updated.fx,
            200,
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV unavailable",
    )
    def test_pipeline(self) -> None:
        profile = ModelPreprocessingProfiles.yolo(
            224,
            224,
        )

        pipeline = ImagePreprocessingPipeline(
            profile
        )

        result = pipeline.process(
            self.image,
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
        )

        self.assertEqual(
            result.tensor.shape,
            (1, 3, 224, 224),
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV unavailable",
    )
    def test_service(self) -> None:
        service = RoboticsImagePreprocessingService()

        profile = ModelPreprocessingProfiles.imagenet()

        service.register_profile(
            profile,
            access=self.access,
        )

        result = service.process(
            self.image,
            camera_id="CAM-001",
            profile_name=profile.profile_name,
            access=self.access,
        )

        self.assertEqual(
            result.tensor.shape,
            (1, 3, 224, 224),
        )

    def test_tenant_isolation(self) -> None:
        other = ImageAccessContext(
            tenant_id="tenant_beta",
            user_id="other",
            allowed_camera_ids=frozenset({
                "CAM-001",
            }),
            can_process=True,
            can_view=True,
        )

        with self.assertRaises(ImageAuthorizationError):
            RoboticsImagePreprocessingService._authorize(
                other,
                "tenant_alpha",
                "CAM-001",
                "can_process",
            )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV unavailable",
    )
    def test_metadata_export(self) -> None:
        profile = ModelPreprocessingProfiles.yolo(
            128,
            128,
        )

        result = ImagePreprocessingPipeline(
            profile
        ).process(
            self.image,
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
        )

        payload = (
            RoboticsImagePreprocessingService
            .export_metadata_json(result)
        )

        self.assertIn(
            "CAM-001",
            payload,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Image Preprocessing and Vision Pipeline\n"
    )

    if cv2 is None:
        print(
            "OpenCV is not installed. "
            "Install opencv-python for the full demonstration."
        )
        return

    access = ImageAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_camera_ids=frozenset({
            "CAM-FRONT-001",
        }),
        can_process=True,
        can_view=True,
        can_manage=True,
    )

    image = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    image[:, :, 0] = 60
    image[:, :, 1] = 110
    image[:, :, 2] = 170

    cv2.rectangle(
        image,
        (200, 150),
        (440, 350),
        (30, 220, 80),
        -1,
    )

    intrinsics = CameraIntrinsics(
        width=640,
        height=480,
        fx=520.0,
        fy=520.0,
        cx=320.0,
        cy=240.0,
    )

    service = RoboticsImagePreprocessingService()

    profile = ModelPreprocessingProfiles.yolo(
        640,
        640,
    )

    service.register_profile(
        profile,
        access=access,
    )

    result = service.process(
        image,
        camera_id="CAM-FRONT-001",
        profile_name=profile.profile_name,
        access=access,
        intrinsics=intrinsics,
    )

    print("PREPROCESSING RESULT\n")

    print(
        service.export_metadata_json(
            result
        )
    )

    original_box = BoundingBox(
        x1=200,
        y1=150,
        x2=440,
        y2=350,
    )

    model_box = (
        result.geometric_transform.transform_box(
            original_box
        )
    )

    recovered_box = (
        result.geometric_transform.inverse_box(
            model_box
        )
    )

    print("\nBOUNDING BOX TRANSFORMATION\n")

    print(
        json.dumps(
            {
                "original": json_safe(original_box),
                "model_coordinates": json_safe(model_box),
                "recovered": json_safe(recovered_box),
            },
            indent=2,
        )
    )

    quality = ImageQualityEvaluator().evaluate(
        image,
        ImageColorSpace.BGR,
    )

    print("\nIMAGE QUALITY\n")

    print(
        json.dumps(
            json_safe(quality),
            indent=2,
        )
    )

    statistics = service.statistics(
        access=access
    )

    print("\nPROCESSING STATISTICS\n")

    print(
        json.dumps(
            json_safe(statistics),
            indent=2,
        )
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestImagePreprocessing
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
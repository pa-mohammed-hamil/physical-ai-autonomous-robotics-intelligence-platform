"""Depth Estimation module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/depth_estimation.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: depth_estimation.py

Depth estimation and 3D perception for autonomous robotics.

Features:
    - Monocular depth estimation (optional ONNX model)
    - Stereo depth estimation using OpenCV StereoSGBM
    - RGB-D camera depth processing
    - Metric depth calculation from stereo disparity
    - Depth normalization and visualization
    - Depth map filtering
    - Depth uncertainty estimation
    - Depth-to-3D point-cloud conversion
    - 3D point projection
    - Obstacle distance estimation
    - Region-of-interest depth analysis
    - Camera intrinsic integration
    - Depth quality assessment
    - Depth frame history
    - Multi-camera isolation
    - Tenant-based access control
    - Thread-safe processing
    - JSON metadata export
    - Synthetic depth generation
    - Unit tests

Dependencies:
    Python 3.11+
    numpy
    Optional: opencv-python, onnxruntime

Install:
    pip install numpy opencv-python onnxruntime

Important:
    Monocular neural-network depth may be relative rather than metric.
    Metric scale requires a model trained for metric depth or a validated
    external scale reference.

    Stereo depth requires calibrated, rectified camera images and a
    measured baseline.

    Depth estimates must not be treated as independently sufficient
    for collision avoidance or safety-critical robot control.
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
from typing import Any, Sequence

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "depth_estimation.py requires numpy. "
        "Install using: pip install numpy"
    ) from exc

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

class DepthEstimationError(Exception):
    """Base depth estimation error."""


class DepthValidationError(DepthEstimationError):
    """Invalid depth configuration or input."""


class DepthDependencyError(DepthEstimationError):
    """Required optional dependency unavailable."""


class DepthProcessingError(DepthEstimationError):
    """Depth processing failed."""


class DepthAuthorizationError(DepthEstimationError):
    """Depth processing access denied."""


class DepthModelError(DepthEstimationError):
    """Neural depth model failed."""


class DepthCalibrationError(DepthEstimationError):
    """Invalid or missing camera calibration."""


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
        raise DepthValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise DepthValidationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def positive_float(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise DepthValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result) or result <= 0:
        raise DepthValidationError(
            f"{name} must be finite and positive"
        )

    return result


def validate_image(
    image: np.ndarray,
    name: str = "image",
) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise DepthValidationError(
            f"{name} must be a NumPy array"
        )

    if image.ndim not in (2, 3):
        raise DepthValidationError(
            f"{name} must be a 2D or 3D array"
        )

    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise DepthValidationError(
            f"{name} cannot be empty"
        )

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise DepthValidationError(
            f"{name} must have 1, 3, or 4 channels"
        )

    if not np.issubdtype(image.dtype, np.number):
        raise DepthValidationError(
            f"{name} must contain numeric pixels"
        )

    if not np.all(np.isfinite(image)):
        raise DepthValidationError(
            f"{name} contains nonfinite values"
        )

    return image


def require_opencv() -> None:
    if cv2 is None:
        raise DepthDependencyError(
            "OpenCV required: pip install opencv-python"
        )


# ============================================================
# ENUMERATIONS
# ============================================================

class DepthEstimationMethod(str, Enum):
    MONOCULAR = "monocular"
    STEREO = "stereo"
    RGBD = "rgbd"
    SYNTHETIC = "synthetic"


class DepthScaleType(str, Enum):
    METRIC = "metric"
    RELATIVE = "relative"


class DepthQuality(str, Enum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


class DepthFilterType(str, Enum):
    NONE = "none"
    MEDIAN = "median"
    GAUSSIAN = "gaussian"
    BILATERAL = "bilateral"


# ============================================================
# CAMERA INTRINSICS
# ============================================================

@dataclass(frozen=True)
class DepthCameraIntrinsics:
    width: int
    height: int

    fx: float
    fy: float

    cx: float
    cy: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
        ):
            raise DepthCalibrationError(
                "Invalid camera resolution"
            )

        positive_float(self.fx, "fx")
        positive_float(self.fy, "fy")

        if not (
            math.isfinite(self.cx)
            and math.isfinite(self.cy)
        ):
            raise DepthCalibrationError(
                "Invalid camera principal point"
            )

    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DepthEstimationConfig:
    minimum_depth_m: float = 0.1
    maximum_depth_m: float = 20.0

    confidence_threshold: float = 0.5

    filter_type: DepthFilterType = DepthFilterType.MEDIAN
    filter_kernel_size: int = 5

    maximum_history_frames: int = 30
    maximum_point_cloud_points: int = 100000

    minimum_valid_pixel_ratio: float = 0.5

    obstacle_distance_threshold_m: float = 1.0

    def __post_init__(self) -> None:
        positive_float(
            self.minimum_depth_m,
            "minimum_depth_m",
        )

        positive_float(
            self.maximum_depth_m,
            "maximum_depth_m",
        )

        if self.minimum_depth_m >= self.maximum_depth_m:
            raise DepthValidationError(
                "Minimum depth must be below maximum depth"
            )

        if not 0 <= self.confidence_threshold <= 1:
            raise DepthValidationError(
                "confidence_threshold must be in [0, 1]"
            )

        if not 0 <= self.minimum_valid_pixel_ratio <= 1:
            raise DepthValidationError(
                "minimum_valid_pixel_ratio must be in [0, 1]"
            )

        if (
            self.filter_kernel_size <= 0
            or self.filter_kernel_size % 2 == 0
        ):
            raise DepthValidationError(
                "Filter kernel size must be positive and odd"
            )

        if (
            self.maximum_history_frames <= 0
            or self.maximum_point_cloud_points <= 0
        ):
            raise DepthValidationError(
                "History and point-cloud limits must be positive"
            )

        positive_float(
            self.obstacle_distance_threshold_m,
            "obstacle_distance_threshold_m",
        )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class DepthAccessContext:
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
# DEPTH FRAME
# ============================================================

@dataclass(frozen=True)
class DepthFrame:
    frame_id: str

    tenant_id: str
    camera_id: str

    method: DepthEstimationMethod
    scale_type: DepthScaleType

    depth_map: np.ndarray = field(repr=False)

    timestamp: datetime = field(
        default_factory=utc_now
    )

    confidence_map: np.ndarray | None = field(
        default=None,
        repr=False,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.frame_id,
            "frame_id",
        )

        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.camera_id,
            "camera_id",
        )

        if self.depth_map.ndim != 2:
            raise DepthValidationError(
                "Depth map must be two-dimensional"
            )

        if self.confidence_map is not None:
            if (
                self.confidence_map.shape
                != self.depth_map.shape
            ):
                raise DepthValidationError(
                    "Confidence map shape mismatch"
                )

    @property
    def width(self) -> int:
        return int(self.depth_map.shape[1])

    @property
    def height(self) -> int:
        return int(self.depth_map.shape[0])

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "tenant_id": self.tenant_id,
            "camera_id": self.camera_id,
            "method": self.method.value,
            "scale_type": self.scale_type.value,
            "width": self.width,
            "height": self.height,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# ============================================================
# DEPTH QUALITY REPORT
# ============================================================

@dataclass(frozen=True)
class DepthQualityReport:
    quality: DepthQuality

    valid_pixel_ratio: float

    minimum_depth: float | None
    maximum_depth: float | None
    mean_depth: float | None
    median_depth: float | None

    warnings: tuple[str, ...]

    timestamp: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# DEPTH FILTERING
# ============================================================

class DepthMapFilter:

    @staticmethod
    def apply(
        depth_map: np.ndarray,
        filter_type: DepthFilterType,
        kernel_size: int = 5,
    ) -> np.ndarray:
        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        )

        if depth.ndim != 2:
            raise DepthValidationError(
                "Depth map must be 2D"
            )

        if (
            kernel_size <= 0
            or kernel_size % 2 == 0
        ):
            raise DepthValidationError(
                "Kernel size must be positive and odd"
            )

        if filter_type == DepthFilterType.NONE:
            return depth.copy()

        require_opencv()

        valid = (
            np.isfinite(depth)
            & (depth > 0)
        )

        if not np.any(valid):
            return np.full_like(
                depth,
                np.nan,
            )

        working = np.where(
            valid,
            depth,
            0.0,
        ).astype(np.float32)

        if filter_type == DepthFilterType.MEDIAN:
            filtered = cv2.medianBlur(
                working,
                kernel_size,
            )

        elif filter_type == DepthFilterType.GAUSSIAN:
            filtered = cv2.GaussianBlur(
                working,
                (kernel_size, kernel_size),
                0,
            )

        elif filter_type == DepthFilterType.BILATERAL:
            filtered = cv2.bilateralFilter(
                working,
                kernel_size,
                0.1,
                5.0,
            )

        else:
            raise DepthValidationError(
                "Unsupported depth filter"
            )

        # Preserve invalid measurements rather than inventing
        # a valid depth where the sensor provided none.
        filtered[~valid] = np.nan

        return filtered


# ============================================================
# STEREO DEPTH ESTIMATION
# ============================================================

@dataclass(frozen=True)
class StereoDepthConfig:
    baseline_m: float
    focal_length_px: float

    minimum_disparity: int = 0
    number_of_disparities: int = 128
    block_size: int = 5

    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2

    def __post_init__(self) -> None:
        positive_float(
            self.baseline_m,
            "baseline_m",
        )

        positive_float(
            self.focal_length_px,
            "focal_length_px",
        )

        if (
            self.number_of_disparities <= 0
            or self.number_of_disparities % 16 != 0
        ):
            raise DepthValidationError(
                "number_of_disparities must be "
                "a positive multiple of 16"
            )

        if (
            self.block_size < 3
            or self.block_size % 2 == 0
        ):
            raise DepthValidationError(
                "block_size must be odd and at least 3"
            )


class StereoDepthEstimator:

    def __init__(
        self,
        config: StereoDepthConfig,
    ) -> None:
        self.config = config

    @staticmethod
    def disparity_to_depth(
        disparity: np.ndarray,
        focal_length_px: float,
        baseline_m: float,
    ) -> np.ndarray:
        focal = positive_float(
            focal_length_px,
            "focal_length_px",
        )

        baseline = positive_float(
            baseline_m,
            "baseline_m",
        )

        disparity = np.asarray(
            disparity,
            dtype=np.float32,
        )

        if disparity.ndim != 2:
            raise DepthValidationError(
                "Disparity must be a 2D array"
            )

        depth = np.full(
            disparity.shape,
            np.nan,
            dtype=np.float32,
        )

        valid = (
            np.isfinite(disparity)
            & (disparity > 0)
        )

        depth[valid] = (
            focal * baseline / disparity[valid]
        )

        return depth

    def estimate(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        *,
        images_are_rectified: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            depth_map_m, disparity_map_px

        The caller must supply calibrated, rectified images.
        """
        require_opencv()

        if not images_are_rectified:
            raise DepthCalibrationError(
                "Stereo images must be calibrated "
                "and rectified before depth estimation"
            )

        left = validate_image(
            left_image,
            "left_image",
        )

        right = validate_image(
            right_image,
            "right_image",
        )

        if left.shape[:2] != right.shape[:2]:
            raise DepthValidationError(
                "Stereo image resolutions must match"
            )

        def grayscale(image: np.ndarray) -> np.ndarray:
            if image.ndim == 2:
                result = image
            elif image.shape[2] == 1:
                result = image[:, :, 0]
            elif image.shape[2] == 4:
                result = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGRA2GRAY,
                )
            else:
                result = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY,
                )

            return np.clip(
                result,
                0,
                255,
            ).astype(np.uint8)

        left_gray = grayscale(left)
        right_gray = grayscale(right)

        block = self.config.block_size

        matcher = cv2.StereoSGBM_create(
            minDisparity=(
                self.config.minimum_disparity
            ),
            numDisparities=(
                self.config.number_of_disparities
            ),
            blockSize=block,
            P1=8 * block * block,
            P2=32 * block * block,
            uniquenessRatio=(
                self.config.uniqueness_ratio
            ),
            speckleWindowSize=(
                self.config.speckle_window_size
            ),
            speckleRange=(
                self.config.speckle_range
            ),
            disp12MaxDiff=1,
        )

        raw_disparity = matcher.compute(
            left_gray,
            right_gray,
        )

        disparity = (
            raw_disparity.astype(np.float32)
            / 16.0
        )

        depth = self.disparity_to_depth(
            disparity,
            focal_length_px=(
                self.config.focal_length_px
            ),
            baseline_m=(
                self.config.baseline_m
            ),
        )

        return depth, disparity


# ============================================================
# RGB-D DEPTH ESTIMATION
# ============================================================

class RGBDDepthEstimator:

    @staticmethod
    def convert_to_meters(
        raw_depth: np.ndarray,
        depth_scale_m: float,
        invalid_value: float = 0.0,
    ) -> np.ndarray:
        scale = positive_float(
            depth_scale_m,
            "depth_scale_m",
        )

        raw = np.asarray(
            raw_depth,
            dtype=np.float32,
        )

        if raw.ndim != 2:
            raise DepthValidationError(
                "RGB-D depth image must be 2D"
            )

        valid = (
            np.isfinite(raw)
            & (raw > 0)
            & (raw != invalid_value)
        )

        depth = np.full(
            raw.shape,
            np.nan,
            dtype=np.float32,
        )

        depth[valid] = raw[valid] * scale

        return depth


# ============================================================
# MONOCULAR ONNX DEPTH ESTIMATION
# ============================================================

@dataclass(frozen=True)
class MonocularModelConfig:
    model_path: str

    input_width: int = 256
    input_height: int = 256

    output_scale_type: DepthScaleType = (
        DepthScaleType.RELATIVE
    )

    input_color_order: str = "RGB"

    def __post_init__(self) -> None:
        if not self.model_path:
            raise DepthValidationError(
                "model_path required"
            )

        if (
            self.input_width <= 0
            or self.input_height <= 0
        ):
            raise DepthValidationError(
                "Invalid model input resolution"
            )

        if self.input_color_order not in (
            "RGB",
            "BGR",
        ):
            raise DepthValidationError(
                "Unsupported color order"
            )


class MonocularONNXDepthEstimator:
    """
    Generic ONNX depth inference adapter.

    Assumptions:
        - Input tensor shape: [1, 3, H, W]
        - Input normalization: pixel / 255
        - Output: [1, 1, H, W], [1, H, W], or [H, W]
        - Output values increase with estimated depth

    Real models may require ImageNet normalization,
    inverse-depth conversion, custom resizing, or
    architecture-specific postprocessing. Adapt this
    class to the model's documented preprocessing.
    """

    def __init__(
        self,
        config: MonocularModelConfig,
    ) -> None:
        if ort is None:
            raise DepthDependencyError(
                "onnxruntime required"
            )

        require_opencv()

        self.config = config

        try:
            self.session = ort.InferenceSession(
                config.model_path,
                providers=[
                    "CPUExecutionProvider",
                ],
            )
        except Exception as exc:
            raise DepthModelError(
                "Unable to load ONNX depth model"
            ) from exc

        inputs = self.session.get_inputs()

        if len(inputs) != 1:
            raise DepthModelError(
                "Expected a single model input"
            )

        self.input_name = inputs[0].name

    def estimate(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        image = validate_image(
            image,
            "image",
        )

        if image.ndim != 3 or image.shape[2] != 3:
            raise DepthValidationError(
                "Monocular model requires a 3-channel image"
            )

        original_height, original_width = (
            image.shape[:2]
        )

        resized = cv2.resize(
            image,
            (
                self.config.input_width,
                self.config.input_height,
            ),
        )

        if self.config.input_color_order == "BGR":
            resized = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2RGB,
            )

        tensor = (
            resized.astype(np.float32)
            / 255.0
        )

        tensor = np.transpose(
            tensor,
            (2, 0, 1),
        )[None, ...]

        try:
            outputs = self.session.run(
                None,
                {
                    self.input_name: tensor,
                },
            )
        except Exception as exc:
            raise DepthModelError(
                "ONNX depth inference failed"
            ) from exc

        if not outputs:
            raise DepthModelError(
                "Model returned no outputs"
            )

        prediction = np.asarray(
            outputs[0],
            dtype=np.float32,
        ).squeeze()

        if prediction.ndim != 2:
            raise DepthModelError(
                "Model output is not a 2D depth map"
            )

        depth = cv2.resize(
            prediction,
            (
                original_width,
                original_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        depth = depth.astype(
            np.float32
        )

        depth[
            ~np.isfinite(depth)
        ] = np.nan

        depth[
            depth <= 0
        ] = np.nan

        return depth


# ============================================================
# DEPTH QUALITY EVALUATOR
# ============================================================

class DepthQualityEvaluator:

    def __init__(
        self,
        config: DepthEstimationConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        depth_map: np.ndarray,
        scale_type: DepthScaleType,
    ) -> DepthQualityReport:
        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        )

        if depth.ndim != 2 or depth.size == 0:
            raise DepthValidationError(
                "Invalid depth map"
            )

        valid = (
            np.isfinite(depth)
            & (depth > 0)
        )

        if scale_type == DepthScaleType.METRIC:
            valid &= (
                depth >= self.config.minimum_depth_m
            )

            valid &= (
                depth <= self.config.maximum_depth_m
            )

        valid_ratio = float(
            np.count_nonzero(valid) / depth.size
        )

        warnings = []

        if not np.any(valid):
            return DepthQualityReport(
                quality=DepthQuality.INVALID,
                valid_pixel_ratio=0.0,
                minimum_depth=None,
                maximum_depth=None,
                mean_depth=None,
                median_depth=None,
                warnings=(
                    "No valid depth measurements",
                ),
            )

        values = depth[valid]

        if (
            valid_ratio
            < self.config.minimum_valid_pixel_ratio
        ):
            warnings.append(
                "Low valid-depth pixel coverage"
            )

        if scale_type == DepthScaleType.RELATIVE:
            warnings.append(
                "Depth scale is relative, not metric"
            )

        if valid_ratio >= 0.9:
            quality = DepthQuality.GOOD

        elif (
            valid_ratio
            >= self.config.minimum_valid_pixel_ratio
        ):
            quality = DepthQuality.ACCEPTABLE

        else:
            quality = DepthQuality.POOR

        return DepthQualityReport(
            quality=quality,
            valid_pixel_ratio=valid_ratio,
            minimum_depth=float(np.min(values)),
            maximum_depth=float(np.max(values)),
            mean_depth=float(np.mean(values)),
            median_depth=float(np.median(values)),
            warnings=tuple(warnings),
        )


# ============================================================
# DEPTH TO POINT CLOUD
# ============================================================

@dataclass(frozen=True)
class PointCloud:
    points_xyz: np.ndarray = field(
        repr=False
    )

    camera_id: str = ""

    frame_id: str = ""

    coordinate_frame: str = "camera"

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        if (
            self.points_xyz.ndim != 2
            or self.points_xyz.shape[1] != 3
        ):
            raise DepthValidationError(
                "Point cloud must have shape [N, 3]"
            )

    @property
    def point_count(self) -> int:
        return int(self.points_xyz.shape[0])


class DepthPointCloudConverter:

    @staticmethod
    def depth_to_point_cloud(
        depth_map: np.ndarray,
        intrinsics: DepthCameraIntrinsics,
        *,
        camera_id: str = "",
        frame_id: str = "",
        stride: int = 2,
        maximum_points: int = 100000,
        minimum_depth_m: float = 0.1,
        maximum_depth_m: float = 20.0,
    ) -> PointCloud:
        if stride <= 0 or maximum_points <= 0:
            raise DepthValidationError(
                "Stride and maximum_points must be positive"
            )

        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        )

        if depth.shape != (
            intrinsics.height,
            intrinsics.width,
        ):
            raise DepthCalibrationError(
                "Depth resolution and intrinsics mismatch"
            )

        ys, xs = np.mgrid[
            0:intrinsics.height:stride,
            0:intrinsics.width:stride,
        ]

        sampled_depth = depth[::stride, ::stride]

        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth >= minimum_depth_m)
            & (sampled_depth <= maximum_depth_m)
        )

        z = sampled_depth[valid]

        x = (
            (xs[valid] - intrinsics.cx)
            * z
            / intrinsics.fx
        )

        y = (
            (ys[valid] - intrinsics.cy)
            * z
            / intrinsics.fy
        )

        points = np.column_stack(
            (x, y, z)
        ).astype(np.float32)

        if len(points) > maximum_points:
            indices = np.linspace(
                0,
                len(points) - 1,
                maximum_points,
                dtype=np.int64,
            )

            points = points[indices]

        return PointCloud(
            points_xyz=points,
            camera_id=camera_id,
            frame_id=frame_id,
        )

    @staticmethod
    def pixel_to_point(
        pixel_x: int,
        pixel_y: int,
        depth_m: float,
        intrinsics: DepthCameraIntrinsics,
    ) -> tuple[float, float, float]:
        depth_m = positive_float(
            depth_m,
            "depth_m",
        )

        if not (
            0 <= pixel_x < intrinsics.width
            and 0 <= pixel_y < intrinsics.height
        ):
            raise DepthValidationError(
                "Pixel outside camera image"
            )

        x = (
            (pixel_x - intrinsics.cx)
            * depth_m
            / intrinsics.fx
        )

        y = (
            (pixel_y - intrinsics.cy)
            * depth_m
            / intrinsics.fy
        )

        return (
            float(x),
            float(y),
            float(depth_m),
        )


# ============================================================
# DEPTH REGION ANALYSIS
# ============================================================

@dataclass(frozen=True)
class DepthRegion:
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
            raise DepthValidationError(
                "Invalid depth region"
            )


@dataclass(frozen=True)
class DepthRegionStatistics:
    valid_pixel_count: int

    valid_pixel_ratio: float

    minimum_depth: float | None
    maximum_depth: float | None
    mean_depth: float | None
    median_depth: float | None


class DepthRegionAnalyzer:

    @staticmethod
    def analyze(
        depth_map: np.ndarray,
        region: DepthRegion,
        *,
        minimum_depth: float = 0.0,
        maximum_depth: float = math.inf,
    ) -> DepthRegionStatistics:
        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        )

        if depth.ndim != 2:
            raise DepthValidationError(
                "Depth map must be 2D"
            )

        height, width = depth.shape

        if (
            region.x + region.width > width
            or region.y + region.height > height
        ):
            raise DepthValidationError(
                "Region exceeds depth map bounds"
            )

        roi = depth[
            region.y:region.y + region.height,
            region.x:region.x + region.width,
        ]

        valid = (
            np.isfinite(roi)
            & (roi > 0)
            & (roi >= minimum_depth)
            & (roi <= maximum_depth)
        )

        count = int(np.count_nonzero(valid))

        ratio = count / roi.size

        if count == 0:
            return DepthRegionStatistics(
                valid_pixel_count=0,
                valid_pixel_ratio=0.0,
                minimum_depth=None,
                maximum_depth=None,
                mean_depth=None,
                median_depth=None,
            )

        values = roi[valid]

        return DepthRegionStatistics(
            valid_pixel_count=count,
            valid_pixel_ratio=float(ratio),
            minimum_depth=float(np.min(values)),
            maximum_depth=float(np.max(values)),
            mean_depth=float(np.mean(values)),
            median_depth=float(np.median(values)),
        )


# ============================================================
# OBSTACLE DISTANCE ESTIMATION
# ============================================================

@dataclass(frozen=True)
class ObstacleDistanceResult:
    minimum_distance_m: float | None
    median_distance_m: float | None

    obstacle_detected: bool

    threshold_m: float

    valid_pixel_ratio: float

    advisory_only: bool = True


class DepthObstacleAnalyzer:

    @staticmethod
    def analyze(
        depth_map: np.ndarray,
        *,
        threshold_m: float = 1.0,
        region: DepthRegion | None = None,
        minimum_valid_pixel_ratio: float = 0.5,
    ) -> ObstacleDistanceResult:
        positive_float(
            threshold_m,
            "threshold_m",
        )

        if not 0 <= minimum_valid_pixel_ratio <= 1:
            raise DepthValidationError(
                "Invalid minimum_valid_pixel_ratio"
            )

        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        )

        if region is None:
            region = DepthRegion(
                x=0,
                y=0,
                width=depth.shape[1],
                height=depth.shape[0],
            )

        stats = DepthRegionAnalyzer.analyze(
            depth,
            region,
        )

        sufficient_coverage = (
            stats.valid_pixel_ratio
            >= minimum_valid_pixel_ratio
        )

        detected = (
            sufficient_coverage
            and stats.minimum_depth is not None
            and stats.minimum_depth < threshold_m
        )

        return ObstacleDistanceResult(
            minimum_distance_m=stats.minimum_depth,
            median_distance_m=stats.median_depth,
            obstacle_detected=detected,
            threshold_m=threshold_m,
            valid_pixel_ratio=stats.valid_pixel_ratio,
        )


# ============================================================
# DEPTH VISUALIZATION
# ============================================================

class DepthVisualizer:

    @staticmethod
    def normalize(
        depth_map: np.ndarray,
        *,
        minimum_depth: float | None = None,
        maximum_depth: float | None = None,
    ) -> np.ndarray:
        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        )

        if depth.ndim != 2:
            raise DepthValidationError(
                "Depth map must be 2D"
            )

        valid = (
            np.isfinite(depth)
            & (depth > 0)
        )

        result = np.zeros(
            depth.shape,
            dtype=np.uint8,
        )

        if not np.any(valid):
            return result

        values = depth[valid]

        low = (
            float(np.min(values))
            if minimum_depth is None
            else float(minimum_depth)
        )

        high = (
            float(np.max(values))
            if maximum_depth is None
            else float(maximum_depth)
        )

        if high <= low:
            return result

        normalized = (
            (depth[valid] - low)
            / (high - low)
        )

        result[valid] = np.clip(
            normalized * 255,
            0,
            255,
        ).astype(np.uint8)

        return result

    @staticmethod
    def colorize(
        depth_map: np.ndarray,
    ) -> np.ndarray:
        normalized = DepthVisualizer.normalize(
            depth_map
        )

        if cv2 is None:
            return np.repeat(
                normalized[:, :, None],
                3,
                axis=2,
            )

        return cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_TURBO,
        )


# ============================================================
# SYNTHETIC DEPTH GENERATOR
# ============================================================

class SyntheticDepthGenerator:

    @staticmethod
    def generate(
        width: int,
        height: int,
        *,
        base_depth_m: float = 3.0,
        obstacle_depth_m: float = 0.8,
    ) -> np.ndarray:
        if width <= 0 or height <= 0:
            raise DepthValidationError(
                "Invalid synthetic depth resolution"
            )

        positive_float(
            base_depth_m,
            "base_depth_m",
        )

        positive_float(
            obstacle_depth_m,
            "obstacle_depth_m",
        )

        depth = np.full(
            (height, width),
            base_depth_m,
            dtype=np.float32,
        )

        x1 = width // 3
        x2 = 2 * width // 3

        y1 = height // 3
        y2 = 2 * height // 3

        depth[
            y1:y2,
            x1:x2,
        ] = obstacle_depth_m

        return depth


# ============================================================
# DEPTH ESTIMATION SERVICE
# ============================================================

class RoboticsDepthEstimationService:

    def __init__(
        self,
        config: DepthEstimationConfig | None = None,
    ) -> None:
        self.config = (
            config or DepthEstimationConfig()
        )

        self._history: dict[
            tuple[str, str],
            deque[DepthFrame],
        ] = {}

        self._lock = threading.RLock()

        self._quality = DepthQualityEvaluator(
            self.config
        )

    @staticmethod
    def _authorize(
        access: DepthAccessContext,
        tenant_id: str,
        camera_id: str,
        permission: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise DepthAuthorizationError(
                "Cross-tenant access denied"
            )

        if camera_id not in access.allowed_camera_ids:
            raise DepthAuthorizationError(
                "Camera access denied"
            )

        if not getattr(access, permission, False):
            raise DepthAuthorizationError(
                f"Missing permission: {permission}"
            )

    def _create_frame(
        self,
        *,
        tenant_id: str,
        camera_id: str,
        depth_map: np.ndarray,
        method: DepthEstimationMethod,
        scale_type: DepthScaleType,
        access: DepthAccessContext,
        confidence_map: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DepthFrame:
        self._authorize(
            access,
            tenant_id,
            camera_id,
            "can_process",
        )

        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        ).copy()

        if depth.ndim != 2 or depth.size == 0:
            raise DepthValidationError(
                "Depth map must be nonempty and 2D"
            )

        valid = (
            np.isfinite(depth)
            & (depth > 0)
        )

        if scale_type == DepthScaleType.METRIC:
            valid &= (
                depth >= self.config.minimum_depth_m
            )

            valid &= (
                depth <= self.config.maximum_depth_m
            )

        depth[~valid] = np.nan

        if self.config.filter_type != DepthFilterType.NONE:
            depth = DepthMapFilter.apply(
                depth,
                self.config.filter_type,
                self.config.filter_kernel_size,
            )

        confidence = None

        if confidence_map is not None:
            confidence = np.asarray(
                confidence_map,
                dtype=np.float32,
            ).copy()

            if confidence.shape != depth.shape:
                raise DepthValidationError(
                    "Confidence map shape mismatch"
                )

            if not np.all(np.isfinite(confidence)):
                raise DepthValidationError(
                    "Confidence map must be finite"
                )

            if np.any(
                (confidence < 0)
                | (confidence > 1)
            ):
                raise DepthValidationError(
                    "Confidence values must be in [0, 1]"
                )

            depth[
                confidence
                < self.config.confidence_threshold
            ] = np.nan

        frame = DepthFrame(
            frame_id=generate_id(),
            tenant_id=tenant_id,
            camera_id=camera_id,
            method=method,
            scale_type=scale_type,
            depth_map=depth,
            confidence_map=confidence,
            metadata=metadata or {},
        )

        key = (
            tenant_id,
            camera_id,
        )

        with self._lock:
            history = self._history.setdefault(
                key,
                deque(
                    maxlen=(
                        self.config.maximum_history_frames
                    )
                ),
            )

            history.append(frame)

        return frame

    # --------------------------------------------------------
    # RGB-D
    # --------------------------------------------------------

    def process_rgbd(
        self,
        *,
        tenant_id: str,
        camera_id: str,
        raw_depth: np.ndarray,
        depth_scale_m: float,
        access: DepthAccessContext,
    ) -> DepthFrame:
        depth = RGBDDepthEstimator.convert_to_meters(
            raw_depth,
            depth_scale_m,
        )

        return self._create_frame(
            tenant_id=tenant_id,
            camera_id=camera_id,
            depth_map=depth,
            method=DepthEstimationMethod.RGBD,
            scale_type=DepthScaleType.METRIC,
            access=access,
        )

    # --------------------------------------------------------
    # STEREO
    # --------------------------------------------------------

    def process_stereo(
        self,
        *,
        tenant_id: str,
        camera_id: str,
        left_image: np.ndarray,
        right_image: np.ndarray,
        stereo_config: StereoDepthConfig,
        access: DepthAccessContext,
        images_are_rectified: bool = False,
    ) -> DepthFrame:
        self._authorize(
            access,
            tenant_id,
            camera_id,
            "can_process",
        )

        estimator = StereoDepthEstimator(
            stereo_config
        )

        depth, disparity = estimator.estimate(
            left_image,
            right_image,
            images_are_rectified=images_are_rectified,
        )

        return self._create_frame(
            tenant_id=tenant_id,
            camera_id=camera_id,
            depth_map=depth,
            method=DepthEstimationMethod.STEREO,
            scale_type=DepthScaleType.METRIC,
            access=access,
            metadata={
                "baseline_m": stereo_config.baseline_m,
                "focal_length_px": (
                    stereo_config.focal_length_px
                ),
                "rectified": images_are_rectified,
            },
        )

    # --------------------------------------------------------
    # MONOCULAR
    # --------------------------------------------------------

    def process_monocular(
        self,
        *,
        tenant_id: str,
        camera_id: str,
        image: np.ndarray,
        estimator: MonocularONNXDepthEstimator,
        access: DepthAccessContext,
    ) -> DepthFrame:
        self._authorize(
            access,
            tenant_id,
            camera_id,
            "can_process",
        )

        depth = estimator.estimate(
            image
        )

        return self._create_frame(
            tenant_id=tenant_id,
            camera_id=camera_id,
            depth_map=depth,
            method=DepthEstimationMethod.MONOCULAR,
            scale_type=(
                estimator.config.output_scale_type
            ),
            access=access,
            metadata={
                "model_path": "[MODEL_PATH_REDACTED]",
            },
        )

    # --------------------------------------------------------
    # SYNTHETIC
    # --------------------------------------------------------

    def process_synthetic(
        self,
        *,
        tenant_id: str,
        camera_id: str,
        width: int,
        height: int,
        access: DepthAccessContext,
    ) -> DepthFrame:
        depth = SyntheticDepthGenerator.generate(
            width,
            height,
        )

        return self._create_frame(
            tenant_id=tenant_id,
            camera_id=camera_id,
            depth_map=depth,
            method=DepthEstimationMethod.SYNTHETIC,
            scale_type=DepthScaleType.METRIC,
            access=access,
            metadata={
                "simulated": True,
            },
        )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    def evaluate_quality(
        self,
        frame: DepthFrame,
        *,
        access: DepthAccessContext,
    ) -> DepthQualityReport:
        self._authorize(
            access,
            frame.tenant_id,
            frame.camera_id,
            "can_view",
        )

        return self._quality.evaluate(
            frame.depth_map,
            frame.scale_type,
        )

    # --------------------------------------------------------
    # POINT CLOUD
    # --------------------------------------------------------

    def create_point_cloud(
        self,
        frame: DepthFrame,
        intrinsics: DepthCameraIntrinsics,
        *,
        access: DepthAccessContext,
        stride: int = 2,
    ) -> PointCloud:
        self._authorize(
            access,
            frame.tenant_id,
            frame.camera_id,
            "can_process",
        )

        if frame.scale_type != DepthScaleType.METRIC:
            raise DepthCalibrationError(
                "Metric point cloud requires metric depth"
            )

        return DepthPointCloudConverter.depth_to_point_cloud(
            frame.depth_map,
            intrinsics,
            camera_id=frame.camera_id,
            frame_id=frame.frame_id,
            stride=stride,
            maximum_points=(
                self.config.maximum_point_cloud_points
            ),
            minimum_depth_m=(
                self.config.minimum_depth_m
            ),
            maximum_depth_m=(
                self.config.maximum_depth_m
            ),
        )

    # --------------------------------------------------------
    # OBSTACLE DISTANCE
    # --------------------------------------------------------

    def analyze_obstacles(
        self,
        frame: DepthFrame,
        *,
        access: DepthAccessContext,
        region: DepthRegion | None = None,
    ) -> ObstacleDistanceResult:
        self._authorize(
            access,
            frame.tenant_id,
            frame.camera_id,
            "can_view",
        )

        if frame.scale_type != DepthScaleType.METRIC:
            raise DepthCalibrationError(
                "Obstacle distance requires metric depth"
            )

        return DepthObstacleAnalyzer.analyze(
            frame.depth_map,
            threshold_m=(
                self.config.obstacle_distance_threshold_m
            ),
            region=region,
            minimum_valid_pixel_ratio=(
                self.config.minimum_valid_pixel_ratio
            ),
        )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    def history(
        self,
        camera_id: str,
        *,
        access: DepthAccessContext,
    ) -> tuple[DepthFrame, ...]:
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

    def latest(
        self,
        camera_id: str,
        *,
        access: DepthAccessContext,
    ) -> DepthFrame | None:
        history = self.history(
            camera_id,
            access=access,
        )

        return (
            history[-1]
            if history
            else None
        )

    # --------------------------------------------------------
    # JSON EXPORT
    # --------------------------------------------------------

    @staticmethod
    def export_frame_metadata(
        frame: DepthFrame,
    ) -> str:
        return json.dumps(
            frame.metadata_dict(),
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestDepthEstimation(unittest.TestCase):

    def setUp(self) -> None:
        self.config = DepthEstimationConfig(
            filter_type=DepthFilterType.NONE,
        )

        self.service = RoboticsDepthEstimationService(
            self.config
        )

        self.access = DepthAccessContext(
            tenant_id="tenant_alpha",
            user_id="engineer",
            allowed_camera_ids=frozenset({
                "CAM-001",
            }),
            can_process=True,
            can_view=True,
        )

        self.intrinsics = DepthCameraIntrinsics(
            width=64,
            height=48,
            fx=100.0,
            fy=100.0,
            cx=32.0,
            cy=24.0,
        )

    def test_synthetic_depth(self) -> None:
        frame = self.service.process_synthetic(
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            width=64,
            height=48,
            access=self.access,
        )

        self.assertEqual(
            frame.depth_map.shape,
            (48, 64),
        )

        self.assertEqual(
            frame.scale_type,
            DepthScaleType.METRIC,
        )

    def test_stereo_disparity_to_depth(self) -> None:
        disparity = np.full(
            (10, 10),
            20.0,
            dtype=np.float32,
        )

        depth = StereoDepthEstimator.disparity_to_depth(
            disparity,
            focal_length_px=100.0,
            baseline_m=0.1,
        )

        self.assertAlmostEqual(
            float(depth[0, 0]),
            0.5,
        )

    def test_rgbd_conversion(self) -> None:
        raw = np.array(
            [
                [1000, 2000],
                [0, 3000],
            ],
            dtype=np.uint16,
        )

        depth = RGBDDepthEstimator.convert_to_meters(
            raw,
            depth_scale_m=0.001,
        )

        self.assertAlmostEqual(
            float(depth[0, 0]),
            1.0,
        )

        self.assertTrue(
            np.isnan(depth[1, 0])
        )

    def test_quality(self) -> None:
        frame = self.service.process_synthetic(
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            width=64,
            height=48,
            access=self.access,
        )

        report = self.service.evaluate_quality(
            frame,
            access=self.access,
        )

        self.assertEqual(
            report.quality,
            DepthQuality.GOOD,
        )

    def test_point_cloud(self) -> None:
        frame = self.service.process_synthetic(
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            width=64,
            height=48,
            access=self.access,
        )

        cloud = self.service.create_point_cloud(
            frame,
            self.intrinsics,
            access=self.access,
        )

        self.assertGreater(
            cloud.point_count,
            0,
        )

        self.assertEqual(
            cloud.points_xyz.shape[1],
            3,
        )

    def test_pixel_to_point(self) -> None:
        point = DepthPointCloudConverter.pixel_to_point(
            32,
            24,
            2.0,
            self.intrinsics,
        )

        self.assertAlmostEqual(
            point[0],
            0.0,
        )

        self.assertAlmostEqual(
            point[1],
            0.0,
        )

        self.assertAlmostEqual(
            point[2],
            2.0,
        )

    def test_obstacle_detection(self) -> None:
        frame = self.service.process_synthetic(
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            width=64,
            height=48,
            access=self.access,
        )

        result = self.service.analyze_obstacles(
            frame,
            access=self.access,
        )

        self.assertTrue(
            result.obstacle_detected
        )

    def test_region_statistics(self) -> None:
        depth = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float32,
        )

        result = DepthRegionAnalyzer.analyze(
            depth,
            DepthRegion(
                0,
                0,
                2,
                2,
            ),
        )

        self.assertAlmostEqual(
            result.mean_depth,
            2.5,
        )

    def test_depth_normalization(self) -> None:
        depth = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float32,
        )

        normalized = DepthVisualizer.normalize(
            depth
        )

        self.assertEqual(
            normalized.dtype,
            np.uint8,
        )

    def test_tenant_isolation(self) -> None:
        other_access = DepthAccessContext(
            tenant_id="tenant_beta",
            user_id="other",
            allowed_camera_ids=frozenset({
                "CAM-001",
            }),
            can_process=True,
            can_view=True,
        )

        with self.assertRaises(
            DepthAuthorizationError
        ):
            self.service.process_synthetic(
                tenant_id="tenant_alpha",
                camera_id="CAM-001",
                width=64,
                height=48,
                access=other_access,
            )

    def test_history(self) -> None:
        for _ in range(3):
            self.service.process_synthetic(
                tenant_id="tenant_alpha",
                camera_id="CAM-001",
                width=64,
                height=48,
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

    def test_json_export(self) -> None:
        frame = self.service.process_synthetic(
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            width=64,
            height=48,
            access=self.access,
        )

        payload = self.service.export_frame_metadata(
            frame
        )

        self.assertIn(
            "CAM-001",
            payload,
        )

    def test_relative_depth_point_cloud_rejected(self) -> None:
        frame = self.service._create_frame(
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            depth_map=np.ones(
                (48, 64),
                dtype=np.float32,
            ),
            method=DepthEstimationMethod.MONOCULAR,
            scale_type=DepthScaleType.RELATIVE,
            access=self.access,
        )

        with self.assertRaises(
            DepthCalibrationError
        ):
            self.service.create_point_cloud(
                frame,
                self.intrinsics,
                access=self.access,
            )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Depth Estimation and 3D Perception Engine\n"
    )

    config = DepthEstimationConfig(
        minimum_depth_m=0.1,
        maximum_depth_m=20.0,
        filter_type=DepthFilterType.NONE,
        obstacle_distance_threshold_m=1.0,
    )

    service = RoboticsDepthEstimationService(
        config
    )

    access = DepthAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_camera_ids=frozenset({
            "CAM-FRONT-001",
        }),
        can_process=True,
        can_view=True,
    )

    intrinsics = DepthCameraIntrinsics(
        width=320,
        height=240,
        fx=280.0,
        fy=280.0,
        cx=160.0,
        cy=120.0,
    )

    frame = service.process_synthetic(
        tenant_id="tenant_alpha",
        camera_id="CAM-FRONT-001",
        width=320,
        height=240,
        access=access,
    )

    print("DEPTH FRAME METADATA\n")

    print(
        service.export_frame_metadata(
            frame
        )
    )

    quality = service.evaluate_quality(
        frame,
        access=access,
    )

    print("\nDEPTH QUALITY\n")

    print(
        json.dumps(
            {
                "quality": quality.quality.value,
                "valid_pixel_ratio": (
                    quality.valid_pixel_ratio
                ),
                "minimum_depth": (
                    quality.minimum_depth
                ),
                "maximum_depth": (
                    quality.maximum_depth
                ),
                "mean_depth": (
                    quality.mean_depth
                ),
                "median_depth": (
                    quality.median_depth
                ),
                "warnings": list(
                    quality.warnings
                ),
            },
            indent=2,
        )
    )

    cloud = service.create_point_cloud(
        frame,
        intrinsics,
        access=access,
        stride=4,
    )

    print("\nPOINT CLOUD\n")

    print(
        json.dumps(
            {
                "point_count": cloud.point_count,
                "coordinate_frame": (
                    cloud.coordinate_frame
                ),
                "first_points": (
                    cloud.points_xyz[:5].tolist()
                ),
            },
            indent=2,
        )
    )

    obstacle = service.analyze_obstacles(
        frame,
        access=access,
    )

    print("\nOBSTACLE DISTANCE ANALYSIS\n")

    print(
        json.dumps(
            {
                "minimum_distance_m": (
                    obstacle.minimum_distance_m
                ),
                "median_distance_m": (
                    obstacle.median_distance_m
                ),
                "obstacle_detected": (
                    obstacle.obstacle_detected
                ),
                "threshold_m": (
                    obstacle.threshold_m
                ),
                "valid_pixel_ratio": (
                    obstacle.valid_pixel_ratio
                ),
                "advisory_only": (
                    obstacle.advisory_only
                ),
            },
            indent=2,
        )
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestDepthEstimation
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
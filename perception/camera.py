"""Camera module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/camera.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: camera.py

Camera acquisition, frame processing, simulated camera streaming,
device lifecycle management, and camera health monitoring.

Features:
    - USB / integrated camera support
    - RTSP and HTTP camera streams
    - Video-file camera sources
    - Simulated camera sources
    - Camera configuration management
    - Camera discovery (OpenCV backend)
    - Start / stop / restart
    - Frame capture
    - Background frame acquisition
    - Thread-safe frame buffering
    - Frame timestamps and sequence numbers
    - Frame resolution and FPS management
    - Brightness and contrast adjustments
    - Frame resizing and rotation
    - Camera calibration integration
    - Camera health monitoring
    - Frame drop detection
    - Reconnection with bounded retries
    - Multi-camera management
    - Tenant and camera access control
    - Camera statistics
    - Event and audit history
    - JSON-safe metadata export
    - Unit tests

Dependencies:
    Python 3.11+
    Optional: opencv-python, numpy

Installation:
    pip install opencv-python numpy

Security:
    - Camera access is permission-controlled.
    - Network camera URLs are disabled by default.
    - Credentials are never included in exported metadata.
    - Camera frames are not persisted automatically.
    - This module does not control robot actuators.

Note:
    The simulated camera backend works without OpenCV.
"""

from __future__ import annotations

import copy
import json
import math
import queue
import re
import threading
import time
import unittest
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Protocol
from urllib.parse import urlsplit, urlunsplit


try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


# ============================================================
# EXCEPTIONS
# ============================================================

class CameraError(Exception):
    """Base camera exception."""


class CameraConfigurationError(CameraError):
    """Invalid camera configuration."""


class CameraConnectionError(CameraError):
    """Camera connection failed."""


class CameraCaptureError(CameraError):
    """Frame acquisition failed."""


class CameraStateError(CameraError):
    """Invalid camera lifecycle transition."""


class CameraAuthorizationError(CameraError):
    """Camera access denied."""


class CameraNotFoundError(CameraError):
    """Camera not registered."""


class CameraDependencyError(CameraError):
    """Optional camera dependency unavailable."""


class CameraCapacityError(CameraError):
    """Camera manager capacity exceeded."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def validate_id(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise CameraConfigurationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise CameraConfigurationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise CameraConfigurationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise CameraConfigurationError(
            f"{name} must be finite"
        )

    return result


def redact_source(source: str) -> str:
    """
    Remove user information, passwords and query strings
    from exported network camera addresses.
    """
    try:
        parts = urlsplit(source)

        if not parts.scheme:
            return source

        hostname = parts.hostname or ""

        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"

        if parts.port is not None:
            hostname += f":{parts.port}"

        return urlunsplit(
            (
                parts.scheme,
                hostname,
                parts.path,
                "",
                "",
            )
        )
    except (ValueError, TypeError):
        return "[REDACTED]"


# ============================================================
# ENUMERATIONS
# ============================================================

class CameraSourceType(str, Enum):
    USB = "usb"
    RTSP = "rtsp"
    HTTP = "http"
    VIDEO_FILE = "video_file"
    SIMULATED = "simulated"


class CameraStatus(str, Enum):
    CREATED = "created"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    STOPPED = "stopped"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class CameraHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CameraEventType(str, Enum):
    REGISTERED = "registered"
    CONNECTED = "connected"
    STARTED = "started"
    FRAME_CAPTURED = "frame_captured"
    FRAME_DROPPED = "frame_dropped"
    CAPTURE_FAILED = "capture_failed"
    RECONNECTED = "reconnected"
    STOPPED = "stopped"
    DISCONNECTED = "disconnected"
    ERROR = "error"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    tenant_id: str
    name: str

    source_type: CameraSourceType

    source: str | int = 0

    width: int = 640
    height: int = 480
    fps: float = 30.0

    buffer_size: int = 10

    capture_timeout_seconds: float = 5.0
    reconnect_interval_seconds: float = 2.0
    maximum_reconnect_attempts: int = 3

    enable_auto_reconnect: bool = True
    allow_network_source: bool = False

    brightness: float = 0.0
    contrast: float = 1.0

    rotation_degrees: int = 0

    calibration_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_id(self.camera_id, "camera_id")
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.name, "name")

        for name in ("width", "height", "buffer_size"):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise CameraConfigurationError(
                    f"{name} must be a positive integer"
                )

        for name in (
            "maximum_reconnect_attempts",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise CameraConfigurationError(
                    f"{name} must be a nonnegative integer"
                )

        for name in (
            "fps",
            "capture_timeout_seconds",
            "reconnect_interval_seconds",
            "contrast",
        ):
            if finite_number(
                getattr(self, name),
                name,
            ) <= 0:
                raise CameraConfigurationError(
                    f"{name} must be positive"
                )

        finite_number(
            self.brightness,
            "brightness",
        )

        if self.rotation_degrees not in (
            0,
            90,
            180,
            270,
        ):
            raise CameraConfigurationError(
                "Rotation must be 0, 90, 180 or 270"
            )

        if self.source_type in (
            CameraSourceType.RTSP,
            CameraSourceType.HTTP,
        ):
            if not self.allow_network_source:
                raise CameraConfigurationError(
                    "Network camera sources are disabled"
                )

            if not isinstance(self.source, str):
                raise CameraConfigurationError(
                    "Network camera source must be a URL"
                )

            parts = urlsplit(self.source)

            allowed_schemes = (
                {"rtsp", "rtsps"}
                if self.source_type == CameraSourceType.RTSP
                else {"http", "https"}
            )

            if (
                parts.scheme.lower() not in allowed_schemes
                or not parts.hostname
            ):
                raise CameraConfigurationError(
                    "Invalid network camera URL"
                )

        if self.source_type == CameraSourceType.USB:
            if (
                isinstance(self.source, bool)
                or not isinstance(self.source, int)
                or self.source < 0
            ):
                raise CameraConfigurationError(
                    "USB source must be a nonnegative device index"
                )

        if self.source_type == CameraSourceType.VIDEO_FILE:
            if (
                not isinstance(self.source, str)
                or not self.source.strip()
            ):
                raise CameraConfigurationError(
                    "Video file path required"
                )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class CameraAccessContext:
    tenant_id: str
    user_id: str

    allowed_camera_ids: frozenset[str] = frozenset()

    can_view: bool = False
    can_capture: bool = False
    can_manage: bool = False

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.user_id, "user_id")


# ============================================================
# FRAME DATA
# ============================================================

@dataclass(frozen=True)
class CameraFrame:
    frame_id: str
    camera_id: str
    tenant_id: str

    sequence_number: int

    timestamp: datetime

    width: int
    height: int

    image: Any = field(repr=False)

    simulated: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "tenant_id": self.tenant_id,
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp.isoformat(),
            "width": self.width,
            "height": self.height,
            "simulated": self.simulated,
            "metadata": copy.deepcopy(self.metadata),
        }


# ============================================================
# CAMERA EVENTS
# ============================================================

@dataclass(frozen=True)
class CameraEvent:
    event_id: str
    camera_id: str
    tenant_id: str

    event_type: CameraEventType

    message: str

    timestamp: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# CAMERA STATISTICS
# ============================================================

@dataclass(frozen=True)
class CameraStatistics:
    camera_id: str

    status: CameraStatus

    frames_captured: int
    frames_dropped: int
    capture_errors: int
    reconnect_attempts: int

    measured_fps: float

    last_frame_timestamp: datetime | None

    uptime_seconds: float


# ============================================================
# CAMERA HEALTH REPORT
# ============================================================

@dataclass(frozen=True)
class CameraHealthReport:
    camera_id: str

    status: CameraHealthStatus

    camera_status: CameraStatus

    frame_age_seconds: float | None

    measured_fps: float

    warnings: tuple[str, ...]

    checked_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# CAMERA BACKEND PROTOCOL
# ============================================================

class CameraBackend(Protocol):

    def open(self) -> None:
        ...

    def read(self) -> Any | None:
        ...

    def close(self) -> None:
        ...

    def is_open(self) -> bool:
        ...


# ============================================================
# SIMULATED CAMERA BACKEND
# ============================================================

class SimulatedCameraBackend:
    """
    Generates deterministic synthetic RGB frames.

    If NumPy is unavailable, frames are represented as
    nested Python lists.
    """

    def __init__(
        self,
        width: int,
        height: int,
    ) -> None:
        self.width = width
        self.height = height

        self._open = False
        self._sequence = 0

        self._lock = threading.RLock()

    def open(self) -> None:
        with self._lock:
            self._open = True

    def is_open(self) -> bool:
        with self._lock:
            return self._open

    def read(self) -> Any | None:
        with self._lock:
            if not self._open:
                return None

            self._sequence += 1

            value = self._sequence % 256

            if np is not None:
                frame = np.zeros(
                    (
                        self.height,
                        self.width,
                        3,
                    ),
                    dtype=np.uint8,
                )

                frame[:, :, 0] = value
                frame[:, :, 1] = 100
                frame[:, :, 2] = 180

                return frame

            return [
                [
                    [value, 100, 180]
                    for _ in range(self.width)
                ]
                for _ in range(self.height)
            ]

    def close(self) -> None:
        with self._lock:
            self._open = False


# ============================================================
# OPENCV CAMERA BACKEND
# ============================================================

class OpenCVCameraBackend:

    def __init__(
        self,
        config: CameraConfig,
    ) -> None:
        self.config = config

        self._capture: Any = None
        self._lock = threading.RLock()

    def open(self) -> None:
        if cv2 is None:
            raise CameraDependencyError(
                "Install opencv-python for physical cameras"
            )

        with self._lock:
            if self._capture is not None:
                self.close()

            capture = cv2.VideoCapture(
                self.config.source
            )

            if not capture.isOpened():
                capture.release()

                raise CameraConnectionError(
                    "Unable to open camera source"
                )

            capture.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                self.config.width,
            )

            capture.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                self.config.height,
            )

            capture.set(
                cv2.CAP_PROP_FPS,
                self.config.fps,
            )

            self._capture = capture

    def read(self) -> Any | None:
        with self._lock:
            if self._capture is None:
                return None

            success, frame = self._capture.read()

            if not success:
                return None

            return frame

    def is_open(self) -> bool:
        with self._lock:
            return (
                self._capture is not None
                and self._capture.isOpened()
            )

    def close(self) -> None:
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None


# ============================================================
# CAMERA FRAME PROCESSOR
# ============================================================

class CameraFrameProcessor:

    @staticmethod
    def dimensions(image: Any) -> tuple[int, int]:
        if np is not None and isinstance(
            image,
            np.ndarray,
        ):
            if image.ndim < 2:
                raise CameraCaptureError(
                    "Invalid image dimensions"
                )

            height, width = image.shape[:2]
            return width, height

        if isinstance(image, list) and image:
            height = len(image)
            width = len(image[0])

            if width <= 0:
                raise CameraCaptureError(
                    "Invalid image width"
                )

            return width, height

        raise CameraCaptureError(
            "Unsupported image format"
        )

    @staticmethod
    def rotate(
        image: Any,
        degrees: int,
    ) -> Any:
        if degrees == 0:
            return image

        if np is not None and isinstance(
            image,
            np.ndarray,
        ):
            turns = {
                90: 3,
                180: 2,
                270: 1,
            }[degrees]

            return np.rot90(
                image,
                turns,
            ).copy()

        if isinstance(image, list):
            if degrees == 90:
                return [
                    list(row)
                    for row in zip(
                        *image[::-1]
                    )
                ]

            if degrees == 180:
                return [
                    row[::-1]
                    for row in image[::-1]
                ]

            if degrees == 270:
                return [
                    list(row)
                    for row in zip(*image)
                ][::-1]

        raise CameraCaptureError(
            "Unsupported rotation format"
        )

    @staticmethod
    def adjust_brightness_contrast(
        image: Any,
        brightness: float,
        contrast: float,
    ) -> Any:
        if brightness == 0 and contrast == 1:
            return image

        if np is not None and isinstance(
            image,
            np.ndarray,
        ):
            result = (
                image.astype(np.float32)
                * contrast
                + brightness
            )

            return np.clip(
                result,
                0,
                255,
            ).astype(np.uint8)

        def adjust_pixel(value: Any) -> Any:
            if isinstance(value, list):
                return [
                    adjust_pixel(item)
                    for item in value
                ]

            return int(
                max(
                    0,
                    min(
                        255,
                        value * contrast + brightness,
                    ),
                )
            )

        return adjust_pixel(image)

    @staticmethod
    def resize(
        image: Any,
        width: int,
        height: int,
    ) -> Any:
        if width <= 0 or height <= 0:
            raise CameraCaptureError(
                "Invalid resize dimensions"
            )

        if cv2 is not None and np is not None:
            if isinstance(image, np.ndarray):
                return cv2.resize(
                    image,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )

        original_width, original_height = (
            CameraFrameProcessor.dimensions(image)
        )

        if np is not None and isinstance(
            image,
            np.ndarray,
        ):
            x_indices = np.minimum(
                (
                    np.arange(width)
                    * original_width
                    / width
                ).astype(int),
                original_width - 1,
            )

            y_indices = np.minimum(
                (
                    np.arange(height)
                    * original_height
                    / height
                ).astype(int),
                original_height - 1,
            )

            return image[
                y_indices[:, None],
                x_indices[None, :],
            ].copy()

        return [
            [
                copy.deepcopy(
                    image[
                        min(
                            int(y * original_height / height),
                            original_height - 1,
                        )
                    ][
                        min(
                            int(x * original_width / width),
                            original_width - 1,
                        )
                    ]
                )
                for x in range(width)
            ]
            for y in range(height)
        ]

    @classmethod
    def process(
        cls,
        image: Any,
        config: CameraConfig,
    ) -> Any:
        result = cls.adjust_brightness_contrast(
            image,
            config.brightness,
            config.contrast,
        )

        result = cls.rotate(
            result,
            config.rotation_degrees,
        )

        return result


# ============================================================
# CAMERA DEVICE
# ============================================================

class CameraDevice:

    def __init__(
        self,
        config: CameraConfig,
        backend: CameraBackend | None = None,
    ) -> None:
        self.config = config

        if backend is None:
            if (
                config.source_type
                == CameraSourceType.SIMULATED
            ):
                backend = SimulatedCameraBackend(
                    config.width,
                    config.height,
                )
            else:
                backend = OpenCVCameraBackend(
                    config
                )

        self.backend = backend

        self._status = CameraStatus.CREATED

        self._frames: deque[CameraFrame] = deque(
            maxlen=config.buffer_size
        )

        self._events: deque[CameraEvent] = deque(
            maxlen=1000
        )

        self._lock = threading.RLock()

        self._stop_event = threading.Event()

        self._worker: threading.Thread | None = None

        self._sequence = 0

        self._frames_captured = 0
        self._frames_dropped = 0
        self._capture_errors = 0
        self._reconnect_attempts = 0

        self._started_monotonic: float | None = None
        self._last_frame_timestamp: datetime | None = None

        self._capture_times: deque[float] = deque(
            maxlen=120
        )

    # --------------------------------------------------------
    # ACCESS
    # --------------------------------------------------------

    def _authorize(
        self,
        access: CameraAccessContext,
        permission: str,
    ) -> None:
        if access.tenant_id != self.config.tenant_id:
            raise CameraAuthorizationError(
                "Cross-tenant camera access denied"
            )

        if (
            self.config.camera_id
            not in access.allowed_camera_ids
        ):
            raise CameraAuthorizationError(
                "Camera access denied"
            )

        if not getattr(access, permission, False):
            raise CameraAuthorizationError(
                f"Missing permission: {permission}"
            )

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    def _emit(
        self,
        event_type: CameraEventType,
        message: str,
    ) -> None:
        with self._lock:
            self._events.append(
                CameraEvent(
                    event_id=new_id(),
                    camera_id=self.config.camera_id,
                    tenant_id=self.config.tenant_id,
                    event_type=event_type,
                    message=message,
                )
            )

    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    def connect(
        self,
        *,
        access: CameraAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            if self._status == CameraStatus.STREAMING:
                raise CameraStateError(
                    "Stop streaming before reconnecting"
                )

            self._status = CameraStatus.CONNECTING

        try:
            self.backend.open()

            if not self.backend.is_open():
                raise CameraConnectionError(
                    "Backend did not open"
                )

            with self._lock:
                self._status = CameraStatus.CONNECTED

            self._emit(
                CameraEventType.CONNECTED,
                "Camera connected",
            )

        except Exception as exc:
            with self._lock:
                self._status = CameraStatus.ERROR

            self._emit(
                CameraEventType.ERROR,
                "Camera connection failed",
            )

            raise CameraConnectionError(
                "Unable to connect camera"
            ) from exc

    # --------------------------------------------------------
    # CAPTURE
    # --------------------------------------------------------

    def _capture_internal(self) -> CameraFrame:
        with self._lock:
            if self._status not in (
                CameraStatus.CONNECTED,
                CameraStatus.STREAMING,
            ):
                raise CameraStateError(
                    "Camera is not connected"
                )

        try:
            image = self.backend.read()

            if image is None:
                raise CameraCaptureError(
                    "Camera returned no frame"
                )

            image = CameraFrameProcessor.process(
                image,
                self.config,
            )

            width, height = (
                CameraFrameProcessor.dimensions(
                    image
                )
            )

            timestamp = utc_now()
            monotonic_timestamp = time.monotonic()

            with self._lock:
                self._sequence += 1
                self._frames_captured += 1

                if len(self._frames) == self._frames.maxlen:
                    self._frames_dropped += 1

                frame = CameraFrame(
                    frame_id=new_id(),
                    camera_id=self.config.camera_id,
                    tenant_id=self.config.tenant_id,
                    sequence_number=self._sequence,
                    timestamp=timestamp,
                    width=width,
                    height=height,
                    image=image,
                    simulated=(
                        self.config.source_type
                        == CameraSourceType.SIMULATED
                    ),
                    metadata={
                        "calibration_id": (
                            self.config.calibration_id
                        ),
                        "source_type": (
                            self.config.source_type.value
                        ),
                    },
                )

                self._frames.append(frame)

                self._last_frame_timestamp = timestamp

                self._capture_times.append(
                    monotonic_timestamp
                )

            return frame

        except Exception as exc:
            with self._lock:
                self._capture_errors += 1

            self._emit(
                CameraEventType.CAPTURE_FAILED,
                "Frame capture failed",
            )

            if isinstance(exc, CameraError):
                raise

            raise CameraCaptureError(
                "Camera frame acquisition failed"
            ) from exc

    def capture(
        self,
        *,
        access: CameraAccessContext,
    ) -> CameraFrame:
        self._authorize(
            access,
            "can_capture",
        )

        return self._capture_internal()

    # --------------------------------------------------------
    # BACKGROUND STREAM
    # --------------------------------------------------------

    def start_stream(
        self,
        *,
        access: CameraAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            if self._status != CameraStatus.CONNECTED:
                raise CameraStateError(
                    "Camera must be connected"
                )

            if (
                self._worker is not None
                and self._worker.is_alive()
            ):
                raise CameraStateError(
                    "Camera worker already running"
                )

            self._stop_event.clear()

            self._status = CameraStatus.STREAMING
            self._started_monotonic = time.monotonic()

            self._worker = threading.Thread(
                target=self._stream_loop,
                name=f"camera-{self.config.camera_id}",
                daemon=True,
            )

            self._worker.start()

        self._emit(
            CameraEventType.STARTED,
            "Camera streaming started",
        )

    def _stream_loop(self) -> None:
        interval = 1.0 / self.config.fps

        while not self._stop_event.is_set():
            started = time.monotonic()

            try:
                self._capture_internal()

            except CameraError:
                if not self.config.enable_auto_reconnect:
                    with self._lock:
                        self._status = CameraStatus.ERROR
                    break

                if not self._reconnect():
                    with self._lock:
                        self._status = CameraStatus.ERROR
                    break

            elapsed = time.monotonic() - started

            self._stop_event.wait(
                max(0.0, interval - elapsed)
            )

    # --------------------------------------------------------
    # RECONNECT
    # --------------------------------------------------------

    def _reconnect(self) -> bool:
        for _ in range(
            self.config.maximum_reconnect_attempts
        ):
            if self._stop_event.is_set():
                return False

            with self._lock:
                self._reconnect_attempts += 1

            self._stop_event.wait(
                self.config.reconnect_interval_seconds
            )

            if self._stop_event.is_set():
                return False

            try:
                self.backend.close()
                self.backend.open()

                if self.backend.is_open():
                    self._emit(
                        CameraEventType.RECONNECTED,
                        "Camera reconnected",
                    )

                    return True

            except Exception:
                continue

        return False

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    def stop_stream(
        self,
        *,
        access: CameraAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            worker = self._worker

            self._stop_event.set()

        if worker is not None:
            worker.join(
                timeout=(
                    self.config.capture_timeout_seconds
                    + 1.0
                )
            )

            if worker.is_alive():
                raise CameraStateError(
                    "Camera worker did not stop; "
                    "backend read may be blocked"
                )

        with self._lock:
            self._worker = None

            if self.backend.is_open():
                self._status = CameraStatus.CONNECTED
            else:
                self._status = CameraStatus.DISCONNECTED

        self._emit(
            CameraEventType.STOPPED,
            "Camera streaming stopped",
        )

    # --------------------------------------------------------
    # DISCONNECT
    # --------------------------------------------------------

    def disconnect(
        self,
        *,
        access: CameraAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            streaming = (
                self._worker is not None
                and self._worker.is_alive()
            )

        if streaming:
            self.stop_stream(
                access=access
            )

        self.backend.close()

        with self._lock:
            self._status = CameraStatus.DISCONNECTED

        self._emit(
            CameraEventType.DISCONNECTED,
            "Camera disconnected",
        )

    # --------------------------------------------------------
    # GET LATEST FRAME
    # --------------------------------------------------------

    def latest_frame(
        self,
        *,
        access: CameraAccessContext,
    ) -> CameraFrame | None:
        self._authorize(
            access,
            "can_capture",
        )

        with self._lock:
            if not self._frames:
                return None

            return self._frames[-1]

    # --------------------------------------------------------
    # FRAME BUFFER
    # --------------------------------------------------------

    def buffered_frames(
        self,
        *,
        access: CameraAccessContext,
    ) -> tuple[CameraFrame, ...]:
        self._authorize(
            access,
            "can_capture",
        )

        with self._lock:
            return tuple(self._frames)

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        access: CameraAccessContext,
    ) -> CameraStatistics:
        self._authorize(
            access,
            "can_view",
        )

        with self._lock:
            times = tuple(self._capture_times)

            if len(times) >= 2:
                duration = times[-1] - times[0]

                measured_fps = (
                    (len(times) - 1) / duration
                    if duration > 0
                    else 0.0
                )
            else:
                measured_fps = 0.0

            uptime = (
                time.monotonic() - self._started_monotonic
                if self._started_monotonic is not None
                else 0.0
            )

            return CameraStatistics(
                camera_id=self.config.camera_id,
                status=self._status,
                frames_captured=self._frames_captured,
                frames_dropped=self._frames_dropped,
                capture_errors=self._capture_errors,
                reconnect_attempts=self._reconnect_attempts,
                measured_fps=measured_fps,
                last_frame_timestamp=(
                    self._last_frame_timestamp
                ),
                uptime_seconds=max(0.0, uptime),
            )

    # --------------------------------------------------------
    # HEALTH
    # --------------------------------------------------------

    def health(
        self,
        *,
        access: CameraAccessContext,
        maximum_frame_age_seconds: float = 5.0,
    ) -> CameraHealthReport:
        stats = self.statistics(
            access=access
        )

        warnings = []

        frame_age = None

        if stats.last_frame_timestamp is not None:
            frame_age = max(
                0.0,
                (
                    utc_now()
                    - stats.last_frame_timestamp
                ).total_seconds(),
            )

        if stats.status in (
            CameraStatus.ERROR,
            CameraStatus.DISCONNECTED,
        ):
            warnings.append(
                "Camera disconnected or in error state"
            )

        if (
            stats.status == CameraStatus.STREAMING
            and frame_age is not None
            and frame_age > maximum_frame_age_seconds
        ):
            warnings.append(
                "Camera frame stream is stale"
            )

        if stats.capture_errors > 0:
            warnings.append(
                "Camera capture errors detected"
            )

        if stats.status in (
            CameraStatus.ERROR,
            CameraStatus.DISCONNECTED,
        ):
            health_status = CameraHealthStatus.UNHEALTHY

        elif warnings:
            health_status = CameraHealthStatus.DEGRADED

        elif stats.status in (
            CameraStatus.CONNECTED,
            CameraStatus.STREAMING,
        ):
            health_status = CameraHealthStatus.HEALTHY

        else:
            health_status = CameraHealthStatus.UNKNOWN

        return CameraHealthReport(
            camera_id=self.config.camera_id,
            status=health_status,
            camera_status=stats.status,
            frame_age_seconds=frame_age,
            measured_fps=stats.measured_fps,
            warnings=tuple(warnings),
        )

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    def events(
        self,
        *,
        access: CameraAccessContext,
    ) -> tuple[CameraEvent, ...]:
        self._authorize(
            access,
            "can_view",
        )

        with self._lock:
            return tuple(self._events)

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    def status(
        self,
        *,
        access: CameraAccessContext,
    ) -> CameraStatus:
        self._authorize(
            access,
            "can_view",
        )

        with self._lock:
            return self._status


# ============================================================
# CAMERA DISCOVERY
# ============================================================

class CameraDiscovery:

    @staticmethod
    def discover_usb_cameras(
        maximum_devices: int = 5,
    ) -> tuple[int, ...]:
        if cv2 is None:
            raise CameraDependencyError(
                "OpenCV required for camera discovery"
            )

        if maximum_devices <= 0:
            raise CameraConfigurationError(
                "maximum_devices must be positive"
            )

        available = []

        for index in range(maximum_devices):
            capture = cv2.VideoCapture(index)

            try:
                if capture.isOpened():
                    available.append(index)
            finally:
                capture.release()

        return tuple(available)


# ============================================================
# MULTI-CAMERA MANAGER
# ============================================================

class RoboticsCameraManager:

    def __init__(
        self,
        maximum_cameras: int = 32,
    ) -> None:
        if maximum_cameras <= 0:
            raise CameraConfigurationError(
                "maximum_cameras must be positive"
            )

        self.maximum_cameras = maximum_cameras

        self._cameras: dict[
            tuple[str, str],
            CameraDevice,
        ] = {}

        self._lock = threading.RLock()

    def register(
        self,
        config: CameraConfig,
        *,
        access: CameraAccessContext,
        backend: CameraBackend | None = None,
    ) -> CameraDevice:
        if (
            not access.can_manage
            or access.tenant_id != config.tenant_id
            or config.camera_id
            not in access.allowed_camera_ids
        ):
            raise CameraAuthorizationError(
                "Camera registration denied"
            )

        key = (
            config.tenant_id,
            config.camera_id,
        )

        with self._lock:
            if key in self._cameras:
                raise CameraConfigurationError(
                    "Camera already registered"
                )

            if len(self._cameras) >= self.maximum_cameras:
                raise CameraCapacityError(
                    "Maximum camera capacity reached"
                )

            device = CameraDevice(
                config,
                backend=backend,
            )

            self._cameras[key] = device

            device._emit(
                CameraEventType.REGISTERED,
                "Camera registered",
            )

            return device

    def get(
        self,
        camera_id: str,
        *,
        access: CameraAccessContext,
    ) -> CameraDevice:
        if camera_id not in access.allowed_camera_ids:
            raise CameraAuthorizationError(
                "Camera access denied"
            )

        key = (
            access.tenant_id,
            camera_id,
        )

        with self._lock:
            camera = self._cameras.get(key)

            if camera is None:
                raise CameraNotFoundError(
                    "Camera not found"
                )

            return camera

    def list_cameras(
        self,
        *,
        access: CameraAccessContext,
    ) -> tuple[str, ...]:
        if not access.can_view:
            raise CameraAuthorizationError(
                "Camera view permission required"
            )

        with self._lock:
            return tuple(
                camera_id
                for (tenant_id, camera_id)
                in self._cameras
                if tenant_id == access.tenant_id
                and camera_id in access.allowed_camera_ids
            )

    def health_summary(
        self,
        *,
        access: CameraAccessContext,
    ) -> dict[str, CameraHealthReport]:
        result = {}

        for camera_id in self.list_cameras(
            access=access
        ):
            camera = self.get(
                camera_id,
                access=access,
            )

            result[camera_id] = camera.health(
                access=access
            )

        return result

    def stop_all(
        self,
        *,
        access: CameraAccessContext,
    ) -> None:
        if not access.can_manage:
            raise CameraAuthorizationError(
                "Camera management permission required"
            )

        for camera_id in self.list_cameras(
            access=access
        ):
            camera = self.get(
                camera_id,
                access=access,
            )

            status = camera.status(
                access=access
            )

            if status == CameraStatus.STREAMING:
                camera.stop_stream(
                    access=access
                )

            if camera.backend.is_open():
                camera.disconnect(
                    access=access
                )


# ============================================================
# JSON EXPORT
# ============================================================

class CameraMetadataExporter:

    @staticmethod
    def config_to_dict(
        config: CameraConfig,
    ) -> dict[str, Any]:
        return {
            "camera_id": config.camera_id,
            "tenant_id": config.tenant_id,
            "name": config.name,
            "source_type": config.source_type.value,
            "source": (
                redact_source(config.source)
                if isinstance(config.source, str)
                else config.source
            ),
            "width": config.width,
            "height": config.height,
            "fps": config.fps,
            "buffer_size": config.buffer_size,
            "rotation_degrees": config.rotation_degrees,
            "calibration_id": config.calibration_id,
        }

    @staticmethod
    def statistics_to_json(
        statistics: CameraStatistics,
    ) -> str:
        return json.dumps(
            {
                "camera_id": statistics.camera_id,
                "status": statistics.status.value,
                "frames_captured": (
                    statistics.frames_captured
                ),
                "frames_dropped": (
                    statistics.frames_dropped
                ),
                "capture_errors": (
                    statistics.capture_errors
                ),
                "reconnect_attempts": (
                    statistics.reconnect_attempts
                ),
                "measured_fps": (
                    statistics.measured_fps
                ),
                "last_frame_timestamp": (
                    statistics.last_frame_timestamp.isoformat()
                    if statistics.last_frame_timestamp
                    else None
                ),
                "uptime_seconds": (
                    statistics.uptime_seconds
                ),
            },
            indent=2,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestRoboticsCamera(unittest.TestCase):

    def setUp(self) -> None:
        self.config = CameraConfig(
            camera_id="CAM-001",
            tenant_id="tenant_alpha",
            name="Simulated Front Camera",
            source_type=CameraSourceType.SIMULATED,
            width=64,
            height=48,
            fps=10,
            buffer_size=5,
        )

        self.access = CameraAccessContext(
            tenant_id="tenant_alpha",
            user_id="engineer",
            allowed_camera_ids=frozenset({
                "CAM-001",
            }),
            can_view=True,
            can_capture=True,
            can_manage=True,
        )

        self.camera = CameraDevice(
            self.config
        )

    def tearDown(self) -> None:
        if self.camera.backend.is_open():
            self.camera.disconnect(
                access=self.access
            )

    def test_connect(self) -> None:
        self.camera.connect(
            access=self.access
        )

        self.assertEqual(
            self.camera.status(
                access=self.access
            ),
            CameraStatus.CONNECTED,
        )

    def test_capture(self) -> None:
        self.camera.connect(
            access=self.access
        )

        frame = self.camera.capture(
            access=self.access
        )

        self.assertEqual(
            frame.width,
            64,
        )

        self.assertEqual(
            frame.height,
            48,
        )

        self.assertTrue(
            frame.simulated
        )

    def test_sequence_number(self) -> None:
        self.camera.connect(
            access=self.access
        )

        first = self.camera.capture(
            access=self.access
        )

        second = self.camera.capture(
            access=self.access
        )

        self.assertEqual(
            second.sequence_number,
            first.sequence_number + 1,
        )

    def test_latest_frame(self) -> None:
        self.camera.connect(
            access=self.access
        )

        frame = self.camera.capture(
            access=self.access
        )

        latest = self.camera.latest_frame(
            access=self.access
        )

        self.assertEqual(
            frame.frame_id,
            latest.frame_id,
        )

    def test_buffer_capacity(self) -> None:
        self.camera.connect(
            access=self.access
        )

        for _ in range(10):
            self.camera.capture(
                access=self.access
            )

        frames = self.camera.buffered_frames(
            access=self.access
        )

        self.assertEqual(
            len(frames),
            5,
        )

    def test_frame_drops(self) -> None:
        self.camera.connect(
            access=self.access
        )

        for _ in range(10):
            self.camera.capture(
                access=self.access
            )

        stats = self.camera.statistics(
            access=self.access
        )

        self.assertEqual(
            stats.frames_dropped,
            5,
        )

    def test_streaming(self) -> None:
        self.camera.connect(
            access=self.access
        )

        self.camera.start_stream(
            access=self.access
        )

        time.sleep(0.25)

        self.camera.stop_stream(
            access=self.access
        )

        stats = self.camera.statistics(
            access=self.access
        )

        self.assertGreater(
            stats.frames_captured,
            0,
        )

    def test_health(self) -> None:
        self.camera.connect(
            access=self.access
        )

        self.camera.capture(
            access=self.access
        )

        report = self.camera.health(
            access=self.access
        )

        self.assertEqual(
            report.status,
            CameraHealthStatus.HEALTHY,
        )

    def test_tenant_isolation(self) -> None:
        other = CameraAccessContext(
            tenant_id="tenant_beta",
            user_id="other",
            allowed_camera_ids=frozenset({
                "CAM-001",
            }),
            can_view=True,
            can_capture=True,
            can_manage=True,
        )

        with self.assertRaises(
            CameraAuthorizationError
        ):
            self.camera.connect(
                access=other
            )

    def test_camera_manager(self) -> None:
        manager = RoboticsCameraManager()

        manager.register(
            self.config,
            access=self.access,
        )

        self.assertIn(
            "CAM-001",
            manager.list_cameras(
                access=self.access
            ),
        )

    def test_rotation(self) -> None:
        image = [
            [[1], [2]],
            [[3], [4]],
        ]

        rotated = CameraFrameProcessor.rotate(
            image,
            90,
        )

        self.assertEqual(
            rotated,
            [
                [[3], [1]],
                [[4], [2]],
            ],
        )

    def test_resize(self) -> None:
        image = [
            [[1], [2]],
            [[3], [4]],
        ]

        resized = CameraFrameProcessor.resize(
            image,
            4,
            4,
        )

        width, height = (
            CameraFrameProcessor.dimensions(
                resized
            )
        )

        self.assertEqual(
            (width, height),
            (4, 4),
        )

    def test_frame_metadata(self) -> None:
        self.camera.connect(
            access=self.access
        )

        frame = self.camera.capture(
            access=self.access
        )

        metadata = frame.metadata_dict()

        self.assertEqual(
            metadata["camera_id"],
            "CAM-001",
        )

        self.assertNotIn(
            "image",
            metadata,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Camera Acquisition and Streaming Engine\n"
    )

    manager = RoboticsCameraManager()

    access = CameraAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_camera_ids=frozenset({
            "CAM-FRONT-001",
        }),
        can_view=True,
        can_capture=True,
        can_manage=True,
    )

    config = CameraConfig(
        camera_id="CAM-FRONT-001",
        tenant_id="tenant_alpha",
        name="Front Robot Camera",
        source_type=CameraSourceType.SIMULATED,
        width=320,
        height=240,
        fps=15,
        buffer_size=20,
        calibration_id="CAL-FRONT-001",
    )

    camera = manager.register(
        config,
        access=access,
    )

    camera.connect(
        access=access
    )

    print("CAMERA CONFIGURATION\n")

    print(
        json.dumps(
            CameraMetadataExporter.config_to_dict(
                config
            ),
            indent=2,
        )
    )

    frame = camera.capture(
        access=access
    )

    print("\nCAPTURED FRAME METADATA\n")

    print(
        json.dumps(
            frame.metadata_dict(),
            indent=2,
        )
    )

    camera.start_stream(
        access=access
    )

    time.sleep(1.0)

    camera.stop_stream(
        access=access
    )

    stats = camera.statistics(
        access=access
    )

    print("\nCAMERA STATISTICS\n")

    print(
        CameraMetadataExporter.statistics_to_json(
            stats
        )
    )

    health = camera.health(
        access=access
    )

    print("\nCAMERA HEALTH\n")

    print(
        json.dumps(
            {
                "camera_id": health.camera_id,
                "status": health.status.value,
                "camera_status": (
                    health.camera_status.value
                ),
                "frame_age_seconds": (
                    health.frame_age_seconds
                ),
                "measured_fps": (
                    health.measured_fps
                ),
                "warnings": list(
                    health.warnings
                ),
            },
            indent=2,
        )
    )

    camera.disconnect(
        access=access
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestRoboticsCamera
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
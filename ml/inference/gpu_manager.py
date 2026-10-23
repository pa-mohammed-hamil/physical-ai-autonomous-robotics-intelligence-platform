"""
ml/inference/gpu_manager.py

Physical AI & Autonomous Robotics Intelligence Platform

GPU resource discovery, monitoring, allocation, and inference scheduling.

Features:
- NVIDIA CUDA GPU detection
- Apple MPS detection
- CPU fallback
- GPU memory monitoring
- GPU utilization monitoring (NVML optional)
- Multi-GPU selection
- Priority-aware GPU allocation
- GPU memory reservation
- Concurrent inference limits
- GPU health monitoring
- CUDA cache cleanup
- Inference device selection
- GPU resource reporting
- Async allocation and release
- Context-managed GPU reservations
- Safe handling of unavailable GPUs

Dependencies:
    pip install torch psutil

Optional:
    pip install nvidia-ml-py

Python 3.11+
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import platform
import time
import uuid

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, AsyncIterator, Mapping, Protocol


logger = logging.getLogger(__name__)


# ============================================================
# Optional Dependencies
# ============================================================

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None


# ============================================================
# Constants
# ============================================================

BYTES_PER_MIB = 1024 * 1024

DEFAULT_MAX_CONCURRENT_TASKS = 4
DEFAULT_MEMORY_SAFETY_MARGIN_MB = 512
DEFAULT_ALLOCATION_TIMEOUT_SECONDS = 30.0
DEFAULT_MONITOR_INTERVAL_SECONDS = 5.0

MAX_ALLOCATION_TIMEOUT_SECONDS = 600.0
MAX_RESERVATIONS_PER_DEVICE = 1000


# ============================================================
# Exceptions
# ============================================================

class GPUManagerError(Exception):
    """Base GPU management exception."""


class GPUValidationError(GPUManagerError):
    """Invalid GPU configuration or request."""


class GPUUnavailableError(GPUManagerError):
    """No suitable GPU is available."""


class GPUAllocationError(GPUManagerError):
    """GPU allocation failed."""


class GPUAllocationTimeoutError(GPUAllocationError):
    """GPU allocation deadline exceeded."""


class GPUMemoryError(GPUAllocationError):
    """Insufficient available GPU memory."""


class GPUReservationNotFoundError(GPUManagerError):
    """GPU reservation does not exist."""


class GPUManagerClosedError(GPUManagerError):
    """GPU manager has been closed."""


# ============================================================
# Utilities
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def bytes_to_mb(value: int | float) -> float:
    return float(value) / BYTES_PER_MIB


def mb_to_bytes(value: int | float) -> int:
    return int(float(value) * BYTES_PER_MIB)


def validate_nonnegative(
    value: int | float,
    name: str,
) -> float:

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise GPUValidationError(
            f"{name} must be a finite nonnegative number"
        )

    return float(value)


def validate_positive(
    value: int | float,
    name: str,
) -> float:

    result = validate_nonnegative(value, name)

    if result <= 0:
        raise GPUValidationError(
            f"{name} must be positive"
        )

    return result


# ============================================================
# Enumerations
# ============================================================

class ComputeDeviceType(str, Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class DeviceHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class AllocationPriority(IntEnum):
    LOW = 10
    NORMAL = 20
    HIGH = 30
    CRITICAL = 40


class AllocationStatus(str, Enum):
    ACTIVE = "active"
    RELEASED = "released"
    EXPIRED = "expired"


class DeviceSelectionStrategy(str, Enum):
    MOST_FREE_MEMORY = "most_free_memory"
    LEAST_RESERVED_MEMORY = "least_reserved_memory"
    LEAST_ACTIVE_TASKS = "least_active_tasks"
    BALANCED = "balanced"


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True, slots=True)
class GPUManagerConfiguration:

    enable_cuda: bool = True
    enable_mps: bool = True
    allow_cpu_fallback: bool = True

    max_concurrent_tasks_per_device: int = (
        DEFAULT_MAX_CONCURRENT_TASKS
    )

    memory_safety_margin_mb: int = (
        DEFAULT_MEMORY_SAFETY_MARGIN_MB
    )

    allocation_timeout_seconds: float = (
        DEFAULT_ALLOCATION_TIMEOUT_SECONDS
    )

    monitor_interval_seconds: float = (
        DEFAULT_MONITOR_INTERVAL_SECONDS
    )

    selection_strategy: DeviceSelectionStrategy = (
        DeviceSelectionStrategy.BALANCED
    )

    enable_nvml: bool = True

    enable_cuda_cache_cleanup: bool = True

    def __post_init__(self) -> None:

        if self.max_concurrent_tasks_per_device < 1:
            raise GPUValidationError(
                "Device concurrency must be positive"
            )

        validate_nonnegative(
            self.memory_safety_margin_mb,
            "memory_safety_margin_mb",
        )

        validate_positive(
            self.allocation_timeout_seconds,
            "allocation_timeout_seconds",
        )

        if (
            self.allocation_timeout_seconds
            > MAX_ALLOCATION_TIMEOUT_SECONDS
        ):
            raise GPUValidationError(
                "Allocation timeout exceeds maximum"
            )

        validate_positive(
            self.monitor_interval_seconds,
            "monitor_interval_seconds",
        )


# ============================================================
# Data Models
# ============================================================

@dataclass(frozen=True, slots=True)
class GPUAllocationScope:

    tenant_id: str
    user_id: str

    model_name: str | None = None
    workflow_id: str | None = None
    task_id: str | None = None

    def __post_init__(self) -> None:

        if not self.tenant_id.strip():
            raise GPUValidationError(
                "Tenant ID is required"
            )

        if not self.user_id.strip():
            raise GPUValidationError(
                "User ID is required"
            )


@dataclass(slots=True)
class ComputeDevice:

    device_id: str
    device_type: ComputeDeviceType
    name: str

    index: int | None = None

    total_memory_mb: float | None = None
    free_memory_mb: float | None = None
    used_memory_mb: float | None = None

    utilization_percent: float | None = None

    temperature_celsius: float | None = None

    reserved_memory_mb: float = 0.0

    active_tasks: int = 0

    health: DeviceHealth = DeviceHealth.UNKNOWN

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    last_updated_at: datetime = field(
        default_factory=utc_now
    )

    @property
    def available_memory_mb(self) -> float | None:

        if self.free_memory_mb is None:
            return None

        return max(
            0.0,
            self.free_memory_mb
            - self.reserved_memory_mb,
        )


@dataclass(frozen=True, slots=True)
class GPUAllocationRequest:

    scope: GPUAllocationScope

    requested_memory_mb: int = 0

    preferred_device: ComputeDeviceType | None = None

    preferred_device_id: str | None = None

    priority: AllocationPriority = (
        AllocationPriority.NORMAL
    )

    allow_cpu_fallback: bool = True

    timeout_seconds: float | None = None

    request_id: str = field(
        default_factory=generate_id
    )

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        validate_nonnegative(
            self.requested_memory_mb,
            "requested_memory_mb",
        )

        if self.timeout_seconds is not None:
            validate_positive(
                self.timeout_seconds,
                "timeout_seconds",
            )


@dataclass(slots=True)
class GPUReservation:

    reservation_id: str

    request_id: str

    device_id: str

    device_type: ComputeDeviceType

    tenant_id: str
    user_id: str

    model_name: str | None

    reserved_memory_mb: int

    priority: AllocationPriority

    status: AllocationStatus = (
        AllocationStatus.ACTIVE
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    released_at: datetime | None = None

    @property
    def torch_device(self) -> str:

        return self.device_id


@dataclass(slots=True)
class GPUManagerStatistics:

    total_allocation_requests: int = 0
    successful_allocations: int = 0
    failed_allocations: int = 0
    timed_out_allocations: int = 0

    cpu_fallback_allocations: int = 0

    released_allocations: int = 0

    peak_active_reservations: int = 0

    @property
    def allocation_success_rate(self) -> float:

        if self.total_allocation_requests == 0:
            return 0.0

        return (
            self.successful_allocations
            / self.total_allocation_requests
        )


# ============================================================
# Protocols
# ============================================================

class GPUAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        ...


class GPUAuditSink(Protocol):

    async def record(
        self,
        *,
        tenant_id: str,
        user_id: str,
        action: str,
        metadata: Mapping[str, Any],
    ) -> None:
        ...


# ============================================================
# Hardware Discovery
# ============================================================

class HardwareDiscovery:

    def __init__(
        self,
        config: GPUManagerConfiguration,
    ) -> None:

        self.config = config

        self._nvml_initialized = False

    def initialize_nvml(self) -> None:

        if (
            not self.config.enable_nvml
            or pynvml is None
        ):
            return

        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True

        except Exception as exc:
            logger.warning(
                "NVML initialization unavailable: %s",
                type(exc).__name__,
            )

    def shutdown_nvml(self) -> None:

        if (
            self._nvml_initialized
            and pynvml is not None
        ):
            try:
                pynvml.nvmlShutdown()
            except Exception:
                logger.exception(
                    "NVML shutdown failed"
                )

        self._nvml_initialized = False

    def discover(self) -> list[ComputeDevice]:

        devices: list[ComputeDevice] = []

        if (
            self.config.enable_cuda
            and torch is not None
            and torch.cuda.is_available()
        ):
            for index in range(
                torch.cuda.device_count()
            ):

                try:

                    properties = (
                        torch.cuda.get_device_properties(
                            index
                        )
                    )

                    total_memory_mb = bytes_to_mb(
                        properties.total_memory
                    )

                    device = ComputeDevice(
                        device_id=f"cuda:{index}",
                        device_type=(
                            ComputeDeviceType.CUDA
                        ),
                        name=properties.name,
                        index=index,
                        total_memory_mb=total_memory_mb,
                        health=DeviceHealth.HEALTHY,
                        metadata={
                            "compute_capability": (
                                f"{properties.major}."
                                f"{properties.minor}"
                            ),
                        },
                    )

                    devices.append(device)

                except Exception:
                    logger.exception(
                        "Failed to discover CUDA device %s",
                        index,
                    )

        if (
            self.config.enable_mps
            and torch is not None
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):

            devices.append(
                ComputeDevice(
                    device_id="mps",
                    device_type=(
                        ComputeDeviceType.MPS
                    ),
                    name="Apple Metal Performance Shaders",
                    health=DeviceHealth.HEALTHY,
                )
            )

        cpu_memory_mb: float | None = None

        if psutil is not None:
            try:
                cpu_memory_mb = bytes_to_mb(
                    psutil.virtual_memory().total
                )
            except Exception:
                pass

        devices.append(
            ComputeDevice(
                device_id="cpu",
                device_type=ComputeDeviceType.CPU,
                name=platform.processor()
                or platform.machine()
                or "CPU",
                total_memory_mb=cpu_memory_mb,
                health=DeviceHealth.HEALTHY,
                metadata={
                    "logical_cpu_count": os.cpu_count(),
                    "platform": platform.system(),
                },
            )
        )

        return devices

    def update_metrics(
        self,
        device: ComputeDevice,
    ) -> None:

        if device.device_type == ComputeDeviceType.CPU:

            if psutil is not None:

                try:
                    memory = psutil.virtual_memory()

                    device.total_memory_mb = bytes_to_mb(
                        memory.total
                    )

                    device.free_memory_mb = bytes_to_mb(
                        memory.available
                    )

                    device.used_memory_mb = bytes_to_mb(
                        memory.used
                    )

                    device.utilization_percent = (
                        psutil.cpu_percent(
                            interval=None
                        )
                    )

                except Exception:
                    logger.exception(
                        "CPU metrics collection failed"
                    )

        elif device.device_type == ComputeDeviceType.CUDA:

            self._update_cuda_metrics(device)

        elif device.device_type == ComputeDeviceType.MPS:

            self._update_mps_metrics(device)

        device.last_updated_at = utc_now()

    def _update_cuda_metrics(
        self,
        device: ComputeDevice,
    ) -> None:

        if (
            torch is None
            or device.index is None
        ):
            device.health = DeviceHealth.UNAVAILABLE
            return

        try:

            if not torch.cuda.is_available():
                device.health = DeviceHealth.UNAVAILABLE
                return

            if self._nvml_initialized and pynvml is not None:

                handle = (
                    pynvml.nvmlDeviceGetHandleByIndex(
                        device.index
                    )
                )

                memory = (
                    pynvml.nvmlDeviceGetMemoryInfo(
                        handle
                    )
                )

                utilization = (
                    pynvml.nvmlDeviceGetUtilizationRates(
                        handle
                    )
                )

                device.total_memory_mb = bytes_to_mb(
                    memory.total
                )

                device.free_memory_mb = bytes_to_mb(
                    memory.free
                )

                device.used_memory_mb = bytes_to_mb(
                    memory.used
                )

                device.utilization_percent = float(
                    utilization.gpu
                )

                try:
                    device.temperature_celsius = float(
                        pynvml.nvmlDeviceGetTemperature(
                            handle,
                            pynvml.NVML_TEMPERATURE_GPU,
                        )
                    )
                except Exception:
                    device.temperature_celsius = None

            else:

                free_bytes, total_bytes = (
                    torch.cuda.mem_get_info(
                        device.index
                    )
                )

                device.total_memory_mb = bytes_to_mb(
                    total_bytes
                )

                device.free_memory_mb = bytes_to_mb(
                    free_bytes
                )

                device.used_memory_mb = bytes_to_mb(
                    total_bytes - free_bytes
                )

            device.health = DeviceHealth.HEALTHY

        except Exception:
            logger.exception(
                "CUDA metrics collection failed for %s",
                device.device_id,
            )

            device.health = DeviceHealth.UNKNOWN

    def _update_mps_metrics(
        self,
        device: ComputeDevice,
    ) -> None:

        if torch is None:
            device.health = DeviceHealth.UNAVAILABLE
            return

        try:

            if not torch.backends.mps.is_available():
                device.health = DeviceHealth.UNAVAILABLE
                return

            if hasattr(
                torch.mps,
                "current_allocated_memory",
            ):
                device.used_memory_mb = bytes_to_mb(
                    torch.mps.current_allocated_memory()
                )

            if hasattr(
                torch.mps,
                "recommended_max_memory",
            ):
                device.total_memory_mb = bytes_to_mb(
                    torch.mps.recommended_max_memory()
                )

            # MPS memory is shared with system RAM.
            # Treat free-memory estimates as advisory only.
            device.free_memory_mb = None

            device.health = DeviceHealth.HEALTHY

        except Exception:
            logger.exception(
                "MPS metrics collection failed"
            )

            device.health = DeviceHealth.UNKNOWN


# ============================================================
# GPU Manager
# ============================================================

class GPUManager:

    """
    Asynchronous GPU allocation manager.

    Memory reservations are scheduling estimates only.
    They do not reserve physical CUDA memory.

    This manager does not execute robot commands.
    """

    def __init__(
        self,
        *,
        authorizer: GPUAuthorizer,
        config: GPUManagerConfiguration | None = None,
        audit_sink: GPUAuditSink | None = None,
    ) -> None:

        self.config = (
            config or GPUManagerConfiguration()
        )

        self.authorizer = authorizer

        self.audit_sink = audit_sink

        self.discovery = HardwareDiscovery(
            self.config
        )

        self.devices: dict[
            str,
            ComputeDevice,
        ] = {}

        self.reservations: dict[
            str,
            GPUReservation,
        ] = {}

        self.statistics = GPUManagerStatistics()

        self._condition = asyncio.Condition()

        self._started = False
        self._closed = False

        self._monitor_task: (
            asyncio.Task[None] | None
        ) = None

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    async def start(self) -> None:

        async with self._condition:

            if self._closed:
                raise GPUManagerClosedError(
                    "GPU manager has been closed"
                )

            if self._started:
                return

            self.discovery.initialize_nvml()

            discovered = self.discovery.discover()

            self.devices = {
                device.device_id: device
                for device in discovered
            }

            self._started = True

        await self.refresh_metrics()

        self._monitor_task = asyncio.create_task(
            self._monitor_loop(),
            name="gpu-resource-monitor",
        )

        logger.info(
            "GPU manager initialized with %s devices",
            len(self.devices),
        )

    # --------------------------------------------------------
    # Refresh Metrics
    # --------------------------------------------------------

    async def refresh_metrics(self) -> None:

        if self._closed:
            return

        async with self._condition:

            for device in self.devices.values():
                self.discovery.update_metrics(device)

            self._condition.notify_all()

    # --------------------------------------------------------
    # Monitoring Loop
    # --------------------------------------------------------

    async def _monitor_loop(self) -> None:

        try:

            while not self._closed:

                await asyncio.sleep(
                    self.config.monitor_interval_seconds
                )

                await self.refresh_metrics()

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                "GPU monitoring loop failed"
            )

    # --------------------------------------------------------
    # Device Eligibility
    # --------------------------------------------------------

    def _is_device_eligible(
        self,
        *,
        device: ComputeDevice,
        request: GPUAllocationRequest,
    ) -> bool:

        if device.health == DeviceHealth.UNAVAILABLE:
            return False

        if request.preferred_device_id is not None:

            if (
                device.device_id
                != request.preferred_device_id
            ):
                return False

        if request.preferred_device is not None:

            if (
                device.device_type
                != request.preferred_device
            ):
                return False

        if (
            device.active_tasks
            >= self.config.max_concurrent_tasks_per_device
        ):
            return False

        if device.device_type == ComputeDeviceType.CPU:

            if request.requested_memory_mb <= 0:
                return True

            if device.available_memory_mb is None:
                return True

            return (
                device.available_memory_mb
                >= request.requested_memory_mb
            )

        if device.device_type == ComputeDeviceType.MPS:

            # MPS does not expose a reliable independent
            # GPU free-memory metric through PyTorch.
            # Use conservative concurrency limits.
            return True

        available = device.available_memory_mb

        if available is None:
            return False

        required = (
            request.requested_memory_mb
            + self.config.memory_safety_margin_mb
        )

        return available >= required

    # --------------------------------------------------------
    # Device Selection
    # --------------------------------------------------------

    def _select_device(
        self,
        request: GPUAllocationRequest,
    ) -> ComputeDevice | None:

        candidates = [
            device
            for device in self.devices.values()
            if self._is_device_eligible(
                device=device,
                request=request,
            )
        ]

        if not candidates:
            return None

        gpu_candidates = [
            device
            for device in candidates
            if device.device_type
            != ComputeDeviceType.CPU
        ]

        if gpu_candidates:
            candidates = gpu_candidates

        elif not (
            self.config.allow_cpu_fallback
            and request.allow_cpu_fallback
        ):
            return None

        strategy = self.config.selection_strategy

        if strategy == DeviceSelectionStrategy.MOST_FREE_MEMORY:

            candidates.sort(
                key=lambda device: (
                    -(
                        device.available_memory_mb
                        if device.available_memory_mb is not None
                        else -1
                    ),
                    device.active_tasks,
                )
            )

        elif strategy == DeviceSelectionStrategy.LEAST_RESERVED_MEMORY:

            candidates.sort(
                key=lambda device: (
                    device.reserved_memory_mb,
                    device.active_tasks,
                )
            )

        elif strategy == DeviceSelectionStrategy.LEAST_ACTIVE_TASKS:

            candidates.sort(
                key=lambda device: (
                    device.active_tasks,
                    device.reserved_memory_mb,
                )
            )

        else:

            candidates.sort(
                key=lambda device: (
                    device.active_tasks
                    / self.config.max_concurrent_tasks_per_device,
                    device.reserved_memory_mb,
                    -(
                        device.available_memory_mb
                        if device.available_memory_mb is not None
                        else 0
                    ),
                )
            )

        return candidates[0]

    # --------------------------------------------------------
    # Allocate
    # --------------------------------------------------------

    async def allocate(
        self,
        request: GPUAllocationRequest,
    ) -> GPUReservation:

        await self.authorizer.require_permission(
            tenant_id=request.scope.tenant_id,
            user_id=request.scope.user_id,
            permission="ml:gpu:allocate",
        )

        if not self._started:
            await self.start()

        timeout = (
            request.timeout_seconds
            or self.config.allocation_timeout_seconds
        )

        loop = asyncio.get_running_loop()

        deadline = loop.time() + timeout

        self.statistics.total_allocation_requests += 1

        while True:

            async with self._condition:

                if self._closed:
                    raise GPUManagerClosedError(
                        "GPU manager has been closed"
                    )

                device = self._select_device(
                    request
                )

                if device is not None:

                    reservation = GPUReservation(
                        reservation_id=generate_id(),
                        request_id=request.request_id,
                        device_id=device.device_id,
                        device_type=device.device_type,
                        tenant_id=(
                            request.scope.tenant_id
                        ),
                        user_id=request.scope.user_id,
                        model_name=(
                            request.scope.model_name
                        ),
                        reserved_memory_mb=(
                            request.requested_memory_mb
                        ),
                        priority=request.priority,
                    )

                    device.reserved_memory_mb += (
                        request.requested_memory_mb
                    )

                    device.active_tasks += 1

                    self.reservations[
                        reservation.reservation_id
                    ] = reservation

                    self.statistics.successful_allocations += 1

                    if (
                        device.device_type
                        == ComputeDeviceType.CPU
                        and request.preferred_device
                        != ComputeDeviceType.CPU
                    ):
                        self.statistics.cpu_fallback_allocations += 1

                    self.statistics.peak_active_reservations = max(
                        self.statistics.peak_active_reservations,
                        len(self.reservations),
                    )

                    break

                remaining = deadline - loop.time()

                if remaining <= 0:

                    self.statistics.timed_out_allocations += 1
                    self.statistics.failed_allocations += 1

                    raise GPUAllocationTimeoutError(
                        "No suitable compute device became available"
                    )

                try:
                    await asyncio.wait_for(
                        self._condition.wait(),
                        timeout=remaining,
                    )

                except asyncio.TimeoutError:

                    self.statistics.timed_out_allocations += 1
                    self.statistics.failed_allocations += 1

                    raise GPUAllocationTimeoutError(
                        "GPU allocation timed out"
                    ) from None

        await self._audit(
            scope=request.scope,
            action="ml.gpu.allocated",
            metadata={
                "reservation_id": (
                    reservation.reservation_id
                ),
                "device_id": reservation.device_id,
                "device_type": (
                    reservation.device_type.value
                ),
                "reserved_memory_mb": (
                    reservation.reserved_memory_mb
                ),
            },
        )

        return reservation

    # --------------------------------------------------------
    # Release
    # --------------------------------------------------------

    async def release(
        self,
        reservation_id: str,
        *,
        scope: GPUAllocationScope,
    ) -> None:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            permission="ml:gpu:release",
        )

        async with self._condition:

            reservation = self.reservations.get(
                reservation_id
            )

            if reservation is None:
                raise GPUReservationNotFoundError(
                    "GPU reservation not found"
                )

            if reservation.tenant_id != scope.tenant_id:
                raise GPUReservationNotFoundError(
                    "GPU reservation not found"
                )

            if reservation.user_id != scope.user_id:
                raise GPUReservationNotFoundError(
                    "GPU reservation not found"
                )

            device = self.devices.get(
                reservation.device_id
            )

            if device is not None:

                device.reserved_memory_mb = max(
                    0.0,
                    device.reserved_memory_mb
                    - reservation.reserved_memory_mb,
                )

                device.active_tasks = max(
                    0,
                    device.active_tasks - 1,
                )

            reservation.status = (
                AllocationStatus.RELEASED
            )

            reservation.released_at = utc_now()

            del self.reservations[
                reservation_id
            ]

            self.statistics.released_allocations += 1

            self._condition.notify_all()

        await self._audit(
            scope=scope,
            action="ml.gpu.released",
            metadata={
                "reservation_id": reservation_id,
                "device_id": reservation.device_id,
            },
        )

    # --------------------------------------------------------
    # Context Manager
    # --------------------------------------------------------

    @asynccontextmanager
    async def reservation(
        self,
        request: GPUAllocationRequest,
    ) -> AsyncIterator[GPUReservation]:

        allocation = await self.allocate(
            request
        )

        try:
            yield allocation

        finally:
            await self.release(
                allocation.reservation_id,
                scope=request.scope,
            )

    # --------------------------------------------------------
    # Get Devices
    # --------------------------------------------------------

    async def list_devices(
        self,
    ) -> list[dict[str, Any]]:

        if not self._started:
            await self.start()

        await self.refresh_metrics()

        async with self._condition:

            return [
                {
                    "device_id": device.device_id,
                    "device_type": (
                        device.device_type.value
                    ),
                    "name": device.name,
                    "index": device.index,
                    "total_memory_mb": (
                        device.total_memory_mb
                    ),
                    "free_memory_mb": (
                        device.free_memory_mb
                    ),
                    "used_memory_mb": (
                        device.used_memory_mb
                    ),
                    "available_memory_mb": (
                        device.available_memory_mb
                    ),
                    "reserved_memory_mb": (
                        device.reserved_memory_mb
                    ),
                    "utilization_percent": (
                        device.utilization_percent
                    ),
                    "temperature_celsius": (
                        device.temperature_celsius
                    ),
                    "active_tasks": (
                        device.active_tasks
                    ),
                    "health": device.health.value,
                    "last_updated_at": (
                        device.last_updated_at.isoformat()
                    ),
                }
                for device in self.devices.values()
            ]

    # --------------------------------------------------------
    # Active Reservations
    # --------------------------------------------------------

    async def list_reservations(
        self,
        *,
        scope: GPUAllocationScope,
    ) -> list[dict[str, Any]]:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            permission="ml:gpu:read",
        )

        async with self._condition:

            return [
                {
                    "reservation_id": (
                        reservation.reservation_id
                    ),
                    "request_id": reservation.request_id,
                    "device_id": reservation.device_id,
                    "device_type": (
                        reservation.device_type.value
                    ),
                    "model_name": reservation.model_name,
                    "reserved_memory_mb": (
                        reservation.reserved_memory_mb
                    ),
                    "priority": int(
                        reservation.priority
                    ),
                    "status": reservation.status.value,
                    "created_at": (
                        reservation.created_at.isoformat()
                    ),
                }
                for reservation in self.reservations.values()
                if reservation.tenant_id == scope.tenant_id
                and reservation.user_id == scope.user_id
            ]

    # --------------------------------------------------------
    # CUDA Cache Cleanup
    # --------------------------------------------------------

    async def cleanup_cuda_cache(
        self,
        *,
        scope: GPUAllocationScope,
    ) -> bool:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            permission="ml:gpu:manage",
        )

        if not self.config.enable_cuda_cache_cleanup:
            return False

        if torch is None:
            return False

        if not torch.cuda.is_available():
            return False

        try:

            torch.cuda.empty_cache()

            await self._audit(
                scope=scope,
                action="ml.gpu.cache_cleanup",
                metadata={
                    "device_type": "cuda",
                },
            )

            return True

        except Exception:
            logger.exception(
                "CUDA cache cleanup failed"
            )

            return False

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def get_statistics(
        self,
    ) -> dict[str, Any]:

        return {
            "total_allocation_requests": (
                self.statistics.total_allocation_requests
            ),
            "successful_allocations": (
                self.statistics.successful_allocations
            ),
            "failed_allocations": (
                self.statistics.failed_allocations
            ),
            "timed_out_allocations": (
                self.statistics.timed_out_allocations
            ),
            "cpu_fallback_allocations": (
                self.statistics.cpu_fallback_allocations
            ),
            "released_allocations": (
                self.statistics.released_allocations
            ),
            "peak_active_reservations": (
                self.statistics.peak_active_reservations
            ),
            "active_reservations": len(
                self.reservations
            ),
            "allocation_success_rate": (
                self.statistics.allocation_success_rate
            ),
            "device_count": len(
                self.devices
            ),
        }

    # --------------------------------------------------------
    # Generate Report
    # --------------------------------------------------------

    async def generate_report(
        self,
    ) -> dict[str, Any]:

        return {
            "generated_at": utc_now().isoformat(),
            "platform": (
                "Physical AI & Autonomous Robotics"
            ),
            "configuration": {
                "enable_cuda": self.config.enable_cuda,
                "enable_mps": self.config.enable_mps,
                "allow_cpu_fallback": (
                    self.config.allow_cpu_fallback
                ),
                "max_concurrent_tasks_per_device": (
                    self.config.max_concurrent_tasks_per_device
                ),
                "memory_safety_margin_mb": (
                    self.config.memory_safety_margin_mb
                ),
                "selection_strategy": (
                    self.config.selection_strategy.value
                ),
            },
            "devices": await self.list_devices(),
            "statistics": self.get_statistics(),
            "safety": {
                "physical_robot_actuation": False,
                "robot_command_execution": False,
                "emergency_stop_reset": False,
            },
        }

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    async def _audit(
        self,
        *,
        scope: GPUAllocationScope,
        action: str,
        metadata: Mapping[str, Any],
    ) -> None:

        if self.audit_sink is None:
            return

        await self.audit_sink.record(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            action=action,
            metadata=metadata,
        )

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    async def shutdown(self) -> None:

        async with self._condition:

            if self._closed:
                return

            self._closed = True

            self._condition.notify_all()

        if self._monitor_task is not None:

            self._monitor_task.cancel()

            await asyncio.gather(
                self._monitor_task,
                return_exceptions=True,
            )

        async with self._condition:

            for reservation in self.reservations.values():

                reservation.status = (
                    AllocationStatus.EXPIRED
                )

                reservation.released_at = utc_now()

            self.reservations.clear()

            for device in self.devices.values():

                device.reserved_memory_mb = 0.0

                device.active_tasks = 0

        self.discovery.shutdown_nvml()

        logger.info(
            "GPU manager shut down"
        )


# ============================================================
# Deny-by-Default Authorizer
# ============================================================

class DenyByDefaultGPUAuthorizer:

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:

        raise PermissionError(
            "GPU authorization provider not configured"
        )


# ============================================================
# Factory
# ============================================================

def create_gpu_manager(
    *,
    authorizer: GPUAuthorizer,
    config: GPUManagerConfiguration | None = None,
    audit_sink: GPUAuditSink | None = None,
) -> GPUManager:

    return GPUManager(
        authorizer=authorizer,
        config=config,
        audit_sink=audit_sink,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "GPUManagerError",
    "GPUValidationError",
    "GPUUnavailableError",
    "GPUAllocationError",
    "GPUAllocationTimeoutError",
    "GPUMemoryError",
    "GPUReservationNotFoundError",
    "GPUManagerClosedError",
    "ComputeDeviceType",
    "DeviceHealth",
    "AllocationPriority",
    "AllocationStatus",
    "DeviceSelectionStrategy",
    "GPUManagerConfiguration",
    "GPUAllocationScope",
    "ComputeDevice",
    "GPUAllocationRequest",
    "GPUReservation",
    "GPUManagerStatistics",
    "GPUAuthorizer",
    "GPUAuditSink",
    "HardwareDiscovery",
    "GPUManager",
    "DenyByDefaultGPUAuthorizer",
    "create_gpu_manager",
]
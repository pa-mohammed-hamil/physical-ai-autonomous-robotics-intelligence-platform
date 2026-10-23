"""
ml/inference/batching.py

Physical AI & Autonomous Robotics Intelligence Platform
Asynchronous ML Inference Batching

Features:
- Dynamic request batching
- Maximum batch size and wait-time controls
- Priority-aware scheduling
- Multiple model support
- GPU/CPU backend integration
- Per-request deadlines
- Backpressure and queue limits
- Cancellation handling
- Batch inference retries
- Input/output validation
- Batch statistics
- Graceful shutdown
- Simulation-only restrictions for robotic action models

Python 3.11+
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
import uuid

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum, Enum
from typing import Any, Callable, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

DEFAULT_MAX_BATCH_SIZE = 16
DEFAULT_MAX_WAIT_MS = 25
DEFAULT_MAX_QUEUE_SIZE = 1000
DEFAULT_MAX_CONCURRENT_BATCHES = 2
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 1

MAX_BATCH_SIZE_LIMIT = 512
MAX_QUEUE_SIZE_LIMIT = 100_000
MAX_RETRIES_LIMIT = 3


# ============================================================
# Exceptions
# ============================================================

class BatchingError(Exception):
    """Base batching exception."""


class BatchValidationError(BatchingError):
    """Invalid inference request or configuration."""


class BatchQueueFullError(BatchingError):
    """Inference queue capacity exceeded."""


class BatchTimeoutError(BatchingError):
    """Inference request exceeded its deadline."""


class BatchInferenceError(BatchingError):
    """Model batch execution failed."""


class BatchOutputError(BatchingError):
    """Model returned invalid batch outputs."""


class BatchShutdownError(BatchingError):
    """Batching service is shutting down."""


class ModelNotRegisteredError(BatchingError):
    """Requested model is unavailable."""


class UnsafeInferenceError(BatchingError):
    """Inference request violates robot safety policy."""


# ============================================================
# Utilities
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def validate_finite_positive(
    value: float,
    name: str,
) -> float:

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise BatchValidationError(
            f"{name} must be a finite positive number"
        )

    return float(value)


# ============================================================
# Enumerations
# ============================================================

class InferencePriority(IntEnum):
    LOW = 10
    NORMAL = 20
    HIGH = 30
    CRITICAL = 40


class InferenceStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class ModelCategory(str, Enum):
    PERCEPTION = "perception"
    VLM = "vlm"
    VLA = "vla"
    WORLD_MODEL = "world_model"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    ANOMALY = "anomaly"
    FAILURE_PREDICTION = "failure_prediction"
    RUL = "rul"
    PART_INSPECTION = "part_inspection"
    CUSTOM = "custom"


class InferenceDevice(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


# ============================================================
# Data Models
# ============================================================

@dataclass(frozen=True, slots=True)
class BatchConfiguration:

    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    max_wait_ms: int = DEFAULT_MAX_WAIT_MS

    max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE

    max_concurrent_batches: int = (
        DEFAULT_MAX_CONCURRENT_BATCHES
    )

    request_timeout_seconds: float = (
        DEFAULT_REQUEST_TIMEOUT_SECONDS
    )

    max_retries: int = DEFAULT_MAX_RETRIES

    retry_backoff_seconds: float = 0.1

    def __post_init__(self) -> None:

        if not 1 <= self.max_batch_size <= MAX_BATCH_SIZE_LIMIT:
            raise BatchValidationError(
                "Invalid maximum batch size"
            )

        if not 0 <= self.max_wait_ms <= 60_000:
            raise BatchValidationError(
                "Invalid batch wait time"
            )

        if not 1 <= self.max_queue_size <= MAX_QUEUE_SIZE_LIMIT:
            raise BatchValidationError(
                "Invalid queue size"
            )

        if self.max_concurrent_batches < 1:
            raise BatchValidationError(
                "Concurrency must be positive"
            )

        validate_finite_positive(
            self.request_timeout_seconds,
            "request_timeout_seconds",
        )

        if not 0 <= self.max_retries <= MAX_RETRIES_LIMIT:
            raise BatchValidationError(
                "Invalid retry limit"
            )

        if (
            not math.isfinite(self.retry_backoff_seconds)
            or self.retry_backoff_seconds < 0
        ):
            raise BatchValidationError(
                "Invalid retry backoff"
            )


@dataclass(frozen=True, slots=True)
class InferenceScope:
    tenant_id: str
    user_id: str
    robot_id: str | None = None


@dataclass(slots=True)
class InferenceRequest:

    model_name: str
    inputs: Any

    scope: InferenceScope

    request_id: str = field(
        default_factory=generate_id
    )

    priority: InferencePriority = (
        InferencePriority.NORMAL
    )

    timeout_seconds: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    simulation_only: bool = True

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:

        if not self.model_name.strip():
            raise BatchValidationError(
                "Model name cannot be empty"
            )

        if not self.scope.tenant_id.strip():
            raise BatchValidationError(
                "Tenant ID is required"
            )

        if not self.scope.user_id.strip():
            raise BatchValidationError(
                "User ID is required"
            )

        if self.timeout_seconds is not None:
            validate_finite_positive(
                self.timeout_seconds,
                "timeout_seconds",
            )


@dataclass(slots=True)
class InferenceResult:

    request_id: str
    model_name: str

    status: InferenceStatus

    output: Any = None

    error: str | None = None

    batch_id: str | None = None

    batch_size: int = 0

    queue_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    device: InferenceDevice = (
        InferenceDevice.CPU
    )

    completed_at: datetime = field(
        default_factory=utc_now
    )

    physical_execution_authorized: bool = False


@dataclass(slots=True)
class BatchStatistics:

    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    timed_out_requests: int = 0
    cancelled_requests: int = 0

    total_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0

    total_batch_items: int = 0

    total_queue_latency_ms: float = 0.0
    total_inference_latency_ms: float = 0.0

    @property
    def average_batch_size(self) -> float:

        if self.total_batches == 0:
            return 0.0

        return (
            self.total_batch_items
            / self.total_batches
        )

    @property
    def average_queue_latency_ms(self) -> float:

        if self.completed_requests == 0:
            return 0.0

        return (
            self.total_queue_latency_ms
            / self.completed_requests
        )

    @property
    def average_inference_latency_ms(self) -> float:

        if self.successful_batches == 0:
            return 0.0

        return (
            self.total_inference_latency_ms
            / self.successful_batches
        )


# ============================================================
# Model Backend Protocol
# ============================================================

class BatchInferenceBackend(Protocol):

    async def predict_batch(
        self,
        inputs: Sequence[Any],
    ) -> Sequence[Any]:
        ...


class InferenceAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        ...


class InferenceAuditSink(Protocol):

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
# Model Registration
# ============================================================

@dataclass(slots=True)
class RegisteredInferenceModel:

    name: str

    backend: BatchInferenceBackend

    category: ModelCategory

    device: InferenceDevice = (
        InferenceDevice.AUTO
    )

    enabled: bool = True

    version: str = "1.0.0"

    max_batch_size: int | None = None

    def __post_init__(self) -> None:

        if not self.name.strip():
            raise BatchValidationError(
                "Model name is required"
            )

        if self.max_batch_size is not None:
            if not (
                1
                <= self.max_batch_size
                <= MAX_BATCH_SIZE_LIMIT
            ):
                raise BatchValidationError(
                    "Invalid model batch limit"
                )


class InferenceModelRegistry:

    def __init__(self) -> None:
        self._models: dict[
            str,
            RegisteredInferenceModel,
        ] = {}

    def register(
        self,
        model: RegisteredInferenceModel,
    ) -> None:

        if model.name in self._models:
            raise BatchValidationError(
                f"Model already registered: {model.name}"
            )

        self._models[model.name] = model

    def get(
        self,
        model_name: str,
    ) -> RegisteredInferenceModel:

        model = self._models.get(model_name)

        if model is None or not model.enabled:
            raise ModelNotRegisteredError(
                f"Model unavailable: {model_name}"
            )

        return model

    def list_models(
        self,
    ) -> list[RegisteredInferenceModel]:

        return list(self._models.values())


# ============================================================
# Internal Queue Item
# ============================================================

@dataclass(slots=True)
class _QueuedInference:

    request: InferenceRequest

    future: asyncio.Future[InferenceResult]

    enqueued_at: float

    deadline: float

    sequence: int


# ============================================================
# Dynamic Batch Scheduler
# ============================================================

class DynamicBatchScheduler:

    """
    Priority-aware asynchronous inference batching.

    All requests within a batch use the same registered
    model and tenant partition.

    The scheduler only performs model inference.
    It never sends commands to physical robots.
    """

    def __init__(
        self,
        *,
        registry: InferenceModelRegistry,
        authorizer: InferenceAuthorizer,
        config: BatchConfiguration | None = None,
        audit_sink: InferenceAuditSink | None = None,
    ) -> None:

        self.registry = registry
        self.authorizer = authorizer

        self.config = config or BatchConfiguration()

        self.audit_sink = audit_sink

        self.statistics = BatchStatistics()

        self._queues: dict[
            tuple[str, str],
            deque[_QueuedInference],
        ] = defaultdict(deque)

        self._condition = asyncio.Condition()

        self._semaphore = asyncio.Semaphore(
            self.config.max_concurrent_batches
        )

        self._workers: set[
            asyncio.Task[None]
        ] = set()

        self._active_batches: set[
            asyncio.Task[None]
        ] = set()

        self._sequence = 0
        self._queued_count = 0

        self._running = False
        self._closing = False

    # --------------------------------------------------------
    # Start Scheduler
    # --------------------------------------------------------

    async def start(self) -> None:

        async with self._condition:

            if self._closing:
                raise BatchShutdownError(
                    "Scheduler has been closed"
                )

            if self._running:
                return

            self._running = True

            worker = asyncio.create_task(
                self._dispatch_loop(),
                name="ml-inference-batch-dispatcher",
            )

            self._workers.add(worker)

            worker.add_done_callback(
                self._workers.discard
            )

    # --------------------------------------------------------
    # Submit Inference Request
    # --------------------------------------------------------

    async def submit(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:

        model = self.registry.get(
            request.model_name
        )

        await self.authorizer.require_permission(
            tenant_id=request.scope.tenant_id,
            user_id=request.scope.user_id,
            permission="ml:inference",
        )

        self._validate_safety(
            request=request,
            model=model,
        )

        if not self._running:
            await self.start()

        loop = asyncio.get_running_loop()

        future: asyncio.Future[InferenceResult] = (
            loop.create_future()
        )

        timeout = (
            request.timeout_seconds
            if request.timeout_seconds is not None
            else self.config.request_timeout_seconds
        )

        now = loop.time()

        async with self._condition:

            if self._closing:
                raise BatchShutdownError(
                    "Scheduler is shutting down"
                )

            if (
                self._queued_count
                >= self.config.max_queue_size
            ):
                raise BatchQueueFullError(
                    "Inference queue is full"
                )

            self._sequence += 1

            item = _QueuedInference(
                request=request,
                future=future,
                enqueued_at=now,
                deadline=now + timeout,
                sequence=self._sequence,
            )

            queue_key = (
                request.scope.tenant_id,
                request.model_name,
            )

            self._queues[queue_key].append(item)

            self._queued_count += 1

            self.statistics.total_requests += 1

            self._condition.notify_all()

        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=timeout,
            )

        except asyncio.TimeoutError:

            if future.done():
                return future.result()

            result = InferenceResult(
                request_id=request.request_id,
                model_name=request.model_name,
                status=InferenceStatus.TIMED_OUT,
                error="Inference deadline exceeded",
            )

            future.set_result(result)

            self.statistics.timed_out_requests += 1

            return result

        except asyncio.CancelledError:

            if not future.done():
                future.cancel()

            self.statistics.cancelled_requests += 1

            raise

    # --------------------------------------------------------
    # Safety Validation
    # --------------------------------------------------------

    @staticmethod
    def _validate_safety(
        *,
        request: InferenceRequest,
        model: RegisteredInferenceModel,
    ) -> None:

        if model.category in (
            ModelCategory.VLA,
            ModelCategory.NAVIGATION,
            ModelCategory.MANIPULATION,
        ):
            if not request.simulation_only:
                raise UnsafeInferenceError(
                    "Robot action-model inference "
                    "must remain simulation-only"
                )

        if request.metadata.get(
            "physical_execution_authorized"
        ) is True:
            raise UnsafeInferenceError(
                "Inference cannot authorize "
                "physical robot execution"
            )

    # --------------------------------------------------------
    # Select Batch
    # --------------------------------------------------------

    def _select_batch(
        self,
    ) -> tuple[
        RegisteredInferenceModel,
        list[_QueuedInference],
    ] | None:

        now = asyncio.get_running_loop().time()

        candidates: list[
            tuple[
                int,
                float,
                tuple[str, str],
            ]
        ] = []

        for queue_key, queue in self._queues.items():

            while queue and (
                queue[0].future.done()
                or queue[0].deadline <= now
            ):
                expired = queue.popleft()

                self._queued_count -= 1

                if not expired.future.done():
                    expired.future.set_result(
                        InferenceResult(
                            request_id=(
                                expired.request.request_id
                            ),
                            model_name=(
                                expired.request.model_name
                            ),
                            status=(
                                InferenceStatus.TIMED_OUT
                            ),
                            error=(
                                "Request expired in queue"
                            ),
                        )
                    )

                    self.statistics.timed_out_requests += 1

            if not queue:
                continue

            highest_priority = max(
                int(item.request.priority)
                for item in queue
            )

            oldest = min(
                item.enqueued_at
                for item in queue
            )

            candidates.append(
                (
                    -highest_priority,
                    oldest,
                    queue_key,
                )
            )

        if not candidates:
            return None

        candidates.sort()

        _, _, selected_key = candidates[0]

        queue = self._queues[selected_key]

        model = self.registry.get(
            selected_key[1]
        )

        batch_limit = min(
            self.config.max_batch_size,
            model.max_batch_size
            or self.config.max_batch_size,
        )

        items = sorted(
            queue,
            key=lambda item: (
                -int(item.request.priority),
                item.sequence,
            ),
        )

        selected = items[:batch_limit]

        selected_sequences = {
            item.sequence
            for item in selected
        }

        self._queues[selected_key] = deque(
            item
            for item in queue
            if item.sequence not in selected_sequences
        )

        self._queued_count -= len(selected)

        return model, selected

    # --------------------------------------------------------
    # Dispatch Loop
    # --------------------------------------------------------

    async def _dispatch_loop(self) -> None:

        try:
            while True:

                async with self._condition:

                    while (
                        self._queued_count == 0
                        and self._running
                    ):
                        await self._condition.wait()

                    if not self._running:
                        return

                    await_ms = (
                        self.config.max_wait_ms
                    )

                    if await_ms > 0:
                        try:
                            await asyncio.wait_for(
                                self._condition.wait(),
                                timeout=await_ms / 1000,
                            )
                        except asyncio.TimeoutError:
                            pass

                    selection = self._select_batch()

                if selection is None:
                    await asyncio.sleep(0)
                    continue

                model, items = selection

                task = asyncio.create_task(
                    self._execute_batch(
                        model=model,
                        items=items,
                    )
                )

                self._active_batches.add(task)

                task.add_done_callback(
                    self._active_batches.discard
                )

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                "Inference dispatcher crashed"
            )
            raise

    # --------------------------------------------------------
    # Execute Batch
    # --------------------------------------------------------

    async def _execute_batch(
        self,
        *,
        model: RegisteredInferenceModel,
        items: Sequence[_QueuedInference],
    ) -> None:

        async with self._semaphore:

            loop = asyncio.get_running_loop()

            active_items = [
                item
                for item in items
                if (
                    not item.future.done()
                    and item.deadline > loop.time()
                )
            ]

            if not active_items:
                return

            batch_id = generate_id()

            started = time.perf_counter()

            self.statistics.total_batches += 1
            self.statistics.total_batch_items += (
                len(active_items)
            )

            inputs = [
                item.request.inputs
                for item in active_items
            ]

            outputs: Sequence[Any] | None = None
            last_error: Exception | None = None

            for attempt in range(
                self.config.max_retries + 1
            ):

                try:

                    remaining = min(
                        item.deadline - loop.time()
                        for item in active_items
                        if not item.future.done()
                    )

                    if remaining <= 0:
                        raise BatchTimeoutError(
                            "Batch deadline exceeded"
                        )

                    outputs = await asyncio.wait_for(
                        model.backend.predict_batch(
                            inputs
                        ),
                        timeout=remaining,
                    )

                    if (
                        isinstance(
                            outputs,
                            (str, bytes, Mapping),
                        )
                        or not isinstance(
                            outputs,
                            Sequence,
                        )
                    ):
                        raise BatchOutputError(
                            "Batch backend must return "
                            "a sequence of outputs"
                        )

                    if len(outputs) != len(active_items):
                        raise BatchOutputError(
                            "Batch output count mismatch"
                        )

                    break

                except Exception as exc:

                    last_error = exc

                    if (
                        attempt
                        >= self.config.max_retries
                    ):
                        break

                    await asyncio.sleep(
                        self.config.retry_backoff_seconds
                        * (2 ** attempt)
                    )

            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000

            self.statistics.total_inference_latency_ms += (
                elapsed_ms
            )

            if outputs is None:

                self.statistics.failed_batches += 1

                for item in active_items:

                    if item.future.done():
                        continue

                    result = InferenceResult(
                        request_id=(
                            item.request.request_id
                        ),
                        model_name=model.name,
                        status=InferenceStatus.FAILED,
                        error=str(last_error),
                        batch_id=batch_id,
                        batch_size=len(active_items),
                        inference_latency_ms=elapsed_ms,
                        device=model.device,
                    )

                    item.future.set_result(result)

                    self.statistics.failed_requests += 1

                return

            self.statistics.successful_batches += 1

            for item, output in zip(
                active_items,
                outputs,
            ):

                if item.future.done():
                    continue

                if (
                    isinstance(output, Mapping)
                    and output.get(
                        "physical_execution_authorized"
                    ) is True
                ):
                    result = InferenceResult(
                        request_id=(
                            item.request.request_id
                        ),
                        model_name=model.name,
                        status=InferenceStatus.FAILED,
                        error=(
                            "Unsafe model output rejected"
                        ),
                        batch_id=batch_id,
                        batch_size=len(active_items),
                        inference_latency_ms=elapsed_ms,
                        device=model.device,
                    )

                    self.statistics.failed_requests += 1

                else:

                    queue_latency_ms = max(
                        0.0,
                        (
                            started
                            - item.enqueued_at
                        ) * 1000,
                    )

                    result = InferenceResult(
                        request_id=(
                            item.request.request_id
                        ),
                        model_name=model.name,
                        status=(
                            InferenceStatus.COMPLETED
                        ),
                        output=output,
                        batch_id=batch_id,
                        batch_size=len(active_items),
                        queue_latency_ms=(
                            queue_latency_ms
                        ),
                        inference_latency_ms=(
                            elapsed_ms
                        ),
                        total_latency_ms=(
                            queue_latency_ms
                            + elapsed_ms
                        ),
                        device=model.device,
                        physical_execution_authorized=False,
                    )

                    self.statistics.completed_requests += 1

                    self.statistics.total_queue_latency_ms += (
                        queue_latency_ms
                    )

                item.future.set_result(result)

    # --------------------------------------------------------
    # Batch Submit
    # --------------------------------------------------------

    async def submit_many(
        self,
        requests: Sequence[InferenceRequest],
    ) -> list[InferenceResult]:

        return list(
            await asyncio.gather(
                *(
                    self.submit(request)
                    for request in requests
                )
            )
        )

    # --------------------------------------------------------
    # Queue Statistics
    # --------------------------------------------------------

    async def get_queue_statistics(
        self,
    ) -> dict[str, Any]:

        async with self._condition:

            queues = {
                f"{tenant_id}:{model_name}": len(queue)
                for (
                    tenant_id,
                    model_name,
                ), queue in self._queues.items()
                if queue
            }

            return {
                "running": self._running,
                "closing": self._closing,
                "queued_requests": (
                    self._queued_count
                ),
                "active_batches": len(
                    self._active_batches
                ),
                "queues": queues,
                "statistics": {
                    "total_requests": (
                        self.statistics.total_requests
                    ),
                    "completed_requests": (
                        self.statistics.completed_requests
                    ),
                    "failed_requests": (
                        self.statistics.failed_requests
                    ),
                    "timed_out_requests": (
                        self.statistics.timed_out_requests
                    ),
                    "cancelled_requests": (
                        self.statistics.cancelled_requests
                    ),
                    "total_batches": (
                        self.statistics.total_batches
                    ),
                    "successful_batches": (
                        self.statistics.successful_batches
                    ),
                    "failed_batches": (
                        self.statistics.failed_batches
                    ),
                    "average_batch_size": (
                        self.statistics.average_batch_size
                    ),
                    "average_queue_latency_ms": (
                        self.statistics.average_queue_latency_ms
                    ),
                    "average_inference_latency_ms": (
                        self.statistics.average_inference_latency_ms
                    ),
                },
            }

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    async def shutdown(
        self,
        *,
        wait_for_active: bool = True,
    ) -> None:

        async with self._condition:

            self._closing = True
            self._running = False

            for queue in self._queues.values():

                while queue:

                    item = queue.popleft()

                    if not item.future.done():
                        item.future.set_result(
                            InferenceResult(
                                request_id=(
                                    item.request.request_id
                                ),
                                model_name=(
                                    item.request.model_name
                                ),
                                status=(
                                    InferenceStatus.CANCELLED
                                ),
                                error=(
                                    "Batch scheduler shut down"
                                ),
                            )
                        )

                        self.statistics.cancelled_requests += 1

            self._queued_count = 0

            self._condition.notify_all()

        for worker in tuple(self._workers):
            worker.cancel()

        if self._workers:
            await asyncio.gather(
                *tuple(self._workers),
                return_exceptions=True,
            )

        if wait_for_active:

            if self._active_batches:
                await asyncio.gather(
                    *tuple(self._active_batches),
                    return_exceptions=True,
                )

        else:

            for task in tuple(
                self._active_batches
            ):
                task.cancel()

            if self._active_batches:
                await asyncio.gather(
                    *tuple(self._active_batches),
                    return_exceptions=True,
                )


# ============================================================
# Example Backend
# ============================================================

class CallableBatchBackend:
    """
    Adapter for an asynchronous or synchronous
    batch inference function.

    For GPU workloads, prefer an asynchronous model
    server or dedicated inference worker.
    """

    def __init__(
        self,
        predict_fn: Callable[
            [Sequence[Any]],
            Any,
        ],
    ) -> None:

        self.predict_fn = predict_fn

    async def predict_batch(
        self,
        inputs: Sequence[Any],
    ) -> Sequence[Any]:

        if inspect.iscoroutinefunction(
            self.predict_fn
        ):
            outputs = await self.predict_fn(
                inputs
            )

        else:
            outputs = await asyncio.to_thread(
                self.predict_fn,
                inputs,
            )

            if inspect.isawaitable(outputs):
                outputs = await outputs

        if (
            isinstance(
                outputs,
                (str, bytes, Mapping),
            )
            or not isinstance(
                outputs,
                Sequence,
            )
        ):
            raise BatchOutputError(
                "Backend returned an invalid output"
            )

        return outputs


# ============================================================
# Factory
# ============================================================

def create_batch_scheduler(
    *,
    registry: InferenceModelRegistry,
    authorizer: InferenceAuthorizer,
    config: BatchConfiguration | None = None,
    audit_sink: InferenceAuditSink | None = None,
) -> DynamicBatchScheduler:

    return DynamicBatchScheduler(
        registry=registry,
        authorizer=authorizer,
        config=config,
        audit_sink=audit_sink,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "BatchingError",
    "BatchValidationError",
    "BatchQueueFullError",
    "BatchTimeoutError",
    "BatchInferenceError",
    "BatchOutputError",
    "BatchShutdownError",
    "ModelNotRegisteredError",
    "UnsafeInferenceError",
    "InferencePriority",
    "InferenceStatus",
    "ModelCategory",
    "InferenceDevice",
    "BatchConfiguration",
    "InferenceScope",
    "InferenceRequest",
    "InferenceResult",
    "BatchStatistics",
    "BatchInferenceBackend",
    "InferenceAuthorizer",
    "InferenceAuditSink",
    "RegisteredInferenceModel",
    "InferenceModelRegistry",
    "CallableBatchBackend",
    "DynamicBatchScheduler",
    "create_batch_scheduler",
]
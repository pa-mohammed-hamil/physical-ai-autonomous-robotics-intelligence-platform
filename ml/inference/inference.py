"""
ml/inference/inference.py

Physical AI & Autonomous Robotics Intelligence Platform

Unified machine learning inference service.

Features:
- Model registration and versioning
- PyTorch, callable and remote inference backends
- CPU, CUDA and Apple MPS support
- Single and batch inference
- Async execution with concurrency limits
- Model-specific preprocessing and postprocessing
- Inference timeout handling
- Model input/output validation
- Tenant-aware authorization
- Model performance monitoring
- Structured inference results
- Error handling and audit logging
- Simulation-only robotic action inference
- No direct physical robot actuation

Python 3.11+

Optional dependencies:
    pip install torch numpy
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import math
import time
import uuid

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


try:
    import torch
except ImportError:
    torch = None


# ============================================================
# Constants
# ============================================================

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_BATCH_SIZE = 32
DEFAULT_MAX_CONCURRENT_INFERENCES = 8

MAX_TIMEOUT_SECONDS = 600.0
MAX_BATCH_SIZE = 512

ROBOT_ACTION_CATEGORIES = {
    "vla",
    "navigation",
    "manipulation",
}


# ============================================================
# Exceptions
# ============================================================

class InferenceError(Exception):
    """Base inference exception."""


class InferenceValidationError(InferenceError):
    """Invalid inference input or configuration."""


class InferencePermissionError(InferenceError):
    """Inference permission denied."""


class InferenceSafetyError(InferenceError):
    """Unsafe inference request or output."""


class InferenceModelNotFoundError(InferenceError):
    """Requested model is not registered."""


class InferenceModelUnavailableError(InferenceError):
    """Requested model is unavailable."""


class InferenceTimeoutError(InferenceError):
    """Inference exceeded its timeout."""


class InferenceExecutionError(InferenceError):
    """Model inference execution failed."""


class InferenceOutputError(InferenceError):
    """Model returned an invalid output."""


class InferenceServiceClosedError(InferenceError):
    """Inference service has been shut down."""


# ============================================================
# Utilities
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def validate_positive(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise InferenceValidationError(
            f"{name} must be a finite positive number"
        )

    return float(value)


def contains_unsafe_authorization(
    value: Any,
    *,
    depth: int = 0,
) -> bool:
    """
    Reject model outputs that claim physical authority.

    This is a defense-in-depth output check, not a
    replacement for independent robot safety controls.
    """
    if depth > 12:
        raise InferenceOutputError(
            "Model output exceeds maximum nesting depth"
        )

    forbidden = {
        "physical_execution_authorized",
        "emergency_stop_reset_authorized",
        "safety_interlock_bypass_authorized",
        "hardware_actuation_authorized",
    }

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in forbidden and item is True:
                return True

            if contains_unsafe_authorization(
                item,
                depth=depth + 1,
            ):
                return True

    elif isinstance(value, (list, tuple)):
        for item in value:
            if contains_unsafe_authorization(
                item,
                depth=depth + 1,
            ):
                return True

    return False


# ============================================================
# Enumerations
# ============================================================

class ModelCategory(str, Enum):
    VLM = "vlm"
    VLA = "vla"
    PERCEPTION = "perception"
    WORLD_MODEL = "world_model"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    ANOMALY = "anomaly"
    FAILURE_PREDICTION = "failure_prediction"
    RUL = "rul"
    PART_INSPECTION = "part_inspection"
    CUSTOM = "custom"


class InferenceDevice(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    REMOTE = "remote"


class ModelStatus(str, Enum):
    REGISTERED = "registered"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class InferenceStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True, slots=True)
class InferenceConfiguration:
    default_timeout_seconds: float = (
        DEFAULT_TIMEOUT_SECONDS
    )

    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE

    max_concurrent_inferences: int = (
        DEFAULT_MAX_CONCURRENT_INFERENCES
    )

    allow_cpu_fallback: bool = True

    enable_inference_mode: bool = True

    def __post_init__(self) -> None:
        validate_positive(
            self.default_timeout_seconds,
            "default_timeout_seconds",
        )

        if (
            self.default_timeout_seconds
            > MAX_TIMEOUT_SECONDS
        ):
            raise InferenceValidationError(
                "Timeout exceeds maximum"
            )

        if not 1 <= self.max_batch_size <= MAX_BATCH_SIZE:
            raise InferenceValidationError(
                "Invalid maximum batch size"
            )

        if self.max_concurrent_inferences < 1:
            raise InferenceValidationError(
                "Concurrency must be positive"
            )


# ============================================================
# Request and Result Models
# ============================================================

@dataclass(frozen=True, slots=True)
class InferenceScope:
    tenant_id: str
    user_id: str

    robot_id: str | None = None
    workflow_id: str | None = None
    task_id: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise InferenceValidationError(
                "Tenant ID is required"
            )

        if not self.user_id.strip():
            raise InferenceValidationError(
                "User ID is required"
            )


@dataclass(slots=True)
class InferenceRequest:
    model_name: str
    inputs: Any
    scope: InferenceScope

    request_id: str = field(
        default_factory=generate_id
    )

    model_version: str | None = None

    timeout_seconds: float | None = None

    simulation_only: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise InferenceValidationError(
                "Model name is required"
            )

        if self.timeout_seconds is not None:
            validate_positive(
                self.timeout_seconds,
                "timeout_seconds",
            )


@dataclass(slots=True)
class InferenceResult:
    request_id: str
    model_name: str
    model_version: str

    status: InferenceStatus

    output: Any = None
    error: str | None = None

    device: InferenceDevice = InferenceDevice.CPU

    inference_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    batch_size: int = 1

    completed_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    physical_execution_authorized: bool = False


@dataclass(slots=True)
class ModelMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timed_out_requests: int = 0

    total_inference_latency_ms: float = 0.0

    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0

        return (
            self.successful_requests
            / self.total_requests
        )

    @property
    def average_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0

        return (
            self.total_inference_latency_ms
            / self.successful_requests
        )


# ============================================================
# Protocols
# ============================================================

class InferenceBackend(Protocol):
    async def predict(self, inputs: Any) -> Any:
        ...

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


class InferenceResultRepository(Protocol):
    async def save_result(
        self,
        result: InferenceResult,
        scope: InferenceScope,
    ) -> None:
        ...


class InferenceInputValidator(Protocol):
    def validate(
        self,
        inputs: Any,
    ) -> None:
        ...


class InferenceOutputValidator(Protocol):
    def validate(
        self,
        output: Any,
    ) -> None:
        ...


# ============================================================
# Device Resolver
# ============================================================

class DeviceResolver:
    @staticmethod
    def resolve(
        preferred: InferenceDevice,
        *,
        allow_cpu_fallback: bool = True,
    ) -> InferenceDevice:
        if preferred == InferenceDevice.REMOTE:
            return InferenceDevice.REMOTE

        if preferred == InferenceDevice.CPU:
            return InferenceDevice.CPU

        if torch is not None:
            if preferred in (
                InferenceDevice.AUTO,
                InferenceDevice.CUDA,
            ):
                if torch.cuda.is_available():
                    return InferenceDevice.CUDA

            if preferred in (
                InferenceDevice.AUTO,
                InferenceDevice.MPS,
            ):
                if (
                    hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                ):
                    return InferenceDevice.MPS

        if allow_cpu_fallback:
            return InferenceDevice.CPU

        raise InferenceModelUnavailableError(
            f"Requested device unavailable: {preferred.value}"
        )

    @staticmethod
    def torch_device(
        device: InferenceDevice,
    ) -> str:
        if device == InferenceDevice.CUDA:
            return "cuda:0"

        if device == InferenceDevice.MPS:
            return "mps"

        return "cpu"


# ============================================================
# Callable Inference Backend
# ============================================================

class CallableInferenceBackend:
    """
    Wrap synchronous or asynchronous prediction functions.
    """

    def __init__(
        self,
        predict_fn: Callable[[Any], Any],
        batch_fn: Callable[
            [Sequence[Any]],
            Any,
        ] | None = None,
    ) -> None:
        self.predict_fn = predict_fn
        self.batch_fn = batch_fn

    @staticmethod
    async def _invoke(
        fn: Callable[..., Any],
        *args: Any,
    ) -> Any:
        if inspect.iscoroutinefunction(fn):
            return await fn(*args)

        result = await asyncio.to_thread(
            fn,
            *args,
        )

        if inspect.isawaitable(result):
            return await result

        return result

    async def predict(self, inputs: Any) -> Any:
        return await self._invoke(
            self.predict_fn,
            inputs,
        )

    async def predict_batch(
        self,
        inputs: Sequence[Any],
    ) -> Sequence[Any]:
        if self.batch_fn is not None:
            result = await self._invoke(
                self.batch_fn,
                inputs,
            )
        else:
            result = await asyncio.gather(
                *(
                    self.predict(item)
                    for item in inputs
                )
            )

        if (
            isinstance(result, (str, bytes, Mapping))
            or not isinstance(result, Sequence)
        ):
            raise InferenceOutputError(
                "Batch backend must return a sequence"
            )

        return result


# ============================================================
# PyTorch Backend
# ============================================================

class PyTorchInferenceBackend:
    """
    Adapter for an already-loaded PyTorch model.

    The caller supplies preprocessing and postprocessing
    suitable for the specific model architecture.
    """

    def __init__(
        self,
        *,
        model: Any,
        device: InferenceDevice = InferenceDevice.AUTO,
        preprocess: Callable[[Any], Any] | None = None,
        postprocess: Callable[[Any], Any] | None = None,
        allow_cpu_fallback: bool = True,
    ) -> None:
        if torch is None:
            raise InferenceModelUnavailableError(
                "PyTorch is not installed"
            )

        self.device = DeviceResolver.resolve(
            device,
            allow_cpu_fallback=allow_cpu_fallback,
        )

        if self.device == InferenceDevice.REMOTE:
            raise InferenceValidationError(
                "PyTorch backend requires a local device"
            )

        self.torch_device = DeviceResolver.torch_device(
            self.device
        )

        self.model = model.to(self.torch_device)
        self.model.eval()

        self.preprocess = preprocess
        self.postprocess = postprocess

        self._model_lock = asyncio.Lock()

    def _prepare_input(self, inputs: Any) -> Any:
        if self.preprocess is not None:
            inputs = self.preprocess(inputs)

        return self._move_to_device(inputs)

    def _move_to_device(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(self.torch_device)

        if isinstance(value, dict):
            return {
                key: self._move_to_device(item)
                for key, item in value.items()
            }

        if isinstance(value, tuple):
            return tuple(
                self._move_to_device(item)
                for item in value
            )

        if isinstance(value, list):
            return [
                self._move_to_device(item)
                for item in value
            ]

        return value

    def _forward(self, inputs: Any) -> Any:
        with torch.inference_mode():
            if isinstance(inputs, Mapping):
                return self.model(**inputs)

            if isinstance(inputs, tuple):
                return self.model(*inputs)

            return self.model(inputs)

    def _convert_output(self, output: Any) -> Any:
        if self.postprocess is not None:
            return self.postprocess(output)

        if torch.is_tensor(output):
            return output.detach().cpu().tolist()

        if isinstance(output, Mapping):
            return {
                key: self._convert_output(value)
                for key, value in output.items()
            }

        if isinstance(output, (tuple, list)):
            return [
                self._convert_output(value)
                for value in output
            ]

        return output

    def _predict_sync(self, inputs: Any) -> Any:
        prepared = self._prepare_input(inputs)

        output = self._forward(prepared)

        return self._convert_output(output)

    async def predict(self, inputs: Any) -> Any:
        async with self._model_lock:
            return await asyncio.to_thread(
                self._predict_sync,
                inputs,
            )

    async def predict_batch(
        self,
        inputs: Sequence[Any],
    ) -> Sequence[Any]:
        # Conservative default: sequential execution.
        # A model-specific batch_fn can provide tensor stacking
        # and more efficient batched forward passes.
        results = []

        for item in inputs:
            results.append(
                await self.predict(item)
            )

        return results


# ============================================================
# Model Registration
# ============================================================

@dataclass(slots=True)
class RegisteredModel:
    name: str
    version: str
    category: ModelCategory
    backend: InferenceBackend

    device: InferenceDevice = InferenceDevice.AUTO

    status: ModelStatus = ModelStatus.READY

    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE

    input_validator: InferenceInputValidator | None = None
    output_validator: InferenceOutputValidator | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    metrics: ModelMetrics = field(
        default_factory=ModelMetrics
    )

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InferenceValidationError(
                "Model name is required"
            )

        if not self.version.strip():
            raise InferenceValidationError(
                "Model version is required"
            )

        if not 1 <= self.max_batch_size <= MAX_BATCH_SIZE:
            raise InferenceValidationError(
                "Invalid model batch size"
            )


class InferenceModelRegistry:
    def __init__(self) -> None:
        self._models: dict[
            tuple[str, str],
            RegisteredModel,
        ] = {}

        self._default_versions: dict[
            str,
            str,
        ] = {}

    def register(
        self,
        model: RegisteredModel,
        *,
        set_default: bool = True,
    ) -> None:
        key = (model.name, model.version)

        if key in self._models:
            raise InferenceValidationError(
                "Model version already registered"
            )

        self._models[key] = model

        if (
            set_default
            or model.name not in self._default_versions
        ):
            self._default_versions[
                model.name
            ] = model.version

    def get(
        self,
        model_name: str,
        version: str | None = None,
    ) -> RegisteredModel:
        selected_version = (
            version
            or self._default_versions.get(model_name)
        )

        if selected_version is None:
            raise InferenceModelNotFoundError(
                f"Model not found: {model_name}"
            )

        model = self._models.get(
            (model_name, selected_version)
        )

        if model is None:
            raise InferenceModelNotFoundError(
                f"Model not found: {model_name}:{selected_version}"
            )

        if model.status not in (
            ModelStatus.READY,
            ModelStatus.DEGRADED,
        ):
            raise InferenceModelUnavailableError(
                f"Model unavailable: {model_name}"
            )

        return model

    def list_models(self) -> list[RegisteredModel]:
        return list(self._models.values())

    def set_default_version(
        self,
        model_name: str,
        version: str,
    ) -> None:
        if (model_name, version) not in self._models:
            raise InferenceModelNotFoundError(
                "Model version not registered"
            )

        self._default_versions[model_name] = version


# ============================================================
# Unified Inference Service
# ============================================================

class InferenceService:
    """
    Unified asynchronous inference service.

    This service produces predictions and recommendations.
    It cannot authorize or execute physical robot actions.
    """

    def __init__(
        self,
        *,
        authorizer: InferenceAuthorizer,
        registry: InferenceModelRegistry | None = None,
        config: InferenceConfiguration | None = None,
        audit_sink: InferenceAuditSink | None = None,
        repository: InferenceResultRepository | None = None,
    ) -> None:
        self.authorizer = authorizer

        self.registry = (
            registry or InferenceModelRegistry()
        )

        self.config = (
            config or InferenceConfiguration()
        )

        self.audit_sink = audit_sink
        self.repository = repository

        self._semaphore = asyncio.Semaphore(
            self.config.max_concurrent_inferences
        )

        self._closed = False

    def register_model(
        self,
        model: RegisteredModel,
        *,
        set_default: bool = True,
    ) -> None:
        if self._closed:
            raise InferenceServiceClosedError(
                "Inference service is closed"
            )

        self.registry.register(
            model,
            set_default=set_default,
        )

    @staticmethod
    def _validate_safety(
        request: InferenceRequest,
        model: RegisteredModel,
    ) -> None:
        if (
            model.category.value in ROBOT_ACTION_CATEGORIES
            and not request.simulation_only
        ):
            raise InferenceSafetyError(
                "Robot action-model inference "
                "must be simulation-only"
            )

        if request.metadata.get(
            "physical_execution_authorized"
        ) is True:
            raise InferenceSafetyError(
                "Inference cannot authorize "
                "physical robot execution"
            )

    @staticmethod
    def _validate_output(
        output: Any,
        model: RegisteredModel,
    ) -> None:
        if output is None:
            raise InferenceOutputError(
                "Model returned no output"
            )

        if contains_unsafe_authorization(output):
            raise InferenceSafetyError(
                "Model output contains an unsafe "
                "physical authorization claim"
            )

        if model.output_validator is not None:
            model.output_validator.validate(output)

    async def _authorize(
        self,
        scope: InferenceScope,
    ) -> None:
        try:
            await self.authorizer.require_permission(
                tenant_id=scope.tenant_id,
                user_id=scope.user_id,
                permission="ml:inference",
            )
        except PermissionError as exc:
            raise InferencePermissionError(
                "Inference permission denied"
            ) from exc

    async def predict(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        if self._closed:
            raise InferenceServiceClosedError(
                "Inference service is closed"
            )

        await self._authorize(request.scope)

        model = self.registry.get(
            request.model_name,
            request.model_version,
        )

        self._validate_safety(request, model)

        if model.input_validator is not None:
            model.input_validator.validate(
                request.inputs
            )

        timeout = (
            request.timeout_seconds
            or self.config.default_timeout_seconds
        )

        validate_positive(
            timeout,
            "timeout_seconds",
        )

        started = time.perf_counter()

        model.metrics.total_requests += 1

        output: Any = None
        error: str | None = None

        try:
            async with self._semaphore:
                output = await asyncio.wait_for(
                    model.backend.predict(
                        request.inputs
                    ),
                    timeout=timeout,
                )

                self._validate_output(
                    output,
                    model,
                )

            status = InferenceStatus.COMPLETED

            model.metrics.successful_requests += 1
            model.metrics.last_success_at = utc_now()

        except asyncio.TimeoutError:
            status = InferenceStatus.TIMED_OUT
            error = "Inference timeout exceeded"

            model.metrics.timed_out_requests += 1
            model.metrics.failed_requests += 1
            model.metrics.last_failure_at = utc_now()

        except InferenceSafetyError as exc:
            status = InferenceStatus.REJECTED
            error = str(exc)

            model.metrics.failed_requests += 1
            model.metrics.last_failure_at = utc_now()

        except Exception as exc:
            status = InferenceStatus.FAILED
            error = (
                f"Inference failed: "
                f"{type(exc).__name__}"
            )

            logger.exception(
                "Inference failed for model %s",
                model.name,
            )

            model.metrics.failed_requests += 1
            model.metrics.last_failure_at = utc_now()

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000

        if status == InferenceStatus.COMPLETED:
            model.metrics.total_inference_latency_ms += (
                elapsed_ms
            )

        device = DeviceResolver.resolve(
            model.device,
            allow_cpu_fallback=(
                self.config.allow_cpu_fallback
            ),
        )

        result = InferenceResult(
            request_id=request.request_id,
            model_name=model.name,
            model_version=model.version,
            status=status,
            output=(
                output
                if status == InferenceStatus.COMPLETED
                else None
            ),
            error=error,
            device=device,
            inference_latency_ms=elapsed_ms,
            total_latency_ms=elapsed_ms,
            batch_size=1,
            metadata={
                "category": model.category.value,
                "simulation_only": (
                    request.simulation_only
                ),
            },
            physical_execution_authorized=False,
        )

        await self._record_result(
            request=request,
            result=result,
        )

        return result

    async def predict_batch(
        self,
        requests: Sequence[InferenceRequest],
    ) -> list[InferenceResult]:
        if not requests:
            return []

        if len(requests) > self.config.max_batch_size:
            raise InferenceValidationError(
                "Batch size exceeds configured maximum"
            )

        # Concurrent single-request execution is the safe
        # generic fallback. A specialized batching service
        # can group requests for true model batch inference.
        return list(
            await asyncio.gather(
                *(
                    self.predict(request)
                    for request in requests
                )
            )
        )

    async def _record_result(
        self,
        *,
        request: InferenceRequest,
        result: InferenceResult,
    ) -> None:
        if self.repository is not None:
            await self.repository.save_result(
                result,
                request.scope,
            )

        if self.audit_sink is not None:
            await self.audit_sink.record(
                tenant_id=request.scope.tenant_id,
                user_id=request.scope.user_id,
                action="ml.inference.executed",
                metadata={
                    "request_id": request.request_id,
                    "model_name": result.model_name,
                    "model_version": result.model_version,
                    "status": result.status.value,
                    "device": result.device.value,
                    "latency_ms": (
                        result.total_latency_ms
                    ),
                    "simulation_only": (
                        request.simulation_only
                    ),
                    "physical_execution_authorized": False,
                },
            )

    def get_model_metrics(
        self,
        model_name: str,
        version: str | None = None,
    ) -> dict[str, Any]:
        model = self.registry.get(
            model_name,
            version,
        )

        metrics = model.metrics

        return {
            "model_name": model.name,
            "version": model.version,
            "category": model.category.value,
            "status": model.status.value,
            "total_requests": metrics.total_requests,
            "successful_requests": (
                metrics.successful_requests
            ),
            "failed_requests": (
                metrics.failed_requests
            ),
            "timed_out_requests": (
                metrics.timed_out_requests
            ),
            "success_rate": metrics.success_rate,
            "average_latency_ms": (
                metrics.average_latency_ms
            ),
            "last_success_at": (
                metrics.last_success_at.isoformat()
                if metrics.last_success_at
                else None
            ),
            "last_failure_at": (
                metrics.last_failure_at.isoformat()
                if metrics.last_failure_at
                else None
            ),
        }

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "name": model.name,
                "version": model.version,
                "category": model.category.value,
                "device": model.device.value,
                "status": model.status.value,
                "max_batch_size": model.max_batch_size,
            }
            for model in self.registry.list_models()
        ]

    def generate_report(self) -> dict[str, Any]:
        return {
            "generated_at": utc_now().isoformat(),
            "service": "unified_ml_inference",
            "registered_models": self.list_models(),
            "model_metrics": [
                {
                    "name": model.name,
                    "version": model.version,
                    "total_requests": (
                        model.metrics.total_requests
                    ),
                    "successful_requests": (
                        model.metrics.successful_requests
                    ),
                    "failed_requests": (
                        model.metrics.failed_requests
                    ),
                    "success_rate": (
                        model.metrics.success_rate
                    ),
                    "average_latency_ms": (
                        model.metrics.average_latency_ms
                    ),
                }
                for model in self.registry.list_models()
            ],
            "safety": {
                "physical_robot_actuation": False,
                "emergency_stop_reset": False,
                "safety_interlock_bypass": False,
                "action_models_simulation_only": True,
            },
        }

    async def shutdown(self) -> None:
        self._closed = True


# ============================================================
# Deny-by-Default Authorizer
# ============================================================

class DenyByDefaultInferenceAuthorizer:
    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        raise InferencePermissionError(
            "Inference authorization provider "
            "has not been configured"
        )


# ============================================================
# Factory
# ============================================================

def create_inference_service(
    *,
    authorizer: InferenceAuthorizer,
    registry: InferenceModelRegistry | None = None,
    config: InferenceConfiguration | None = None,
    audit_sink: InferenceAuditSink | None = None,
    repository: InferenceResultRepository | None = None,
) -> InferenceService:
    return InferenceService(
        authorizer=authorizer,
        registry=registry,
        config=config,
        audit_sink=audit_sink,
        repository=repository,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "InferenceError",
    "InferenceValidationError",
    "InferencePermissionError",
    "InferenceSafetyError",
    "InferenceModelNotFoundError",
    "InferenceModelUnavailableError",
    "InferenceTimeoutError",
    "InferenceExecutionError",
    "InferenceOutputError",
    "InferenceServiceClosedError",
    "ModelCategory",
    "InferenceDevice",
    "ModelStatus",
    "InferenceStatus",
    "InferenceConfiguration",
    "InferenceScope",
    "InferenceRequest",
    "InferenceResult",
    "ModelMetrics",
    "InferenceBackend",
    "InferenceAuthorizer",
    "InferenceAuditSink",
    "InferenceResultRepository",
    "InferenceInputValidator",
    "InferenceOutputValidator",
    "DeviceResolver",
    "CallableInferenceBackend",
    "PyTorchInferenceBackend",
    "RegisteredModel",
    "InferenceModelRegistry",
    "InferenceService",
    "DenyByDefaultInferenceAuthorizer",
    "create_inference_service",
]
"""Model Serving module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/mlops/model_serving.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
MLOps — Model Serving

Features:
    - Multi-model inference serving
    - Model version routing
    - Production, staging and canary deployments
    - Weighted traffic distribution
    - Dynamic batching
    - Asynchronous inference
    - Concurrent request processing
    - Model loading and unloading
    - Model warm-up
    - Model health checks
    - Model artifact verification
    - Input and output validation
    - Inference latency monitoring
    - Throughput and error-rate metrics
    - Request timeout handling
    - Rate limiting
    - Tenant isolation
    - Model registry adapter interface
    - ONNX Runtime adapter
    - PyTorch adapter
    - Synthetic inference adapter
    - Structured audit logging
    - Thread-safe serving
    - Optional FastAPI integration
    - Unit tests
    - Demonstration

Python 3.11+

Optional dependencies:
    pip install numpy onnxruntime
    pip install torch
    pip install fastapi uvicorn pydantic

Safety:
    This service performs model inference only.
    It does not directly command physical robots.
    Model outputs are predictions, not actuator commands.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import statistics
import tempfile
import threading
import time
import unittest
import uuid

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class ModelServingError(Exception):
    pass


class ModelValidationError(ModelServingError):
    pass


class ModelNotFoundError(ModelServingError):
    pass


class ModelLoadError(ModelServingError):
    pass


class ModelInferenceError(ModelServingError):
    pass


class ModelTimeoutError(ModelServingError):
    pass


class ModelAuthorizationError(ModelServingError):
    pass


class ModelRateLimitError(ModelServingError):
    pass


class ModelIntegrityError(ModelServingError):
    pass


class ModelRoutingError(ModelServingError):
    pass


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
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
    ):
        raise ModelValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def validate_finite_number(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise ModelValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise ModelValidationError(
            f"{name} must be finite"
        )

    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "tolist"):
        return json_safe(value.tolist())

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelValidationError(
                "Non-finite output value"
            )

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    raise ModelValidationError(
        f"Unsupported JSON value: {type(value).__name__}"
    )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# ENUMERATIONS
# ============================================================

class ModelFramework(str, Enum):
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SCIKIT_LEARN = "scikit_learn"
    TRANSFORMERS = "transformers"
    CUSTOM = "custom"
    SYNTHETIC = "synthetic"


class ModelTask(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    ROBOT_PERCEPTION = "robot_perception"
    ROBOT_HEALTH = "robot_health"
    NAVIGATION = "navigation"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    EMBEDDING = "embedding"
    GENERATIVE_AI = "generative_ai"
    MULTIMODAL = "multimodal"
    OTHER = "other"


class ModelStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"


class ModelStatus(str, Enum):
    REGISTERED = "registered"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNLOADED = "unloaded"


class InferenceStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class RoutingStrategy(str, Enum):
    PRODUCTION = "production"
    WEIGHTED = "weighted"
    EXPLICIT_VERSION = "explicit_version"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ModelServingConfig:
    maximum_workers: int = 8

    maximum_batch_size: int = 32

    default_timeout_seconds: float = 10.0

    maximum_input_bytes: int = 2 * 1024 * 1024

    requests_per_minute: int = 600

    metrics_history_size: int = 10000

    verify_artifacts: bool = True

    allow_explicit_version_routing: bool = True

    def __post_init__(self) -> None:
        if self.maximum_workers <= 0:
            raise ModelValidationError(
                "maximum_workers must be positive"
            )

        if self.maximum_batch_size <= 0:
            raise ModelValidationError(
                "maximum_batch_size must be positive"
            )

        if self.default_timeout_seconds <= 0:
            raise ModelValidationError(
                "default_timeout_seconds must be positive"
            )

        if self.maximum_input_bytes <= 0:
            raise ModelValidationError(
                "maximum_input_bytes must be positive"
            )

        if self.requests_per_minute <= 0:
            raise ModelValidationError(
                "requests_per_minute must be positive"
            )

        if self.metrics_history_size <= 0:
            raise ModelValidationError(
                "metrics_history_size must be positive"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class ServingAccess:
    tenant_id: str
    user_id: str

    can_infer: bool = False
    can_manage: bool = False
    can_monitor: bool = False

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
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class ModelSpecification:
    model_id: str
    version_id: str
    tenant_id: str

    model_name: str

    framework: ModelFramework
    task: ModelTask
    stage: ModelStage

    artifact_path: Path | None = None
    artifact_sha256: str | None = None

    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "version_id",
            "tenant_id",
            "model_name",
        ):
            validate_identifier(
                getattr(self, name),
                name,
            )

        if self.artifact_sha256 is not None:
            digest = self.artifact_sha256

            if (
                len(digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in digest
                )
            ):
                raise ModelValidationError(
                    "Invalid artifact SHA-256"
                )


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    tenant_id: str
    model_id: str

    inputs: dict[str, Any]

    version_id: str | None = None

    timeout_seconds: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    tenant_id: str
    model_id: str
    version_id: str

    outputs: dict[str, Any]

    status: InferenceStatus

    latency_ms: float

    created_at: datetime

    completed_at: datetime

    error: str | None = None


@dataclass(frozen=True)
class ModelHealth:
    model_id: str
    version_id: str

    status: ModelStatus

    loaded: bool

    total_requests: int
    successful_requests: int
    failed_requests: int

    error_rate: float

    average_latency_ms: float
    p95_latency_ms: float

    checked_at: datetime


@dataclass(frozen=True)
class ModelTrafficWeight:
    version_id: str
    weight: float


@dataclass(frozen=True)
class InferenceMetric:
    request_id: str
    tenant_id: str
    model_id: str
    version_id: str

    status: InferenceStatus
    latency_ms: float

    timestamp: datetime


# ============================================================
# INPUT VALIDATION
# ============================================================

class InferenceInputValidator:

    @staticmethod
    def validate(
        inputs: Mapping[str, Any],
        specification: ModelSpecification,
        maximum_bytes: int,
    ) -> dict[str, Any]:
        if not isinstance(inputs, Mapping):
            raise ModelValidationError(
                "Inference inputs must be a mapping"
            )

        normalized = json_safe(
            dict(inputs)
        )

        if not isinstance(normalized, dict):
            raise ModelValidationError(
                "Invalid inference input structure"
            )

        encoded = json.dumps(
            normalized,
            allow_nan=False,
        ).encode("utf-8")

        if len(encoded) > maximum_bytes:
            raise ModelValidationError(
                "Inference input exceeds maximum size"
            )

        for input_name in specification.input_names:
            if input_name not in normalized:
                raise ModelValidationError(
                    f"Missing required input: {input_name}"
                )

        return normalized


class InferenceOutputValidator:

    @staticmethod
    def validate(
        outputs: Mapping[str, Any],
        specification: ModelSpecification,
    ) -> dict[str, Any]:
        if not isinstance(outputs, Mapping):
            raise ModelValidationError(
                "Model outputs must be a mapping"
            )

        normalized = json_safe(
            dict(outputs)
        )

        for output_name in specification.output_names:
            if output_name not in normalized:
                raise ModelValidationError(
                    f"Missing expected output: {output_name}"
                )

        return normalized


# ============================================================
# MODEL ADAPTER CONTRACT
# ============================================================

class ModelAdapter(Protocol):

    def load(
        self,
        specification: ModelSpecification,
    ) -> None:
        ...

    def predict(
        self,
        inputs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...

    def predict_batch(
        self,
        batch: Sequence[Mapping[str, Any]],
    ) -> Sequence[Mapping[str, Any]]:
        ...

    def unload(self) -> None:
        ...


# ============================================================
# SYNTHETIC MODEL ADAPTER
# ============================================================

class SyntheticModelAdapter:
    """
    Deterministic synthetic inference adapter.

    This adapter is intended for development,
    benchmarking and unit testing.
    """

    def __init__(self) -> None:
        self._loaded = False

        self._specification: (
            ModelSpecification | None
        ) = None

    def load(
        self,
        specification: ModelSpecification,
    ) -> None:
        self._specification = specification
        self._loaded = True

    def predict(
        self,
        inputs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not self._loaded:
            raise ModelLoadError(
                "Synthetic model is not loaded"
            )

        features = inputs.get(
            "features",
            [],
        )

        if not isinstance(features, list):
            raise ModelValidationError(
                "features must be a list"
            )

        numeric = [
            validate_finite_number(
                value,
                "feature",
            )
            for value in features
        ]

        score = (
            sum(numeric) / len(numeric)
            if numeric
            else 0.0
        )

        probability = (
            1.0 / (1.0 + math.exp(-max(
                -60.0,
                min(60.0, score),
            )))
        )

        return {
            "prediction": int(probability >= 0.5),
            "confidence": probability,
            "score": score,
            "synthetic": True,
        }

    def predict_batch(
        self,
        batch: Sequence[Mapping[str, Any]],
    ) -> Sequence[Mapping[str, Any]]:
        return [
            self.predict(item)
            for item in batch
        ]

    def unload(self) -> None:
        self._loaded = False
        self._specification = None


# ============================================================
# ONNX RUNTIME ADAPTER
# ============================================================

class ONNXRuntimeAdapter:

    def __init__(
        self,
        providers: Sequence[str] | None = None,
    ) -> None:
        self.providers = list(
            providers or ["CPUExecutionProvider"]
        )

        self._session: Any = None
        self._np: Any = None

    def load(
        self,
        specification: ModelSpecification,
    ) -> None:
        if specification.artifact_path is None:
            raise ModelLoadError(
                "ONNX artifact path is required"
            )

        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelLoadError(
                "Install numpy and onnxruntime"
            ) from exc

        self._np = np

        try:
            self._session = ort.InferenceSession(
                str(specification.artifact_path),
                providers=self.providers,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"ONNX model loading failed: {exc}"
            ) from exc

    def predict(
        self,
        inputs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self._session is None:
            raise ModelLoadError(
                "ONNX model is not loaded"
            )

        session_inputs = {
            item.name: self._np.asarray(
                inputs[item.name]
            )
            for item in self._session.get_inputs()
        }

        output_names = [
            item.name
            for item in self._session.get_outputs()
        ]

        predictions = self._session.run(
            output_names,
            session_inputs,
        )

        return {
            name: json_safe(value)
            for name, value in zip(
                output_names,
                predictions,
            )
        }

    def predict_batch(
        self,
        batch: Sequence[Mapping[str, Any]],
    ) -> Sequence[Mapping[str, Any]]:
        return [
            self.predict(item)
            for item in batch
        ]

    def unload(self) -> None:
        self._session = None
        self._np = None


# ============================================================
# PYTORCH ADAPTER
# ============================================================

class PyTorchAdapter:
    """
    Safe-by-default PyTorch adapter.

    Requires an application-supplied model factory.
    Loads state_dict weights using weights_only=True.

    Arbitrary serialized Python model objects
    are intentionally not supported.
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        *,
        device: str = "cpu",
        input_name: str = "features",
    ) -> None:
        self.model_factory = model_factory
        self.device = device
        self.input_name = input_name

        self._torch: Any = None
        self._model: Any = None

    def load(
        self,
        specification: ModelSpecification,
    ) -> None:
        if specification.artifact_path is None:
            raise ModelLoadError(
                "PyTorch artifact path is required"
            )

        try:
            import torch
        except ImportError as exc:
            raise ModelLoadError(
                "Install torch"
            ) from exc

        model = self.model_factory()

        try:
            state_dict = torch.load(
                str(specification.artifact_path),
                map_location=self.device,
                weights_only=True,
            )

            model.load_state_dict(
                state_dict,
                strict=True,
            )

            model.to(self.device)
            model.eval()

        except Exception as exc:
            raise ModelLoadError(
                f"PyTorch loading failed: {exc}"
            ) from exc

        self._torch = torch
        self._model = model

    def predict(
        self,
        inputs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self._model is None:
            raise ModelLoadError(
                "PyTorch model is not loaded"
            )

        if self.input_name not in inputs:
            raise ModelValidationError(
                f"Missing input: {self.input_name}"
            )

        tensor = self._torch.as_tensor(
            inputs[self.input_name],
            dtype=self._torch.float32,
            device=self.device,
        )

        with self._torch.inference_mode():
            prediction = self._model(
                tensor
            )

        return {
            "prediction": json_safe(
                prediction.detach().cpu().numpy()
            )
        }

    def predict_batch(
        self,
        batch: Sequence[Mapping[str, Any]],
    ) -> Sequence[Mapping[str, Any]]:
        return [
            self.predict(item)
            for item in batch
        ]

    def unload(self) -> None:
        self._model = None
        self._torch = None


# ============================================================
# MODEL REGISTRY ADAPTER
# ============================================================

class ModelRegistryAdapter(Protocol):

    def resolve_model(
        self,
        tenant_id: str,
        model_id: str,
        version_id: str,
    ) -> ModelSpecification:
        ...


class InMemoryModelRegistryAdapter:

    def __init__(self) -> None:
        self._models: dict[
            tuple[str, str, str],
            ModelSpecification,
        ] = {}

        self._lock = threading.RLock()

    def register(
        self,
        specification: ModelSpecification,
    ) -> None:
        key = (
            specification.tenant_id,
            specification.model_id,
            specification.version_id,
        )

        with self._lock:
            if key in self._models:
                raise ModelValidationError(
                    "Model version already registered"
                )

            self._models[key] = specification

    def resolve_model(
        self,
        tenant_id: str,
        model_id: str,
        version_id: str,
    ) -> ModelSpecification:
        key = (
            tenant_id,
            model_id,
            version_id,
        )

        with self._lock:
            specification = self._models.get(
                key
            )

        if specification is None:
            raise ModelNotFoundError(
                "Model version not found in registry"
            )

        return specification


# ============================================================
# MODEL ARTIFACT VERIFICATION
# ============================================================

class ModelArtifactVerifier:

    @staticmethod
    def verify(
        specification: ModelSpecification,
    ) -> bool:
        if specification.framework == ModelFramework.SYNTHETIC:
            return True

        if (
            specification.artifact_path is None
            or specification.artifact_sha256 is None
        ):
            return False

        path = Path(
            specification.artifact_path
        )

        if path.is_symlink() or not path.is_file():
            return False

        return (
            sha256_file(path)
            == specification.artifact_sha256
        )


# ============================================================
# MODEL INSTANCE
# ============================================================

class LoadedModel:

    def __init__(
        self,
        specification: ModelSpecification,
        adapter: ModelAdapter,
    ) -> None:
        self.specification = specification
        self.adapter = adapter

        self.status = ModelStatus.REGISTERED

        self.loaded_at: datetime | None = None

        self._lock = threading.RLock()

    def load(self) -> None:
        with self._lock:
            if self.status == ModelStatus.READY:
                return

            self.status = ModelStatus.LOADING

            try:
                self.adapter.load(
                    self.specification
                )

                self.loaded_at = utc_now()
                self.status = ModelStatus.READY

            except Exception as exc:
                self.status = ModelStatus.FAILED

                raise ModelLoadError(
                    f"Model loading failed: {exc}"
                ) from exc

    def predict(
        self,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            if self.status != ModelStatus.READY:
                raise ModelInferenceError(
                    "Model is not ready"
                )

            try:
                outputs = self.adapter.predict(
                    inputs
                )

                return InferenceOutputValidator.validate(
                    outputs,
                    self.specification,
                )

            except ModelServingError:
                raise

            except Exception as exc:
                raise ModelInferenceError(
                    f"Inference failed: {exc}"
                ) from exc

    def predict_batch(
        self,
        inputs: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        with self._lock:
            if self.status != ModelStatus.READY:
                raise ModelInferenceError(
                    "Model is not ready"
                )

            outputs = self.adapter.predict_batch(
                inputs
            )

            if len(outputs) != len(inputs):
                raise ModelInferenceError(
                    "Batch output size mismatch"
                )

            return [
                InferenceOutputValidator.validate(
                    output,
                    self.specification,
                )
                for output in outputs
            ]

    def unload(self) -> None:
        with self._lock:
            self.adapter.unload()

            self.status = ModelStatus.UNLOADED
            self.loaded_at = None


# ============================================================
# MODEL ROUTER
# ============================================================

class ModelServingRouter:

    def __init__(self) -> None:
        self._weights: dict[
            tuple[str, str],
            tuple[ModelTrafficWeight, ...],
        ] = {}

        self._lock = threading.RLock()

    def configure_weights(
        self,
        tenant_id: str,
        model_id: str,
        weights: Sequence[ModelTrafficWeight],
    ) -> None:
        if not weights:
            raise ModelRoutingError(
                "Traffic weights cannot be empty"
            )

        total = sum(
            validate_finite_number(
                item.weight,
                "weight",
            )
            for item in weights
        )

        if any(
            item.weight < 0
            for item in weights
        ):
            raise ModelRoutingError(
                "Traffic weights cannot be negative"
            )

        if not math.isclose(
            total,
            100.0,
            abs_tol=1e-6,
        ):
            raise ModelRoutingError(
                "Traffic weights must total 100%"
            )

        version_ids = [
            item.version_id
            for item in weights
        ]

        if len(version_ids) != len(set(version_ids)):
            raise ModelRoutingError(
                "Duplicate model versions in traffic configuration"
            )

        with self._lock:
            self._weights[
                (tenant_id, model_id)
            ] = tuple(weights)

    def resolve(
        self,
        tenant_id: str,
        model_id: str,
        *,
        explicit_version: str | None = None,
        available_versions: Sequence[str] = (),
    ) -> str:
        available = set(
            available_versions
        )

        if explicit_version is not None:
            if explicit_version not in available:
                raise ModelNotFoundError(
                    "Requested model version is unavailable"
                )

            return explicit_version

        with self._lock:
            weights = self._weights.get(
                (tenant_id, model_id)
            )

        if weights:
            eligible = [
                item
                for item in weights
                if item.version_id in available
                and item.weight > 0
            ]

            if not eligible:
                raise ModelRoutingError(
                    "No configured traffic target is available"
                )

            return random.choices(
                [item.version_id for item in eligible],
                weights=[item.weight for item in eligible],
                k=1,
            )[0]

        if len(available) == 1:
            return next(iter(available))

        raise ModelRoutingError(
            "Multiple versions available; configure routing"
        )


# ============================================================
# RATE LIMITER
# ============================================================

class SlidingWindowRateLimiter:

    def __init__(
        self,
        requests_per_minute: int,
    ) -> None:
        self.limit = requests_per_minute

        self._requests: dict[
            tuple[str, str],
            deque[float],
        ] = defaultdict(deque)

        self._lock = threading.RLock()

    def check(
        self,
        tenant_id: str,
        user_id: str,
    ) -> None:
        key = (
            tenant_id,
            user_id,
        )

        now = time.monotonic()

        with self._lock:
            timestamps = self._requests[key]

            while (
                timestamps
                and now - timestamps[0] >= 60.0
            ):
                timestamps.popleft()

            if len(timestamps) >= self.limit:
                raise ModelRateLimitError(
                    "Inference rate limit exceeded"
                )

            timestamps.append(now)


# ============================================================
# INFERENCE METRICS
# ============================================================

class ModelServingMetrics:

    def __init__(
        self,
        history_size: int = 10000,
    ) -> None:
        self._metrics: deque[InferenceMetric] = deque(
            maxlen=history_size
        )

        self._lock = threading.RLock()

    def record(
        self,
        metric: InferenceMetric,
    ) -> None:
        with self._lock:
            self._metrics.append(
                metric
            )

    def get_metrics(
        self,
        tenant_id: str,
        model_id: str,
        version_id: str,
    ) -> tuple[InferenceMetric, ...]:
        with self._lock:
            return tuple(
                metric
                for metric in self._metrics
                if (
                    metric.tenant_id == tenant_id
                    and metric.model_id == model_id
                    and metric.version_id == version_id
                )
            )

    @staticmethod
    def percentile(
        values: Sequence[float],
        percentile: float,
    ) -> float:
        if not values:
            return 0.0

        ordered = sorted(values)

        position = (
            (len(ordered) - 1)
            * percentile
            / 100.0
        )

        lower = math.floor(position)
        upper = math.ceil(position)

        if lower == upper:
            return ordered[lower]

        fraction = position - lower

        return (
            ordered[lower] * (1.0 - fraction)
            + ordered[upper] * fraction
        )

    def health(
        self,
        model: LoadedModel,
    ) -> ModelHealth:
        specification = model.specification

        metrics = self.get_metrics(
            specification.tenant_id,
            specification.model_id,
            specification.version_id,
        )

        total = len(metrics)

        successes = sum(
            metric.status == InferenceStatus.SUCCESS
            for metric in metrics
        )

        failures = total - successes

        latencies = [
            metric.latency_ms
            for metric in metrics
        ]

        error_rate = (
            failures / total
            if total
            else 0.0
        )

        return ModelHealth(
            model_id=specification.model_id,
            version_id=specification.version_id,
            status=model.status,
            loaded=(
                model.status == ModelStatus.READY
            ),
            total_requests=total,
            successful_requests=successes,
            failed_requests=failures,
            error_rate=error_rate,
            average_latency_ms=(
                statistics.mean(latencies)
                if latencies
                else 0.0
            ),
            p95_latency_ms=self.percentile(
                latencies,
                95,
            ),
            checked_at=utc_now(),
        )


# ============================================================
# MODEL SERVING SERVICE
# ============================================================

class ModelServingService:

    def __init__(
        self,
        config: ModelServingConfig | None = None,
        *,
        registry: ModelRegistryAdapter | None = None,
    ) -> None:
        self.config = (
            config or ModelServingConfig()
        )

        self.registry = (
            registry or InMemoryModelRegistryAdapter()
        )

        self.router = ModelServingRouter()

        self.metrics = ModelServingMetrics(
            self.config.metrics_history_size
        )

        self.rate_limiter = SlidingWindowRateLimiter(
            self.config.requests_per_minute
        )

        self._models: dict[
            tuple[str, str, str],
            LoadedModel,
        ] = {}

        self._lock = threading.RLock()

        self._executor = ThreadPoolExecutor(
            max_workers=self.config.maximum_workers,
            thread_name_prefix="model-serving",
        )

        self._closed = False

    def __enter__(self) -> ModelServingService:
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    @staticmethod
    def _authorize(
        access: ServingAccess,
        permission: str,
    ) -> None:
        if not getattr(
            access,
            permission,
            False,
        ):
            raise ModelAuthorizationError(
                f"Missing permission: {permission}"
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ModelServingError(
                "Model serving service is closed"
            )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    def load_model(
        self,
        model_id: str,
        version_id: str,
        access: ServingAccess,
        *,
        adapter: ModelAdapter,
    ) -> ModelSpecification:
        self._authorize(
            access,
            "can_manage",
        )

        self._ensure_open()

        specification = self.registry.resolve_model(
            access.tenant_id,
            model_id,
            version_id,
        )

        if self.config.verify_artifacts:
            if not ModelArtifactVerifier.verify(
                specification
            ):
                raise ModelIntegrityError(
                    "Model artifact integrity verification failed"
                )

        key = (
            access.tenant_id,
            model_id,
            version_id,
        )

        with self._lock:
            if key in self._models:
                raise ModelValidationError(
                    "Model version already loaded"
                )

            model = LoadedModel(
                specification,
                adapter,
            )

            model.load()

            self._models[key] = model

        return specification

    # --------------------------------------------------------
    # UNLOAD MODEL
    # --------------------------------------------------------

    def unload_model(
        self,
        model_id: str,
        version_id: str,
        access: ServingAccess,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        key = (
            access.tenant_id,
            model_id,
            version_id,
        )

        with self._lock:
            model = self._models.pop(
                key,
                None,
            )

            if model is None:
                raise ModelNotFoundError(
                    "Loaded model not found"
                )

            model.unload()

    # --------------------------------------------------------
    # CONFIGURE ROUTING
    # --------------------------------------------------------

    def configure_traffic(
        self,
        model_id: str,
        weights: Sequence[ModelTrafficWeight],
        access: ServingAccess,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            available = {
                version_id
                for tenant_id, registered_model_id, version_id
                in self._models
                if (
                    tenant_id == access.tenant_id
                    and registered_model_id == model_id
                )
            }

            for item in weights:
                if item.version_id not in available:
                    raise ModelNotFoundError(
                        f"Version not loaded: {item.version_id}"
                    )

            self.router.configure_weights(
                access.tenant_id,
                model_id,
                weights,
            )

    # --------------------------------------------------------
    # RESOLVE MODEL
    # --------------------------------------------------------

    def _resolve_model(
        self,
        request: InferenceRequest,
    ) -> LoadedModel:
        with self._lock:
            versions = [
                version_id
                for tenant_id, model_id, version_id
                in self._models
                if (
                    tenant_id == request.tenant_id
                    and model_id == request.model_id
                )
            ]

            if not versions:
                raise ModelNotFoundError(
                    "No loaded versions for requested model"
                )

            if (
                request.version_id is not None
                and not self.config.allow_explicit_version_routing
            ):
                raise ModelRoutingError(
                    "Explicit version routing is disabled"
                )

            version_id = self.router.resolve(
                request.tenant_id,
                request.model_id,
                explicit_version=request.version_id,
                available_versions=versions,
            )

            return self._models[
                (
                    request.tenant_id,
                    request.model_id,
                    version_id,
                )
            ]

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    def predict(
        self,
        request: InferenceRequest,
        access: ServingAccess,
    ) -> InferenceResponse:
        self._authorize(
            access,
            "can_infer",
        )

        self._ensure_open()

        if request.tenant_id != access.tenant_id:
            raise ModelAuthorizationError(
                "Cross-tenant inference is forbidden"
            )

        self.rate_limiter.check(
            access.tenant_id,
            access.user_id,
        )

        model = self._resolve_model(
            request
        )

        specification = model.specification

        inputs = InferenceInputValidator.validate(
            request.inputs,
            specification,
            self.config.maximum_input_bytes,
        )

        timeout_seconds = (
            request.timeout_seconds
            if request.timeout_seconds is not None
            else self.config.default_timeout_seconds
        )

        if timeout_seconds <= 0:
            raise ModelValidationError(
                "timeout_seconds must be positive"
            )

        started_at = utc_now()
        started = time.perf_counter()

        status = InferenceStatus.SUCCESS
        outputs: dict[str, Any] = {}
        error: str | None = None

        try:
            # Synchronous execution measures elapsed time.
            # It cannot forcibly interrupt a blocking model.
            outputs = model.predict(
                inputs
            )

            elapsed = (
                time.perf_counter() - started
            )

            if elapsed > timeout_seconds:
                status = InferenceStatus.TIMEOUT
                outputs = {}
                error = "Inference exceeded timeout"

        except Exception as exc:
            status = InferenceStatus.FAILED
            error = str(exc)

        latency_ms = (
            time.perf_counter() - started
        ) * 1000.0

        completed_at = utc_now()

        self.metrics.record(
            InferenceMetric(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                model_id=request.model_id,
                version_id=specification.version_id,
                status=status,
                latency_ms=latency_ms,
                timestamp=completed_at,
            )
        )

        return InferenceResponse(
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            model_id=request.model_id,
            version_id=specification.version_id,
            outputs=outputs,
            status=status,
            latency_ms=latency_ms,
            created_at=started_at,
            completed_at=completed_at,
            error=error,
        )

    # --------------------------------------------------------
    # ASYNC INFERENCE
    # --------------------------------------------------------

    async def predict_async(
        self,
        request: InferenceRequest,
        access: ServingAccess,
    ) -> InferenceResponse:
        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            self._executor,
            self.predict,
            request,
            access,
        )

    # --------------------------------------------------------
    # BATCH INFERENCE
    # --------------------------------------------------------

    def predict_batch(
        self,
        requests: Sequence[InferenceRequest],
        access: ServingAccess,
    ) -> tuple[InferenceResponse, ...]:
        self._authorize(
            access,
            "can_infer",
        )

        if len(requests) > self.config.maximum_batch_size:
            raise ModelValidationError(
                "Batch size exceeds configured maximum"
            )

        if not requests:
            return ()

        # Individual routing and validation preserve
        # tenant isolation and per-request metrics.
        return tuple(
            self.predict(
                request,
                access,
            )
            for request in requests
        )

    async def predict_batch_async(
        self,
        requests: Sequence[InferenceRequest],
        access: ServingAccess,
    ) -> tuple[InferenceResponse, ...]:
        if len(requests) > self.config.maximum_batch_size:
            raise ModelValidationError(
                "Batch size exceeds configured maximum"
            )

        results = await asyncio.gather(
            *(
                self.predict_async(
                    request,
                    access,
                )
                for request in requests
            )
        )

        return tuple(results)

    # --------------------------------------------------------
    # WARM UP
    # --------------------------------------------------------

    def warmup(
        self,
        model_id: str,
        version_id: str,
        sample_inputs: Mapping[str, Any],
        access: ServingAccess,
        *,
        iterations: int = 3,
    ) -> dict[str, Any]:
        self._authorize(
            access,
            "can_manage",
        )

        if iterations <= 0:
            raise ModelValidationError(
                "iterations must be positive"
            )

        request = InferenceRequest(
            request_id=generate_id(),
            tenant_id=access.tenant_id,
            model_id=model_id,
            version_id=version_id,
            inputs=dict(sample_inputs),
        )

        internal_access = ServingAccess(
            tenant_id=access.tenant_id,
            user_id=access.user_id,
            can_infer=True,
        )

        latencies = []

        for _ in range(iterations):
            response = self.predict(
                request,
                internal_access,
            )

            if response.status != InferenceStatus.SUCCESS:
                raise ModelInferenceError(
                    f"Warm-up failed: {response.error}"
                )

            latencies.append(
                response.latency_ms
            )

        return {
            "model_id": model_id,
            "version_id": version_id,
            "iterations": iterations,
            "average_latency_ms": statistics.mean(
                latencies
            ),
            "minimum_latency_ms": min(latencies),
            "maximum_latency_ms": max(latencies),
        }

    # --------------------------------------------------------
    # HEALTH CHECK
    # --------------------------------------------------------

    def health_check(
        self,
        model_id: str,
        version_id: str,
        access: ServingAccess,
    ) -> ModelHealth:
        self._authorize(
            access,
            "can_monitor",
        )

        key = (
            access.tenant_id,
            model_id,
            version_id,
        )

        with self._lock:
            model = self._models.get(
                key
            )

        if model is None:
            raise ModelNotFoundError(
                "Loaded model not found"
            )

        return self.metrics.health(
            model
        )

    # --------------------------------------------------------
    # LIST LOADED MODELS
    # --------------------------------------------------------

    def list_loaded_models(
        self,
        access: ServingAccess,
    ) -> tuple[dict[str, Any], ...]:
        self._authorize(
            access,
            "can_monitor",
        )

        with self._lock:
            return tuple(
                {
                    "model_id": model.specification.model_id,
                    "version_id": model.specification.version_id,
                    "model_name": model.specification.model_name,
                    "framework": model.specification.framework.value,
                    "task": model.specification.task.value,
                    "stage": model.specification.stage.value,
                    "status": model.status.value,
                    "loaded_at": (
                        model.loaded_at.isoformat()
                        if model.loaded_at
                        else None
                    ),
                }
                for key, model in self._models.items()
                if key[0] == access.tenant_id
            )

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return

            self._closed = True

            for model in self._models.values():
                model.unload()

            self._models.clear()

        self._executor.shutdown(
            wait=True,
        )


# ============================================================
# OPTIONAL FASTAPI INTEGRATION
# ============================================================

def create_fastapi_app(
    service: ModelServingService,
    access_resolver: Callable[[str, str], ServingAccess],
) -> Any:
    """
    Create a FastAPI application.

    access_resolver must authenticate and authorize
    the caller using trusted server-side credentials.

    Never use unverified client-supplied tenant or
    user identifiers as proof of authorization.
    """

    try:
        from fastapi import FastAPI, Header, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ModelServingError(
            "Install fastapi and pydantic"
        ) from exc

    class PredictPayload(BaseModel):
        inputs: dict[str, Any]

        version_id: str | None = None

        timeout_seconds: float | None = Field(
            default=None,
            gt=0,
        )

    app = FastAPI(
        title="Physical AI Model Serving",
        version="1.0.0",
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "running",
            "service": "model_serving",
            "timestamp": utc_now().isoformat(),
        }

    @app.post("/v1/models/{model_id}/predict")
    async def predict_endpoint(
        model_id: str,
        payload: PredictPayload,
        authorization: str = Header(...),
    ) -> dict[str, Any]:
        try:
            # The resolver is responsible for verifying
            # the authorization token and returning
            # authenticated tenant/user permissions.
            access = access_resolver(
                authorization,
                model_id,
            )

            request = InferenceRequest(
                request_id=generate_id(),
                tenant_id=access.tenant_id,
                model_id=model_id,
                version_id=payload.version_id,
                inputs=payload.inputs,
                timeout_seconds=payload.timeout_seconds,
            )

            response = await service.predict_async(
                request,
                access,
            )

            return json_safe(
                {
                    "request_id": response.request_id,
                    "model_id": response.model_id,
                    "version_id": response.version_id,
                    "status": response.status,
                    "outputs": response.outputs,
                    "latency_ms": response.latency_ms,
                    "error": response.error,
                }
            )

        except ModelAuthorizationError as exc:
            raise HTTPException(
                status_code=403,
                detail=str(exc),
            ) from exc

        except ModelNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

        except ModelRateLimitError as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
            ) from exc

        except ModelValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        except ModelServingError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc

    return app


# ============================================================
# SYNTHETIC MODEL FACTORY
# ============================================================

class SyntheticServingFactory:

    @staticmethod
    def specification(
        *,
        tenant_id: str = "robotics_lab",
        model_id: str = "robot-perception",
        version_id: str = "v1",
        stage: ModelStage = ModelStage.PRODUCTION,
    ) -> ModelSpecification:
        return ModelSpecification(
            model_id=model_id,
            version_id=version_id,
            tenant_id=tenant_id,
            model_name="Robot Perception Model",
            framework=ModelFramework.SYNTHETIC,
            task=ModelTask.ROBOT_PERCEPTION,
            stage=stage,
            input_names=("features",),
            output_names=(
                "prediction",
                "confidence",
                "score",
            ),
            metadata={
                "synthetic": True,
                "robot_actuation": False,
            },
        )

    @staticmethod
    def request(
        *,
        tenant_id: str = "robotics_lab",
        model_id: str = "robot-perception",
        version_id: str | None = None,
        features: Sequence[float] = (
            0.4,
            0.8,
            0.2,
        ),
    ) -> InferenceRequest:
        return InferenceRequest(
            request_id=generate_id(),
            tenant_id=tenant_id,
            model_id=model_id,
            version_id=version_id,
            inputs={
                "features": list(features),
            },
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestModelServing(unittest.TestCase):

    def setUp(self) -> None:
        self.registry = InMemoryModelRegistryAdapter()

        self.specification = (
            SyntheticServingFactory.specification()
        )

        self.registry.register(
            self.specification
        )

        self.service = ModelServingService(
            ModelServingConfig(
                maximum_workers=4,
                maximum_batch_size=8,
                requests_per_minute=1000,
            ),
            registry=self.registry,
        )

        self.manager = ServingAccess(
            tenant_id="robotics_lab",
            user_id="mlops_engineer",
            can_infer=True,
            can_manage=True,
            can_monitor=True,
        )

        self.service.load_model(
            "robot-perception",
            "v1",
            self.manager,
            adapter=SyntheticModelAdapter(),
        )

    def tearDown(self) -> None:
        self.service.close()

    def test_model_loading(self) -> None:
        models = self.service.list_loaded_models(
            self.manager
        )

        self.assertEqual(
            len(models),
            1,
        )

    def test_single_inference(self) -> None:
        response = self.service.predict(
            SyntheticServingFactory.request(),
            self.manager,
        )

        self.assertEqual(
            response.status,
            InferenceStatus.SUCCESS,
        )

    def test_output_fields(self) -> None:
        response = self.service.predict(
            SyntheticServingFactory.request(),
            self.manager,
        )

        self.assertIn(
            "prediction",
            response.outputs,
        )

        self.assertIn(
            "confidence",
            response.outputs,
        )

    def test_explicit_version(self) -> None:
        response = self.service.predict(
            SyntheticServingFactory.request(
                version_id="v1"
            ),
            self.manager,
        )

        self.assertEqual(
            response.version_id,
            "v1",
        )

    def test_invalid_input(self) -> None:
        request = InferenceRequest(
            request_id=generate_id(),
            tenant_id="robotics_lab",
            model_id="robot-perception",
            inputs={},
        )

        with self.assertRaises(
            ModelValidationError
        ):
            self.service.predict(
                request,
                self.manager,
            )

    def test_batch_inference(self) -> None:
        requests = [
            SyntheticServingFactory.request()
            for _ in range(5)
        ]

        responses = self.service.predict_batch(
            requests,
            self.manager,
        )

        self.assertEqual(
            len(responses),
            5,
        )

    def test_batch_size_limit(self) -> None:
        requests = [
            SyntheticServingFactory.request()
            for _ in range(9)
        ]

        with self.assertRaises(
            ModelValidationError
        ):
            self.service.predict_batch(
                requests,
                self.manager,
            )

    def test_async_inference(self) -> None:
        async def run() -> InferenceResponse:
            return await self.service.predict_async(
                SyntheticServingFactory.request(),
                self.manager,
            )

        response = asyncio.run(
            run()
        )

        self.assertEqual(
            response.status,
            InferenceStatus.SUCCESS,
        )

    def test_async_batch(self) -> None:
        async def run() -> tuple[InferenceResponse, ...]:
            return await self.service.predict_batch_async(
                [
                    SyntheticServingFactory.request()
                    for _ in range(4)
                ],
                self.manager,
            )

        responses = asyncio.run(
            run()
        )

        self.assertEqual(
            len(responses),
            4,
        )

    def test_health_check(self) -> None:
        self.service.predict(
            SyntheticServingFactory.request(),
            self.manager,
        )

        health = self.service.health_check(
            "robot-perception",
            "v1",
            self.manager,
        )

        self.assertEqual(
            health.total_requests,
            1,
        )

        self.assertEqual(
            health.successful_requests,
            1,
        )

    def test_warmup(self) -> None:
        result = self.service.warmup(
            "robot-perception",
            "v1",
            {"features": [0.1, 0.2, 0.3]},
            self.manager,
            iterations=3,
        )

        self.assertEqual(
            result["iterations"],
            3,
        )

    def test_tenant_isolation(self) -> None:
        other = ServingAccess(
            tenant_id="other_tenant",
            user_id="other_user",
            can_infer=True,
        )

        with self.assertRaises(
            ModelAuthorizationError
        ):
            self.service.predict(
                SyntheticServingFactory.request(),
                other,
            )

    def test_permission_denied(self) -> None:
        access = ServingAccess(
            tenant_id="robotics_lab",
            user_id="readonly",
        )

        with self.assertRaises(
            ModelAuthorizationError
        ):
            self.service.predict(
                SyntheticServingFactory.request(),
                access,
            )

    def test_weighted_routing(self) -> None:
        second = SyntheticServingFactory.specification(
            version_id="v2",
            stage=ModelStage.CANARY,
        )

        self.registry.register(
            second
        )

        self.service.load_model(
            "robot-perception",
            "v2",
            self.manager,
            adapter=SyntheticModelAdapter(),
        )

        self.service.configure_traffic(
            "robot-perception",
            [
                ModelTrafficWeight(
                    version_id="v1",
                    weight=100.0,
                ),
                ModelTrafficWeight(
                    version_id="v2",
                    weight=0.0,
                ),
            ],
            self.manager,
        )

        response = self.service.predict(
            SyntheticServingFactory.request(),
            self.manager,
        )

        self.assertEqual(
            response.version_id,
            "v1",
        )

    def test_unload_model(self) -> None:
        self.service.unload_model(
            "robot-perception",
            "v1",
            self.manager,
        )

        self.assertEqual(
            len(
                self.service.list_loaded_models(
                    self.manager
                )
            ),
            0,
        )

    def test_duplicate_loading(self) -> None:
        with self.assertRaises(
            ModelValidationError
        ):
            self.service.load_model(
                "robot-perception",
                "v1",
                self.manager,
                adapter=SyntheticModelAdapter(),
            )

    def test_model_not_found(self) -> None:
        with self.assertRaises(
            ModelNotFoundError
        ):
            self.service.predict(
                SyntheticServingFactory.request(
                    model_id="missing-model"
                ),
                self.manager,
            )

    def test_invalid_traffic_weights(self) -> None:
        with self.assertRaises(
            ModelRoutingError
        ):
            self.service.configure_traffic(
                "robot-perception",
                [
                    ModelTrafficWeight(
                        version_id="v1",
                        weight=70.0,
                    ),
                ],
                self.manager,
            )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    registry = InMemoryModelRegistryAdapter()

    production = SyntheticServingFactory.specification(
        tenant_id="physical_ai_lab",
        model_id="autonomous-robot-perception",
        version_id="v1.0.0",
        stage=ModelStage.PRODUCTION,
    )

    canary = SyntheticServingFactory.specification(
        tenant_id="physical_ai_lab",
        model_id="autonomous-robot-perception",
        version_id="v2.0.0",
        stage=ModelStage.CANARY,
    )

    registry.register(production)
    registry.register(canary)

    access = ServingAccess(
        tenant_id="physical_ai_lab",
        user_id="mlops_engineer",
        can_infer=True,
        can_manage=True,
        can_monitor=True,
    )

    with ModelServingService(
        ModelServingConfig(
            maximum_workers=4,
            maximum_batch_size=16,
        ),
        registry=registry,
    ) as service:
        service.load_model(
            production.model_id,
            production.version_id,
            access,
            adapter=SyntheticModelAdapter(),
        )

        service.load_model(
            canary.model_id,
            canary.version_id,
            access,
            adapter=SyntheticModelAdapter(),
        )

        service.configure_traffic(
            production.model_id,
            [
                ModelTrafficWeight(
                    version_id="v1.0.0",
                    weight=90.0,
                ),
                ModelTrafficWeight(
                    version_id="v2.0.0",
                    weight=10.0,
                ),
            ],
            access,
        )

        responses = []

        for index in range(20):
            request = InferenceRequest(
                request_id=generate_id(),
                tenant_id="physical_ai_lab",
                model_id=production.model_id,
                inputs={
                    "features": [
                        index / 20.0,
                        0.65,
                        0.42,
                    ],
                },
            )

            responses.append(
                service.predict(
                    request,
                    access,
                )
            )

        production_health = service.health_check(
            production.model_id,
            production.version_id,
            access,
        )

        canary_health = service.health_check(
            canary.model_id,
            canary.version_id,
            access,
        )

        print("\nPHYSICAL AI MODEL SERVING")
        print("=" * 70)

        print(
            json.dumps(
                json_safe(
                    {
                        "loaded_models": (
                            service.list_loaded_models(
                                access
                            )
                        ),
                        "total_inference_requests": len(
                            responses
                        ),
                        "successful_requests": sum(
                            response.status
                            == InferenceStatus.SUCCESS
                            for response in responses
                        ),
                        "production_health": (
                            production_health
                        ).__dict__,
                        "canary_health": (
                            canary_health
                        ).__dict__,
                        "sample_prediction": (
                            responses[0].outputs
                        ),
                        "synthetic_model": True,
                        "physical_robot_control": False,
                    }
                ),
                indent=2,
                allow_nan=False,
            )
        )

    print("\nRUNNING UNIT TESTS")
    print("=" * 70)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestModelServing
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
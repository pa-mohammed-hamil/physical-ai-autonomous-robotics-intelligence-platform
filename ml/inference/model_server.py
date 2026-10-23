"""
Physical AI & Autonomous Robotics Intelligence Platform
File: ml/inference/model_server.py

Production-oriented FastAPI ML model serving API.

Features:
- Model registration and version management
- Single and batch inference
- Model health and readiness endpoints
- CPU, CUDA and MPS inference support
- Tenant-aware API key authentication
- Model-level access control
- Request validation
- Inference timeouts
- Concurrent request limits
- Prometheus-compatible metrics
- Model performance statistics
- Simulation-only VLA/navigation/manipulation inference
- Safe error handling
- Graceful shutdown

Dependencies:
    pip install fastapi uvicorn pydantic

Optional:
    pip install torch prometheus-client

Run:
    export MODEL_SERVER_API_KEYS='{"dev-key":{"tenant_id":"demo","user_id":"developer","models":["*"],"admin":true}}'
    uvicorn ml.inference.model_server:app --host 127.0.0.1 --port 8001

Security:
    Use HTTPS, managed secrets, and a trusted authentication
    gateway for production deployments.

    This API performs inference only. It never executes
    physical robot commands.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger("robotics.model_server")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class ServerSettings:
    app_name: str = "Physical AI Model Server"
    version: str = "1.0.0"

    max_batch_size: int = 32
    max_concurrent_inferences: int = 8

    default_timeout_seconds: float = 30.0
    max_timeout_seconds: float = 120.0

    enable_metrics: bool = True

    @classmethod
    def from_environment(cls) -> "ServerSettings":
        return cls(
            max_batch_size=int(
                os.getenv("MODEL_SERVER_MAX_BATCH_SIZE", "32")
            ),
            max_concurrent_inferences=int(
                os.getenv("MODEL_SERVER_CONCURRENCY", "8")
            ),
            default_timeout_seconds=float(
                os.getenv("MODEL_SERVER_TIMEOUT", "30")
            ),
            max_timeout_seconds=float(
                os.getenv("MODEL_SERVER_MAX_TIMEOUT", "120")
            ),
        )

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")

        if self.max_concurrent_inferences < 1:
            raise ValueError(
                "max_concurrent_inferences must be positive"
            )

        if not (
            0 < self.default_timeout_seconds
            <= self.max_timeout_seconds
        ):
            raise ValueError("Invalid inference timeout settings")


settings = ServerSettings.from_environment()


# ============================================================
# Enumerations
# ============================================================

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


class ModelDevice(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    REMOTE = "remote"


class ModelState(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class PredictionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


ACTION_MODEL_CATEGORIES = {
    ModelCategory.VLA,
    ModelCategory.NAVIGATION,
    ModelCategory.MANIPULATION,
}


# ============================================================
# Authentication
# ============================================================

@dataclass(frozen=True)
class APIPrincipal:
    tenant_id: str
    user_id: str
    allowed_models: frozenset[str]
    is_admin: bool = False

    def can_access_model(self, model_name: str) -> bool:
        return (
            "*" in self.allowed_models
            or model_name in self.allowed_models
        )


class APIKeyStore:
    """
    Load API keys from MODEL_SERVER_API_KEYS.

    Example:
    {
      "secret-key": {
        "tenant_id": "demo",
        "user_id": "developer",
        "models": ["perception-v1"],
        "admin": false
      }
    }
    """

    def __init__(self) -> None:
        self._keys: dict[str, APIPrincipal] = {}

    def load_from_environment(self) -> None:
        raw = os.getenv("MODEL_SERVER_API_KEYS")

        if not raw:
            logger.warning(
                "No model server API keys configured; "
                "protected endpoints will reject requests"
            )
            return

        try:
            data = json.loads(raw)

            if not isinstance(data, dict):
                raise ValueError("API key configuration must be an object")

            for api_key, principal in data.items():
                if (
                    not isinstance(api_key, str)
                    or not api_key
                    or not isinstance(principal, dict)
                ):
                    raise ValueError("Invalid API key entry")

                tenant_id = principal.get("tenant_id")
                user_id = principal.get("user_id")
                models = principal.get("models", [])
                admin = principal.get("admin", False)

                if not isinstance(tenant_id, str) or not tenant_id:
                    raise ValueError("Invalid tenant_id")

                if not isinstance(user_id, str) or not user_id:
                    raise ValueError("Invalid user_id")

                if (
                    not isinstance(models, list)
                    or not all(isinstance(m, str) for m in models)
                ):
                    raise ValueError("Invalid models list")

                if not isinstance(admin, bool):
                    raise ValueError("Invalid admin flag")

                self._keys[api_key] = APIPrincipal(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    allowed_models=frozenset(models),
                    is_admin=admin,
                )

        except (ValueError, json.JSONDecodeError):
            logger.exception("Invalid API key configuration")
            raise

    def authenticate(self, api_key: str) -> APIPrincipal | None:
        for stored_key, principal in self._keys.items():
            if hmac.compare_digest(stored_key, api_key):
                return principal

        return None


api_key_store = APIKeyStore()


async def get_principal(
    x_api_key: str | None = Header(default=None),
) -> APIPrincipal:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )

    principal = api_key_store.authenticate(x_api_key)

    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return principal


async def require_admin(
    principal: APIPrincipal = Depends(get_principal),
) -> APIPrincipal:
    if not principal.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator permission required",
        )

    return principal


# ============================================================
# Request Models
# ============================================================

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1, max_length=128)
    model_version: str | None = None

    inputs: Any

    robot_id: str | None = None

    simulation_only: bool = True

    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        le=120,
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1, max_length=128)
    model_version: str | None = None

    inputs: list[Any] = Field(min_length=1)

    simulation_only: bool = True

    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        le=120,
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    request_id: str

    model_name: str
    model_version: str

    status: PredictionStatus

    output: Any = None
    error: str | None = None

    device: ModelDevice

    latency_ms: float

    batch_size: int = 1

    simulation_only: bool = True

    physical_execution_authorized: bool = False

    completed_at: datetime


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    category: ModelCategory
    device: ModelDevice
    state: ModelState

    max_batch_size: int

    total_requests: int
    successful_requests: int
    failed_requests: int

    average_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime


# ============================================================
# Backend Protocol
# ============================================================

class ModelBackend(Protocol):
    async def predict(self, inputs: Any) -> Any:
        ...

    async def predict_batch(
        self,
        inputs: Sequence[Any],
    ) -> Sequence[Any]:
        ...


# ============================================================
# Model Metrics
# ============================================================

@dataclass
class ModelMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    total_latency_ms: float = 0.0

    @property
    def average_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0

        return (
            self.total_latency_ms
            / self.successful_requests
        )


# ============================================================
# Model Registration
# ============================================================

@dataclass
class RegisteredModel:
    name: str
    version: str
    category: ModelCategory

    backend: ModelBackend

    device: ModelDevice = ModelDevice.CPU
    state: ModelState = ModelState.READY

    max_batch_size: int = 32

    tenant_id: str | None = None

    metrics: ModelMetrics = field(
        default_factory=ModelMetrics
    )

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("Model name and version required")

        if self.max_batch_size < 1:
            raise ValueError("Invalid model batch size")


# ============================================================
# Model Registry
# ============================================================

class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[
            tuple[str, str],
            RegisteredModel,
        ] = {}

        self._defaults: dict[str, str] = {}

    def register(
        self,
        model: RegisteredModel,
        *,
        set_default: bool = True,
    ) -> None:
        key = (model.name, model.version)

        if key in self._models:
            raise ValueError(
                f"Model already registered: {key}"
            )

        self._models[key] = model

        if set_default or model.name not in self._defaults:
            self._defaults[model.name] = model.version

    def get(
        self,
        name: str,
        version: str | None = None,
    ) -> RegisteredModel:
        selected_version = version or self._defaults.get(name)

        if selected_version is None:
            raise KeyError("Model not found")

        model = self._models.get(
            (name, selected_version)
        )

        if model is None:
            raise KeyError("Model not found")

        if model.state not in {
            ModelState.READY,
            ModelState.DEGRADED,
        }:
            raise RuntimeError("Model unavailable")

        return model

    def list_models(self) -> list[RegisteredModel]:
        return list(self._models.values())

    def set_state(
        self,
        name: str,
        version: str,
        state: ModelState,
    ) -> None:
        model = self._models.get((name, version))

        if model is None:
            raise KeyError("Model not found")

        model.state = state


model_registry = ModelRegistry()


# ============================================================
# Safety Validation
# ============================================================

FORBIDDEN_AUTHORIZATION_FIELDS = {
    "physical_execution_authorized",
    "hardware_actuation_authorized",
    "emergency_stop_reset_authorized",
    "safety_interlock_bypass_authorized",
}


def validate_model_output(
    output: Any,
    *,
    depth: int = 0,
) -> None:
    if depth > 12:
        raise ValueError("Output nesting limit exceeded")

    if isinstance(output, Mapping):
        for key, value in output.items():
            if (
                key in FORBIDDEN_AUTHORIZATION_FIELDS
                and value is True
            ):
                raise ValueError(
                    "Unsafe physical authorization claim"
                )

            validate_model_output(
                value,
                depth=depth + 1,
            )

    elif isinstance(output, (list, tuple)):
        for item in output:
            validate_model_output(
                item,
                depth=depth + 1,
            )


def validate_inference_safety(
    model: RegisteredModel,
    *,
    simulation_only: bool,
    metadata: Mapping[str, Any],
) -> None:
    if (
        model.category in ACTION_MODEL_CATEGORIES
        and not simulation_only
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Robotic action models require "
                "simulation_only=true"
            ),
        )

    for key in FORBIDDEN_AUTHORIZATION_FIELDS:
        if metadata.get(key) is True:
            raise HTTPException(
                status_code=422,
                detail="Physical execution authorization is forbidden",
            )


def check_model_access(
    model: RegisteredModel,
    principal: APIPrincipal,
) -> None:
    if not principal.can_access_model(model.name):
        raise HTTPException(
            status_code=403,
            detail="Model access denied",
        )

    if (
        model.tenant_id is not None
        and model.tenant_id != principal.tenant_id
    ):
        raise HTTPException(
            status_code=404,
            detail="Model not found",
        )


# ============================================================
# Inference Engine
# ============================================================

class ModelInferenceEngine:
    def __init__(
        self,
        registry: ModelRegistry,
        config: ServerSettings,
    ) -> None:
        self.registry = registry
        self.config = config

        self._semaphore = asyncio.Semaphore(
            config.max_concurrent_inferences
        )

        self._closed = False

    def _get_model(
        self,
        name: str,
        version: str | None,
        principal: APIPrincipal,
    ) -> RegisteredModel:
        try:
            model = self.registry.get(name, version)

        except KeyError:
            raise HTTPException(
                status_code=404,
                detail="Model not found",
            ) from None

        except RuntimeError:
            raise HTTPException(
                status_code=503,
                detail="Model unavailable",
            ) from None

        check_model_access(model, principal)

        return model

    async def predict(
        self,
        request: PredictRequest,
        principal: APIPrincipal,
    ) -> PredictionResponse:
        if self._closed:
            raise HTTPException(
                status_code=503,
                detail="Inference service shutting down",
            )

        model = self._get_model(
            request.model_name,
            request.model_version,
            principal,
        )

        validate_inference_safety(
            model,
            simulation_only=request.simulation_only,
            metadata=request.metadata,
        )

        timeout = (
            request.timeout_seconds
            or self.config.default_timeout_seconds
        )

        timeout = min(
            timeout,
            self.config.max_timeout_seconds,
        )

        request_id = str(uuid.uuid4())
        started = time.perf_counter()

        model.metrics.total_requests += 1

        try:
            async with self._semaphore:
                output = await asyncio.wait_for(
                    model.backend.predict(request.inputs),
                    timeout=timeout,
                )

                validate_model_output(output)

            prediction_status = PredictionStatus.COMPLETED
            error = None

        except asyncio.TimeoutError:
            prediction_status = PredictionStatus.TIMED_OUT
            output = None
            error = "Inference timeout exceeded"

        except Exception:
            logger.exception(
                "Model inference failed: %s",
                model.name,
            )

            prediction_status = PredictionStatus.FAILED
            output = None
            error = "Model inference failed"

        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        if prediction_status == PredictionStatus.COMPLETED:
            model.metrics.successful_requests += 1
            model.metrics.total_latency_ms += latency_ms
        else:
            model.metrics.failed_requests += 1

        return PredictionResponse(
            request_id=request_id,
            model_name=model.name,
            model_version=model.version,
            status=prediction_status,
            output=output,
            error=error,
            device=model.device,
            latency_ms=round(latency_ms, 3),
            batch_size=1,
            simulation_only=request.simulation_only,
            physical_execution_authorized=False,
            completed_at=datetime.now(timezone.utc),
        )

    async def predict_batch(
        self,
        request: BatchPredictRequest,
        principal: APIPrincipal,
    ) -> PredictionResponse:
        if self._closed:
            raise HTTPException(
                status_code=503,
                detail="Inference service shutting down",
            )

        model = self._get_model(
            request.model_name,
            request.model_version,
            principal,
        )

        validate_inference_safety(
            model,
            simulation_only=request.simulation_only,
            metadata=request.metadata,
        )

        batch_size = len(request.inputs)

        if batch_size > min(
            self.config.max_batch_size,
            model.max_batch_size,
        ):
            raise HTTPException(
                status_code=422,
                detail="Batch size exceeds model limit",
            )

        timeout = (
            request.timeout_seconds
            or self.config.default_timeout_seconds
        )

        timeout = min(
            timeout,
            self.config.max_timeout_seconds,
        )

        request_id = str(uuid.uuid4())
        started = time.perf_counter()

        model.metrics.total_requests += batch_size

        try:
            async with self._semaphore:
                outputs = await asyncio.wait_for(
                    model.backend.predict_batch(request.inputs),
                    timeout=timeout,
                )

                if (
                    isinstance(outputs, (str, bytes, Mapping))
                    or not isinstance(outputs, Sequence)
                    or len(outputs) != batch_size
                ):
                    raise ValueError(
                        "Invalid batch output count or type"
                    )

                validate_model_output(outputs)

            prediction_status = PredictionStatus.COMPLETED
            error = None

        except asyncio.TimeoutError:
            prediction_status = PredictionStatus.TIMED_OUT
            outputs = None
            error = "Batch inference timeout exceeded"

        except Exception:
            logger.exception(
                "Batch inference failed: %s",
                model.name,
            )

            prediction_status = PredictionStatus.FAILED
            outputs = None
            error = "Batch inference failed"

        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        if prediction_status == PredictionStatus.COMPLETED:
            model.metrics.successful_requests += batch_size
            model.metrics.total_latency_ms += latency_ms
        else:
            model.metrics.failed_requests += batch_size

        return PredictionResponse(
            request_id=request_id,
            model_name=model.name,
            model_version=model.version,
            status=prediction_status,
            output=outputs,
            error=error,
            device=model.device,
            latency_ms=round(latency_ms, 3),
            batch_size=batch_size,
            simulation_only=request.simulation_only,
            physical_execution_authorized=False,
            completed_at=datetime.now(timezone.utc),
        )

    async def shutdown(self) -> None:
        self._closed = True


inference_engine = ModelInferenceEngine(
    registry=model_registry,
    config=settings,
)


# ============================================================
# Example Inference Backend
# ============================================================

class DemoPerceptionBackend:
    """
    Demonstration backend.

    Replace with an actual trained model backend.
    """

    async def predict(self, inputs: Any) -> Any:
        return {
            "detections": [],
            "input_received": inputs is not None,
            "model_type": "demo_perception",
            "simulation_only": True,
        }

    async def predict_batch(
        self,
        inputs: Sequence[Any],
    ) -> Sequence[Any]:
        return [
            await self.predict(item)
            for item in inputs
        ]


def register_demo_models() -> None:
    """
    Explicitly register demo models when needed.

    Demo models are not registered automatically
    in production.
    """
    try:
        model_registry.register(
            RegisteredModel(
                name="demo-perception",
                version="1.0.0",
                category=ModelCategory.PERCEPTION,
                backend=DemoPerceptionBackend(),
                device=ModelDevice.CPU,
                max_batch_size=32,
            )
        )

    except ValueError:
        pass


# ============================================================
# Application Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key_store.load_from_environment()

    if os.getenv(
        "MODEL_SERVER_ENABLE_DEMO",
        "false",
    ).lower() == "true":
        register_demo_models()

    logger.info("Model server started")

    yield

    await inference_engine.shutdown()

    logger.info("Model server stopped")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Secure ML inference service for the "
        "Physical AI & Autonomous Robotics platform."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================
# Health Endpoints
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="model-server",
        timestamp=datetime.now(timezone.utc),
    )


@app.get(
    "/ready",
    tags=["Health"],
)
async def readiness_check():
    if inference_engine._closed:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "reason": "Service shutting down",
            },
        )

    ready_models = [
        model
        for model in model_registry.list_models()
        if model.state == ModelState.READY
    ]

    return {
        "status": (
            "ready"
            if ready_models
            else "no_models_loaded"
        ),
        "ready_model_count": len(ready_models),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ============================================================
# Inference Endpoints
# ============================================================

@app.post(
    "/api/v1/inference/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
)
async def predict(
    request: PredictRequest,
    principal: APIPrincipal = Depends(get_principal),
):
    return await inference_engine.predict(
        request,
        principal,
    )


@app.post(
    "/api/v1/inference/batch",
    response_model=PredictionResponse,
    tags=["Inference"],
)
async def predict_batch(
    request: BatchPredictRequest,
    principal: APIPrincipal = Depends(get_principal),
):
    return await inference_engine.predict_batch(
        request,
        principal,
    )


# ============================================================
# Model Management Endpoints
# ============================================================

@app.get(
    "/api/v1/models",
    tags=["Models"],
)
async def list_models(
    principal: APIPrincipal = Depends(get_principal),
):
    models = []

    for model in model_registry.list_models():
        if not principal.can_access_model(model.name):
            continue

        if (
            model.tenant_id is not None
            and model.tenant_id != principal.tenant_id
        ):
            continue

        models.append(
            {
                "name": model.name,
                "version": model.version,
                "category": model.category.value,
                "device": model.device.value,
                "state": model.state.value,
                "max_batch_size": model.max_batch_size,
            }
        )

    return {
        "count": len(models),
        "models": models,
    }


@app.get(
    "/api/v1/models/{model_name}",
    response_model=ModelInfoResponse,
    tags=["Models"],
)
async def get_model(
    model_name: str,
    version: str | None = None,
    principal: APIPrincipal = Depends(get_principal),
):
    model = inference_engine._get_model(
        model_name,
        version,
        principal,
    )

    return ModelInfoResponse(
        name=model.name,
        version=model.version,
        category=model.category,
        device=model.device,
        state=model.state,
        max_batch_size=model.max_batch_size,
        total_requests=model.metrics.total_requests,
        successful_requests=(
            model.metrics.successful_requests
        ),
        failed_requests=model.metrics.failed_requests,
        average_latency_ms=(
            model.metrics.average_latency_ms
        ),
    )


class ModelStateUpdate(BaseModel):
    state: ModelState


@app.patch(
    "/api/v1/models/{model_name}/{version}/state",
    tags=["Models"],
)
async def update_model_state(
    model_name: str,
    version: str,
    payload: ModelStateUpdate,
    principal: APIPrincipal = Depends(require_admin),
):
    try:
        model = model_registry.get(
            model_name,
            version,
        )

        check_model_access(model, principal)

        model_registry.set_state(
            model_name,
            version,
            payload.state,
        )

    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Model not found",
        ) from None

    return {
        "model_name": model_name,
        "version": version,
        "state": payload.state.value,
        "updated": True,
    }


# ============================================================
# Server Statistics
# ============================================================

@app.get(
    "/api/v1/statistics",
    tags=["Monitoring"],
)
async def get_statistics(
    principal: APIPrincipal = Depends(require_admin),
):
    models = [
        model
        for model in model_registry.list_models()
        if (
            principal.can_access_model(model.name)
            and (
                model.tenant_id is None
                or model.tenant_id == principal.tenant_id
            )
        )
    ]

    return {
        "model_count": len(models),
        "total_requests": sum(
            model.metrics.total_requests
            for model in models
        ),
        "successful_requests": sum(
            model.metrics.successful_requests
            for model in models
        ),
        "failed_requests": sum(
            model.metrics.failed_requests
            for model in models
        ),
        "models": [
            {
                "name": model.name,
                "version": model.version,
                "requests": model.metrics.total_requests,
                "success_rate": (
                    model.metrics.successful_requests
                    / model.metrics.total_requests
                    if model.metrics.total_requests
                    else 0.0
                ),
                "average_latency_ms": (
                    model.metrics.average_latency_ms
                ),
            }
            for model in models
        ],
    }


# ============================================================
# Prometheus Metrics
# ============================================================

@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    tags=["Monitoring"],
)
async def prometheus_metrics(
    principal: APIPrincipal = Depends(require_admin),
):
    lines = [
        "# HELP model_server_requests_total Total inference requests",
        "# TYPE model_server_requests_total counter",
    ]

    for model in model_registry.list_models():
        if not principal.can_access_model(model.name):
            continue

        if (
            model.tenant_id is not None
            and model.tenant_id != principal.tenant_id
        ):
            continue

        labels = (
            f'model="{model.name}",'
            f'version="{model.version}"'
        )

        lines.append(
            f"model_server_requests_total"
            f"{{{labels}}} "
            f"{model.metrics.total_requests}"
        )

    lines.extend(
        [
            "# HELP model_server_success_total Successful predictions",
            "# TYPE model_server_success_total counter",
        ]
    )

    for model in model_registry.list_models():
        if not principal.can_access_model(model.name):
            continue

        if (
            model.tenant_id is not None
            and model.tenant_id != principal.tenant_id
        ):
            continue

        labels = (
            f'model="{model.name}",'
            f'version="{model.version}"'
        )

        lines.append(
            f"model_server_success_total"
            f"{{{labels}}} "
            f"{model.metrics.successful_requests}"
        )

    return "\n".join(lines) + "\n"


# ============================================================
# Exception Handlers
# ============================================================

@app.exception_handler(ValueError)
async def validation_exception_handler(
    request: Request,
    exc: ValueError,
):
    logger.warning(
        "Validation failure on %s",
        request.url.path,
    )

    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request or model output",
        },
    )


# ============================================================
# Server Entry Point
# ============================================================

def run_server(
    host: str = "127.0.0.1",
    port: int = 8001,
) -> None:
    import uvicorn

    uvicorn.run(
        "ml.inference.model_server:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    run_server()
"""Model Service module.

TODO: implement and test; no real robot control is enabled.
"""
"""Navigation Service module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/services/model_service.py

AI/ML Model Registry and Inference Service

Features:
    - AI/ML model registration
    - Model version management
    - Model metadata and artifacts
    - Model lifecycle management
    - Model validation and approval
    - Model deployment simulation
    - Active model routing
    - Model inference
    - Custom inference handlers
    - Model performance monitoring
    - Latency tracking
    - Model rollback
    - Model comparison
    - Model registry persistence
    - JSON report export

Supported model categories:
    - Computer vision
    - Object detection
    - Image segmentation
    - Predictive maintenance
    - Anomaly detection
    - Remaining useful life prediction
    - Navigation
    - Manipulation
    - Vision Language Action (VLA)
    - Reinforcement learning
    - Multimodal AI
    - LLM / Agentic AI

Python 3.11+
Standard library only.

IMPORTANT:
This service provides model orchestration and
registered inference-handler execution.

Actual PyTorch, ONNX, TensorFlow and Hugging Face
inference requires separately implemented adapters.

No physical robot commands are issued.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class ModelServiceError(Exception):
    pass


class ModelValidationError(ModelServiceError):
    pass


class ModelNotFoundError(ModelServiceError):
    pass


class ModelStateError(ModelServiceError):
    pass


class ModelInferenceError(ModelServiceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ModelCategory(str, Enum):
    COMPUTER_VISION = "computer_vision"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    ANOMALY_DETECTION = "anomaly_detection"
    RUL_PREDICTION = "rul_prediction"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    VLA = "vision_language_action"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    MULTIMODAL = "multimodal"
    LLM = "llm"
    AGENT = "agent"
    OTHER = "other"


class ModelFramework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    HUGGING_FACE = "hugging_face"
    SCIKIT_LEARN = "scikit_learn"
    CUSTOM = "custom"


class ModelStatus(str, Enum):
    REGISTERED = "registered"
    VALIDATED = "validated"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"


class InferenceStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def require_text(
    value: str,
    name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError(
            f"{name} is required"
        )

    return value.strip()


def finite_number(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ModelValidationError(
            f"{name} must be finite"
        )

    return float(value)


def calculate_sha256(
    path: str | Path,
) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while chunk := handle.read(
            1024 * 1024
        ):
            digest.update(chunk)

    return digest.hexdigest()


def json_safe(
    value: Any,
) -> Any:
    """
    Convert JSON-compatible objects without
    silently serializing arbitrary Python objects.
    """
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
        )
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ModelServiceConfig:
    registry_directory: str = (
        "data/models/registry"
    )

    report_directory: str = (
        "data/models/reports"
    )

    maximum_models: int = 10000

    maximum_inference_history: int = 10000

    maximum_latency_ms: float = 5000.0

    minimum_validation_score: float = 0.75

    require_approval_before_deployment: bool = True

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        if self.maximum_models < 1:
            raise ModelValidationError(
                "maximum_models must be positive"
            )

        if self.maximum_inference_history < 1:
            raise ModelValidationError(
                "maximum_inference_history must be positive"
            )

        if self.maximum_latency_ms <= 0:
            raise ModelValidationError(
                "maximum_latency_ms must be positive"
            )

        if not (
            0.0 <= self.minimum_validation_score <= 1.0
        ):
            raise ModelValidationError(
                "Invalid minimum validation score"
            )

        if self.allow_hardware_control:
            raise ModelValidationError(
                "Hardware control is disabled"
            )


# ============================================================
# MODEL METADATA
# ============================================================

@dataclass
class RegisteredModel:
    model_id: str
    name: str
    version: str

    category: ModelCategory
    framework: ModelFramework

    artifact_path: str | None = None
    artifact_sha256: str | None = None

    status: ModelStatus = (
        ModelStatus.REGISTERED
    )

    validation_score: float | None = None

    validation_metrics: dict[str, float] = field(
        default_factory=dict
    )

    description: str = ""

    created_at: str = field(
        default_factory=utc_now
    )

    updated_at: str = field(
        default_factory=utc_now
    )

    approved_by: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "category": self.category.value,
            "framework": self.framework.value,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "status": self.status.value,
            "validation_score": (
                self.validation_score
            ),
            "validation_metrics": (
                self.validation_metrics
            ),
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "approved_by": self.approved_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> RegisteredModel:
        return cls(
            model_id=str(data["model_id"]),
            name=str(data["name"]),
            version=str(data["version"]),
            category=ModelCategory(
                data["category"]
            ),
            framework=ModelFramework(
                data["framework"]
            ),
            artifact_path=data.get(
                "artifact_path"
            ),
            artifact_sha256=data.get(
                "artifact_sha256"
            ),
            status=ModelStatus(
                data.get(
                    "status",
                    "registered",
                )
            ),
            validation_score=data.get(
                "validation_score"
            ),
            validation_metrics=dict(
                data.get(
                    "validation_metrics",
                    {},
                )
            ),
            description=str(
                data.get(
                    "description",
                    "",
                )
            ),
            created_at=str(
                data.get(
                    "created_at",
                    utc_now(),
                )
            ),
            updated_at=str(
                data.get(
                    "updated_at",
                    utc_now(),
                )
            ),
            approved_by=data.get(
                "approved_by"
            ),
            metadata=dict(
                data.get(
                    "metadata",
                    {},
                )
            ),
        )


# ============================================================
# INFERENCE RESULT
# ============================================================

@dataclass
class ModelInferenceResult:
    inference_id: str
    model_id: str

    model_name: str
    model_version: str

    status: InferenceStatus

    output: Any

    latency_ms: float

    timestamp: str

    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "inference_id": self.inference_id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status": self.status.value,
            "output": self.output,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "error": self.error,
        }


# ============================================================
# MODEL REGISTRY
# ============================================================

class RoboticsModelRegistry:

    def __init__(
        self,
        config: ModelServiceConfig,
    ) -> None:
        self.config = config

        self.models: dict[
            str,
            RegisteredModel
        ] = {}

        self.active_models: dict[
            ModelCategory,
            str
        ] = {}

        self._lock = threading.RLock()

        self.registry_root = Path(
            config.registry_directory
        )

        self.registry_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def register(
        self,
        *,
        name: str,
        version: str,
        category: ModelCategory,
        framework: ModelFramework,
        artifact_path: str | None = None,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> RegisteredModel:
        name = require_text(name, "name")
        version = require_text(
            version,
            "version",
        )

        with self._lock:
            if len(self.models) >= self.config.maximum_models:
                raise ModelStateError(
                    "Model registry capacity reached"
                )

            if any(
                model.name == name
                and model.version == version
                for model in self.models.values()
            ):
                raise ModelValidationError(
                    "Model name and version already exist"
                )

            artifact_sha256 = None

            if artifact_path is not None:
                path = Path(artifact_path)

                if not path.is_file():
                    raise ModelValidationError(
                        "Model artifact not found"
                    )

                artifact_path = str(
                    path.resolve()
                )

                artifact_sha256 = (
                    calculate_sha256(path)
                )

            model = RegisteredModel(
                model_id=str(uuid.uuid4()),
                name=name,
                version=version,
                category=ModelCategory(category),
                framework=ModelFramework(framework),
                artifact_path=artifact_path,
                artifact_sha256=artifact_sha256,
                description=description,
                metadata=json_safe(
                    dict(metadata or {})
                ),
            )

            self.models[
                model.model_id
            ] = model

            return model

    def get(
        self,
        model_id: str,
    ) -> RegisteredModel:
        model = self.models.get(
            model_id
        )

        if model is None:
            raise ModelNotFoundError(
                f"Model not found: {model_id}"
            )

        return model

    def list_models(
        self,
        *,
        category: ModelCategory | None = None,
        status: ModelStatus | None = None,
        name: str | None = None,
    ) -> list[RegisteredModel]:
        models = list(
            self.models.values()
        )

        if category is not None:
            models = [
                model
                for model in models
                if model.category == category
            ]

        if status is not None:
            models = [
                model
                for model in models
                if model.status == status
            ]

        if name is not None:
            models = [
                model
                for model in models
                if model.name == name
            ]

        return sorted(
            models,
            key=lambda model: (
                model.name,
                model.version,
            ),
        )

    def validate_model(
        self,
        model_id: str,
        *,
        validation_score: float,
        metrics: Mapping[str, float] | None = None,
    ) -> RegisteredModel:
        score = finite_number(
            validation_score,
            "validation_score",
        )

        if not 0 <= score <= 1:
            raise ModelValidationError(
                "Validation score must be between 0 and 1"
            )

        with self._lock:
            model = self.get(model_id)

            if model.status not in (
                ModelStatus.REGISTERED,
                ModelStatus.VALIDATED,
            ):
                raise ModelStateError(
                    "Model cannot be validated in current state"
                )

            model.validation_score = score

            model.validation_metrics = {
                key: finite_number(value, key)
                for key, value in dict(
                    metrics or {}
                ).items()
            }

            model.status = ModelStatus.VALIDATED
            model.updated_at = utc_now()

            return model

    def approve(
        self,
        model_id: str,
        *,
        approved_by: str,
    ) -> RegisteredModel:
        require_text(
            approved_by,
            "approved_by",
        )

        with self._lock:
            model = self.get(model_id)

            if model.status != ModelStatus.VALIDATED:
                raise ModelStateError(
                    "Model must be validated first"
                )

            if (
                model.validation_score is None
                or model.validation_score
                < self.config.minimum_validation_score
            ):
                raise ModelStateError(
                    "Validation score below deployment threshold"
                )

            model.status = ModelStatus.APPROVED
            model.approved_by = approved_by
            model.updated_at = utc_now()

            return model

    def deploy(
        self,
        model_id: str,
    ) -> RegisteredModel:
        with self._lock:
            model = self.get(model_id)

            if self.config.require_approval_before_deployment:
                if model.status != ModelStatus.APPROVED:
                    raise ModelStateError(
                        "Model approval required"
                    )

            elif model.status not in (
                ModelStatus.VALIDATED,
                ModelStatus.APPROVED,
            ):
                raise ModelStateError(
                    "Model must be validated before deployment"
                )

            previous_id = self.active_models.get(
                model.category
            )

            if previous_id is not None:
                previous = self.get(
                    previous_id
                )

                if previous.status == ModelStatus.DEPLOYED:
                    previous.status = ModelStatus.APPROVED
                    previous.updated_at = utc_now()

            model.status = ModelStatus.DEPLOYED
            model.updated_at = utc_now()

            self.active_models[
                model.category
            ] = model_id

            return model

    def active(
        self,
        category: ModelCategory,
    ) -> RegisteredModel:
        category = ModelCategory(
            category
        )

        model_id = self.active_models.get(
            category
        )

        if model_id is None:
            raise ModelNotFoundError(
                f"No active model for {category.value}"
            )

        return self.get(model_id)

    def rollback(
        self,
        category: ModelCategory,
        previous_model_id: str,
    ) -> RegisteredModel:
        """
        Rollback requires an explicitly selected,
        previously approved model.
        """
        with self._lock:
            previous = self.get(
                previous_model_id
            )

            if previous.category != category:
                raise ModelValidationError(
                    "Rollback model category mismatch"
                )

            if previous.status != ModelStatus.APPROVED:
                raise ModelStateError(
                    "Rollback target must be approved"
                )

            return self.deploy(
                previous_model_id
            )

    def archive(
        self,
        model_id: str,
    ) -> RegisteredModel:
        with self._lock:
            model = self.get(
                model_id
            )

            if model.status == ModelStatus.DEPLOYED:
                raise ModelStateError(
                    "Cannot archive an active model"
                )

            model.status = ModelStatus.ARCHIVED
            model.updated_at = utc_now()

            return model

    def save(
        self,
        filename: str = "model_registry.json",
    ) -> Path:
        with self._lock:
            output = (
                self.registry_root
                / filename
            )

            payload = {
                "generated_at": utc_now(),
                "models": [
                    model.to_dict()
                    for model in self.models.values()
                ],
                "active_models": {
                    category.value: model_id
                    for category, model_id
                    in self.active_models.items()
                },
            }

            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary = output.with_suffix(
                output.suffix + ".tmp"
            )

            temporary.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

            temporary.replace(output)

            return output

    def load(
        self,
        filename: str = "model_registry.json",
    ) -> None:
        path = self.registry_root / filename

        if not path.is_file():
            raise ModelNotFoundError(
                "Registry file not found"
            )

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        models = {
            model.model_id: model
            for model in (
                RegisteredModel.from_dict(data)
                for data in payload["models"]
            )
        }

        active_models = {
            ModelCategory(category): model_id
            for category, model_id
            in payload.get(
                "active_models",
                {},
            ).items()
        }

        for category, model_id in active_models.items():
            model = models.get(model_id)

            if (
                model is None
                or model.category != category
                or model.status != ModelStatus.DEPLOYED
            ):
                raise ModelValidationError(
                    "Invalid active model registry state"
                )

        with self._lock:
            self.models = models
            self.active_models = active_models


# ============================================================
# INFERENCE HANDLER REGISTRY
# ============================================================

InferenceHandler = Callable[
    [Any],
    Any,
]


class RoboticsInferenceEngine:

    def __init__(
        self,
        registry: RoboticsModelRegistry,
        config: ModelServiceConfig,
    ) -> None:
        self.registry = registry
        self.config = config

        self.handlers: dict[
            str,
            InferenceHandler
        ] = {}

        self.history: deque[
            ModelInferenceResult
        ] = deque(
            maxlen=config.maximum_inference_history
        )

        self._lock = threading.RLock()

    def register_handler(
        self,
        model_id: str,
        handler: InferenceHandler,
    ) -> None:
        self.registry.get(model_id)

        if not callable(handler):
            raise ModelValidationError(
                "Inference handler must be callable"
            )

        with self._lock:
            self.handlers[
                model_id
            ] = handler

    def infer(
        self,
        *,
        category: ModelCategory,
        inputs: Any,
    ) -> ModelInferenceResult:
        model = self.registry.active(
            category
        )

        with self._lock:
            handler = self.handlers.get(
                model.model_id
            )

        if handler is None:
            raise ModelInferenceError(
                "No inference handler registered "
                f"for model {model.model_id}"
            )

        started = time.perf_counter()

        try:
            output = handler(inputs)

            output = json_safe(output)

            status = InferenceStatus.SUCCESS
            error = None

        except Exception as exc:
            output = None
            status = InferenceStatus.FAILED
            error = (
                f"{type(exc).__name__}: {exc}"
            )

        latency_ms = (
            time.perf_counter() - started
        ) * 1000.0

        result = ModelInferenceResult(
            inference_id=str(uuid.uuid4()),
            model_id=model.model_id,
            model_name=model.name,
            model_version=model.version,
            status=status,
            output=output,
            latency_ms=latency_ms,
            timestamp=utc_now(),
            error=error,
        )

        with self._lock:
            self.history.append(
                result
            )

        return result

    def statistics(
        self,
        *,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            records = [
                record
                for record in self.history
                if (
                    model_id is None
                    or record.model_id == model_id
                )
            ]

        if not records:
            return {
                "total_inferences": 0,
                "successful_inferences": 0,
                "failed_inferences": 0,
                "average_latency_ms": None,
                "maximum_latency_ms": None,
                "latency_threshold_exceeded": 0,
            }

        latencies = [
            record.latency_ms
            for record in records
        ]

        successes = sum(
            record.status == InferenceStatus.SUCCESS
            for record in records
        )

        return {
            "total_inferences": len(records),
            "successful_inferences": successes,
            "failed_inferences": (
                len(records) - successes
            ),
            "success_rate": (
                successes / len(records)
            ),
            "average_latency_ms": (
                sum(latencies) / len(latencies)
            ),
            "maximum_latency_ms": max(
                latencies
            ),
            "latency_threshold_exceeded": sum(
                latency > self.config.maximum_latency_ms
                for latency in latencies
            ),
        }


# ============================================================
# MODEL COMPARISON
# ============================================================

class RoboticsModelComparator:

    @staticmethod
    def compare(
        models: list[RegisteredModel],
        *,
        metric: str,
        higher_is_better: bool = True,
    ) -> list[dict[str, Any]]:
        results = []

        for model in models:
            score = model.validation_metrics.get(
                metric
            )

            if score is None:
                continue

            results.append({
                "model_id": model.model_id,
                "name": model.name,
                "version": model.version,
                "metric": metric,
                "score": score,
            })

        results.sort(
            key=lambda item: item["score"],
            reverse=higher_is_better,
        )

        return results


# ============================================================
# MODEL SERVICE
# ============================================================

class RoboticsModelService:

    def __init__(
        self,
        config: ModelServiceConfig | None = None,
    ) -> None:
        self.config = (
            config or ModelServiceConfig()
        )

        self.registry = RoboticsModelRegistry(
            self.config
        )

        self.inference_engine = (
            RoboticsInferenceEngine(
                self.registry,
                self.config,
            )
        )

    def register_model(
        self,
        **kwargs: Any,
    ) -> RegisteredModel:
        return self.registry.register(
            **kwargs
        )

    def validate_model(
        self,
        model_id: str,
        *,
        validation_score: float,
        metrics: Mapping[str, float] | None = None,
    ) -> RegisteredModel:
        return self.registry.validate_model(
            model_id,
            validation_score=validation_score,
            metrics=metrics,
        )

    def approve_model(
        self,
        model_id: str,
        *,
        approved_by: str,
    ) -> RegisteredModel:
        return self.registry.approve(
            model_id,
            approved_by=approved_by,
        )

    def deploy_model(
        self,
        model_id: str,
    ) -> RegisteredModel:
        return self.registry.deploy(
            model_id
        )

    def register_inference_handler(
        self,
        model_id: str,
        handler: InferenceHandler,
    ) -> None:
        self.inference_engine.register_handler(
            model_id,
            handler,
        )

    def predict(
        self,
        *,
        category: ModelCategory,
        inputs: Any,
    ) -> ModelInferenceResult:
        return self.inference_engine.infer(
            category=category,
            inputs=inputs,
        )

    def rollback_model(
        self,
        category: ModelCategory,
        previous_model_id: str,
    ) -> RegisteredModel:
        return self.registry.rollback(
            category,
            previous_model_id,
        )

    def statistics(
        self,
    ) -> dict[str, Any]:
        models = list(
            self.registry.models.values()
        )

        return {
            "total_models": len(models),
            "active_models": {
                category.value: model_id
                for category, model_id
                in self.registry.active_models.items()
            },
            "status_distribution": dict(
                Counter(
                    model.status.value
                    for model in models
                )
            ),
            "category_distribution": dict(
                Counter(
                    model.category.value
                    for model in models
                )
            ),
            "inference": (
                self.inference_engine.statistics()
            ),
            "simulation_only": True,
        }

    def export_report(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "generated_at": utc_now(),
            "statistics": self.statistics(),
            "models": [
                model.to_dict()
                for model in self.registry.models.values()
            ],
            "inference_history": [
                record.to_dict()
                for record
                in self.inference_engine.history
            ],
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
# EXAMPLE INFERENCE HANDLER
# ============================================================

def example_predictive_maintenance_handler(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Demonstration rule-based handler.

    Replace with an actual trained ML model
    for real predictive maintenance.
    """
    temperature = finite_number(
        inputs.get(
            "motor_temperature_c",
            25.0,
        ),
        "motor_temperature_c",
    )

    vibration = finite_number(
        inputs.get(
            "vibration_mm_s",
            0.0,
        ),
        "vibration_mm_s",
    )

    risk_score = min(
        1.0,
        max(
            0.0,
            (
                max(
                    temperature - 50.0,
                    0.0,
                ) / 60.0
                + vibration / 20.0
            ) / 2.0,
        ),
    )

    return {
        "failure_risk_score": risk_score,
        "risk_level": (
            "critical"
            if risk_score >= 0.8
            else "high"
            if risk_score >= 0.6
            else "medium"
            if risk_score >= 0.3
            else "low"
        ),
        "model_type": "rule_based_demo",
    }


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsModelService()

    model = service.register_model(
        name="robot_predictive_maintenance",
        version="1.0.0",
        category=(
            ModelCategory.PREDICTIVE_MAINTENANCE
        ),
        framework=ModelFramework.CUSTOM,
        description=(
            "Demonstration robot maintenance "
            "risk estimation model"
        ),
        metadata={
            "simulation_only": True,
            "training_dataset": (
                "synthetic_robot_telemetry"
            ),
        },
    )

    service.validate_model(
        model.model_id,
        validation_score=0.91,
        metrics={
            "accuracy": 0.92,
            "precision": 0.90,
            "recall": 0.88,
            "f1": 0.89,
        },
    )

    service.approve_model(
        model.model_id,
        approved_by="simulation_admin",
    )

    service.deploy_model(
        model.model_id
    )

    service.register_inference_handler(
        model.model_id,
        example_predictive_maintenance_handler,
    )

    result = service.predict(
        category=(
            ModelCategory.PREDICTIVE_MAINTENANCE
        ),
        inputs={
            "motor_temperature_c": 82.0,
            "vibration_mm_s": 9.0,
        },
    )

    service.registry.save()

    service.export_report(
        "data/models/reports/model_report.json"
    )

    print(
        json.dumps(
            {
                "model": model.to_dict(),
                "inference": result.to_dict(),
                "statistics": service.statistics(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

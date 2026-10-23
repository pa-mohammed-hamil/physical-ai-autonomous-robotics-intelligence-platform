"""
Physical AI & Autonomous Robotics Intelligence Platform
File: ml/registry/model_metadata.py

Model metadata management for ML/MLOps.

Features:
- Model identity and semantic versioning
- Model architecture and framework metadata
- Training dataset lineage
- Evaluation metrics and benchmark results
- Input/output schema definitions
- CPU/GPU deployment requirements
- Model artifact checksums
- Model provenance and reproducibility
- Approval and deployment lifecycle
- Model safety classification
- Simulation-only robotic action models
- Tenant-aware model metadata
- JSON serialization and persistence
- Metadata validation
- Audit history
- Model comparison and reporting

Python 3.11+
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol


# ============================================================
# Exceptions
# ============================================================

class ModelMetadataError(Exception):
    """Base model metadata exception."""


class MetadataValidationError(ModelMetadataError):
    """Invalid model metadata."""


class ModelMetadataNotFoundError(ModelMetadataError):
    """Requested model metadata does not exist."""


class ModelMetadataConflictError(ModelMetadataError):
    """Metadata already exists or has a version conflict."""


class MetadataPermissionError(ModelMetadataError):
    """Metadata access denied."""


class ModelApprovalError(ModelMetadataError):
    """Model approval requirements were not satisfied."""


class ModelSafetyError(ModelMetadataError):
    """Model metadata violates the safety policy."""


class MetadataStorageError(ModelMetadataError):
    """Metadata persistence failed."""


# ============================================================
# Constants
# ============================================================

SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?"
    r"(?:\+[0-9A-Za-z.-]+)?$"
)

IDENTIFIER_PATTERN = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$"
)

MAX_METADATA_SIZE_BYTES = 2 * 1024 * 1024

ROBOT_ACTION_CATEGORIES = {
    "vla",
    "navigation",
    "manipulation",
}


# ============================================================
# Utilities
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def validate_identifier(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not IDENTIFIER_PATTERN.fullmatch(value)
    ):
        raise MetadataValidationError(
            f"Invalid {name}: use letters, numbers, "
            "underscores, hyphens or periods"
        )

    return value


def validate_semver(version: str) -> str:
    if not SEMVER_PATTERN.fullmatch(version):
        raise MetadataValidationError(
            f"Invalid semantic version: {version}"
        )

    return version


def validate_metric(
    name: str,
    value: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise MetadataValidationError(
            f"Metric {name} must be finite and numeric"
        )

    return float(value)


def calculate_sha256(
    file_path: str | Path,
) -> str:
    digest = hashlib.sha256()

    with Path(file_path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def parse_datetime(
    value: str | datetime,
) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

    if result.tzinfo is None:
        raise MetadataValidationError(
            "Timestamps must include timezone information"
        )

    return result.astimezone(timezone.utc)


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


class ModelFramework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    HUGGING_FACE = "hugging_face"
    CUSTOM = "custom"


class ModelLifecycleStage(str, Enum):
    DEVELOPMENT = "development"
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVOKED = "revoked"


class SafetyClassification(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ComputeDevice(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    REMOTE = "remote"


class ModelPrecision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    MIXED = "mixed"


# ============================================================
# Identity and Ownership
# ============================================================

@dataclass(slots=True)
class ModelIdentity:
    model_id: str
    name: str
    version: str
    category: ModelCategory

    description: str = ""
    architecture: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.model_id,
            "model_id",
        )

        validate_identifier(
            self.name,
            "model name",
        )

        validate_semver(self.version)

        self.created_at = parse_datetime(
            self.created_at
        )


@dataclass(slots=True)
class ModelOwnership:
    tenant_id: str
    owner_id: str

    team: str | None = None
    organization: str | None = None

    visibility: str = "private"

    def __post_init__(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.owner_id,
            "owner_id",
        )

        if self.visibility not in {
            "private",
            "tenant",
        }:
            raise MetadataValidationError(
                "Unsupported model visibility"
            )


# ============================================================
# Dataset Lineage
# ============================================================

@dataclass(slots=True)
class DatasetReference:
    dataset_id: str
    version: str

    name: str = ""
    uri: str | None = None

    sha256: str | None = None

    sample_count: int | None = None

    split: str | None = None

    license: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(
            self.dataset_id,
            "dataset_id",
        )

        if not self.version:
            raise MetadataValidationError(
                "Dataset version is required"
            )

        if (
            self.sample_count is not None
            and self.sample_count < 0
        ):
            raise MetadataValidationError(
                "Dataset sample count cannot be negative"
            )

        if self.sha256 is not None:
            if not re.fullmatch(
                r"[0-9a-fA-F]{64}",
                self.sha256,
            ):
                raise MetadataValidationError(
                    "Invalid dataset SHA-256 checksum"
                )


@dataclass(slots=True)
class TrainingMetadata:
    training_run_id: str | None = None

    datasets: list[DatasetReference] = field(
        default_factory=list
    )

    random_seed: int | None = None

    epochs: int | None = None
    batch_size: int | None = None

    learning_rate: float | None = None

    optimizer: str | None = None
    scheduler: str | None = None

    training_duration_seconds: float | None = None

    source_commit: str | None = None

    experiment_id: str | None = None

    hyperparameters: dict[str, Any] = field(
        default_factory=dict
    )

    environment: dict[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.epochs is not None and self.epochs < 1:
            raise MetadataValidationError(
                "Epochs must be positive"
            )

        if (
            self.batch_size is not None
            and self.batch_size < 1
        ):
            raise MetadataValidationError(
                "Batch size must be positive"
            )

        if self.learning_rate is not None:
            if (
                not math.isfinite(self.learning_rate)
                or self.learning_rate <= 0
            ):
                raise MetadataValidationError(
                    "Learning rate must be positive"
                )


# ============================================================
# Evaluation Metadata
# ============================================================

@dataclass(slots=True)
class BenchmarkResult:
    benchmark_name: str
    benchmark_version: str

    metrics: dict[str, float] = field(
        default_factory=dict
    )

    dataset_id: str | None = None

    evaluated_at: datetime = field(
        default_factory=utc_now
    )

    notes: str = ""

    def __post_init__(self) -> None:
        if not self.benchmark_name.strip():
            raise MetadataValidationError(
                "Benchmark name is required"
            )

        self.metrics = {
            name: validate_metric(name, value)
            for name, value in self.metrics.items()
        }

        self.evaluated_at = parse_datetime(
            self.evaluated_at
        )


@dataclass(slots=True)
class EvaluationMetadata:
    validation_metrics: dict[str, float] = field(
        default_factory=dict
    )

    test_metrics: dict[str, float] = field(
        default_factory=dict
    )

    benchmarks: list[BenchmarkResult] = field(
        default_factory=list
    )

    evaluation_dataset_id: str | None = None

    evaluated_at: datetime | None = None

    evaluator_id: str | None = None

    passed_quality_gate: bool = False

    quality_gate_notes: str = ""

    def __post_init__(self) -> None:
        self.validation_metrics = {
            name: validate_metric(name, value)
            for name, value in self.validation_metrics.items()
        }

        self.test_metrics = {
            name: validate_metric(name, value)
            for name, value in self.test_metrics.items()
        }

        if self.evaluated_at is not None:
            self.evaluated_at = parse_datetime(
                self.evaluated_at
            )


# ============================================================
# Model Input and Output Schemas
# ============================================================

@dataclass(slots=True)
class TensorSchema:
    name: str
    dtype: str

    shape: list[int | None]

    description: str = ""

    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise MetadataValidationError(
                "Tensor name is required"
            )

        if not self.dtype:
            raise MetadataValidationError(
                "Tensor dtype is required"
            )

        for dimension in self.shape:
            if (
                dimension is not None
                and (
                    isinstance(dimension, bool)
                    or not isinstance(dimension, int)
                    or dimension < 1
                )
            ):
                raise MetadataValidationError(
                    "Tensor dimensions must be positive "
                    "integers or null"
                )


@dataclass(slots=True)
class ModelInterface:
    inputs: list[TensorSchema] = field(
        default_factory=list
    )

    outputs: list[TensorSchema] = field(
        default_factory=list
    )

    preprocessing_version: str | None = None

    postprocessing_version: str | None = None

    tokenizer_id: str | None = None

    max_sequence_length: int | None = None

    supported_modalities: list[str] = field(
        default_factory=list
    )


# ============================================================
# Compute Requirements
# ============================================================

@dataclass(slots=True)
class ComputeRequirements:
    supported_devices: list[ComputeDevice] = field(
        default_factory=lambda: [ComputeDevice.CPU]
    )

    preferred_device: ComputeDevice = (
        ComputeDevice.CPU
    )

    precision: ModelPrecision = (
        ModelPrecision.FP32
    )

    minimum_ram_mb: int = 0
    minimum_vram_mb: int = 0

    estimated_model_memory_mb: int = 0

    max_batch_size: int = 1

    recommended_concurrency: int = 1

    supports_quantization: bool = False

    supports_dynamic_batching: bool = False

    def __post_init__(self) -> None:
        if self.preferred_device not in self.supported_devices:
            raise MetadataValidationError(
                "Preferred device must be supported"
            )

        for name in (
            "minimum_ram_mb",
            "minimum_vram_mb",
            "estimated_model_memory_mb",
        ):
            if getattr(self, name) < 0:
                raise MetadataValidationError(
                    f"{name} cannot be negative"
                )

        if self.max_batch_size < 1:
            raise MetadataValidationError(
                "Maximum batch size must be positive"
            )

        if self.recommended_concurrency < 1:
            raise MetadataValidationError(
                "Recommended concurrency must be positive"
            )


# ============================================================
# Model Artifacts
# ============================================================

@dataclass(slots=True)
class ModelArtifact:
    artifact_id: str
    artifact_type: str

    uri: str

    sha256: str

    size_bytes: int

    format: str

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.artifact_id,
            "artifact_id",
        )

        if not re.fullmatch(
            r"[0-9a-fA-F]{64}",
            self.sha256,
        ):
            raise MetadataValidationError(
                "Invalid model artifact SHA-256"
            )

        if self.size_bytes < 0:
            raise MetadataValidationError(
                "Artifact size cannot be negative"
            )

        self.created_at = parse_datetime(
            self.created_at
        )


# ============================================================
# Safety Metadata
# ============================================================

@dataclass(slots=True)
class ModelSafetyMetadata:
    classification: SafetyClassification = (
        SafetyClassification.MODERATE
    )

    simulation_only: bool = True

    requires_human_review: bool = False

    physical_execution_authorized: bool = False

    emergency_stop_reset_authorized: bool = False

    safety_interlock_bypass_authorized: bool = False

    safety_evaluation_passed: bool = False

    safety_evaluation_report_uri: str | None = None

    known_limitations: list[str] = field(
        default_factory=list
    )

    prohibited_uses: list[str] = field(
        default_factory=list
    )

    def validate_for_category(
        self,
        category: ModelCategory,
    ) -> None:
        if (
            category.value in ROBOT_ACTION_CATEGORIES
            and not self.simulation_only
        ):
            raise ModelSafetyError(
                "Robotic action models must be "
                "simulation-only in this platform"
            )

        if self.physical_execution_authorized:
            raise ModelSafetyError(
                "Model metadata cannot authorize "
                "physical robot execution"
            )

        if self.emergency_stop_reset_authorized:
            raise ModelSafetyError(
                "Model metadata cannot authorize "
                "emergency-stop reset"
            )

        if self.safety_interlock_bypass_authorized:
            raise ModelSafetyError(
                "Model metadata cannot authorize "
                "safety-interlock bypass"
            )


# ============================================================
# Approval Metadata
# ============================================================

@dataclass(slots=True)
class ModelApproval:
    status: ApprovalStatus = ApprovalStatus.PENDING

    reviewer_id: str | None = None

    reviewed_at: datetime | None = None

    notes: str = ""

    approval_reference: str | None = None

    def __post_init__(self) -> None:
        if self.reviewed_at is not None:
            self.reviewed_at = parse_datetime(
                self.reviewed_at
            )


# ============================================================
# Audit Events
# ============================================================

@dataclass(slots=True)
class MetadataAuditEvent:
    event_id: str
    actor_id: str
    action: str

    timestamp: datetime = field(
        default_factory=utc_now
    )

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# Complete Model Metadata
# ============================================================

@dataclass(slots=True)
class ModelMetadata:
    identity: ModelIdentity
    ownership: ModelOwnership

    framework: ModelFramework

    lifecycle_stage: ModelLifecycleStage = (
        ModelLifecycleStage.DEVELOPMENT
    )

    training: TrainingMetadata = field(
        default_factory=TrainingMetadata
    )

    evaluation: EvaluationMetadata = field(
        default_factory=EvaluationMetadata
    )

    interface: ModelInterface = field(
        default_factory=ModelInterface
    )

    compute: ComputeRequirements = field(
        default_factory=ComputeRequirements
    )

    artifacts: list[ModelArtifact] = field(
        default_factory=list
    )

    safety: ModelSafetyMetadata = field(
        default_factory=ModelSafetyMetadata
    )

    approval: ModelApproval = field(
        default_factory=ModelApproval
    )

    tags: list[str] = field(
        default_factory=list
    )

    custom_metadata: dict[str, Any] = field(
        default_factory=dict
    )

    audit_history: list[MetadataAuditEvent] = field(
        default_factory=list
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.safety.validate_for_category(
            self.identity.category
        )

        validate_semver(
            self.schema_version
        )

        self.updated_at = parse_datetime(
            self.updated_at
        )

        if (
            self.lifecycle_stage
            == ModelLifecycleStage.PRODUCTION
        ):
            self.validate_production_readiness()

    def validate_production_readiness(self) -> None:
        if (
            self.approval.status
            != ApprovalStatus.APPROVED
        ):
            raise ModelApprovalError(
                "Production models require approval"
            )

        if not self.evaluation.passed_quality_gate:
            raise ModelApprovalError(
                "Production models must pass "
                "evaluation quality gates"
            )

        if not self.safety.safety_evaluation_passed:
            raise ModelSafetyError(
                "Production models must pass "
                "safety evaluation"
            )

        if not self.artifacts:
            raise MetadataValidationError(
                "Production models require artifacts"
            )

    def add_audit_event(
        self,
        *,
        actor_id: str,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.audit_history.append(
            MetadataAuditEvent(
                event_id=generate_id(),
                actor_id=actor_id,
                action=action,
                details=details or {},
            )
        )

        self.updated_at = utc_now()

    @property
    def registry_key(self) -> str:
        return (
            f"{self.ownership.tenant_id}:"
            f"{self.identity.name}:"
            f"{self.identity.version}"
        )

    def to_dict(self) -> dict[str, Any]:
        def normalize(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value

            if isinstance(value, datetime):
                return value.isoformat()

            if isinstance(value, dict):
                return {
                    key: normalize(item)
                    for key, item in value.items()
                }

            if isinstance(value, list):
                return [
                    normalize(item)
                    for item in value
                ]

            return value

        return normalize(asdict(self))

    def to_json(
        self,
        *,
        indent: int = 2,
    ) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "model_id": self.identity.model_id,
            "name": self.identity.name,
            "version": self.identity.version,
            "category": self.identity.category.value,
            "framework": self.framework.value,
            "tenant_id": self.ownership.tenant_id,
            "lifecycle_stage": (
                self.lifecycle_stage.value
            ),
            "approval_status": (
                self.approval.status.value
            ),
            "safety_classification": (
                self.safety.classification.value
            ),
            "simulation_only": (
                self.safety.simulation_only
            ),
            "preferred_device": (
                self.compute.preferred_device.value
            ),
            "artifact_count": len(self.artifacts),
            "updated_at": self.updated_at.isoformat(),
        }


# ============================================================
# Serialization
# ============================================================

def metadata_from_dict(
    data: Mapping[str, Any],
) -> ModelMetadata:
    identity_data = dict(data["identity"])
    ownership_data = dict(data["ownership"])

    training_data = dict(
        data.get("training", {})
    )

    evaluation_data = dict(
        data.get("evaluation", {})
    )

    interface_data = dict(
        data.get("interface", {})
    )

    compute_data = dict(
        data.get("compute", {})
    )

    safety_data = dict(
        data.get("safety", {})
    )

    approval_data = dict(
        data.get("approval", {})
    )

    identity_data["category"] = ModelCategory(
        identity_data["category"]
    )

    training_data["datasets"] = [
        DatasetReference(**item)
        for item in training_data.get(
            "datasets", []
        )
    ]

    evaluation_data["benchmarks"] = [
        BenchmarkResult(**item)
        for item in evaluation_data.get(
            "benchmarks", []
        )
    ]

    interface_data["inputs"] = [
        TensorSchema(**item)
        for item in interface_data.get(
            "inputs", []
        )
    ]

    interface_data["outputs"] = [
        TensorSchema(**item)
        for item in interface_data.get(
            "outputs", []
        )
    ]

    if "supported_devices" in compute_data:
        compute_data["supported_devices"] = [
            ComputeDevice(device)
            for device in compute_data[
                "supported_devices"
            ]
        ]

    if "preferred_device" in compute_data:
        compute_data["preferred_device"] = (
            ComputeDevice(
                compute_data["preferred_device"]
            )
        )

    if "precision" in compute_data:
        compute_data["precision"] = (
            ModelPrecision(
                compute_data["precision"]
            )
        )

    if "classification" in safety_data:
        safety_data["classification"] = (
            SafetyClassification(
                safety_data["classification"]
            )
        )

    if "status" in approval_data:
        approval_data["status"] = (
            ApprovalStatus(
                approval_data["status"]
            )
        )

    return ModelMetadata(
        identity=ModelIdentity(
            **identity_data
        ),
        ownership=ModelOwnership(
            **ownership_data
        ),
        framework=ModelFramework(
            data["framework"]
        ),
        lifecycle_stage=ModelLifecycleStage(
            data.get(
                "lifecycle_stage",
                "development",
            )
        ),
        training=TrainingMetadata(
            **training_data
        ),
        evaluation=EvaluationMetadata(
            **evaluation_data
        ),
        interface=ModelInterface(
            **interface_data
        ),
        compute=ComputeRequirements(
            **compute_data
        ),
        artifacts=[
            ModelArtifact(**item)
            for item in data.get(
                "artifacts", []
            )
        ],
        safety=ModelSafetyMetadata(
            **safety_data
        ),
        approval=ModelApproval(
            **approval_data
        ),
        tags=list(
            data.get("tags", [])
        ),
        custom_metadata=dict(
            data.get("custom_metadata", {})
        ),
        audit_history=[
            MetadataAuditEvent(**item)
            for item in data.get(
                "audit_history", []
            )
        ],
        updated_at=data.get(
            "updated_at",
            utc_now(),
        ),
        schema_version=data.get(
            "schema_version",
            "1.0.0",
        ),
    )


# ============================================================
# Authorization Protocol
# ============================================================

class MetadataAuthorizer(Protocol):
    def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        ...


class DenyByDefaultMetadataAuthorizer:
    def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        raise MetadataPermissionError(
            "Metadata authorization is not configured"
        )


# ============================================================
# Metadata Repository
# ============================================================

class ModelMetadataRepository:
    """
    JSON-backed metadata repository.

    Designed for local development and testing.

    For production, use a transactional database
    with tenant-scoped authorization, audit logs,
    and optimistic concurrency control.
    """

    def __init__(
        self,
        *,
        storage_directory: str | Path,
        authorizer: MetadataAuthorizer,
    ) -> None:
        self.storage_directory = Path(
            storage_directory
        ).resolve()

        self.storage_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.authorizer = authorizer

        self._lock = threading.RLock()

    def _metadata_path(
        self,
        tenant_id: str,
        model_name: str,
        version: str,
    ) -> Path:
        validate_identifier(
            tenant_id,
            "tenant_id",
        )

        validate_identifier(
            model_name,
            "model_name",
        )

        validate_semver(version)

        directory = (
            self.storage_directory
            / tenant_id
            / model_name
        )

        return directory / f"{version}.json"

    def save(
        self,
        metadata: ModelMetadata,
        *,
        actor_id: str,
        overwrite: bool = False,
    ) -> Path:
        tenant_id = metadata.ownership.tenant_id

        self.authorizer.require_permission(
            tenant_id=tenant_id,
            user_id=actor_id,
            permission="ml:metadata:write",
        )

        metadata.validate()

        path = self._metadata_path(
            tenant_id,
            metadata.identity.name,
            metadata.identity.version,
        )

        metadata.add_audit_event(
            actor_id=actor_id,
            action="metadata.saved",
            details={
                "overwrite": overwrite,
            },
        )

        payload = metadata.to_json()

        if (
            len(payload.encode("utf-8"))
            > MAX_METADATA_SIZE_BYTES
        ):
            raise MetadataValidationError(
                "Metadata exceeds maximum size"
            )

        with self._lock:
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if path.exists() and not overwrite:
                raise ModelMetadataConflictError(
                    "Model metadata already exists"
                )

            temporary_path: Path | None = None

            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=path.parent,
                    prefix=".metadata-",
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_file.write(payload)
                    temporary_file.flush()

                    os.fsync(
                        temporary_file.fileno()
                    )

                    temporary_path = Path(
                        temporary_file.name
                    )

                os.replace(
                    temporary_path,
                    path,
                )

            except OSError as exc:
                raise MetadataStorageError(
                    "Unable to persist model metadata"
                ) from exc

            finally:
                if (
                    temporary_path is not None
                    and temporary_path.exists()
                ):
                    temporary_path.unlink()

        return path

    def load(
        self,
        *,
        tenant_id: str,
        model_name: str,
        version: str,
        actor_id: str,
    ) -> ModelMetadata:
        self.authorizer.require_permission(
            tenant_id=tenant_id,
            user_id=actor_id,
            permission="ml:metadata:read",
        )

        path = self._metadata_path(
            tenant_id,
            model_name,
            version,
        )

        with self._lock:
            if not path.is_file():
                raise ModelMetadataNotFoundError(
                    "Model metadata not found"
                )

            if path.stat().st_size > MAX_METADATA_SIZE_BYTES:
                raise MetadataStorageError(
                    "Metadata file exceeds size limit"
                )

            try:
                data = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

                metadata = metadata_from_dict(
                    data
                )

            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                raise MetadataStorageError(
                    "Unable to load model metadata"
                ) from exc

        if metadata.ownership.tenant_id != tenant_id:
            raise MetadataStorageError(
                "Stored metadata tenant mismatch"
            )

        return metadata

    def list_versions(
        self,
        *,
        tenant_id: str,
        model_name: str,
        actor_id: str,
    ) -> list[str]:
        self.authorizer.require_permission(
            tenant_id=tenant_id,
            user_id=actor_id,
            permission="ml:metadata:read",
        )

        directory = self._metadata_path(
            tenant_id,
            model_name,
            "0.0.0",
        ).parent

        if not directory.exists():
            return []

        return sorted(
            path.stem
            for path in directory.glob("*.json")
            if SEMVER_PATTERN.fullmatch(
                path.stem
            )
        )

    def delete(
        self,
        *,
        tenant_id: str,
        model_name: str,
        version: str,
        actor_id: str,
    ) -> None:
        self.authorizer.require_permission(
            tenant_id=tenant_id,
            user_id=actor_id,
            permission="ml:metadata:delete",
        )

        path = self._metadata_path(
            tenant_id,
            model_name,
            version,
        )

        with self._lock:
            if not path.is_file():
                raise ModelMetadataNotFoundError(
                    "Model metadata not found"
                )

            path.unlink()


# ============================================================
# Model Comparison
# ============================================================

def compare_model_metadata(
    first: ModelMetadata,
    second: ModelMetadata,
) -> dict[str, Any]:
    metric_names = (
        set(first.evaluation.test_metrics)
        | set(second.evaluation.test_metrics)
    )

    metric_comparison = {}

    for metric_name in sorted(metric_names):
        first_value = (
            first.evaluation.test_metrics.get(
                metric_name
            )
        )

        second_value = (
            second.evaluation.test_metrics.get(
                metric_name
            )
        )

        delta = None

        if (
            first_value is not None
            and second_value is not None
        ):
            delta = second_value - first_value

        metric_comparison[metric_name] = {
            "first": first_value,
            "second": second_value,
            "delta": delta,
        }

    return {
        "first_model": first.summary(),
        "second_model": second.summary(),
        "test_metrics": metric_comparison,
        "compute": {
            "first_memory_mb": (
                first.compute.estimated_model_memory_mb
            ),
            "second_memory_mb": (
                second.compute.estimated_model_memory_mb
            ),
        },
        "safety": {
            "first_classification": (
                first.safety.classification.value
            ),
            "second_classification": (
                second.safety.classification.value
            ),
            "first_simulation_only": (
                first.safety.simulation_only
            ),
            "second_simulation_only": (
                second.safety.simulation_only
            ),
        },
    }


# ============================================================
# Factory
# ============================================================

def create_model_metadata(
    *,
    name: str,
    version: str,
    category: ModelCategory,
    framework: ModelFramework,
    tenant_id: str,
    owner_id: str,
    architecture: str = "",
    description: str = "",
) -> ModelMetadata:
    return ModelMetadata(
        identity=ModelIdentity(
            model_id=generate_id(),
            name=name,
            version=version,
            category=category,
            architecture=architecture,
            description=description,
        ),
        ownership=ModelOwnership(
            tenant_id=tenant_id,
            owner_id=owner_id,
        ),
        framework=framework,
        safety=ModelSafetyMetadata(
            classification=(
                SafetyClassification.HIGH
                if category.value in ROBOT_ACTION_CATEGORIES
                else SafetyClassification.MODERATE
            ),
            simulation_only=True,
            requires_human_review=(
                category.value
                in ROBOT_ACTION_CATEGORIES
            ),
        ),
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "ModelMetadataError",
    "MetadataValidationError",
    "ModelMetadataNotFoundError",
    "ModelMetadataConflictError",
    "MetadataPermissionError",
    "ModelApprovalError",
    "ModelSafetyError",
    "MetadataStorageError",
    "ModelCategory",
    "ModelFramework",
    "ModelLifecycleStage",
    "ApprovalStatus",
    "SafetyClassification",
    "ComputeDevice",
    "ModelPrecision",
    "ModelIdentity",
    "ModelOwnership",
    "DatasetReference",
    "TrainingMetadata",
    "BenchmarkResult",
    "EvaluationMetadata",
    "TensorSchema",
    "ModelInterface",
    "ComputeRequirements",
    "ModelArtifact",
    "ModelSafetyMetadata",
    "ModelApproval",
    "MetadataAuditEvent",
    "ModelMetadata",
    "MetadataAuthorizer",
    "DenyByDefaultMetadataAuthorizer",
    "ModelMetadataRepository",
    "metadata_from_dict",
    "compare_model_metadata",
    "create_model_metadata",
    "calculate_sha256",
]
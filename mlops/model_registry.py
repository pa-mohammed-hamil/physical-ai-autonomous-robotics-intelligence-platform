"""Model Registry module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/mlops/model_registry.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
MLOps — Model Registry

Features:
    - Register ML, DL, GenAI and robotics models
    - Model version management
    - Model artifact SHA-256 verification
    - Content-addressed artifact storage
    - Model metadata and hyperparameters
    - Training experiment and dataset lineage
    - Model evaluation metrics
    - Model stage transitions
    - Approval workflows
    - Production model promotion
    - Model version comparison
    - Model rollback
    - Model archival
    - Model search and filtering
    - Multi-tenant access control
    - SQLite persistence
    - Audit logging
    - JSON export
    - Thread-safe operations
    - Unit tests
    - Synthetic demonstration

Python 3.11+
Standard library only.

Important:
    This module manages model registry metadata and artifacts.

    It does not load model weights, deploy containers,
    modify Kubernetes resources or control physical robots.

    Model promotion is a registry state transition, not
    authorization for physical robot actuation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import threading
import unittest
import uuid

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class ModelRegistryError(Exception):
    pass


class RegistryValidationError(ModelRegistryError):
    pass


class RegistryNotFoundError(ModelRegistryError):
    pass


class RegistryAuthorizationError(ModelRegistryError):
    pass


class RegistryConflictError(ModelRegistryError):
    pass


class RegistryIntegrityError(ModelRegistryError):
    pass


class RegistryApprovalError(ModelRegistryError):
    pass


class RegistryTransitionError(ModelRegistryError):
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
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
    ):
        raise RegistryValidationError(
            f"Invalid {field_name}"
        )

    return value.strip()


def ensure_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise RegistryValidationError(
            "Expected datetime"
        )

    if value.tzinfo is None:
        raise RegistryValidationError(
            "Datetime must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def timestamp(value: datetime) -> str:
    return ensure_utc(value).isoformat(
        timespec="microseconds"
    )


def parse_timestamp(value: str) -> datetime:
    return ensure_utc(
        datetime.fromisoformat(value)
    )


def validate_number(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise RegistryValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RegistryValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise RegistryValidationError(
            f"{name} must be finite"
        )

    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return timestamp(value)

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: json_safe(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    return value


def json_encode(value: Any) -> str:
    try:
        return json.dumps(
            json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RegistryValidationError(
            "Value must be JSON serializable"
        ) from exc


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)
            total += len(chunk)

    return digest.hexdigest(), total


def validate_sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise RegistryValidationError(
            "Invalid SHA-256 digest"
        )

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class ModelFramework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    JAX = "jax"
    HUGGING_FACE = "hugging_face"
    OPENVINO = "openvino"
    TENSORRT = "tensorrt"
    ROS2 = "ros2"
    CUSTOM = "custom"


class ModelCategory(str, Enum):
    COMPUTER_VISION = "computer_vision"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    DEPTH_ESTIMATION = "depth_estimation"
    POSE_ESTIMATION = "pose_estimation"
    SENSOR_FUSION = "sensor_fusion"
    SLAM = "slam"
    NAVIGATION = "navigation"
    MOTION_PLANNING = "motion_planning"
    ROBOT_CONTROL = "robot_control"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    IMITATION_LEARNING = "imitation_learning"
    WORLD_MODEL = "world_model"
    VISION_LANGUAGE_ACTION = "vision_language_action"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    GENERATIVE_AI = "generative_ai"
    LANGUAGE_MODEL = "language_model"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    MULTIMODAL = "multimodal"
    OTHER = "other"


class ModelStage(str, Enum):
    REGISTERED = "registered"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class RegistryEventType(str, Enum):
    MODEL_CREATED = "model_created"
    VERSION_REGISTERED = "version_registered"
    METRICS_UPDATED = "metrics_updated"
    APPROVAL_RECORDED = "approval_recorded"
    STAGE_CHANGED = "stage_changed"
    MODEL_ROLLED_BACK = "model_rolled_back"
    VERSION_ARCHIVED = "version_archived"
    ARTIFACT_CHECKED_OUT = "artifact_checked_out"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ModelRegistryConfig:
    database_path: Path = Path(
        "./data/mlops/model_registry.sqlite3"
    )

    artifact_root: Path = Path(
        "./data/mlops/registry_artifacts"
    )

    maximum_artifact_bytes: int = 5 * 1024**3
    chunk_size: int = 1024 * 1024

    require_production_approval: bool = True
    prevent_self_approval: bool = True

    verify_artifacts_on_registration: bool = True
    verify_artifacts_on_checkout: bool = True
    verify_artifacts_on_promotion: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "database_path",
            Path(self.database_path).expanduser().resolve(),
        )

        object.__setattr__(
            self,
            "artifact_root",
            Path(self.artifact_root).expanduser().resolve(),
        )

        if (
            self.maximum_artifact_bytes <= 0
            or self.chunk_size <= 0
        ):
            raise RegistryValidationError(
                "Artifact limits must be positive"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class RegistryAccess:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_write: bool = False
    can_approve: bool = False
    can_promote: bool = False
    can_manage: bool = False

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
class RegisteredModel:
    model_id: str
    tenant_id: str
    model_name: str
    category: ModelCategory

    description: str

    created_at: datetime
    created_by: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ModelVersion:
    version_id: str
    model_id: str
    tenant_id: str

    version: str
    framework: ModelFramework
    stage: ModelStage

    artifact_sha256: str
    artifact_size_bytes: int
    artifact_filename: str

    created_at: datetime
    created_by: str

    experiment_id: str | None = None
    run_id: str | None = None
    dataset_version_id: str | None = None

    parent_version_id: str | None = None

    hyperparameters: dict[str, Any] = field(
        default_factory=dict
    )

    metrics: dict[str, float] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ModelApproval:
    approval_id: str
    tenant_id: str
    model_id: str
    version_id: str

    decision: ApprovalDecision

    approved_by: str
    approved_at: datetime

    reason: str


@dataclass(frozen=True)
class ModelStageTransition:
    transition_id: str
    tenant_id: str
    model_id: str
    version_id: str

    previous_stage: ModelStage
    new_stage: ModelStage

    changed_by: str
    changed_at: datetime

    reason: str


@dataclass(frozen=True)
class ModelArtifactIntegrity:
    version_id: str

    valid: bool
    exists: bool

    expected_sha256: str
    actual_sha256: str | None

    expected_size_bytes: int
    actual_size_bytes: int | None


@dataclass(frozen=True)
class ModelVersionComparison:
    left_version_id: str
    right_version_id: str

    same_artifact: bool
    same_framework: bool
    same_dataset_version: bool

    metric_differences: dict[str, float | None]

    hyperparameter_changes: dict[
        str,
        dict[str, Any],
    ]


# ============================================================
# ARTIFACT STORAGE
# ============================================================

class RegistryArtifactStore:

    def __init__(
        self,
        config: ModelRegistryConfig,
    ) -> None:
        self.config = config

        self.root = (
            config.artifact_root
            / "objects"
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.RLock()

    def path_for_hash(
        self,
        digest: str,
    ) -> Path:
        validate_sha256(digest)

        return (
            self.root
            / digest[:2]
            / digest[2:4]
            / digest
        )

    def store(
        self,
        source_path: Path,
    ) -> tuple[str, int]:
        source = Path(
            source_path
        ).expanduser()

        if source.is_symlink():
            raise RegistryValidationError(
                "Symlink artifacts are not supported"
            )

        source = source.resolve(strict=True)

        if not source.is_file():
            raise RegistryValidationError(
                "Artifact source must be a file"
            )

        with self._lock:
            staging = self.root / ".staging"

            staging.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary = (
                staging
                / f"{generate_id()}.tmp"
            )

            digest = hashlib.sha256()
            size = 0

            try:
                with (
                    source.open("rb") as reader,
                    temporary.open("xb") as writer,
                ):
                    while True:
                        chunk = reader.read(
                            self.config.chunk_size
                        )

                        if not chunk:
                            break

                        size += len(chunk)

                        if (
                            size
                            > self.config.maximum_artifact_bytes
                        ):
                            raise RegistryValidationError(
                                "Artifact exceeds maximum size"
                            )

                        digest.update(chunk)
                        writer.write(chunk)

                    writer.flush()
                    os.fsync(writer.fileno())

                checksum = digest.hexdigest()

                destination = self.path_for_hash(
                    checksum
                )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                if destination.exists():
                    existing_hash, existing_size = (
                        sha256_file(
                            destination,
                            self.config.chunk_size,
                        )
                    )

                    if (
                        existing_hash != checksum
                        or existing_size != size
                    ):
                        raise RegistryIntegrityError(
                            "Stored object failed integrity verification"
                        )

                else:
                    os.replace(
                        temporary,
                        destination,
                    )

                if self.config.verify_artifacts_on_registration:
                    actual_hash, actual_size = sha256_file(
                        destination,
                        self.config.chunk_size,
                    )

                    if (
                        actual_hash != checksum
                        or actual_size != size
                    ):
                        raise RegistryIntegrityError(
                            "Artifact verification failed"
                        )

                return checksum, size

            finally:
                temporary.unlink(
                    missing_ok=True
                )

    def verify(
        self,
        digest: str,
        expected_size: int,
        version_id: str,
    ) -> ModelArtifactIntegrity:
        path = self.path_for_hash(
            digest
        )

        if not path.is_file():
            return ModelArtifactIntegrity(
                version_id=version_id,
                valid=False,
                exists=False,
                expected_sha256=digest,
                actual_sha256=None,
                expected_size_bytes=expected_size,
                actual_size_bytes=None,
            )

        actual_hash, actual_size = sha256_file(
            path,
            self.config.chunk_size,
        )

        return ModelArtifactIntegrity(
            version_id=version_id,
            valid=(
                actual_hash == digest
                and actual_size == expected_size
            ),
            exists=True,
            expected_sha256=digest,
            actual_sha256=actual_hash,
            expected_size_bytes=expected_size,
            actual_size_bytes=actual_size,
        )

    def checkout(
        self,
        digest: str,
        expected_size: int,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        source = self.path_for_hash(
            digest
        )

        destination = (
            Path(destination).expanduser().absolute()
        )

        with self._lock:
            if self.config.verify_artifacts_on_checkout:
                integrity = self.verify(
                    digest,
                    expected_size,
                    version_id="checkout",
                )

                if not integrity.valid:
                    raise RegistryIntegrityError(
                        "Artifact integrity verification failed"
                    )

            if not source.is_file():
                raise RegistryNotFoundError(
                    "Artifact object missing"
                )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if destination.exists() and not overwrite:
                raise RegistryConflictError(
                    "Checkout destination already exists"
                )

            temporary = (
                destination.parent
                / f".{destination.name}.{generate_id()}.tmp"
            )

            try:
                shutil.copyfile(
                    source,
                    temporary,
                )

                actual_hash, actual_size = sha256_file(
                    temporary,
                    self.config.chunk_size,
                )

                if (
                    actual_hash != digest
                    or actual_size != expected_size
                ):
                    raise RegistryIntegrityError(
                        "Copied artifact failed integrity verification"
                    )

                if overwrite:
                    os.replace(
                        temporary,
                        destination,
                    )
                else:
                    try:
                        with destination.open("xb") as writer:
                            with temporary.open("rb") as reader:
                                shutil.copyfileobj(
                                    reader,
                                    writer,
                                    self.config.chunk_size,
                                )
                    except FileExistsError as exc:
                        raise RegistryConflictError(
                            "Checkout destination already exists"
                        ) from exc

                return destination

            finally:
                temporary.unlink(
                    missing_ok=True
                )


# ============================================================
# MODEL REGISTRY
# ============================================================

class ModelRegistry:

    def __init__(
        self,
        config: ModelRegistryConfig | None = None,
    ) -> None:
        self.config = (
            config or ModelRegistryConfig()
        )

        self._lock = threading.RLock()

        self.config.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.artifacts = RegistryArtifactStore(
            self.config
        )

        self._db = sqlite3.connect(
            str(self.config.database_path),
            check_same_thread=False,
            timeout=10.0,
        )

        self._db.row_factory = sqlite3.Row

        self._db.execute(
            "PRAGMA foreign_keys = ON"
        )

        self._db.execute(
            "PRAGMA journal_mode = WAL"
        )

        self._initialize_database()

    def __enter__(self) -> ModelRegistry:
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --------------------------------------------------------
    # AUTHORIZATION
    # --------------------------------------------------------

    @staticmethod
    def _authorize(
        access: RegistryAccess,
        permission: str,
    ) -> None:
        if not getattr(
            access,
            permission,
            False,
        ):
            raise RegistryAuthorizationError(
                f"Missing permission: {permission}"
            )

    @staticmethod
    def _read_access(
        access: RegistryAccess,
    ) -> RegistryAccess:
        return RegistryAccess(
            tenant_id=access.tenant_id,
            user_id=access.user_id,
            can_read=True,
        )

    # --------------------------------------------------------
    # DATABASE INITIALIZATION
    # --------------------------------------------------------

    def _initialize_database(self) -> None:
        with self._lock, self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE (
                        tenant_id,
                        model_name
                    )
                );

                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    artifact_sha256 TEXT NOT NULL,
                    artifact_size_bytes INTEGER NOT NULL,
                    artifact_filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    experiment_id TEXT,
                    run_id TEXT,
                    dataset_version_id TEXT,
                    parent_version_id TEXT,
                    hyperparameters_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(model_id)
                        REFERENCES models(model_id),
                    UNIQUE (
                        tenant_id,
                        model_id,
                        version
                    )
                );

                CREATE INDEX IF NOT EXISTS
                idx_registry_versions
                ON model_versions (
                    tenant_id,
                    model_id,
                    stage
                );

                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    approved_by TEXT NOT NULL,
                    approved_at TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    FOREIGN KEY(version_id)
                        REFERENCES model_versions(version_id)
                );

                CREATE TABLE IF NOT EXISTS transitions (
                    transition_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    previous_stage TEXT NOT NULL,
                    new_stage TEXT NOT NULL,
                    changed_by TEXT NOT NULL,
                    changed_at TEXT NOT NULL,
                    reason TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    audit_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    model_id TEXT,
                    version_id TEXT,
                    timestamp TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                """
            )

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        access: RegistryAccess,
        event_type: RegistryEventType,
        *,
        model_id: str | None = None,
        version_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO audit_log (
                audit_id,
                tenant_id,
                user_id,
                event_type,
                model_id,
                version_id,
                timestamp,
                details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                generate_id(),
                access.tenant_id,
                access.user_id,
                event_type.value,
                model_id,
                version_id,
                timestamp(utc_now()),
                json_encode(dict(details or {})),
            ),
        )

    # --------------------------------------------------------
    # ROW CONVERSION
    # --------------------------------------------------------

    @staticmethod
    def _row_to_model(
        row: sqlite3.Row,
    ) -> RegisteredModel:
        return RegisteredModel(
            model_id=row["model_id"],
            tenant_id=row["tenant_id"],
            model_name=row["model_name"],
            category=ModelCategory(
                row["category"]
            ),
            description=row["description"],
            created_at=parse_timestamp(
                row["created_at"]
            ),
            created_by=row["created_by"],
            metadata=json.loads(
                row["metadata_json"]
            ),
        )

    @staticmethod
    def _row_to_version(
        row: sqlite3.Row,
    ) -> ModelVersion:
        return ModelVersion(
            version_id=row["version_id"],
            model_id=row["model_id"],
            tenant_id=row["tenant_id"],
            version=row["version"],
            framework=ModelFramework(
                row["framework"]
            ),
            stage=ModelStage(
                row["stage"]
            ),
            artifact_sha256=row[
                "artifact_sha256"
            ],
            artifact_size_bytes=row[
                "artifact_size_bytes"
            ],
            artifact_filename=row[
                "artifact_filename"
            ],
            created_at=parse_timestamp(
                row["created_at"]
            ),
            created_by=row["created_by"],
            experiment_id=row["experiment_id"],
            run_id=row["run_id"],
            dataset_version_id=row[
                "dataset_version_id"
            ],
            parent_version_id=row[
                "parent_version_id"
            ],
            hyperparameters=json.loads(
                row["hyperparameters_json"]
            ),
            metrics=json.loads(
                row["metrics_json"]
            ),
            metadata=json.loads(
                row["metadata_json"]
            ),
        )

    # --------------------------------------------------------
    # REGISTER MODEL
    # --------------------------------------------------------

    def register_model(
        self,
        model_name: str,
        category: ModelCategory,
        access: RegistryAccess,
        *,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> RegisteredModel:
        self._authorize(
            access,
            "can_write",
        )

        model = RegisteredModel(
            model_id=generate_id(),
            tenant_id=access.tenant_id,
            model_name=validate_identifier(
                model_name,
                "model_name",
            ),
            category=ModelCategory(category),
            description=description,
            created_at=utc_now(),
            created_by=access.user_id,
            metadata=json.loads(
                json_encode(dict(metadata or {}))
            ),
        )

        with self._lock, self._db:
            try:
                self._db.execute(
                    """
                    INSERT INTO models (
                        model_id,
                        tenant_id,
                        model_name,
                        category,
                        description,
                        created_at,
                        created_by,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model.model_id,
                        model.tenant_id,
                        model.model_name,
                        model.category.value,
                        model.description,
                        timestamp(model.created_at),
                        model.created_by,
                        json_encode(model.metadata),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise RegistryConflictError(
                    "Model already registered"
                ) from exc

            self._audit(
                access,
                RegistryEventType.MODEL_CREATED,
                model_id=model.model_id,
            )

        return model

    # --------------------------------------------------------
    # GET MODEL
    # --------------------------------------------------------

    def get_model(
        self,
        model_id: str,
        access: RegistryAccess,
    ) -> RegisteredModel:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            row = self._db.execute(
                """
                SELECT *
                FROM models
                WHERE tenant_id = ?
                AND model_id = ?
                """,
                (
                    access.tenant_id,
                    model_id,
                ),
            ).fetchone()

        if row is None:
            raise RegistryNotFoundError(
                "Model not found"
            )

        return self._row_to_model(
            row
        )

    # --------------------------------------------------------
    # LIST MODELS
    # --------------------------------------------------------

    def list_models(
        self,
        access: RegistryAccess,
        *,
        category: ModelCategory | None = None,
        search: str | None = None,
    ) -> tuple[RegisteredModel, ...]:
        self._authorize(
            access,
            "can_read",
        )

        query = """
            SELECT *
            FROM models
            WHERE tenant_id = ?
        """

        parameters: list[Any] = [
            access.tenant_id
        ]

        if category is not None:
            query += " AND category = ?"
            parameters.append(
                ModelCategory(category).value
            )

        if search is not None:
            query += " AND model_name LIKE ?"
            escaped = (
                search.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            query = query.replace(
                "model_name LIKE ?",
                "model_name LIKE ? ESCAPE '\\'",
            )
            parameters.append(
                f"%{escaped}%"
            )

        query += """
            ORDER BY created_at DESC
        """

        with self._lock:
            rows = self._db.execute(
                query,
                parameters,
            ).fetchall()

        return tuple(
            self._row_to_model(row)
            for row in rows
        )

    # --------------------------------------------------------
    # REGISTER MODEL VERSION
    # --------------------------------------------------------

    def register_version(
        self,
        model_id: str,
        version: str,
        framework: ModelFramework,
        artifact_path: Path,
        access: RegistryAccess,
        *,
        experiment_id: str | None = None,
        run_id: str | None = None,
        dataset_version_id: str | None = None,
        parent_version_id: str | None = None,
        hyperparameters: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelVersion:
        self._authorize(
            access,
            "can_write",
        )

        self.get_model(
            model_id,
            self._read_access(access),
        )

        if parent_version_id is not None:
            parent = self.get_version(
                parent_version_id,
                self._read_access(access),
            )

            if parent.model_id != model_id:
                raise RegistryValidationError(
                    "Parent version belongs to another model"
                )

        version = validate_identifier(
            version,
            "version",
        )

        source = Path(artifact_path)

        if (
            source.is_symlink()
            or not source.is_file()
        ):
            raise RegistryValidationError(
                "Artifact path must be a regular file"
            )

        validated_metrics = {
            str(name): validate_number(value, name)
            for name, value in dict(
                metrics or {}
            ).items()
        }

        validated_hyperparameters = json.loads(
            json_encode(
                dict(hyperparameters or {})
            )
        )

        validated_metadata = json.loads(
            json_encode(
                dict(metadata or {})
            )
        )

        checksum, size = self.artifacts.store(
            source
        )

        record = ModelVersion(
            version_id=generate_id(),
            model_id=model_id,
            tenant_id=access.tenant_id,
            version=version,
            framework=ModelFramework(framework),
            stage=ModelStage.REGISTERED,
            artifact_sha256=checksum,
            artifact_size_bytes=size,
            artifact_filename=source.name,
            created_at=utc_now(),
            created_by=access.user_id,
            experiment_id=experiment_id,
            run_id=run_id,
            dataset_version_id=dataset_version_id,
            parent_version_id=parent_version_id,
            hyperparameters=validated_hyperparameters,
            metrics=validated_metrics,
            metadata=validated_metadata,
        )

        with self._lock, self._db:
            try:
                self._db.execute(
                    """
                    INSERT INTO model_versions (
                        version_id,
                        model_id,
                        tenant_id,
                        version,
                        framework,
                        stage,
                        artifact_sha256,
                        artifact_size_bytes,
                        artifact_filename,
                        created_at,
                        created_by,
                        experiment_id,
                        run_id,
                        dataset_version_id,
                        parent_version_id,
                        hyperparameters_json,
                        metrics_json,
                        metadata_json
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        record.version_id,
                        record.model_id,
                        record.tenant_id,
                        record.version,
                        record.framework.value,
                        record.stage.value,
                        record.artifact_sha256,
                        record.artifact_size_bytes,
                        record.artifact_filename,
                        timestamp(record.created_at),
                        record.created_by,
                        record.experiment_id,
                        record.run_id,
                        record.dataset_version_id,
                        record.parent_version_id,
                        json_encode(
                            record.hyperparameters
                        ),
                        json_encode(
                            record.metrics
                        ),
                        json_encode(
                            record.metadata
                        ),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise RegistryConflictError(
                    "Model version already exists"
                ) from exc

            self._audit(
                access,
                RegistryEventType.VERSION_REGISTERED,
                model_id=model_id,
                version_id=record.version_id,
                details={
                    "version": version,
                    "framework": record.framework.value,
                    "sha256": checksum,
                },
            )

        return record

    # --------------------------------------------------------
    # GET VERSION
    # --------------------------------------------------------

    def get_version(
        self,
        version_id: str,
        access: RegistryAccess,
    ) -> ModelVersion:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            row = self._db.execute(
                """
                SELECT *
                FROM model_versions
                WHERE tenant_id = ?
                AND version_id = ?
                """,
                (
                    access.tenant_id,
                    version_id,
                ),
            ).fetchone()

        if row is None:
            raise RegistryNotFoundError(
                "Model version not found"
            )

        return self._row_to_version(
            row
        )

    # --------------------------------------------------------
    # LIST VERSIONS
    # --------------------------------------------------------

    def list_versions(
        self,
        model_id: str,
        access: RegistryAccess,
        *,
        stage: ModelStage | None = None,
    ) -> tuple[ModelVersion, ...]:
        self.get_model(
            model_id,
            access,
        )

        query = """
            SELECT *
            FROM model_versions
            WHERE tenant_id = ?
            AND model_id = ?
        """

        parameters: list[Any] = [
            access.tenant_id,
            model_id,
        ]

        if stage is not None:
            query += " AND stage = ?"
            parameters.append(
                ModelStage(stage).value
            )

        query += """
            ORDER BY created_at DESC, version_id DESC
        """

        with self._lock:
            rows = self._db.execute(
                query,
                parameters,
            ).fetchall()

        return tuple(
            self._row_to_version(row)
            for row in rows
        )

    # --------------------------------------------------------
    # UPDATE METRICS
    # --------------------------------------------------------

    def update_metrics(
        self,
        version_id: str,
        metrics: Mapping[str, float],
        access: RegistryAccess,
    ) -> ModelVersion:
        self._authorize(
            access,
            "can_write",
        )

        version = self.get_version(
            version_id,
            self._read_access(access),
        )

        if version.stage in (
            ModelStage.PRODUCTION,
            ModelStage.ARCHIVED,
        ):
            raise RegistryTransitionError(
                "Metrics cannot be modified in production "
                "or after archival"
            )

        validated = {
            str(name): validate_number(value, name)
            for name, value in metrics.items()
        }

        merged = {
            **version.metrics,
            **validated,
        }

        with self._lock, self._db:
            self._db.execute(
                """
                UPDATE model_versions
                SET metrics_json = ?
                WHERE tenant_id = ?
                AND version_id = ?
                """,
                (
                    json_encode(merged),
                    access.tenant_id,
                    version_id,
                ),
            )

            # A metric change invalidates previous
            # promotion approvals.
            self._db.execute(
                """
                DELETE FROM approvals
                WHERE tenant_id = ?
                AND version_id = ?
                """,
                (
                    access.tenant_id,
                    version_id,
                ),
            )

            self._audit(
                access,
                RegistryEventType.METRICS_UPDATED,
                model_id=version.model_id,
                version_id=version_id,
                details={"metrics": validated},
            )

        return self.get_version(
            version_id,
            self._read_access(access),
        )

    # --------------------------------------------------------
    # ARTIFACT INTEGRITY
    # --------------------------------------------------------

    def verify_artifact(
        self,
        version_id: str,
        access: RegistryAccess,
    ) -> ModelArtifactIntegrity:
        version = self.get_version(
            version_id,
            access,
        )

        return self.artifacts.verify(
            version.artifact_sha256,
            version.artifact_size_bytes,
            version.version_id,
        )

    # --------------------------------------------------------
    # CHECKOUT ARTIFACT
    # --------------------------------------------------------

    def checkout_artifact(
        self,
        version_id: str,
        destination: Path,
        access: RegistryAccess,
        *,
        overwrite: bool = False,
    ) -> Path:
        version = self.get_version(
            version_id,
            access,
        )

        result = self.artifacts.checkout(
            version.artifact_sha256,
            version.artifact_size_bytes,
            destination,
            overwrite=overwrite,
        )

        with self._lock, self._db:
            self._audit(
                access,
                RegistryEventType.ARTIFACT_CHECKED_OUT,
                model_id=version.model_id,
                version_id=version_id,
                details={
                    "destination": str(result),
                },
            )

        return result

    # --------------------------------------------------------
    # APPROVAL
    # --------------------------------------------------------

    def approve_version(
        self,
        version_id: str,
        access: RegistryAccess,
        *,
        decision: ApprovalDecision = (
            ApprovalDecision.APPROVED
        ),
        reason: str = "",
    ) -> ModelApproval:
        self._authorize(
            access,
            "can_approve",
        )

        version = self.get_version(
            version_id,
            self._read_access(access),
        )

        if version.stage == ModelStage.ARCHIVED:
            raise RegistryApprovalError(
                "Archived versions cannot be approved"
            )

        if (
            self.config.prevent_self_approval
            and version.created_by == access.user_id
        ):
            raise RegistryApprovalError(
                "Model creator cannot self-approve"
            )

        approval = ModelApproval(
            approval_id=generate_id(),
            tenant_id=access.tenant_id,
            model_id=version.model_id,
            version_id=version_id,
            decision=ApprovalDecision(decision),
            approved_by=access.user_id,
            approved_at=utc_now(),
            reason=reason,
        )

        with self._lock, self._db:
            self._db.execute(
                """
                INSERT INTO approvals (
                    approval_id,
                    tenant_id,
                    model_id,
                    version_id,
                    decision,
                    approved_by,
                    approved_at,
                    reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval.approval_id,
                    approval.tenant_id,
                    approval.model_id,
                    approval.version_id,
                    approval.decision.value,
                    approval.approved_by,
                    timestamp(approval.approved_at),
                    approval.reason,
                ),
            )

            self._audit(
                access,
                RegistryEventType.APPROVAL_RECORDED,
                model_id=version.model_id,
                version_id=version_id,
                details={
                    "decision": approval.decision.value,
                    "reason": reason,
                },
            )

        return approval

    def _latest_approval(
        self,
        version_id: str,
        tenant_id: str,
    ) -> ApprovalDecision | None:
        row = self._db.execute(
            """
            SELECT decision
            FROM approvals
            WHERE tenant_id = ?
            AND version_id = ?
            ORDER BY approved_at DESC, approval_id DESC
            LIMIT 1
            """,
            (
                tenant_id,
                version_id,
            ),
        ).fetchone()

        return (
            ApprovalDecision(row["decision"])
            if row
            else None
        )

    # --------------------------------------------------------
    # STAGE TRANSITIONS
    # --------------------------------------------------------

    @staticmethod
    def _allowed_transition(
        previous: ModelStage,
        target: ModelStage,
    ) -> bool:
        allowed = {
            ModelStage.REGISTERED: {
                ModelStage.DEVELOPMENT,
                ModelStage.ARCHIVED,
            },
            ModelStage.DEVELOPMENT: {
                ModelStage.STAGING,
                ModelStage.ARCHIVED,
            },
            ModelStage.STAGING: {
                ModelStage.DEVELOPMENT,
                ModelStage.PRODUCTION,
                ModelStage.ARCHIVED,
            },
            ModelStage.PRODUCTION: {
                ModelStage.STAGING,
                ModelStage.ARCHIVED,
            },
            ModelStage.ARCHIVED: set(),
        }

        return target in allowed[previous]

    def transition_stage(
        self,
        version_id: str,
        target_stage: ModelStage,
        access: RegistryAccess,
        *,
        reason: str = "",
    ) -> ModelStageTransition:
        self._authorize(
            access,
            "can_promote",
        )

        target_stage = ModelStage(target_stage)

        with self._lock, self._db:
            version = self.get_version(
                version_id,
                self._read_access(access),
            )

            if not self._allowed_transition(
                version.stage,
                target_stage,
            ):
                raise RegistryTransitionError(
                    f"Invalid transition: "
                    f"{version.stage.value} -> "
                    f"{target_stage.value}"
                )

            if target_stage == ModelStage.PRODUCTION:
                if self.config.require_production_approval:
                    approval = self._latest_approval(
                        version_id,
                        access.tenant_id,
                    )

                    if approval != ApprovalDecision.APPROVED:
                        raise RegistryApprovalError(
                            "Production promotion requires approval"
                        )

                if self.config.verify_artifacts_on_promotion:
                    integrity = self.verify_artifact(
                        version_id,
                        self._read_access(access),
                    )

                    if not integrity.valid:
                        raise RegistryIntegrityError(
                            "Artifact failed production integrity check"
                        )

                existing = self._db.execute(
                    """
                    SELECT version_id
                    FROM model_versions
                    WHERE tenant_id = ?
                    AND model_id = ?
                    AND stage = ?
                    AND version_id != ?
                    """,
                    (
                        access.tenant_id,
                        version.model_id,
                        ModelStage.PRODUCTION.value,
                        version_id,
                    ),
                ).fetchall()

                for row in existing:
                    self._change_stage(
                        row["version_id"],
                        ModelStage.PRODUCTION,
                        ModelStage.STAGING,
                        access,
                        reason="Superseded by production promotion",
                    )

            transition = self._change_stage(
                version_id,
                version.stage,
                target_stage,
                access,
                reason=reason,
            )

        return transition

    def _change_stage(
        self,
        version_id: str,
        previous: ModelStage,
        target: ModelStage,
        access: RegistryAccess,
        *,
        reason: str,
    ) -> ModelStageTransition:
        version = self.get_version(
            version_id,
            self._read_access(access),
        )

        transition = ModelStageTransition(
            transition_id=generate_id(),
            tenant_id=access.tenant_id,
            model_id=version.model_id,
            version_id=version_id,
            previous_stage=previous,
            new_stage=target,
            changed_by=access.user_id,
            changed_at=utc_now(),
            reason=reason,
        )

        self._db.execute(
            """
            UPDATE model_versions
            SET stage = ?
            WHERE tenant_id = ?
            AND version_id = ?
            """,
            (
                target.value,
                access.tenant_id,
                version_id,
            ),
        )

        self._db.execute(
            """
            INSERT INTO transitions (
                transition_id,
                tenant_id,
                model_id,
                version_id,
                previous_stage,
                new_stage,
                changed_by,
                changed_at,
                reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transition.transition_id,
                transition.tenant_id,
                transition.model_id,
                transition.version_id,
                transition.previous_stage.value,
                transition.new_stage.value,
                transition.changed_by,
                timestamp(transition.changed_at),
                transition.reason,
            ),
        )

        self._audit(
            access,
            RegistryEventType.STAGE_CHANGED,
            model_id=version.model_id,
            version_id=version_id,
            details={
                "previous": previous.value,
                "new": target.value,
                "reason": reason,
            },
        )

        return transition

    # --------------------------------------------------------
    # GET PRODUCTION VERSION
    # --------------------------------------------------------

    def get_production_version(
        self,
        model_id: str,
        access: RegistryAccess,
    ) -> ModelVersion | None:
        self.get_model(
            model_id,
            access,
        )

        versions = self.list_versions(
            model_id,
            access,
            stage=ModelStage.PRODUCTION,
        )

        return (
            versions[0]
            if versions
            else None
        )

    # --------------------------------------------------------
    # ROLLBACK
    # --------------------------------------------------------

    def rollback(
        self,
        model_id: str,
        target_version_id: str,
        access: RegistryAccess,
        *,
        reason: str = "Manual rollback",
    ) -> ModelVersion:
        self._authorize(
            access,
            "can_promote",
        )

        with self._lock, self._db:
            self.get_model(
                model_id,
                self._read_access(access),
            )

            target = self.get_version(
                target_version_id,
                self._read_access(access),
            )

            if target.model_id != model_id:
                raise RegistryValidationError(
                    "Rollback target belongs to another model"
                )

            if target.stage != ModelStage.STAGING:
                raise RegistryTransitionError(
                    "Rollback target must be in staging"
                )

            self.transition_stage(
                target_version_id,
                ModelStage.PRODUCTION,
                access,
                reason=reason,
            )

            self._audit(
                access,
                RegistryEventType.MODEL_ROLLED_BACK,
                model_id=model_id,
                version_id=target_version_id,
                details={"reason": reason},
            )

        return self.get_version(
            target_version_id,
            self._read_access(access),
        )

    # --------------------------------------------------------
    # ARCHIVE
    # --------------------------------------------------------

    def archive_version(
        self,
        version_id: str,
        access: RegistryAccess,
        *,
        reason: str = "",
    ) -> ModelVersion:
        self._authorize(
            access,
            "can_manage",
        )

        version = self.get_version(
            version_id,
            self._read_access(access),
        )

        if version.stage == ModelStage.PRODUCTION:
            raise RegistryTransitionError(
                "Move production version out of production "
                "before archival"
            )

        self.transition_stage(
            version_id,
            ModelStage.ARCHIVED,
            access,
            reason=reason,
        )

        with self._lock, self._db:
            self._audit(
                access,
                RegistryEventType.VERSION_ARCHIVED,
                model_id=version.model_id,
                version_id=version_id,
            )

        return self.get_version(
            version_id,
            self._read_access(access),
        )

    # --------------------------------------------------------
    # COMPARE VERSIONS
    # --------------------------------------------------------

    def compare_versions(
        self,
        left_version_id: str,
        right_version_id: str,
        access: RegistryAccess,
    ) -> ModelVersionComparison:
        left = self.get_version(
            left_version_id,
            access,
        )

        right = self.get_version(
            right_version_id,
            access,
        )

        if left.model_id != right.model_id:
            raise RegistryValidationError(
                "Versions must belong to the same model"
            )

        metric_differences = {}

        for name in sorted(
            set(left.metrics)
            | set(right.metrics)
        ):
            if (
                name in left.metrics
                and name in right.metrics
            ):
                metric_differences[name] = (
                    right.metrics[name]
                    - left.metrics[name]
                )
            else:
                metric_differences[name] = None

        hyperparameter_changes = {}

        for name in sorted(
            set(left.hyperparameters)
            | set(right.hyperparameters)
        ):
            left_value = left.hyperparameters.get(
                name
            )

            right_value = right.hyperparameters.get(
                name
            )

            if left_value != right_value:
                hyperparameter_changes[name] = {
                    "left": left_value,
                    "right": right_value,
                }

        return ModelVersionComparison(
            left_version_id=left_version_id,
            right_version_id=right_version_id,
            same_artifact=(
                left.artifact_sha256
                == right.artifact_sha256
            ),
            same_framework=(
                left.framework == right.framework
            ),
            same_dataset_version=(
                left.dataset_version_id
                == right.dataset_version_id
            ),
            metric_differences=metric_differences,
            hyperparameter_changes=hyperparameter_changes,
        )

    # --------------------------------------------------------
    # LINEAGE
    # --------------------------------------------------------

    def get_lineage(
        self,
        version_id: str,
        access: RegistryAccess,
    ) -> dict[str, Any]:
        version = self.get_version(
            version_id,
            access,
        )

        ancestors = []
        visited = {version_id}

        current = version

        while current.parent_version_id is not None:
            parent_id = current.parent_version_id

            if parent_id in visited:
                raise RegistryIntegrityError(
                    "Version lineage contains a cycle"
                )

            visited.add(parent_id)

            parent = self.get_version(
                parent_id,
                access,
            )

            if parent.model_id != version.model_id:
                raise RegistryIntegrityError(
                    "Cross-model lineage detected"
                )

            ancestors.append(
                {
                    "version_id": parent.version_id,
                    "version": parent.version,
                    "sha256": parent.artifact_sha256,
                }
            )

            current = parent

        return {
            "model_id": version.model_id,
            "version_id": version.version_id,
            "version": version.version,
            "experiment_id": version.experiment_id,
            "run_id": version.run_id,
            "dataset_version_id": (
                version.dataset_version_id
            ),
            "ancestors": ancestors,
        }

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        access: RegistryAccess,
        *,
        model_id: str | None = None,
        limit: int = 100,
    ) -> tuple[dict[str, Any], ...]:
        self._authorize(
            access,
            "can_read",
        )

        if limit <= 0:
            raise RegistryValidationError(
                "limit must be positive"
            )

        query = """
            SELECT *
            FROM audit_log
            WHERE tenant_id = ?
        """

        parameters: list[Any] = [
            access.tenant_id
        ]

        if model_id is not None:
            self.get_model(
                model_id,
                access,
            )

            query += " AND model_id = ?"
            parameters.append(model_id)

        query += """
            ORDER BY timestamp DESC
            LIMIT ?
        """

        parameters.append(limit)

        with self._lock:
            rows = self._db.execute(
                query,
                parameters,
            ).fetchall()

        return tuple(
            {
                "audit_id": row["audit_id"],
                "event_type": row["event_type"],
                "model_id": row["model_id"],
                "version_id": row["version_id"],
                "user_id": row["user_id"],
                "timestamp": row["timestamp"],
                "details": json.loads(
                    row["details_json"]
                ),
            }
            for row in rows
        )

    # --------------------------------------------------------
    # JSON EXPORT
    # --------------------------------------------------------

    def export_model_json(
        self,
        model_id: str,
        access: RegistryAccess,
    ) -> str:
        model = self.get_model(
            model_id,
            access,
        )

        versions = self.list_versions(
            model_id,
            access,
        )

        production = self.get_production_version(
            model_id,
            access,
        )

        return json.dumps(
            json_safe(
                {
                    "model": model,
                    "versions": versions,
                    "production_version": production,
                    "audit_history": self.audit_history(
                        access,
                        model_id=model_id,
                    ),
                }
            ),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC MODEL FACTORY
# ============================================================

class SyntheticModelFactory:

    @staticmethod
    def create_artifact(
        directory: Path,
        *,
        version: str,
    ) -> Path:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            directory
            / f"robot_model_{version}.bin"
        )

        payload = json_encode(
            {
                "model": "robot-perception",
                "version": version,
                "synthetic": True,
            }
        ).encode("utf-8")

        path.write_bytes(
            payload
            + bytes(range(256)) * 8
        )

        return path


# ============================================================
# UNIT TESTS
# ============================================================

class TestModelRegistry(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

        root = Path(
            self.temporary.name
        )

        self.registry = ModelRegistry(
            ModelRegistryConfig(
                database_path=root / "registry.sqlite3",
                artifact_root=root / "artifacts",
            )
        )

        self.access = RegistryAccess(
            tenant_id="robotics_lab",
            user_id="model_creator",
            can_read=True,
            can_write=True,
            can_promote=True,
            can_manage=True,
        )

        self.approver = RegistryAccess(
            tenant_id="robotics_lab",
            user_id="model_reviewer",
            can_read=True,
            can_approve=True,
        )

        self.model = self.registry.register_model(
            model_name="robot-perception",
            category=ModelCategory.OBJECT_DETECTION,
            access=self.access,
        )

        artifact = SyntheticModelFactory.create_artifact(
            root,
            version="v1",
        )

        self.version = self.registry.register_version(
            model_id=self.model.model_id,
            version="v1",
            framework=ModelFramework.PYTORCH,
            artifact_path=artifact,
            access=self.access,
            metrics={
                "precision": 0.91,
                "recall": 0.88,
            },
        )

    def tearDown(self) -> None:
        self.registry.close()
        self.temporary.cleanup()

    def test_register_model(self) -> None:
        self.assertEqual(
            self.model.model_name,
            "robot-perception",
        )

    def test_register_version(self) -> None:
        self.assertEqual(
            self.version.version,
            "v1",
        )

    def test_get_model(self) -> None:
        model = self.registry.get_model(
            self.model.model_id,
            self.access,
        )

        self.assertEqual(
            model.model_id,
            self.model.model_id,
        )

    def test_get_version(self) -> None:
        version = self.registry.get_version(
            self.version.version_id,
            self.access,
        )

        self.assertEqual(
            version.framework,
            ModelFramework.PYTORCH,
        )

    def test_duplicate_model(self) -> None:
        with self.assertRaises(
            RegistryConflictError
        ):
            self.registry.register_model(
                "robot-perception",
                ModelCategory.OBJECT_DETECTION,
                self.access,
            )

    def test_duplicate_version(self) -> None:
        artifact = SyntheticModelFactory.create_artifact(
            Path(self.temporary.name),
            version="duplicate",
        )

        with self.assertRaises(
            RegistryConflictError
        ):
            self.registry.register_version(
                self.model.model_id,
                "v1",
                ModelFramework.PYTORCH,
                artifact,
                self.access,
            )

    def test_verify_artifact(self) -> None:
        result = self.registry.verify_artifact(
            self.version.version_id,
            self.access,
        )

        self.assertTrue(result.valid)

    def test_checkout_artifact(self) -> None:
        destination = (
            Path(self.temporary.name)
            / "checkout"
            / "model.bin"
        )

        result = self.registry.checkout_artifact(
            self.version.version_id,
            destination,
            self.access,
        )

        self.assertTrue(result.exists())

    def test_update_metrics(self) -> None:
        updated = self.registry.update_metrics(
            self.version.version_id,
            {"f1": 0.895},
            self.access,
        )

        self.assertIn(
            "f1",
            updated.metrics,
        )

    def test_self_approval_rejected(self) -> None:
        self_approver = RegistryAccess(
            tenant_id="robotics_lab",
            user_id="model_creator",
            can_read=True,
            can_approve=True,
        )

        with self.assertRaises(
            RegistryApprovalError
        ):
            self.registry.approve_version(
                self.version.version_id,
                self_approver,
            )

    def test_stage_promotion(self) -> None:
        self.registry.transition_stage(
            self.version.version_id,
            ModelStage.DEVELOPMENT,
            self.access,
        )

        self.registry.transition_stage(
            self.version.version_id,
            ModelStage.STAGING,
            self.access,
        )

        self.registry.approve_version(
            self.version.version_id,
            self.approver,
        )

        transition = self.registry.transition_stage(
            self.version.version_id,
            ModelStage.PRODUCTION,
            self.access,
        )

        self.assertEqual(
            transition.new_stage,
            ModelStage.PRODUCTION,
        )

    def test_production_requires_approval(self) -> None:
        self.registry.transition_stage(
            self.version.version_id,
            ModelStage.DEVELOPMENT,
            self.access,
        )

        self.registry.transition_stage(
            self.version.version_id,
            ModelStage.STAGING,
            self.access,
        )

        with self.assertRaises(
            RegistryApprovalError
        ):
            self.registry.transition_stage(
                self.version.version_id,
                ModelStage.PRODUCTION,
                self.access,
            )

    def test_compare_versions(self) -> None:
        artifact = SyntheticModelFactory.create_artifact(
            Path(self.temporary.name),
            version="v2",
        )

        second = self.registry.register_version(
            self.model.model_id,
            "v2",
            ModelFramework.PYTORCH,
            artifact,
            self.access,
            metrics={
                "precision": 0.95,
            },
        )

        result = self.registry.compare_versions(
            self.version.version_id,
            second.version_id,
            self.access,
        )

        self.assertFalse(
            result.same_artifact
        )

    def test_lineage(self) -> None:
        artifact = SyntheticModelFactory.create_artifact(
            Path(self.temporary.name),
            version="v2",
        )

        child = self.registry.register_version(
            self.model.model_id,
            "v2",
            ModelFramework.PYTORCH,
            artifact,
            self.access,
            parent_version_id=self.version.version_id,
        )

        lineage = self.registry.get_lineage(
            child.version_id,
            self.access,
        )

        self.assertEqual(
            len(lineage["ancestors"]),
            1,
        )

    def test_tenant_isolation(self) -> None:
        other = RegistryAccess(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
        )

        with self.assertRaises(
            RegistryNotFoundError
        ):
            self.registry.get_model(
                self.model.model_id,
                other,
            )

    def test_export(self) -> None:
        payload = json.loads(
            self.registry.export_model_json(
                self.model.model_id,
                self.access,
            )
        )

        self.assertEqual(
            len(payload["versions"]),
            1,
        )

    def test_audit(self) -> None:
        events = self.registry.audit_history(
            self.access
        )

        self.assertGreaterEqual(
            len(events),
            2,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        config = ModelRegistryConfig(
            database_path=root / "registry.sqlite3",
            artifact_root=root / "artifacts",
        )

        with ModelRegistry(config) as registry:
            creator = RegistryAccess(
                tenant_id="physical_ai_lab",
                user_id="ml_engineer",
                can_read=True,
                can_write=True,
                can_promote=True,
                can_manage=True,
            )

            reviewer = RegistryAccess(
                tenant_id="physical_ai_lab",
                user_id="ml_reviewer",
                can_read=True,
                can_approve=True,
            )

            model = registry.register_model(
                model_name="autonomous-robot-perception",
                category=ModelCategory.OBJECT_DETECTION,
                access=creator,
                description=(
                    "Robot perception model registry example"
                ),
                metadata={
                    "robot_type": "autonomous_mobile_robot",
                    "environment": "warehouse",
                    "synthetic": True,
                },
            )

            artifact_v1 = (
                SyntheticModelFactory.create_artifact(
                    root,
                    version="v1",
                )
            )

            version_v1 = registry.register_version(
                model_id=model.model_id,
                version="v1.0.0",
                framework=ModelFramework.PYTORCH,
                artifact_path=artifact_v1,
                access=creator,
                experiment_id="experiment-001",
                run_id="training-run-001",
                dataset_version_id="warehouse-dataset-v1",
                hyperparameters={
                    "learning_rate": 0.0001,
                    "batch_size": 16,
                    "epochs": 50,
                },
                metrics={
                    "precision": 0.91,
                    "recall": 0.88,
                    "f1": 0.895,
                },
            )

            registry.transition_stage(
                version_v1.version_id,
                ModelStage.DEVELOPMENT,
                creator,
            )

            registry.transition_stage(
                version_v1.version_id,
                ModelStage.STAGING,
                creator,
            )

            registry.approve_version(
                version_v1.version_id,
                reviewer,
                reason="Synthetic evaluation approved",
            )

            registry.transition_stage(
                version_v1.version_id,
                ModelStage.PRODUCTION,
                creator,
                reason="Registry promotion demonstration",
            )

            production = registry.get_production_version(
                model.model_id,
                creator,
            )

            integrity = registry.verify_artifact(
                version_v1.version_id,
                creator,
            )

            print("\nPHYSICAL AI MODEL REGISTRY")
            print("=" * 70)

            print(
                json.dumps(
                    json_safe(
                        {
                            "model": model,
                            "production_version": production,
                            "artifact_integrity": integrity,
                            "lineage": registry.get_lineage(
                                version_v1.version_id,
                                creator,
                            ),
                            "synthetic_data": True,
                            "actual_deployment": False,
                            "robot_actuation": False,
                        }
                    ),
                    indent=2,
                    allow_nan=False,
                )
            )

    print("\nRUNNING UNIT TESTS")
    print("=" * 70)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestModelRegistry
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
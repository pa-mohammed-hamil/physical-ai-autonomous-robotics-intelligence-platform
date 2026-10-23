"""Model Artifacts module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/mlops/model_artifacts.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
MLOps — Model Artifact Management

Features:
    - Model artifact registration
    - Content-addressed artifact storage (SHA-256)
    - Artifact integrity verification
    - Model version association
    - Artifact metadata and lineage
    - Model checkpoint management
    - ONNX, PyTorch, TensorFlow and other artifact formats
    - Model configuration storage
    - Tokenizer and preprocessing artifact support
    - Evaluation report storage
    - Training metrics storage
    - Robotics policy and perception model support
    - Artifact version comparison
    - Artifact download / checkout
    - Artifact archival and restoration
    - Artifact listing and filtering
    - Multi-tenant isolation
    - Role-based authorization
    - SQLite metadata persistence
    - Atomic artifact writes
    - Thread-safe operations
    - Audit logging
    - JSON export
    - Unit tests
    - Demonstration

Python 3.11+
Standard library only.

Security:
    Artifacts are stored as opaque files.

    This module never deserializes model weights, pickle files,
    PyTorch checkpoints or executable model packages.

    Model loading must occur in a separate trusted execution
    environment with appropriate security controls.

    Artifact integrity verification does not establish that
    an artifact is safe or trustworthy.

Safety:
    Artifact registration or retrieval does not authorize
    deployment or physical robot actuation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import unittest
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class ModelArtifactError(Exception):
    pass


class ArtifactValidationError(ModelArtifactError):
    pass


class ArtifactNotFoundError(ModelArtifactError):
    pass


class ArtifactAuthorizationError(ModelArtifactError):
    pass


class ArtifactConflictError(ModelArtifactError):
    pass


class ArtifactIntegrityError(ModelArtifactError):
    pass


class ArtifactStorageError(ModelArtifactError):
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
        raise ArtifactValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def ensure_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ArtifactValidationError(
            "Expected datetime"
        )

    if value.tzinfo is None:
        raise ArtifactValidationError(
            "Datetime must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def timestamp_text(value: datetime) -> str:
    return ensure_utc(value).isoformat(
        timespec="microseconds"
    )


def parse_timestamp(value: str) -> datetime:
    return ensure_utc(
        datetime.fromisoformat(value)
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return timestamp_text(value)

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


def json_text(value: Any) -> str:
    try:
        return json.dumps(
            json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_digest(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            char not in "0123456789abcdef"
            for char in value
        )
    ):
        raise ArtifactValidationError(
            "Invalid SHA-256 digest"
        )

    return value


def validate_filename(value: str) -> str:
    validate_identifier(
        value,
        "filename",
    )

    if (
        value in (".", "..")
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or Path(value).name != value
    ):
        raise ArtifactValidationError(
            "Artifact filename must be a simple filename"
        )

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class ArtifactType(str, Enum):
    MODEL_WEIGHTS = "model_weights"
    CHECKPOINT = "checkpoint"
    ONNX_MODEL = "onnx_model"
    TORCHSCRIPT_MODEL = "torchscript_model"
    TENSORFLOW_MODEL = "tensorflow_model"
    TFLITE_MODEL = "tflite_model"
    TOKENIZER = "tokenizer"
    PREPROCESSOR = "preprocessor"
    FEATURE_SCHEMA = "feature_schema"
    MODEL_CONFIG = "model_config"
    TRAINING_CONFIG = "training_config"
    EVALUATION_REPORT = "evaluation_report"
    METRICS = "metrics"
    LABEL_MAP = "label_map"
    CALIBRATION = "calibration"
    ROBOT_POLICY = "robot_policy"
    WORLD_MODEL = "world_model"
    EMBEDDING_MODEL = "embedding_model"
    RERANKER_MODEL = "reranker_model"
    OTHER = "other"


class ArtifactFramework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    HUGGING_FACE = "hugging_face"
    JAX = "jax"
    OPENVINO = "openvino"
    TENSORRT = "tensorrt"
    ROS2 = "ros2"
    CUSTOM = "custom"
    NONE = "none"


class ArtifactStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class ArtifactIntegrityStatus(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    CORRUPTED = "corrupted"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ModelArtifactConfig:
    storage_root: Path = Path(
        "./data/mlops/model_artifacts"
    )

    database_path: Path = Path(
        "./data/mlops/model_artifacts.sqlite3"
    )

    maximum_artifact_bytes: int = 5 * 1024**3
    chunk_size: int = 1024 * 1024

    sqlite_timeout_seconds: float = 10.0

    verify_after_write: bool = True
    verify_before_checkout: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "storage_root",
            Path(self.storage_root).expanduser().resolve(),
        )

        object.__setattr__(
            self,
            "database_path",
            Path(self.database_path).expanduser().resolve(),
        )

        if (
            self.maximum_artifact_bytes <= 0
            or self.chunk_size <= 0
            or self.sqlite_timeout_seconds <= 0
        ):
            raise ArtifactValidationError(
                "Configuration values must be positive"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class ArtifactAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_write: bool = False
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
class ModelArtifact:
    artifact_id: str

    tenant_id: str
    model_name: str
    model_version: str

    artifact_name: str
    artifact_type: ArtifactType
    framework: ArtifactFramework

    filename: str
    sha256: str
    size_bytes: int

    storage_key: str

    created_at: datetime
    created_by: str

    status: ArtifactStatus = ArtifactStatus.ACTIVE

    experiment_id: str | None = None
    run_id: str | None = None
    dataset_version_id: str | None = None

    parent_artifact_id: str | None = None

    description: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ArtifactIntegrityReport:
    artifact_id: str
    status: ArtifactIntegrityStatus

    expected_sha256: str
    actual_sha256: str | None

    expected_size_bytes: int
    actual_size_bytes: int | None

    checked_at: datetime

    valid: bool


@dataclass(frozen=True)
class ArtifactComparison:
    left_artifact_id: str
    right_artifact_id: str

    identical_content: bool

    same_model: bool
    same_version: bool
    same_type: bool
    same_framework: bool

    size_difference_bytes: int

    metadata_changes: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ArtifactStatistics:
    tenant_id: str

    total_artifacts: int
    active_artifacts: int
    archived_artifacts: int

    logical_storage_bytes: int
    unique_storage_bytes: int

    artifact_types: dict[str, int]
    frameworks: dict[str, int]


# ============================================================
# CONTENT-ADDRESSED OBJECT STORE
# ============================================================

class ContentAddressedArtifactStore:

    def __init__(
        self,
        config: ModelArtifactConfig,
    ) -> None:
        self.config = config

        self.root = (
            config.storage_root
            / "objects"
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.RLock()

    def object_path(
        self,
        digest: str,
    ) -> Path:
        validate_digest(digest)

        return (
            self.root
            / digest[:2]
            / digest[2:4]
            / digest
        )

    def store_file(
        self,
        source_path: Path,
    ) -> tuple[str, int, str]:
        source = Path(source_path).expanduser().resolve(
            strict=True
        )

        if not source.is_file():
            raise ArtifactValidationError(
                "Source must be a regular file"
            )

        if source.is_symlink():
            raise ArtifactValidationError(
                "Symlink source files are not allowed"
            )

        with self._lock:
            staging_directory = (
                self.root / ".staging"
            )

            staging_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_path = (
                staging_directory
                / f"{generate_id()}.tmp"
            )

            digest = hashlib.sha256()
            total_size = 0

            try:
                with (
                    source.open("rb") as reader,
                    temporary_path.open("xb") as writer,
                ):
                    while True:
                        chunk = reader.read(
                            self.config.chunk_size
                        )

                        if not chunk:
                            break

                        total_size += len(chunk)

                        if (
                            total_size
                            > self.config.maximum_artifact_bytes
                        ):
                            raise ArtifactValidationError(
                                "Artifact exceeds maximum size"
                            )

                        digest.update(chunk)
                        writer.write(chunk)

                    writer.flush()
                    os.fsync(writer.fileno())

                checksum = digest.hexdigest()

                destination = self.object_path(
                    checksum
                )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                if destination.exists():
                    existing_hash, existing_size = sha256_file(
                        destination,
                        self.config.chunk_size,
                    )

                    if (
                        existing_hash != checksum
                        or existing_size != total_size
                    ):
                        raise ArtifactIntegrityError(
                            "Existing content-addressed object "
                            "failed integrity verification"
                        )

                    temporary_path.unlink(
                        missing_ok=True
                    )

                else:
                    os.replace(
                        temporary_path,
                        destination,
                    )

                if self.config.verify_after_write:
                    verified_hash, verified_size = sha256_file(
                        destination,
                        self.config.chunk_size,
                    )

                    if (
                        verified_hash != checksum
                        or verified_size != total_size
                    ):
                        raise ArtifactIntegrityError(
                            "Artifact verification failed"
                        )

                return (
                    checksum,
                    total_size,
                    checksum,
                )

            finally:
                temporary_path.unlink(
                    missing_ok=True
                )

    def store_bytes(
        self,
        data: bytes,
    ) -> tuple[str, int, str]:
        if not isinstance(data, bytes):
            raise ArtifactValidationError(
                "Artifact data must be bytes"
            )

        if len(data) > self.config.maximum_artifact_bytes:
            raise ArtifactValidationError(
                "Artifact exceeds maximum size"
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = (
                Path(temporary)
                / "artifact.bin"
            )

            path.write_bytes(data)

            return self.store_file(
                path
            )

    def verify(
        self,
        digest: str,
        expected_size: int,
    ) -> tuple[
        ArtifactIntegrityStatus,
        str | None,
        int | None,
    ]:
        path = self.object_path(
            digest
        )

        if not path.is_file():
            return (
                ArtifactIntegrityStatus.MISSING,
                None,
                None,
            )

        actual_digest, actual_size = sha256_file(
            path,
            self.config.chunk_size,
        )

        if (
            actual_digest != digest
            or actual_size != expected_size
        ):
            return (
                ArtifactIntegrityStatus.CORRUPTED,
                actual_digest,
                actual_size,
            )

        return (
            ArtifactIntegrityStatus.VALID,
            actual_digest,
            actual_size,
        )

    def checkout(
        self,
        digest: str,
        expected_size: int,
        destination: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        source = self.object_path(
            digest
        )

        destination = (
            Path(destination).expanduser().absolute()
        )

        with self._lock:
            if self.config.verify_before_checkout:
                status, _, _ = self.verify(
                    digest,
                    expected_size,
                )

                if status != ArtifactIntegrityStatus.VALID:
                    raise ArtifactIntegrityError(
                        f"Artifact integrity status: {status.value}"
                    )

            if not source.is_file():
                raise ArtifactNotFoundError(
                    "Artifact object missing"
                )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if destination.exists() and not overwrite:
                raise ArtifactConflictError(
                    "Destination already exists"
                )

            temporary_path = (
                destination.parent
                / f".{destination.name}.{generate_id()}.tmp"
            )

            try:
                shutil.copyfile(
                    source,
                    temporary_path,
                )

                copied_hash, copied_size = sha256_file(
                    temporary_path,
                    self.config.chunk_size,
                )

                if (
                    copied_hash != digest
                    or copied_size != expected_size
                ):
                    raise ArtifactIntegrityError(
                        "Checkout copy failed integrity verification"
                    )

                if overwrite:
                    os.replace(
                        temporary_path,
                        destination,
                    )

                else:
                    # Exclusive creation avoids silently
                    # replacing a destination created
                    # during the checkout operation.
                    try:
                        with destination.open("xb") as output:
                            with temporary_path.open("rb") as reader:
                                shutil.copyfileobj(
                                    reader,
                                    output,
                                    self.config.chunk_size,
                                )
                    except FileExistsError as exc:
                        raise ArtifactConflictError(
                            "Destination already exists"
                        ) from exc

                    temporary_path.unlink(
                        missing_ok=True
                    )

                return destination

            finally:
                temporary_path.unlink(
                    missing_ok=True
                )


# ============================================================
# MODEL ARTIFACT MANAGER
# ============================================================

class ModelArtifactManager:

    def __init__(
        self,
        config: ModelArtifactConfig | None = None,
    ) -> None:
        self.config = (
            config or ModelArtifactConfig()
        )

        self._lock = threading.RLock()

        self.config.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.object_store = ContentAddressedArtifactStore(
            self.config
        )

        self._connection = sqlite3.connect(
            str(self.config.database_path),
            timeout=self.config.sqlite_timeout_seconds,
            check_same_thread=False,
        )

        self._connection.row_factory = sqlite3.Row

        self._connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        self._connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        self._initialize_database()

    def __enter__(self) -> ModelArtifactManager:
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
            self._connection.close()

    @staticmethod
    def _authorize(
        access: ArtifactAccessContext,
        permission: str,
    ) -> None:
        if not getattr(
            access,
            permission,
            False,
        ):
            raise ArtifactAuthorizationError(
                f"Missing permission: {permission}"
            )

    @staticmethod
    def _read_access(
        access: ArtifactAccessContext,
    ) -> ArtifactAccessContext:
        return ArtifactAccessContext(
            tenant_id=access.tenant_id,
            user_id=access.user_id,
            can_read=True,
        )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    def _initialize_database(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    artifact_name TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    storage_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    status TEXT NOT NULL,
                    experiment_id TEXT,
                    run_id TEXT,
                    dataset_version_id TEXT,
                    parent_artifact_id TEXT,
                    description TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE (
                        tenant_id,
                        model_name,
                        model_version,
                        artifact_name
                    )
                );

                CREATE INDEX IF NOT EXISTS
                idx_artifacts_model
                ON model_artifacts (
                    tenant_id,
                    model_name,
                    model_version
                );

                CREATE INDEX IF NOT EXISTS
                idx_artifacts_checksum
                ON model_artifacts (
                    sha256
                );

                CREATE TABLE IF NOT EXISTS artifact_audit (
                    audit_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    artifact_id TEXT,
                    timestamp TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                """
            )

    def _audit(
        self,
        access: ArtifactAccessContext,
        action: str,
        *,
        artifact_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO artifact_audit (
                audit_id,
                tenant_id,
                user_id,
                action,
                artifact_id,
                timestamp,
                details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                generate_id(),
                access.tenant_id,
                access.user_id,
                action,
                artifact_id,
                timestamp_text(utc_now()),
                json_text(
                    dict(details or {})
                ),
            ),
        )

    # --------------------------------------------------------
    # ROW CONVERSION
    # --------------------------------------------------------

    @staticmethod
    def _row_to_artifact(
        row: sqlite3.Row,
    ) -> ModelArtifact:
        return ModelArtifact(
            artifact_id=row["artifact_id"],
            tenant_id=row["tenant_id"],
            model_name=row["model_name"],
            model_version=row["model_version"],
            artifact_name=row["artifact_name"],
            artifact_type=ArtifactType(
                row["artifact_type"]
            ),
            framework=ArtifactFramework(
                row["framework"]
            ),
            filename=row["filename"],
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
            storage_key=row["storage_key"],
            created_at=parse_timestamp(
                row["created_at"]
            ),
            created_by=row["created_by"],
            status=ArtifactStatus(
                row["status"]
            ),
            experiment_id=row["experiment_id"],
            run_id=row["run_id"],
            dataset_version_id=row[
                "dataset_version_id"
            ],
            parent_artifact_id=row[
                "parent_artifact_id"
            ],
            description=row["description"],
            metadata=json.loads(
                row["metadata_json"]
            ),
        )

    # --------------------------------------------------------
    # REGISTER ARTIFACT
    # --------------------------------------------------------

    def register_artifact(
        self,
        model_name: str,
        model_version: str,
        artifact_name: str,
        artifact_type: ArtifactType,
        framework: ArtifactFramework,
        access: ArtifactAccessContext,
        *,
        source_path: Path | None = None,
        content: bytes | None = None,
        filename: str | None = None,
        experiment_id: str | None = None,
        run_id: str | None = None,
        dataset_version_id: str | None = None,
        parent_artifact_id: str | None = None,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelArtifact:
        self._authorize(
            access,
            "can_write",
        )

        validate_identifier(
            model_name,
            "model_name",
        )

        validate_identifier(
            model_version,
            "model_version",
        )

        validate_identifier(
            artifact_name,
            "artifact_name",
        )

        if (source_path is None) == (content is None):
            raise ArtifactValidationError(
                "Provide exactly one of source_path or content"
            )

        if parent_artifact_id is not None:
            self.get_artifact(
                parent_artifact_id,
                self._read_access(access),
            )

        if source_path is not None:
            source = Path(source_path)

            artifact_filename = validate_filename(
                filename or source.name
            )

            digest, size, storage_key = (
                self.object_store.store_file(
                    source
                )
            )

        else:
            artifact_filename = validate_filename(
                filename or f"{artifact_name}.bin"
            )

            digest, size, storage_key = (
                self.object_store.store_bytes(
                    content
                )
            )

        artifact = ModelArtifact(
            artifact_id=generate_id(),
            tenant_id=access.tenant_id,
            model_name=model_name,
            model_version=model_version,
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            framework=framework,
            filename=artifact_filename,
            sha256=digest,
            size_bytes=size,
            storage_key=storage_key,
            created_at=utc_now(),
            created_by=access.user_id,
            experiment_id=experiment_id,
            run_id=run_id,
            dataset_version_id=dataset_version_id,
            parent_artifact_id=parent_artifact_id,
            description=description,
            metadata=json.loads(
                json_text(
                    dict(metadata or {})
                )
            ),
        )

        with self._lock, self._connection:
            try:
                self._connection.execute(
                    """
                    INSERT INTO model_artifacts (
                        artifact_id,
                        tenant_id,
                        model_name,
                        model_version,
                        artifact_name,
                        artifact_type,
                        framework,
                        filename,
                        sha256,
                        size_bytes,
                        storage_key,
                        created_at,
                        created_by,
                        status,
                        experiment_id,
                        run_id,
                        dataset_version_id,
                        parent_artifact_id,
                        description,
                        metadata_json
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        artifact.artifact_id,
                        artifact.tenant_id,
                        artifact.model_name,
                        artifact.model_version,
                        artifact.artifact_name,
                        artifact.artifact_type.value,
                        artifact.framework.value,
                        artifact.filename,
                        artifact.sha256,
                        artifact.size_bytes,
                        artifact.storage_key,
                        timestamp_text(
                            artifact.created_at
                        ),
                        artifact.created_by,
                        artifact.status.value,
                        artifact.experiment_id,
                        artifact.run_id,
                        artifact.dataset_version_id,
                        artifact.parent_artifact_id,
                        artifact.description,
                        json_text(
                            artifact.metadata
                        ),
                    ),
                )

            except sqlite3.IntegrityError as exc:
                raise ArtifactConflictError(
                    "Artifact name already exists for "
                    "this model version"
                ) from exc

            self._audit(
                access,
                "artifact_registered",
                artifact_id=artifact.artifact_id,
                details={
                    "model_name": model_name,
                    "model_version": model_version,
                    "sha256": digest,
                    "size_bytes": size,
                },
            )

        return artifact

    # --------------------------------------------------------
    # REGISTER JSON ARTIFACT
    # --------------------------------------------------------

    def register_json_artifact(
        self,
        model_name: str,
        model_version: str,
        artifact_name: str,
        artifact_type: ArtifactType,
        payload: Any,
        access: ArtifactAccessContext,
        *,
        framework: ArtifactFramework = (
            ArtifactFramework.NONE
        ),
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelArtifact:
        content = json.dumps(
            json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")

        return self.register_artifact(
            model_name=model_name,
            model_version=model_version,
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            framework=framework,
            access=access,
            content=content,
            filename=f"{artifact_name}.json",
            metadata=metadata,
        )

    # --------------------------------------------------------
    # GET ARTIFACT
    # --------------------------------------------------------

    def get_artifact(
        self,
        artifact_id: str,
        access: ArtifactAccessContext,
    ) -> ModelArtifact:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            row = self._connection.execute(
                """
                SELECT *
                FROM model_artifacts
                WHERE tenant_id = ?
                AND artifact_id = ?
                """,
                (
                    access.tenant_id,
                    artifact_id,
                ),
            ).fetchone()

            if row is None:
                raise ArtifactNotFoundError(
                    "Artifact not found"
                )

            return self._row_to_artifact(
                row
            )

    # --------------------------------------------------------
    # LIST ARTIFACTS
    # --------------------------------------------------------

    def list_artifacts(
        self,
        access: ArtifactAccessContext,
        *,
        model_name: str | None = None,
        model_version: str | None = None,
        artifact_type: ArtifactType | None = None,
        status: ArtifactStatus | None = None,
        limit: int = 1000,
    ) -> tuple[ModelArtifact, ...]:
        self._authorize(
            access,
            "can_read",
        )

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
            or limit > 100000
        ):
            raise ArtifactValidationError(
                "Invalid limit"
            )

        query = """
            SELECT *
            FROM model_artifacts
            WHERE tenant_id = ?
        """

        parameters: list[Any] = [
            access.tenant_id
        ]

        filters = (
            ("model_name", model_name),
            ("model_version", model_version),
            (
                "artifact_type",
                artifact_type.value
                if artifact_type is not None
                else None,
            ),
            (
                "status",
                status.value
                if status is not None
                else None,
            ),
        )

        for column, value in filters:
            if value is not None:
                query += f" AND {column} = ?"
                parameters.append(value)

        query += """
            ORDER BY created_at DESC, artifact_id DESC
            LIMIT ?
        """

        parameters.append(limit)

        with self._lock:
            rows = self._connection.execute(
                query,
                parameters,
            ).fetchall()

        return tuple(
            self._row_to_artifact(row)
            for row in rows
        )

    # --------------------------------------------------------
    # VERIFY ARTIFACT
    # --------------------------------------------------------

    def verify_artifact(
        self,
        artifact_id: str,
        access: ArtifactAccessContext,
    ) -> ArtifactIntegrityReport:
        artifact = self.get_artifact(
            artifact_id,
            access,
        )

        status, actual_hash, actual_size = (
            self.object_store.verify(
                artifact.sha256,
                artifact.size_bytes,
            )
        )

        return ArtifactIntegrityReport(
            artifact_id=artifact.artifact_id,
            status=status,
            expected_sha256=artifact.sha256,
            actual_sha256=actual_hash,
            expected_size_bytes=artifact.size_bytes,
            actual_size_bytes=actual_size,
            checked_at=utc_now(),
            valid=(
                status == ArtifactIntegrityStatus.VALID
            ),
        )

    # --------------------------------------------------------
    # CHECKOUT ARTIFACT
    # --------------------------------------------------------

    def checkout_artifact(
        self,
        artifact_id: str,
        destination: Path,
        access: ArtifactAccessContext,
        *,
        overwrite: bool = False,
    ) -> Path:
        artifact = self.get_artifact(
            artifact_id,
            access,
        )

        result = self.object_store.checkout(
            artifact.sha256,
            artifact.size_bytes,
            destination,
            overwrite=overwrite,
        )

        with self._lock, self._connection:
            self._audit(
                access,
                "artifact_checked_out",
                artifact_id=artifact_id,
                details={
                    "destination": str(result),
                },
            )

        return result

    # --------------------------------------------------------
    # READ JSON ARTIFACT
    # --------------------------------------------------------

    def read_json_artifact(
        self,
        artifact_id: str,
        access: ArtifactAccessContext,
    ) -> Any:
        artifact = self.get_artifact(
            artifact_id,
            access,
        )

        if not artifact.filename.lower().endswith(
            ".json"
        ):
            raise ArtifactValidationError(
                "Artifact is not declared as JSON"
            )

        integrity = self.verify_artifact(
            artifact_id,
            access,
        )

        if not integrity.valid:
            raise ArtifactIntegrityError(
                "Cannot read invalid artifact"
            )

        path = self.object_store.object_path(
            artifact.sha256
        )

        try:
            return json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ArtifactValidationError(
                "Artifact contains invalid JSON"
            ) from exc

    # --------------------------------------------------------
    # ARCHIVE ARTIFACT
    # --------------------------------------------------------

    def archive_artifact(
        self,
        artifact_id: str,
        access: ArtifactAccessContext,
    ) -> ModelArtifact:
        self._authorize(
            access,
            "can_manage",
        )

        self.get_artifact(
            artifact_id,
            self._read_access(access),
        )

        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE model_artifacts
                SET status = ?
                WHERE tenant_id = ?
                AND artifact_id = ?
                """,
                (
                    ArtifactStatus.ARCHIVED.value,
                    access.tenant_id,
                    artifact_id,
                ),
            )

            self._audit(
                access,
                "artifact_archived",
                artifact_id=artifact_id,
            )

        return self.get_artifact(
            artifact_id,
            self._read_access(access),
        )

    # --------------------------------------------------------
    # RESTORE ARTIFACT
    # --------------------------------------------------------

    def restore_artifact(
        self,
        artifact_id: str,
        access: ArtifactAccessContext,
    ) -> ModelArtifact:
        self._authorize(
            access,
            "can_manage",
        )

        self.get_artifact(
            artifact_id,
            self._read_access(access),
        )

        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE model_artifacts
                SET status = ?
                WHERE tenant_id = ?
                AND artifact_id = ?
                """,
                (
                    ArtifactStatus.ACTIVE.value,
                    access.tenant_id,
                    artifact_id,
                ),
            )

            self._audit(
                access,
                "artifact_restored",
                artifact_id=artifact_id,
            )

        return self.get_artifact(
            artifact_id,
            self._read_access(access),
        )

    # --------------------------------------------------------
    # COMPARE ARTIFACTS
    # --------------------------------------------------------

    def compare_artifacts(
        self,
        left_artifact_id: str,
        right_artifact_id: str,
        access: ArtifactAccessContext,
    ) -> ArtifactComparison:
        left = self.get_artifact(
            left_artifact_id,
            access,
        )

        right = self.get_artifact(
            right_artifact_id,
            access,
        )

        metadata_changes = {}

        for key in sorted(
            set(left.metadata)
            | set(right.metadata)
        ):
            left_value = left.metadata.get(
                key
            )

            right_value = right.metadata.get(
                key
            )

            if left_value != right_value:
                metadata_changes[key] = {
                    "left": left_value,
                    "right": right_value,
                }

        return ArtifactComparison(
            left_artifact_id=left.artifact_id,
            right_artifact_id=right.artifact_id,
            identical_content=(
                left.sha256 == right.sha256
                and left.size_bytes == right.size_bytes
            ),
            same_model=(
                left.model_name == right.model_name
            ),
            same_version=(
                left.model_version == right.model_version
            ),
            same_type=(
                left.artifact_type == right.artifact_type
            ),
            same_framework=(
                left.framework == right.framework
            ),
            size_difference_bytes=(
                right.size_bytes - left.size_bytes
            ),
            metadata_changes=metadata_changes,
        )

    # --------------------------------------------------------
    # ARTIFACT LINEAGE
    # --------------------------------------------------------

    def get_lineage(
        self,
        artifact_id: str,
        access: ArtifactAccessContext,
    ) -> dict[str, Any]:
        artifact = self.get_artifact(
            artifact_id,
            access,
        )

        ancestors = []
        visited = {
            artifact.artifact_id
        }

        current = artifact

        while current.parent_artifact_id is not None:
            parent_id = current.parent_artifact_id

            if parent_id in visited:
                raise ArtifactIntegrityError(
                    "Artifact lineage contains a cycle"
                )

            visited.add(parent_id)

            current = self.get_artifact(
                parent_id,
                access,
            )

            ancestors.append(
                {
                    "artifact_id": current.artifact_id,
                    "model_version": current.model_version,
                    "sha256": current.sha256,
                }
            )

        return {
            "artifact_id": artifact.artifact_id,
            "model_name": artifact.model_name,
            "model_version": artifact.model_version,
            "experiment_id": artifact.experiment_id,
            "run_id": artifact.run_id,
            "dataset_version_id": artifact.dataset_version_id,
            "parent_artifact_id": artifact.parent_artifact_id,
            "ancestors": ancestors,
        }

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        access: ArtifactAccessContext,
    ) -> ArtifactStatistics:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM model_artifacts
                WHERE tenant_id = ?
                """,
                (
                    access.tenant_id,
                ),
            ).fetchall()

        artifacts = [
            self._row_to_artifact(row)
            for row in rows
        ]

        type_counts: dict[str, int] = {}
        framework_counts: dict[str, int] = {}

        unique_objects: dict[str, int] = {}

        for artifact in artifacts:
            type_name = artifact.artifact_type.value

            type_counts[type_name] = (
                type_counts.get(type_name, 0) + 1
            )

            framework_name = artifact.framework.value

            framework_counts[framework_name] = (
                framework_counts.get(framework_name, 0) + 1
            )

            unique_objects[
                artifact.sha256
            ] = artifact.size_bytes

        return ArtifactStatistics(
            tenant_id=access.tenant_id,
            total_artifacts=len(artifacts),
            active_artifacts=sum(
                artifact.status == ArtifactStatus.ACTIVE
                for artifact in artifacts
            ),
            archived_artifacts=sum(
                artifact.status == ArtifactStatus.ARCHIVED
                for artifact in artifacts
            ),
            logical_storage_bytes=sum(
                artifact.size_bytes
                for artifact in artifacts
            ),
            unique_storage_bytes=sum(
                unique_objects.values()
            ),
            artifact_types=type_counts,
            frameworks=framework_counts,
        )

    # --------------------------------------------------------
    # EXPORT MANIFEST
    # --------------------------------------------------------

    def export_manifest_json(
        self,
        artifact_id: str,
        access: ArtifactAccessContext,
    ) -> str:
        artifact = self.get_artifact(
            artifact_id,
            access,
        )

        integrity = self.verify_artifact(
            artifact_id,
            access,
        )

        lineage = self.get_lineage(
            artifact_id,
            access,
        )

        return json.dumps(
            json_safe(
                {
                    "artifact": artifact,
                    "integrity": integrity,
                    "lineage": lineage,
                }
            ),
            indent=2,
            allow_nan=False,
        )

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        access: ArtifactAccessContext,
        *,
        artifact_id: str | None = None,
        limit: int = 1000,
    ) -> tuple[dict[str, Any], ...]:
        self._authorize(
            access,
            "can_read",
        )

        if limit <= 0:
            raise ArtifactValidationError(
                "limit must be positive"
            )

        query = """
            SELECT *
            FROM artifact_audit
            WHERE tenant_id = ?
        """

        parameters: list[Any] = [
            access.tenant_id
        ]

        if artifact_id is not None:
            query += " AND artifact_id = ?"
            parameters.append(artifact_id)

        query += """
            ORDER BY timestamp DESC
            LIMIT ?
        """

        parameters.append(limit)

        with self._lock:
            rows = self._connection.execute(
                query,
                parameters,
            ).fetchall()

        return tuple(
            {
                "audit_id": row["audit_id"],
                "tenant_id": row["tenant_id"],
                "user_id": row["user_id"],
                "action": row["action"],
                "artifact_id": row["artifact_id"],
                "timestamp": row["timestamp"],
                "details": json.loads(
                    row["details_json"]
                ),
            }
            for row in rows
        )


# ============================================================
# SYNTHETIC ROBOTICS ARTIFACT FACTORY
# ============================================================

class SyntheticModelArtifactFactory:

    @staticmethod
    def model_weights(
        *,
        model_name: str = "robot-perception-model",
        version: str = "v1",
    ) -> bytes:
        """
        Generate opaque synthetic bytes for artifact tests.

        These are not real model weights and cannot be loaded
        by PyTorch, TensorFlow or ONNX runtimes.
        """
        header = json_text(
            {
                "synthetic": True,
                "model_name": model_name,
                "version": version,
                "architecture": "mock-vision-encoder",
            }
        ).encode("utf-8")

        payload = bytes(
            index % 256
            for index in range(2048)
        )

        return header + b"\n" + payload

    @staticmethod
    def model_configuration() -> dict[str, Any]:
        return {
            "model_name": "robot-perception-model",
            "architecture": "mock-vision-encoder",
            "input_shape": [1, 3, 640, 640],
            "output_classes": [
                "person",
                "forklift",
                "pallet",
                "robot",
            ],
            "confidence_threshold": 0.65,
            "runtime": "prototype",
            "synthetic": True,
        }

    @staticmethod
    def evaluation_report() -> dict[str, Any]:
        return {
            "dataset": "synthetic-warehouse-validation",
            "dataset_version": "v1",
            "metrics": {
                "precision": 0.91,
                "recall": 0.88,
                "f1_score": 0.895,
            },
            "synthetic": True,
            "verified_real_world_performance": False,
        }


# ============================================================
# UNIT TESTS
# ============================================================

class TestModelArtifactManager(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

        root = Path(
            self.temporary.name
        )

        self.config = ModelArtifactConfig(
            storage_root=root / "objects",
            database_path=root / "artifacts.sqlite3",
        )

        self.manager = ModelArtifactManager(
            self.config
        )

        self.access = ArtifactAccessContext(
            tenant_id="robotics_lab",
            user_id="mlops_engineer",
            can_read=True,
            can_write=True,
            can_manage=True,
        )

        self.artifact = self.manager.register_artifact(
            model_name="robot-perception",
            model_version="v1",
            artifact_name="weights",
            artifact_type=ArtifactType.MODEL_WEIGHTS,
            framework=ArtifactFramework.PYTORCH,
            access=self.access,
            content=(
                SyntheticModelArtifactFactory.model_weights()
            ),
            filename="weights.bin",
            experiment_id="experiment-001",
            run_id="run-001",
            dataset_version_id="dataset-v1",
        )

    def tearDown(self) -> None:
        self.manager.close()
        self.temporary.cleanup()

    def test_register_artifact(self) -> None:
        self.assertEqual(
            self.artifact.model_version,
            "v1",
        )

    def test_get_artifact(self) -> None:
        result = self.manager.get_artifact(
            self.artifact.artifact_id,
            self.access,
        )

        self.assertEqual(
            result.sha256,
            self.artifact.sha256,
        )

    def test_verify_artifact(self) -> None:
        result = self.manager.verify_artifact(
            self.artifact.artifact_id,
            self.access,
        )

        self.assertTrue(
            result.valid
        )

    def test_checkout_artifact(self) -> None:
        destination = (
            Path(self.temporary.name)
            / "downloaded_weights.bin"
        )

        result = self.manager.checkout_artifact(
            self.artifact.artifact_id,
            destination,
            self.access,
        )

        self.assertTrue(
            result.exists()
        )

        self.assertEqual(
            sha256_file(result)[0],
            self.artifact.sha256,
        )

    def test_duplicate_artifact(self) -> None:
        with self.assertRaises(
            ArtifactConflictError
        ):
            self.manager.register_artifact(
                model_name="robot-perception",
                model_version="v1",
                artifact_name="weights",
                artifact_type=ArtifactType.MODEL_WEIGHTS,
                framework=ArtifactFramework.PYTORCH,
                access=self.access,
                content=b"duplicate",
            )

    def test_json_artifact(self) -> None:
        artifact = self.manager.register_json_artifact(
            model_name="robot-perception",
            model_version="v1",
            artifact_name="model_config",
            artifact_type=ArtifactType.MODEL_CONFIG,
            payload=(
                SyntheticModelArtifactFactory.model_configuration()
            ),
            access=self.access,
        )

        payload = self.manager.read_json_artifact(
            artifact.artifact_id,
            self.access,
        )

        self.assertEqual(
            payload["architecture"],
            "mock-vision-encoder",
        )

    def test_archive_artifact(self) -> None:
        result = self.manager.archive_artifact(
            self.artifact.artifact_id,
            self.access,
        )

        self.assertEqual(
            result.status,
            ArtifactStatus.ARCHIVED,
        )

    def test_restore_artifact(self) -> None:
        self.manager.archive_artifact(
            self.artifact.artifact_id,
            self.access,
        )

        result = self.manager.restore_artifact(
            self.artifact.artifact_id,
            self.access,
        )

        self.assertEqual(
            result.status,
            ArtifactStatus.ACTIVE,
        )

    def test_content_deduplication(self) -> None:
        second = self.manager.register_artifact(
            model_name="robot-perception",
            model_version="v2",
            artifact_name="weights",
            artifact_type=ArtifactType.MODEL_WEIGHTS,
            framework=ArtifactFramework.PYTORCH,
            access=self.access,
            content=(
                SyntheticModelArtifactFactory.model_weights()
            ),
        )

        self.assertEqual(
            second.sha256,
            self.artifact.sha256,
        )

        stats = self.manager.statistics(
            self.access
        )

        self.assertEqual(
            stats.total_artifacts,
            2,
        )

        self.assertEqual(
            stats.unique_storage_bytes,
            self.artifact.size_bytes,
        )

    def test_compare_artifacts(self) -> None:
        second = self.manager.register_artifact(
            model_name="robot-perception",
            model_version="v2",
            artifact_name="weights",
            artifact_type=ArtifactType.MODEL_WEIGHTS,
            framework=ArtifactFramework.PYTORCH,
            access=self.access,
            content=b"different-weights",
        )

        comparison = self.manager.compare_artifacts(
            self.artifact.artifact_id,
            second.artifact_id,
            self.access,
        )

        self.assertFalse(
            comparison.identical_content
        )

        self.assertTrue(
            comparison.same_model
        )

    def test_lineage(self) -> None:
        child = self.manager.register_artifact(
            model_name="robot-perception",
            model_version="v2",
            artifact_name="fine_tuned_weights",
            artifact_type=ArtifactType.MODEL_WEIGHTS,
            framework=ArtifactFramework.PYTORCH,
            access=self.access,
            content=b"fine-tuned",
            parent_artifact_id=self.artifact.artifact_id,
        )

        lineage = self.manager.get_lineage(
            child.artifact_id,
            self.access,
        )

        self.assertEqual(
            lineage["parent_artifact_id"],
            self.artifact.artifact_id,
        )

        self.assertEqual(
            len(lineage["ancestors"]),
            1,
        )

    def test_tenant_isolation(self) -> None:
        other = ArtifactAccessContext(
            tenant_id="another_tenant",
            user_id="other_user",
            can_read=True,
        )

        with self.assertRaises(
            ArtifactNotFoundError
        ):
            self.manager.get_artifact(
                self.artifact.artifact_id,
                other,
            )

    def test_invalid_filename(self) -> None:
        with self.assertRaises(
            ArtifactValidationError
        ):
            self.manager.register_artifact(
                model_name="robot-perception",
                model_version="v2",
                artifact_name="unsafe",
                artifact_type=ArtifactType.OTHER,
                framework=ArtifactFramework.NONE,
                access=self.access,
                content=b"data",
                filename="../unsafe.bin",
            )

    def test_export_manifest(self) -> None:
        output = self.manager.export_manifest_json(
            self.artifact.artifact_id,
            self.access,
        )

        payload = json.loads(
            output
        )

        self.assertEqual(
            payload["artifact"]["sha256"],
            self.artifact.sha256,
        )

    def test_audit_history(self) -> None:
        history = self.manager.audit_history(
            self.access
        )

        self.assertGreaterEqual(
            len(history),
            1,
        )

    def test_list_artifacts(self) -> None:
        artifacts = self.manager.list_artifacts(
            self.access,
            model_name="robot-perception",
        )

        self.assertEqual(
            len(artifacts),
            1,
        )

    def test_integrity_detects_corruption(self) -> None:
        path = self.manager.object_store.object_path(
            self.artifact.sha256
        )

        path.write_bytes(
            b"corrupted-artifact"
        )

        result = self.manager.verify_artifact(
            self.artifact.artifact_id,
            self.access,
        )

        self.assertEqual(
            result.status,
            ArtifactIntegrityStatus.CORRUPTED,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(
            temporary
        )

        config = ModelArtifactConfig(
            storage_root=root / "artifact_objects",
            database_path=root / "artifact_registry.sqlite3",
        )

        with ModelArtifactManager(config) as manager:
            access = ArtifactAccessContext(
                tenant_id="physical_ai_lab",
                user_id="mlops_engineer",
                can_read=True,
                can_write=True,
                can_manage=True,
            )

            weights = manager.register_artifact(
                model_name="autonomous-robot-perception",
                model_version="v1.0.0",
                artifact_name="model_weights",
                artifact_type=ArtifactType.MODEL_WEIGHTS,
                framework=ArtifactFramework.PYTORCH,
                access=access,
                content=(
                    SyntheticModelArtifactFactory.model_weights()
                ),
                filename="model_weights.bin",
                experiment_id="robot-perception-experiment",
                run_id="training-run-001",
                dataset_version_id="warehouse-dataset-v1",
                description=(
                    "Synthetic robot perception model artifact"
                ),
                metadata={
                    "architecture": "mock-vision-encoder",
                    "environment": "warehouse",
                    "synthetic": True,
                },
            )

            configuration = manager.register_json_artifact(
                model_name="autonomous-robot-perception",
                model_version="v1.0.0",
                artifact_name="model_configuration",
                artifact_type=ArtifactType.MODEL_CONFIG,
                payload=(
                    SyntheticModelArtifactFactory.model_configuration()
                ),
                access=access,
            )

            evaluation = manager.register_json_artifact(
                model_name="autonomous-robot-perception",
                model_version="v1.0.0",
                artifact_name="evaluation_report",
                artifact_type=ArtifactType.EVALUATION_REPORT,
                payload=(
                    SyntheticModelArtifactFactory.evaluation_report()
                ),
                access=access,
            )

            integrity = manager.verify_artifact(
                weights.artifact_id,
                access,
            )

            checkout_path = (
                root
                / "checkout"
                / "robot_model_weights.bin"
            )

            manager.checkout_artifact(
                weights.artifact_id,
                checkout_path,
                access,
            )

            statistics_result = manager.statistics(
                access
            )

            print(
                "\nPHYSICAL AI MODEL ARTIFACT MANAGER"
            )
            print("=" * 70)

            print(
                json.dumps(
                    json_safe(
                        {
                            "model_name": weights.model_name,
                            "model_version": weights.model_version,
                            "artifacts": [
                                {
                                    "artifact_id": item.artifact_id,
                                    "artifact_name": item.artifact_name,
                                    "artifact_type": item.artifact_type,
                                    "sha256": item.sha256,
                                    "size_bytes": item.size_bytes,
                                }
                                for item in (
                                    weights,
                                    configuration,
                                    evaluation,
                                )
                            ],
                            "integrity": integrity,
                            "checkout_exists": checkout_path.exists(),
                            "statistics": statistics_result,
                            "synthetic_data": True,
                            "model_execution": False,
                            "physical_robot_control": False,
                        }
                    ),
                    indent=2,
                    allow_nan=False,
                )
            )

    print(
        "\nRUNNING UNIT TESTS"
    )
    print("=" * 70)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestModelArtifactManager
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
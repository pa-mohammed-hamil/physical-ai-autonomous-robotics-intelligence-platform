"""
Physical AI & Autonomous Robotics Intelligence Platform
File: ml/registry/model_registry.py

ML Model Registry and Lifecycle Management.

Features:
- Multi-model registration
- Semantic versioning
- Model artifact management
- SHA-256 artifact verification
- Tenant-aware model isolation
- Model lifecycle management
- Staging and production promotion
- Human approval requirements
- Model rollback
- Model lineage
- Deployment aliases
- Model search and filtering
- Registry audit events
- Atomic JSON persistence
- Model metadata integration
- Simulation-only robotics safety controls

Python 3.11+

Integration:
    ml/registry/model_metadata.py

Security:
    Registry operations require an authorization provider.
    Model registration never executes arbitrary model files.
    Registry promotion does not authorize robot actuation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol

try:
    from .model_metadata import (
        ModelMetadata,
        ModelMetadataRepository,
        ModelLifecycleStage,
        ApprovalStatus,
        MetadataAuthorizer,
        metadata_from_dict,
    )
except ImportError:
    from model_metadata import (
        ModelMetadata,
        ModelMetadataRepository,
        ModelLifecycleStage,
        ApprovalStatus,
        MetadataAuthorizer,
        metadata_from_dict,
    )


logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

REGISTRY_SCHEMA_VERSION = "1.0.0"

MAX_REGISTRY_FILE_BYTES = 4 * 1024 * 1024

MAX_MODEL_VERSIONS = 1000

MODEL_NAME_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
)

SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?"
    r"(?:\+[0-9A-Za-z.-]+)?$"
)

ACTION_MODEL_CATEGORIES = {
    "vla",
    "navigation",
    "manipulation",
}


# ============================================================
# Exceptions
# ============================================================

class ModelRegistryError(Exception):
    """Base registry exception."""


class RegistryValidationError(ModelRegistryError):
    """Invalid registry request or data."""


class RegistryPermissionError(ModelRegistryError):
    """Registry operation was not authorized."""


class ModelAlreadyRegisteredError(ModelRegistryError):
    """Model version already exists."""


class ModelNotFoundError(ModelRegistryError):
    """Model version could not be found."""


class ModelPromotionError(ModelRegistryError):
    """Model cannot be promoted."""


class ModelRollbackError(ModelRegistryError):
    """Model rollback failed."""


class ModelArtifactError(ModelRegistryError):
    """Model artifact validation failed."""


class ModelSafetyError(ModelRegistryError):
    """Model violates robotics safety requirements."""


class RegistryStorageError(ModelRegistryError):
    """Registry storage operation failed."""


# ============================================================
# Utilities
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def validate_name(
    value: str,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not MODEL_NAME_PATTERN.fullmatch(value)
    ):
        raise RegistryValidationError(
            f"Invalid {field_name}"
        )

    return value


def validate_version(version: str) -> str:
    if not SEMVER_PATTERN.fullmatch(version):
        raise RegistryValidationError(
            f"Invalid semantic version: {version}"
        )

    return version


def parse_timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    if timestamp.tzinfo is None:
        raise RegistryValidationError(
            "Timestamp must include timezone"
        )

    return timestamp.astimezone(timezone.utc)


def atomic_write_json(
    path: Path,
    data: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )

    if len(payload.encode("utf-8")) > MAX_REGISTRY_FILE_BYTES:
        raise RegistryStorageError(
            "Registry file exceeds size limit"
        )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".registry-",
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
        raise RegistryStorageError(
            "Unable to persist model registry"
        ) from exc

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


# ============================================================
# Enumerations
# ============================================================

class RegistryStage(str, Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class RegistryModelStatus(str, Enum):
    REGISTERED = "registered"
    READY = "ready"
    DEGRADED = "degraded"
    DISABLED = "disabled"
    FAILED = "failed"


class DeploymentAlias(str, Enum):
    LATEST = "latest"
    STAGING = "staging"
    PRODUCTION = "production"
    CHAMPION = "champion"
    CHALLENGER = "challenger"


class RegistryEventType(str, Enum):
    REGISTERED = "registered"
    UPDATED = "updated"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    ARCHIVED = "archived"
    ALIAS_UPDATED = "alias_updated"
    STATUS_UPDATED = "status_updated"
    DELETED = "deleted"


# ============================================================
# Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class RegistryScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_name(
            self.tenant_id,
            "tenant_id",
        )

        validate_name(
            self.user_id,
            "user_id",
        )


# ============================================================
# Model Registry Record
# ============================================================

@dataclass(slots=True)
class RegisteredModelRecord:
    registry_id: str

    tenant_id: str
    model_name: str
    model_version: str

    category: str
    framework: str

    stage: RegistryStage = RegistryStage.DEVELOPMENT

    status: RegistryModelStatus = (
        RegistryModelStatus.REGISTERED
    )

    metadata_path: str = ""

    artifact_uris: list[str] = field(
        default_factory=list
    )

    artifact_checksums: dict[str, str] = field(
        default_factory=dict
    )

    created_by: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    promoted_at: datetime | None = None

    previous_production_version: str | None = None

    tags: list[str] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        validate_name(
            self.tenant_id,
            "tenant_id",
        )

        validate_name(
            self.model_name,
            "model_name",
        )

        validate_version(
            self.model_version
        )

    @property
    def registry_key(self) -> str:
        return (
            f"{self.tenant_id}:"
            f"{self.model_name}:"
            f"{self.model_version}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_id": self.registry_id,
            "tenant_id": self.tenant_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "category": self.category,
            "framework": self.framework,
            "stage": self.stage.value,
            "status": self.status.value,
            "metadata_path": self.metadata_path,
            "artifact_uris": list(self.artifact_uris),
            "artifact_checksums": dict(
                self.artifact_checksums
            ),
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "promoted_at": (
                self.promoted_at.isoformat()
                if self.promoted_at
                else None
            ),
            "previous_production_version": (
                self.previous_production_version
            ),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "RegisteredModelRecord":
        return cls(
            registry_id=data["registry_id"],
            tenant_id=data["tenant_id"],
            model_name=data["model_name"],
            model_version=data["model_version"],
            category=data["category"],
            framework=data["framework"],
            stage=RegistryStage(
                data.get("stage", "development")
            ),
            status=RegistryModelStatus(
                data.get("status", "registered")
            ),
            metadata_path=data.get(
                "metadata_path", ""
            ),
            artifact_uris=list(
                data.get("artifact_uris", [])
            ),
            artifact_checksums=dict(
                data.get("artifact_checksums", {})
            ),
            created_by=data.get(
                "created_by", ""
            ),
            created_at=parse_timestamp(
                data["created_at"]
            ),
            updated_at=parse_timestamp(
                data["updated_at"]
            ),
            promoted_at=(
                parse_timestamp(data["promoted_at"])
                if data.get("promoted_at")
                else None
            ),
            previous_production_version=data.get(
                "previous_production_version"
            ),
            tags=list(
                data.get("tags", [])
            ),
        )


# ============================================================
# Registry Audit Events
# ============================================================

@dataclass(slots=True)
class RegistryAuditEvent:
    event_id: str
    tenant_id: str
    model_name: str
    model_version: str
    actor_id: str
    event_type: RegistryEventType

    timestamp: datetime = field(
        default_factory=utc_now
    )

    details: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "actor_id": self.actor_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


# ============================================================
# Authorization Protocol
# ============================================================

class RegistryAuthorizer(Protocol):
    def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        ...


class DenyByDefaultRegistryAuthorizer:
    def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        raise RegistryPermissionError(
            "Registry authorization is not configured"
        )


# ============================================================
# Model Registry
# ============================================================

class ModelRegistry:
    """
    Tenant-isolated model registry.

    The registry manages metadata and deployment pointers.
    It does not deserialize model artifacts or execute
    robotic commands.
    """

    def __init__(
        self,
        *,
        storage_directory: str | Path,
        authorizer: RegistryAuthorizer,
        metadata_repository: ModelMetadataRepository,
    ) -> None:
        self.storage_directory = Path(
            storage_directory
        ).resolve()

        self.storage_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.authorizer = authorizer
        self.metadata_repository = metadata_repository

        self._lock = threading.RLock()

        self._records: dict[
            str,
            RegisteredModelRecord,
        ] = {}

        self._aliases: dict[
            str,
            str,
        ] = {}

        self._audit_events: list[
            RegistryAuditEvent
        ] = []

        self._registry_file = (
            self.storage_directory
            / "model_registry.json"
        )

        self._load_registry()

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    def _authorize(
        self,
        scope: RegistryScope,
        permission: str,
    ) -> None:
        self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            permission=permission,
        )

    # --------------------------------------------------------
    # Keys
    # --------------------------------------------------------

    @staticmethod
    def _model_key(
        tenant_id: str,
        model_name: str,
        version: str,
    ) -> str:
        return (
            f"{tenant_id}:"
            f"{model_name}:"
            f"{version}"
        )

    @staticmethod
    def _alias_key(
        tenant_id: str,
        model_name: str,
        alias: DeploymentAlias,
    ) -> str:
        return (
            f"{tenant_id}:"
            f"{model_name}:"
            f"alias:{alias.value}"
        )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    def _load_registry(self) -> None:
        if not self._registry_file.exists():
            return

        with self._lock:
            try:
                if (
                    self._registry_file.stat().st_size
                    > MAX_REGISTRY_FILE_BYTES
                ):
                    raise RegistryStorageError(
                        "Registry file exceeds size limit"
                    )

                data = json.loads(
                    self._registry_file.read_text(
                        encoding="utf-8"
                    )
                )

                self._records = {
                    key: RegisteredModelRecord.from_dict(
                        value
                    )
                    for key, value in data.get(
                        "records", {}
                    ).items()
                }

                self._aliases = dict(
                    data.get("aliases", {})
                )

            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                raise RegistryStorageError(
                    "Unable to load registry"
                ) from exc

    def _save_registry(self) -> None:
        data = {
            "schema_version": (
                REGISTRY_SCHEMA_VERSION
            ),
            "updated_at": utc_now().isoformat(),
            "records": {
                key: record.to_dict()
                for key, record in self._records.items()
            },
            "aliases": dict(self._aliases),
        }

        atomic_write_json(
            self._registry_file,
            data,
        )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    def _audit(
        self,
        *,
        scope: RegistryScope,
        record: RegisteredModelRecord,
        event_type: RegistryEventType,
        details: dict[str, Any] | None = None,
    ) -> None:
        event = RegistryAuditEvent(
            event_id=generate_id(),
            tenant_id=scope.tenant_id,
            model_name=record.model_name,
            model_version=record.model_version,
            actor_id=scope.user_id,
            event_type=event_type,
            details=details or {},
        )

        self._audit_events.append(event)

        logger.info(
            "Registry event=%s tenant=%s model=%s version=%s",
            event_type.value,
            scope.tenant_id,
            record.model_name,
            record.model_version,
        )

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    @staticmethod
    def _validate_model_safety(
        metadata: ModelMetadata,
    ) -> None:
        metadata.safety.validate_for_category(
            metadata.identity.category
        )

        if (
            metadata.identity.category.value
            in ACTION_MODEL_CATEGORIES
            and not metadata.safety.simulation_only
        ):
            raise ModelSafetyError(
                "Robot action models must remain "
                "simulation-only"
            )

        if metadata.safety.physical_execution_authorized:
            raise ModelSafetyError(
                "Model cannot authorize physical actuation"
            )

    # --------------------------------------------------------
    # Register Model
    # --------------------------------------------------------

    def register_model(
        self,
        *,
        metadata: ModelMetadata,
        scope: RegistryScope,
    ) -> RegisteredModelRecord:
        self._authorize(
            scope,
            "ml:registry:register",
        )

        if metadata.ownership.tenant_id != scope.tenant_id:
            raise RegistryPermissionError(
                "Cross-tenant registration is forbidden"
            )

        self._validate_model_safety(metadata)

        model_name = metadata.identity.name
        version = metadata.identity.version

        key = self._model_key(
            scope.tenant_id,
            model_name,
            version,
        )

        with self._lock:
            if key in self._records:
                raise ModelAlreadyRegisteredError(
                    "Model version already registered"
                )

            version_count = sum(
                1
                for record in self._records.values()
                if record.tenant_id == scope.tenant_id
                and record.model_name == model_name
            )

            if version_count >= MAX_MODEL_VERSIONS:
                raise RegistryValidationError(
                    "Maximum model versions reached"
                )

            metadata_path = (
                self.metadata_repository.save(
                    metadata,
                    actor_id=scope.user_id,
                    overwrite=False,
                )
            )

            record = RegisteredModelRecord(
                registry_id=generate_id(),
                tenant_id=scope.tenant_id,
                model_name=model_name,
                model_version=version,
                category=(
                    metadata.identity.category.value
                ),
                framework=metadata.framework.value,
                stage=RegistryStage(
                    metadata.lifecycle_stage.value
                    if metadata.lifecycle_stage.value
                    in RegistryStage._value2member_map_
                    else RegistryStage.DEVELOPMENT.value
                ),
                status=RegistryModelStatus.REGISTERED,
                metadata_path=str(metadata_path),
                artifact_uris=[
                    artifact.uri
                    for artifact in metadata.artifacts
                ],
                artifact_checksums={
                    artifact.artifact_id: artifact.sha256
                    for artifact in metadata.artifacts
                },
                created_by=scope.user_id,
                tags=list(metadata.tags),
            )

            self._records[key] = record

            latest_key = self._alias_key(
                scope.tenant_id,
                model_name,
                DeploymentAlias.LATEST,
            )

            self._aliases[latest_key] = version

            self._save_registry()

            self._audit(
                scope=scope,
                record=record,
                event_type=RegistryEventType.REGISTERED,
            )

            return record

    # --------------------------------------------------------
    # Get Model
    # --------------------------------------------------------

    def get_model(
        self,
        *,
        model_name: str,
        version: str,
        scope: RegistryScope,
    ) -> RegisteredModelRecord:
        self._authorize(
            scope,
            "ml:registry:read",
        )

        key = self._model_key(
            scope.tenant_id,
            model_name,
            version,
        )

        with self._lock:
            record = self._records.get(key)

            if record is None:
                raise ModelNotFoundError(
                    "Model version not found"
                )

            return RegisteredModelRecord.from_dict(
                record.to_dict()
            )

    # --------------------------------------------------------
    # List Models
    # --------------------------------------------------------

    def list_models(
        self,
        *,
        scope: RegistryScope,
        category: str | None = None,
        stage: RegistryStage | None = None,
        model_name: str | None = None,
    ) -> list[RegisteredModelRecord]:
        self._authorize(
            scope,
            "ml:registry:read",
        )

        with self._lock:
            records = [
                record
                for record in self._records.values()
                if record.tenant_id == scope.tenant_id
            ]

            if category is not None:
                records = [
                    record
                    for record in records
                    if record.category == category
                ]

            if stage is not None:
                records = [
                    record
                    for record in records
                    if record.stage == stage
                ]

            if model_name is not None:
                records = [
                    record
                    for record in records
                    if record.model_name == model_name
                ]

            records.sort(
                key=lambda record: (
                    record.model_name,
                    record.model_version,
                )
            )

            return [
                RegisteredModelRecord.from_dict(
                    record.to_dict()
                )
                for record in records
            ]

    # --------------------------------------------------------
    # Get Metadata
    # --------------------------------------------------------

    def get_metadata(
        self,
        *,
        model_name: str,
        version: str,
        scope: RegistryScope,
    ) -> ModelMetadata:
        self.get_model(
            model_name=model_name,
            version=version,
            scope=scope,
        )

        return self.metadata_repository.load(
            tenant_id=scope.tenant_id,
            model_name=model_name,
            version=version,
            actor_id=scope.user_id,
        )

    # --------------------------------------------------------
    # Promote Model
    # --------------------------------------------------------

    def promote_model(
        self,
        *,
        model_name: str,
        version: str,
        target_stage: RegistryStage,
        scope: RegistryScope,
    ) -> RegisteredModelRecord:
        self._authorize(
            scope,
            "ml:registry:promote",
        )

        if target_stage not in {
            RegistryStage.VALIDATION,
            RegistryStage.STAGING,
            RegistryStage.PRODUCTION,
        }:
            raise ModelPromotionError(
                "Invalid promotion target"
            )

        metadata = self.get_metadata(
            model_name=model_name,
            version=version,
            scope=scope,
        )

        self._validate_model_safety(metadata)

        if target_stage == RegistryStage.PRODUCTION:
            metadata.validate_production_readiness()

        key = self._model_key(
            scope.tenant_id,
            model_name,
            version,
        )

        with self._lock:
            record = self._records.get(key)

            if record is None:
                raise ModelNotFoundError(
                    "Model not found"
                )

            if record.stage == RegistryStage.ARCHIVED:
                raise ModelPromotionError(
                    "Archived model cannot be promoted"
                )

            previous_stage = record.stage

            previous_production = self._aliases.get(
                self._alias_key(
                    scope.tenant_id,
                    model_name,
                    DeploymentAlias.PRODUCTION,
                )
            )

            record.stage = target_stage
            record.updated_at = utc_now()
            record.promoted_at = utc_now()

            if target_stage == RegistryStage.STAGING:
                self._aliases[
                    self._alias_key(
                        scope.tenant_id,
                        model_name,
                        DeploymentAlias.STAGING,
                    )
                ] = version

            if target_stage == RegistryStage.PRODUCTION:
                record.previous_production_version = (
                    previous_production
                )

                self._aliases[
                    self._alias_key(
                        scope.tenant_id,
                        model_name,
                        DeploymentAlias.PRODUCTION,
                    )
                ] = version

                self._aliases[
                    self._alias_key(
                        scope.tenant_id,
                        model_name,
                        DeploymentAlias.CHAMPION,
                    )
                ] = version

            self._save_registry()

            self._audit(
                scope=scope,
                record=record,
                event_type=RegistryEventType.PROMOTED,
                details={
                    "from_stage": previous_stage.value,
                    "to_stage": target_stage.value,
                    "previous_production": (
                        previous_production
                    ),
                },
            )

            return RegisteredModelRecord.from_dict(
                record.to_dict()
            )

    # --------------------------------------------------------
    # Resolve Alias
    # --------------------------------------------------------

    def resolve_alias(
        self,
        *,
        model_name: str,
        alias: DeploymentAlias,
        scope: RegistryScope,
    ) -> RegisteredModelRecord:
        self._authorize(
            scope,
            "ml:registry:read",
        )

        alias_key = self._alias_key(
            scope.tenant_id,
            model_name,
            alias,
        )

        with self._lock:
            version = self._aliases.get(
                alias_key
            )

        if version is None:
            raise ModelNotFoundError(
                f"Alias not found: {alias.value}"
            )

        return self.get_model(
            model_name=model_name,
            version=version,
            scope=scope,
        )

    # --------------------------------------------------------
    # Set Alias
    # --------------------------------------------------------

    def set_alias(
        self,
        *,
        model_name: str,
        version: str,
        alias: DeploymentAlias,
        scope: RegistryScope,
    ) -> None:
        self._authorize(
            scope,
            "ml:registry:manage",
        )

        if alias in {
            DeploymentAlias.PRODUCTION,
            DeploymentAlias.CHAMPION,
        }:
            raise RegistryValidationError(
                "Production aliases must be updated "
                "through promotion or rollback"
            )

        key = self._model_key(
            scope.tenant_id,
            model_name,
            version,
        )

        with self._lock:
            record = self._records.get(key)

            if record is None:
                raise ModelNotFoundError(
                    "Model version not found"
                )

            self._aliases[
                self._alias_key(
                    scope.tenant_id,
                    model_name,
                    alias,
                )
            ] = version

            self._save_registry()

            self._audit(
                scope=scope,
                record=record,
                event_type=RegistryEventType.ALIAS_UPDATED,
                details={
                    "alias": alias.value,
                },
            )

    # --------------------------------------------------------
    # Rollback Production
    # --------------------------------------------------------

    def rollback_production(
        self,
        *,
        model_name: str,
        target_version: str,
        scope: RegistryScope,
    ) -> RegisteredModelRecord:
        self._authorize(
            scope,
            "ml:registry:rollback",
        )

        metadata = self.get_metadata(
            model_name=model_name,
            version=target_version,
            scope=scope,
        )

        self._validate_model_safety(metadata)
        metadata.validate_production_readiness()

        key = self._model_key(
            scope.tenant_id,
            model_name,
            target_version,
        )

        with self._lock:
            record = self._records.get(key)

            if record is None:
                raise ModelNotFoundError(
                    "Rollback target not found"
                )

            if record.status in {
                RegistryModelStatus.DISABLED,
                RegistryModelStatus.FAILED,
            }:
                raise ModelRollbackError(
                    "Rollback target is unavailable"
                )

            alias_key = self._alias_key(
                scope.tenant_id,
                model_name,
                DeploymentAlias.PRODUCTION,
            )

            previous_version = self._aliases.get(
                alias_key
            )

            if previous_version == target_version:
                return RegisteredModelRecord.from_dict(
                    record.to_dict()
                )

            self._aliases[alias_key] = target_version

            self._aliases[
                self._alias_key(
                    scope.tenant_id,
                    model_name,
                    DeploymentAlias.CHAMPION,
                )
            ] = target_version

            record.previous_production_version = (
                previous_version
            )

            record.stage = RegistryStage.PRODUCTION
            record.updated_at = utc_now()

            self._save_registry()

            self._audit(
                scope=scope,
                record=record,
                event_type=RegistryEventType.ROLLED_BACK,
                details={
                    "previous_version": previous_version,
                    "target_version": target_version,
                },
            )

            return RegisteredModelRecord.from_dict(
                record.to_dict()
            )

    # --------------------------------------------------------
    # Update Model Status
    # --------------------------------------------------------

    def update_model_status(
        self,
        *,
        model_name: str,
        version: str,
        new_status: RegistryModelStatus,
        scope: RegistryScope,
    ) -> RegisteredModelRecord:
        self._authorize(
            scope,
            "ml:registry:manage",
        )

        key = self._model_key(
            scope.tenant_id,
            model_name,
            version,
        )

        with self._lock:
            record = self._records.get(key)

            if record is None:
                raise ModelNotFoundError(
                    "Model not found"
                )

            old_status = record.status

            record.status = new_status
            record.updated_at = utc_now()

            self._save_registry()

            self._audit(
                scope=scope,
                record=record,
                event_type=RegistryEventType.STATUS_UPDATED,
                details={
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                },
            )

            return RegisteredModelRecord.from_dict(
                record.to_dict()
            )

    # --------------------------------------------------------
    # Archive Model
    # --------------------------------------------------------

    def archive_model(
        self,
        *,
        model_name: str,
        version: str,
        scope: RegistryScope,
    ) -> RegisteredModelRecord:
        self._authorize(
            scope,
            "ml:registry:archive",
        )

        key = self._model_key(
            scope.tenant_id,
            model_name,
            version,
        )

        with self._lock:
            record = self._records.get(key)

            if record is None:
                raise ModelNotFoundError(
                    "Model not found"
                )

            production_key = self._alias_key(
                scope.tenant_id,
                model_name,
                DeploymentAlias.PRODUCTION,
            )

            if self._aliases.get(production_key) == version:
                raise ModelPromotionError(
                    "Cannot archive active production model"
                )

            record.stage = RegistryStage.ARCHIVED
            record.status = RegistryModelStatus.DISABLED
            record.updated_at = utc_now()

            for alias_key, alias_version in list(
                self._aliases.items()
            ):
                if (
                    alias_version == version
                    and alias_key.startswith(
                        f"{scope.tenant_id}:{model_name}:alias:"
                    )
                ):
                    del self._aliases[alias_key]

            self._save_registry()

            self._audit(
                scope=scope,
                record=record,
                event_type=RegistryEventType.ARCHIVED,
            )

            return RegisteredModelRecord.from_dict(
                record.to_dict()
            )

    # --------------------------------------------------------
    # Model Lineage
    # --------------------------------------------------------

    def get_model_lineage(
        self,
        *,
        model_name: str,
        version: str,
        scope: RegistryScope,
    ) -> dict[str, Any]:
        metadata = self.get_metadata(
            model_name=model_name,
            version=version,
            scope=scope,
        )

        return {
            "model_name": model_name,
            "version": version,
            "tenant_id": scope.tenant_id,
            "training_run_id": (
                metadata.training.training_run_id
            ),
            "experiment_id": (
                metadata.training.experiment_id
            ),
            "source_commit": (
                metadata.training.source_commit
            ),
            "datasets": [
                {
                    "dataset_id": dataset.dataset_id,
                    "version": dataset.version,
                    "name": dataset.name,
                    "sha256": dataset.sha256,
                    "split": dataset.split,
                }
                for dataset in metadata.training.datasets
            ],
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "uri": artifact.uri,
                    "sha256": artifact.sha256,
                    "format": artifact.format,
                }
                for artifact in metadata.artifacts
            ],
            "evaluation": {
                "quality_gate_passed": (
                    metadata.evaluation.passed_quality_gate
                ),
                "test_metrics": (
                    metadata.evaluation.test_metrics
                ),
            },
        }

    # --------------------------------------------------------
    # Audit History
    # --------------------------------------------------------

    def get_audit_history(
        self,
        *,
        scope: RegistryScope,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        self._authorize(
            scope,
            "ml:registry:audit",
        )

        with self._lock:
            events = [
                event.to_dict()
                for event in self._audit_events
                if event.tenant_id == scope.tenant_id
                and (
                    model_name is None
                    or event.model_name == model_name
                )
            ]

        return events

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def get_statistics(
        self,
        *,
        scope: RegistryScope,
    ) -> dict[str, Any]:
        records = self.list_models(
            scope=scope
        )

        stage_counts = {
            stage.value: sum(
                record.stage == stage
                for record in records
            )
            for stage in RegistryStage
        }

        category_counts: dict[str, int] = {}

        for record in records:
            category_counts[record.category] = (
                category_counts.get(
                    record.category, 0
                ) + 1
            )

        return {
            "tenant_id": scope.tenant_id,
            "total_model_versions": len(records),
            "unique_models": len(
                {
                    record.model_name
                    for record in records
                }
            ),
            "stage_counts": stage_counts,
            "category_counts": category_counts,
            "production_models": [
                {
                    "name": record.model_name,
                    "version": record.model_version,
                }
                for record in records
                if record.stage
                == RegistryStage.PRODUCTION
            ],
        }

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    def generate_report(
        self,
        *,
        scope: RegistryScope,
    ) -> dict[str, Any]:
        return {
            "generated_at": utc_now().isoformat(),
            "platform": (
                "Physical AI & Autonomous Robotics "
                "Intelligence Platform"
            ),
            "registry_schema_version": (
                REGISTRY_SCHEMA_VERSION
            ),
            "statistics": self.get_statistics(
                scope=scope
            ),
            "models": [
                record.to_dict()
                for record in self.list_models(
                    scope=scope
                )
            ],
            "safety": {
                "physical_robot_actuation": False,
                "emergency_stop_reset": False,
                "safety_interlock_bypass": False,
                "robot_action_models_simulation_only": True,
            },
        }


# ============================================================
# Factory
# ============================================================

def create_model_registry(
    *,
    storage_directory: str | Path,
    metadata_directory: str | Path,
    authorizer: RegistryAuthorizer,
    metadata_authorizer: MetadataAuthorizer,
) -> ModelRegistry:
    metadata_repository = ModelMetadataRepository(
        storage_directory=metadata_directory,
        authorizer=metadata_authorizer,
    )

    return ModelRegistry(
        storage_directory=storage_directory,
        authorizer=authorizer,
        metadata_repository=metadata_repository,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "ModelRegistryError",
    "RegistryValidationError",
    "RegistryPermissionError",
    "ModelAlreadyRegisteredError",
    "ModelNotFoundError",
    "ModelPromotionError",
    "ModelRollbackError",
    "ModelArtifactError",
    "ModelSafetyError",
    "RegistryStorageError",
    "RegistryStage",
    "RegistryModelStatus",
    "DeploymentAlias",
    "RegistryEventType",
    "RegistryScope",
    "RegisteredModelRecord",
    "RegistryAuditEvent",
    "RegistryAuthorizer",
    "DenyByDefaultRegistryAuthorizer",
    "ModelRegistry",
    "create_model_registry",
]
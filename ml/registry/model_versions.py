"""
Physical AI & Autonomous Robotics Intelligence Platform
File: ml/registry/model_versions.py

Model Version Management and Deployment History.

Features:
- Semantic version parsing and comparison
- Major, minor and patch version increments
- Pre-release and build metadata
- Model version registration
- Version history and lineage
- Version compatibility checks
- Model lifecycle transitions
- Staging and production version tracking
- Champion/challenger version selection
- Production rollback planning
- Approval and safety validation
- Tenant-aware version management
- JSON persistence
- Atomic file writes
- Audit history
- Model version comparison
- Version reporting

Python 3.11+

This module manages version records and deployment pointers.
It does not load model artifacts or execute robot commands.
"""

from __future__ import annotations

import copy
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
from functools import total_ordering
from pathlib import Path
from typing import Any, Mapping, Protocol


logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

SCHEMA_VERSION = "1.0.0"

MAX_VERSION_RECORDS = 10000
MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024

SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    r"(?:-("
    r"(?:0|[1-9]\d*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*))*"
    r"))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
)

ACTION_MODEL_CATEGORIES = frozenset({
    "vla",
    "navigation",
    "manipulation",
})


# ============================================================
# Exceptions
# ============================================================

class ModelVersionError(Exception):
    """Base model version exception."""


class InvalidVersionError(ModelVersionError):
    """Invalid semantic version."""


class VersionConflictError(ModelVersionError):
    """Model version already exists."""


class VersionNotFoundError(ModelVersionError):
    """Requested version does not exist."""


class VersionPermissionError(ModelVersionError):
    """Version operation is not authorized."""


class VersionTransitionError(ModelVersionError):
    """Invalid model lifecycle transition."""


class VersionApprovalError(ModelVersionError):
    """Model has not satisfied approval requirements."""


class VersionSafetyError(ModelVersionError):
    """Model version violates robotics safety policy."""


class VersionRollbackError(ModelVersionError):
    """Production rollback cannot be completed."""


class VersionStorageError(ModelVersionError):
    """Model version storage failed."""


# ============================================================
# Utilities
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def validate_identifier(
    value: str,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not IDENTIFIER_PATTERN.fullmatch(value)
    ):
        raise ValueError(
            f"Invalid {field_name}"
        )

    return value


def parse_timestamp(
    value: datetime | str,
) -> datetime:
    if isinstance(value, datetime):
        timestamp = value
    else:
        timestamp = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

    if timestamp.tzinfo is None:
        raise ValueError(
            "Timestamp must include timezone information"
        )

    return timestamp.astimezone(timezone.utc)


# ============================================================
# Semantic Version
# ============================================================

@total_ordering
@dataclass(frozen=True, slots=True, eq=False)
class SemanticVersion:
    major: int
    minor: int
    patch: int

    prerelease: str | None = None
    build: str | None = None

    def __post_init__(self) -> None:
        for name in ("major", "minor", "patch"):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise InvalidVersionError(
                    f"{name} must be a nonnegative integer"
                )

        if not SEMVER_PATTERN.fullmatch(str(self)):
            raise InvalidVersionError(
                "Invalid semantic version"
            )

    @classmethod
    def parse(
        cls,
        version: str,
    ) -> "SemanticVersion":
        if not isinstance(version, str):
            raise InvalidVersionError(
                "Version must be a string"
            )

        match = SEMVER_PATTERN.fullmatch(version)

        if match is None:
            raise InvalidVersionError(
                f"Invalid semantic version: {version}"
            )

        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
            build=match.group(5),
        )

    def __str__(self) -> str:
        result = (
            f"{self.major}."
            f"{self.minor}."
            f"{self.patch}"
        )

        if self.prerelease is not None:
            result += f"-{self.prerelease}"

        if self.build is not None:
            result += f"+{self.build}"

        return result

    @staticmethod
    def _compare_identifiers(
        left: str,
        right: str,
    ) -> int:
        left_numeric = left.isdigit()
        right_numeric = right.isdigit()

        if left_numeric and right_numeric:
            a = int(left)
            b = int(right)
            return (a > b) - (a < b)

        if left_numeric:
            return -1

        if right_numeric:
            return 1

        return (left > right) - (left < right)

    def compare_precedence(
        self,
        other: "SemanticVersion",
    ) -> int:
        if not isinstance(other, SemanticVersion):
            raise TypeError(
                "Expected SemanticVersion"
            )

        left_core = (
            self.major,
            self.minor,
            self.patch,
        )

        right_core = (
            other.major,
            other.minor,
            other.patch,
        )

        if left_core != right_core:
            return (
                (left_core > right_core)
                - (left_core < right_core)
            )

        if (
            self.prerelease is None
            and other.prerelease is None
        ):
            return 0

        if self.prerelease is None:
            return 1

        if other.prerelease is None:
            return -1

        left_parts = self.prerelease.split(".")
        right_parts = other.prerelease.split(".")

        for left, right in zip(
            left_parts,
            right_parts,
        ):
            comparison = self._compare_identifiers(
                left,
                right,
            )

            if comparison:
                return comparison

        return (
            (len(left_parts) > len(right_parts))
            - (len(left_parts) < len(right_parts))
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented

        return self.compare_precedence(other) == 0

    def __lt__(
        self,
        other: "SemanticVersion",
    ) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented

        return self.compare_precedence(other) < 0

    def bump_major(self) -> "SemanticVersion":
        return SemanticVersion(
            self.major + 1,
            0,
            0,
        )

    def bump_minor(self) -> "SemanticVersion":
        return SemanticVersion(
            self.major,
            self.minor + 1,
            0,
        )

    def bump_patch(self) -> "SemanticVersion":
        return SemanticVersion(
            self.major,
            self.minor,
            self.patch + 1,
        )

    def with_prerelease(
        self,
        label: str,
    ) -> "SemanticVersion":
        return SemanticVersion(
            self.major,
            self.minor,
            self.patch,
            prerelease=label,
        )

    def is_prerelease(self) -> bool:
        return self.prerelease is not None

    def is_compatible_with(
        self,
        other: "SemanticVersion",
    ) -> bool:
        """
        Conservative API compatibility convention.

        Major >= 1:
            Same major version.

        Major == 0:
            Same major and minor version.

        This is a policy heuristic, not proof that model
        tensor schemas or output semantics are compatible.
        """
        if self.major != other.major:
            return False

        if self.major == 0:
            return self.minor == other.minor

        return True


# ============================================================
# Enumerations
# ============================================================

class VersionStage(str, Enum):
    DEVELOPMENT = "development"
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class VersionStatus(str, Enum):
    REGISTERED = "registered"
    READY = "ready"
    DEGRADED = "degraded"
    DISABLED = "disabled"
    FAILED = "failed"


class VersionChangeType(str, Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    PRERELEASE = "prerelease"


class VersionAlias(str, Enum):
    LATEST = "latest"
    STAGING = "staging"
    PRODUCTION = "production"
    CHAMPION = "champion"
    CHALLENGER = "challenger"


class VersionEventType(str, Enum):
    REGISTERED = "registered"
    STAGE_CHANGED = "stage_changed"
    STATUS_CHANGED = "status_changed"
    APPROVED = "approved"
    REVOKED = "revoked"
    ALIAS_UPDATED = "alias_updated"
    ROLLED_BACK = "rolled_back"
    ARCHIVED = "archived"


# ============================================================
# Scope and Authorization
# ============================================================

@dataclass(frozen=True, slots=True)
class VersionScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.user_id,
            "user_id",
        )


class VersionAuthorizer(Protocol):
    def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        ...


class DenyByDefaultVersionAuthorizer:
    def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        raise VersionPermissionError(
            "Version authorization provider is not configured"
        )


# ============================================================
# Version Record
# ============================================================

@dataclass(slots=True)
class ModelVersionRecord:
    version_id: str
    tenant_id: str
    model_name: str
    version: str
    category: str

    stage: VersionStage = VersionStage.DEVELOPMENT
    status: VersionStatus = VersionStatus.REGISTERED

    artifact_uri: str | None = None
    artifact_sha256: str | None = None

    source_commit: str | None = None
    training_run_id: str | None = None
    parent_version: str | None = None

    input_schema_hash: str | None = None
    output_schema_hash: str | None = None

    quality_gate_passed: bool = False
    safety_evaluation_passed: bool = False

    approved: bool = False
    approved_by: str | None = None
    approved_at: datetime | None = None

    simulation_only: bool = True

    physical_execution_authorized: bool = False
    emergency_stop_reset_authorized: bool = False
    safety_interlock_bypass_authorized: bool = False

    metrics: dict[str, float] = field(
        default_factory=dict
    )

    tags: list[str] = field(
        default_factory=list
    )

    created_by: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    description: str = ""

    def __post_init__(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.model_name,
            "model_name",
        )

        SemanticVersion.parse(
            self.version
        )

        if self.parent_version is not None:
            SemanticVersion.parse(
                self.parent_version
            )

        if not isinstance(self.category, str) or not self.category:
            raise ValueError(
                "Model category is required"
            )

        if (
            self.category in ACTION_MODEL_CATEGORIES
            and not self.simulation_only
        ):
            raise VersionSafetyError(
                "Robot action models must remain simulation-only"
            )

        if (
            self.physical_execution_authorized
            or self.emergency_stop_reset_authorized
            or self.safety_interlock_bypass_authorized
        ):
            raise VersionSafetyError(
                "Model versions cannot authorize "
                "physical actuation or bypass safety systems"
            )

        for name, value in self.metrics.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"Invalid model metric: {name}"
                )

        self.created_at = parse_timestamp(
            self.created_at
        )

        self.updated_at = parse_timestamp(
            self.updated_at
        )

        if self.approved_at is not None:
            self.approved_at = parse_timestamp(
                self.approved_at
            )

    @property
    def key(self) -> str:
        return (
            f"{self.tenant_id}:"
            f"{self.model_name}:"
            f"{self.version}"
        )

    def validate_production_readiness(self) -> None:
        if not self.approved:
            raise VersionApprovalError(
                "Production requires human approval"
            )

        if not self.quality_gate_passed:
            raise VersionApprovalError(
                "Model quality gate has not passed"
            )

        if not self.safety_evaluation_passed:
            raise VersionSafetyError(
                "Model safety evaluation has not passed"
            )

        if not self.artifact_uri:
            raise VersionTransitionError(
                "Production requires a model artifact"
            )

        if self.status not in {
            VersionStatus.READY,
            VersionStatus.DEGRADED,
        }:
            raise VersionTransitionError(
                "Model is not ready for deployment"
            )

        if (
            self.category in ACTION_MODEL_CATEGORIES
            and not self.simulation_only
        ):
            raise VersionSafetyError(
                "Robotic action inference must be simulation-only"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "tenant_id": self.tenant_id,
            "model_name": self.model_name,
            "version": self.version,
            "category": self.category,
            "stage": self.stage.value,
            "status": self.status.value,
            "artifact_uri": self.artifact_uri,
            "artifact_sha256": self.artifact_sha256,
            "source_commit": self.source_commit,
            "training_run_id": self.training_run_id,
            "parent_version": self.parent_version,
            "input_schema_hash": self.input_schema_hash,
            "output_schema_hash": self.output_schema_hash,
            "quality_gate_passed": self.quality_gate_passed,
            "safety_evaluation_passed": (
                self.safety_evaluation_passed
            ),
            "approved": self.approved,
            "approved_by": self.approved_by,
            "approved_at": (
                self.approved_at.isoformat()
                if self.approved_at
                else None
            ),
            "simulation_only": self.simulation_only,
            "physical_execution_authorized": False,
            "emergency_stop_reset_authorized": False,
            "safety_interlock_bypass_authorized": False,
            "metrics": dict(self.metrics),
            "tags": list(self.tags),
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "description": self.description,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ModelVersionRecord":
        values = dict(data)

        values["stage"] = VersionStage(
            values.get("stage", "development")
        )

        values["status"] = VersionStatus(
            values.get("status", "registered")
        )

        for key in (
            "created_at",
            "updated_at",
            "approved_at",
        ):
            if values.get(key) is not None:
                values[key] = parse_timestamp(
                    values[key]
                )

        return cls(**values)


# ============================================================
# Version Audit Event
# ============================================================

@dataclass(slots=True)
class VersionAuditEvent:
    event_id: str
    tenant_id: str
    model_name: str
    version: str
    actor_id: str
    event_type: VersionEventType

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
            "version": self.version,
            "actor_id": self.actor_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "details": copy.deepcopy(self.details),
        }


# ============================================================
# Version Compatibility
# ============================================================

@dataclass(frozen=True, slots=True)
class VersionCompatibilityResult:
    compatible: bool
    semantic_compatible: bool
    input_schema_compatible: bool
    output_schema_compatible: bool
    reasons: tuple[str, ...]


def check_version_compatibility(
    current: ModelVersionRecord,
    candidate: ModelVersionRecord,
) -> VersionCompatibilityResult:
    reasons: list[str] = []

    same_model = (
        current.tenant_id == candidate.tenant_id
        and current.model_name == candidate.model_name
    )

    if not same_model:
        reasons.append(
            "Versions belong to different models or tenants"
        )

    current_semver = SemanticVersion.parse(
        current.version
    )

    candidate_semver = SemanticVersion.parse(
        candidate.version
    )

    semantic_compatible = (
        candidate_semver.is_compatible_with(
            current_semver
        )
    )

    if not semantic_compatible:
        reasons.append(
            "Semantic versions are not compatible"
        )

    input_compatible = (
        current.input_schema_hash is not None
        and candidate.input_schema_hash is not None
        and current.input_schema_hash
        == candidate.input_schema_hash
    )

    output_compatible = (
        current.output_schema_hash is not None
        and candidate.output_schema_hash is not None
        and current.output_schema_hash
        == candidate.output_schema_hash
    )

    if not input_compatible:
        reasons.append(
            "Input schema compatibility is unverified "
            "or the schema changed"
        )

    if not output_compatible:
        reasons.append(
            "Output schema compatibility is unverified "
            "or the schema changed"
        )

    compatible = (
        same_model
        and semantic_compatible
        and input_compatible
        and output_compatible
    )

    return VersionCompatibilityResult(
        compatible=compatible,
        semantic_compatible=semantic_compatible,
        input_schema_compatible=input_compatible,
        output_schema_compatible=output_compatible,
        reasons=tuple(reasons),
    )


# ============================================================
# Version Manager
# ============================================================

class ModelVersionManager:
    """
    Persistent model version and deployment alias manager.

    Suitable for local development or a single writer.

    Production multi-instance deployments should use a
    transactional database, distributed locking, and an
    append-only audit store.
    """

    def __init__(
        self,
        *,
        storage_directory: str | Path,
        authorizer: VersionAuthorizer,
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

        self._records: dict[
            str,
            ModelVersionRecord,
        ] = {}

        self._aliases: dict[str, str] = {}

        self._events: list[
            VersionAuditEvent
        ] = []

        self._storage_path = (
            self.storage_directory
            / "model_versions.json"
        )

        self._load()

    @staticmethod
    def _key(
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
        alias: VersionAlias,
    ) -> str:
        return (
            f"{tenant_id}:"
            f"{model_name}:"
            f"alias:{alias.value}"
        )

    def _authorize(
        self,
        scope: VersionScope,
        permission: str,
    ) -> None:
        self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            permission=permission,
        )

    def _load(self) -> None:
        if not self._storage_path.exists():
            return

        with self._lock:
            try:
                if (
                    self._storage_path.stat().st_size
                    > MAX_FILE_SIZE_BYTES
                ):
                    raise VersionStorageError(
                        "Version database exceeds size limit"
                    )

                data = json.loads(
                    self._storage_path.read_text(
                        encoding="utf-8"
                    )
                )

                self._records = {
                    key: ModelVersionRecord.from_dict(
                        value
                    )
                    for key, value in data.get(
                        "records", {}
                    ).items()
                }

                self._aliases = dict(
                    data.get("aliases", {})
                )

                self._events = [
                    VersionAuditEvent(
                        event_id=item["event_id"],
                        tenant_id=item["tenant_id"],
                        model_name=item["model_name"],
                        version=item["version"],
                        actor_id=item["actor_id"],
                        event_type=VersionEventType(
                            item["event_type"]
                        ),
                        timestamp=parse_timestamp(
                            item["timestamp"]
                        ),
                        details=dict(
                            item.get("details", {})
                        ),
                    )
                    for item in data.get(
                        "events", []
                    )
                ]

            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                raise VersionStorageError(
                    "Failed to load model versions"
                ) from exc

    def _save(self) -> None:
        data = {
            "schema_version": SCHEMA_VERSION,
            "updated_at": utc_now().isoformat(),
            "records": {
                key: record.to_dict()
                for key, record in self._records.items()
            },
            "aliases": dict(self._aliases),
            "events": [
                event.to_dict()
                for event in self._events
            ],
        }

        payload = json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

        if len(payload.encode("utf-8")) > MAX_FILE_SIZE_BYTES:
            raise VersionStorageError(
                "Version database exceeds size limit"
            )

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.storage_directory,
                prefix=".versions-",
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
                self._storage_path,
            )

        except OSError as exc:
            raise VersionStorageError(
                "Unable to persist model versions"
            ) from exc

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

    def _audit(
        self,
        *,
        scope: VersionScope,
        record: ModelVersionRecord,
        event_type: VersionEventType,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._events.append(
            VersionAuditEvent(
                event_id=generate_id(),
                tenant_id=scope.tenant_id,
                model_name=record.model_name,
                version=record.version,
                actor_id=scope.user_id,
                event_type=event_type,
                details=details or {},
            )
        )

    def register_version(
        self,
        *,
        record: ModelVersionRecord,
        scope: VersionScope,
    ) -> ModelVersionRecord:
        self._authorize(
            scope,
            "ml:versions:register",
        )

        if record.tenant_id != scope.tenant_id:
            raise VersionPermissionError(
                "Cross-tenant registration denied"
            )

        key = record.key

        with self._lock:
            if key in self._records:
                raise VersionConflictError(
                    "Model version already registered"
                )

            if len(self._records) >= MAX_VERSION_RECORDS:
                raise VersionStorageError(
                    "Version record limit reached"
                )

            if record.parent_version is not None:
                parent_key = self._key(
                    scope.tenant_id,
                    record.model_name,
                    record.parent_version,
                )

                if parent_key not in self._records:
                    raise VersionNotFoundError(
                        "Parent model version not found"
                    )

            self._records[key] = copy.deepcopy(record)

            latest_alias = self._alias_key(
                scope.tenant_id,
                record.model_name,
                VersionAlias.LATEST,
            )

            previous_latest = self._aliases.get(
                latest_alias
            )

            if previous_latest is None:
                self._aliases[
                    latest_alias
                ] = record.version

            else:
                new_version = SemanticVersion.parse(
                    record.version
                )

                old_version = SemanticVersion.parse(
                    previous_latest
                )

                if new_version > old_version:
                    self._aliases[
                        latest_alias
                    ] = record.version

            self._audit(
                scope=scope,
                record=record,
                event_type=VersionEventType.REGISTERED,
            )

            self._save()

            return copy.deepcopy(record)

    def get_version(
        self,
        *,
        model_name: str,
        version: str,
        scope: VersionScope,
    ) -> ModelVersionRecord:
        self._authorize(
            scope,
            "ml:versions:read",
        )

        key = self._key(
            scope.tenant_id,
            model_name,
            version,
        )

        with self._lock:
            record = self._records.get(key)

            if record is None:
                raise VersionNotFoundError(
                    "Model version not found"
                )

            return copy.deepcopy(record)

    def list_versions(
        self,
        *,
        model_name: str,
        scope: VersionScope,
        include_archived: bool = False,
    ) -> list[ModelVersionRecord]:
        self._authorize(
            scope,
            "ml:versions:read",
        )

        with self._lock:
            records = [
                copy.deepcopy(record)
                for record in self._records.values()
                if record.tenant_id == scope.tenant_id
                and record.model_name == model_name
                and (
                    include_archived
                    or record.stage
                    != VersionStage.ARCHIVED
                )
            ]

        records.sort(
            key=lambda record: (
                SemanticVersion.parse(record.version),
                record.version,
            ),
            reverse=True,
        )

        return records

    def get_latest_version(
        self,
        *,
        model_name: str,
        scope: VersionScope,
    ) -> ModelVersionRecord:
        versions = self.list_versions(
            model_name=model_name,
            scope=scope,
        )

        if not versions:
            raise VersionNotFoundError(
                "No model versions found"
            )

        return versions[0]

    def suggest_next_version(
        self,
        *,
        model_name: str,
        change_type: VersionChangeType,
        scope: VersionScope,
    ) -> str:
        try:
            latest = self.get_latest_version(
                model_name=model_name,
                scope=scope,
            )

            version = SemanticVersion.parse(
                latest.version
            )

        except VersionNotFoundError:
            return "0.1.0"

        if change_type == VersionChangeType.MAJOR:
            return str(version.bump_major())

        if change_type == VersionChangeType.MINOR:
            return str(version.bump_minor())

        if change_type == VersionChangeType.PATCH:
            return str(version.bump_patch())

        return str(
            version.bump_patch().with_prerelease(
                "rc.1"
            )
        )

    def approve_version(
        self,
        *,
        model_name: str,
        version: str,
        scope: VersionScope,
    ) -> ModelVersionRecord:
        self._authorize(
            scope,
            "ml:versions:approve",
        )

        key = self._key(
            scope.tenant_id,
            model_name,
            version,
        )

        with self._lock:
            record = self._records.get(key)

            if record is None:
                raise VersionNotFoundError(
                    "Model version not found"
                )

            if not record.quality_gate_passed:
                raise VersionApprovalError(
                    "Model quality gate has not passed"
                )

            if not record.safety_evaluation_passed:
                raise VersionSafetyError(
                    "Model safety evaluation has not passed"
                )

            record.approved = True
            record.approved_by = scope.user_id
            record.approved_at = utc_now()
            record.updated_at = utc_now()

            self._audit(
                scope=scope,
                record=record,
                event_type=VersionEventType.APPROVED,
            )

            self._save()

            return copy.deepcopy(record)

    def revoke_approval(
        self,
        *,
        model_name: str,
        version: str,
        scope: VersionScope,
    ) -> ModelVersionRecord:
        self._authorize(
            scope,
            "ml:versions:approve",
        )

        key = self._key(
            scope.tenant_id,
            model_name,
            version,
        )

        with self._lock:
            record = self._records.get(key)

            if record is None:
                raise VersionNotFoundError(
                    "Model version not found"
                )

            record.approved = False
            record.approved_by = None
            record.approved_at = None
            record.updated_at = utc_now()

            self._audit(
                scope=scope,
                record=record,
                event_type=VersionEventType.REVOKED,
            )

            self._save()

            return copy.deepcopy(record)

    def transition_stage(
        self,
        *,
        model_name: str,
        version: str,
        target_stage: VersionStage,
        scope: VersionScope,
    ) -> ModelVersionRecord:
        self._authorize(
            scope,
            "ml:versions:promote",
        )

        key = self._key(
            scope.tenant_id,
            model_name,
            version,
        )

        allowed_transitions = {
            VersionStage.DEVELOPMENT: {
                VersionStage.TRAINING,
                VersionStage.VALIDATION,
                VersionStage.ARCHIVED,
            },
            VersionStage.TRAINING: {
                VersionStage.VALIDATION,
                VersionStage.ARCHIVED,
            },
            VersionStage.VALIDATION: {
                VersionStage.STAGING,
                VersionStage.ARCHIVED,
            },
            VersionStage.STAGING: {
                VersionStage.PRODUCTION,
                VersionStage.ARCHIVED,
            },
            VersionStage.PRODUCTION: {
                VersionStage.ARCHIVED,
            },
            VersionStage.ARCHIVED: set(),
        }

        with self._lock:
            record = self._records.get(key)

            if record is None:
                raise VersionNotFoundError(
                    "Model version not found"
                )

            if target_stage == record.stage:
                return copy.deepcopy(record)

            if target_stage not in allowed_transitions[
                record.stage
            ]:
                raise VersionTransitionError(
                    f"Invalid transition: "
                    f"{record.stage.value} -> "
                    f"{target_stage.value}"
                )

            if target_stage == VersionStage.PRODUCTION:
                record.validate_production_readiness()

            if target_stage == VersionStage.STAGING:
                if not record.quality_gate_passed:
                    raise VersionApprovalError(
                        "Staging requires a passed quality gate"
                    )

            if target_stage == VersionStage.ARCHIVED:
                production_alias = self._alias_key(
                    scope.tenant_id,
                    model_name,
                    VersionAlias.PRODUCTION,
                )

                if self._aliases.get(
                    production_alias
                ) == version:
                    raise VersionTransitionError(
                        "Cannot archive active production version"
                    )

            previous_stage = record.stage

            record.stage = target_stage
            record.updated_at = utc_now()

            if target_stage == VersionStage.STAGING:
                self._aliases[
                    self._alias_key(
                        scope.tenant_id,
                        model_name,
                        VersionAlias.STAGING,
                    )
                ] = version

            if target_stage == VersionStage.PRODUCTION:
                self._aliases[
                    self._alias_key(
                        scope.tenant_id,
                        model_name,
                        VersionAlias.PRODUCTION,
                    )
                ] = version

                self._aliases[
                    self._alias_key(
                        scope.tenant_id,
                        model_name,
                        VersionAlias.CHAMPION,
                    )
                ] = version

            if target_stage == VersionStage.ARCHIVED:
                for alias_key, alias_version in list(
                    self._aliases.items()
                ):
                    if (
                        alias_version == version
                        and alias_key.startswith(
                            f"{scope.tenant_id}:"
                            f"{model_name}:alias:"
                        )
                    ):
                        del self._aliases[alias_key]

            self._audit(
                scope=scope,
                record=record,
                event_type=(
                    VersionEventType.ARCHIVED
                    if target_stage == VersionStage.ARCHIVED
                    else VersionEventType.STAGE_CHANGED
                ),
                details={
                    "previous_stage": previous_stage.value,
                    "target_stage": target_stage.value,
                },
            )

            self._save()

            return copy.deepcopy(record)

    def resolve_alias(
        self,
        *,
        model_name: str,
        alias: VersionAlias,
        scope: VersionScope,
    ) -> ModelVersionRecord:
        self._authorize(
            scope,
            "ml:versions:read",
        )

        alias_key = self._alias_key(
            scope.tenant_id,
            model_name,
            alias,
        )

        with self._lock:
            version = self._aliases.get(alias_key)

        if version is None:
            raise VersionNotFoundError(
                f"Alias not found: {alias.value}"
            )

        return self.get_version(
            model_name=model_name,
            version=version,
            scope=scope,
        )

    def set_challenger(
        self,
        *,
        model_name: str,
        version: str,
        scope: VersionScope,
    ) -> ModelVersionRecord:
        self._authorize(
            scope,
            "ml:versions:manage",
        )

        key = self._key(
            scope.tenant_id,
            model_name,
            version,
        )

        with self._lock:
            record = self._records.get(key)

            if record is None:
                raise VersionNotFoundError(
                    "Challenger model not found"
                )

            if record.stage not in {
                VersionStage.STAGING,
                VersionStage.PRODUCTION,
            }:
                raise VersionTransitionError(
                    "Challenger must be staged or production-ready"
                )

            self._aliases[
                self._alias_key(
                    scope.tenant_id,
                    model_name,
                    VersionAlias.CHALLENGER,
                )
            ] = version

            self._audit(
                scope=scope,
                record=record,
                event_type=VersionEventType.ALIAS_UPDATED,
                details={
                    "alias": VersionAlias.CHALLENGER.value,
                },
            )

            self._save()

            return copy.deepcopy(record)

    def rollback_production(
        self,
        *,
        model_name: str,
        target_version: str,
        scope: VersionScope,
    ) -> ModelVersionRecord:
        self._authorize(
            scope,
            "ml:versions:rollback",
        )

        key = self._key(
            scope.tenant_id,
            model_name,
            target_version,
        )

        with self._lock:
            target = self._records.get(key)

            if target is None:
                raise VersionNotFoundError(
                    "Rollback target not found"
                )

            target.validate_production_readiness()

            if target.stage not in {
                VersionStage.STAGING,
                VersionStage.PRODUCTION,
            }:
                raise VersionRollbackError(
                    "Rollback target must be staged "
                    "or previously deployed"
                )

            alias_key = self._alias_key(
                scope.tenant_id,
                model_name,
                VersionAlias.PRODUCTION,
            )

            previous_version = self._aliases.get(
                alias_key
            )

            if previous_version == target_version:
                return copy.deepcopy(target)

            self._aliases[
                alias_key
            ] = target_version

            self._aliases[
                self._alias_key(
                    scope.tenant_id,
                    model_name,
                    VersionAlias.CHAMPION,
                )
            ] = target_version

            target.stage = VersionStage.PRODUCTION
            target.updated_at = utc_now()

            self._audit(
                scope=scope,
                record=target,
                event_type=VersionEventType.ROLLED_BACK,
                details={
                    "previous_version": previous_version,
                    "target_version": target_version,
                },
            )

            self._save()

            return copy.deepcopy(target)

    def update_status(
        self,
        *,
        model_name: str,
        version: str,
        new_status: VersionStatus,
        scope: VersionScope,
    ) -> ModelVersionRecord:
        self._authorize(
            scope,
            "ml:versions:manage",
        )

        key = self._key(
            scope.tenant_id,
            model_name,
            version,
        )

        with self._lock:
            record = self._records.get(key)

            if record is None:
                raise VersionNotFoundError(
                    "Model version not found"
                )

            old_status = record.status

            record.status = new_status
            record.updated_at = utc_now()

            self._audit(
                scope=scope,
                record=record,
                event_type=VersionEventType.STATUS_CHANGED,
                details={
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                },
            )

            self._save()

            return copy.deepcopy(record)

    def get_lineage(
        self,
        *,
        model_name: str,
        version: str,
        scope: VersionScope,
    ) -> list[dict[str, Any]]:
        self._authorize(
            scope,
            "ml:versions:read",
        )

        lineage = []
        visited = set()

        current_version: str | None = version

        while current_version is not None:
            if current_version in visited:
                raise VersionStorageError(
                    "Model version lineage cycle detected"
                )

            visited.add(current_version)

            record = self.get_version(
                model_name=model_name,
                version=current_version,
                scope=scope,
            )

            lineage.append(
                {
                    "version": record.version,
                    "parent_version": record.parent_version,
                    "training_run_id": record.training_run_id,
                    "source_commit": record.source_commit,
                    "created_at": (
                        record.created_at.isoformat()
                    ),
                }
            )

            current_version = record.parent_version

        return lineage

    def compare_versions(
        self,
        *,
        model_name: str,
        first_version: str,
        second_version: str,
        scope: VersionScope,
    ) -> dict[str, Any]:
        first = self.get_version(
            model_name=model_name,
            version=first_version,
            scope=scope,
        )

        second = self.get_version(
            model_name=model_name,
            version=second_version,
            scope=scope,
        )

        compatibility = check_version_compatibility(
            first,
            second,
        )

        metric_names = (
            set(first.metrics)
            | set(second.metrics)
        )

        metrics = {}

        for name in sorted(metric_names):
            first_value = first.metrics.get(name)
            second_value = second.metrics.get(name)

            delta = None

            if (
                first_value is not None
                and second_value is not None
            ):
                delta = second_value - first_value

            metrics[name] = {
                "first": first_value,
                "second": second_value,
                "delta": delta,
            }

        return {
            "model_name": model_name,
            "first_version": first_version,
            "second_version": second_version,
            "compatibility": {
                "compatible": compatibility.compatible,
                "semantic_compatible": (
                    compatibility.semantic_compatible
                ),
                "input_schema_compatible": (
                    compatibility.input_schema_compatible
                ),
                "output_schema_compatible": (
                    compatibility.output_schema_compatible
                ),
                "reasons": list(compatibility.reasons),
            },
            "metrics": metrics,
            "approval": {
                "first": first.approved,
                "second": second.approved,
            },
            "safety": {
                "first_passed": (
                    first.safety_evaluation_passed
                ),
                "second_passed": (
                    second.safety_evaluation_passed
                ),
                "simulation_only": (
                    first.simulation_only
                    and second.simulation_only
                ),
            },
        }

    def get_audit_history(
        self,
        *,
        scope: VersionScope,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        self._authorize(
            scope,
            "ml:versions:audit",
        )

        with self._lock:
            return [
                event.to_dict()
                for event in self._events
                if event.tenant_id == scope.tenant_id
                and (
                    model_name is None
                    or event.model_name == model_name
                )
            ]

    def get_statistics(
        self,
        *,
        scope: VersionScope,
    ) -> dict[str, Any]:
        self._authorize(
            scope,
            "ml:versions:read",
        )

        with self._lock:
            records = [
                record
                for record in self._records.values()
                if record.tenant_id == scope.tenant_id
            ]

            stage_counts = {
                stage.value: sum(
                    record.stage == stage
                    for record in records
                )
                for stage in VersionStage
            }

            return {
                "tenant_id": scope.tenant_id,
                "total_versions": len(records),
                "unique_models": len({
                    record.model_name
                    for record in records
                }),
                "approved_versions": sum(
                    record.approved
                    for record in records
                ),
                "stage_counts": stage_counts,
                "active_production_aliases": sum(
                    1
                    for key in self._aliases
                    if key.startswith(
                        f"{scope.tenant_id}:"
                    )
                    and key.endswith(
                        ":alias:production"
                    )
                ),
            }

    def generate_report(
        self,
        *,
        scope: VersionScope,
    ) -> dict[str, Any]:
        return {
            "generated_at": utc_now().isoformat(),
            "platform": (
                "Physical AI & Autonomous Robotics "
                "Intelligence Platform"
            ),
            "schema_version": SCHEMA_VERSION,
            "statistics": self.get_statistics(
                scope=scope
            ),
            "safety": {
                "physical_actuation_authorized": False,
                "emergency_stop_reset_authorized": False,
                "safety_interlock_bypass_authorized": False,
                "robot_action_models_simulation_only": True,
            },
        }


# ============================================================
# Factory
# ============================================================

def create_model_version_manager(
    *,
    storage_directory: str | Path,
    authorizer: VersionAuthorizer,
) -> ModelVersionManager:
    return ModelVersionManager(
        storage_directory=storage_directory,
        authorizer=authorizer,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "ModelVersionError",
    "InvalidVersionError",
    "VersionConflictError",
    "VersionNotFoundError",
    "VersionPermissionError",
    "VersionTransitionError",
    "VersionApprovalError",
    "VersionSafetyError",
    "VersionRollbackError",
    "VersionStorageError",
    "SemanticVersion",
    "VersionStage",
    "VersionStatus",
    "VersionChangeType",
    "VersionAlias",
    "VersionEventType",
    "VersionScope",
    "VersionAuthorizer",
    "DenyByDefaultVersionAuthorizer",
    "ModelVersionRecord",
    "VersionAuditEvent",
    "VersionCompatibilityResult",
    "check_version_compatibility",
    "ModelVersionManager",
    "create_model_version_manager",
]
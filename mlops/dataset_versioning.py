"""Dataset Versioning module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/mlops/dataset_versioning.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Dataset Versioning and Integrity Management

Features:
    - Immutable dataset versions
    - SHA-256 file hashing
    - Content-addressed object storage
    - Dataset registration and version creation
    - Dataset manifests and metadata
    - Dataset lineage and parent versions
    - Version comparison
    - Dataset integrity verification
    - Version restoration and checkout
    - Dataset tags
    - Dataset statistics
    - Multi-tenant access control
    - Thread-safe dataset management
    - Atomic metadata persistence
    - Safe file path validation
    - JSON export
    - Synthetic examples
    - Unit tests

Supported datasets:
    - Robot camera images
    - LiDAR point clouds
    - RGB-D datasets
    - Sensor telemetry
    - Robot navigation datasets
    - SLAM datasets
    - Robot manipulation datasets
    - Reinforcement learning datasets
    - Robot inspection images
    - Predictive maintenance datasets
    - Training, validation and test datasets

Python 3.11+
Standard library only.

Security:
    - Dataset objects are stored using content hashes.
    - Version manifests are immutable after creation.
    - Source files are copied into managed storage.
    - Checkout destinations are validated.
    - Tenant authorization is enforced.
    - Hash verification detects unexpected modifications.

Important:
    This module manages dataset artifacts. It does not
    certify dataset safety, provenance, licensing, quality
    or suitability for real-world robot control.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import unittest
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class DatasetVersioningError(Exception):
    pass


class DatasetValidationError(DatasetVersioningError):
    pass


class DatasetNotFoundError(DatasetVersioningError):
    pass


class DatasetVersionNotFoundError(DatasetVersioningError):
    pass


class DatasetAuthorizationError(DatasetVersioningError):
    pass


class DatasetIntegrityError(DatasetVersioningError):
    pass


class DatasetConflictError(DatasetVersioningError):
    pass


class DatasetCapacityError(DatasetVersioningError):
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
        or len(value) > 128
    ):
        raise DatasetValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def validate_relative_path(
    value: str,
) -> str:
    if not isinstance(value, str):
        raise DatasetValidationError(
            "Dataset path must be a string"
        )

    if (
        not value
        or "\x00" in value
        or "\\" in value
    ):
        raise DatasetValidationError(
            "Invalid dataset relative path"
        )

    path = Path(value)

    if (
        path.is_absolute()
        or any(
            part in ("", ".", "..")
            for part in value.split("/")
        )
        or ":" in value.split("/")[0]
    ):
        raise DatasetValidationError(
            "Dataset path must be relative and cannot traverse directories"
        )

    return path.as_posix()


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(data: Any) -> bytes:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def atomic_write_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".dataset-",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                data,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )

            stream.flush()
            os.fsync(stream.fileno())

        os.replace(
            temporary_name,
            path,
        )

    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "__dataclass_fields__"):
        return {
            name: json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list, set, frozenset)):
        return [
            json_safe(item)
            for item in value
        ]

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class DatasetType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    LIDAR = "lidar"
    RGBD = "rgbd"
    SLAM = "slam"
    SENSOR = "sensor"
    TELEMETRY = "telemetry"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    INSPECTION = "inspection"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    MULTIMODAL = "multimodal"
    GENERAL = "general"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    UNSPECIFIED = "unspecified"


class DatasetVersionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class DatasetChangeType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DatasetVersioningConfig:
    storage_root: Path = Path("./data/dataset_registry")

    maximum_files_per_version: int = 100000
    maximum_file_size_bytes: int = 10 * 1024**3
    maximum_dataset_size_bytes: int = 100 * 1024**3

    maximum_versions_per_dataset: int = 10000

    verify_objects_on_checkout: bool = True
    reject_symbolic_links: bool = True

    def __post_init__(self) -> None:
        root = Path(self.storage_root).expanduser().resolve()

        object.__setattr__(
            self,
            "storage_root",
            root,
        )

        for name in (
            "maximum_files_per_version",
            "maximum_file_size_bytes",
            "maximum_dataset_size_bytes",
            "maximum_versions_per_dataset",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise DatasetValidationError(
                    f"{name} must be a positive integer"
                )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class DatasetFile:
    relative_path: str
    sha256: str
    size_bytes: int

    split: DatasetSplit = DatasetSplit.UNSPECIFIED

    def __post_init__(self) -> None:
        validate_relative_path(
            self.relative_path
        )

        if (
            len(self.sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.sha256
            )
        ):
            raise DatasetValidationError(
                "Invalid SHA-256 hash"
            )

        if self.size_bytes < 0:
            raise DatasetValidationError(
                "File size cannot be negative"
            )


@dataclass(frozen=True)
class DatasetVersion:
    dataset_id: str
    version_id: str
    version_number: int

    tenant_id: str

    parent_version_id: str | None

    files: tuple[DatasetFile, ...]

    manifest_hash: str

    created_at: datetime
    created_by: str

    description: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    status: DatasetVersionStatus = (
        DatasetVersionStatus.ACTIVE
    )

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_size_bytes(self) -> int:
        return sum(
            item.size_bytes
            for item in self.files
        )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    tenant_id: str

    name: str
    dataset_type: DatasetType

    created_at: datetime
    created_by: str

    description: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class DatasetFileChange:
    relative_path: str

    change_type: DatasetChangeType

    old_sha256: str | None
    new_sha256: str | None

    old_size_bytes: int | None
    new_size_bytes: int | None


@dataclass(frozen=True)
class DatasetVersionDiff:
    dataset_id: str

    source_version_id: str
    target_version_id: str

    changes: tuple[DatasetFileChange, ...]

    added_count: int
    removed_count: int
    modified_count: int
    unchanged_count: int


@dataclass(frozen=True)
class DatasetIntegrityReport:
    dataset_id: str
    version_id: str

    valid: bool

    checked_files: int

    missing_objects: tuple[str, ...]
    corrupted_objects: tuple[str, ...]

    manifest_valid: bool

    checked_at: datetime


@dataclass(frozen=True)
class DatasetStatistics:
    dataset_id: str
    version_id: str

    total_files: int
    total_size_bytes: int

    file_extensions: dict[str, int]
    split_counts: dict[str, int]

    unique_content_objects: int


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class DatasetAccessContext:
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
# CONTENT-ADDRESSED STORAGE
# ============================================================

class ContentAddressedObjectStore:

    def __init__(
        self,
        config: DatasetVersioningConfig,
    ) -> None:
        self.config = config

        self.root = (
            config.storage_root / "objects"
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.RLock()

    def object_path(
        self,
        sha256: str,
    ) -> Path:
        if (
            len(sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in sha256
            )
        ):
            raise DatasetValidationError(
                "Invalid object hash"
            )

        return (
            self.root
            / sha256[:2]
            / sha256[2:4]
            / sha256
        )

    def store_file(
        self,
        source: Path,
    ) -> tuple[str, int]:
        source = Path(source)

        if not source.is_file():
            raise DatasetValidationError(
                f"Source file not found: {source}"
            )

        if (
            self.config.reject_symbolic_links
            and source.is_symlink()
        ):
            raise DatasetValidationError(
                "Symbolic links are not allowed"
            )

        size = source.stat().st_size

        if size > self.config.maximum_file_size_bytes:
            raise DatasetCapacityError(
                "File exceeds configured size limit"
            )

        # Copy and hash in one pass so the stored hash
        # describes the copied bytes, not an earlier
        # potentially changing source snapshot.
        with self._lock:
            self.root.mkdir(
                parents=True,
                exist_ok=True,
            )

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".object-",
                dir=str(self.root),
            )

            digest = hashlib.sha256()
            copied_size = 0

            try:
                with (
                    source.open("rb") as input_stream,
                    os.fdopen(descriptor, "wb") as output_stream,
                ):
                    while True:
                        chunk = input_stream.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        copied_size += len(chunk)

                        if (
                            copied_size
                            > self.config.maximum_file_size_bytes
                        ):
                            raise DatasetCapacityError(
                                "File exceeds configured size limit"
                            )

                        digest.update(chunk)
                        output_stream.write(chunk)

                    output_stream.flush()
                    os.fsync(output_stream.fileno())

                content_hash = digest.hexdigest()

                destination = self.object_path(
                    content_hash
                )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                if destination.exists():
                    if (
                        destination.stat().st_size != copied_size
                        or sha256_file(destination)
                        != content_hash
                    ):
                        raise DatasetIntegrityError(
                            "Existing content object is corrupted"
                        )
                else:
                    os.replace(
                        temporary_name,
                        destination,
                    )

                return (
                    content_hash,
                    copied_size,
                )

            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)

    def verify(
        self,
        sha256: str,
        expected_size: int | None = None,
    ) -> bool:
        path = self.object_path(
            sha256
        )

        if not path.is_file():
            return False

        if (
            expected_size is not None
            and path.stat().st_size != expected_size
        ):
            return False

        return sha256_file(path) == sha256

    def restore(
        self,
        sha256: str,
        destination: Path,
        expected_size: int | None = None,
    ) -> None:
        source = self.object_path(
            sha256
        )

        if not source.is_file():
            raise DatasetIntegrityError(
                "Dataset object is missing"
            )

        if not self.verify(
            sha256,
            expected_size,
        ):
            raise DatasetIntegrityError(
                "Dataset object failed integrity verification"
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copyfile(
            source,
            destination,
        )


# ============================================================
# MANIFEST BUILDER
# ============================================================

class DatasetManifestBuilder:

    @staticmethod
    def manifest_payload(
        dataset_id: str,
        version_number: int,
        parent_version_id: str | None,
        files: Iterable[DatasetFile],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        sorted_files = sorted(
            files,
            key=lambda item: item.relative_path,
        )

        return {
            "dataset_id": dataset_id,
            "version_number": version_number,
            "parent_version_id": parent_version_id,
            "files": [
                json_safe(item)
                for item in sorted_files
            ],
            "metadata": json_safe(dict(metadata)),
        }

    @classmethod
    def calculate_hash(
        cls,
        dataset_id: str,
        version_number: int,
        parent_version_id: str | None,
        files: Iterable[DatasetFile],
        metadata: Mapping[str, Any],
    ) -> str:
        payload = cls.manifest_payload(
            dataset_id,
            version_number,
            parent_version_id,
            files,
            metadata,
        )

        return sha256_bytes(
            canonical_json(payload)
        )


# ============================================================
# DATASET VERSION COMPARISON
# ============================================================

class DatasetVersionComparator:

    @staticmethod
    def compare(
        source: DatasetVersion,
        target: DatasetVersion,
    ) -> DatasetVersionDiff:
        if source.dataset_id != target.dataset_id:
            raise DatasetValidationError(
                "Cannot compare versions from different datasets"
            )

        source_files = {
            item.relative_path: item
            for item in source.files
        }

        target_files = {
            item.relative_path: item
            for item in target.files
        }

        changes = []

        for path in sorted(
            set(source_files) | set(target_files)
        ):
            old = source_files.get(path)
            new = target_files.get(path)

            if old is None:
                change_type = DatasetChangeType.ADDED

            elif new is None:
                change_type = DatasetChangeType.REMOVED

            elif (
                old.sha256 != new.sha256
                or old.split != new.split
            ):
                change_type = DatasetChangeType.MODIFIED

            else:
                change_type = DatasetChangeType.UNCHANGED

            changes.append(
                DatasetFileChange(
                    relative_path=path,
                    change_type=change_type,
                    old_sha256=(
                        old.sha256 if old else None
                    ),
                    new_sha256=(
                        new.sha256 if new else None
                    ),
                    old_size_bytes=(
                        old.size_bytes if old else None
                    ),
                    new_size_bytes=(
                        new.size_bytes if new else None
                    ),
                )
            )

        return DatasetVersionDiff(
            dataset_id=source.dataset_id,
            source_version_id=source.version_id,
            target_version_id=target.version_id,
            changes=tuple(changes),
            added_count=sum(
                item.change_type == DatasetChangeType.ADDED
                for item in changes
            ),
            removed_count=sum(
                item.change_type == DatasetChangeType.REMOVED
                for item in changes
            ),
            modified_count=sum(
                item.change_type == DatasetChangeType.MODIFIED
                for item in changes
            ),
            unchanged_count=sum(
                item.change_type == DatasetChangeType.UNCHANGED
                for item in changes
            ),
        )


# ============================================================
# DATASET STATISTICS
# ============================================================

class DatasetStatisticsCalculator:

    @staticmethod
    def calculate(
        version: DatasetVersion,
    ) -> DatasetStatistics:
        extensions: dict[str, int] = {}
        splits: dict[str, int] = {}

        unique_hashes = set()

        for item in version.files:
            extension = (
                Path(item.relative_path).suffix.lower()
                or "<none>"
            )

            extensions[extension] = (
                extensions.get(extension, 0) + 1
            )

            split = item.split.value

            splits[split] = (
                splits.get(split, 0) + 1
            )

            unique_hashes.add(item.sha256)

        return DatasetStatistics(
            dataset_id=version.dataset_id,
            version_id=version.version_id,
            total_files=version.file_count,
            total_size_bytes=version.total_size_bytes,
            file_extensions=extensions,
            split_counts=splits,
            unique_content_objects=len(unique_hashes),
        )


# ============================================================
# DATASET REGISTRY
# ============================================================

class DatasetVersioningService:

    def __init__(
        self,
        config: DatasetVersioningConfig | None = None,
    ) -> None:
        self.config = (
            config or DatasetVersioningConfig()
        )

        self.object_store = (
            ContentAddressedObjectStore(self.config)
        )

        self.manifest_builder = (
            DatasetManifestBuilder()
        )

        self.comparator = (
            DatasetVersionComparator()
        )

        self.statistics_calculator = (
            DatasetStatisticsCalculator()
        )

        self._datasets: dict[
            tuple[str, str],
            DatasetRecord,
        ] = {}

        self._versions: dict[
            tuple[str, str],
            list[DatasetVersion],
        ] = {}

        self._tags: dict[
            tuple[str, str, str],
            str,
        ] = {}

        self._lock = threading.RLock()

        self._load_registry()

    # --------------------------------------------------------
    # AUTHORIZATION
    # --------------------------------------------------------

    @staticmethod
    def _authorize(
        access: DatasetAccessContext,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise DatasetAuthorizationError(
                f"Missing permission: {permission}"
            )

    # --------------------------------------------------------
    # STORAGE PATHS
    # --------------------------------------------------------

    def _registry_path(self) -> Path:
        return (
            self.config.storage_root
            / "registry.json"
        )

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    def _persist(self) -> None:
        payload = {
            "schema_version": 1,
            "datasets": [
                record.to_dict()
                if hasattr(record, "to_dict")
                else json_safe(record)
                for record in self._datasets.values()
            ],
            "versions": [
                version.to_dict()
                for versions in self._versions.values()
                for version in versions
            ],
            "tags": [
                {
                    "tenant_id": tenant_id,
                    "dataset_id": dataset_id,
                    "tag": tag,
                    "version_id": version_id,
                }
                for (
                    tenant_id,
                    dataset_id,
                    tag,
                ), version_id in self._tags.items()
            ],
        }

        atomic_write_json(
            self._registry_path(),
            payload,
        )

    def _load_registry(self) -> None:
        path = self._registry_path()

        if not path.exists():
            return

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as stream:
                payload = json.load(stream)

            if payload.get("schema_version") != 1:
                raise DatasetIntegrityError(
                    "Unsupported registry schema"
                )

            for item in payload.get("datasets", []):
                record = DatasetRecord(
                    dataset_id=item["dataset_id"],
                    tenant_id=item["tenant_id"],
                    name=item["name"],
                    dataset_type=DatasetType(
                        item["dataset_type"]
                    ),
                    created_at=datetime.fromisoformat(
                        item["created_at"]
                    ),
                    created_by=item["created_by"],
                    description=item.get(
                        "description", ""
                    ),
                    metadata=item.get(
                        "metadata", {}
                    ),
                )

                self._datasets[
                    (
                        record.tenant_id,
                        record.dataset_id,
                    )
                ] = record

            for item in payload.get("versions", []):
                files = tuple(
                    DatasetFile(
                        relative_path=file["relative_path"],
                        sha256=file["sha256"],
                        size_bytes=file["size_bytes"],
                        split=DatasetSplit(
                            file["split"]
                        ),
                    )
                    for file in item["files"]
                )

                version = DatasetVersion(
                    dataset_id=item["dataset_id"],
                    version_id=item["version_id"],
                    version_number=item["version_number"],
                    tenant_id=item["tenant_id"],
                    parent_version_id=item.get(
                        "parent_version_id"
                    ),
                    files=files,
                    manifest_hash=item["manifest_hash"],
                    created_at=datetime.fromisoformat(
                        item["created_at"]
                    ),
                    created_by=item["created_by"],
                    description=item.get(
                        "description", ""
                    ),
                    metadata=item.get(
                        "metadata", {}
                    ),
                    status=DatasetVersionStatus(
                        item.get(
                            "status",
                            DatasetVersionStatus.ACTIVE.value,
                        )
                    ),
                )

                key = (
                    version.tenant_id,
                    version.dataset_id,
                )

                self._versions.setdefault(
                    key,
                    [],
                ).append(version)

            for item in payload.get("tags", []):
                self._tags[
                    (
                        item["tenant_id"],
                        item["dataset_id"],
                        item["tag"],
                    )
                ] = item["version_id"]

        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise DatasetIntegrityError(
                "Dataset registry is invalid"
            ) from exc

    # --------------------------------------------------------
    # DATASET REGISTRATION
    # --------------------------------------------------------

    def register_dataset(
        self,
        name: str,
        dataset_type: DatasetType,
        access: DatasetAccessContext,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> DatasetRecord:
        self._authorize(
            access,
            "can_manage",
        )

        name = validate_identifier(
            name,
            "dataset name",
        )

        record = DatasetRecord(
            dataset_id=generate_id(),
            tenant_id=access.tenant_id,
            name=name,
            dataset_type=DatasetType(dataset_type),
            created_at=utc_now(),
            created_by=access.user_id,
            description=description,
            metadata=dict(metadata or {}),
        )

        with self._lock:
            for existing in self._datasets.values():
                if (
                    existing.tenant_id == access.tenant_id
                    and existing.name == name
                ):
                    raise DatasetConflictError(
                        "Dataset name already exists"
                    )

            key = (
                access.tenant_id,
                record.dataset_id,
            )

            self._datasets[key] = record
            self._versions[key] = []

            self._persist()

        return record

    # --------------------------------------------------------
    # DATASET LOOKUP
    # --------------------------------------------------------

    def get_dataset(
        self,
        dataset_id: str,
        access: DatasetAccessContext,
    ) -> DatasetRecord:
        self._authorize(
            access,
            "can_read",
        )

        key = (
            access.tenant_id,
            dataset_id,
        )

        with self._lock:
            record = self._datasets.get(key)

            if record is None:
                raise DatasetNotFoundError(
                    "Dataset not found"
                )

            return record

    def list_datasets(
        self,
        access: DatasetAccessContext,
    ) -> tuple[DatasetRecord, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            return tuple(
                record
                for record in self._datasets.values()
                if record.tenant_id == access.tenant_id
            )

    # --------------------------------------------------------
    # FILE COLLECTION
    # --------------------------------------------------------

    def _collect_files(
        self,
        source_directory: Path,
        split_mapping: Mapping[str, DatasetSplit] | None,
    ) -> tuple[DatasetFile, ...]:
        source_directory = Path(
            source_directory
        )

        if not source_directory.is_dir():
            raise DatasetValidationError(
                "Dataset source directory not found"
            )

        if (
            self.config.reject_symbolic_links
            and source_directory.is_symlink()
        ):
            raise DatasetValidationError(
                "Dataset source directory cannot be a symlink"
            )

        files = []
        total_size = 0

        # os.walk with followlinks=False prevents traversal
        # into linked directories; reject them explicitly.
        for root, directories, filenames in os.walk(
            source_directory,
            followlinks=False,
        ):
            root_path = Path(root)

            if self.config.reject_symbolic_links:
                for directory in directories:
                    if (root_path / directory).is_symlink():
                        raise DatasetValidationError(
                            "Symbolic link directories are not allowed"
                        )

            for filename in filenames:
                source = root_path / filename

                if (
                    self.config.reject_symbolic_links
                    and source.is_symlink()
                ):
                    raise DatasetValidationError(
                        "Symbolic link files are not allowed"
                    )

                if not source.is_file():
                    continue

                relative_path = validate_relative_path(
                    source.relative_to(
                        source_directory
                    ).as_posix()
                )

                content_hash, size = (
                    self.object_store.store_file(
                        source
                    )
                )

                total_size += size

                if (
                    total_size
                    > self.config.maximum_dataset_size_bytes
                ):
                    raise DatasetCapacityError(
                        "Dataset size capacity exceeded"
                    )

                split = DatasetSplit.UNSPECIFIED

                if split_mapping:
                    split = DatasetSplit(
                        split_mapping.get(
                            relative_path,
                            DatasetSplit.UNSPECIFIED,
                        )
                    )

                files.append(
                    DatasetFile(
                        relative_path=relative_path,
                        sha256=content_hash,
                        size_bytes=size,
                        split=split,
                    )
                )

                if (
                    len(files)
                    > self.config.maximum_files_per_version
                ):
                    raise DatasetCapacityError(
                        "Dataset file count exceeded"
                    )

        return tuple(
            sorted(
                files,
                key=lambda item: item.relative_path,
            )
        )

    # --------------------------------------------------------
    # CREATE VERSION
    # --------------------------------------------------------

    def create_version(
        self,
        dataset_id: str,
        source_directory: Path,
        access: DatasetAccessContext,
        *,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
        split_mapping: Mapping[
            str, DatasetSplit
        ] | None = None,
        parent_version_id: str | None = None,
    ) -> DatasetVersion:
        self._authorize(
            access,
            "can_write",
        )

        key = (
            access.tenant_id,
            dataset_id,
        )

        # Serialize version numbering and publication.
        with self._lock:
            if key not in self._datasets:
                raise DatasetNotFoundError(
                    "Dataset not found"
                )

            versions = self._versions.setdefault(
                key,
                [],
            )

            if (
                len(versions)
                >= self.config.maximum_versions_per_dataset
            ):
                raise DatasetCapacityError(
                    "Maximum dataset versions reached"
                )

            if parent_version_id is None and versions:
                parent_version_id = (
                    versions[-1].version_id
                )

            if parent_version_id is not None:
                if not any(
                    version.version_id == parent_version_id
                    for version in versions
                ):
                    raise DatasetVersionNotFoundError(
                        "Parent version not found"
                    )

            files = self._collect_files(
                source_directory,
                split_mapping,
            )

            version_number = len(versions) + 1

            version_metadata = dict(
                metadata or {}
            )

            manifest_hash = (
                self.manifest_builder.calculate_hash(
                    dataset_id=dataset_id,
                    version_number=version_number,
                    parent_version_id=parent_version_id,
                    files=files,
                    metadata=version_metadata,
                )
            )

            version = DatasetVersion(
                dataset_id=dataset_id,
                version_id=generate_id(),
                version_number=version_number,
                tenant_id=access.tenant_id,
                parent_version_id=parent_version_id,
                files=files,
                manifest_hash=manifest_hash,
                created_at=utc_now(),
                created_by=access.user_id,
                description=description,
                metadata=version_metadata,
            )

            versions.append(version)

            try:
                self._persist()
            except Exception:
                versions.pop()
                raise

            return version

    # --------------------------------------------------------
    # GET VERSION
    # --------------------------------------------------------

    def get_version(
        self,
        dataset_id: str,
        version_id: str,
        access: DatasetAccessContext,
    ) -> DatasetVersion:
        self._authorize(
            access,
            "can_read",
        )

        key = (
            access.tenant_id,
            dataset_id,
        )

        with self._lock:
            if key not in self._datasets:
                raise DatasetNotFoundError(
                    "Dataset not found"
                )

            for version in self._versions.get(
                key,
                [],
            ):
                if version.version_id == version_id:
                    return version

        raise DatasetVersionNotFoundError(
            "Dataset version not found"
        )

    def list_versions(
        self,
        dataset_id: str,
        access: DatasetAccessContext,
    ) -> tuple[DatasetVersion, ...]:
        self._authorize(
            access,
            "can_read",
        )

        key = (
            access.tenant_id,
            dataset_id,
        )

        with self._lock:
            if key not in self._datasets:
                raise DatasetNotFoundError(
                    "Dataset not found"
                )

            return tuple(
                self._versions.get(key, [])
            )

    def latest_version(
        self,
        dataset_id: str,
        access: DatasetAccessContext,
    ) -> DatasetVersion:
        versions = self.list_versions(
            dataset_id,
            access,
        )

        if not versions:
            raise DatasetVersionNotFoundError(
                "Dataset has no versions"
            )

        return versions[-1]

    # --------------------------------------------------------
    # COMPARE VERSIONS
    # --------------------------------------------------------

    def compare_versions(
        self,
        dataset_id: str,
        source_version_id: str,
        target_version_id: str,
        access: DatasetAccessContext,
    ) -> DatasetVersionDiff:
        source = self.get_version(
            dataset_id,
            source_version_id,
            access,
        )

        target = self.get_version(
            dataset_id,
            target_version_id,
            access,
        )

        return self.comparator.compare(
            source,
            target,
        )

    # --------------------------------------------------------
    # INTEGRITY VERIFICATION
    # --------------------------------------------------------

    def verify_version(
        self,
        dataset_id: str,
        version_id: str,
        access: DatasetAccessContext,
    ) -> DatasetIntegrityReport:
        version = self.get_version(
            dataset_id,
            version_id,
            access,
        )

        expected_manifest_hash = (
            self.manifest_builder.calculate_hash(
                dataset_id=version.dataset_id,
                version_number=version.version_number,
                parent_version_id=version.parent_version_id,
                files=version.files,
                metadata=version.metadata,
            )
        )

        manifest_valid = (
            expected_manifest_hash
            == version.manifest_hash
        )

        missing = []
        corrupted = []

        for item in version.files:
            path = self.object_store.object_path(
                item.sha256
            )

            if not path.is_file():
                missing.append(
                    item.relative_path
                )
                continue

            if not self.object_store.verify(
                item.sha256,
                item.size_bytes,
            ):
                corrupted.append(
                    item.relative_path
                )

        valid = (
            manifest_valid
            and not missing
            and not corrupted
        )

        return DatasetIntegrityReport(
            dataset_id=dataset_id,
            version_id=version_id,
            valid=valid,
            checked_files=len(version.files),
            missing_objects=tuple(missing),
            corrupted_objects=tuple(corrupted),
            manifest_valid=manifest_valid,
            checked_at=utc_now(),
        )

    # --------------------------------------------------------
    # CHECKOUT VERSION
    # --------------------------------------------------------

    def checkout_version(
        self,
        dataset_id: str,
        version_id: str,
        destination_directory: Path,
        access: DatasetAccessContext,
        *,
        overwrite: bool = False,
    ) -> Path:
        version = self.get_version(
            dataset_id,
            version_id,
            access,
        )

        destination = Path(
            destination_directory
        ).expanduser().resolve()

        if (
            destination == self.config.storage_root
            or self.config.storage_root in destination.parents
        ):
            raise DatasetValidationError(
                "Cannot checkout inside dataset registry storage"
            )

        if destination.exists():
            if not destination.is_dir():
                raise DatasetValidationError(
                    "Checkout destination is not a directory"
                )

            if any(destination.iterdir()) and not overwrite:
                raise DatasetConflictError(
                    "Checkout destination is not empty"
                )

        if self.config.verify_objects_on_checkout:
            report = self.verify_version(
                dataset_id,
                version_id,
                access,
            )

            if not report.valid:
                raise DatasetIntegrityError(
                    "Dataset version failed integrity verification"
                )

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        for item in version.files:
            relative_path = validate_relative_path(
                item.relative_path
            )

            target = destination / relative_path

            # Prevent writing through an existing symlink
            # or symlinked ancestor.
            if target.exists() and target.is_symlink():
                raise DatasetValidationError(
                    "Checkout target cannot be a symlink"
                )

            for parent in (
                target,
                *target.parents,
            ):
                if parent == destination.parent:
                    break

                if parent.is_symlink():
                    raise DatasetValidationError(
                        "Checkout path contains a symlink"
                    )

            if (
                target.exists()
                and not overwrite
            ):
                raise DatasetConflictError(
                    f"File already exists: {relative_path}"
                )

            self.object_store.restore(
                item.sha256,
                target,
                item.size_bytes,
            )

        return destination

    # --------------------------------------------------------
    # DATASET TAGS
    # --------------------------------------------------------

    def tag_version(
        self,
        dataset_id: str,
        version_id: str,
        tag: str,
        access: DatasetAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        tag = validate_identifier(
            tag,
            "tag",
        )

        read_access = replace(
            access,
            can_read=True,
        )

        self.get_version(
            dataset_id,
            version_id,
            read_access,
        )

        key = (
            access.tenant_id,
            dataset_id,
            tag,
        )

        with self._lock:
            previous = self._tags.get(key)

            self._tags[key] = version_id

            try:
                self._persist()
            except Exception:
                if previous is None:
                    self._tags.pop(
                        key,
                        None,
                    )
                else:
                    self._tags[key] = previous
                raise

    def resolve_tag(
        self,
        dataset_id: str,
        tag: str,
        access: DatasetAccessContext,
    ) -> DatasetVersion:
        self._authorize(
            access,
            "can_read",
        )

        key = (
            access.tenant_id,
            dataset_id,
            tag,
        )

        with self._lock:
            version_id = self._tags.get(
                key
            )

        if version_id is None:
            raise DatasetVersionNotFoundError(
                "Dataset tag not found"
            )

        return self.get_version(
            dataset_id,
            version_id,
            access,
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        dataset_id: str,
        version_id: str,
        access: DatasetAccessContext,
    ) -> DatasetStatistics:
        version = self.get_version(
            dataset_id,
            version_id,
            access,
        )

        return self.statistics_calculator.calculate(
            version
        )

    # --------------------------------------------------------
    # LINEAGE
    # --------------------------------------------------------

    def lineage(
        self,
        dataset_id: str,
        version_id: str,
        access: DatasetAccessContext,
    ) -> tuple[DatasetVersion, ...]:
        current = self.get_version(
            dataset_id,
            version_id,
            access,
        )

        lineage = []
        visited = set()

        while current is not None:
            if current.version_id in visited:
                raise DatasetIntegrityError(
                    "Cycle detected in dataset lineage"
                )

            visited.add(
                current.version_id
            )

            lineage.append(
                current
            )

            if current.parent_version_id is None:
                break

            current = self.get_version(
                dataset_id,
                current.parent_version_id,
                access,
            )

        return tuple(lineage)

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    def export_version_json(
        self,
        dataset_id: str,
        version_id: str,
        access: DatasetAccessContext,
    ) -> str:
        version = self.get_version(
            dataset_id,
            version_id,
            access,
        )

        return json.dumps(
            version.to_dict(),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC DATASET FACTORY
# ============================================================

class SyntheticDatasetFactory:

    @staticmethod
    def create_robotics_dataset(
        directory: Path,
        *,
        version: int = 1,
    ) -> dict[str, DatasetSplit]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        files = {
            "train/camera/frame_001.txt": (
                f"robot-camera-frame-v{version}-001"
            ),
            "train/lidar/scan_001.txt": (
                f"lidar-scan-v{version}-001"
            ),
            "validation/telemetry/sample_001.csv": (
                "timestamp,temperature,vibration\n"
                f"0,{40 + version},0.12\n"
            ),
            "test/navigation/path_001.json": (
                json.dumps(
                    {
                        "robot_id": "ROBOT-001",
                        "version": version,
                        "waypoints": [
                            [0, 0],
                            [1, 0],
                            [2, 1],
                        ],
                    }
                )
            ),
        }

        splits = {}

        for relative_path, content in files.items():
            path = directory / relative_path

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            path.write_text(
                content,
                encoding="utf-8",
            )

            if relative_path.startswith("train/"):
                split = DatasetSplit.TRAIN

            elif relative_path.startswith("validation/"):
                split = DatasetSplit.VALIDATION

            else:
                split = DatasetSplit.TEST

            splits[relative_path] = split

        return splits


# ============================================================
# UNIT TESTS
# ============================================================

class TestDatasetVersioning(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

        self.root = Path(
            self.temporary.name
        )

        self.config = DatasetVersioningConfig(
            storage_root=self.root / "registry",
        )

        self.service = DatasetVersioningService(
            self.config
        )

        self.access = DatasetAccessContext(
            tenant_id="tenant_alpha",
            user_id="ml_engineer",
            can_read=True,
            can_write=True,
            can_manage=True,
        )

        self.dataset = self.service.register_dataset(
            name="robotics_visual_slam",
            dataset_type=DatasetType.SLAM,
            access=self.access,
        )

        self.source = self.root / "source"

        self.splits = (
            SyntheticDatasetFactory.create_robotics_dataset(
                self.source,
                version=1,
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create_version(self) -> DatasetVersion:
        return self.service.create_version(
            dataset_id=self.dataset.dataset_id,
            source_directory=self.source,
            access=self.access,
            split_mapping=self.splits,
        )

    def test_register_dataset(self) -> None:
        self.assertEqual(
            self.dataset.name,
            "robotics_visual_slam",
        )

    def test_create_version(self) -> None:
        version = self._create_version()

        self.assertEqual(
            version.version_number,
            1,
        )

        self.assertEqual(
            version.file_count,
            4,
        )

    def test_latest_version(self) -> None:
        version = self._create_version()

        latest = self.service.latest_version(
            self.dataset.dataset_id,
            self.access,
        )

        self.assertEqual(
            latest.version_id,
            version.version_id,
        )

    def test_content_hash(self) -> None:
        version = self._create_version()

        for item in version.files:
            self.assertEqual(
                len(item.sha256),
                64,
            )

    def test_integrity(self) -> None:
        version = self._create_version()

        report = self.service.verify_version(
            self.dataset.dataset_id,
            version.version_id,
            self.access,
        )

        self.assertTrue(
            report.valid
        )

    def test_version_comparison(self) -> None:
        first = self._create_version()

        SyntheticDatasetFactory.create_robotics_dataset(
            self.source,
            version=2,
        )

        second = self._create_version()

        diff = self.service.compare_versions(
            self.dataset.dataset_id,
            first.version_id,
            second.version_id,
            self.access,
        )

        self.assertGreater(
            diff.modified_count,
            0,
        )

    def test_checkout(self) -> None:
        version = self._create_version()

        destination = self.root / "checkout"

        self.service.checkout_version(
            self.dataset.dataset_id,
            version.version_id,
            destination,
            self.access,
        )

        self.assertTrue(
            (
                destination
                / "train/camera/frame_001.txt"
            ).exists()
        )

    def test_version_tag(self) -> None:
        version = self._create_version()

        self.service.tag_version(
            self.dataset.dataset_id,
            version.version_id,
            "production",
            self.access,
        )

        resolved = self.service.resolve_tag(
            self.dataset.dataset_id,
            "production",
            self.access,
        )

        self.assertEqual(
            resolved.version_id,
            version.version_id,
        )

    def test_lineage(self) -> None:
        first = self._create_version()

        second = self._create_version()

        lineage = self.service.lineage(
            self.dataset.dataset_id,
            second.version_id,
            self.access,
        )

        self.assertEqual(
            len(lineage),
            2,
        )

        self.assertEqual(
            lineage[-1].version_id,
            first.version_id,
        )

    def test_statistics(self) -> None:
        version = self._create_version()

        statistics = self.service.statistics(
            self.dataset.dataset_id,
            version.version_id,
            self.access,
        )

        self.assertEqual(
            statistics.total_files,
            4,
        )

        self.assertEqual(
            statistics.split_counts["train"],
            2,
        )

    def test_registry_persistence(self) -> None:
        version = self._create_version()

        reloaded = DatasetVersioningService(
            self.config
        )

        restored = reloaded.get_version(
            self.dataset.dataset_id,
            version.version_id,
            self.access,
        )

        self.assertEqual(
            restored.manifest_hash,
            version.manifest_hash,
        )

    def test_tenant_isolation(self) -> None:
        other_access = DatasetAccessContext(
            tenant_id="tenant_beta",
            user_id="other_user",
            can_read=True,
        )

        with self.assertRaises(
            DatasetNotFoundError
        ):
            self.service.get_dataset(
                self.dataset.dataset_id,
                other_access,
            )

    def test_read_permission(self) -> None:
        denied = DatasetAccessContext(
            tenant_id="tenant_alpha",
            user_id="reader",
            can_read=False,
        )

        with self.assertRaises(
            DatasetAuthorizationError
        ):
            self.service.list_datasets(
                denied
            )

    def test_path_traversal_rejected(self) -> None:
        with self.assertRaises(
            DatasetValidationError
        ):
            validate_relative_path(
                "../secret.txt"
            )

    def test_duplicate_dataset_name(self) -> None:
        with self.assertRaises(
            DatasetConflictError
        ):
            self.service.register_dataset(
                name="robotics_visual_slam",
                dataset_type=DatasetType.SLAM,
                access=self.access,
            )

    def test_missing_object_detection(self) -> None:
        version = self._create_version()

        item = version.files[0]

        object_path = (
            self.service.object_store.object_path(
                item.sha256
            )
        )

        object_path.unlink()

        report = self.service.verify_version(
            self.dataset.dataset_id,
            version.version_id,
            self.access,
        )

        self.assertFalse(
            report.valid
        )

        self.assertIn(
            item.relative_path,
            report.missing_objects,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        service = DatasetVersioningService(
            DatasetVersioningConfig(
                storage_root=root / "registry",
            )
        )

        access = DatasetAccessContext(
            tenant_id="robotics_lab",
            user_id="mlops_engineer",
            can_read=True,
            can_write=True,
            can_manage=True,
        )

        dataset = service.register_dataset(
            name="physical_ai_robotics_dataset",
            dataset_type=DatasetType.MULTIMODAL,
            access=access,
            description=(
                "Versioned camera, LiDAR, telemetry "
                "and navigation datasets"
            ),
            metadata={
                "platform": "physical_ai",
                "robot_family": "autonomous_mobile_robot",
            },
        )

        source = root / "dataset"

        splits = (
            SyntheticDatasetFactory.create_robotics_dataset(
                source,
                version=1,
            )
        )

        version_1 = service.create_version(
            dataset_id=dataset.dataset_id,
            source_directory=source,
            access=access,
            description="Initial robotics dataset",
            split_mapping=splits,
        )

        SyntheticDatasetFactory.create_robotics_dataset(
            source,
            version=2,
        )

        version_2 = service.create_version(
            dataset_id=dataset.dataset_id,
            source_directory=source,
            access=access,
            description="Updated robotics dataset",
            split_mapping=splits,
        )

        service.tag_version(
            dataset.dataset_id,
            version_2.version_id,
            "latest",
            access,
        )

        diff = service.compare_versions(
            dataset.dataset_id,
            version_1.version_id,
            version_2.version_id,
            access,
        )

        integrity = service.verify_version(
            dataset.dataset_id,
            version_2.version_id,
            access,
        )

        statistics = service.statistics(
            dataset.dataset_id,
            version_2.version_id,
            access,
        )

        checkout = service.checkout_version(
            dataset.dataset_id,
            version_1.version_id,
            root / "restored_dataset",
            access,
        )

        print("\nDATASET VERSIONING ENGINE")
        print("=" * 65)

        print(
            json.dumps(
                {
                    "dataset_id": dataset.dataset_id,
                    "dataset_name": dataset.name,
                    "dataset_type": dataset.dataset_type.value,
                    "version_1": version_1.version_id,
                    "version_2": version_2.version_id,
                    "latest_version_number": version_2.version_number,
                    "version_diff": json_safe(diff),
                    "integrity": json_safe(integrity),
                    "statistics": json_safe(statistics),
                    "checkout_path": str(checkout),
                    "lineage": [
                        version.version_number
                        for version in service.lineage(
                            dataset.dataset_id,
                            version_2.version_id,
                            access,
                        )
                    ],
                },
                indent=2,
                allow_nan=False,
            )
        )

    print("\nRUNNING UNIT TESTS")
    print("=" * 65)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestDatasetVersioning
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
"""Feature Store module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/mlops/feature_store.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Feature Store — Offline / Online Feature Management

Features:
    - Feature registry and feature groups
    - Feature schema validation
    - Feature versioning
    - Offline feature storage (SQLite)
    - Online feature serving (SQLite-backed latest-value index)
    - Point-in-time feature retrieval
    - Historical feature queries
    - Batch feature ingestion
    - Online feature synchronization
    - Feature freshness monitoring
    - Feature quality validation
    - Missing-value detection
    - Feature statistics
    - Feature lineage metadata
    - Dataset and model version references
    - Robotics sensor feature support
    - Computer vision embedding features
    - Reinforcement learning state features
    - Model training feature retrieval
    - Multi-tenant isolation
    - Thread-safe operations
    - JSON and CSV export
    - Audit logging
    - Synthetic robotics telemetry
    - Unit tests

Python 3.11+
Standard library only.

Architecture:
    Feature Registry
          |
          v
    Schema Validation
          |
          v
    Offline Feature Store (SQLite)
          |
          v
    Online Materialization (SQLite)
          |
          v
    Low-overhead Feature Retrieval

Important:
    This implementation is intended for local development,
    testing and prototype workloads.

    Production deployment should use a durable distributed
    offline store, a suitable low-latency online database,
    centralized authorization and operational monitoring.

    Feature values are advisory inputs to AI models.
    This module does not control physical robot hardware.
"""

from __future__ import annotations

import csv
import io
import json
import math
import sqlite3
import statistics
import tempfile
import threading
import unittest
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class FeatureStoreError(Exception):
    pass


class FeatureValidationError(FeatureStoreError):
    pass


class FeatureNotFoundError(FeatureStoreError):
    pass


class FeatureAuthorizationError(FeatureStoreError):
    pass


class FeatureConflictError(FeatureStoreError):
    pass


class FeatureFreshnessError(FeatureStoreError):
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
        raise FeatureValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def ensure_utc(
    value: datetime,
) -> datetime:
    if not isinstance(value, datetime):
        raise FeatureValidationError(
            "Timestamp must be datetime"
        )

    if value.tzinfo is None:
        raise FeatureValidationError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(
        timezone.utc
    )


def timestamp_text(
    value: datetime,
) -> str:
    return ensure_utc(value).isoformat(
        timespec="microseconds"
    )


def parse_timestamp(
    value: str,
) -> datetime:
    return ensure_utc(
        datetime.fromisoformat(value)
    )


def finite_number(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise FeatureValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise FeatureValidationError(
            f"{name} must be finite"
        )

    return result


def json_value(
    value: Any,
) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise FeatureValidationError(
            "Value must be JSON serializable"
        ) from exc


def json_text(
    value: Any,
) -> str:
    return json.dumps(
        json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


# ============================================================
# ENUMERATIONS
# ============================================================

class FeatureDataType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    CATEGORY = "category"
    FLOAT_VECTOR = "float_vector"
    INTEGER_VECTOR = "integer_vector"
    JSON = "json"


class FeatureGroupType(str, Enum):
    ROBOT_TELEMETRY = "robot_telemetry"
    ROBOT_HEALTH = "robot_health"
    COMPUTER_VISION = "computer_vision"
    LIDAR = "lidar"
    NAVIGATION = "navigation"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    SENSOR_FUSION = "sensor_fusion"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    MODEL_PREDICTIONS = "model_predictions"
    GENERIC = "generic"


class FeatureStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class FeatureQualityStatus(str, Enum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class FeatureFreshnessStatus(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    MISSING = "missing"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class FeatureStoreConfig:
    database_path: Path = Path(
        "./data/mlops/feature_store.sqlite3"
    )

    sqlite_timeout_seconds: float = 10.0

    maximum_batch_size: int = 10000
    maximum_vector_dimension: int = 4096
    maximum_history_limit: int = 100000

    default_freshness_seconds: int = 300

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "database_path",
            Path(self.database_path).expanduser().resolve(),
        )

        finite_number(
            self.sqlite_timeout_seconds,
            "sqlite_timeout_seconds",
        )

        if self.sqlite_timeout_seconds <= 0:
            raise FeatureValidationError(
                "SQLite timeout must be positive"
            )

        for name in (
            "maximum_batch_size",
            "maximum_vector_dimension",
            "maximum_history_limit",
            "default_freshness_seconds",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise FeatureValidationError(
                    f"{name} must be a positive integer"
                )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class FeatureAccessContext:
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
# FEATURE SCHEMA
# ============================================================

@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    data_type: FeatureDataType

    nullable: bool = False

    description: str = ""
    unit: str = ""

    minimum: float | None = None
    maximum: float | None = None

    vector_dimension: int | None = None

    allowed_categories: tuple[str, ...] = ()

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.name,
            "feature name",
        )

        if self.minimum is not None:
            finite_number(
                self.minimum,
                "minimum",
            )

        if self.maximum is not None:
            finite_number(
                self.maximum,
                "maximum",
            )

        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise FeatureValidationError(
                "minimum cannot exceed maximum"
            )

        if self.vector_dimension is not None:
            if (
                isinstance(self.vector_dimension, bool)
                or not isinstance(self.vector_dimension, int)
                or self.vector_dimension <= 0
            ):
                raise FeatureValidationError(
                    "vector_dimension must be positive"
                )

        json_value(self.metadata)


@dataclass(frozen=True)
class FeatureGroup:
    group_id: str

    tenant_id: str
    name: str
    version: int

    group_type: FeatureGroupType

    entity_type: str

    features: tuple[FeatureDefinition, ...]

    created_at: datetime
    created_by: str

    status: FeatureStatus = FeatureStatus.ACTIVE

    freshness_seconds: int = 300

    dataset_id: str | None = None
    dataset_version_id: str | None = None

    source_name: str = ""

    description: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(
            feature.name
            for feature in self.features
        )


# ============================================================
# FEATURE RECORDS
# ============================================================

@dataclass(frozen=True)
class FeatureRecord:
    record_id: str

    tenant_id: str
    group_id: str
    entity_id: str

    event_timestamp: datetime
    ingestion_timestamp: datetime

    values: dict[str, Any]

    source_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class FeatureQualityResult:
    group_id: str
    entity_id: str

    status: FeatureQualityStatus

    valid: bool

    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class FeatureFreshnessResult:
    group_id: str
    entity_id: str

    status: FeatureFreshnessStatus

    latest_event_timestamp: datetime | None

    age_seconds: float | None
    threshold_seconds: int


@dataclass(frozen=True)
class FeatureStatistics:
    feature_name: str

    sample_count: int
    missing_count: int

    minimum: float | None = None
    maximum: float | None = None

    mean: float | None = None
    standard_deviation: float | None = None


@dataclass(frozen=True)
class FeatureTrainingRow:
    entity_id: str
    observation_timestamp: datetime

    features: dict[str, Any]


# ============================================================
# FEATURE VALIDATOR
# ============================================================

class FeatureSchemaValidator:

    def __init__(
        self,
        maximum_vector_dimension: int = 4096,
    ) -> None:
        self.maximum_vector_dimension = (
            maximum_vector_dimension
        )

    def validate_value(
        self,
        definition: FeatureDefinition,
        value: Any,
    ) -> Any:
        if value is None:
            if definition.nullable:
                return None

            raise FeatureValidationError(
                f"{definition.name} cannot be null"
            )

        data_type = definition.data_type

        if data_type == FeatureDataType.INTEGER:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise FeatureValidationError(
                    f"{definition.name} must be integer"
                )

            result = value

        elif data_type == FeatureDataType.FLOAT:
            result = finite_number(
                value,
                definition.name,
            )

        elif data_type == FeatureDataType.BOOLEAN:
            if not isinstance(value, bool):
                raise FeatureValidationError(
                    f"{definition.name} must be boolean"
                )

            result = value

        elif data_type in (
            FeatureDataType.STRING,
            FeatureDataType.CATEGORY,
        ):
            if not isinstance(value, str):
                raise FeatureValidationError(
                    f"{definition.name} must be string"
                )

            result = value

            if (
                data_type == FeatureDataType.CATEGORY
                and definition.allowed_categories
                and value not in definition.allowed_categories
            ):
                raise FeatureValidationError(
                    f"Unsupported category for {definition.name}"
                )

        elif data_type in (
            FeatureDataType.FLOAT_VECTOR,
            FeatureDataType.INTEGER_VECTOR,
        ):
            if (
                not isinstance(value, (list, tuple))
                or isinstance(value, (str, bytes))
            ):
                raise FeatureValidationError(
                    f"{definition.name} must be a vector"
                )

            if (
                len(value)
                > self.maximum_vector_dimension
            ):
                raise FeatureValidationError(
                    "Vector dimension exceeds configured limit"
                )

            if (
                definition.vector_dimension is not None
                and len(value) != definition.vector_dimension
            ):
                raise FeatureValidationError(
                    f"Invalid vector dimension for {definition.name}"
                )

            if data_type == FeatureDataType.INTEGER_VECTOR:
                if any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    for item in value
                ):
                    raise FeatureValidationError(
                        f"{definition.name} requires integer vector"
                    )

                result = list(value)

            else:
                result = [
                    finite_number(
                        item,
                        definition.name,
                    )
                    for item in value
                ]

        elif data_type == FeatureDataType.JSON:
            result = json_value(
                value
            )

        else:
            raise FeatureValidationError(
                "Unsupported feature type"
            )

        if (
            definition.minimum is not None
            or definition.maximum is not None
        ):
            if isinstance(result, (int, float)) and not isinstance(
                result,
                bool,
            ):
                numeric = float(result)

                if (
                    definition.minimum is not None
                    and numeric < definition.minimum
                ):
                    raise FeatureValidationError(
                        f"{definition.name} below minimum"
                    )

                if (
                    definition.maximum is not None
                    and numeric > definition.maximum
                ):
                    raise FeatureValidationError(
                        f"{definition.name} above maximum"
                    )

        return result

    def validate_record(
        self,
        group: FeatureGroup,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected = set(
            group.feature_names
        )

        provided = set(
            values
        )

        unknown = provided - expected

        if unknown:
            raise FeatureValidationError(
                f"Unknown features: {sorted(unknown)}"
            )

        normalized = {}

        for definition in group.features:
            value = values.get(
                definition.name
            )

            normalized[
                definition.name
            ] = self.validate_value(
                definition,
                value,
            )

        return normalized

    def quality_check(
        self,
        group: FeatureGroup,
        entity_id: str,
        values: Mapping[str, Any],
    ) -> FeatureQualityResult:
        errors = []
        warnings = []

        try:
            normalized = self.validate_record(
                group,
                values,
            )

            for name, value in normalized.items():
                if value is None:
                    warnings.append(
                        f"Missing nullable feature: {name}"
                    )

        except FeatureValidationError as exc:
            errors.append(
                str(exc)
            )

        if errors:
            status = FeatureQualityStatus.FAILED

        elif warnings:
            status = FeatureQualityStatus.WARNING

        else:
            status = FeatureQualityStatus.PASSED

        return FeatureQualityResult(
            group_id=group.group_id,
            entity_id=entity_id,
            status=status,
            valid=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )


# ============================================================
# SQLITE FEATURE STORE
# ============================================================

class FeatureStore:

    def __init__(
        self,
        config: FeatureStoreConfig | None = None,
    ) -> None:
        self.config = (
            config or FeatureStoreConfig()
        )

        self.validator = FeatureSchemaValidator(
            self.config.maximum_vector_dimension
        )

        self._lock = threading.RLock()

        self.config.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
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

        self._initialize_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> FeatureStore:
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    # --------------------------------------------------------
    # AUTHORIZATION
    # --------------------------------------------------------

    @staticmethod
    def _authorize(
        access: FeatureAccessContext,
        permission: str,
    ) -> None:
        if not getattr(
            access,
            permission,
            False,
        ):
            raise FeatureAuthorizationError(
                f"Missing permission: {permission}"
            )

    # --------------------------------------------------------
    # DATABASE INITIALIZATION
    # --------------------------------------------------------

    def _initialize_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feature_groups (
                    group_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    group_type TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    schema_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    status TEXT NOT NULL,
                    freshness_seconds INTEGER NOT NULL,
                    dataset_id TEXT,
                    dataset_version_id TEXT,
                    source_name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE(tenant_id, name, version)
                );

                CREATE TABLE IF NOT EXISTS offline_features (
                    record_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    event_timestamp TEXT NOT NULL,
                    ingestion_timestamp TEXT NOT NULL,
                    values_json TEXT NOT NULL,
                    source_id TEXT,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(group_id)
                        REFERENCES feature_groups(group_id)
                );

                CREATE INDEX IF NOT EXISTS
                idx_offline_lookup
                ON offline_features (
                    tenant_id,
                    group_id,
                    entity_id,
                    event_timestamp DESC,
                    ingestion_timestamp DESC
                );

                CREATE TABLE IF NOT EXISTS online_features (
                    tenant_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    event_timestamp TEXT NOT NULL,
                    ingestion_timestamp TEXT NOT NULL,
                    values_json TEXT NOT NULL,
                    PRIMARY KEY (
                        tenant_id,
                        group_id,
                        entity_id
                    )
                );

                CREATE TABLE IF NOT EXISTS feature_audit (
                    audit_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    group_id TEXT,
                    entity_id TEXT,
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
        access: FeatureAccessContext,
        action: str,
        *,
        group_id: str | None = None,
        entity_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO feature_audit (
                audit_id,
                tenant_id,
                user_id,
                action,
                group_id,
                entity_id,
                timestamp,
                details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                generate_id(),
                access.tenant_id,
                access.user_id,
                action,
                group_id,
                entity_id,
                timestamp_text(utc_now()),
                json_text(
                    dict(details or {})
                ),
            ),
        )

    # --------------------------------------------------------
    # CREATE FEATURE GROUP
    # --------------------------------------------------------

    def create_feature_group(
        self,
        name: str,
        group_type: FeatureGroupType,
        entity_type: str,
        features: Sequence[FeatureDefinition],
        access: FeatureAccessContext,
        *,
        freshness_seconds: int | None = None,
        dataset_id: str | None = None,
        dataset_version_id: str | None = None,
        source_name: str = "",
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> FeatureGroup:
        self._authorize(
            access,
            "can_manage",
        )

        validate_identifier(
            name,
            "group name",
        )

        validate_identifier(
            entity_type,
            "entity type",
        )

        if not features:
            raise FeatureValidationError(
                "Feature group cannot be empty"
            )

        feature_names = [
            feature.name
            for feature in features
        ]

        if len(feature_names) != len(set(feature_names)):
            raise FeatureValidationError(
                "Duplicate feature definitions"
            )

        freshness = (
            freshness_seconds
            if freshness_seconds is not None
            else self.config.default_freshness_seconds
        )

        if (
            isinstance(freshness, bool)
            or not isinstance(freshness, int)
            or freshness <= 0
        ):
            raise FeatureValidationError(
                "freshness_seconds must be positive"
            )

        for feature in features:
            if (
                feature.vector_dimension is not None
                and feature.vector_dimension
                > self.config.maximum_vector_dimension
            ):
                raise FeatureValidationError(
                    "Vector dimension exceeds limit"
                )

        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT COALESCE(MAX(version), 0)
                AS latest_version
                FROM feature_groups
                WHERE tenant_id = ?
                AND name = ?
                """,
                (
                    access.tenant_id,
                    name,
                ),
            ).fetchone()

            version = (
                row["latest_version"] + 1
            )

            group = FeatureGroup(
                group_id=generate_id(),
                tenant_id=access.tenant_id,
                name=name,
                version=version,
                group_type=group_type,
                entity_type=entity_type,
                features=tuple(features),
                created_at=utc_now(),
                created_by=access.user_id,
                freshness_seconds=freshness,
                dataset_id=dataset_id,
                dataset_version_id=dataset_version_id,
                source_name=source_name,
                description=description,
                metadata=json_value(
                    dict(metadata or {})
                ),
            )

            self._connection.execute(
                """
                INSERT INTO feature_groups (
                    group_id,
                    tenant_id,
                    name,
                    version,
                    group_type,
                    entity_type,
                    schema_json,
                    created_at,
                    created_by,
                    status,
                    freshness_seconds,
                    dataset_id,
                    dataset_version_id,
                    source_name,
                    description,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    group.group_id,
                    group.tenant_id,
                    group.name,
                    group.version,
                    group.group_type.value,
                    group.entity_type,
                    json_text(
                        [
                            {
                                **{
                                    key: value
                                    for key, value
                                    in feature.__dict__.items()
                                    if key != "data_type"
                                },
                                "data_type": feature.data_type.value,
                            }
                            for feature in features
                        ]
                    ),
                    timestamp_text(group.created_at),
                    group.created_by,
                    group.status.value,
                    group.freshness_seconds,
                    group.dataset_id,
                    group.dataset_version_id,
                    group.source_name,
                    group.description,
                    json_text(group.metadata),
                ),
            )

            self._audit(
                access,
                "feature_group_created",
                group_id=group.group_id,
                details={
                    "name": name,
                    "version": version,
                },
            )

            return group

    # --------------------------------------------------------
    # ROW CONVERSION
    # --------------------------------------------------------

    @staticmethod
    def _row_to_group(
        row: sqlite3.Row,
    ) -> FeatureGroup:
        definitions = json.loads(
            row["schema_json"]
        )

        features = tuple(
            FeatureDefinition(
                name=item["name"],
                data_type=FeatureDataType(
                    item["data_type"]
                ),
                nullable=item.get(
                    "nullable",
                    False,
                ),
                description=item.get(
                    "description",
                    "",
                ),
                unit=item.get(
                    "unit",
                    "",
                ),
                minimum=item.get(
                    "minimum"
                ),
                maximum=item.get(
                    "maximum"
                ),
                vector_dimension=item.get(
                    "vector_dimension"
                ),
                allowed_categories=tuple(
                    item.get(
                        "allowed_categories",
                        [],
                    )
                ),
                metadata=item.get(
                    "metadata",
                    {},
                ),
            )
            for item in definitions
        )

        return FeatureGroup(
            group_id=row["group_id"],
            tenant_id=row["tenant_id"],
            name=row["name"],
            version=row["version"],
            group_type=FeatureGroupType(
                row["group_type"]
            ),
            entity_type=row["entity_type"],
            features=features,
            created_at=parse_timestamp(
                row["created_at"]
            ),
            created_by=row["created_by"],
            status=FeatureStatus(
                row["status"]
            ),
            freshness_seconds=row[
                "freshness_seconds"
            ],
            dataset_id=row["dataset_id"],
            dataset_version_id=row[
                "dataset_version_id"
            ],
            source_name=row["source_name"],
            description=row["description"],
            metadata=json.loads(
                row["metadata_json"]
            ),
        )

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
    ) -> FeatureRecord:
        return FeatureRecord(
            record_id=row["record_id"],
            tenant_id=row["tenant_id"],
            group_id=row["group_id"],
            entity_id=row["entity_id"],
            event_timestamp=parse_timestamp(
                row["event_timestamp"]
            ),
            ingestion_timestamp=parse_timestamp(
                row["ingestion_timestamp"]
            ),
            values=json.loads(
                row["values_json"]
            ),
            source_id=(
                row["source_id"]
                if "source_id" in row.keys()
                else None
            ),
            metadata=(
                json.loads(
                    row["metadata_json"]
                )
                if "metadata_json" in row.keys()
                else {}
            ),
        )

    # --------------------------------------------------------
    # GET FEATURE GROUP
    # --------------------------------------------------------

    def get_feature_group(
        self,
        group_id: str,
        access: FeatureAccessContext,
    ) -> FeatureGroup:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            row = self._connection.execute(
                """
                SELECT *
                FROM feature_groups
                WHERE tenant_id = ?
                AND group_id = ?
                """,
                (
                    access.tenant_id,
                    group_id,
                ),
            ).fetchone()

            if row is None:
                raise FeatureNotFoundError(
                    "Feature group not found"
                )

            return self._row_to_group(
                row
            )

    # --------------------------------------------------------
    # LIST FEATURE GROUPS
    # --------------------------------------------------------

    def list_feature_groups(
        self,
        access: FeatureAccessContext,
        *,
        group_type: FeatureGroupType | None = None,
    ) -> tuple[FeatureGroup, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            if group_type is None:
                rows = self._connection.execute(
                    """
                    SELECT *
                    FROM feature_groups
                    WHERE tenant_id = ?
                    ORDER BY name, version
                    """,
                    (
                        access.tenant_id,
                    ),
                ).fetchall()

            else:
                rows = self._connection.execute(
                    """
                    SELECT *
                    FROM feature_groups
                    WHERE tenant_id = ?
                    AND group_type = ?
                    ORDER BY name, version
                    """,
                    (
                        access.tenant_id,
                        group_type.value,
                    ),
                ).fetchall()

            return tuple(
                self._row_to_group(row)
                for row in rows
            )

    # --------------------------------------------------------
    # INGEST FEATURES
    # --------------------------------------------------------

    def ingest_features(
        self,
        group_id: str,
        entity_id: str,
        values: Mapping[str, Any],
        event_timestamp: datetime,
        access: FeatureAccessContext,
        *,
        source_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        materialize_online: bool = True,
    ) -> FeatureRecord:
        results = self.ingest_batch(
            group_id,
            [
                {
                    "entity_id": entity_id,
                    "values": dict(values),
                    "event_timestamp": event_timestamp,
                    "source_id": source_id,
                    "metadata": dict(metadata or {}),
                }
            ],
            access,
            materialize_online=materialize_online,
        )

        return results[0]

    # --------------------------------------------------------
    # BATCH INGESTION
    # --------------------------------------------------------

    def ingest_batch(
        self,
        group_id: str,
        records: Sequence[Mapping[str, Any]],
        access: FeatureAccessContext,
        *,
        materialize_online: bool = True,
    ) -> tuple[FeatureRecord, ...]:
        self._authorize(
            access,
            "can_write",
        )

        if not records:
            return ()

        if len(records) > self.config.maximum_batch_size:
            raise FeatureValidationError(
                "Batch exceeds configured limit"
            )

        group = self.get_feature_group(
            group_id,
            FeatureAccessContext(
                tenant_id=access.tenant_id,
                user_id=access.user_id,
                can_read=True,
            ),
        )

        if group.status != FeatureStatus.ACTIVE:
            raise FeatureConflictError(
                "Feature group is not active"
            )

        ingestion_time = utc_now()

        prepared = []

        for item in records:
            entity_id = validate_identifier(
                item["entity_id"],
                "entity_id",
            )

            event_time = ensure_utc(
                item["event_timestamp"]
            )

            normalized = self.validator.validate_record(
                group,
                item["values"],
            )

            source_id = item.get(
                "source_id"
            )

            if source_id is not None:
                validate_identifier(
                    source_id,
                    "source_id",
                )

            record = FeatureRecord(
                record_id=generate_id(),
                tenant_id=access.tenant_id,
                group_id=group_id,
                entity_id=entity_id,
                event_timestamp=event_time,
                ingestion_timestamp=ingestion_time,
                values=normalized,
                source_id=source_id,
                metadata=json_value(
                    dict(item.get("metadata") or {})
                ),
            )

            prepared.append(
                record
            )

        with self._lock, self._connection:
            for record in prepared:
                self._connection.execute(
                    """
                    INSERT INTO offline_features (
                        record_id,
                        tenant_id,
                        group_id,
                        entity_id,
                        event_timestamp,
                        ingestion_timestamp,
                        values_json,
                        source_id,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.tenant_id,
                        record.group_id,
                        record.entity_id,
                        timestamp_text(
                            record.event_timestamp
                        ),
                        timestamp_text(
                            record.ingestion_timestamp
                        ),
                        json_text(
                            record.values
                        ),
                        record.source_id,
                        json_text(
                            record.metadata
                        ),
                    ),
                )

                if materialize_online:
                    self._upsert_online(
                        record
                    )

            self._audit(
                access,
                "features_ingested",
                group_id=group_id,
                details={
                    "record_count": len(prepared),
                    "online_materialized": materialize_online,
                },
            )

        return tuple(prepared)

    # --------------------------------------------------------
    # ONLINE MATERIALIZATION
    # --------------------------------------------------------

    def _upsert_online(
        self,
        record: FeatureRecord,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO online_features (
                tenant_id,
                group_id,
                entity_id,
                record_id,
                event_timestamp,
                ingestion_timestamp,
                values_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                tenant_id,
                group_id,
                entity_id
            )
            DO UPDATE SET
                record_id = excluded.record_id,
                event_timestamp = excluded.event_timestamp,
                ingestion_timestamp = excluded.ingestion_timestamp,
                values_json = excluded.values_json
            WHERE
                excluded.event_timestamp
                    > online_features.event_timestamp
                OR (
                    excluded.event_timestamp
                        = online_features.event_timestamp
                    AND excluded.ingestion_timestamp
                        >= online_features.ingestion_timestamp
                )
            """,
            (
                record.tenant_id,
                record.group_id,
                record.entity_id,
                record.record_id,
                timestamp_text(
                    record.event_timestamp
                ),
                timestamp_text(
                    record.ingestion_timestamp
                ),
                json_text(
                    record.values
                ),
            ),
        )

    def materialize_online(
        self,
        group_id: str,
        access: FeatureAccessContext,
    ) -> int:
        self._authorize(
            access,
            "can_write",
        )

        self.get_feature_group(
            group_id,
            FeatureAccessContext(
                tenant_id=access.tenant_id,
                user_id=access.user_id,
                can_read=True,
            ),
        )

        with self._lock, self._connection:
            rows = self._connection.execute(
                """
                SELECT *
                FROM offline_features
                WHERE tenant_id = ?
                AND group_id = ?
                ORDER BY
                    event_timestamp ASC,
                    ingestion_timestamp ASC,
                    record_id ASC
                """,
                (
                    access.tenant_id,
                    group_id,
                ),
            ).fetchall()

            latest: dict[str, FeatureRecord] = {}

            for row in rows:
                record = self._row_to_record(
                    row
                )

                latest[
                    record.entity_id
                ] = record

            for record in latest.values():
                self._upsert_online(
                    record
                )

            self._audit(
                access,
                "online_materialized",
                group_id=group_id,
                details={
                    "entity_count": len(latest),
                },
            )

            return len(latest)

    # --------------------------------------------------------
    # GET ONLINE FEATURES
    # --------------------------------------------------------

    def get_online_features(
        self,
        group_id: str,
        entity_id: str,
        access: FeatureAccessContext,
        *,
        require_fresh: bool = False,
        as_of: datetime | None = None,
    ) -> FeatureRecord | None:
        self._authorize(
            access,
            "can_read",
        )

        group = self.get_feature_group(
            group_id,
            access,
        )

        validate_identifier(
            entity_id,
            "entity_id",
        )

        with self._lock:
            row = self._connection.execute(
                """
                SELECT
                    o.record_id,
                    o.tenant_id,
                    o.group_id,
                    o.entity_id,
                    o.event_timestamp,
                    o.ingestion_timestamp,
                    o.values_json,
                    f.source_id,
                    f.metadata_json
                FROM online_features AS o
                JOIN offline_features AS f
                  ON f.record_id = o.record_id
                WHERE o.tenant_id = ?
                AND o.group_id = ?
                AND o.entity_id = ?
                """,
                (
                    access.tenant_id,
                    group_id,
                    entity_id,
                ),
            ).fetchone()

            if row is None:
                return None

            record = self._row_to_record(
                row
            )

            if require_fresh:
                reference_time = (
                    ensure_utc(as_of)
                    if as_of is not None
                    else utc_now()
                )

                age = (
                    reference_time
                    - record.event_timestamp
                ).total_seconds()

                if (
                    age < 0
                    or age > group.freshness_seconds
                ):
                    raise FeatureFreshnessError(
                        "Online feature is not fresh"
                    )

            return record

    # --------------------------------------------------------
    # POINT-IN-TIME FEATURE RETRIEVAL
    # --------------------------------------------------------

    def get_features_at(
        self,
        group_id: str,
        entity_id: str,
        as_of: datetime,
        access: FeatureAccessContext,
        *,
        available_as_of: datetime | None = None,
    ) -> FeatureRecord | None:
        """
        Return the latest feature event at or before `as_of`.

        `available_as_of` additionally constrains ingestion time,
        preventing use of records not yet available at that time.

        For leakage-resistant historical training, pass the
        observation timestamp as both arguments.
        """
        self._authorize(
            access,
            "can_read",
        )

        self.get_feature_group(
            group_id,
            access,
        )

        validate_identifier(
            entity_id,
            "entity_id",
        )

        event_cutoff = timestamp_text(
            as_of
        )

        ingestion_cutoff = (
            timestamp_text(available_as_of)
            if available_as_of is not None
            else None
        )

        with self._lock:
            row = self._connection.execute(
                """
                SELECT *
                FROM offline_features
                WHERE tenant_id = ?
                AND group_id = ?
                AND entity_id = ?
                AND event_timestamp <= ?
                AND (
                    ? IS NULL
                    OR ingestion_timestamp <= ?
                )
                ORDER BY
                    event_timestamp DESC,
                    ingestion_timestamp DESC,
                    record_id DESC
                LIMIT 1
                """,
                (
                    access.tenant_id,
                    group_id,
                    entity_id,
                    event_cutoff,
                    ingestion_cutoff,
                    ingestion_cutoff,
                ),
            ).fetchone()

            if row is None:
                return None

            return self._row_to_record(
                row
            )

    # --------------------------------------------------------
    # HISTORICAL FEATURES
    # --------------------------------------------------------

    def get_feature_history(
        self,
        group_id: str,
        entity_id: str,
        access: FeatureAccessContext,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> tuple[FeatureRecord, ...]:
        self._authorize(
            access,
            "can_read",
        )

        self.get_feature_group(
            group_id,
            access,
        )

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
            or limit > self.config.maximum_history_limit
        ):
            raise FeatureValidationError(
                "Invalid history limit"
            )

        query = """
            SELECT *
            FROM offline_features
            WHERE tenant_id = ?
            AND group_id = ?
            AND entity_id = ?
        """

        parameters: list[Any] = [
            access.tenant_id,
            group_id,
            entity_id,
        ]

        if start_time is not None:
            query += (
                " AND event_timestamp >= ?"
            )

            parameters.append(
                timestamp_text(
                    start_time
                )
            )

        if end_time is not None:
            query += (
                " AND event_timestamp <= ?"
            )

            parameters.append(
                timestamp_text(
                    end_time
                )
            )

        query += """
            ORDER BY
                event_timestamp ASC,
                ingestion_timestamp ASC,
                record_id ASC
            LIMIT ?
        """

        parameters.append(
            limit
        )

        with self._lock:
            rows = self._connection.execute(
                query,
                parameters,
            ).fetchall()

            return tuple(
                self._row_to_record(row)
                for row in rows
            )

    # --------------------------------------------------------
    # TRAINING DATASET
    # --------------------------------------------------------

    def build_training_dataset(
        self,
        group_id: str,
        observations: Sequence[
            tuple[str, datetime]
        ],
        access: FeatureAccessContext,
        *,
        require_available_at_observation: bool = True,
    ) -> tuple[FeatureTrainingRow, ...]:
        """
        Construct historical features for entity/timestamp pairs.

        The default mode enforces both event-time and
        ingestion-time availability at the observation time.

        The caller must still ensure labels and any downstream
        transformations are leakage-free.
        """
        self._authorize(
            access,
            "can_read",
        )

        group = self.get_feature_group(
            group_id,
            access,
        )

        results = []

        for entity_id, observation_time in observations:
            record = self.get_features_at(
                group_id,
                entity_id,
                observation_time,
                access,
                available_as_of=(
                    observation_time
                    if require_available_at_observation
                    else None
                ),
            )

            features = (
                record.values
                if record is not None
                else {
                    name: None
                    for name in group.feature_names
                }
            )

            results.append(
                FeatureTrainingRow(
                    entity_id=entity_id,
                    observation_timestamp=ensure_utc(
                        observation_time
                    ),
                    features=dict(features),
                )
            )

        return tuple(results)

    # --------------------------------------------------------
    # FEATURE FRESHNESS
    # --------------------------------------------------------

    def check_freshness(
        self,
        group_id: str,
        entity_id: str,
        access: FeatureAccessContext,
        *,
        as_of: datetime | None = None,
    ) -> FeatureFreshnessResult:
        group = self.get_feature_group(
            group_id,
            access,
        )

        reference_time = (
            ensure_utc(as_of)
            if as_of is not None
            else utc_now()
        )

        record = self.get_online_features(
            group_id,
            entity_id,
            access,
        )

        if record is None:
            return FeatureFreshnessResult(
                group_id=group_id,
                entity_id=entity_id,
                status=FeatureFreshnessStatus.MISSING,
                latest_event_timestamp=None,
                age_seconds=None,
                threshold_seconds=group.freshness_seconds,
            )

        age = (
            reference_time
            - record.event_timestamp
        ).total_seconds()

        status = (
            FeatureFreshnessStatus.FRESH
            if 0 <= age <= group.freshness_seconds
            else FeatureFreshnessStatus.STALE
        )

        return FeatureFreshnessResult(
            group_id=group_id,
            entity_id=entity_id,
            status=status,
            latest_event_timestamp=record.event_timestamp,
            age_seconds=age,
            threshold_seconds=group.freshness_seconds,
        )

    # --------------------------------------------------------
    # FEATURE QUALITY
    # --------------------------------------------------------

    def validate_features(
        self,
        group_id: str,
        entity_id: str,
        values: Mapping[str, Any],
        access: FeatureAccessContext,
    ) -> FeatureQualityResult:
        group = self.get_feature_group(
            group_id,
            access,
        )

        return self.validator.quality_check(
            group,
            entity_id,
            values,
        )

    # --------------------------------------------------------
    # FEATURE STATISTICS
    # --------------------------------------------------------

    def feature_statistics(
        self,
        group_id: str,
        feature_name: str,
        access: FeatureAccessContext,
        *,
        entity_id: str | None = None,
    ) -> FeatureStatistics:
        group = self.get_feature_group(
            group_id,
            access,
        )

        definitions = {
            feature.name: feature
            for feature in group.features
        }

        if feature_name not in definitions:
            raise FeatureNotFoundError(
                "Feature not found in group"
            )

        query = """
            SELECT values_json
            FROM offline_features
            WHERE tenant_id = ?
            AND group_id = ?
        """

        parameters: list[Any] = [
            access.tenant_id,
            group_id,
        ]

        if entity_id is not None:
            query += " AND entity_id = ?"
            parameters.append(entity_id)

        with self._lock:
            rows = self._connection.execute(
                query,
                parameters,
            ).fetchall()

        values = [
            json.loads(
                row["values_json"]
            ).get(feature_name)
            for row in rows
        ]

        missing = sum(
            value is None
            for value in values
        )

        numeric = []

        if definitions[feature_name].data_type in (
            FeatureDataType.INTEGER,
            FeatureDataType.FLOAT,
        ):
            numeric = [
                float(value)
                for value in values
                if (
                    value is not None
                    and not isinstance(value, bool)
                )
            ]

        return FeatureStatistics(
            feature_name=feature_name,
            sample_count=len(values),
            missing_count=missing,
            minimum=(
                min(numeric)
                if numeric
                else None
            ),
            maximum=(
                max(numeric)
                if numeric
                else None
            ),
            mean=(
                statistics.fmean(numeric)
                if numeric
                else None
            ),
            standard_deviation=(
                statistics.pstdev(numeric)
                if numeric
                else None
            ),
        )

    # --------------------------------------------------------
    # FEATURE LINEAGE
    # --------------------------------------------------------

    def get_feature_lineage(
        self,
        group_id: str,
        access: FeatureAccessContext,
    ) -> dict[str, Any]:
        group = self.get_feature_group(
            group_id,
            access,
        )

        return {
            "group_id": group.group_id,
            "group_name": group.name,
            "group_version": group.version,
            "entity_type": group.entity_type,
            "dataset_id": group.dataset_id,
            "dataset_version_id": group.dataset_version_id,
            "source_name": group.source_name,
            "feature_names": list(
                group.feature_names
            ),
            "metadata": group.metadata,
        }

    # --------------------------------------------------------
    # EXPORT CSV
    # --------------------------------------------------------

    def export_history_csv(
        self,
        group_id: str,
        entity_id: str,
        access: FeatureAccessContext,
        *,
        limit: int = 10000,
    ) -> str:
        group = self.get_feature_group(
            group_id,
            access,
        )

        records = self.get_feature_history(
            group_id,
            entity_id,
            access,
            limit=limit,
        )

        output = io.StringIO()

        writer = csv.writer(
            output
        )

        writer.writerow(
            [
                "record_id",
                "entity_id",
                "event_timestamp",
                "ingestion_timestamp",
                *group.feature_names,
            ]
        )

        for record in records:
            writer.writerow(
                [
                    record.record_id,
                    record.entity_id,
                    timestamp_text(
                        record.event_timestamp
                    ),
                    timestamp_text(
                        record.ingestion_timestamp
                    ),
                    *[
                        (
                            json_text(
                                record.values.get(name)
                            )
                            if isinstance(
                                record.values.get(name),
                                (list, dict),
                            )
                            else record.values.get(name)
                        )
                        for name in group.feature_names
                    ],
                ]
            )

        return output.getvalue()

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        access: FeatureAccessContext,
        *,
        group_id: str | None = None,
        limit: int = 1000,
    ) -> tuple[dict[str, Any], ...]:
        self._authorize(
            access,
            "can_read",
        )

        if limit <= 0:
            raise FeatureValidationError(
                "limit must be positive"
            )

        query = """
            SELECT *
            FROM feature_audit
            WHERE tenant_id = ?
        """

        parameters: list[Any] = [
            access.tenant_id
        ]

        if group_id is not None:
            query += " AND group_id = ?"
            parameters.append(group_id)

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
                    "group_id": row["group_id"],
                    "entity_id": row["entity_id"],
                    "timestamp": row["timestamp"],
                    "details": json.loads(
                        row["details_json"]
                    ),
                }
                for row in rows
            )


# ============================================================
# SYNTHETIC ROBOTICS FEATURE FACTORY
# ============================================================

class SyntheticRoboticsFeatureFactory:

    @staticmethod
    def definitions() -> tuple[
        FeatureDefinition,
        ...
    ]:
        return (
            FeatureDefinition(
                name="motor_temperature",
                data_type=FeatureDataType.FLOAT,
                unit="celsius",
                minimum=-40,
                maximum=200,
            ),
            FeatureDefinition(
                name="battery_soc",
                data_type=FeatureDataType.FLOAT,
                unit="percent",
                minimum=0,
                maximum=100,
            ),
            FeatureDefinition(
                name="vibration_rms",
                data_type=FeatureDataType.FLOAT,
                unit="mm/s",
                minimum=0,
            ),
            FeatureDefinition(
                name="robot_speed",
                data_type=FeatureDataType.FLOAT,
                unit="m/s",
                minimum=0,
            ),
            FeatureDefinition(
                name="operating_mode",
                data_type=FeatureDataType.CATEGORY,
                allowed_categories=(
                    "idle",
                    "navigation",
                    "inspection",
                    "charging",
                    "emergency",
                ),
            ),
            FeatureDefinition(
                name="vision_embedding",
                data_type=FeatureDataType.FLOAT_VECTOR,
                vector_dimension=4,
            ),
            FeatureDefinition(
                name="obstacle_detected",
                data_type=FeatureDataType.BOOLEAN,
            ),
        )

    @staticmethod
    def generate(
        entity_id: str,
        samples: int = 100,
        *,
        start_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        if samples <= 0:
            raise FeatureValidationError(
                "samples must be positive"
            )

        start = (
            ensure_utc(start_time)
            if start_time is not None
            else utc_now() - timedelta(
                seconds=samples
            )
        )

        result = []

        for index in range(samples):
            result.append(
                {
                    "entity_id": entity_id,
                    "event_timestamp": (
                        start
                        + timedelta(
                            seconds=index
                        )
                    ),
                    "values": {
                        "motor_temperature": (
                            40.0
                            + math.sin(index * 0.1) * 3.0
                        ),
                        "battery_soc": max(
                            0.0,
                            95.0 - index * 0.1,
                        ),
                        "vibration_rms": (
                            0.2
                            + abs(
                                math.sin(index * 0.15)
                            ) * 0.05
                        ),
                        "robot_speed": (
                            1.0
                            + math.sin(index * 0.08) * 0.2
                        ),
                        "operating_mode": (
                            "navigation"
                            if index % 5
                            else "inspection"
                        ),
                        "vision_embedding": [
                            math.sin(index * 0.1),
                            math.cos(index * 0.1),
                            math.sin(index * 0.2),
                            math.cos(index * 0.2),
                        ],
                        "obstacle_detected": (
                            index % 7 == 0
                        ),
                    },
                    "source_id": "synthetic-simulator",
                }
            )

        return result


# ============================================================
# UNIT TESTS
# ============================================================

class TestFeatureStore(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

        self.config = FeatureStoreConfig(
            database_path=(
                Path(self.temporary.name)
                / "feature_store.sqlite3"
            )
        )

        self.store = FeatureStore(
            self.config
        )

        self.access = FeatureAccessContext(
            tenant_id="robotics_lab",
            user_id="mlops_engineer",
            can_read=True,
            can_write=True,
            can_manage=True,
        )

        self.group = self.store.create_feature_group(
            name="robot_telemetry",
            group_type=FeatureGroupType.ROBOT_TELEMETRY,
            entity_type="robot",
            features=(
                SyntheticRoboticsFeatureFactory.definitions()
            ),
            access=self.access,
            freshness_seconds=300,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def _sample(
        self,
        *,
        samples: int = 10,
        start_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        return SyntheticRoboticsFeatureFactory.generate(
            "robot-001",
            samples,
            start_time=start_time,
        )

    def test_create_group(self) -> None:
        self.assertEqual(
            self.group.version,
            1,
        )

    def test_group_versioning(self) -> None:
        second = self.store.create_feature_group(
            name="robot_telemetry",
            group_type=FeatureGroupType.ROBOT_TELEMETRY,
            entity_type="robot",
            features=(
                SyntheticRoboticsFeatureFactory.definitions()
            ),
            access=self.access,
        )

        self.assertEqual(
            second.version,
            2,
        )

    def test_ingest_batch(self) -> None:
        records = self.store.ingest_batch(
            self.group.group_id,
            self._sample(),
            self.access,
        )

        self.assertEqual(
            len(records),
            10,
        )

    def test_online_retrieval(self) -> None:
        records = self.store.ingest_batch(
            self.group.group_id,
            self._sample(),
            self.access,
        )

        online = self.store.get_online_features(
            self.group.group_id,
            "robot-001",
            self.access,
        )

        self.assertIsNotNone(
            online
        )

        self.assertEqual(
            online.record_id,
            records[-1].record_id,
        )

    def test_point_in_time(self) -> None:
        start = utc_now() - timedelta(
            minutes=5
        )

        self.store.ingest_batch(
            self.group.group_id,
            self._sample(
                samples=10,
                start_time=start,
            ),
            self.access,
        )

        result = self.store.get_features_at(
            self.group.group_id,
            "robot-001",
            start + timedelta(
                seconds=4
            ),
            self.access,
        )

        self.assertIsNotNone(
            result
        )

        self.assertEqual(
            result.event_timestamp,
            start + timedelta(
                seconds=4
            ),
        )

    def test_history(self) -> None:
        self.store.ingest_batch(
            self.group.group_id,
            self._sample(),
            self.access,
        )

        history = self.store.get_feature_history(
            self.group.group_id,
            "robot-001",
            self.access,
        )

        self.assertEqual(
            len(history),
            10,
        )

    def test_invalid_temperature(self) -> None:
        item = self._sample(
            samples=1
        )[0]

        item["values"][
            "motor_temperature"
        ] = 500.0

        with self.assertRaises(
            FeatureValidationError
        ):
            self.store.ingest_batch(
                self.group.group_id,
                [item],
                self.access,
            )

    def test_invalid_embedding_dimension(self) -> None:
        item = self._sample(
            samples=1
        )[0]

        item["values"][
            "vision_embedding"
        ] = [0.1, 0.2]

        with self.assertRaises(
            FeatureValidationError
        ):
            self.store.ingest_batch(
                self.group.group_id,
                [item],
                self.access,
            )

    def test_quality_check(self) -> None:
        item = self._sample(
            samples=1
        )[0]

        result = self.store.validate_features(
            self.group.group_id,
            "robot-001",
            item["values"],
            self.access,
        )

        self.assertTrue(
            result.valid
        )

    def test_freshness(self) -> None:
        self.store.ingest_batch(
            self.group.group_id,
            self._sample(
                samples=1,
                start_time=utc_now(),
            ),
            self.access,
        )

        result = self.store.check_freshness(
            self.group.group_id,
            "robot-001",
            self.access,
        )

        self.assertEqual(
            result.status,
            FeatureFreshnessStatus.FRESH,
        )

    def test_stale_features(self) -> None:
        self.store.ingest_batch(
            self.group.group_id,
            self._sample(
                samples=1,
                start_time=(
                    utc_now()
                    - timedelta(
                        hours=1
                    )
                ),
            ),
            self.access,
        )

        result = self.store.check_freshness(
            self.group.group_id,
            "robot-001",
            self.access,
        )

        self.assertEqual(
            result.status,
            FeatureFreshnessStatus.STALE,
        )

    def test_statistics(self) -> None:
        self.store.ingest_batch(
            self.group.group_id,
            self._sample(),
            self.access,
        )

        result = self.store.feature_statistics(
            self.group.group_id,
            "motor_temperature",
            self.access,
        )

        self.assertEqual(
            result.sample_count,
            10,
        )

        self.assertIsNotNone(
            result.mean
        )

    def test_training_dataset(self) -> None:
        start = utc_now() - timedelta(
            minutes=10
        )

        self.store.ingest_batch(
            self.group.group_id,
            self._sample(
                samples=5,
                start_time=start,
            ),
            self.access,
        )

        rows = self.store.build_training_dataset(
            self.group.group_id,
            [
                (
                    "robot-001",
                    start + timedelta(
                        seconds=3
                    ),
                )
            ],
            self.access,
            require_available_at_observation=False,
        )

        self.assertEqual(
            len(rows),
            1,
        )

        self.assertIsNotNone(
            rows[0].features[
                "motor_temperature"
            ]
        )

    def test_training_ingestion_cutoff(self) -> None:
        start = utc_now() - timedelta(
            minutes=10
        )

        self.store.ingest_batch(
            self.group.group_id,
            self._sample(
                samples=5,
                start_time=start,
            ),
            self.access,
        )

        rows = self.store.build_training_dataset(
            self.group.group_id,
            [
                (
                    "robot-001",
                    start + timedelta(
                        seconds=3
                    ),
                )
            ],
            self.access,
        )

        self.assertIsNone(
            rows[0].features[
                "motor_temperature"
            ]
        )

    def test_online_materialization(self) -> None:
        self.store.ingest_batch(
            self.group.group_id,
            self._sample(),
            self.access,
            materialize_online=False,
        )

        before = self.store.get_online_features(
            self.group.group_id,
            "robot-001",
            self.access,
        )

        self.assertIsNone(
            before
        )

        count = self.store.materialize_online(
            self.group.group_id,
            self.access,
        )

        self.assertEqual(
            count,
            1,
        )

    def test_tenant_isolation(self) -> None:
        other = FeatureAccessContext(
            tenant_id="another_tenant",
            user_id="other_user",
            can_read=True,
        )

        with self.assertRaises(
            FeatureNotFoundError
        ):
            self.store.get_feature_group(
                self.group.group_id,
                other,
            )

    def test_csv_export(self) -> None:
        self.store.ingest_batch(
            self.group.group_id,
            self._sample(),
            self.access,
        )

        output = self.store.export_history_csv(
            self.group.group_id,
            "robot-001",
            self.access,
        )

        self.assertIn(
            "motor_temperature",
            output,
        )

    def test_lineage(self) -> None:
        lineage = self.store.get_feature_lineage(
            self.group.group_id,
            self.access,
        )

        self.assertEqual(
            lineage["group_name"],
            "robot_telemetry",
        )

    def test_audit(self) -> None:
        history = self.store.audit_history(
            self.access,
        )

        self.assertGreaterEqual(
            len(history),
            1,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        config = FeatureStoreConfig(
            database_path=(
                Path(temporary)
                / "feature_store.sqlite3"
            )
        )

        with FeatureStore(config) as store:
            access = FeatureAccessContext(
                tenant_id="physical_ai_lab",
                user_id="mlops_engineer",
                can_read=True,
                can_write=True,
                can_manage=True,
            )

            group = store.create_feature_group(
                name="autonomous_robot_features",
                group_type=FeatureGroupType.ROBOT_TELEMETRY,
                entity_type="robot",
                features=(
                    SyntheticRoboticsFeatureFactory.definitions()
                ),
                access=access,
                freshness_seconds=300,
                dataset_id="robotics-telemetry-dataset",
                dataset_version_id="v1",
                source_name="robotics-simulator",
                description=(
                    "Telemetry and perception features "
                    "for autonomous robot AI models"
                ),
            )

            records = (
                SyntheticRoboticsFeatureFactory.generate(
                    entity_id="robot-001",
                    samples=100,
                )
            )

            ingested = store.ingest_batch(
                group.group_id,
                records,
                access,
            )

            online = store.get_online_features(
                group.group_id,
                "robot-001",
                access,
            )

            freshness = store.check_freshness(
                group.group_id,
                "robot-001",
                access,
            )

            statistics_result = (
                store.feature_statistics(
                    group.group_id,
                    "motor_temperature",
                    access,
                )
            )

            lineage = store.get_feature_lineage(
                group.group_id,
                access,
            )

            print(
                "\nPHYSICAL AI FEATURE STORE"
            )
            print("=" * 70)

            print(
                json.dumps(
                    {
                        "feature_group": {
                            "group_id": group.group_id,
                            "name": group.name,
                            "version": group.version,
                            "type": group.group_type.value,
                            "features": list(
                                group.feature_names
                            ),
                        },
                        "ingested_records": len(ingested),
                        "online_entity": (
                            online.entity_id
                            if online
                            else None
                        ),
                        "online_features": (
                            online.values
                            if online
                            else None
                        ),
                        "freshness": {
                            "status": freshness.status.value,
                            "age_seconds": freshness.age_seconds,
                        },
                        "temperature_statistics": {
                            "mean": statistics_result.mean,
                            "minimum": statistics_result.minimum,
                            "maximum": statistics_result.maximum,
                        },
                        "lineage": lineage,
                        "synthetic_data": True,
                        "physical_robot_control": False,
                    },
                    indent=2,
                    allow_nan=False,
                )
            )

    print(
        "\nRUNNING UNIT TESTS"
    )
    print("=" * 70)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestFeatureStore
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
"""Model Rollback module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/mlops/model_rollback.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
MLOps — Model Rollback Manager

Features
--------
- Model deployment rollback planning
- Previous stable version recovery
- Manual and automatic rollback recommendations
- Canary deployment rollback
- Blue-green deployment rollback
- Rolling deployment rollback
- Model artifact integrity verification
- Model registry integration through adapters
- Health-based rollback evaluation
- Latency and error-rate monitoring
- Model accuracy degradation detection
- Model drift-triggered rollback recommendations
- Human approval workflow
- Separation of duties
- Rollback execution through an explicit deployment adapter
- Rollback verification
- Rollback failure handling
- Audit logging
- Multi-tenant isolation
- SQLite persistence
- Thread-safe operations
- Synthetic demonstration
- Unit tests

Python 3.11+
Standard library only.

Safety
------
Rollback is advisory by default.

The manager does not directly control robots, actuators,
navigation controllers, or safety-critical equipment.

A deployment adapter must explicitly implement infrastructure
changes. Production execution requires human approval.

The default adapter is a simulation and never changes
real infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
import threading
import unittest
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol


# ============================================================
# EXCEPTIONS
# ============================================================

class ModelRollbackError(Exception):
    pass


class RollbackValidationError(ModelRollbackError):
    pass


class RollbackNotFoundError(ModelRollbackError):
    pass


class RollbackAuthorizationError(ModelRollbackError):
    pass


class RollbackConflictError(ModelRollbackError):
    pass


class RollbackApprovalError(ModelRollbackError):
    pass


class RollbackIntegrityError(ModelRollbackError):
    pass


class RollbackExecutionError(ModelRollbackError):
    pass


class RollbackVerificationError(ModelRollbackError):
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
        raise RollbackValidationError(
            f"Invalid {field_name}"
        )

    return value.strip()


def validate_number(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise RollbackValidationError(
            f"{field_name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RollbackValidationError(
            f"{field_name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise RollbackValidationError(
            f"{field_name} must be finite"
        )

    if minimum is not None and result < minimum:
        raise RollbackValidationError(
            f"{field_name} must be >= {minimum}"
        )

    if maximum is not None and result > maximum:
        raise RollbackValidationError(
            f"{field_name} must be <= {maximum}"
        )

    return result


def ensure_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise RollbackValidationError(
            "Expected datetime"
        )

    if value.tzinfo is None:
        raise RollbackValidationError(
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

    if isinstance(value, (tuple, list)):
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
        raise RollbackValidationError(
            "Data must be JSON serializable"
        ) from exc


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

class DeploymentEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    EDGE = "edge"


class DeploymentStrategy(str, Enum):
    RECREATE = "recreate"
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"


class RollbackTrigger(str, Enum):
    MANUAL = "manual"
    HIGH_ERROR_RATE = "high_error_rate"
    HIGH_LATENCY = "high_latency"
    ACCURACY_DEGRADATION = "accuracy_degradation"
    MODEL_DRIFT = "model_drift"
    HEALTH_CHECK_FAILURE = "health_check_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SAFETY_REVIEW = "safety_review"


class RollbackStatus(str, Enum):
    PLANNED = "planned"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RollbackMode(str, Enum):
    SIMULATION = "simulation"
    EXTERNAL_ADAPTER = "external_adapter"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ModelRollbackConfig:
    database_path: Path = Path(
        "./data/mlops/model_rollback.sqlite3"
    )

    require_approval: bool = True
    prevent_self_approval: bool = True

    maximum_error_rate: float = 0.05
    maximum_p95_latency_ms: float = 250.0
    minimum_accuracy: float = 0.80
    maximum_drift_score: float = 0.20

    minimum_health_samples: int = 10

    verify_target_artifact: bool = True
    verify_deployment_after_rollback: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "database_path",
            Path(self.database_path).expanduser().resolve(),
        )

        validate_number(
            self.maximum_error_rate,
            "maximum_error_rate",
            minimum=0,
            maximum=1,
        )

        validate_number(
            self.maximum_p95_latency_ms,
            "maximum_p95_latency_ms",
            minimum=0,
        )

        validate_number(
            self.minimum_accuracy,
            "minimum_accuracy",
            minimum=0,
            maximum=1,
        )

        validate_number(
            self.maximum_drift_score,
            "maximum_drift_score",
            minimum=0,
        )

        if self.minimum_health_samples <= 0:
            raise RollbackValidationError(
                "minimum_health_samples must be positive"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class RollbackAccess:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_plan: bool = False
    can_approve: bool = False
    can_execute: bool = False
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
class DeploymentSnapshot:
    deployment_id: str
    tenant_id: str
    model_id: str

    current_version_id: str
    previous_stable_version_id: str | None

    environment: DeploymentEnvironment
    strategy: DeploymentStrategy

    traffic_percentage: float = 100.0

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "deployment_id",
            "tenant_id",
            "model_id",
            "current_version_id",
        ):
            validate_identifier(
                getattr(self, name),
                name,
            )

        validate_number(
            self.traffic_percentage,
            "traffic_percentage",
            minimum=0,
            maximum=100,
        )


@dataclass(frozen=True)
class ModelArtifactReference:
    version_id: str
    model_id: str

    artifact_path: Path
    expected_sha256: str

    framework: str = "unknown"

    def __post_init__(self) -> None:
        validate_identifier(
            self.version_id,
            "version_id",
        )

        validate_identifier(
            self.model_id,
            "model_id",
        )

        if (
            len(self.expected_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.expected_sha256
            )
        ):
            raise RollbackValidationError(
                "Invalid artifact SHA-256"
            )


@dataclass(frozen=True)
class ModelHealthMetrics:
    sample_count: int

    error_rate: float
    p95_latency_ms: float

    accuracy: float | None = None
    drift_score: float | None = None

    cpu_percent: float | None = None
    gpu_percent: float | None = None
    memory_percent: float | None = None

    health_status: HealthStatus = HealthStatus.UNKNOWN

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise RollbackValidationError(
                "sample_count cannot be negative"
            )

        validate_number(
            self.error_rate,
            "error_rate",
            minimum=0,
            maximum=1,
        )

        validate_number(
            self.p95_latency_ms,
            "p95_latency_ms",
            minimum=0,
        )

        for name in (
            "accuracy",
            "drift_score",
            "cpu_percent",
            "gpu_percent",
            "memory_percent",
        ):
            value = getattr(self, name)

            if value is not None:
                validate_number(
                    value,
                    name,
                    minimum=0,
                    maximum=(
                        1
                        if name == "accuracy"
                        else (
                            100
                            if name.endswith("_percent")
                            else None
                        )
                    ),
                )


@dataclass(frozen=True)
class RollbackRecommendation:
    rollback_recommended: bool

    triggers: tuple[RollbackTrigger, ...]

    severity: HealthStatus

    reasons: tuple[str, ...]

    evaluated_at: datetime


@dataclass(frozen=True)
class RollbackPlan:
    rollback_id: str

    tenant_id: str
    deployment_id: str
    model_id: str

    source_version_id: str
    target_version_id: str

    environment: DeploymentEnvironment
    strategy: DeploymentStrategy

    trigger: RollbackTrigger
    status: RollbackStatus

    requested_by: str
    requested_at: datetime

    reason: str

    execution_mode: RollbackMode

    approved_by: str | None = None
    approved_at: datetime | None = None

    executed_at: datetime | None = None
    completed_at: datetime | None = None

    failure_reason: str | None = None


@dataclass(frozen=True)
class RollbackExecutionResult:
    rollback_id: str

    success: bool

    deployment_id: str
    target_version_id: str

    mode: RollbackMode

    started_at: datetime
    completed_at: datetime

    verified: bool

    message: str


# ============================================================
# HEALTH EVALUATOR
# ============================================================

class RollbackHealthEvaluator:

    @staticmethod
    def evaluate(
        metrics: ModelHealthMetrics,
        config: ModelRollbackConfig,
    ) -> RollbackRecommendation:
        if metrics.sample_count < config.minimum_health_samples:
            return RollbackRecommendation(
                rollback_recommended=False,
                triggers=(),
                severity=HealthStatus.UNKNOWN,
                reasons=(
                    "Insufficient monitoring samples",
                ),
                evaluated_at=utc_now(),
            )

        triggers: list[RollbackTrigger] = []
        reasons: list[str] = []

        if metrics.error_rate > config.maximum_error_rate:
            triggers.append(
                RollbackTrigger.HIGH_ERROR_RATE
            )
            reasons.append(
                "Inference error rate exceeds threshold"
            )

        if (
            metrics.p95_latency_ms
            > config.maximum_p95_latency_ms
        ):
            triggers.append(
                RollbackTrigger.HIGH_LATENCY
            )
            reasons.append(
                "P95 inference latency exceeds threshold"
            )

        if (
            metrics.accuracy is not None
            and metrics.accuracy < config.minimum_accuracy
        ):
            triggers.append(
                RollbackTrigger.ACCURACY_DEGRADATION
            )
            reasons.append(
                "Model accuracy below threshold"
            )

        if (
            metrics.drift_score is not None
            and metrics.drift_score
            > config.maximum_drift_score
        ):
            triggers.append(
                RollbackTrigger.MODEL_DRIFT
            )
            reasons.append(
                "Model drift indicator exceeds threshold"
            )

        if metrics.health_status == HealthStatus.CRITICAL:
            triggers.append(
                RollbackTrigger.HEALTH_CHECK_FAILURE
            )
            reasons.append(
                "Deployment health is critical"
            )

        if any(
            value is not None and value > 95
            for value in (
                metrics.cpu_percent,
                metrics.gpu_percent,
                metrics.memory_percent,
            )
        ):
            triggers.append(
                RollbackTrigger.RESOURCE_EXHAUSTION
            )
            reasons.append(
                "Compute resource utilization exceeds 95%"
            )

        recommended = bool(triggers)

        severity = (
            HealthStatus.CRITICAL
            if (
                RollbackTrigger.HIGH_ERROR_RATE in triggers
                or RollbackTrigger.HEALTH_CHECK_FAILURE
                in triggers
            )
            else (
                HealthStatus.DEGRADED
                if recommended
                else HealthStatus.HEALTHY
            )
        )

        return RollbackRecommendation(
            rollback_recommended=recommended,
            triggers=tuple(dict.fromkeys(triggers)),
            severity=severity,
            reasons=tuple(reasons),
            evaluated_at=utc_now(),
        )


# ============================================================
# ARTIFACT VERIFIER
# ============================================================

class RollbackArtifactVerifier:

    @staticmethod
    def verify(
        artifact: ModelArtifactReference,
    ) -> bool:
        path = Path(
            artifact.artifact_path
        )

        if path.is_symlink() or not path.is_file():
            return False

        actual_sha256 = sha256_file(
            path
        )

        return (
            actual_sha256
            == artifact.expected_sha256
        )


# ============================================================
# DEPLOYMENT ADAPTER CONTRACT
# ============================================================

class RollbackDeploymentAdapter(Protocol):

    def get_deployment(
        self,
        deployment_id: str,
        tenant_id: str,
    ) -> DeploymentSnapshot:
        ...

    def rollback(
        self,
        deployment_id: str,
        tenant_id: str,
        target_version_id: str,
        strategy: DeploymentStrategy,
    ) -> bool:
        ...

    def verify_version(
        self,
        deployment_id: str,
        tenant_id: str,
        expected_version_id: str,
    ) -> bool:
        ...


# ============================================================
# SIMULATED DEPLOYMENT ADAPTER
# ============================================================

class SimulatedRollbackAdapter:
    """
    In-memory deployment adapter.

    Suitable for tests and demonstrations only.
    Does not change real infrastructure.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self._deployments: dict[
            tuple[str, str],
            DeploymentSnapshot,
        ] = {}

    def register_deployment(
        self,
        deployment: DeploymentSnapshot,
    ) -> None:
        key = (
            deployment.tenant_id,
            deployment.deployment_id,
        )

        with self._lock:
            if key in self._deployments:
                raise RollbackConflictError(
                    "Deployment already registered"
                )

            self._deployments[key] = deployment

    def get_deployment(
        self,
        deployment_id: str,
        tenant_id: str,
    ) -> DeploymentSnapshot:
        key = (
            tenant_id,
            deployment_id,
        )

        with self._lock:
            deployment = self._deployments.get(
                key
            )

            if deployment is None:
                raise RollbackNotFoundError(
                    "Deployment not found"
                )

            return deployment

    def rollback(
        self,
        deployment_id: str,
        tenant_id: str,
        target_version_id: str,
        strategy: DeploymentStrategy,
    ) -> bool:
        key = (
            tenant_id,
            deployment_id,
        )

        with self._lock:
            deployment = self.get_deployment(
                deployment_id,
                tenant_id,
            )

            self._deployments[key] = DeploymentSnapshot(
                deployment_id=deployment.deployment_id,
                tenant_id=deployment.tenant_id,
                model_id=deployment.model_id,
                current_version_id=target_version_id,
                previous_stable_version_id=(
                    deployment.current_version_id
                ),
                environment=deployment.environment,
                strategy=strategy,
                traffic_percentage=100.0,
            )

        return True

    def verify_version(
        self,
        deployment_id: str,
        tenant_id: str,
        expected_version_id: str,
    ) -> bool:
        deployment = self.get_deployment(
            deployment_id,
            tenant_id,
        )

        return (
            deployment.current_version_id
            == expected_version_id
        )


# ============================================================
# MODEL ROLLBACK MANAGER
# ============================================================

class ModelRollbackManager:

    def __init__(
        self,
        config: ModelRollbackConfig | None = None,
        *,
        deployment_adapter: (
            RollbackDeploymentAdapter | None
        ) = None,
    ) -> None:
        self.config = (
            config or ModelRollbackConfig()
        )

        self.adapter = (
            deployment_adapter
            if deployment_adapter is not None
            else SimulatedRollbackAdapter()
        )

        self._lock = threading.RLock()

        self.config.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
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

    def __enter__(self) -> ModelRollbackManager:
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
        access: RollbackAccess,
        permission: str,
    ) -> None:
        if not getattr(
            access,
            permission,
            False,
        ):
            raise RollbackAuthorizationError(
                f"Missing permission: {permission}"
            )

    @staticmethod
    def _read_access(
        access: RollbackAccess,
    ) -> RollbackAccess:
        return RollbackAccess(
            tenant_id=access.tenant_id,
            user_id=access.user_id,
            can_read=True,
        )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    def _initialize_database(self) -> None:
        with self._lock, self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS rollback_plans (
                    rollback_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    deployment_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    source_version_id TEXT NOT NULL,
                    target_version_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    approved_by TEXT,
                    approved_at TEXT,
                    executed_at TEXT,
                    completed_at TEXT,
                    failure_reason TEXT
                );

                CREATE INDEX IF NOT EXISTS
                idx_rollback_deployment
                ON rollback_plans (
                    tenant_id,
                    deployment_id,
                    requested_at
                );

                CREATE TABLE IF NOT EXISTS rollback_artifacts (
                    rollback_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    expected_sha256 TEXT NOT NULL,
                    framework TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rollback_audit (
                    audit_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    rollback_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
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
        access: RollbackAccess,
        rollback_id: str,
        action: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO rollback_audit (
                audit_id,
                tenant_id,
                rollback_id,
                user_id,
                action,
                timestamp,
                details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                generate_id(),
                access.tenant_id,
                rollback_id,
                access.user_id,
                action,
                timestamp(utc_now()),
                json_encode(dict(details or {})),
            ),
        )

    # --------------------------------------------------------
    # ROW CONVERSION
    # --------------------------------------------------------

    @staticmethod
    def _row_to_plan(
        row: sqlite3.Row,
    ) -> RollbackPlan:
        return RollbackPlan(
            rollback_id=row["rollback_id"],
            tenant_id=row["tenant_id"],
            deployment_id=row["deployment_id"],
            model_id=row["model_id"],
            source_version_id=row[
                "source_version_id"
            ],
            target_version_id=row[
                "target_version_id"
            ],
            environment=DeploymentEnvironment(
                row["environment"]
            ),
            strategy=DeploymentStrategy(
                row["strategy"]
            ),
            trigger=RollbackTrigger(
                row["trigger_type"]
            ),
            status=RollbackStatus(
                row["status"]
            ),
            requested_by=row["requested_by"],
            requested_at=parse_timestamp(
                row["requested_at"]
            ),
            reason=row["reason"],
            execution_mode=RollbackMode(
                row["execution_mode"]
            ),
            approved_by=row["approved_by"],
            approved_at=(
                parse_timestamp(
                    row["approved_at"]
                )
                if row["approved_at"]
                else None
            ),
            executed_at=(
                parse_timestamp(
                    row["executed_at"]
                )
                if row["executed_at"]
                else None
            ),
            completed_at=(
                parse_timestamp(
                    row["completed_at"]
                )
                if row["completed_at"]
                else None
            ),
            failure_reason=row["failure_reason"],
        )

    # --------------------------------------------------------
    # GET PLAN
    # --------------------------------------------------------

    def get_plan(
        self,
        rollback_id: str,
        access: RollbackAccess,
    ) -> RollbackPlan:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            row = self._db.execute(
                """
                SELECT *
                FROM rollback_plans
                WHERE tenant_id = ?
                AND rollback_id = ?
                """,
                (
                    access.tenant_id,
                    rollback_id,
                ),
            ).fetchone()

        if row is None:
            raise RollbackNotFoundError(
                "Rollback plan not found"
            )

        return self._row_to_plan(
            row
        )

    # --------------------------------------------------------
    # HEALTH EVALUATION
    # --------------------------------------------------------

    def evaluate_health(
        self,
        metrics: ModelHealthMetrics,
    ) -> RollbackRecommendation:
        return RollbackHealthEvaluator.evaluate(
            metrics,
            self.config,
        )

    # --------------------------------------------------------
    # CREATE ROLLBACK PLAN
    # --------------------------------------------------------

    def create_plan(
        self,
        deployment_id: str,
        access: RollbackAccess,
        *,
        target_version_id: str | None = None,
        trigger: RollbackTrigger = RollbackTrigger.MANUAL,
        reason: str = "",
        artifact: ModelArtifactReference | None = None,
        execution_mode: RollbackMode = (
            RollbackMode.SIMULATION
        ),
    ) -> RollbackPlan:
        self._authorize(
            access,
            "can_plan",
        )

        deployment = self.adapter.get_deployment(
            deployment_id,
            access.tenant_id,
        )

        target = (
            target_version_id
            or deployment.previous_stable_version_id
        )

        if target is None:
            raise RollbackValidationError(
                "No previous stable model version available"
            )

        validate_identifier(
            target,
            "target_version_id",
        )

        if target == deployment.current_version_id:
            raise RollbackConflictError(
                "Target is already the active version"
            )

        if artifact is not None:
            if (
                artifact.version_id != target
                or artifact.model_id != deployment.model_id
            ):
                raise RollbackValidationError(
                    "Artifact does not match rollback target"
                )

        execution_mode = RollbackMode(
            execution_mode
        )

        if (
            execution_mode == RollbackMode.EXTERNAL_ADAPTER
            and isinstance(
                self.adapter,
                SimulatedRollbackAdapter,
            )
        ):
            raise RollbackValidationError(
                "External execution requires a real adapter"
            )

        status = (
            RollbackStatus.PENDING_APPROVAL
            if self.config.require_approval
            else RollbackStatus.PLANNED
        )

        plan = RollbackPlan(
            rollback_id=generate_id(),
            tenant_id=access.tenant_id,
            deployment_id=deployment.deployment_id,
            model_id=deployment.model_id,
            source_version_id=deployment.current_version_id,
            target_version_id=target,
            environment=deployment.environment,
            strategy=deployment.strategy,
            trigger=RollbackTrigger(trigger),
            status=status,
            requested_by=access.user_id,
            requested_at=utc_now(),
            reason=reason,
            execution_mode=execution_mode,
        )

        with self._lock, self._db:
            existing = self._db.execute(
                """
                SELECT rollback_id
                FROM rollback_plans
                WHERE tenant_id = ?
                AND deployment_id = ?
                AND status IN (
                    'planned',
                    'pending_approval',
                    'approved',
                    'executing',
                    'verifying'
                )
                """,
                (
                    access.tenant_id,
                    deployment_id,
                ),
            ).fetchone()

            if existing is not None:
                raise RollbackConflictError(
                    "An active rollback plan already exists"
                )

            self._db.execute(
                """
                INSERT INTO rollback_plans (
                    rollback_id,
                    tenant_id,
                    deployment_id,
                    model_id,
                    source_version_id,
                    target_version_id,
                    environment,
                    strategy,
                    trigger_type,
                    status,
                    requested_by,
                    requested_at,
                    reason,
                    execution_mode
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    plan.rollback_id,
                    plan.tenant_id,
                    plan.deployment_id,
                    plan.model_id,
                    plan.source_version_id,
                    plan.target_version_id,
                    plan.environment.value,
                    plan.strategy.value,
                    plan.trigger.value,
                    plan.status.value,
                    plan.requested_by,
                    timestamp(plan.requested_at),
                    plan.reason,
                    plan.execution_mode.value,
                ),
            )

            if artifact is not None:
                self._db.execute(
                    """
                    INSERT INTO rollback_artifacts (
                        rollback_id,
                        version_id,
                        model_id,
                        artifact_path,
                        expected_sha256,
                        framework
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plan.rollback_id,
                        artifact.version_id,
                        artifact.model_id,
                        str(artifact.artifact_path),
                        artifact.expected_sha256,
                        artifact.framework,
                    ),
                )

            self._audit(
                access,
                plan.rollback_id,
                "rollback_plan_created",
                {
                    "source": plan.source_version_id,
                    "target": plan.target_version_id,
                    "trigger": plan.trigger.value,
                    "mode": plan.execution_mode.value,
                },
            )

        return plan

    # --------------------------------------------------------
    # APPROVE / REJECT
    # --------------------------------------------------------

    def decide_plan(
        self,
        rollback_id: str,
        decision: ApprovalDecision,
        access: RollbackAccess,
        *,
        reason: str = "",
    ) -> RollbackPlan:
        self._authorize(
            access,
            "can_approve",
        )

        with self._lock, self._db:
            plan = self.get_plan(
                rollback_id,
                self._read_access(access),
            )

            if plan.status != RollbackStatus.PENDING_APPROVAL:
                raise RollbackApprovalError(
                    "Plan is not pending approval"
                )

            if (
                self.config.prevent_self_approval
                and plan.requested_by == access.user_id
            ):
                raise RollbackApprovalError(
                    "Requester cannot approve own rollback"
                )

            decision = ApprovalDecision(
                decision
            )

            new_status = (
                RollbackStatus.APPROVED
                if decision == ApprovalDecision.APPROVE
                else RollbackStatus.REJECTED
            )

            self._db.execute(
                """
                UPDATE rollback_plans
                SET status = ?,
                    approved_by = ?,
                    approved_at = ?
                WHERE tenant_id = ?
                AND rollback_id = ?
                """,
                (
                    new_status.value,
                    access.user_id,
                    timestamp(utc_now()),
                    access.tenant_id,
                    rollback_id,
                ),
            )

            self._audit(
                access,
                rollback_id,
                "rollback_decision",
                {
                    "decision": decision.value,
                    "reason": reason,
                },
            )

        return self.get_plan(
            rollback_id,
            self._read_access(access),
        )

    # --------------------------------------------------------
    # VERIFY TARGET ARTIFACT
    # --------------------------------------------------------

    def verify_target_artifact(
        self,
        rollback_id: str,
        access: RollbackAccess,
    ) -> bool:
        plan = self.get_plan(
            rollback_id,
            access,
        )

        with self._lock:
            row = self._db.execute(
                """
                SELECT *
                FROM rollback_artifacts
                WHERE rollback_id = ?
                """,
                (rollback_id,),
            ).fetchone()

        if row is None:
            raise RollbackIntegrityError(
                "No target artifact reference registered"
            )

        artifact = ModelArtifactReference(
            version_id=row["version_id"],
            model_id=row["model_id"],
            artifact_path=Path(
                row["artifact_path"]
            ),
            expected_sha256=row[
                "expected_sha256"
            ],
            framework=row["framework"],
        )

        if (
            artifact.version_id != plan.target_version_id
            or artifact.model_id != plan.model_id
        ):
            raise RollbackIntegrityError(
                "Artifact reference does not match plan"
            )

        return RollbackArtifactVerifier.verify(
            artifact
        )

    # --------------------------------------------------------
    # STATUS UPDATE
    # --------------------------------------------------------

    def _set_status(
        self,
        rollback_id: str,
        access: RollbackAccess,
        status: RollbackStatus,
        *,
        failure_reason: str | None = None,
    ) -> None:
        executed_at = (
            timestamp(utc_now())
            if status == RollbackStatus.EXECUTING
            else None
        )

        completed_at = (
            timestamp(utc_now())
            if status in (
                RollbackStatus.COMPLETED,
                RollbackStatus.FAILED,
            )
            else None
        )

        self._db.execute(
            """
            UPDATE rollback_plans
            SET status = ?,
                executed_at = COALESCE(?, executed_at),
                completed_at = COALESCE(?, completed_at),
                failure_reason = ?
            WHERE tenant_id = ?
            AND rollback_id = ?
            """,
            (
                status.value,
                executed_at,
                completed_at,
                failure_reason,
                access.tenant_id,
                rollback_id,
            ),
        )

    # --------------------------------------------------------
    # EXECUTE ROLLBACK
    # --------------------------------------------------------

    def execute(
        self,
        rollback_id: str,
        access: RollbackAccess,
    ) -> RollbackExecutionResult:
        self._authorize(
            access,
            "can_execute",
        )

        with self._lock:
            plan = self.get_plan(
                rollback_id,
                self._read_access(access),
            )

            allowed_status = (
                RollbackStatus.APPROVED
                if self.config.require_approval
                else RollbackStatus.PLANNED
            )

            if plan.status != allowed_status:
                raise RollbackApprovalError(
                    "Rollback is not authorized for execution"
                )

            deployment = self.adapter.get_deployment(
                plan.deployment_id,
                access.tenant_id,
            )

            if (
                deployment.model_id != plan.model_id
                or deployment.current_version_id
                != plan.source_version_id
            ):
                raise RollbackConflictError(
                    "Deployment state changed since planning"
                )

            if self.config.verify_target_artifact:
                if not self.verify_target_artifact(
                    rollback_id,
                    self._read_access(access),
                ):
                    raise RollbackIntegrityError(
                        "Rollback target artifact failed verification"
                    )

            started_at = utc_now()

            with self._db:
                self._set_status(
                    rollback_id,
                    access,
                    RollbackStatus.EXECUTING,
                )

                self._audit(
                    access,
                    rollback_id,
                    "rollback_execution_started",
                )

            try:
                success = self.adapter.rollback(
                    plan.deployment_id,
                    access.tenant_id,
                    plan.target_version_id,
                    plan.strategy,
                )

                if not success:
                    raise RollbackExecutionError(
                        "Deployment adapter reported failure"
                    )

                with self._db:
                    self._set_status(
                        rollback_id,
                        access,
                        RollbackStatus.VERIFYING,
                    )

                verified = (
                    self.adapter.verify_version(
                        plan.deployment_id,
                        access.tenant_id,
                        plan.target_version_id,
                    )
                    if self.config.verify_deployment_after_rollback
                    else False
                )

                if (
                    self.config.verify_deployment_after_rollback
                    and not verified
                ):
                    raise RollbackVerificationError(
                        "Deployment version verification failed"
                    )

                with self._db:
                    self._set_status(
                        rollback_id,
                        access,
                        RollbackStatus.COMPLETED,
                    )

                    self._audit(
                        access,
                        rollback_id,
                        "rollback_completed",
                        {
                            "verified": verified,
                            "target": plan.target_version_id,
                        },
                    )

                return RollbackExecutionResult(
                    rollback_id=rollback_id,
                    success=True,
                    deployment_id=plan.deployment_id,
                    target_version_id=plan.target_version_id,
                    mode=plan.execution_mode,
                    started_at=started_at,
                    completed_at=utc_now(),
                    verified=verified,
                    message=(
                        "Rollback completed in simulation"
                        if plan.execution_mode
                        == RollbackMode.SIMULATION
                        else "Deployment adapter rollback completed"
                    ),
                )

            except Exception as exc:
                with self._db:
                    self._set_status(
                        rollback_id,
                        access,
                        RollbackStatus.FAILED,
                        failure_reason=str(exc),
                    )

                    self._audit(
                        access,
                        rollback_id,
                        "rollback_failed",
                        {"error": str(exc)},
                    )

                raise RollbackExecutionError(
                    f"Rollback failed: {exc}"
                ) from exc

    # --------------------------------------------------------
    # CANCEL
    # --------------------------------------------------------

    def cancel_plan(
        self,
        rollback_id: str,
        access: RollbackAccess,
    ) -> RollbackPlan:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock, self._db:
            plan = self.get_plan(
                rollback_id,
                self._read_access(access),
            )

            if plan.status not in (
                RollbackStatus.PLANNED,
                RollbackStatus.PENDING_APPROVAL,
                RollbackStatus.APPROVED,
            ):
                raise RollbackConflictError(
                    "Rollback cannot be cancelled"
                )

            self._set_status(
                rollback_id,
                access,
                RollbackStatus.CANCELLED,
            )

            self._audit(
                access,
                rollback_id,
                "rollback_cancelled",
            )

        return self.get_plan(
            rollback_id,
            self._read_access(access),
        )

    # --------------------------------------------------------
    # LIST PLANS
    # --------------------------------------------------------

    def list_plans(
        self,
        access: RollbackAccess,
        *,
        deployment_id: str | None = None,
        status: RollbackStatus | None = None,
    ) -> tuple[RollbackPlan, ...]:
        self._authorize(
            access,
            "can_read",
        )

        query = """
            SELECT *
            FROM rollback_plans
            WHERE tenant_id = ?
        """

        parameters: list[Any] = [
            access.tenant_id
        ]

        if deployment_id is not None:
            query += " AND deployment_id = ?"
            parameters.append(deployment_id)

        if status is not None:
            query += " AND status = ?"
            parameters.append(
                RollbackStatus(status).value
            )

        query += """
            ORDER BY requested_at DESC
        """

        with self._lock:
            rows = self._db.execute(
                query,
                parameters,
            ).fetchall()

        return tuple(
            self._row_to_plan(row)
            for row in rows
        )

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        rollback_id: str,
        access: RollbackAccess,
    ) -> tuple[dict[str, Any], ...]:
        self.get_plan(
            rollback_id,
            access,
        )

        with self._lock:
            rows = self._db.execute(
                """
                SELECT *
                FROM rollback_audit
                WHERE tenant_id = ?
                AND rollback_id = ?
                ORDER BY timestamp, audit_id
                """,
                (
                    access.tenant_id,
                    rollback_id,
                ),
            ).fetchall()

        return tuple(
            {
                "audit_id": row["audit_id"],
                "user_id": row["user_id"],
                "action": row["action"],
                "timestamp": row["timestamp"],
                "details": json.loads(
                    row["details_json"]
                ),
            }
            for row in rows
        )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    def export_report_json(
        self,
        rollback_id: str,
        access: RollbackAccess,
    ) -> str:
        plan = self.get_plan(
            rollback_id,
            access,
        )

        return json.dumps(
            json_safe(
                {
                    "rollback_plan": plan,
                    "audit_history": self.audit_history(
                        rollback_id,
                        access,
                    ),
                }
            ),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC FACTORY
# ============================================================

class SyntheticRollbackFactory:

    @staticmethod
    def create_artifact(
        directory: Path,
        version: str,
    ) -> ModelArtifactReference:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            directory
            / f"robot_model_{version}.bin"
        )

        path.write_bytes(
            (
                f"SYNTHETIC_ROBOT_MODEL:{version}"
            ).encode("utf-8")
            + bytes(range(256)) * 4
        )

        return ModelArtifactReference(
            version_id=version,
            model_id="robot-perception",
            artifact_path=path,
            expected_sha256=sha256_file(path),
            framework="pytorch",
        )

    @staticmethod
    def unhealthy_metrics() -> ModelHealthMetrics:
        return ModelHealthMetrics(
            sample_count=200,
            error_rate=0.18,
            p95_latency_ms=420.0,
            accuracy=0.61,
            drift_score=0.35,
            cpu_percent=94.0,
            gpu_percent=97.0,
            memory_percent=88.0,
            health_status=HealthStatus.CRITICAL,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestModelRollback(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

        root = Path(
            self.temporary.name
        )

        self.adapter = SimulatedRollbackAdapter()

        self.adapter.register_deployment(
            DeploymentSnapshot(
                deployment_id="deployment-001",
                tenant_id="robotics_lab",
                model_id="robot-perception",
                current_version_id="v2",
                previous_stable_version_id="v1",
                environment=DeploymentEnvironment.PRODUCTION,
                strategy=DeploymentStrategy.CANARY,
            )
        )

        self.manager = ModelRollbackManager(
            ModelRollbackConfig(
                database_path=(
                    root / "rollback.sqlite3"
                )
            ),
            deployment_adapter=self.adapter,
        )

        self.creator = RollbackAccess(
            tenant_id="robotics_lab",
            user_id="mlops_engineer",
            can_read=True,
            can_plan=True,
            can_execute=True,
            can_manage=True,
        )

        self.approver = RollbackAccess(
            tenant_id="robotics_lab",
            user_id="reviewer",
            can_read=True,
            can_approve=True,
        )

        self.artifact = (
            SyntheticRollbackFactory.create_artifact(
                root,
                "v1",
            )
        )

    def tearDown(self) -> None:
        self.manager.close()
        self.temporary.cleanup()

    def _approved_plan(self) -> RollbackPlan:
        plan = self.manager.create_plan(
            "deployment-001",
            self.creator,
            artifact=self.artifact,
        )

        return self.manager.decide_plan(
            plan.rollback_id,
            ApprovalDecision.APPROVE,
            self.approver,
        )

    def test_health_evaluation(self) -> None:
        recommendation = (
            self.manager.evaluate_health(
                SyntheticRollbackFactory.unhealthy_metrics()
            )
        )

        self.assertTrue(
            recommendation.rollback_recommended
        )

    def test_healthy_model(self) -> None:
        metrics = ModelHealthMetrics(
            sample_count=100,
            error_rate=0.01,
            p95_latency_ms=80.0,
            accuracy=0.95,
            drift_score=0.05,
            health_status=HealthStatus.HEALTHY,
        )

        recommendation = self.manager.evaluate_health(
            metrics
        )

        self.assertFalse(
            recommendation.rollback_recommended
        )

    def test_insufficient_samples(self) -> None:
        metrics = ModelHealthMetrics(
            sample_count=2,
            error_rate=1.0,
            p95_latency_ms=1000.0,
        )

        result = self.manager.evaluate_health(
            metrics
        )

        self.assertFalse(
            result.rollback_recommended
        )

    def test_create_plan(self) -> None:
        plan = self.manager.create_plan(
            "deployment-001",
            self.creator,
            artifact=self.artifact,
        )

        self.assertEqual(
            plan.target_version_id,
            "v1",
        )

    def test_duplicate_active_plan(self) -> None:
        self.manager.create_plan(
            "deployment-001",
            self.creator,
            artifact=self.artifact,
        )

        with self.assertRaises(
            RollbackConflictError
        ):
            self.manager.create_plan(
                "deployment-001",
                self.creator,
                artifact=self.artifact,
            )

    def test_approval(self) -> None:
        plan = self._approved_plan()

        self.assertEqual(
            plan.status,
            RollbackStatus.APPROVED,
        )

    def test_self_approval(self) -> None:
        plan = self.manager.create_plan(
            "deployment-001",
            self.creator,
            artifact=self.artifact,
        )

        self_approver = RollbackAccess(
            tenant_id="robotics_lab",
            user_id="mlops_engineer",
            can_read=True,
            can_approve=True,
        )

        with self.assertRaises(
            RollbackApprovalError
        ):
            self.manager.decide_plan(
                plan.rollback_id,
                ApprovalDecision.APPROVE,
                self_approver,
            )

    def test_execution_requires_approval(self) -> None:
        plan = self.manager.create_plan(
            "deployment-001",
            self.creator,
            artifact=self.artifact,
        )

        with self.assertRaises(
            RollbackApprovalError
        ):
            self.manager.execute(
                plan.rollback_id,
                self.creator,
            )

    def test_execute_rollback(self) -> None:
        plan = self._approved_plan()

        result = self.manager.execute(
            plan.rollback_id,
            self.creator,
        )

        self.assertTrue(
            result.success
        )

        self.assertTrue(
            result.verified
        )

        deployment = self.adapter.get_deployment(
            "deployment-001",
            "robotics_lab",
        )

        self.assertEqual(
            deployment.current_version_id,
            "v1",
        )

    def test_artifact_verification(self) -> None:
        plan = self.manager.create_plan(
            "deployment-001",
            self.creator,
            artifact=self.artifact,
        )

        self.assertTrue(
            self.manager.verify_target_artifact(
                plan.rollback_id,
                self.creator,
            )
        )

    def test_corrupted_artifact(self) -> None:
        plan = self._approved_plan()

        self.artifact.artifact_path.write_bytes(
            b"CORRUPTED"
        )

        with self.assertRaises(
            RollbackIntegrityError
        ):
            self.manager.execute(
                plan.rollback_id,
                self.creator,
            )

    def test_cancel_plan(self) -> None:
        plan = self.manager.create_plan(
            "deployment-001",
            self.creator,
            artifact=self.artifact,
        )

        cancelled = self.manager.cancel_plan(
            plan.rollback_id,
            self.creator,
        )

        self.assertEqual(
            cancelled.status,
            RollbackStatus.CANCELLED,
        )

    def test_tenant_isolation(self) -> None:
        other = RollbackAccess(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
        )

        plan = self.manager.create_plan(
            "deployment-001",
            self.creator,
            artifact=self.artifact,
        )

        with self.assertRaises(
            RollbackNotFoundError
        ):
            self.manager.get_plan(
                plan.rollback_id,
                other,
            )

    def test_export(self) -> None:
        plan = self.manager.create_plan(
            "deployment-001",
            self.creator,
            artifact=self.artifact,
        )

        payload = json.loads(
            self.manager.export_report_json(
                plan.rollback_id,
                self.creator,
            )
        )

        self.assertEqual(
            payload["rollback_plan"]["target_version_id"],
            "v1",
        )

    def test_audit(self) -> None:
        plan = self._approved_plan()

        history = self.manager.audit_history(
            plan.rollback_id,
            self.creator,
        )

        self.assertGreaterEqual(
            len(history),
            2,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        adapter = SimulatedRollbackAdapter()

        adapter.register_deployment(
            DeploymentSnapshot(
                deployment_id="warehouse-edge-01",
                tenant_id="physical_ai_lab",
                model_id="robot-perception",
                current_version_id="v2.0.0",
                previous_stable_version_id="v1.0.0",
                environment=DeploymentEnvironment.PRODUCTION,
                strategy=DeploymentStrategy.CANARY,
                traffic_percentage=100.0,
            )
        )

        config = ModelRollbackConfig(
            database_path=root / "rollback.sqlite3",
        )

        with ModelRollbackManager(
            config,
            deployment_adapter=adapter,
        ) as manager:
            creator = RollbackAccess(
                tenant_id="physical_ai_lab",
                user_id="mlops_engineer",
                can_read=True,
                can_plan=True,
                can_execute=True,
                can_manage=True,
            )

            reviewer = RollbackAccess(
                tenant_id="physical_ai_lab",
                user_id="safety_reviewer",
                can_read=True,
                can_approve=True,
            )

            artifact = (
                SyntheticRollbackFactory.create_artifact(
                    root,
                    "v1.0.0",
                )
            )

            recommendation = manager.evaluate_health(
                SyntheticRollbackFactory.unhealthy_metrics()
            )

            plan = manager.create_plan(
                "warehouse-edge-01",
                creator,
                target_version_id="v1.0.0",
                trigger=RollbackTrigger.HIGH_ERROR_RATE,
                reason=(
                    "Synthetic deployment health degradation"
                ),
                artifact=artifact,
                execution_mode=RollbackMode.SIMULATION,
            )

            manager.decide_plan(
                plan.rollback_id,
                ApprovalDecision.APPROVE,
                reviewer,
                reason="Approved for simulation",
            )

            result = manager.execute(
                plan.rollback_id,
                creator,
            )

            final_plan = manager.get_plan(
                plan.rollback_id,
                creator,
            )

            print("\nPHYSICAL AI MODEL ROLLBACK")
            print("=" * 70)

            print(
                json.dumps(
                    json_safe(
                        {
                            "recommendation": recommendation,
                            "rollback_plan": final_plan,
                            "execution_result": result,
                            "deployment_after": (
                                adapter.get_deployment(
                                    "warehouse-edge-01",
                                    "physical_ai_lab",
                                )
                            ),
                            "synthetic_data": True,
                            "real_infrastructure_modified": False,
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
        TestModelRollback
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
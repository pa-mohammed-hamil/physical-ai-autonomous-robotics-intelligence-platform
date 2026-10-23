"""Deployment Manager module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/mlops/deployment_manager.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
AI Model Deployment Manager

Features:
    - Model deployment registration
    - Model artifact integrity verification
    - Dataset version and model lineage
    - Development, staging and production environments
    - Deployment lifecycle management
    - Blue-green deployment
    - Canary deployment and traffic splitting
    - Rolling deployment planning
    - Model rollback
    - Deployment approval workflow
    - Health checks and readiness evaluation
    - Latency, error rate and throughput monitoring
    - Deployment performance metrics
    - Deployment history and audit logs
    - Deployment configuration validation
    - Resource requirements
    - Multi-tenant access control
    - Thread-safe deployment management
    - JSON persistence and export
    - Unit tests and demonstration

Python 3.11+
Standard library only.

Important:
    This module is a deployment control-plane reference
    implementation. It manages desired deployment state,
    rollout decisions and audit records.

    It does not directly start containers, modify Kubernetes,
    deploy to cloud infrastructure or actuate robots.

    Production integration requires authenticated infrastructure
    adapters, distributed locking, durable transactions,
    monitoring integrations and safety certification where
    applicable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import threading
import unittest
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class DeploymentManagerError(Exception):
    pass


class DeploymentValidationError(DeploymentManagerError):
    pass


class DeploymentNotFoundError(DeploymentManagerError):
    pass


class DeploymentAuthorizationError(DeploymentManagerError):
    pass


class DeploymentConflictError(DeploymentManagerError):
    pass


class DeploymentIntegrityError(DeploymentManagerError):
    pass


class DeploymentApprovalError(DeploymentManagerError):
    pass


class DeploymentHealthError(DeploymentManagerError):
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
        raise DeploymentValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def validate_number(
    value: float,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise DeploymentValidationError(
            f"{name} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DeploymentValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise DeploymentValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and number < minimum:
        raise DeploymentValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and number > maximum:
        raise DeploymentValidationError(
            f"{name} must be <= {maximum}"
        )

    return number


def validate_hash(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise DeploymentValidationError(
            "Invalid SHA-256 artifact hash"
        )

    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


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


def atomic_write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary = tempfile.mkstemp(
        prefix=".deployment-",
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
                json_safe(payload),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )

            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)

    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# ============================================================
# ENUMERATIONS
# ============================================================

class DeploymentEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    EDGE = "edge"


class DeploymentStatus(str, Enum):
    REGISTERED = "registered"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    STOPPED = "stopped"


class DeploymentStrategy(str, Enum):
    RECREATE = "recreate"
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"


class ModelFramework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    TRANSFORMERS = "transformers"
    CUSTOM = "custom"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class DeploymentEventType(str, Enum):
    CREATED = "created"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYMENT_STARTED = "deployment_started"
    ACTIVATED = "activated"
    HEALTH_UPDATED = "health_updated"
    TRAFFIC_UPDATED = "traffic_updated"
    ROLLBACK_STARTED = "rollback_started"
    ROLLBACK_COMPLETED = "rollback_completed"
    STOPPED = "stopped"
    FAILED = "failed"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DeploymentManagerConfig:
    storage_root: Path = Path(
        "./data/deployment_manager"
    )

    maximum_deployments: int = 10000

    default_canary_percentage: float = 10.0

    maximum_error_rate: float = 0.05
    maximum_p95_latency_ms: float = 500.0

    minimum_health_samples: int = 20

    require_production_approval: bool = True
    require_edge_approval: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "storage_root",
            Path(self.storage_root).expanduser().resolve(),
        )

        if self.maximum_deployments <= 0:
            raise DeploymentValidationError(
                "maximum_deployments must be positive"
            )

        validate_number(
            self.default_canary_percentage,
            "default_canary_percentage",
            minimum=0,
            maximum=100,
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
            minimum=0.001,
        )

        if self.minimum_health_samples <= 0:
            raise DeploymentValidationError(
                "minimum_health_samples must be positive"
            )


# ============================================================
# RESOURCE REQUIREMENTS
# ============================================================

@dataclass(frozen=True)
class DeploymentResources:
    cpu_cores: float = 1.0
    memory_mb: int = 2048
    gpu_count: int = 0
    gpu_memory_mb: int = 0
    replicas: int = 1

    def __post_init__(self) -> None:
        validate_number(
            self.cpu_cores,
            "cpu_cores",
            minimum=0.001,
        )

        for name in (
            "memory_mb",
            "gpu_count",
            "gpu_memory_mb",
            "replicas",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise DeploymentValidationError(
                    f"Invalid {name}"
                )

        if self.memory_mb == 0 or self.replicas == 0:
            raise DeploymentValidationError(
                "Memory and replicas must be positive"
            )


# ============================================================
# MODEL ARTIFACT
# ============================================================

@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    model_version: str

    framework: ModelFramework

    artifact_uri: str
    artifact_sha256: str

    dataset_id: str | None = None
    dataset_version_id: str | None = None

    metrics: dict[str, float] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.model_id,
            "model_id",
        )

        validate_identifier(
            self.model_version,
            "model_version",
        )

        validate_identifier(
            self.artifact_uri,
            "artifact_uri",
        )

        validate_hash(
            self.artifact_sha256
        )

        for name, value in self.metrics.items():
            validate_number(
                value,
                f"metric {name}",
            )


# ============================================================
# DEPLOYMENT SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class DeploymentSpecification:
    deployment_name: str

    environment: DeploymentEnvironment
    strategy: DeploymentStrategy

    model: ModelArtifact

    resources: DeploymentResources = field(
        default_factory=DeploymentResources
    )

    endpoint_name: str = ""

    initial_traffic_percentage: float = 100.0

    environment_variables: dict[str, str] = field(
        default_factory=dict
    )

    labels: dict[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.deployment_name,
            "deployment_name",
        )

        validate_number(
            self.initial_traffic_percentage,
            "initial_traffic_percentage",
            minimum=0,
            maximum=100,
        )

        if self.endpoint_name:
            validate_identifier(
                self.endpoint_name,
                "endpoint_name",
            )

        for key in self.environment_variables:
            validate_identifier(
                key,
                "environment variable name",
            )


# ============================================================
# HEALTH METRICS
# ============================================================

@dataclass(frozen=True)
class DeploymentHealthMetrics:
    request_count: int
    error_count: int

    average_latency_ms: float
    p95_latency_ms: float

    throughput_rps: float

    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float = 0.0

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        if (
            self.request_count < 0
            or self.error_count < 0
            or self.error_count > self.request_count
        ):
            raise DeploymentValidationError(
                "Invalid request/error counts"
            )

        for name in (
            "average_latency_ms",
            "p95_latency_ms",
            "throughput_rps",
        ):
            validate_number(
                getattr(self, name),
                name,
                minimum=0,
            )

        for name in (
            "cpu_utilization",
            "memory_utilization",
            "gpu_utilization",
        ):
            validate_number(
                getattr(self, name),
                name,
                minimum=0,
                maximum=1,
            )

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0

        return self.error_count / self.request_count


@dataclass(frozen=True)
class DeploymentHealthReport:
    deployment_id: str
    status: HealthStatus

    error_rate: float
    p95_latency_ms: float

    reasons: tuple[str, ...]

    evaluated_at: datetime


# ============================================================
# APPROVALS
# ============================================================

@dataclass
class DeploymentApproval:
    approval_id: str
    deployment_id: str

    requested_by: str
    requested_at: datetime

    status: ApprovalStatus = (
        ApprovalStatus.PENDING
    )

    reviewed_by: str | None = None
    reviewed_at: datetime | None = None

    reason: str = ""


# ============================================================
# AUDIT EVENTS
# ============================================================

@dataclass(frozen=True)
class DeploymentAuditEvent:
    event_id: str
    deployment_id: str
    tenant_id: str

    event_type: DeploymentEventType

    actor_id: str
    timestamp: datetime

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# DEPLOYMENT RECORD
# ============================================================

@dataclass
class DeploymentRecord:
    deployment_id: str
    tenant_id: str

    specification: DeploymentSpecification

    status: DeploymentStatus

    created_by: str
    created_at: datetime

    updated_at: datetime

    traffic_percentage: float = 0.0

    previous_deployment_id: str | None = None

    health_status: HealthStatus = (
        HealthStatus.UNKNOWN
    )

    latest_metrics: DeploymentHealthMetrics | None = None

    approval_id: str | None = None

    revision: int = 1

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class DeploymentAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_deploy: bool = False
    can_approve: bool = False
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
# ARTIFACT VERIFIER
# ============================================================

class ModelArtifactVerifier:

    @staticmethod
    def verify_local_artifact(
        artifact: ModelArtifact,
    ) -> bool:
        uri = artifact.artifact_uri

        if uri.startswith("file://"):
            path = Path(uri[7:])

        elif "://" not in uri:
            path = Path(uri)

        else:
            raise DeploymentValidationError(
                "Remote artifact verification requires "
                "an authenticated artifact-store adapter"
            )

        if not path.is_file():
            raise DeploymentIntegrityError(
                "Model artifact file not found"
            )

        actual_hash = sha256_file(
            path
        )

        if actual_hash != artifact.artifact_sha256:
            raise DeploymentIntegrityError(
                "Model artifact SHA-256 mismatch"
            )

        return True


# ============================================================
# HEALTH EVALUATOR
# ============================================================

class DeploymentHealthEvaluator:

    def __init__(
        self,
        config: DeploymentManagerConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        deployment_id: str,
        metrics: DeploymentHealthMetrics,
    ) -> DeploymentHealthReport:
        reasons = []

        if (
            metrics.request_count
            < self.config.minimum_health_samples
        ):
            return DeploymentHealthReport(
                deployment_id=deployment_id,
                status=HealthStatus.UNKNOWN,
                error_rate=metrics.error_rate,
                p95_latency_ms=metrics.p95_latency_ms,
                reasons=(
                    "Insufficient health-check samples",
                ),
                evaluated_at=utc_now(),
            )

        if (
            metrics.error_rate
            > self.config.maximum_error_rate
        ):
            reasons.append(
                "Error rate exceeds configured threshold"
            )

        if (
            metrics.p95_latency_ms
            > self.config.maximum_p95_latency_ms
        ):
            reasons.append(
                "P95 latency exceeds configured threshold"
            )

        if metrics.memory_utilization >= 0.98:
            reasons.append(
                "Critical memory utilization"
            )

        if metrics.cpu_utilization >= 0.98:
            reasons.append(
                "Critical CPU utilization"
            )

        if reasons:
            status = HealthStatus.UNHEALTHY

        elif (
            metrics.memory_utilization >= 0.90
            or metrics.cpu_utilization >= 0.90
        ):
            status = HealthStatus.DEGRADED

            reasons.append(
                "Resource utilization is elevated"
            )

        else:
            status = HealthStatus.HEALTHY

        return DeploymentHealthReport(
            deployment_id=deployment_id,
            status=status,
            error_rate=metrics.error_rate,
            p95_latency_ms=metrics.p95_latency_ms,
            reasons=tuple(reasons),
            evaluated_at=utc_now(),
        )


# ============================================================
# ROLLOUT PLANNER
# ============================================================

@dataclass(frozen=True)
class RolloutStep:
    step_number: int
    traffic_percentage: float
    description: str


@dataclass(frozen=True)
class RolloutPlan:
    deployment_id: str
    strategy: DeploymentStrategy
    steps: tuple[RolloutStep, ...]


class DeploymentRolloutPlanner:

    @staticmethod
    def create_plan(
        deployment_id: str,
        strategy: DeploymentStrategy,
        *,
        canary_percentage: float = 10.0,
    ) -> RolloutPlan:
        validate_number(
            canary_percentage,
            "canary_percentage",
            minimum=0.001,
            maximum=100,
        )

        if strategy == DeploymentStrategy.CANARY:
            percentages = [
                canary_percentage,
                25.0,
                50.0,
                100.0,
            ]

        elif strategy == DeploymentStrategy.ROLLING:
            percentages = [
                25.0,
                50.0,
                75.0,
                100.0,
            ]

        elif strategy == DeploymentStrategy.BLUE_GREEN:
            percentages = [
                0.0,
                100.0,
            ]

        else:
            percentages = [
                100.0,
            ]

        percentages = sorted(
            set(percentages)
        )

        steps = tuple(
            RolloutStep(
                step_number=index + 1,
                traffic_percentage=percentage,
                description=(
                    f"Set desired traffic allocation "
                    f"to {percentage:.1f}%"
                ),
            )
            for index, percentage in enumerate(
                percentages
            )
        )

        return RolloutPlan(
            deployment_id=deployment_id,
            strategy=strategy,
            steps=steps,
        )


# ============================================================
# DEPLOYMENT MANAGER
# ============================================================

class DeploymentManager:

    def __init__(
        self,
        config: DeploymentManagerConfig | None = None,
    ) -> None:
        self.config = (
            config or DeploymentManagerConfig()
        )

        self.artifact_verifier = (
            ModelArtifactVerifier()
        )

        self.health_evaluator = (
            DeploymentHealthEvaluator(
                self.config
            )
        )

        self.rollout_planner = (
            DeploymentRolloutPlanner()
        )

        self._deployments: dict[
            tuple[str, str],
            DeploymentRecord,
        ] = {}

        self._approvals: dict[
            str,
            DeploymentApproval,
        ] = {}

        self._events: list[
            DeploymentAuditEvent
        ] = []

        self._lock = threading.RLock()

        self._load()

    # --------------------------------------------------------
    # AUTHORIZATION
    # --------------------------------------------------------

    @staticmethod
    def _authorize(
        access: DeploymentAccessContext,
        permission: str,
    ) -> None:
        if not getattr(
            access,
            permission,
            False,
        ):
            raise DeploymentAuthorizationError(
                f"Missing permission: {permission}"
            )

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    def _storage_path(self) -> Path:
        return (
            self.config.storage_root
            / "deployments.json"
        )

    def _persist(self) -> None:
        atomic_write_json(
            self._storage_path(),
            {
                "schema_version": 1,
                "deployments": [
                    record.to_dict()
                    for record in self._deployments.values()
                ],
                "approvals": [
                    json_safe(approval)
                    for approval in self._approvals.values()
                ],
                "events": [
                    json_safe(event)
                    for event in self._events
                ],
            },
        )

    @staticmethod
    def _parse_model(
        data: Mapping[str, Any],
    ) -> ModelArtifact:
        return ModelArtifact(
            model_id=data["model_id"],
            model_version=data["model_version"],
            framework=ModelFramework(
                data["framework"]
            ),
            artifact_uri=data["artifact_uri"],
            artifact_sha256=data["artifact_sha256"],
            dataset_id=data.get("dataset_id"),
            dataset_version_id=data.get(
                "dataset_version_id"
            ),
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def _parse_specification(
        cls,
        data: Mapping[str, Any],
    ) -> DeploymentSpecification:
        return DeploymentSpecification(
            deployment_name=data["deployment_name"],
            environment=DeploymentEnvironment(
                data["environment"]
            ),
            strategy=DeploymentStrategy(
                data["strategy"]
            ),
            model=cls._parse_model(
                data["model"]
            ),
            resources=DeploymentResources(
                **data["resources"]
            ),
            endpoint_name=data.get(
                "endpoint_name",
                "",
            ),
            initial_traffic_percentage=data.get(
                "initial_traffic_percentage",
                100.0,
            ),
            environment_variables=data.get(
                "environment_variables",
                {},
            ),
            labels=data.get(
                "labels",
                {},
            ),
        )

    @staticmethod
    def _parse_metrics(
        data: Mapping[str, Any] | None,
    ) -> DeploymentHealthMetrics | None:
        if data is None:
            return None

        values = dict(data)

        values["timestamp"] = datetime.fromisoformat(
            values["timestamp"]
        )

        return DeploymentHealthMetrics(
            **values
        )

    def _load(self) -> None:
        path = self._storage_path()

        if not path.exists():
            return

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            if payload.get("schema_version") != 1:
                raise DeploymentIntegrityError(
                    "Unsupported deployment registry schema"
                )

            for item in payload.get(
                "deployments",
                [],
            ):
                record = DeploymentRecord(
                    deployment_id=item["deployment_id"],
                    tenant_id=item["tenant_id"],
                    specification=self._parse_specification(
                        item["specification"]
                    ),
                    status=DeploymentStatus(
                        item["status"]
                    ),
                    created_by=item["created_by"],
                    created_at=datetime.fromisoformat(
                        item["created_at"]
                    ),
                    updated_at=datetime.fromisoformat(
                        item["updated_at"]
                    ),
                    traffic_percentage=item.get(
                        "traffic_percentage",
                        0.0,
                    ),
                    previous_deployment_id=item.get(
                        "previous_deployment_id"
                    ),
                    health_status=HealthStatus(
                        item.get(
                            "health_status",
                            "unknown",
                        )
                    ),
                    latest_metrics=self._parse_metrics(
                        item.get("latest_metrics")
                    ),
                    approval_id=item.get(
                        "approval_id"
                    ),
                    revision=item.get(
                        "revision",
                        1,
                    ),
                )

                self._deployments[
                    (
                        record.tenant_id,
                        record.deployment_id,
                    )
                ] = record

            for item in payload.get(
                "approvals",
                [],
            ):
                approval = DeploymentApproval(
                    approval_id=item["approval_id"],
                    deployment_id=item["deployment_id"],
                    requested_by=item["requested_by"],
                    requested_at=datetime.fromisoformat(
                        item["requested_at"]
                    ),
                    status=ApprovalStatus(
                        item["status"]
                    ),
                    reviewed_by=item.get(
                        "reviewed_by"
                    ),
                    reviewed_at=(
                        datetime.fromisoformat(
                            item["reviewed_at"]
                        )
                        if item.get("reviewed_at")
                        else None
                    ),
                    reason=item.get(
                        "reason",
                        "",
                    ),
                )

                self._approvals[
                    approval.approval_id
                ] = approval

            for item in payload.get(
                "events",
                [],
            ):
                self._events.append(
                    DeploymentAuditEvent(
                        event_id=item["event_id"],
                        deployment_id=item["deployment_id"],
                        tenant_id=item["tenant_id"],
                        event_type=DeploymentEventType(
                            item["event_type"]
                        ),
                        actor_id=item["actor_id"],
                        timestamp=datetime.fromisoformat(
                            item["timestamp"]
                        ),
                        details=item.get(
                            "details",
                            {},
                        ),
                    )
                )

        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise DeploymentIntegrityError(
                "Invalid deployment registry"
            ) from exc

    # --------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------

    def _get(
        self,
        deployment_id: str,
        tenant_id: str,
    ) -> DeploymentRecord:
        record = self._deployments.get(
            (
                tenant_id,
                deployment_id,
            )
        )

        if record is None:
            raise DeploymentNotFoundError(
                "Deployment not found"
            )

        return record

    def _event(
        self,
        record: DeploymentRecord,
        event_type: DeploymentEventType,
        actor_id: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._events.append(
            DeploymentAuditEvent(
                event_id=generate_id(),
                deployment_id=record.deployment_id,
                tenant_id=record.tenant_id,
                event_type=event_type,
                actor_id=actor_id,
                timestamp=utc_now(),
                details=dict(details or {}),
            )
        )

    @staticmethod
    def _touch(
        record: DeploymentRecord,
    ) -> None:
        record.revision += 1
        record.updated_at = utc_now()

    def _requires_approval(
        self,
        environment: DeploymentEnvironment,
    ) -> bool:
        if environment == DeploymentEnvironment.PRODUCTION:
            return self.config.require_production_approval

        if environment == DeploymentEnvironment.EDGE:
            return self.config.require_edge_approval

        return False

    # --------------------------------------------------------
    # REGISTER DEPLOYMENT
    # --------------------------------------------------------

    def register_deployment(
        self,
        specification: DeploymentSpecification,
        access: DeploymentAccessContext,
        *,
        verify_artifact: bool = True,
        previous_deployment_id: str | None = None,
    ) -> DeploymentRecord:
        self._authorize(
            access,
            "can_deploy",
        )

        with self._lock:
            if (
                len(self._deployments)
                >= self.config.maximum_deployments
            ):
                raise DeploymentConflictError(
                    "Deployment capacity reached"
                )

            for existing in self._deployments.values():
                if (
                    existing.tenant_id == access.tenant_id
                    and existing.specification.deployment_name
                    == specification.deployment_name
                    and existing.status not in (
                        DeploymentStatus.STOPPED,
                        DeploymentStatus.ROLLED_BACK,
                        DeploymentStatus.FAILED,
                    )
                ):
                    raise DeploymentConflictError(
                        "Active deployment name already exists"
                    )

            if previous_deployment_id is not None:
                previous = self._get(
                    previous_deployment_id,
                    access.tenant_id,
                )

                if (
                    previous.specification.endpoint_name
                    != specification.endpoint_name
                ):
                    raise DeploymentValidationError(
                        "Previous deployment endpoint mismatch"
                    )

            if verify_artifact:
                self.artifact_verifier.verify_local_artifact(
                    specification.model
                )

            requires_approval = (
                self._requires_approval(
                    specification.environment
                )
            )

            record = DeploymentRecord(
                deployment_id=generate_id(),
                tenant_id=access.tenant_id,
                specification=copy.deepcopy(
                    specification
                ),
                status=(
                    DeploymentStatus.PENDING_APPROVAL
                    if requires_approval
                    else DeploymentStatus.APPROVED
                ),
                created_by=access.user_id,
                created_at=utc_now(),
                updated_at=utc_now(),
                previous_deployment_id=(
                    previous_deployment_id
                ),
            )

            key = (
                access.tenant_id,
                record.deployment_id,
            )

            self._deployments[key] = record

            self._event(
                record,
                DeploymentEventType.CREATED,
                access.user_id,
            )

            if requires_approval:
                approval = DeploymentApproval(
                    approval_id=generate_id(),
                    deployment_id=record.deployment_id,
                    requested_by=access.user_id,
                    requested_at=utc_now(),
                )

                record.approval_id = (
                    approval.approval_id
                )

                self._approvals[
                    approval.approval_id
                ] = approval

                self._event(
                    record,
                    DeploymentEventType.APPROVAL_REQUESTED,
                    access.user_id,
                )

            self._persist()

            return copy.deepcopy(record)

    # --------------------------------------------------------
    # GET / LIST
    # --------------------------------------------------------

    def get_deployment(
        self,
        deployment_id: str,
        access: DeploymentAccessContext,
    ) -> DeploymentRecord:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            return copy.deepcopy(
                self._get(
                    deployment_id,
                    access.tenant_id,
                )
            )

    def list_deployments(
        self,
        access: DeploymentAccessContext,
        *,
        environment: DeploymentEnvironment | None = None,
        status: DeploymentStatus | None = None,
    ) -> tuple[DeploymentRecord, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            results = []

            for record in self._deployments.values():
                if record.tenant_id != access.tenant_id:
                    continue

                if (
                    environment is not None
                    and record.specification.environment
                    != environment
                ):
                    continue

                if (
                    status is not None
                    and record.status != status
                ):
                    continue

                results.append(
                    copy.deepcopy(record)
                )

            return tuple(results)

    # --------------------------------------------------------
    # APPROVAL WORKFLOW
    # --------------------------------------------------------

    def review_approval(
        self,
        deployment_id: str,
        access: DeploymentAccessContext,
        *,
        approve: bool,
        reason: str = "",
    ) -> DeploymentRecord:
        self._authorize(
            access,
            "can_approve",
        )

        with self._lock:
            record = self._get(
                deployment_id,
                access.tenant_id,
            )

            if record.status != DeploymentStatus.PENDING_APPROVAL:
                raise DeploymentApprovalError(
                    "Deployment is not awaiting approval"
                )

            if record.approval_id is None:
                raise DeploymentApprovalError(
                    "Approval record is missing"
                )

            approval = self._approvals[
                record.approval_id
            ]

            if approval.requested_by == access.user_id:
                raise DeploymentApprovalError(
                    "Requester cannot approve their own deployment"
                )

            approval.status = (
                ApprovalStatus.APPROVED
                if approve
                else ApprovalStatus.REJECTED
            )

            approval.reviewed_by = access.user_id
            approval.reviewed_at = utc_now()
            approval.reason = reason

            record.status = (
                DeploymentStatus.APPROVED
                if approve
                else DeploymentStatus.FAILED
            )

            self._touch(record)

            self._event(
                record,
                (
                    DeploymentEventType.APPROVED
                    if approve
                    else DeploymentEventType.REJECTED
                ),
                access.user_id,
                {
                    "reason": reason,
                },
            )

            self._persist()

            return copy.deepcopy(record)

    # --------------------------------------------------------
    # START DEPLOYMENT
    # --------------------------------------------------------

    def start_deployment(
        self,
        deployment_id: str,
        access: DeploymentAccessContext,
    ) -> DeploymentRecord:
        self._authorize(
            access,
            "can_deploy",
        )

        with self._lock:
            record = self._get(
                deployment_id,
                access.tenant_id,
            )

            if record.status != DeploymentStatus.APPROVED:
                raise DeploymentConflictError(
                    "Deployment must be approved before starting"
                )

            record.status = DeploymentStatus.DEPLOYING

            self._touch(record)

            self._event(
                record,
                DeploymentEventType.DEPLOYMENT_STARTED,
                access.user_id,
            )

            self._persist()

            return copy.deepcopy(record)

    # --------------------------------------------------------
    # ACTIVATE DEPLOYMENT
    # --------------------------------------------------------

    def activate_deployment(
        self,
        deployment_id: str,
        access: DeploymentAccessContext,
    ) -> DeploymentRecord:
        self._authorize(
            access,
            "can_deploy",
        )

        with self._lock:
            record = self._get(
                deployment_id,
                access.tenant_id,
            )

            if record.status != DeploymentStatus.DEPLOYING:
                raise DeploymentConflictError(
                    "Deployment must be in deploying state"
                )

            record.status = DeploymentStatus.ACTIVE

            if record.specification.strategy == DeploymentStrategy.CANARY:
                traffic = (
                    self.config.default_canary_percentage
                )

            elif record.specification.strategy == DeploymentStrategy.BLUE_GREEN:
                traffic = 0.0

            else:
                traffic = (
                    record.specification.initial_traffic_percentage
                )

            record.traffic_percentage = traffic

            self._touch(record)

            self._event(
                record,
                DeploymentEventType.ACTIVATED,
                access.user_id,
                {
                    "traffic_percentage": traffic,
                    "desired_state_only": True,
                },
            )

            self._persist()

            return copy.deepcopy(record)

    # --------------------------------------------------------
    # HEALTH REPORTING
    # --------------------------------------------------------

    def update_health(
        self,
        deployment_id: str,
        metrics: DeploymentHealthMetrics,
        access: DeploymentAccessContext,
    ) -> DeploymentHealthReport:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            record = self._get(
                deployment_id,
                access.tenant_id,
            )

            if record.status not in (
                DeploymentStatus.ACTIVE,
                DeploymentStatus.DEGRADED,
            ):
                raise DeploymentConflictError(
                    "Deployment is not active"
                )

            report = self.health_evaluator.evaluate(
                deployment_id,
                metrics,
            )

            record.latest_metrics = metrics
            record.health_status = report.status

            if report.status == HealthStatus.UNHEALTHY:
                record.status = DeploymentStatus.DEGRADED

            elif report.status == HealthStatus.HEALTHY:
                record.status = DeploymentStatus.ACTIVE

            self._touch(record)

            self._event(
                record,
                DeploymentEventType.HEALTH_UPDATED,
                access.user_id,
                {
                    "health_status": report.status.value,
                    "error_rate": report.error_rate,
                    "p95_latency_ms": report.p95_latency_ms,
                    "reasons": list(report.reasons),
                },
            )

            self._persist()

            return report

    # --------------------------------------------------------
    # TRAFFIC MANAGEMENT
    # --------------------------------------------------------

    def update_traffic(
        self,
        deployment_id: str,
        traffic_percentage: float,
        access: DeploymentAccessContext,
    ) -> DeploymentRecord:
        self._authorize(
            access,
            "can_deploy",
        )

        traffic_percentage = validate_number(
            traffic_percentage,
            "traffic_percentage",
            minimum=0,
            maximum=100,
        )

        with self._lock:
            record = self._get(
                deployment_id,
                access.tenant_id,
            )

            if record.status not in (
                DeploymentStatus.ACTIVE,
                DeploymentStatus.DEGRADED,
            ):
                raise DeploymentConflictError(
                    "Deployment is not active"
                )

            if (
                traffic_percentage
                > record.traffic_percentage
                and record.health_status
                != HealthStatus.HEALTHY
            ):
                raise DeploymentHealthError(
                    "Traffic cannot increase without healthy metrics"
                )

            endpoint = (
                record.specification.endpoint_name
            )

            if endpoint:
                other_traffic = sum(
                    deployment.traffic_percentage
                    for deployment in self._deployments.values()
                    if (
                        deployment.tenant_id == access.tenant_id
                        and deployment.deployment_id
                        != deployment_id
                        and deployment.specification.environment
                        == record.specification.environment
                        and deployment.specification.endpoint_name
                        == endpoint
                        and deployment.status in (
                            DeploymentStatus.ACTIVE,
                            DeploymentStatus.DEGRADED,
                        )
                    )
                )

                if other_traffic + traffic_percentage > 100.000001:
                    raise DeploymentConflictError(
                        "Endpoint traffic allocation exceeds 100%"
                    )

            record.traffic_percentage = (
                traffic_percentage
            )

            self._touch(record)

            self._event(
                record,
                DeploymentEventType.TRAFFIC_UPDATED,
                access.user_id,
                {
                    "traffic_percentage": traffic_percentage,
                    "desired_state_only": True,
                },
            )

            self._persist()

            return copy.deepcopy(record)

    # --------------------------------------------------------
    # ROLLOUT PLAN
    # --------------------------------------------------------

    def rollout_plan(
        self,
        deployment_id: str,
        access: DeploymentAccessContext,
    ) -> RolloutPlan:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            record = self._get(
                deployment_id,
                access.tenant_id,
            )

            return self.rollout_planner.create_plan(
                deployment_id,
                record.specification.strategy,
                canary_percentage=(
                    self.config.default_canary_percentage
                ),
            )

    # --------------------------------------------------------
    # ROLLBACK
    # --------------------------------------------------------

    def rollback_deployment(
        self,
        deployment_id: str,
        access: DeploymentAccessContext,
    ) -> DeploymentRecord:
        self._authorize(
            access,
            "can_deploy",
        )

        with self._lock:
            record = self._get(
                deployment_id,
                access.tenant_id,
            )

            if record.status not in (
                DeploymentStatus.ACTIVE,
                DeploymentStatus.DEGRADED,
            ):
                raise DeploymentConflictError(
                    "Only active deployments can be rolled back"
                )

            if record.previous_deployment_id is None:
                raise DeploymentConflictError(
                    "No previous deployment available"
                )

            previous = self._get(
                record.previous_deployment_id,
                access.tenant_id,
            )

            if previous.status not in (
                DeploymentStatus.ACTIVE,
                DeploymentStatus.DEGRADED,
            ):
                raise DeploymentConflictError(
                    "Previous deployment is not available"
                )

            record.status = DeploymentStatus.ROLLING_BACK

            self._event(
                record,
                DeploymentEventType.ROLLBACK_STARTED,
                access.user_id,
                {
                    "previous_deployment_id": (
                        previous.deployment_id
                    ),
                },
            )

            # Desired-state rollback only. Infrastructure
            # must independently confirm traffic changes.
            record.traffic_percentage = 0.0
            record.status = DeploymentStatus.ROLLED_BACK

            self._touch(record)

            self._event(
                record,
                DeploymentEventType.ROLLBACK_COMPLETED,
                access.user_id,
                {
                    "previous_deployment_id": (
                        previous.deployment_id
                    ),
                    "desired_state_only": True,
                },
            )

            self._persist()

            return copy.deepcopy(record)

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    def stop_deployment(
        self,
        deployment_id: str,
        access: DeploymentAccessContext,
    ) -> DeploymentRecord:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            record = self._get(
                deployment_id,
                access.tenant_id,
            )

            if record.status == DeploymentStatus.STOPPED:
                return copy.deepcopy(record)

            record.status = DeploymentStatus.STOPPED
            record.traffic_percentage = 0.0

            self._touch(record)

            self._event(
                record,
                DeploymentEventType.STOPPED,
                access.user_id,
                {
                    "desired_state_only": True,
                },
            )

            self._persist()

            return copy.deepcopy(record)

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def deployment_history(
        self,
        deployment_id: str,
        access: DeploymentAccessContext,
    ) -> tuple[DeploymentAuditEvent, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            self._get(
                deployment_id,
                access.tenant_id,
            )

            return tuple(
                copy.deepcopy(event)
                for event in self._events
                if (
                    event.tenant_id == access.tenant_id
                    and event.deployment_id == deployment_id
                )
            )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    def export_deployment_json(
        self,
        deployment_id: str,
        access: DeploymentAccessContext,
    ) -> str:
        record = self.get_deployment(
            deployment_id,
            access,
        )

        return json.dumps(
            record.to_dict(),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC DEPLOYMENT FACTORY
# ============================================================

class SyntheticDeploymentFactory:

    @staticmethod
    def create_model_artifact(
        directory: Path,
        *,
        version: str = "1.0.0",
    ) -> ModelArtifact:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        artifact_path = (
            directory / f"robot_model_{version}.bin"
        )

        artifact_path.write_bytes(
            (
                "synthetic-robotics-model-"
                + version
            ).encode("utf-8")
        )

        return ModelArtifact(
            model_id="robot-perception-model",
            model_version=version,
            framework=ModelFramework.ONNX,
            artifact_uri=str(artifact_path),
            artifact_sha256=sha256_file(
                artifact_path
            ),
            dataset_id="robotics-dataset",
            dataset_version_id="dataset-v1",
            metrics={
                "accuracy": 0.95,
                "f1_score": 0.93,
            },
            metadata={
                "task": "robot_perception",
                "synthetic": True,
            },
        )

    @staticmethod
    def healthy_metrics() -> DeploymentHealthMetrics:
        return DeploymentHealthMetrics(
            request_count=1000,
            error_count=5,
            average_latency_ms=45.0,
            p95_latency_ms=90.0,
            throughput_rps=120.0,
            cpu_utilization=0.45,
            memory_utilization=0.55,
            gpu_utilization=0.35,
        )

    @staticmethod
    def unhealthy_metrics() -> DeploymentHealthMetrics:
        return DeploymentHealthMetrics(
            request_count=1000,
            error_count=120,
            average_latency_ms=450.0,
            p95_latency_ms=950.0,
            throughput_rps=25.0,
            cpu_utilization=0.99,
            memory_utilization=0.98,
            gpu_utilization=0.95,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestDeploymentManager(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

        self.root = Path(
            self.temporary.name
        )

        self.config = DeploymentManagerConfig(
            storage_root=self.root / "registry",
        )

        self.manager = DeploymentManager(
            self.config
        )

        self.deployer = DeploymentAccessContext(
            tenant_id="tenant_alpha",
            user_id="mlops_engineer",
            can_read=True,
            can_deploy=True,
            can_manage=True,
        )

        self.approver = DeploymentAccessContext(
            tenant_id="tenant_alpha",
            user_id="platform_admin",
            can_read=True,
            can_approve=True,
            can_manage=True,
        )

        self.artifact = (
            SyntheticDeploymentFactory.create_model_artifact(
                self.root / "models"
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _specification(
        self,
        *,
        environment: DeploymentEnvironment = (
            DeploymentEnvironment.STAGING
        ),
        strategy: DeploymentStrategy = (
            DeploymentStrategy.CANARY
        ),
        name: str = "robot-perception-v1",
    ) -> DeploymentSpecification:
        return DeploymentSpecification(
            deployment_name=name,
            environment=environment,
            strategy=strategy,
            model=self.artifact,
            resources=DeploymentResources(
                cpu_cores=2.0,
                memory_mb=4096,
                gpu_count=1,
                gpu_memory_mb=8192,
                replicas=2,
            ),
            endpoint_name="robot-perception",
        )

    def _active_deployment(self) -> DeploymentRecord:
        record = self.manager.register_deployment(
            self._specification(),
            self.deployer,
        )

        self.manager.start_deployment(
            record.deployment_id,
            self.deployer,
        )

        return self.manager.activate_deployment(
            record.deployment_id,
            self.deployer,
        )

    def test_artifact_verification(self) -> None:
        self.assertTrue(
            ModelArtifactVerifier.verify_local_artifact(
                self.artifact
            )
        )

    def test_register_deployment(self) -> None:
        record = self.manager.register_deployment(
            self._specification(),
            self.deployer,
        )

        self.assertEqual(
            record.status,
            DeploymentStatus.APPROVED,
        )

    def test_production_approval_required(self) -> None:
        record = self.manager.register_deployment(
            self._specification(
                environment=DeploymentEnvironment.PRODUCTION
            ),
            self.deployer,
        )

        self.assertEqual(
            record.status,
            DeploymentStatus.PENDING_APPROVAL,
        )

    def test_approval_workflow(self) -> None:
        record = self.manager.register_deployment(
            self._specification(
                environment=DeploymentEnvironment.PRODUCTION
            ),
            self.deployer,
        )

        approved = self.manager.review_approval(
            record.deployment_id,
            self.approver,
            approve=True,
        )

        self.assertEqual(
            approved.status,
            DeploymentStatus.APPROVED,
        )

    def test_self_approval_denied(self) -> None:
        record = self.manager.register_deployment(
            self._specification(
                environment=DeploymentEnvironment.PRODUCTION
            ),
            self.deployer,
        )

        self_approver = replace(
            self.deployer,
            can_approve=True,
        )

        with self.assertRaises(
            DeploymentApprovalError
        ):
            self.manager.review_approval(
                record.deployment_id,
                self_approver,
                approve=True,
            )

    def test_start_deployment(self) -> None:
        record = self.manager.register_deployment(
            self._specification(),
            self.deployer,
        )

        started = self.manager.start_deployment(
            record.deployment_id,
            self.deployer,
        )

        self.assertEqual(
            started.status,
            DeploymentStatus.DEPLOYING,
        )

    def test_activate_canary(self) -> None:
        record = self._active_deployment()

        self.assertEqual(
            record.status,
            DeploymentStatus.ACTIVE,
        )

        self.assertEqual(
            record.traffic_percentage,
            self.config.default_canary_percentage,
        )

    def test_healthy_metrics(self) -> None:
        record = self._active_deployment()

        report = self.manager.update_health(
            record.deployment_id,
            SyntheticDeploymentFactory.healthy_metrics(),
            self.deployer,
        )

        self.assertEqual(
            report.status,
            HealthStatus.HEALTHY,
        )

    def test_unhealthy_metrics(self) -> None:
        record = self._active_deployment()

        report = self.manager.update_health(
            record.deployment_id,
            SyntheticDeploymentFactory.unhealthy_metrics(),
            self.deployer,
        )

        self.assertEqual(
            report.status,
            HealthStatus.UNHEALTHY,
        )

    def test_traffic_increase_requires_health(self) -> None:
        record = self._active_deployment()

        with self.assertRaises(
            DeploymentHealthError
        ):
            self.manager.update_traffic(
                record.deployment_id,
                50.0,
                self.deployer,
            )

    def test_traffic_increase_after_health(self) -> None:
        record = self._active_deployment()

        self.manager.update_health(
            record.deployment_id,
            SyntheticDeploymentFactory.healthy_metrics(),
            self.deployer,
        )

        updated = self.manager.update_traffic(
            record.deployment_id,
            50.0,
            self.deployer,
        )

        self.assertEqual(
            updated.traffic_percentage,
            50.0,
        )

    def test_rollout_plan(self) -> None:
        record = self._active_deployment()

        plan = self.manager.rollout_plan(
            record.deployment_id,
            self.deployer,
        )

        self.assertEqual(
            plan.steps[-1].traffic_percentage,
            100.0,
        )

    def test_stop_deployment(self) -> None:
        record = self._active_deployment()

        stopped = self.manager.stop_deployment(
            record.deployment_id,
            self.deployer,
        )

        self.assertEqual(
            stopped.status,
            DeploymentStatus.STOPPED,
        )

    def test_audit_history(self) -> None:
        record = self._active_deployment()

        history = self.manager.deployment_history(
            record.deployment_id,
            self.deployer,
        )

        self.assertGreaterEqual(
            len(history),
            3,
        )

    def test_persistence(self) -> None:
        record = self._active_deployment()

        reloaded = DeploymentManager(
            self.config
        )

        restored = reloaded.get_deployment(
            record.deployment_id,
            self.deployer,
        )

        self.assertEqual(
            restored.status,
            DeploymentStatus.ACTIVE,
        )

    def test_cross_tenant_isolation(self) -> None:
        record = self._active_deployment()

        other = DeploymentAccessContext(
            tenant_id="tenant_beta",
            user_id="other",
            can_read=True,
        )

        with self.assertRaises(
            DeploymentNotFoundError
        ):
            self.manager.get_deployment(
                record.deployment_id,
                other,
            )

    def test_artifact_corruption(self) -> None:
        path = Path(
            self.artifact.artifact_uri
        )

        path.write_bytes(
            b"corrupted-artifact"
        )

        with self.assertRaises(
            DeploymentIntegrityError
        ):
            ModelArtifactVerifier.verify_local_artifact(
                self.artifact
            )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        manager = DeploymentManager(
            DeploymentManagerConfig(
                storage_root=root / "registry",
            )
        )

        deployer = DeploymentAccessContext(
            tenant_id="robotics_lab",
            user_id="mlops_engineer",
            can_read=True,
            can_deploy=True,
            can_manage=True,
        )

        approver = DeploymentAccessContext(
            tenant_id="robotics_lab",
            user_id="platform_admin",
            can_read=True,
            can_approve=True,
            can_manage=True,
        )

        artifact = (
            SyntheticDeploymentFactory.create_model_artifact(
                root / "models",
                version="2.0.0",
            )
        )

        specification = DeploymentSpecification(
            deployment_name="robot-vision-production-v2",
            environment=DeploymentEnvironment.PRODUCTION,
            strategy=DeploymentStrategy.CANARY,
            model=artifact,
            resources=DeploymentResources(
                cpu_cores=4.0,
                memory_mb=8192,
                gpu_count=1,
                gpu_memory_mb=16384,
                replicas=3,
            ),
            endpoint_name="robot-vision",
            labels={
                "application": "physical_ai",
                "component": "visual_perception",
            },
        )

        deployment = manager.register_deployment(
            specification,
            deployer,
        )

        manager.review_approval(
            deployment.deployment_id,
            approver,
            approve=True,
            reason="Validated for staged rollout",
        )

        manager.start_deployment(
            deployment.deployment_id,
            deployer,
        )

        manager.activate_deployment(
            deployment.deployment_id,
            deployer,
        )

        health = manager.update_health(
            deployment.deployment_id,
            SyntheticDeploymentFactory.healthy_metrics(),
            deployer,
        )

        manager.update_traffic(
            deployment.deployment_id,
            25.0,
            deployer,
        )

        current = manager.get_deployment(
            deployment.deployment_id,
            deployer,
        )

        rollout = manager.rollout_plan(
            deployment.deployment_id,
            deployer,
        )

        print("\nPHYSICAL AI DEPLOYMENT MANAGER")
        print("=" * 70)

        print(
            json.dumps(
                {
                    "deployment_id": current.deployment_id,
                    "deployment_name": (
                        current.specification.deployment_name
                    ),
                    "model_version": (
                        current.specification.model.model_version
                    ),
                    "environment": (
                        current.specification.environment.value
                    ),
                    "strategy": (
                        current.specification.strategy.value
                    ),
                    "status": current.status.value,
                    "desired_traffic_percentage": (
                        current.traffic_percentage
                    ),
                    "health": json_safe(health),
                    "rollout_plan": json_safe(rollout),
                    "audit_events": len(
                        manager.deployment_history(
                            current.deployment_id,
                            deployer,
                        )
                    ),
                    "infrastructure_changes_executed": False,
                },
                indent=2,
                allow_nan=False,
            )
        )

    print("\nRUNNING UNIT TESTS")
    print("=" * 70)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestDeploymentManager
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
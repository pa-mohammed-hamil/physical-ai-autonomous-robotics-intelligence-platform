"""Model Registry module.

TODO: implement and test; no real robot control is enabled.
"""
"""Incident module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/db/models/model_registry.py

Physical AI & Autonomous Robotics Intelligence Platform

SQLAlchemy 2.x ORM models for:
- AI/ML model registry
- Model versions and artifact tracking
- Model lifecycle management
- Model approval and governance
- Deployment records
- Model performance and evaluation metrics
- Model lineage and reproducibility
- Model registry audit events
- Multi-tenant isolation

Requires:
    SQLAlchemy >= 2.0

Notes:
    Model approval does not authorize deployment to physical
    robots. Deployment services must independently enforce
    authorization, safety validation and rollout policies.
"""

from __future__ import annotations

import math
import uuid

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    validates,
)

from app.db.base import Base


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    return str(uuid.uuid4())


def serialize_datetime(
    value: Optional[datetime],
) -> Optional[str]:
    return value.isoformat() if value else None


def validate_finite(
    value: Optional[float],
    field_name: str,
) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")

    return number


# ============================================================
# Enumerations
# ============================================================


class RegistryModelType(str, Enum):
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    COMPUTER_VISION = "computer_vision"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    POSE_ESTIMATION = "pose_estimation"
    DEPTH_ESTIMATION = "depth_estimation"
    SENSOR_FUSION = "sensor_fusion"
    NAVIGATION = "navigation"
    LOCALIZATION = "localization"
    MOTION_PLANNING = "motion_planning"
    MANIPULATION = "manipulation"
    GRASPING = "grasping"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    IMITATION_LEARNING = "imitation_learning"
    DIFFUSION_POLICY = "diffusion_policy"
    VISION_LANGUAGE_ACTION = "vision_language_action"
    WORLD_MODEL = "world_model"
    LANGUAGE_MODEL = "language_model"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    RAG = "rag"
    AI_AGENT = "ai_agent"
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    REMAINING_USEFUL_LIFE = "remaining_useful_life"
    TIME_SERIES = "time_series"
    MULTIMODAL = "multimodal"
    OTHER = "other"


class RegistryModelStatus(str, Enum):
    DRAFT = "draft"
    DEVELOPMENT = "development"
    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    DISABLED = "disabled"


class ModelVersionStatus(str, Enum):
    REGISTERED = "registered"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REJECTED = "rejected"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelStage(str, Enum):
    NONE = "none"
    EXPERIMENTAL = "experimental"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelFramework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    KERAS = "keras"
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    HUGGINGFACE = "huggingface"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    JAX = "jax"
    ULTRALYTICS = "ultralytics"
    ROS = "ros"
    CUSTOM = "custom"
    OTHER = "other"


class ModelDeploymentStatus(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"
    ROLLED_BACK = "rolled_back"


class ModelDeploymentEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    EDGE = "edge"
    ROBOT = "robot"
    CLOUD = "cloud"
    SIMULATION = "simulation"


class ModelRegistryEventType(str, Enum):
    MODEL_REGISTERED = "model_registered"
    MODEL_UPDATED = "model_updated"
    MODEL_DELETED = "model_deleted"
    MODEL_RESTORED = "model_restored"
    VERSION_REGISTERED = "version_registered"
    VERSION_VALIDATED = "version_validated"
    VERSION_APPROVED = "version_approved"
    VERSION_REJECTED = "version_rejected"
    VERSION_PROMOTED = "version_promoted"
    VERSION_DEPRECATED = "version_deprecated"
    DEPLOYMENT_CREATED = "deployment_created"
    DEPLOYMENT_STARTED = "deployment_started"
    DEPLOYMENT_SUCCEEDED = "deployment_succeeded"
    DEPLOYMENT_FAILED = "deployment_failed"
    DEPLOYMENT_STOPPED = "deployment_stopped"
    DEPLOYMENT_ROLLED_BACK = "deployment_rolled_back"
    METRIC_RECORDED = "metric_recorded"
    ARTIFACT_UPDATED = "artifact_updated"
    SECURITY_REVIEW = "security_review"
    SAFETY_REVIEW = "safety_review"
    NOTE_ADDED = "note_added"


# ============================================================
# Model Registry
# ============================================================


class ModelRegistry(Base):
    """
    Registered AI model and its logical identity.

    Individual trained artifacts belong to ModelRegistryVersion.
    """

    __tablename__ = "model_registry"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_model_registry_tenant_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "name",
            name="uq_model_registry_tenant_name",
        ),
        Index(
            "ix_registry_tenant_type",
            "tenant_id",
            "model_type",
        ),
        Index(
            "ix_registry_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_registry_tenant_owner",
            "tenant_id",
            "owner_id",
        ),
        Index(
            "ix_registry_tenant_created",
            "tenant_id",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    organization_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    display_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    model_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=RegistryModelType.MACHINE_LEARNING.value,
    )

    framework: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=ModelFramework.CUSTOM.value,
    )

    task_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=RegistryModelStatus.DRAFT.value,
    )

    owner_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    created_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    repository_url: Mapped[Optional[str]] = mapped_column(
        String(2048),
        nullable=True,
    )

    documentation_url: Mapped[Optional[str]] = mapped_column(
        String(2048),
        nullable=True,
    )

    license_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    tags_json: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    configuration_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    versions: Mapped[list["ModelRegistryVersion"]] = relationship(
        "ModelRegistryVersion",
        back_populates="registry",
        cascade="save-update, merge",
        lazy="selectin",
    )

    events: Mapped[list["ModelRegistryEvent"]] = relationship(
        "ModelRegistryEvent",
        back_populates="registry",
        cascade="save-update, merge",
        lazy="selectin",
    )

    @property
    def is_active(self) -> bool:
        return (
            not self.is_deleted
            and self.status in {
                RegistryModelStatus.READY.value,
                RegistryModelStatus.ACTIVE.value,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "model_type": self.model_type,
            "framework": self.framework,
            "task_type": self.task_type,
            "status": self.status,
            "owner_id": self.owner_id,
            "repository_url": self.repository_url,
            "documentation_url": self.documentation_url,
            "license_name": self.license_name,
            "tags": self.tags_json,
            "configuration": self.configuration_json,
            "metadata": self.metadata_json,
            "is_deleted": self.is_deleted,
            "created_at": serialize_datetime(self.created_at),
            "updated_at": serialize_datetime(self.updated_at),
        }

    def __repr__(self) -> str:
        return (
            f"<ModelRegistry("
            f"id={self.id!r}, "
            f"name={self.name!r}, "
            f"status={self.status!r}"
            f")>"
        )


# ============================================================
# Model Registry Version
# ============================================================


class ModelRegistryVersion(Base):
    """
    Immutable model artifact identity with mutable
    governance and lifecycle metadata.

    Artifact checksum and URI should not be modified
    after version registration. Enforce this through
    service policy and database permissions.
    """

    __tablename__ = "model_registry_versions"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "registry_id"],
            ["model_registry.tenant_id", "model_registry.id"],
            name="fk_registry_version_tenant_registry",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_registry_version_tenant_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "registry_id",
            "version",
            name="uq_registry_model_version",
        ),
        Index(
            "ix_registry_version_tenant_registry",
            "tenant_id",
            "registry_id",
        ),
        Index(
            "ix_registry_version_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_registry_version_tenant_stage",
            "tenant_id",
            "stage",
        ),
        Index(
            "ix_registry_version_tenant_created",
            "tenant_id",
            "created_at",
        ),
        CheckConstraint(
            "model_size_bytes IS NULL OR model_size_bytes >= 0",
            name="ck_registry_version_size",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    registry_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ModelVersionStatus.REGISTERED.value,
    )

    stage: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ModelStage.NONE.value,
    )

    framework_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    python_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    artifact_uri: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
    )

    artifact_checksum: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
    )

    artifact_format: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    model_size_bytes: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    experiment_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    experiment_run_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    dataset_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    dataset_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    evaluation_result_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    source_commit: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    container_image_digest: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    metrics_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    parameters_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    input_schema_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    output_schema_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    dependencies_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    training_config_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    safety_report_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    is_approved: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    approved_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    approved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    registered_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    registry: Mapped["ModelRegistry"] = relationship(
        "ModelRegistry",
        back_populates="versions",
    )

    deployments: Mapped[list["RegistryModelDeployment"]] = (
        relationship(
            "RegistryModelDeployment",
            back_populates="model_version",
            cascade="save-update, merge",
            lazy="selectin",
        )
    )

    metrics: Mapped[list["RegistryModelMetric"]] = relationship(
        "RegistryModelMetric",
        back_populates="model_version",
        cascade="save-update, merge",
        lazy="selectin",
    )

    def approve(self, user_id: str) -> None:
        if self.status not in {
            ModelVersionStatus.REGISTERED.value,
            ModelVersionStatus.VALIDATED.value,
        }:
            raise ValueError(
                "Model version cannot be approved "
                "from its current status"
            )

        self.is_approved = True
        self.approved_by = user_id
        self.approved_at = utc_now()
        self.status = ModelVersionStatus.APPROVED.value
        self.updated_at = utc_now()

    def reject(self) -> None:
        if self.status in {
            ModelVersionStatus.DEPLOYED.value,
            ModelVersionStatus.ARCHIVED.value,
        }:
            raise ValueError(
                "Cannot reject deployed or archived versions"
            )

        self.is_approved = False
        self.approved_by = None
        self.approved_at = None
        self.status = ModelVersionStatus.REJECTED.value
        self.updated_at = utc_now()

    @property
    def is_deployable(self) -> bool:
        return (
            self.is_approved
            and self.status in {
                ModelVersionStatus.APPROVED.value,
                ModelVersionStatus.DEPLOYED.value,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "registry_id": self.registry_id,
            "version": self.version,
            "status": self.status,
            "stage": self.stage,
            "framework_version": self.framework_version,
            "artifact_uri": self.artifact_uri,
            "artifact_checksum": self.artifact_checksum,
            "artifact_format": self.artifact_format,
            "model_size_bytes": self.model_size_bytes,
            "experiment_id": self.experiment_id,
            "experiment_run_id": self.experiment_run_id,
            "dataset_id": self.dataset_id,
            "evaluation_result_id": self.evaluation_result_id,
            "source_commit": self.source_commit,
            "metrics": self.metrics_json,
            "parameters": self.parameters_json,
            "is_approved": self.is_approved,
            "approved_by": self.approved_by,
            "approved_at": serialize_datetime(self.approved_at),
            "created_at": serialize_datetime(self.created_at),
            "updated_at": serialize_datetime(self.updated_at),
        }


# ============================================================
# Model Deployment
# ============================================================


class RegistryModelDeployment(Base):
    """
    Deployment record for a registered model version.

    This model records deployment state. It does not
    execute model rollout or robot-control operations.
    """

    __tablename__ = "registry_model_deployments"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "model_version_id"],
            [
                "model_registry_versions.tenant_id",
                "model_registry_versions.id",
            ],
            name="fk_registry_deployment_tenant_version",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_registry_deployment_tenant_version",
            "tenant_id",
            "model_version_id",
        ),
        Index(
            "ix_registry_deployment_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_registry_deployment_tenant_environment",
            "tenant_id",
            "environment",
        ),
        Index(
            "ix_registry_deployment_tenant_robot",
            "tenant_id",
            "robot_id",
        ),
        CheckConstraint(
            "traffic_percentage IS NULL OR "
            "(traffic_percentage >= 0 AND traffic_percentage <= 100)",
            name="ck_registry_deployment_traffic",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    model_version_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    deployment_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    environment: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=ModelDeploymentEnvironment.STAGING.value,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ModelDeploymentStatus.PENDING.value,
    )

    endpoint_url: Mapped[Optional[str]] = mapped_column(
        String(2048),
        nullable=True,
    )

    robot_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    fleet_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    cluster_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    namespace: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    traffic_percentage: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    configuration_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    resource_limits_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    deployed_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    deployed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    stopped_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    model_version: Mapped["ModelRegistryVersion"] = relationship(
        "ModelRegistryVersion",
        back_populates="deployments",
    )

    @validates("traffic_percentage")
    def validate_traffic(
        self,
        key: str,
        value: Optional[float],
    ) -> Optional[float]:
        number = validate_finite(value, key)

        if number is not None and not 0 <= number <= 100:
            raise ValueError(
                "traffic_percentage must be between 0 and 100"
            )

        return number

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "model_version_id": self.model_version_id,
            "deployment_name": self.deployment_name,
            "environment": self.environment,
            "status": self.status,
            "endpoint_url": self.endpoint_url,
            "robot_id": self.robot_id,
            "fleet_id": self.fleet_id,
            "cluster_name": self.cluster_name,
            "namespace": self.namespace,
            "traffic_percentage": self.traffic_percentage,
            "configuration": self.configuration_json,
            "resource_limits": self.resource_limits_json,
            "deployed_by": self.deployed_by,
            "deployed_at": serialize_datetime(self.deployed_at),
            "stopped_at": serialize_datetime(self.stopped_at),
            "error_message": self.error_message,
            "created_at": serialize_datetime(self.created_at),
        }


# ============================================================
# Model Metric
# ============================================================


class RegistryModelMetric(Base):
    """
    Evaluation or monitoring metric for a model version.
    """

    __tablename__ = "registry_model_metrics"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "model_version_id"],
            [
                "model_registry_versions.tenant_id",
                "model_registry_versions.id",
            ],
            name="fk_registry_metric_tenant_version",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_registry_metric_tenant_version",
            "tenant_id",
            "model_version_id",
        ),
        Index(
            "ix_registry_metric_tenant_name",
            "tenant_id",
            "metric_name",
        ),
        CheckConstraint(
            "sample_count IS NULL OR sample_count >= 0",
            name="ck_registry_metric_sample_count",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    model_version_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    metric_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    metric_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    metric_unit: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    metric_category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    dataset_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    sample_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    threshold_value: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    higher_is_better: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    passed: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    model_version: Mapped["ModelRegistryVersion"] = relationship(
        "ModelRegistryVersion",
        back_populates="metrics",
    )

    @validates("metric_value", "threshold_value")
    def validate_numeric(
        self,
        key: str,
        value: Optional[float],
    ) -> Optional[float]:
        return validate_finite(value, key)

    def evaluate_threshold(self) -> Optional[bool]:
        if self.threshold_value is None:
            self.passed = None
            return None

        if self.higher_is_better:
            self.passed = (
                self.metric_value >= self.threshold_value
            )
        else:
            self.passed = (
                self.metric_value <= self.threshold_value
            )

        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "model_version_id": self.model_version_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_unit": self.metric_unit,
            "metric_category": self.metric_category,
            "dataset_name": self.dataset_name,
            "sample_count": self.sample_count,
            "threshold_value": self.threshold_value,
            "higher_is_better": self.higher_is_better,
            "passed": self.passed,
            "metadata": self.metadata_json,
            "recorded_at": serialize_datetime(self.recorded_at),
        }


# ============================================================
# Model Registry Event
# ============================================================


class ModelRegistryEvent(Base):
    """
    Append-only model registry lifecycle event.
    """

    __tablename__ = "model_registry_events"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "registry_id"],
            ["model_registry.tenant_id", "model_registry.id"],
            name="fk_registry_event_tenant_registry",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_registry_event_tenant_registry",
            "tenant_id",
            "registry_id",
        ),
        Index(
            "ix_registry_event_tenant_type",
            "tenant_id",
            "event_type",
        ),
        Index(
            "ix_registry_event_tenant_created",
            "tenant_id",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    registry_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    model_version_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
    )

    deployment_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    actor_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    previous_status: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
    )

    new_status: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    registry: Mapped["ModelRegistry"] = relationship(
        "ModelRegistry",
        back_populates="events",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "registry_id": self.registry_id,
            "model_version_id": self.model_version_id,
            "deployment_id": self.deployment_id,
            "event_type": self.event_type,
            "message": self.message,
            "actor_id": self.actor_id,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "metadata": self.metadata_json,
            "created_at": serialize_datetime(self.created_at),
        }


# ============================================================
# Exports
# ============================================================


__all__ = [
    "ModelRegistry",
    "ModelRegistryVersion",
    "RegistryModelDeployment",
    "RegistryModelMetric",
    "ModelRegistryEvent",
    "RegistryModelType",
    "RegistryModelStatus",
    "ModelVersionStatus",
    "ModelStage",
    "ModelFramework",
    "ModelDeploymentStatus",
    "ModelDeploymentEnvironment",
    "ModelRegistryEventType",
    "utc_now",
    "generate_uuid",
    "serialize_datetime",
    "validate_finite",
]
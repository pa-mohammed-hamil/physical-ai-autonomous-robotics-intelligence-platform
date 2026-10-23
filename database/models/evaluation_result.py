"""
backend/app/db/models/evaluation_result.py

Physical AI & Autonomous Robotics Intelligence Platform

SQLAlchemy 2.x ORM models for storing evaluation results.

Supports:
- Machine learning and deep learning evaluation
- Robotics perception and navigation benchmarks
- Manipulation and motion planning evaluation
- Vision-Language-Action (VLA) evaluation
- Neural world model evaluation
- Reinforcement learning evaluation
- AI agent and multi-agent evaluation
- RAG and LLM evaluation
- Robot safety and reliability evaluation
- Simulation and real-world evaluation
- Model version comparisons
- Evaluation metrics and threshold checks
- Failure analysis and test-case results
- Evaluation artifacts
- Human review and approval

Dependencies:
    sqlalchemy>=2.0

Database:
    PostgreSQL recommended.

Security:
    Evaluation results are tenant-scoped.
    Artifact references must be authorized before access.
    Evaluation approval does not automatically authorize
    deployment or physical robot control.
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


def enum_value(value: Enum | str) -> str:
    if isinstance(value, Enum):
        return str(value.value)

    return str(value)


def validate_finite_number(
    value: Optional[float],
    field_name: str,
) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be numeric"
        )

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(
            f"{field_name} must be finite"
        )

    return number


# ============================================================
# Enumerations
# ============================================================


class EvaluationResultStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class EvaluationResultType(str, Enum):
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
    VISION_LANGUAGE_ACTION = "vision_language_action"
    WORLD_MODEL = "world_model"
    ROBOT_SAFETY = "robot_safety"
    ROBOT_HEALTH = "robot_health"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    ANOMALY_DETECTION = "anomaly_detection"
    REMAINING_USEFUL_LIFE = "remaining_useful_life"
    AI_AGENT = "ai_agent"
    MULTI_AGENT = "multi_agent"
    LLM = "llm"
    RAG = "rag"
    SIMULATION = "simulation"
    HARDWARE = "hardware"
    END_TO_END = "end_to_end"
    OTHER = "other"


class EvaluationEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    SIMULATION = "simulation"
    DIGITAL_TWIN = "digital_twin"
    REAL_WORLD = "real_world"
    EDGE = "edge"
    ROBOT = "robot"
    LABORATORY = "laboratory"


class EvaluationMetricCategory(str, Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    IOU = "iou"
    MAP = "map"
    LOSS = "loss"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    COMPUTE = "compute"
    ENERGY = "energy"
    COST = "cost"
    SAFETY = "safety"
    RELIABILITY = "reliability"
    ROBUSTNESS = "robustness"
    SUCCESS_RATE = "success_rate"
    COLLISION_RATE = "collision_rate"
    PATH_EFFICIENCY = "path_efficiency"
    TRAJECTORY_ERROR = "trajectory_error"
    GRASP_SUCCESS = "grasp_success"
    TASK_COMPLETION = "task_completion"
    POLICY_REWARD = "policy_reward"
    WORLD_MODEL_PREDICTION = "world_model_prediction"
    GROUNDEDNESS = "groundedness"
    FAITHFULNESS = "faithfulness"
    HALLUCINATION = "hallucination"
    RETRIEVAL = "retrieval"
    CITATION_ACCURACY = "citation_accuracy"
    TOOL_CALL_ACCURACY = "tool_call_accuracy"
    OTHER = "other"


class EvaluationReviewStatus(str, Enum):
    NOT_REVIEWED = "not_reviewed"
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


class EvaluationFailureType(str, Enum):
    MODEL_ERROR = "model_error"
    INFERENCE_ERROR = "inference_error"
    DATA_ERROR = "data_error"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SAFETY_VIOLATION = "safety_violation"
    COLLISION = "collision"
    NAVIGATION_FAILURE = "navigation_failure"
    MANIPULATION_FAILURE = "manipulation_failure"
    PERCEPTION_FAILURE = "perception_failure"
    SENSOR_FAILURE = "sensor_failure"
    HARDWARE_FAILURE = "hardware_failure"
    COMMUNICATION_FAILURE = "communication_failure"
    AGENT_FAILURE = "agent_failure"
    TOOL_FAILURE = "tool_failure"
    POLICY_FAILURE = "policy_failure"
    BENCHMARK_FAILURE = "benchmark_failure"
    OTHER = "other"


class EvaluationArtifactType(str, Enum):
    REPORT = "report"
    METRICS = "metrics"
    LOG = "log"
    TRACE = "trace"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DATASET = "dataset"
    PREDICTIONS = "predictions"
    CONFUSION_MATRIX = "confusion_matrix"
    ROC_CURVE = "roc_curve"
    PRECISION_RECALL_CURVE = "precision_recall_curve"
    TRAJECTORY = "trajectory"
    SIMULATION_RECORDING = "simulation_recording"
    SAFETY_REPORT = "safety_report"
    OTHER = "other"


# ============================================================
# Main Evaluation Result
# ============================================================


class EvaluationResult(Base):
    """
    Stores one evaluation execution result.

    A result can reference an existing evaluation run,
    AI model, model version, robot, agent, benchmark,
    dataset, or simulation.

    Relationships to other domain tables are intentionally
    represented by identifiers to avoid circular imports.
    Services must verify referenced resources belong to
    the same tenant.
    """

    __tablename__ = "evaluation_results"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_evaluation_result_tenant_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "result_number",
            name="uq_evaluation_result_number",
        ),
        Index(
            "ix_eval_result_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_eval_result_tenant_type",
            "tenant_id",
            "evaluation_type",
        ),
        Index(
            "ix_eval_result_tenant_model",
            "tenant_id",
            "model_id",
        ),
        Index(
            "ix_eval_result_tenant_robot",
            "tenant_id",
            "robot_id",
        ),
        Index(
            "ix_eval_result_tenant_agent",
            "tenant_id",
            "agent_id",
        ),
        Index(
            "ix_eval_result_tenant_benchmark",
            "tenant_id",
            "benchmark_name",
        ),
        Index(
            "ix_eval_result_tenant_created",
            "tenant_id",
            "created_at",
        ),
        CheckConstraint(
            "duration_ms IS NULL OR duration_ms >= 0",
            name="ck_eval_result_duration_nonnegative",
        ),
        CheckConstraint(
            "total_test_cases >= 0",
            name="ck_eval_result_total_cases",
        ),
        CheckConstraint(
            "passed_test_cases >= 0",
            name="ck_eval_result_passed_cases",
        ),
        CheckConstraint(
            "failed_test_cases >= 0",
            name="ck_eval_result_failed_cases",
        ),
        CheckConstraint(
            "skipped_test_cases >= 0",
            name="ck_eval_result_skipped_cases",
        ),
        CheckConstraint(
            "passed_test_cases + failed_test_cases "
            "+ skipped_test_cases <= total_test_cases",
            name="ck_eval_result_case_counts",
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

    result_number: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    evaluation_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    model_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_version_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_version: Mapped[Optional[str]] = mapped_column(
        String(100),
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

    agent_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    experiment_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    simulation_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    evaluation_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=EvaluationResultType.MACHINE_LEARNING.value,
    )

    environment: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=EvaluationEnvironment.TESTING.value,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=EvaluationResultStatus.PENDING.value,
    )

    benchmark_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    benchmark_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    dataset_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    dataset_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    evaluation_framework: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    evaluation_framework_version: Mapped[Optional[str]] = (
        mapped_column(
            String(100),
            nullable=True,
        )
    )

    score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    max_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    normalized_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    passed: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )

    total_test_cases: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    passed_test_cases: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    failed_test_cases: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    skipped_test_cases: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    metrics_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    configuration_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    environment_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    error_code: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    failure_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    recommendations: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    duration_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    review_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=EvaluationReviewStatus.NOT_REVIEWED.value,
    )

    reviewed_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    review_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
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

    metric_records: Mapped[list["EvaluationResultMetric"]] = (
        relationship(
            "EvaluationResultMetric",
            back_populates="evaluation_result",
            cascade="save-update, merge",
            lazy="selectin",
        )
    )

    test_cases: Mapped[list["EvaluationTestCaseResult"]] = (
        relationship(
            "EvaluationTestCaseResult",
            back_populates="evaluation_result",
            cascade="save-update, merge",
            lazy="selectin",
        )
    )

    failures: Mapped[list["EvaluationFailure"]] = relationship(
        "EvaluationFailure",
        back_populates="evaluation_result",
        cascade="save-update, merge",
        lazy="selectin",
    )

    artifacts: Mapped[list["EvaluationArtifact"]] = relationship(
        "EvaluationArtifact",
        back_populates="evaluation_result",
        cascade="save-update, merge",
        lazy="selectin",
    )

    @validates(
        "score",
        "max_score",
        "normalized_score",
        "duration_ms",
    )
    def validate_numeric(
        self,
        key: str,
        value: Optional[float],
    ) -> Optional[float]:
        number = validate_finite_number(
            value,
            key,
        )

        if (
            number is not None
            and key == "duration_ms"
            and number < 0
        ):
            raise ValueError(
                "duration_ms cannot be negative"
            )

        if (
            number is not None
            and key == "normalized_score"
            and not 0 <= number <= 1
        ):
            raise ValueError(
                "normalized_score must be between 0 and 1"
            )

        return number

    def start(self) -> None:
        if self.status != EvaluationResultStatus.PENDING.value:
            raise ValueError(
                "Only pending evaluations can start"
            )

        now = utc_now()

        self.status = EvaluationResultStatus.RUNNING.value
        self.started_at = now
        self.updated_at = now

    def complete(
        self,
        *,
        score: Optional[float] = None,
        passed: Optional[bool] = None,
    ) -> None:
        if self.status != EvaluationResultStatus.RUNNING.value:
            raise ValueError(
                "Only running evaluations can complete"
            )

        now = utc_now()

        if score is not None:
            self.score = score

        self.passed = passed
        self.status = EvaluationResultStatus.COMPLETED.value
        self.completed_at = now
        self.updated_at = now

        if self.started_at is not None:
            self.duration_ms = max(
                0.0,
                (
                    now - self.started_at
                ).total_seconds() * 1000,
            )

    def fail(
        self,
        error_message: str,
        *,
        error_code: Optional[str] = None,
    ) -> None:
        if self.status not in {
            EvaluationResultStatus.PENDING.value,
            EvaluationResultStatus.RUNNING.value,
        }:
            raise ValueError(
                "Evaluation cannot fail from current status"
            )

        now = utc_now()

        self.status = EvaluationResultStatus.FAILED.value
        self.passed = False
        self.error_message = error_message
        self.error_code = error_code
        self.completed_at = now
        self.updated_at = now

    @property
    def success_rate(self) -> Optional[float]:
        if self.total_test_cases <= 0:
            return None

        return (
            self.passed_test_cases
            / self.total_test_cases
        )

    @property
    def is_completed(self) -> bool:
        return (
            self.status
            == EvaluationResultStatus.COMPLETED.value
        )

    @property
    def is_approved(self) -> bool:
        return (
            self.review_status
            == EvaluationReviewStatus.APPROVED.value
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "result_number": self.result_number,
            "evaluation_id": self.evaluation_id,
            "model_id": self.model_id,
            "model_version_id": self.model_version_id,
            "robot_id": self.robot_id,
            "agent_id": self.agent_id,
            "title": self.title,
            "evaluation_type": self.evaluation_type,
            "environment": self.environment,
            "status": self.status,
            "benchmark_name": self.benchmark_name,
            "dataset_name": self.dataset_name,
            "score": self.score,
            "max_score": self.max_score,
            "normalized_score": self.normalized_score,
            "passed": self.passed,
            "total_test_cases": self.total_test_cases,
            "passed_test_cases": self.passed_test_cases,
            "failed_test_cases": self.failed_test_cases,
            "skipped_test_cases": self.skipped_test_cases,
            "success_rate": self.success_rate,
            "metrics": self.metrics_json,
            "duration_ms": self.duration_ms,
            "review_status": self.review_status,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "started_at": (
                self.started_at.isoformat()
                if self.started_at
                else None
            ),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at
                else None
            ),
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
        }


# ============================================================
# Evaluation Metric
# ============================================================


class EvaluationResultMetric(Base):
    """
    Individual benchmark metric.

    Examples:
        accuracy
        precision
        recall
        F1
        mAP
        IoU
        latency
        collision_rate
        navigation_success_rate
        grasp_success_rate
        world_model_prediction_error
        RAG faithfulness
        agent tool-call accuracy
    """

    __tablename__ = "evaluation_result_metrics"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "evaluation_result_id"],
            [
                "evaluation_results.tenant_id",
                "evaluation_results.id",
            ],
            name="fk_eval_metric_tenant_result",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_eval_metric_tenant_result",
            "tenant_id",
            "evaluation_result_id",
        ),
        Index(
            "ix_eval_metric_tenant_name",
            "tenant_id",
            "metric_name",
        ),
        CheckConstraint(
            "sample_count IS NULL OR sample_count >= 0",
            name="ck_eval_metric_sample_count",
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

    evaluation_result_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    metric_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    metric_category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=EvaluationMetricCategory.OTHER.value,
    )

    metric_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    metric_unit: Mapped[Optional[str]] = mapped_column(
        String(64),
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

    sample_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    confidence_interval_lower: Mapped[Optional[float]] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    confidence_interval_upper: Mapped[Optional[float]] = (
        mapped_column(
            Float,
            nullable=True,
        )
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

    evaluation_result: Mapped["EvaluationResult"] = (
        relationship(
            "EvaluationResult",
            back_populates="metric_records",
        )
    )

    @validates(
        "metric_value",
        "threshold_value",
        "confidence_interval_lower",
        "confidence_interval_upper",
    )
    def validate_numeric(
        self,
        key: str,
        value: Optional[float],
    ) -> Optional[float]:
        return validate_finite_number(
            value,
            key,
        )

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
            "evaluation_result_id": self.evaluation_result_id,
            "metric_name": self.metric_name,
            "metric_category": self.metric_category,
            "metric_value": self.metric_value,
            "metric_unit": self.metric_unit,
            "threshold_value": self.threshold_value,
            "higher_is_better": self.higher_is_better,
            "passed": self.passed,
            "sample_count": self.sample_count,
            "confidence_interval_lower": (
                self.confidence_interval_lower
            ),
            "confidence_interval_upper": (
                self.confidence_interval_upper
            ),
            "metadata": self.metadata_json,
        }


# ============================================================
# Evaluation Test Case Result
# ============================================================


class EvaluationTestCaseResult(Base):
    """
    Per-test-case evaluation result.

    Stores structured inputs and outputs only when
    permitted by the dataset's privacy and security
    policy. Large artifacts should use object storage.
    """

    __tablename__ = "evaluation_test_case_results"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "evaluation_result_id"],
            [
                "evaluation_results.tenant_id",
                "evaluation_results.id",
            ],
            name="fk_eval_test_case_tenant_result",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "tenant_id",
            "evaluation_result_id",
            "test_case_id",
            name="uq_eval_test_case_result",
        ),
        Index(
            "ix_eval_case_tenant_result",
            "tenant_id",
            "evaluation_result_id",
        ),
        Index(
            "ix_eval_case_tenant_passed",
            "tenant_id",
            "passed",
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

    evaluation_result_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    test_case_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    test_case_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    passed: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )

    score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    execution_time_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    input_json: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
    )

    expected_output_json: Mapped[
        Optional[dict[str, Any]]
    ] = mapped_column(
        JSON,
        nullable=True,
    )

    actual_output_json: Mapped[
        Optional[dict[str, Any]]
    ] = mapped_column(
        JSON,
        nullable=True,
    )

    metrics_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
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

    evaluation_result: Mapped["EvaluationResult"] = (
        relationship(
            "EvaluationResult",
            back_populates="test_cases",
        )
    )

    @validates("score", "execution_time_ms")
    def validate_numeric(
        self,
        key: str,
        value: Optional[float],
    ) -> Optional[float]:
        number = validate_finite_number(
            value,
            key,
        )

        if (
            key == "execution_time_ms"
            and number is not None
            and number < 0
        ):
            raise ValueError(
                "execution_time_ms cannot be negative"
            )

        return number

    def to_dict(
        self,
        *,
        include_payloads: bool = False,
    ) -> dict[str, Any]:
        result = {
            "id": self.id,
            "evaluation_result_id": self.evaluation_result_id,
            "test_case_id": self.test_case_id,
            "test_case_name": self.test_case_name,
            "category": self.category,
            "passed": self.passed,
            "score": self.score,
            "execution_time_ms": self.execution_time_ms,
            "metrics": self.metrics_json,
            "error_message": self.error_message,
        }

        if include_payloads:
            result["input"] = self.input_json
            result["expected_output"] = (
                self.expected_output_json
            )
            result["actual_output"] = (
                self.actual_output_json
            )

        return result


# ============================================================
# Evaluation Failure
# ============================================================


class EvaluationFailure(Base):
    """
    Failure analysis for unsuccessful evaluations.

    Examples:
    - Robot collision
    - Navigation timeout
    - Unsafe manipulation
    - Sensor malfunction
    - AI model inference failure
    - Agent tool execution failure
    - Simulation instability
    """

    __tablename__ = "evaluation_failures"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "evaluation_result_id"],
            [
                "evaluation_results.tenant_id",
                "evaluation_results.id",
            ],
            name="fk_eval_failure_tenant_result",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_eval_failure_tenant_result",
            "tenant_id",
            "evaluation_result_id",
        ),
        Index(
            "ix_eval_failure_tenant_type",
            "tenant_id",
            "failure_type",
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

    evaluation_result_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    test_case_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    failure_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=EvaluationFailureType.OTHER.value,
    )

    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="medium",
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    root_cause: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    recommended_action: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    is_safety_related: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    evaluation_result: Mapped["EvaluationResult"] = (
        relationship(
            "EvaluationResult",
            back_populates="failures",
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "evaluation_result_id": self.evaluation_result_id,
            "test_case_id": self.test_case_id,
            "failure_type": self.failure_type,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "root_cause": self.root_cause,
            "recommended_action": self.recommended_action,
            "is_safety_related": self.is_safety_related,
            "metadata": self.metadata_json,
            "occurred_at": (
                self.occurred_at.isoformat()
                if self.occurred_at
                else None
            ),
        }


# ============================================================
# Evaluation Artifact
# ============================================================


class EvaluationArtifact(Base):
    """
    Stores metadata for evaluation artifacts.

    Artifact content should normally be stored in
    S3, GCS, Azure Blob Storage or an authorized
    internal object storage system.

    Artifact URIs must not contain credentials or
    long-lived signed access tokens.
    """

    __tablename__ = "evaluation_artifacts"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "evaluation_result_id"],
            [
                "evaluation_results.tenant_id",
                "evaluation_results.id",
            ],
            name="fk_eval_artifact_tenant_result",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_eval_artifact_tenant_result",
            "tenant_id",
            "evaluation_result_id",
        ),
        Index(
            "ix_eval_artifact_tenant_type",
            "tenant_id",
            "artifact_type",
        ),
        CheckConstraint(
            "file_size_bytes IS NULL OR file_size_bytes >= 0",
            name="ck_eval_artifact_file_size",
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

    evaluation_result_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    artifact_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    artifact_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=EvaluationArtifactType.OTHER.value,
    )

    artifact_uri: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
    )

    content_type: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    file_size_bytes: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    checksum_sha256: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
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

    evaluation_result: Mapped["EvaluationResult"] = (
        relationship(
            "EvaluationResult",
            back_populates="artifacts",
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "evaluation_result_id": self.evaluation_result_id,
            "artifact_name": self.artifact_name,
            "artifact_type": self.artifact_type,
            "artifact_uri": self.artifact_uri,
            "content_type": self.content_type,
            "file_size_bytes": self.file_size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "description": self.description,
            "metadata": self.metadata_json,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
        }


# ============================================================
# Exports
# ============================================================


__all__ = [
    "EvaluationResult",
    "EvaluationResultMetric",
    "EvaluationTestCaseResult",
    "EvaluationFailure",
    "EvaluationArtifact",
    "EvaluationResultStatus",
    "EvaluationResultType",
    "EvaluationEnvironment",
    "EvaluationMetricCategory",
    "EvaluationReviewStatus",
    "EvaluationFailureType",
    "EvaluationArtifactType",
    "utc_now",
    "generate_uuid",
]   
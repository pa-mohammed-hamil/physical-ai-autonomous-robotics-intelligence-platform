"""Experiment Tracker module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/mlops/experiment_tracker.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Experiment Tracking and ML Research Management

Features:
    - Experiment creation and management
    - Training run tracking
    - Hyperparameter logging
    - Training, validation and test metrics
    - Metric history and learning curves
    - Model artifact tracking
    - Dataset version lineage
    - Model version and checkpoint tracking
    - Experiment comparison
    - Best-run selection
    - Hyperparameter search result analysis
    - Robotics simulation experiment tracking
    - Reinforcement learning experiment tracking
    - Computer vision experiment tracking
    - RAG / LLM evaluation tracking
    - GPU and compute resource metadata
    - Reproducibility metadata
    - Experiment tags and notes
    - Run lifecycle management
    - Thread-safe operations
    - Multi-tenant access control
    - Atomic JSON persistence
    - Audit events
    - CSV and JSON export
    - Unit tests and demonstration

Python 3.11+
Standard library only.

This is a local experiment-tracking reference implementation.
It does not execute training jobs, allocate GPUs, upload remote
artifacts, or control physical robots.

For production use, integrate durable transactional storage,
authenticated artifact stores, distributed coordination and
centralized access management.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import threading
import unittest
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class ExperimentTrackerError(Exception):
    pass


class ExperimentValidationError(ExperimentTrackerError):
    pass


class ExperimentNotFoundError(ExperimentTrackerError):
    pass


class RunNotFoundError(ExperimentTrackerError):
    pass


class ExperimentAuthorizationError(ExperimentTrackerError):
    pass


class ExperimentConflictError(ExperimentTrackerError):
    pass


class ExperimentPersistenceError(ExperimentTrackerError):
    pass


class ArtifactIntegrityError(ExperimentTrackerError):
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
        raise ExperimentValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def validate_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ExperimentValidationError(
            f"{name} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise ExperimentValidationError(
            f"{name} must be finite"
        )

    if minimum is not None and number < minimum:
        raise ExperimentValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and number > maximum:
        raise ExperimentValidationError(
            f"{name} must be <= {maximum}"
        )

    return number


def validate_json_value(
    value: Any,
) -> Any:
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
        )
        return json.loads(serialized)
    except (TypeError, ValueError) as exc:
        raise ExperimentValidationError(
            "Value must be JSON serializable"
        ) from exc


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def json_safe(
    value: Any,
) -> Any:
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

    if isinstance(value, (list, tuple, set)):
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

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".experiment-",
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

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


# ============================================================
# ENUMERATIONS
# ============================================================

class ExperimentType(str, Enum):
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    COMPUTER_VISION = "computer_vision"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ROBOTICS_SIMULATION = "robotics_simulation"
    ROBOT_NAVIGATION = "robot_navigation"
    ROBOT_MANIPULATION = "robot_manipulation"
    PHYSICAL_AI = "physical_ai"
    GENERATIVE_AI = "generative_ai"
    RAG = "rag"
    AGENTIC_AI = "agentic_ai"
    MULTIMODAL_AI = "multimodal_ai"
    OTHER = "other"


class ExperimentStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class MetricSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    EVALUATION = "evaluation"
    SIMULATION = "simulation"


class ArtifactType(str, Enum):
    MODEL = "model"
    CHECKPOINT = "checkpoint"
    DATASET = "dataset"
    CONFIGURATION = "configuration"
    PLOT = "plot"
    REPORT = "report"
    LOG = "log"
    VIDEO = "video"
    IMAGE = "image"
    ROBOT_TRAJECTORY = "robot_trajectory"
    SIMULATION_RECORDING = "simulation_recording"
    OTHER = "other"


class AuditAction(str, Enum):
    EXPERIMENT_CREATED = "experiment_created"
    EXPERIMENT_ARCHIVED = "experiment_archived"
    RUN_CREATED = "run_created"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    PARAMETERS_LOGGED = "parameters_logged"
    METRICS_LOGGED = "metrics_logged"
    ARTIFACT_REGISTERED = "artifact_registered"
    TAGS_UPDATED = "tags_updated"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ExperimentTrackerConfig:
    storage_root: Path = Path(
        "./data/mlops/experiment_tracker"
    )

    maximum_experiments: int = 10000
    maximum_runs_per_experiment: int = 100000
    maximum_metrics_per_run: int = 1000000

    verify_local_artifacts: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "storage_root",
            Path(self.storage_root).expanduser().resolve(),
        )

        for name in (
            "maximum_experiments",
            "maximum_runs_per_experiment",
            "maximum_metrics_per_run",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ExperimentValidationError(
                    f"{name} must be a positive integer"
                )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class ExperimentAccessContext:
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
# REPRODUCIBILITY METADATA
# ============================================================

@dataclass(frozen=True)
class ReproducibilityMetadata:
    python_version: str = ""
    framework_name: str = ""
    framework_version: str = ""

    git_commit: str = ""
    git_branch: str = ""

    docker_image: str = ""
    environment_name: str = ""

    random_seed: int | None = None

    dataset_id: str | None = None
    dataset_version_id: str | None = None

    hardware_name: str = ""
    gpu_name: str = ""
    gpu_count: int = 0

    cuda_version: str = ""

    extra: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.gpu_count, bool)
            or not isinstance(self.gpu_count, int)
            or self.gpu_count < 0
        ):
            raise ExperimentValidationError(
                "gpu_count must be a nonnegative integer"
            )

        if self.random_seed is not None:
            if (
                isinstance(self.random_seed, bool)
                or not isinstance(self.random_seed, int)
            ):
                raise ExperimentValidationError(
                    "random_seed must be an integer"
                )

        validate_json_value(
            self.extra
        )


# ============================================================
# EXPERIMENT MODELS
# ============================================================

@dataclass
class Experiment:
    experiment_id: str
    tenant_id: str

    name: str
    experiment_type: ExperimentType

    description: str

    created_by: str
    created_at: datetime
    updated_at: datetime

    status: ExperimentStatus = ExperimentStatus.ACTIVE

    tags: dict[str, str] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    revision: int = 1


@dataclass(frozen=True)
class MetricPoint:
    metric_name: str
    value: float

    step: int
    split: MetricSplit

    timestamp: datetime


@dataclass(frozen=True)
class RunArtifact:
    artifact_id: str

    name: str
    artifact_type: ArtifactType

    uri: str
    sha256: str | None

    size_bytes: int | None

    registered_at: datetime

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ExperimentRun:
    run_id: str
    experiment_id: str
    tenant_id: str

    run_name: str

    created_by: str
    created_at: datetime

    started_at: datetime | None = None
    ended_at: datetime | None = None

    status: RunStatus = RunStatus.CREATED

    parameters: dict[str, Any] = field(
        default_factory=dict
    )

    metrics: list[MetricPoint] = field(
        default_factory=list
    )

    artifacts: list[RunArtifact] = field(
        default_factory=list
    )

    tags: dict[str, str] = field(
        default_factory=dict
    )

    notes: str = ""

    reproducibility: ReproducibilityMetadata = field(
        default_factory=ReproducibilityMetadata
    )

    failure_reason: str = ""

    revision: int = 1

    def latest_metrics(
        self,
        split: MetricSplit | None = None,
    ) -> dict[str, float]:
        latest: dict[
            str,
            tuple[int, datetime, float],
        ] = {}

        for point in self.metrics:
            if (
                split is not None
                and point.split != split
            ):
                continue

            key = (
                f"{point.split.value}/"
                f"{point.metric_name}"
            )

            previous = latest.get(key)

            if (
                previous is None
                or (point.step, point.timestamp)
                > (previous[0], previous[1])
            ):
                latest[key] = (
                    point.step,
                    point.timestamp,
                    point.value,
                )

        return {
            key: entry[2]
            for key, entry in latest.items()
        }


@dataclass(frozen=True)
class ExperimentAuditEvent:
    event_id: str
    tenant_id: str

    experiment_id: str
    run_id: str | None

    action: AuditAction
    actor_id: str

    timestamp: datetime

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# COMPARISON MODELS
# ============================================================

@dataclass(frozen=True)
class RunComparison:
    run_id: str
    run_name: str
    status: RunStatus

    parameters: dict[str, Any]
    latest_metrics: dict[str, float]

    duration_seconds: float | None


@dataclass(frozen=True)
class ExperimentComparison:
    experiment_id: str

    run_count: int

    runs: tuple[RunComparison, ...]


@dataclass(frozen=True)
class BestRunResult:
    experiment_id: str
    run_id: str

    metric_name: str
    split: MetricSplit
    direction: MetricDirection

    metric_value: float


@dataclass(frozen=True)
class HyperparameterSummary:
    parameter_name: str

    values: tuple[Any, ...]
    run_count: int


# ============================================================
# ARTIFACT VERIFICATION
# ============================================================

class ExperimentArtifactVerifier:

    @staticmethod
    def inspect_local_file(
        uri: str,
    ) -> tuple[str, int]:
        if uri.startswith("file://"):
            path = Path(uri[7:])

        elif "://" not in uri:
            path = Path(uri)

        else:
            raise ExperimentValidationError(
                "Remote artifacts require an artifact-store adapter"
            )

        if not path.is_file():
            raise ArtifactIntegrityError(
                "Artifact file does not exist"
            )

        return (
            sha256_file(path),
            path.stat().st_size,
        )

    @classmethod
    def verify(
        cls,
        artifact: RunArtifact,
    ) -> bool:
        actual_hash, actual_size = (
            cls.inspect_local_file(
                artifact.uri
            )
        )

        if (
            artifact.sha256 is not None
            and actual_hash != artifact.sha256
        ):
            raise ArtifactIntegrityError(
                "Artifact SHA-256 mismatch"
            )

        if (
            artifact.size_bytes is not None
            and actual_size != artifact.size_bytes
        ):
            raise ArtifactIntegrityError(
                "Artifact size mismatch"
            )

        return True


# ============================================================
# METRIC ANALYTICS
# ============================================================

class ExperimentMetricAnalyzer:

    @staticmethod
    def metric_history(
        run: ExperimentRun,
        metric_name: str,
        split: MetricSplit,
    ) -> tuple[MetricPoint, ...]:
        return tuple(
            sorted(
                (
                    point
                    for point in run.metrics
                    if (
                        point.metric_name == metric_name
                        and point.split == split
                    )
                ),
                key=lambda point: (
                    point.step,
                    point.timestamp,
                ),
            )
        )

    @staticmethod
    def latest_value(
        run: ExperimentRun,
        metric_name: str,
        split: MetricSplit,
    ) -> float | None:
        history = (
            ExperimentMetricAnalyzer.metric_history(
                run,
                metric_name,
                split,
            )
        )

        if not history:
            return None

        return history[-1].value

    @staticmethod
    def best_value(
        run: ExperimentRun,
        metric_name: str,
        split: MetricSplit,
        direction: MetricDirection,
    ) -> float | None:
        history = (
            ExperimentMetricAnalyzer.metric_history(
                run,
                metric_name,
                split,
            )
        )

        if not history:
            return None

        values = [
            point.value
            for point in history
        ]

        if direction == MetricDirection.MAXIMIZE:
            return max(values)

        return min(values)

    @staticmethod
    def moving_average(
        points: Sequence[MetricPoint],
        window_size: int = 5,
    ) -> tuple[tuple[int, float], ...]:
        if window_size <= 0:
            raise ExperimentValidationError(
                "window_size must be positive"
            )

        ordered = sorted(
            points,
            key=lambda point: (
                point.step,
                point.timestamp,
            ),
        )

        result = []

        for index, point in enumerate(ordered):
            start = max(
                0,
                index - window_size + 1,
            )

            window = ordered[
                start:index + 1
            ]

            average = sum(
                item.value
                for item in window
            ) / len(window)

            result.append(
                (
                    point.step,
                    average,
                )
            )

        return tuple(result)


# ============================================================
# EXPERIMENT TRACKER
# ============================================================

class ExperimentTracker:

    def __init__(
        self,
        config: ExperimentTrackerConfig | None = None,
    ) -> None:
        self.config = (
            config or ExperimentTrackerConfig()
        )

        self.metric_analyzer = (
            ExperimentMetricAnalyzer()
        )

        self.artifact_verifier = (
            ExperimentArtifactVerifier()
        )

        self._experiments: dict[
            tuple[str, str],
            Experiment,
        ] = {}

        self._runs: dict[
            tuple[str, str],
            ExperimentRun,
        ] = {}

        self._events: list[
            ExperimentAuditEvent
        ] = []

        self._lock = threading.RLock()

        self._load()

    # --------------------------------------------------------
    # AUTHORIZATION
    # --------------------------------------------------------

    @staticmethod
    def _authorize(
        access: ExperimentAccessContext,
        permission: str,
    ) -> None:
        if not getattr(
            access,
            permission,
            False,
        ):
            raise ExperimentAuthorizationError(
                f"Missing permission: {permission}"
            )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    def _storage_path(self) -> Path:
        return (
            self.config.storage_root
            / "experiments.json"
        )

    def _persist(self) -> None:
        atomic_write_json(
            self._storage_path(),
            {
                "schema_version": 1,
                "experiments": [
                    json_safe(experiment)
                    for experiment
                    in self._experiments.values()
                ],
                "runs": [
                    json_safe(run)
                    for run in self._runs.values()
                ],
                "events": [
                    json_safe(event)
                    for event in self._events
                ],
            },
        )

    @staticmethod
    def _parse_datetime(
        value: str | None,
    ) -> datetime | None:
        if value is None:
            return None

        return datetime.fromisoformat(
            value
        )

    @classmethod
    def _parse_experiment(
        cls,
        data: Mapping[str, Any],
    ) -> Experiment:
        return Experiment(
            experiment_id=data["experiment_id"],
            tenant_id=data["tenant_id"],
            name=data["name"],
            experiment_type=ExperimentType(
                data["experiment_type"]
            ),
            description=data["description"],
            created_by=data["created_by"],
            created_at=datetime.fromisoformat(
                data["created_at"]
            ),
            updated_at=datetime.fromisoformat(
                data["updated_at"]
            ),
            status=ExperimentStatus(
                data["status"]
            ),
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {}),
            revision=data.get("revision", 1),
        )

    @classmethod
    def _parse_run(
        cls,
        data: Mapping[str, Any],
    ) -> ExperimentRun:
        reproducibility = dict(
            data.get(
                "reproducibility",
                {},
            )
        )

        metrics = [
            MetricPoint(
                metric_name=item["metric_name"],
                value=item["value"],
                step=item["step"],
                split=MetricSplit(
                    item["split"]
                ),
                timestamp=datetime.fromisoformat(
                    item["timestamp"]
                ),
            )
            for item in data.get(
                "metrics",
                [],
            )
        ]

        artifacts = [
            RunArtifact(
                artifact_id=item["artifact_id"],
                name=item["name"],
                artifact_type=ArtifactType(
                    item["artifact_type"]
                ),
                uri=item["uri"],
                sha256=item.get("sha256"),
                size_bytes=item.get(
                    "size_bytes"
                ),
                registered_at=datetime.fromisoformat(
                    item["registered_at"]
                ),
                metadata=item.get(
                    "metadata",
                    {},
                ),
            )
            for item in data.get(
                "artifacts",
                [],
            )
        ]

        return ExperimentRun(
            run_id=data["run_id"],
            experiment_id=data["experiment_id"],
            tenant_id=data["tenant_id"],
            run_name=data["run_name"],
            created_by=data["created_by"],
            created_at=datetime.fromisoformat(
                data["created_at"]
            ),
            started_at=cls._parse_datetime(
                data.get("started_at")
            ),
            ended_at=cls._parse_datetime(
                data.get("ended_at")
            ),
            status=RunStatus(
                data["status"]
            ),
            parameters=data.get(
                "parameters",
                {},
            ),
            metrics=metrics,
            artifacts=artifacts,
            tags=data.get("tags", {}),
            notes=data.get("notes", ""),
            reproducibility=ReproducibilityMetadata(
                **reproducibility
            ),
            failure_reason=data.get(
                "failure_reason",
                "",
            ),
            revision=data.get(
                "revision",
                1,
            ),
        )

    @staticmethod
    def _parse_event(
        data: Mapping[str, Any],
    ) -> ExperimentAuditEvent:
        return ExperimentAuditEvent(
            event_id=data["event_id"],
            tenant_id=data["tenant_id"],
            experiment_id=data["experiment_id"],
            run_id=data.get("run_id"),
            action=AuditAction(
                data["action"]
            ),
            actor_id=data["actor_id"],
            timestamp=datetime.fromisoformat(
                data["timestamp"]
            ),
            details=data.get(
                "details",
                {},
            ),
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
                raise ExperimentPersistenceError(
                    "Unsupported experiment registry schema"
                )

            for item in payload.get(
                "experiments",
                [],
            ):
                experiment = (
                    self._parse_experiment(
                        item
                    )
                )

                self._experiments[
                    (
                        experiment.tenant_id,
                        experiment.experiment_id,
                    )
                ] = experiment

            for item in payload.get(
                "runs",
                [],
            ):
                run = self._parse_run(
                    item
                )

                self._runs[
                    (
                        run.tenant_id,
                        run.run_id,
                    )
                ] = run

            for item in payload.get(
                "events",
                [],
            ):
                self._events.append(
                    self._parse_event(
                        item
                    )
                )

        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise ExperimentPersistenceError(
                "Invalid experiment registry"
            ) from exc

    # --------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------

    def _get_experiment(
        self,
        experiment_id: str,
        tenant_id: str,
    ) -> Experiment:
        experiment = self._experiments.get(
            (
                tenant_id,
                experiment_id,
            )
        )

        if experiment is None:
            raise ExperimentNotFoundError(
                "Experiment not found"
            )

        return experiment

    def _get_run(
        self,
        run_id: str,
        tenant_id: str,
    ) -> ExperimentRun:
        run = self._runs.get(
            (
                tenant_id,
                run_id,
            )
        )

        if run is None:
            raise RunNotFoundError(
                "Experiment run not found"
            )

        return run

    @staticmethod
    def _touch_run(
        run: ExperimentRun,
    ) -> None:
        run.revision += 1

    @staticmethod
    def _touch_experiment(
        experiment: Experiment,
    ) -> None:
        experiment.revision += 1
        experiment.updated_at = utc_now()

    def _audit(
        self,
        *,
        tenant_id: str,
        experiment_id: str,
        run_id: str | None,
        action: AuditAction,
        actor_id: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._events.append(
            ExperimentAuditEvent(
                event_id=generate_id(),
                tenant_id=tenant_id,
                experiment_id=experiment_id,
                run_id=run_id,
                action=action,
                actor_id=actor_id,
                timestamp=utc_now(),
                details=dict(details or {}),
            )
        )

    # --------------------------------------------------------
    # CREATE EXPERIMENT
    # --------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        experiment_type: ExperimentType,
        access: ExperimentAccessContext,
        *,
        description: str = "",
        tags: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Experiment:
        self._authorize(
            access,
            "can_write",
        )

        validate_identifier(
            name,
            "experiment name",
        )

        normalized_metadata = validate_json_value(
            dict(metadata or {})
        )

        with self._lock:
            if (
                len(self._experiments)
                >= self.config.maximum_experiments
            ):
                raise ExperimentConflictError(
                    "Experiment capacity reached"
                )

            for existing in self._experiments.values():
                if (
                    existing.tenant_id == access.tenant_id
                    and existing.name == name
                    and existing.status == ExperimentStatus.ACTIVE
                ):
                    raise ExperimentConflictError(
                        "Active experiment name already exists"
                    )

            experiment = Experiment(
                experiment_id=generate_id(),
                tenant_id=access.tenant_id,
                name=name,
                experiment_type=experiment_type,
                description=description,
                created_by=access.user_id,
                created_at=utc_now(),
                updated_at=utc_now(),
                tags=dict(tags or {}),
                metadata=normalized_metadata,
            )

            self._experiments[
                (
                    access.tenant_id,
                    experiment.experiment_id,
                )
            ] = experiment

            self._audit(
                tenant_id=access.tenant_id,
                experiment_id=experiment.experiment_id,
                run_id=None,
                action=AuditAction.EXPERIMENT_CREATED,
                actor_id=access.user_id,
            )

            self._persist()

            return copy.deepcopy(experiment)

    # --------------------------------------------------------
    # GET EXPERIMENT
    # --------------------------------------------------------

    def get_experiment(
        self,
        experiment_id: str,
        access: ExperimentAccessContext,
    ) -> Experiment:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            return copy.deepcopy(
                self._get_experiment(
                    experiment_id,
                    access.tenant_id,
                )
            )

    # --------------------------------------------------------
    # LIST EXPERIMENTS
    # --------------------------------------------------------

    def list_experiments(
        self,
        access: ExperimentAccessContext,
        *,
        experiment_type: ExperimentType | None = None,
        status: ExperimentStatus | None = None,
    ) -> tuple[Experiment, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            experiments = []

            for experiment in self._experiments.values():
                if experiment.tenant_id != access.tenant_id:
                    continue

                if (
                    experiment_type is not None
                    and experiment.experiment_type
                    != experiment_type
                ):
                    continue

                if (
                    status is not None
                    and experiment.status != status
                ):
                    continue

                experiments.append(
                    copy.deepcopy(experiment)
                )

            return tuple(
                sorted(
                    experiments,
                    key=lambda item: item.created_at,
                )
            )

    # --------------------------------------------------------
    # ARCHIVE EXPERIMENT
    # --------------------------------------------------------

    def archive_experiment(
        self,
        experiment_id: str,
        access: ExperimentAccessContext,
    ) -> Experiment:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            experiment = self._get_experiment(
                experiment_id,
                access.tenant_id,
            )

            if experiment.status == ExperimentStatus.ARCHIVED:
                return copy.deepcopy(experiment)

            active_runs = [
                run
                for run in self._runs.values()
                if (
                    run.tenant_id == access.tenant_id
                    and run.experiment_id == experiment_id
                    and run.status == RunStatus.RUNNING
                )
            ]

            if active_runs:
                raise ExperimentConflictError(
                    "Cannot archive an experiment with running runs"
                )

            experiment.status = ExperimentStatus.ARCHIVED

            self._touch_experiment(
                experiment
            )

            self._audit(
                tenant_id=access.tenant_id,
                experiment_id=experiment_id,
                run_id=None,
                action=AuditAction.EXPERIMENT_ARCHIVED,
                actor_id=access.user_id,
            )

            self._persist()

            return copy.deepcopy(experiment)

    # --------------------------------------------------------
    # CREATE RUN
    # --------------------------------------------------------

    def create_run(
        self,
        experiment_id: str,
        run_name: str,
        access: ExperimentAccessContext,
        *,
        parameters: Mapping[str, Any] | None = None,
        reproducibility: ReproducibilityMetadata | None = None,
        tags: Mapping[str, str] | None = None,
        notes: str = "",
    ) -> ExperimentRun:
        self._authorize(
            access,
            "can_write",
        )

        validate_identifier(
            run_name,
            "run_name",
        )

        normalized_parameters = validate_json_value(
            dict(parameters or {})
        )

        with self._lock:
            experiment = self._get_experiment(
                experiment_id,
                access.tenant_id,
            )

            if experiment.status != ExperimentStatus.ACTIVE:
                raise ExperimentConflictError(
                    "Cannot create runs in archived experiments"
                )

            run_count = sum(
                1
                for run in self._runs.values()
                if (
                    run.tenant_id == access.tenant_id
                    and run.experiment_id == experiment_id
                )
            )

            if (
                run_count
                >= self.config.maximum_runs_per_experiment
            ):
                raise ExperimentConflictError(
                    "Run capacity reached"
                )

            run = ExperimentRun(
                run_id=generate_id(),
                experiment_id=experiment_id,
                tenant_id=access.tenant_id,
                run_name=run_name,
                created_by=access.user_id,
                created_at=utc_now(),
                parameters=normalized_parameters,
                reproducibility=(
                    copy.deepcopy(reproducibility)
                    if reproducibility is not None
                    else ReproducibilityMetadata()
                ),
                tags=dict(tags or {}),
                notes=notes,
            )

            self._runs[
                (
                    access.tenant_id,
                    run.run_id,
                )
            ] = run

            self._touch_experiment(
                experiment
            )

            self._audit(
                tenant_id=access.tenant_id,
                experiment_id=experiment_id,
                run_id=run.run_id,
                action=AuditAction.RUN_CREATED,
                actor_id=access.user_id,
            )

            self._persist()

            return copy.deepcopy(run)

    # --------------------------------------------------------
    # GET RUN
    # --------------------------------------------------------

    def get_run(
        self,
        run_id: str,
        access: ExperimentAccessContext,
    ) -> ExperimentRun:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            return copy.deepcopy(
                self._get_run(
                    run_id,
                    access.tenant_id,
                )
            )

    # --------------------------------------------------------
    # LIST RUNS
    # --------------------------------------------------------

    def list_runs(
        self,
        experiment_id: str,
        access: ExperimentAccessContext,
        *,
        status: RunStatus | None = None,
    ) -> tuple[ExperimentRun, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            self._get_experiment(
                experiment_id,
                access.tenant_id,
            )

            runs = [
                copy.deepcopy(run)
                for run in self._runs.values()
                if (
                    run.tenant_id == access.tenant_id
                    and run.experiment_id == experiment_id
                    and (
                        status is None
                        or run.status == status
                    )
                )
            ]

            return tuple(
                sorted(
                    runs,
                    key=lambda run: run.created_at,
                )
            )

    # --------------------------------------------------------
    # START RUN
    # --------------------------------------------------------

    def start_run(
        self,
        run_id: str,
        access: ExperimentAccessContext,
    ) -> ExperimentRun:
        self._authorize(
            access,
            "can_write",
        )

        with self._lock:
            run = self._get_run(
                run_id,
                access.tenant_id,
            )

            experiment = self._get_experiment(
                run.experiment_id,
                access.tenant_id,
            )

            if experiment.status != ExperimentStatus.ACTIVE:
                raise ExperimentConflictError(
                    "Experiment is archived"
                )

            if run.status != RunStatus.CREATED:
                raise ExperimentConflictError(
                    "Run must be in created state"
                )

            run.status = RunStatus.RUNNING
            run.started_at = utc_now()

            self._touch_run(
                run
            )

            self._audit(
                tenant_id=access.tenant_id,
                experiment_id=run.experiment_id,
                run_id=run.run_id,
                action=AuditAction.RUN_STARTED,
                actor_id=access.user_id,
            )

            self._persist()

            return copy.deepcopy(run)

    # --------------------------------------------------------
    # FINISH RUN
    # --------------------------------------------------------

    def finish_run(
        self,
        run_id: str,
        access: ExperimentAccessContext,
        *,
        status: RunStatus = RunStatus.COMPLETED,
        failure_reason: str = "",
    ) -> ExperimentRun:
        self._authorize(
            access,
            "can_write",
        )

        if status not in (
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        ):
            raise ExperimentValidationError(
                "Invalid terminal run status"
            )

        with self._lock:
            run = self._get_run(
                run_id,
                access.tenant_id,
            )

            if run.status != RunStatus.RUNNING:
                raise ExperimentConflictError(
                    "Only running runs can be finished"
                )

            run.status = status
            run.ended_at = utc_now()
            run.failure_reason = failure_reason

            self._touch_run(
                run
            )

            action = {
                RunStatus.COMPLETED: AuditAction.RUN_COMPLETED,
                RunStatus.FAILED: AuditAction.RUN_FAILED,
                RunStatus.CANCELLED: AuditAction.RUN_CANCELLED,
            }[status]

            self._audit(
                tenant_id=access.tenant_id,
                experiment_id=run.experiment_id,
                run_id=run.run_id,
                action=action,
                actor_id=access.user_id,
                details={
                    "failure_reason": failure_reason,
                },
            )

            self._persist()

            return copy.deepcopy(run)

    # --------------------------------------------------------
    # LOG PARAMETERS
    # --------------------------------------------------------

    def log_parameters(
        self,
        run_id: str,
        parameters: Mapping[str, Any],
        access: ExperimentAccessContext,
    ) -> ExperimentRun:
        self._authorize(
            access,
            "can_write",
        )

        normalized = validate_json_value(
            dict(parameters)
        )

        with self._lock:
            run = self._get_run(
                run_id,
                access.tenant_id,
            )

            if run.status not in (
                RunStatus.CREATED,
                RunStatus.RUNNING,
            ):
                raise ExperimentConflictError(
                    "Cannot modify parameters of a finished run"
                )

            for name, value in normalized.items():
                validate_identifier(
                    name,
                    "parameter name",
                )

                if (
                    name in run.parameters
                    and run.parameters[name] != value
                ):
                    raise ExperimentConflictError(
                        f"Parameter is immutable: {name}"
                    )

            run.parameters.update(
                normalized
            )

            self._touch_run(
                run
            )

            self._audit(
                tenant_id=access.tenant_id,
                experiment_id=run.experiment_id,
                run_id=run.run_id,
                action=AuditAction.PARAMETERS_LOGGED,
                actor_id=access.user_id,
                details={
                    "parameter_names": list(
                        normalized
                    ),
                },
            )

            self._persist()

            return copy.deepcopy(run)

    # --------------------------------------------------------
    # LOG METRIC
    # --------------------------------------------------------

    def log_metric(
        self,
        run_id: str,
        metric_name: str,
        value: float,
        access: ExperimentAccessContext,
        *,
        step: int = 0,
        split: MetricSplit = MetricSplit.TRAIN,
    ) -> MetricPoint:
        self._authorize(
            access,
            "can_write",
        )

        validate_identifier(
            metric_name,
            "metric_name",
        )

        metric_value = validate_number(
            value,
            "metric value",
        )

        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
        ):
            raise ExperimentValidationError(
                "step must be a nonnegative integer"
            )

        with self._lock:
            run = self._get_run(
                run_id,
                access.tenant_id,
            )

            if run.status != RunStatus.RUNNING:
                raise ExperimentConflictError(
                    "Metrics can only be logged to running runs"
                )

            if (
                len(run.metrics)
                >= self.config.maximum_metrics_per_run
            ):
                raise ExperimentConflictError(
                    "Metric capacity reached"
                )

            point = MetricPoint(
                metric_name=metric_name,
                value=metric_value,
                step=step,
                split=split,
                timestamp=utc_now(),
            )

            run.metrics.append(
                point
            )

            self._touch_run(
                run
            )

            self._audit(
                tenant_id=access.tenant_id,
                experiment_id=run.experiment_id,
                run_id=run.run_id,
                action=AuditAction.METRICS_LOGGED,
                actor_id=access.user_id,
                details={
                    "metric_name": metric_name,
                    "step": step,
                    "split": split.value,
                },
            )

            self._persist()

            return point

    # --------------------------------------------------------
    # LOG METRIC BATCH
    # --------------------------------------------------------

    def log_metrics(
        self,
        run_id: str,
        metrics: Mapping[str, float],
        access: ExperimentAccessContext,
        *,
        step: int = 0,
        split: MetricSplit = MetricSplit.TRAIN,
    ) -> tuple[MetricPoint, ...]:
        self._authorize(
            access,
            "can_write",
        )

        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
        ):
            raise ExperimentValidationError(
                "step must be a nonnegative integer"
            )

        validated = {
            validate_identifier(name, "metric name"):
            validate_number(value, name)
            for name, value in metrics.items()
        }

        with self._lock:
            run = self._get_run(
                run_id,
                access.tenant_id,
            )

            if run.status != RunStatus.RUNNING:
                raise ExperimentConflictError(
                    "Run is not running"
                )

            if (
                len(run.metrics) + len(validated)
                > self.config.maximum_metrics_per_run
            ):
                raise ExperimentConflictError(
                    "Metric capacity reached"
                )

            timestamp = utc_now()

            points = tuple(
                MetricPoint(
                    metric_name=name,
                    value=value,
                    step=step,
                    split=split,
                    timestamp=timestamp,
                )
                for name, value in validated.items()
            )

            run.metrics.extend(
                points
            )

            self._touch_run(
                run
            )

            self._audit(
                tenant_id=access.tenant_id,
                experiment_id=run.experiment_id,
                run_id=run.run_id,
                action=AuditAction.METRICS_LOGGED,
                actor_id=access.user_id,
                details={
                    "metric_names": list(validated),
                    "step": step,
                    "split": split.value,
                },
            )

            self._persist()

            return points

    # --------------------------------------------------------
    # REGISTER ARTIFACT
    # --------------------------------------------------------

    def register_artifact(
        self,
        run_id: str,
        name: str,
        artifact_type: ArtifactType,
        uri: str,
        access: ExperimentAccessContext,
        *,
        verify_local: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunArtifact:
        self._authorize(
            access,
            "can_write",
        )

        validate_identifier(
            name,
            "artifact name",
        )

        validate_identifier(
            uri,
            "artifact URI",
        )

        verify = (
            self.config.verify_local_artifacts
            if verify_local is None
            else verify_local
        )

        digest = None
        size = None

        if verify:
            digest, size = (
                self.artifact_verifier.inspect_local_file(
                    uri
                )
            )

        normalized_metadata = validate_json_value(
            dict(metadata or {})
        )

        with self._lock:
            run = self._get_run(
                run_id,
                access.tenant_id,
            )

            if run.status not in (
                RunStatus.CREATED,
                RunStatus.RUNNING,
            ):
                raise ExperimentConflictError(
                    "Cannot register artifacts on a finished run"
                )

            artifact = RunArtifact(
                artifact_id=generate_id(),
                name=name,
                artifact_type=artifact_type,
                uri=uri,
                sha256=digest,
                size_bytes=size,
                registered_at=utc_now(),
                metadata=normalized_metadata,
            )

            run.artifacts.append(
                artifact
            )

            self._touch_run(
                run
            )

            self._audit(
                tenant_id=access.tenant_id,
                experiment_id=run.experiment_id,
                run_id=run.run_id,
                action=AuditAction.ARTIFACT_REGISTERED,
                actor_id=access.user_id,
                details={
                    "artifact_id": artifact.artifact_id,
                    "artifact_name": name,
                },
            )

            self._persist()

            return artifact

    # --------------------------------------------------------
    # METRIC HISTORY
    # --------------------------------------------------------

    def get_metric_history(
        self,
        run_id: str,
        metric_name: str,
        split: MetricSplit,
        access: ExperimentAccessContext,
    ) -> tuple[MetricPoint, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            run = self._get_run(
                run_id,
                access.tenant_id,
            )

            return self.metric_analyzer.metric_history(
                run,
                metric_name,
                split,
            )

    # --------------------------------------------------------
    # COMPARE RUNS
    # --------------------------------------------------------

    def compare_runs(
        self,
        experiment_id: str,
        access: ExperimentAccessContext,
        *,
        run_ids: Sequence[str] | None = None,
    ) -> ExperimentComparison:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            self._get_experiment(
                experiment_id,
                access.tenant_id,
            )

            requested_ids = (
                set(run_ids)
                if run_ids is not None
                else None
            )

            selected = [
                run
                for run in self._runs.values()
                if (
                    run.tenant_id == access.tenant_id
                    and run.experiment_id == experiment_id
                    and (
                        requested_ids is None
                        or run.run_id in requested_ids
                    )
                )
            ]

            if requested_ids is not None:
                found_ids = {
                    run.run_id
                    for run in selected
                }

                if found_ids != requested_ids:
                    raise RunNotFoundError(
                        "One or more runs do not belong to this experiment"
                    )

            comparisons = []

            for run in selected:
                duration = None

                if (
                    run.started_at is not None
                    and run.ended_at is not None
                ):
                    duration = (
                        run.ended_at - run.started_at
                    ).total_seconds()

                comparisons.append(
                    RunComparison(
                        run_id=run.run_id,
                        run_name=run.run_name,
                        status=run.status,
                        parameters=copy.deepcopy(
                            run.parameters
                        ),
                        latest_metrics=run.latest_metrics(),
                        duration_seconds=duration,
                    )
                )

            return ExperimentComparison(
                experiment_id=experiment_id,
                run_count=len(comparisons),
                runs=tuple(comparisons),
            )

    # --------------------------------------------------------
    # SELECT BEST RUN
    # --------------------------------------------------------

    def best_run(
        self,
        experiment_id: str,
        metric_name: str,
        split: MetricSplit,
        direction: MetricDirection,
        access: ExperimentAccessContext,
        *,
        completed_only: bool = True,
    ) -> BestRunResult | None:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            self._get_experiment(
                experiment_id,
                access.tenant_id,
            )

            candidates = []

            for run in self._runs.values():
                if (
                    run.tenant_id != access.tenant_id
                    or run.experiment_id != experiment_id
                ):
                    continue

                if (
                    completed_only
                    and run.status != RunStatus.COMPLETED
                ):
                    continue

                value = (
                    self.metric_analyzer.latest_value(
                        run,
                        metric_name,
                        split,
                    )
                )

                if value is not None:
                    candidates.append(
                        (
                            value,
                            run,
                        )
                    )

            if not candidates:
                return None

            selector = (
                max
                if direction == MetricDirection.MAXIMIZE
                else min
            )

            value, run = selector(
                candidates,
                key=lambda item: item[0],
            )

            return BestRunResult(
                experiment_id=experiment_id,
                run_id=run.run_id,
                metric_name=metric_name,
                split=split,
                direction=direction,
                metric_value=value,
            )

    # --------------------------------------------------------
    # HYPERPARAMETER SUMMARY
    # --------------------------------------------------------

    def hyperparameter_summary(
        self,
        experiment_id: str,
        access: ExperimentAccessContext,
    ) -> tuple[HyperparameterSummary, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            self._get_experiment(
                experiment_id,
                access.tenant_id,
            )

            parameters: dict[
                str,
                list[Any],
            ] = {}

            for run in self._runs.values():
                if (
                    run.tenant_id != access.tenant_id
                    or run.experiment_id != experiment_id
                ):
                    continue

                for name, value in run.parameters.items():
                    parameters.setdefault(
                        name,
                        [],
                    ).append(
                        copy.deepcopy(value)
                    )

            results = []

            for name, values in sorted(
                parameters.items()
            ):
                unique = {}

                for value in values:
                    key = json.dumps(
                        value,
                        sort_keys=True,
                        allow_nan=False,
                    )

                    unique[key] = value

                results.append(
                    HyperparameterSummary(
                        parameter_name=name,
                        values=tuple(
                            unique.values()
                        ),
                        run_count=len(values),
                    )
                )

            return tuple(results)

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        experiment_id: str,
        access: ExperimentAccessContext,
        *,
        run_id: str | None = None,
    ) -> tuple[ExperimentAuditEvent, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            self._get_experiment(
                experiment_id,
                access.tenant_id,
            )

            if run_id is not None:
                run = self._get_run(
                    run_id,
                    access.tenant_id,
                )

                if run.experiment_id != experiment_id:
                    raise RunNotFoundError(
                        "Run does not belong to experiment"
                    )

            return tuple(
                copy.deepcopy(event)
                for event in self._events
                if (
                    event.tenant_id == access.tenant_id
                    and event.experiment_id == experiment_id
                    and (
                        run_id is None
                        or event.run_id == run_id
                    )
                )
            )

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_experiment_json(
        self,
        experiment_id: str,
        access: ExperimentAccessContext,
    ) -> str:
        self._authorize(
            access,
            "can_read",
        )

        experiment = self.get_experiment(
            experiment_id,
            access,
        )

        runs = self.list_runs(
            experiment_id,
            access,
        )

        return json.dumps(
            json_safe(
                {
                    "experiment": experiment,
                    "runs": runs,
                }
            ),
            indent=2,
            allow_nan=False,
        )

    # --------------------------------------------------------
    # EXPORT METRICS CSV
    # --------------------------------------------------------

    def export_metrics_csv(
        self,
        run_id: str,
        access: ExperimentAccessContext,
    ) -> str:
        run = self.get_run(
            run_id,
            access,
        )

        output = io.StringIO()

        writer = csv.writer(
            output
        )

        writer.writerow(
            [
                "run_id",
                "metric_name",
                "value",
                "step",
                "split",
                "timestamp",
            ]
        )

        for point in sorted(
            run.metrics,
            key=lambda item: (
                item.step,
                item.timestamp,
            ),
        ):
            writer.writerow(
                [
                    run.run_id,
                    point.metric_name,
                    point.value,
                    point.step,
                    point.split.value,
                    point.timestamp.isoformat(),
                ]
            )

        return output.getvalue()


# ============================================================
# SYNTHETIC ROBOTICS EXPERIMENT FACTORY
# ============================================================

class SyntheticExperimentFactory:

    @staticmethod
    def training_metrics(
        epochs: int = 10,
        *,
        learning_rate: float = 0.001,
    ) -> list[dict[str, float]]:
        if epochs <= 0:
            raise ExperimentValidationError(
                "epochs must be positive"
            )

        validate_number(
            learning_rate,
            "learning_rate",
            minimum=0.000000001,
        )

        results = []

        for epoch in range(epochs):
            progress = epoch + 1

            train_loss = (
                1.5 / (1 + progress * 0.45)
            )

            validation_loss = (
                1.7 / (1 + progress * 0.36)
            )

            accuracy = min(
                0.99,
                0.55 + progress * 0.035,
            )

            results.append(
                {
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "accuracy": accuracy,
                    "learning_rate": learning_rate,
                }
            )

        return results

    @staticmethod
    def robotics_metrics(
        episodes: int = 10,
    ) -> list[dict[str, float]]:
        if episodes <= 0:
            raise ExperimentValidationError(
                "episodes must be positive"
            )

        return [
            {
                "episode_reward": (
                    20.0 + episode * 8.0
                ),
                "success_rate": min(
                    1.0,
                    0.3 + episode * 0.06,
                ),
                "collision_rate": max(
                    0.0,
                    0.35 - episode * 0.025,
                ),
                "path_efficiency": min(
                    1.0,
                    0.5 + episode * 0.04,
                ),
            }
            for episode in range(episodes)
        ]


# ============================================================
# UNIT TESTS
# ============================================================

class TestExperimentTracker(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

        self.root = Path(
            self.temporary.name
        )

        self.config = ExperimentTrackerConfig(
            storage_root=self.root / "registry",
        )

        self.tracker = ExperimentTracker(
            self.config
        )

        self.access = ExperimentAccessContext(
            tenant_id="robotics_lab",
            user_id="ml_engineer",
            can_read=True,
            can_write=True,
            can_manage=True,
        )

        self.experiment = (
            self.tracker.create_experiment(
                name="robot-vision-training",
                experiment_type=ExperimentType.COMPUTER_VISION,
                access=self.access,
                description="Robot perception training experiments",
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create_run(
        self,
        name: str = "run-1",
    ) -> ExperimentRun:
        return self.tracker.create_run(
            experiment_id=self.experiment.experiment_id,
            run_name=name,
            access=self.access,
            parameters={
                "learning_rate": 0.001,
                "batch_size": 32,
            },
        )

    def test_create_experiment(self) -> None:
        self.assertEqual(
            self.experiment.status,
            ExperimentStatus.ACTIVE,
        )

    def test_get_experiment(self) -> None:
        result = self.tracker.get_experiment(
            self.experiment.experiment_id,
            self.access,
        )

        self.assertEqual(
            result.name,
            "robot-vision-training",
        )

    def test_create_run(self) -> None:
        run = self._create_run()

        self.assertEqual(
            run.status,
            RunStatus.CREATED,
        )

    def test_start_run(self) -> None:
        run = self._create_run()

        started = self.tracker.start_run(
            run.run_id,
            self.access,
        )

        self.assertEqual(
            started.status,
            RunStatus.RUNNING,
        )

    def test_complete_run(self) -> None:
        run = self._create_run()

        self.tracker.start_run(
            run.run_id,
            self.access,
        )

        completed = self.tracker.finish_run(
            run.run_id,
            self.access,
        )

        self.assertEqual(
            completed.status,
            RunStatus.COMPLETED,
        )

    def test_log_metric(self) -> None:
        run = self._create_run()

        self.tracker.start_run(
            run.run_id,
            self.access,
        )

        point = self.tracker.log_metric(
            run.run_id,
            "loss",
            0.25,
            self.access,
            step=1,
        )

        self.assertEqual(
            point.value,
            0.25,
        )

    def test_log_metric_batch(self) -> None:
        run = self._create_run()

        self.tracker.start_run(
            run.run_id,
            self.access,
        )

        points = self.tracker.log_metrics(
            run.run_id,
            {
                "accuracy": 0.95,
                "f1_score": 0.92,
            },
            self.access,
            step=1,
            split=MetricSplit.VALIDATION,
        )

        self.assertEqual(
            len(points),
            2,
        )

    def test_parameter_immutability(self) -> None:
        run = self._create_run()

        with self.assertRaises(
            ExperimentConflictError
        ):
            self.tracker.log_parameters(
                run.run_id,
                {
                    "learning_rate": 0.1,
                },
                self.access,
            )

    def test_metric_history(self) -> None:
        run = self._create_run()

        self.tracker.start_run(
            run.run_id,
            self.access,
        )

        for step in range(5):
            self.tracker.log_metric(
                run.run_id,
                "loss",
                1.0 / (step + 1),
                self.access,
                step=step,
            )

        history = self.tracker.get_metric_history(
            run.run_id,
            "loss",
            MetricSplit.TRAIN,
            self.access,
        )

        self.assertEqual(
            len(history),
            5,
        )

    def test_best_run(self) -> None:
        for index, accuracy in enumerate(
            [0.80, 0.95, 0.88]
        ):
            run = self._create_run(
                f"run-{index}"
            )

            self.tracker.start_run(
                run.run_id,
                self.access,
            )

            self.tracker.log_metric(
                run.run_id,
                "accuracy",
                accuracy,
                self.access,
                step=1,
                split=MetricSplit.VALIDATION,
            )

            self.tracker.finish_run(
                run.run_id,
                self.access,
            )

        best = self.tracker.best_run(
            self.experiment.experiment_id,
            "accuracy",
            MetricSplit.VALIDATION,
            MetricDirection.MAXIMIZE,
            self.access,
        )

        self.assertIsNotNone(
            best
        )

        self.assertEqual(
            best.metric_value,
            0.95,
        )

    def test_compare_runs(self) -> None:
        self._create_run("run-a")
        self._create_run("run-b")

        comparison = self.tracker.compare_runs(
            self.experiment.experiment_id,
            self.access,
        )

        self.assertEqual(
            comparison.run_count,
            2,
        )

    def test_artifact_registration(self) -> None:
        run = self._create_run()

        artifact_path = (
            self.root / "model.bin"
        )

        artifact_path.write_bytes(
            b"synthetic-model-weights"
        )

        artifact = self.tracker.register_artifact(
            run.run_id,
            "model-checkpoint",
            ArtifactType.CHECKPOINT,
            str(artifact_path),
            self.access,
        )

        self.assertTrue(
            self.tracker.artifact_verifier.verify(
                artifact
            )
        )

    def test_artifact_integrity_failure(self) -> None:
        run = self._create_run()

        artifact_path = (
            self.root / "model.bin"
        )

        artifact_path.write_bytes(
            b"original"
        )

        artifact = self.tracker.register_artifact(
            run.run_id,
            "model",
            ArtifactType.MODEL,
            str(artifact_path),
            self.access,
        )

        artifact_path.write_bytes(
            b"modified"
        )

        with self.assertRaises(
            ArtifactIntegrityError
        ):
            self.tracker.artifact_verifier.verify(
                artifact
            )

    def test_persistence(self) -> None:
        run = self._create_run()

        reloaded = ExperimentTracker(
            self.config
        )

        restored = reloaded.get_run(
            run.run_id,
            self.access,
        )

        self.assertEqual(
            restored.run_id,
            run.run_id,
        )

    def test_tenant_isolation(self) -> None:
        other = ExperimentAccessContext(
            tenant_id="another_tenant",
            user_id="other_user",
            can_read=True,
        )

        with self.assertRaises(
            ExperimentNotFoundError
        ):
            self.tracker.get_experiment(
                self.experiment.experiment_id,
                other,
            )

    def test_archive_experiment(self) -> None:
        archived = self.tracker.archive_experiment(
            self.experiment.experiment_id,
            self.access,
        )

        self.assertEqual(
            archived.status,
            ExperimentStatus.ARCHIVED,
        )

    def test_csv_export(self) -> None:
        run = self._create_run()

        self.tracker.start_run(
            run.run_id,
            self.access,
        )

        self.tracker.log_metric(
            run.run_id,
            "accuracy",
            0.95,
            self.access,
        )

        csv_text = self.tracker.export_metrics_csv(
            run.run_id,
            self.access,
        )

        self.assertIn(
            "accuracy",
            csv_text,
        )

    def test_json_export(self) -> None:
        exported = self.tracker.export_experiment_json(
            self.experiment.experiment_id,
            self.access,
        )

        payload = json.loads(
            exported
        )

        self.assertEqual(
            payload["experiment"]["name"],
            "robot-vision-training",
        )

    def test_metric_moving_average(self) -> None:
        points = [
            MetricPoint(
                metric_name="loss",
                value=float(value),
                step=index,
                split=MetricSplit.TRAIN,
                timestamp=utc_now(),
            )
            for index, value in enumerate(
                [1, 2, 3, 4, 5]
            )
        ]

        result = (
            ExperimentMetricAnalyzer.moving_average(
                points,
                window_size=3,
            )
        )

        self.assertEqual(
            result[-1][1],
            4.0,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        tracker = ExperimentTracker(
            ExperimentTrackerConfig(
                storage_root=root / "registry",
            )
        )

        access = ExperimentAccessContext(
            tenant_id="physical_ai_lab",
            user_id="mlops_engineer",
            can_read=True,
            can_write=True,
            can_manage=True,
        )

        experiment = tracker.create_experiment(
            name="autonomous-robot-navigation-rl",
            experiment_type=ExperimentType.REINFORCEMENT_LEARNING,
            access=access,
            description=(
                "Synthetic reinforcement learning "
                "experiment for autonomous navigation"
            ),
            tags={
                "project": "physical_ai",
                "algorithm": "ppo",
            },
            metadata={
                "environment": "warehouse_simulation",
                "robot_type": "autonomous_mobile_robot",
            },
        )

        for index, learning_rate in enumerate(
            [0.001, 0.0005, 0.0001]
        ):
            run = tracker.create_run(
                experiment_id=experiment.experiment_id,
                run_name=f"ppo-training-{index + 1}",
                access=access,
                parameters={
                    "learning_rate": learning_rate,
                    "batch_size": 64,
                    "gamma": 0.99,
                    "clip_epsilon": 0.2,
                    "epochs": 10,
                },
                reproducibility=ReproducibilityMetadata(
                    python_version="3.11",
                    framework_name="pytorch",
                    framework_version="2.x",
                    random_seed=42 + index,
                    dataset_id="robot-simulation-dataset",
                    dataset_version_id="v1",
                    hardware_name="synthetic-training-node",
                    gpu_count=1,
                ),
            )

            tracker.start_run(
                run.run_id,
                access,
            )

            training_data = (
                SyntheticExperimentFactory.robotics_metrics(
                    episodes=10
                )
            )

            for episode, metrics in enumerate(
                training_data
            ):
                adjusted = {
                    name: (
                        value + index * 0.02
                        if name in (
                            "success_rate",
                            "path_efficiency",
                        )
                        else value
                    )
                    for name, value in metrics.items()
                }

                tracker.log_metrics(
                    run.run_id,
                    adjusted,
                    access,
                    step=episode,
                    split=MetricSplit.SIMULATION,
                )

            tracker.finish_run(
                run.run_id,
                access,
            )

        comparison = tracker.compare_runs(
            experiment.experiment_id,
            access,
        )

        best = tracker.best_run(
            experiment.experiment_id,
            "success_rate",
            MetricSplit.SIMULATION,
            MetricDirection.MAXIMIZE,
            access,
        )

        summary = tracker.hyperparameter_summary(
            experiment.experiment_id,
            access,
        )

        print("\nPHYSICAL AI EXPERIMENT TRACKER")
        print("=" * 70)

        print(
            json.dumps(
                json_safe(
                    {
                        "experiment_id": experiment.experiment_id,
                        "experiment_name": experiment.name,
                        "experiment_type": experiment.experiment_type,
                        "run_count": comparison.run_count,
                        "best_run": best,
                        "hyperparameters": summary,
                        "runs": comparison.runs,
                        "audit_event_count": len(
                            tracker.audit_history(
                                experiment.experiment_id,
                                access,
                            )
                        ),
                        "training_executed": False,
                        "synthetic_metrics": True,
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
        TestExperimentTracker
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
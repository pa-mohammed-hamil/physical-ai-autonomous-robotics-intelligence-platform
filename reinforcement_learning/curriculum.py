"""Curriculum module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/curriculum.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Curriculum Learning and Adaptive Training Engine

Python 3.11+
Dependencies: Standard library only

Features:
    - Curriculum learning for robotics and reinforcement learning
    - Progressive task difficulty
    - Adaptive training stages
    - Simulation scenario scheduling
    - Task prerequisite management
    - Success-rate and reward evaluation
    - Training performance tracking
    - Automatic difficulty recommendations
    - Domain randomization configuration
    - Sim-to-real readiness assessment
    - Training checkpoint references
    - Regression detection
    - Safety and failure-rate constraints
    - Human-reviewed stage promotion
    - Multi-tenant isolation
    - Thread-safe in-memory storage
    - Audit logging and JSON export

Integrations:
    action_space.py
    checkpoints.py
    simulation_adapter.py
    scenario_manager.py
    environment_generator.py
    domain_randomization.py
    reinforcement_learning.py
    robot_state.py
    safety_monitor.py
    evaluation_service.py
    model_registry.py

Safety:
    Training progression is simulation-only.

    This module does not execute robot commands,
    deploy trained policies to physical robots,
    or authorize real-world robot operation.

    Passing curriculum stages does not establish
    physical robot safety or sim-to-real readiness.
"""

from __future__ import annotations

import copy
import json
import math
import statistics
import threading
import uuid

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class CurriculumError(Exception):
    pass


class CurriculumValidationError(CurriculumError):
    pass


class CurriculumNotFound(CurriculumError):
    pass


class CurriculumConflict(CurriculumError):
    pass


class CurriculumPermissionDenied(CurriculumError):
    pass


# ============================================================
# ENUMS
# ============================================================

class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    EXPERT = "expert"


class CurriculumStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    LOCKED = "locked"
    AVAILABLE = "available"
    TRAINING = "training"
    READY_FOR_REVIEW = "ready_for_review"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingTaskType(str, Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    LOCOMOTION = "locomotion"
    GRASPING = "grasping"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    OBJECT_TRACKING = "object_tracking"
    PERCEPTION = "perception"
    MULTI_ROBOT = "multi_robot"
    HUMAN_ROBOT_INTERACTION = "human_robot_interaction"
    CUSTOM = "custom"


class AdaptationStrategy(str, Enum):
    FIXED = "fixed"
    PERFORMANCE_BASED = "performance_based"
    MASTERY_BASED = "mastery_based"


class CurriculumRecommendation(str, Enum):
    CONTINUE = "continue"
    PROMOTE = "promote"
    REPEAT = "repeat"
    REGRESS = "regress"
    REVIEW = "review"
    INSUFFICIENT_DATA = "insufficient_data"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            char.isalnum() or char in "._-"
            for char in value
        )
    ):
        raise CurriculumValidationError(
            f"Invalid {name}"
        )

    return value


def validate_text(
    value: str,
    name: str,
    *,
    allow_empty: bool = False,
    maximum: int = 4000,
) -> str:
    if not isinstance(value, str):
        raise CurriculumValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if not allow_empty and not result:
        raise CurriculumValidationError(
            f"{name} cannot be empty"
        )

    if len(result) > maximum:
        raise CurriculumValidationError(
            f"{name} exceeds maximum length"
        )

    return result


def validate_integer(
    value: int,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 10000000,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise CurriculumValidationError(
            f"Invalid {name}"
        )

    return value


def finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise CurriculumValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise CurriculumValidationError(
            f"Invalid {name}"
        ) from exc

    if not math.isfinite(result):
        raise CurriculumValidationError(
            f"{name} must be finite"
        )

    return result


def probability(value: Any, name: str) -> float:
    result = finite_number(value, name)

    if not 0 <= result <= 1:
        raise CurriculumValidationError(
            f"{name} must be between 0 and 1"
        )

    return result


def validate_datetime(
    value: datetime,
    name: str,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise CurriculumValidationError(
            f"{name} must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list, set, frozenset)):
        return [json_safe(item) for item in value]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise CurriculumValidationError(
                "Non-finite JSON value"
            )

    return value


def validate_json_mapping(
    value: Mapping[str, Any],
    name: str,
    *,
    maximum_bytes: int = 1048576,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CurriculumValidationError(
            f"{name} must be a mapping"
        )

    if not all(isinstance(k, str) for k in value):
        raise CurriculumValidationError(
            f"{name} keys must be strings"
        )

    try:
        result = copy.deepcopy(dict(value))
        encoded = json.dumps(
            result,
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise CurriculumValidationError(
            f"{name} must contain JSON-compatible data"
        ) from exc

    if len(encoded.encode("utf-8")) > maximum_bytes:
        raise CurriculumValidationError(
            f"{name} exceeds maximum size"
        )

    return result


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CurriculumConfig:
    maximum_curricula: int = 10000
    maximum_stages_per_curriculum: int = 100
    maximum_episodes_per_stage: int = 100000
    maximum_audit_events: int = 20000

    evaluation_window: int = 20
    minimum_evaluation_episodes: int = 10

    default_success_threshold: float = 0.85
    default_maximum_failure_rate: float = 0.20
    default_maximum_safety_violation_rate: float = 0.0

    regression_success_threshold: float = 0.40

    simulation_only: bool = True
    human_approval_required: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_curricula",
            "maximum_stages_per_curriculum",
            "maximum_episodes_per_stage",
            "maximum_audit_events",
            "evaluation_window",
            "minimum_evaluation_episodes",
        ):
            validate_integer(
                getattr(self, name),
                name,
                minimum=1,
            )

        if (
            self.minimum_evaluation_episodes
            > self.evaluation_window
        ):
            raise CurriculumValidationError(
                "Minimum evaluation episodes exceed window"
            )

        for name in (
            "default_success_threshold",
            "default_maximum_failure_rate",
            "default_maximum_safety_violation_rate",
            "regression_success_threshold",
        ):
            probability(
                getattr(self, name),
                name,
            )

        if self.simulation_only is not True:
            raise CurriculumValidationError(
                "Only simulation training is supported"
            )

        if self.human_approval_required is not True:
            raise CurriculumValidationError(
                "Stage promotion must require human approval"
            )


# ============================================================
# DOMAIN RANDOMIZATION
# ============================================================

@dataclass(frozen=True)
class RandomizationParameter:
    name: str
    minimum: float
    maximum: float
    enabled: bool = True

    def __post_init__(self) -> None:
        validate_id(self.name, "name")

        minimum = finite_number(
            self.minimum,
            "minimum",
        )

        maximum = finite_number(
            self.maximum,
            "maximum",
        )

        if minimum > maximum:
            raise CurriculumValidationError(
                "Randomization minimum exceeds maximum"
            )

        if not isinstance(self.enabled, bool):
            raise CurriculumValidationError(
                "enabled must be boolean"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class DomainRandomizationProfile:
    parameters: tuple[RandomizationParameter, ...] = ()
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, tuple):
            raise CurriculumValidationError(
                "parameters must be a tuple"
            )

        names = set()

        for parameter in self.parameters:
            if not isinstance(
                parameter,
                RandomizationParameter,
            ):
                raise CurriculumValidationError(
                    "Invalid randomization parameter"
                )

            if parameter.name in names:
                raise CurriculumValidationError(
                    "Duplicate randomization parameter"
                )

            names.add(parameter.name)

        validate_integer(
            self.seed,
            "seed",
            maximum=2**32 - 1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": [
                parameter.to_dict()
                for parameter in self.parameters
            ],
            "seed": self.seed,
        }


# ============================================================
# CURRICULUM STAGE
# ============================================================

@dataclass(frozen=True)
class CurriculumStage:
    stage_id: str
    name: str

    stage_order: int
    difficulty: DifficultyLevel

    task_type: TrainingTaskType

    scenario_ids: tuple[str, ...] = ()

    prerequisites: tuple[str, ...] = ()

    target_success_rate: float = 0.85
    minimum_episodes: int = 20

    maximum_failure_rate: float = 0.20
    maximum_safety_violation_rate: float = 0.0

    minimum_mean_reward: float | None = None

    maximum_episode_steps: int = 1000

    randomization: DomainRandomizationProfile = field(
        default_factory=DomainRandomizationProfile
    )

    environment_parameters: Mapping[str, Any] = field(
        default_factory=dict
    )

    description: str = ""

    def __post_init__(self) -> None:
        validate_id(self.stage_id, "stage_id")
        validate_text(self.name, "name")

        validate_integer(
            self.stage_order,
            "stage_order",
            minimum=1,
        )

        if not isinstance(
            self.difficulty,
            DifficultyLevel,
        ):
            raise CurriculumValidationError(
                "Invalid difficulty"
            )

        if not isinstance(
            self.task_type,
            TrainingTaskType,
        ):
            raise CurriculumValidationError(
                "Invalid task type"
            )

        for name in (
            "scenario_ids",
            "prerequisites",
        ):
            values = getattr(self, name)

            if not isinstance(values, tuple):
                raise CurriculumValidationError(
                    f"{name} must be a tuple"
                )

            if len(values) != len(set(values)):
                raise CurriculumValidationError(
                    f"Duplicate values in {name}"
                )

            for item in values:
                validate_id(item, name)

        if self.stage_id in self.prerequisites:
            raise CurriculumValidationError(
                "Stage cannot depend on itself"
            )

        for name in (
            "target_success_rate",
            "maximum_failure_rate",
            "maximum_safety_violation_rate",
        ):
            probability(
                getattr(self, name),
                name,
            )

        validate_integer(
            self.minimum_episodes,
            "minimum_episodes",
            minimum=1,
        )

        validate_integer(
            self.maximum_episode_steps,
            "maximum_episode_steps",
            minimum=1,
        )

        if self.minimum_mean_reward is not None:
            finite_number(
                self.minimum_mean_reward,
                "minimum_mean_reward",
            )

        if not isinstance(
            self.randomization,
            DomainRandomizationProfile,
        ):
            raise CurriculumValidationError(
                "Invalid domain randomization profile"
            )

        validate_json_mapping(
            self.environment_parameters,
            "environment_parameters",
        )

        validate_text(
            self.description,
            "description",
            allow_empty=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "stage_order": self.stage_order,
            "difficulty": self.difficulty.value,
            "task_type": self.task_type.value,
            "scenario_ids": list(self.scenario_ids),
            "prerequisites": list(self.prerequisites),
            "target_success_rate": self.target_success_rate,
            "minimum_episodes": self.minimum_episodes,
            "maximum_failure_rate": self.maximum_failure_rate,
            "maximum_safety_violation_rate": (
                self.maximum_safety_violation_rate
            ),
            "minimum_mean_reward": self.minimum_mean_reward,
            "maximum_episode_steps": self.maximum_episode_steps,
            "randomization": self.randomization.to_dict(),
            "environment_parameters": copy.deepcopy(
                dict(self.environment_parameters)
            ),
            "description": self.description,
        }


# ============================================================
# CURRICULUM DEFINITION
# ============================================================

@dataclass(frozen=True)
class RobotCurriculum:
    curriculum_id: str

    tenant_id: str
    robot_id: str
    model_id: str

    name: str

    stages: tuple[CurriculumStage, ...]

    strategy: AdaptationStrategy = (
        AdaptationStrategy.PERFORMANCE_BASED
    )

    version: int = 1

    description: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "curriculum_id",
            "tenant_id",
            "robot_id",
            "model_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        validate_text(self.name, "name")

        if not isinstance(self.stages, tuple):
            raise CurriculumValidationError(
                "stages must be a tuple"
            )

        if not self.stages:
            raise CurriculumValidationError(
                "Curriculum must contain stages"
            )

        if not isinstance(
            self.strategy,
            AdaptationStrategy,
        ):
            raise CurriculumValidationError(
                "Invalid adaptation strategy"
            )

        stage_ids = set()
        stage_orders = set()

        for stage in self.stages:
            if not isinstance(stage, CurriculumStage):
                raise CurriculumValidationError(
                    "Invalid curriculum stage"
                )

            if stage.stage_id in stage_ids:
                raise CurriculumValidationError(
                    "Duplicate stage ID"
                )

            if stage.stage_order in stage_orders:
                raise CurriculumValidationError(
                    "Duplicate stage order"
                )

            stage_ids.add(stage.stage_id)
            stage_orders.add(stage.stage_order)

        stage_by_id = {
            stage.stage_id: stage
            for stage in self.stages
        }

        for stage in self.stages:
            for dependency_id in stage.prerequisites:
                dependency = stage_by_id.get(
                    dependency_id
                )

                if dependency is None:
                    raise CurriculumValidationError(
                        f"Missing prerequisite: {dependency_id}"
                    )

                if (
                    dependency.stage_order
                    >= stage.stage_order
                ):
                    raise CurriculumValidationError(
                        "Prerequisites must precede "
                        "their dependent stage"
                    )

        validate_integer(
            self.version,
            "version",
            minimum=1,
        )

        validate_text(
            self.description,
            "description",
            allow_empty=True,
        )

        validate_datetime(
            self.created_at,
            "created_at",
        )

    def ordered_stages(
        self,
    ) -> tuple[CurriculumStage, ...]:
        return tuple(
            sorted(
                self.stages,
                key=lambda stage: stage.stage_order,
            )
        )

    def get_stage(
        self,
        stage_id: str,
    ) -> CurriculumStage:
        validate_id(stage_id, "stage_id")

        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage

        raise CurriculumNotFound(
            f"Stage not found: {stage_id}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "curriculum_id": self.curriculum_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "model_id": self.model_id,
            "name": self.name,
            "strategy": self.strategy.value,
            "version": self.version,
            "description": self.description,
            "stages": [
                stage.to_dict()
                for stage in self.ordered_stages()
            ],
            "created_at": self.created_at.isoformat(),
            "simulation_only": True,
        }


# ============================================================
# TRAINING EPISODE
# ============================================================

@dataclass(frozen=True)
class TrainingEpisode:
    episode_id: str

    tenant_id: str
    curriculum_id: str
    stage_id: str

    success: bool
    reward: float
    episode_steps: int

    failure: bool = False
    safety_violations: int = 0

    scenario_id: str = ""

    checkpoint_reference: str = ""

    metrics: Mapping[str, Any] = field(
        default_factory=dict
    )

    recorded_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "tenant_id",
            "curriculum_id",
            "stage_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if not isinstance(self.success, bool):
            raise CurriculumValidationError(
                "success must be boolean"
            )

        if not isinstance(self.failure, bool):
            raise CurriculumValidationError(
                "failure must be boolean"
            )

        if self.success and self.failure:
            raise CurriculumValidationError(
                "Episode cannot be both successful and failed"
            )

        finite_number(self.reward, "reward")

        validate_integer(
            self.episode_steps,
            "episode_steps",
            minimum=1,
        )

        validate_integer(
            self.safety_violations,
            "safety_violations",
        )

        validate_text(
            self.scenario_id,
            "scenario_id",
            allow_empty=True,
            maximum=128,
        )

        validate_text(
            self.checkpoint_reference,
            "checkpoint_reference",
            allow_empty=True,
            maximum=512,
        )

        validate_json_mapping(
            self.metrics,
            "metrics",
        )

        validate_datetime(
            self.recorded_at,
            "recorded_at",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "tenant_id": self.tenant_id,
            "curriculum_id": self.curriculum_id,
            "stage_id": self.stage_id,
            "success": self.success,
            "reward": self.reward,
            "episode_steps": self.episode_steps,
            "failure": self.failure,
            "safety_violations": self.safety_violations,
            "scenario_id": self.scenario_id,
            "checkpoint_reference": (
                self.checkpoint_reference
            ),
            "metrics": copy.deepcopy(
                dict(self.metrics)
            ),
            "recorded_at": self.recorded_at.isoformat(),
        }


# ============================================================
# STAGE PERFORMANCE
# ============================================================

@dataclass(frozen=True)
class StagePerformance:
    curriculum_id: str
    stage_id: str

    episode_count: int
    evaluation_episode_count: int

    success_rate: float
    failure_rate: float
    safety_violation_rate: float

    mean_reward: float
    reward_std: float

    mean_episode_steps: float

    target_success_rate: float

    minimum_episodes_met: bool

    mastery_criteria_met: bool

    recommendation: CurriculumRecommendation

    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "curriculum_id": self.curriculum_id,
            "stage_id": self.stage_id,
            "episode_count": self.episode_count,
            "evaluation_episode_count": (
                self.evaluation_episode_count
            ),
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "safety_violation_rate": (
                self.safety_violation_rate
            ),
            "mean_reward": self.mean_reward,
            "reward_std": self.reward_std,
            "mean_episode_steps": (
                self.mean_episode_steps
            ),
            "target_success_rate": (
                self.target_success_rate
            ),
            "minimum_episodes_met": (
                self.minimum_episodes_met
            ),
            "mastery_criteria_met": (
                self.mastery_criteria_met
            ),
            "recommendation": self.recommendation.value,
            "reasons": list(self.reasons),
            "physical_deployment_approved": False,
        }


# ============================================================
# RUNTIME STATE
# ============================================================

@dataclass
class CurriculumRuntime:
    curriculum: RobotCurriculum

    status: CurriculumStatus = CurriculumStatus.DRAFT

    current_stage_id: str | None = None

    stage_statuses: dict[str, StageStatus] = field(
        default_factory=dict
    )

    episodes: dict[
        str,
        list[TrainingEpisode],
    ] = field(default_factory=dict)

    episode_ids: set[str] = field(
        default_factory=set
    )

    reviewed_by: dict[str, str] = field(
        default_factory=dict
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "curriculum": self.curriculum.to_dict(),
            "status": self.status.value,
            "current_stage_id": self.current_stage_id,
            "stage_statuses": {
                key: value.value
                for key, value in self.stage_statuses.items()
            },
            "episode_counts": {
                key: len(value)
                for key, value in self.episodes.items()
            },
            "reviewed_by": dict(self.reviewed_by),
            "updated_at": self.updated_at.isoformat(),
        }


# ============================================================
# CURRICULUM LEARNING ENGINE
# ============================================================

class RoboticsCurriculumEngine:

    def __init__(
        self,
        config: CurriculumConfig | None = None,
    ) -> None:
        self.config = config or CurriculumConfig()

        self._curricula: dict[
            tuple[str, str],
            CurriculumRuntime,
        ] = {}

        self._audit: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _log(
        self,
        *,
        tenant_id: str,
        curriculum_id: str,
        action: str,
        actor_id: str = "system",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._audit.append(
            {
                "event_id": new_id(),
                "timestamp": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "curriculum_id": curriculum_id,
                "action": action,
                "actor_id": actor_id,
                "details": json_safe(
                    dict(details or {})
                ),
            }
        )

    def _runtime(
        self,
        tenant_id: str,
        curriculum_id: str,
    ) -> CurriculumRuntime:
        validate_id(tenant_id, "tenant_id")
        validate_id(curriculum_id, "curriculum_id")

        runtime = self._curricula.get(
            (tenant_id, curriculum_id)
        )

        if runtime is None:
            raise CurriculumNotFound(
                "Curriculum not found"
            )

        return runtime

    # --------------------------------------------------------
    # REGISTER CURRICULUM
    # --------------------------------------------------------

    def register_curriculum(
        self,
        curriculum: RobotCurriculum,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise CurriculumPermissionDenied(
                "Authorized curriculum registration required"
            )

        if not isinstance(
            curriculum,
            RobotCurriculum,
        ):
            raise CurriculumValidationError(
                "Invalid curriculum"
            )

        validate_id(actor_id, "actor_id")

        if (
            len(curriculum.stages)
            > self.config.maximum_stages_per_curriculum
        ):
            raise CurriculumValidationError(
                "Too many curriculum stages"
            )

        key = (
            curriculum.tenant_id,
            curriculum.curriculum_id,
        )

        with self._lock:
            if key in self._curricula:
                raise CurriculumConflict(
                    "Curriculum already exists"
                )

            if (
                len(self._curricula)
                >= self.config.maximum_curricula
            ):
                raise CurriculumConflict(
                    "Curriculum capacity reached"
                )

            stages = curriculum.ordered_stages()

            runtime = CurriculumRuntime(
                curriculum=curriculum,
                stage_statuses={
                    stage.stage_id: StageStatus.LOCKED
                    for stage in stages
                },
                episodes={
                    stage.stage_id: []
                    for stage in stages
                },
            )

            runtime.stage_statuses[
                stages[0].stage_id
            ] = StageStatus.AVAILABLE

            self._curricula[key] = runtime

            self._log(
                tenant_id=curriculum.tenant_id,
                curriculum_id=curriculum.curriculum_id,
                action="curriculum_registered",
                actor_id=actor_id,
                details={
                    "robot_id": curriculum.robot_id,
                    "model_id": curriculum.model_id,
                    "stage_count": len(stages),
                },
            )

            return runtime.to_dict()

    # --------------------------------------------------------
    # START CURRICULUM
    # --------------------------------------------------------

    def start_curriculum(
        self,
        *,
        tenant_id: str,
        curriculum_id: str,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise CurriculumPermissionDenied(
                "Authorized curriculum start required"
            )

        validate_id(actor_id, "actor_id")

        with self._lock:
            runtime = self._runtime(
                tenant_id,
                curriculum_id,
            )

            if runtime.status != CurriculumStatus.DRAFT:
                raise CurriculumConflict(
                    "Only draft curricula can start"
                )

            first_stage = (
                runtime.curriculum.ordered_stages()[0]
            )

            runtime.status = CurriculumStatus.ACTIVE
            runtime.current_stage_id = first_stage.stage_id

            runtime.stage_statuses[
                first_stage.stage_id
            ] = StageStatus.TRAINING

            runtime.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                curriculum_id=curriculum_id,
                action="curriculum_started",
                actor_id=actor_id,
                details={
                    "stage_id": first_stage.stage_id,
                },
            )

            return runtime.to_dict()

    # --------------------------------------------------------
    # GET CURRICULUM
    # --------------------------------------------------------

    def get_curriculum(
        self,
        *,
        tenant_id: str,
        curriculum_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._runtime(
                tenant_id,
                curriculum_id,
            ).to_dict()

    # --------------------------------------------------------
    # LIST CURRICULA
    # --------------------------------------------------------

    def list_curricula(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        status: CurriculumStatus | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        if status is not None and not isinstance(
            status,
            CurriculumStatus,
        ):
            raise CurriculumValidationError(
                "Invalid curriculum status"
            )

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            results = [
                runtime
                for (tenant, _), runtime
                in self._curricula.items()
                if (
                    tenant == tenant_id
                    and (
                        robot_id is None
                        or runtime.curriculum.robot_id
                        == robot_id
                    )
                    and (
                        status is None
                        or runtime.status == status
                    )
                )
            ]

            results.sort(
                key=lambda item: item.updated_at,
                reverse=True,
            )

            return [
                runtime.to_dict()
                for runtime in results[:limit]
            ]

    # --------------------------------------------------------
    # RECORD TRAINING EPISODE
    # --------------------------------------------------------

    def record_episode(
        self,
        episode: TrainingEpisode,
    ) -> dict[str, Any]:
        if not isinstance(
            episode,
            TrainingEpisode,
        ):
            raise CurriculumValidationError(
                "Invalid training episode"
            )

        with self._lock:
            runtime = self._runtime(
                episode.tenant_id,
                episode.curriculum_id,
            )

            if runtime.status != CurriculumStatus.ACTIVE:
                raise CurriculumConflict(
                    "Curriculum is not active"
                )

            if runtime.current_stage_id != episode.stage_id:
                raise CurriculumConflict(
                    "Episode does not match active stage"
                )

            if (
                runtime.stage_statuses[episode.stage_id]
                != StageStatus.TRAINING
            ):
                raise CurriculumConflict(
                    "Stage is not accepting episodes"
                )

            if episode.episode_id in runtime.episode_ids:
                raise CurriculumConflict(
                    "Duplicate training episode"
                )

            stage = runtime.curriculum.get_stage(
                episode.stage_id
            )

            if (
                episode.episode_steps
                > stage.maximum_episode_steps
            ):
                raise CurriculumValidationError(
                    "Episode exceeds stage step limit"
                )

            if (
                stage.scenario_ids
                and episode.scenario_id
                not in stage.scenario_ids
            ):
                raise CurriculumValidationError(
                    "Episode scenario is not configured "
                    "for this stage"
                )

            episodes = runtime.episodes[
                episode.stage_id
            ]

            if (
                len(episodes)
                >= self.config.maximum_episodes_per_stage
            ):
                raise CurriculumConflict(
                    "Stage episode capacity reached"
                )

            episodes.append(episode)
            runtime.episode_ids.add(
                episode.episode_id
            )

            runtime.updated_at = utc_now()

            self._log(
                tenant_id=episode.tenant_id,
                curriculum_id=episode.curriculum_id,
                action="training_episode_recorded",
                details={
                    "episode_id": episode.episode_id,
                    "stage_id": episode.stage_id,
                    "success": episode.success,
                    "safety_violations": (
                        episode.safety_violations
                    ),
                },
            )

            return episode.to_dict()

    # --------------------------------------------------------
    # EVALUATE STAGE
    # --------------------------------------------------------

    def evaluate_stage(
        self,
        *,
        tenant_id: str,
        curriculum_id: str,
        stage_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            runtime = self._runtime(
                tenant_id,
                curriculum_id,
            )

            stage = runtime.curriculum.get_stage(
                stage_id
            )

            episodes = runtime.episodes[
                stage_id
            ]

            performance = self._calculate_performance(
                curriculum_id,
                stage,
                episodes,
                runtime.curriculum.strategy,
            )

            return performance.to_dict()

    def _calculate_performance(
        self,
        curriculum_id: str,
        stage: CurriculumStage,
        episodes: Sequence[TrainingEpisode],
        strategy: AdaptationStrategy,
    ) -> StagePerformance:
        count = len(episodes)

        evaluation = list(
            episodes[-self.config.evaluation_window:]
        )

        evaluation_count = len(evaluation)

        if evaluation_count == 0:
            return StagePerformance(
                curriculum_id=curriculum_id,
                stage_id=stage.stage_id,
                episode_count=0,
                evaluation_episode_count=0,
                success_rate=0.0,
                failure_rate=0.0,
                safety_violation_rate=0.0,
                mean_reward=0.0,
                reward_std=0.0,
                mean_episode_steps=0.0,
                target_success_rate=stage.target_success_rate,
                minimum_episodes_met=False,
                mastery_criteria_met=False,
                recommendation=(
                    CurriculumRecommendation.INSUFFICIENT_DATA
                ),
                reasons=(
                    "No training episodes recorded.",
                ),
            )

        successes = sum(
            episode.success
            for episode in evaluation
        )

        failures = sum(
            episode.failure
            for episode in evaluation
        )

        unsafe_episodes = sum(
            episode.safety_violations > 0
            for episode in evaluation
        )

        rewards = [
            episode.reward
            for episode in evaluation
        ]

        steps = [
            episode.episode_steps
            for episode in evaluation
        ]

        success_rate = successes / evaluation_count
        failure_rate = failures / evaluation_count
        safety_rate = (
            unsafe_episodes / evaluation_count
        )

        mean_reward = statistics.fmean(rewards)

        reward_std = (
            statistics.pstdev(rewards)
            if evaluation_count > 1
            else 0.0
        )

        mean_steps = statistics.fmean(steps)

        minimum_met = (
            count >= stage.minimum_episodes
            and evaluation_count
            >= self.config.minimum_evaluation_episodes
        )

        reward_met = (
            stage.minimum_mean_reward is None
            or mean_reward >= stage.minimum_mean_reward
        )

        success_met = (
            success_rate >= stage.target_success_rate
        )

        failure_met = (
            failure_rate <= stage.maximum_failure_rate
        )

        safety_met = (
            safety_rate
            <= stage.maximum_safety_violation_rate
        )

        mastery = (
            minimum_met
            and success_met
            and failure_met
            and safety_met
            and reward_met
        )

        reasons = []

        if not minimum_met:
            reasons.append(
                "More evaluation episodes are required."
            )

        if not success_met:
            reasons.append(
                "Success rate is below the stage target."
            )

        if not failure_met:
            reasons.append(
                "Failure rate exceeds the stage limit."
            )

        if not safety_met:
            reasons.append(
                "Safety violation rate exceeds "
                "the stage limit."
            )

        if not reward_met:
            reasons.append(
                "Mean reward is below the stage target."
            )

        if not minimum_met:
            recommendation = (
                CurriculumRecommendation.INSUFFICIENT_DATA
            )

        elif not safety_met:
            recommendation = (
                CurriculumRecommendation.REVIEW
            )

        elif mastery:
            if strategy == AdaptationStrategy.FIXED:
                recommendation = (
                    CurriculumRecommendation.REVIEW
                )
            else:
                recommendation = (
                    CurriculumRecommendation.PROMOTE
                )

        elif (
            success_rate
            < self.config.regression_success_threshold
        ):
            recommendation = (
                CurriculumRecommendation.REGRESS
            )

        elif success_rate < stage.target_success_rate:
            recommendation = (
                CurriculumRecommendation.REPEAT
            )

        else:
            recommendation = (
                CurriculumRecommendation.CONTINUE
            )

        if mastery:
            reasons.append(
                "Configured stage performance "
                "thresholds have been met."
            )

        return StagePerformance(
            curriculum_id=curriculum_id,
            stage_id=stage.stage_id,
            episode_count=count,
            evaluation_episode_count=evaluation_count,
            success_rate=round(success_rate, 6),
            failure_rate=round(failure_rate, 6),
            safety_violation_rate=round(
                safety_rate,
                6,
            ),
            mean_reward=round(mean_reward, 6),
            reward_std=round(reward_std, 6),
            mean_episode_steps=round(
                mean_steps,
                3,
            ),
            target_success_rate=stage.target_success_rate,
            minimum_episodes_met=minimum_met,
            mastery_criteria_met=mastery,
            recommendation=recommendation,
            reasons=tuple(reasons),
        )

    # --------------------------------------------------------
    # PROMOTE STAGE
    # --------------------------------------------------------

    def promote_stage(
        self,
        *,
        tenant_id: str,
        curriculum_id: str,
        reviewer_id: str,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise CurriculumPermissionDenied(
                "Authorized human stage review required"
            )

        validate_id(reviewer_id, "reviewer_id")

        with self._lock:
            runtime = self._runtime(
                tenant_id,
                curriculum_id,
            )

            if runtime.status != CurriculumStatus.ACTIVE:
                raise CurriculumConflict(
                    "Curriculum is not active"
                )

            current_id = runtime.current_stage_id

            if current_id is None:
                raise CurriculumConflict(
                    "No active curriculum stage"
                )

            stage = runtime.curriculum.get_stage(
                current_id
            )

            performance = self._calculate_performance(
                curriculum_id,
                stage,
                runtime.episodes[current_id],
                runtime.curriculum.strategy,
            )

            if not performance.mastery_criteria_met:
                raise CurriculumConflict(
                    "Stage mastery criteria are not met"
                )

            ordered = runtime.curriculum.ordered_stages()

            current_index = next(
                index
                for index, item in enumerate(ordered)
                if item.stage_id == current_id
            )

            runtime.stage_statuses[
                current_id
            ] = StageStatus.COMPLETED

            runtime.reviewed_by[
                current_id
            ] = reviewer_id

            next_index = current_index + 1

            if next_index >= len(ordered):
                runtime.status = (
                    CurriculumStatus.COMPLETED
                )

                runtime.current_stage_id = None

                next_stage_id = None

            else:
                next_stage = ordered[next_index]

                incomplete = [
                    dependency
                    for dependency in next_stage.prerequisites
                    if runtime.stage_statuses.get(
                        dependency
                    ) != StageStatus.COMPLETED
                ]

                if incomplete:
                    raise CurriculumConflict(
                        "Next-stage prerequisites are incomplete"
                    )

                runtime.current_stage_id = (
                    next_stage.stage_id
                )

                runtime.stage_statuses[
                    next_stage.stage_id
                ] = StageStatus.TRAINING

                next_stage_id = next_stage.stage_id

            runtime.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                curriculum_id=curriculum_id,
                action="curriculum_stage_promoted",
                actor_id=reviewer_id,
                details={
                    "completed_stage": current_id,
                    "next_stage": next_stage_id,
                },
            )

            return {
                "curriculum_id": curriculum_id,
                "completed_stage_id": current_id,
                "next_stage_id": next_stage_id,
                "curriculum_status": runtime.status.value,
                "reviewed_by": reviewer_id,
                "physical_deployment_approved": False,
            }

    # --------------------------------------------------------
    # PAUSE / RESUME
    # --------------------------------------------------------

    def pause_curriculum(
        self,
        *,
        tenant_id: str,
        curriculum_id: str,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise CurriculumPermissionDenied(
                "Authorized pause required"
            )

        validate_id(actor_id, "actor_id")

        with self._lock:
            runtime = self._runtime(
                tenant_id,
                curriculum_id,
            )

            if runtime.status != CurriculumStatus.ACTIVE:
                raise CurriculumConflict(
                    "Only active curricula can be paused"
                )

            runtime.status = CurriculumStatus.PAUSED
            runtime.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                curriculum_id=curriculum_id,
                action="curriculum_paused",
                actor_id=actor_id,
            )

            return runtime.to_dict()

    def resume_curriculum(
        self,
        *,
        tenant_id: str,
        curriculum_id: str,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise CurriculumPermissionDenied(
                "Authorized resume required"
            )

        validate_id(actor_id, "actor_id")

        with self._lock:
            runtime = self._runtime(
                tenant_id,
                curriculum_id,
            )

            if runtime.status != CurriculumStatus.PAUSED:
                raise CurriculumConflict(
                    "Only paused curricula can resume"
                )

            runtime.status = CurriculumStatus.ACTIVE
            runtime.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                curriculum_id=curriculum_id,
                action="curriculum_resumed",
                actor_id=actor_id,
            )

            return runtime.to_dict()

    # --------------------------------------------------------
    # TRAINING PLAN
    # --------------------------------------------------------

    def training_plan(
        self,
        *,
        tenant_id: str,
        curriculum_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            runtime = self._runtime(
                tenant_id,
                curriculum_id,
            )

            current_id = runtime.current_stage_id

            if current_id is None:
                return {
                    "curriculum_id": curriculum_id,
                    "status": runtime.status.value,
                    "active_stage": None,
                    "simulation_only": True,
                }

            stage = runtime.curriculum.get_stage(
                current_id
            )

            performance = self._calculate_performance(
                curriculum_id,
                stage,
                runtime.episodes[current_id],
                runtime.curriculum.strategy,
            )

            return {
                "curriculum_id": curriculum_id,
                "robot_id": runtime.curriculum.robot_id,
                "model_id": runtime.curriculum.model_id,
                "status": runtime.status.value,
                "active_stage": stage.to_dict(),
                "performance": performance.to_dict(),
                "remaining_minimum_episodes": max(
                    0,
                    stage.minimum_episodes
                    - len(runtime.episodes[current_id]),
                ),
                "recommended_action": (
                    performance.recommendation.value
                ),
                "simulation_only": True,
                "physical_execution_authorized": False,
            }

    # --------------------------------------------------------
    # CURRICULUM PROGRESS
    # --------------------------------------------------------

    def curriculum_progress(
        self,
        *,
        tenant_id: str,
        curriculum_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            runtime = self._runtime(
                tenant_id,
                curriculum_id,
            )

            statuses = runtime.stage_statuses

            total = len(statuses)

            completed = sum(
                status == StageStatus.COMPLETED
                for status in statuses.values()
            )

            total_episodes = sum(
                len(episodes)
                for episodes in runtime.episodes.values()
            )

            return {
                "curriculum_id": curriculum_id,
                "tenant_id": tenant_id,
                "robot_id": runtime.curriculum.robot_id,
                "model_id": runtime.curriculum.model_id,
                "status": runtime.status.value,
                "current_stage_id": runtime.current_stage_id,
                "total_stages": total,
                "completed_stages": completed,
                "progress_percentage": round(
                    completed / total * 100,
                    2,
                ) if total else 0.0,
                "total_training_episodes": total_episodes,
                "stage_statuses": {
                    stage_id: status.value
                    for stage_id, status in statuses.items()
                },
                "physical_deployment_approved": False,
            }

    # --------------------------------------------------------
    # SIM-TO-REAL READINESS
    # --------------------------------------------------------

    def sim_to_real_readiness(
        self,
        *,
        tenant_id: str,
        curriculum_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            runtime = self._runtime(
                tenant_id,
                curriculum_id,
            )

            stages = runtime.curriculum.ordered_stages()

            performance = [
                self._calculate_performance(
                    curriculum_id,
                    stage,
                    runtime.episodes[stage.stage_id],
                    runtime.curriculum.strategy,
                )
                for stage in stages
            ]

            completed = (
                runtime.status == CurriculumStatus.COMPLETED
            )

            mastery = all(
                item.mastery_criteria_met
                for item in performance
            )

            violations = sum(
                episode.safety_violations
                for episodes in runtime.episodes.values()
                for episode in episodes
            )

            return {
                "curriculum_id": curriculum_id,
                "simulation_curriculum_completed": completed,
                "all_stage_thresholds_met": mastery,
                "total_recorded_safety_violations": violations,
                "simulation_training_ready": (
                    completed and mastery
                ),
                "physical_deployment_ready": False,
                "physical_deployment_approved": False,
                "required_external_evaluations": [
                    "Independent safety assessment",
                    "Hardware compatibility validation",
                    "Real-world sensor validation",
                    "Out-of-distribution testing",
                    "Domain gap evaluation",
                    "Collision and emergency-stop testing",
                    "Qualified human approval",
                ],
                "warning": (
                    "Simulation performance alone cannot "
                    "establish real-world robot safety."
                ),
            }

    # --------------------------------------------------------
    # EPISODE HISTORY
    # --------------------------------------------------------

    def episode_history(
        self,
        *,
        tenant_id: str,
        curriculum_id: str,
        stage_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            runtime = self._runtime(
                tenant_id,
                curriculum_id,
            )

            runtime.curriculum.get_stage(stage_id)

            episodes = runtime.episodes[
                stage_id
            ]

            return [
                episode.to_dict()
                for episode in episodes[-limit:]
            ]

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        *,
        tenant_id: str,
        curriculum_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if curriculum_id is not None:
            validate_id(
                curriculum_id,
                "curriculum_id",
            )

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            events = [
                copy.deepcopy(event)
                for event in self._audit
                if (
                    event["tenant_id"] == tenant_id
                    and (
                        curriculum_id is None
                        or event["curriculum_id"]
                        == curriculum_id
                    )
                )
            ]

            return events[-limit:]

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
    ) -> Path:
        validate_id(tenant_id, "tenant_id")

        output = Path(path)

        with self._lock:
            curricula = [
                runtime.to_dict()
                for (tenant, _), runtime
                in self._curricula.items()
                if tenant == tenant_id
            ]

            episodes = [
                episode.to_dict()
                for (tenant, _), runtime
                in self._curricula.items()
                if tenant == tenant_id
                for stage_episodes
                in runtime.episodes.values()
                for episode in stage_episodes
            ]

            audit = [
                copy.deepcopy(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ]

        payload = {
            "generated_at": utc_now().isoformat(),
            "tenant_id": tenant_id,
            "curricula": curricula,
            "training_episodes": episodes,
            "audit_events": audit,
            "simulation_only": True,
            "physical_deployment_approved": False,
        }

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                json_safe(payload),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output


# ============================================================
# DEFAULT CURRICULUM FACTORY
# ============================================================

def create_navigation_curriculum(
    *,
    tenant_id: str,
    robot_id: str,
    model_id: str,
    curriculum_id: str = "navigation_curriculum",
) -> RobotCurriculum:
    stages = (
        CurriculumStage(
            stage_id="stage_001",
            name="Basic Navigation",
            stage_order=1,
            difficulty=DifficultyLevel.BEGINNER,
            task_type=TrainingTaskType.NAVIGATION,
            scenario_ids=(
                "empty_environment",
            ),
            target_success_rate=0.80,
            minimum_episodes=20,
            maximum_failure_rate=0.20,
            maximum_safety_violation_rate=0.0,
            maximum_episode_steps=500,
            environment_parameters={
                "obstacle_count": 0,
                "dynamic_obstacles": False,
            },
            description="Basic point-to-point navigation",
        ),
        CurriculumStage(
            stage_id="stage_002",
            name="Static Obstacle Avoidance",
            stage_order=2,
            difficulty=DifficultyLevel.EASY,
            task_type=TrainingTaskType.OBSTACLE_AVOIDANCE,
            scenario_ids=(
                "static_obstacles",
            ),
            prerequisites=(
                "stage_001",
            ),
            target_success_rate=0.85,
            minimum_episodes=30,
            maximum_failure_rate=0.15,
            maximum_safety_violation_rate=0.0,
            maximum_episode_steps=750,
            environment_parameters={
                "obstacle_count": 5,
                "dynamic_obstacles": False,
            },
        ),
        CurriculumStage(
            stage_id="stage_003",
            name="Dynamic Obstacle Navigation",
            stage_order=3,
            difficulty=DifficultyLevel.MODERATE,
            task_type=TrainingTaskType.OBSTACLE_AVOIDANCE,
            scenario_ids=(
                "dynamic_obstacles",
            ),
            prerequisites=(
                "stage_002",
            ),
            target_success_rate=0.90,
            minimum_episodes=40,
            maximum_failure_rate=0.10,
            maximum_safety_violation_rate=0.0,
            maximum_episode_steps=1000,
            randomization=DomainRandomizationProfile(
                parameters=(
                    RandomizationParameter(
                        name="friction",
                        minimum=0.4,
                        maximum=1.0,
                    ),
                    RandomizationParameter(
                        name="sensor_noise",
                        minimum=0.0,
                        maximum=0.1,
                    ),
                    RandomizationParameter(
                        name="obstacle_speed",
                        minimum=0.1,
                        maximum=1.0,
                    ),
                ),
                seed=42,
            ),
            environment_parameters={
                "obstacle_count": 10,
                "dynamic_obstacles": True,
            },
        ),
        CurriculumStage(
            stage_id="stage_004",
            name="Complex Warehouse Navigation",
            stage_order=4,
            difficulty=DifficultyLevel.HARD,
            task_type=TrainingTaskType.NAVIGATION,
            scenario_ids=(
                "warehouse_complex",
            ),
            prerequisites=(
                "stage_003",
            ),
            target_success_rate=0.95,
            minimum_episodes=50,
            maximum_failure_rate=0.05,
            maximum_safety_violation_rate=0.0,
            maximum_episode_steps=1500,
            environment_parameters={
                "obstacle_count": 20,
                "dynamic_obstacles": True,
                "narrow_corridors": True,
            },
        ),
    )

    return RobotCurriculum(
        curriculum_id=curriculum_id,
        tenant_id=tenant_id,
        robot_id=robot_id,
        model_id=model_id,
        name="Adaptive Robot Navigation Curriculum",
        stages=stages,
        strategy=AdaptationStrategy.MASTERY_BASED,
        description=(
            "Progressive navigation training from "
            "empty environments to complex warehouses."
        ),
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    engine = RoboticsCurriculumEngine()

    curriculum = create_navigation_curriculum(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        model_id="navigation_policy_v1",
    )

    engine.register_curriculum(
        curriculum,
        authorized=True,
        actor_id="training_admin",
    )

    engine.start_curriculum(
        tenant_id="tenant_alpha",
        curriculum_id="navigation_curriculum",
        authorized=True,
        actor_id="training_admin",
    )

    for index in range(20):
        episode = TrainingEpisode(
            episode_id=f"episode_{index + 1:03d}",
            tenant_id="tenant_alpha",
            curriculum_id="navigation_curriculum",
            stage_id="stage_001",
            success=True,
            reward=95.0 + index * 0.1,
            episode_steps=120 + index,
            failure=False,
            safety_violations=0,
            scenario_id="empty_environment",
            checkpoint_reference=(
                f"model_checkpoint_{index + 1:03d}"
            ),
            metrics={
                "collision_count": 0,
                "distance_to_goal": 0.05,
            },
        )

        engine.record_episode(episode)

    print("\nSTAGE PERFORMANCE")

    print(
        json.dumps(
            engine.evaluate_stage(
                tenant_id="tenant_alpha",
                curriculum_id="navigation_curriculum",
                stage_id="stage_001",
            ),
            indent=2,
        )
    )

    print("\nSTAGE PROMOTION")

    print(
        json.dumps(
            engine.promote_stage(
                tenant_id="tenant_alpha",
                curriculum_id="navigation_curriculum",
                reviewer_id="training_reviewer_001",
                authorized=True,
            ),
            indent=2,
        )
    )

    print("\nTRAINING PLAN")

    print(
        json.dumps(
            engine.training_plan(
                tenant_id="tenant_alpha",
                curriculum_id="navigation_curriculum",
            ),
            indent=2,
        )
    )

    print("\nCURRICULUM PROGRESS")

    print(
        json.dumps(
            engine.curriculum_progress(
                tenant_id="tenant_alpha",
                curriculum_id="navigation_curriculum",
            ),
            indent=2,
        )
    )

    print("\nSIM-TO-REAL READINESS")

    print(
        json.dumps(
            engine.sim_to_real_readiness(
                tenant_id="tenant_alpha",
                curriculum_id="navigation_curriculum",
            ),
            indent=2,
        )
    )

    output = engine.export_json(
        "data/robotics/curriculum_report.json",
        tenant_id="tenant_alpha",
    )

    print("\nExported:", output)


if __name__ == "__main__":
    main()
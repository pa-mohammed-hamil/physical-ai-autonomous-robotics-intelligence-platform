"""Policy module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/policy.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Reinforcement Learning Policy Engine

Python 3.11+
Dependencies:
    pip install numpy torch

Features:
    - Discrete and continuous action policies
    - Stochastic and deterministic action selection
    - Actor-Critic neural networks
    - Gaussian continuous-action policies
    - Action probability and log-probability
    - Action masking
    - Observation normalization
    - Policy entropy
    - Value estimation
    - Policy registration and versioning
    - Batch inference
    - Simulation-only action generation
    - Human-reviewed simulation approval
    - Checkpoint saving and loading
    - Multi-tenant isolation
    - Audit logging
    - JSON report export

Integrations:
    environment.py
    observation_space.py
    action_space.py
    imitation_learning.py
    offline_rl.py
    policy_evaluation.py
    curriculum.py
    domain_randomization.py
    simulator.py
    safety_monitor.py

Safety:
    This module generates actions for offline analysis
    and simulation only.

    It does not communicate with physical robots,
    actuators, ROS2 command topics, or motor controllers.

    Policy approval does not authorize physical deployment.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import threading
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal


# ============================================================
# EXCEPTIONS
# ============================================================

class PolicyError(Exception):
    pass


class PolicyValidationError(PolicyError):
    pass


class PolicyNotFound(PolicyError):
    pass


class PolicyConflict(PolicyError):
    pass


class PolicyPermissionDenied(PolicyError):
    pass


# ============================================================
# ENUMS
# ============================================================

class PolicyType(str, Enum):
    DISCRETE_ACTOR_CRITIC = "discrete_actor_critic"
    CONTINUOUS_ACTOR_CRITIC = "continuous_actor_critic"


class ActionSelectionMode(str, Enum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"


class PolicyStatus(str, Enum):
    REGISTERED = "registered"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    APPROVED_FOR_SIMULATION = "approved_for_simulation"
    ARCHIVED = "archived"


class ExecutionMode(str, Enum):
    SIMULATION = "simulation"
    OFFLINE_ANALYSIS = "offline_analysis"


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
            character.isalnum() or character in "._-"
            for character in value
        )
    ):
        raise PolicyValidationError(
            f"Invalid {name}"
        )

    return value


def finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, str))
        or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        )
    ):
        raise PolicyValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise PolicyValidationError(
            f"{name} must be finite"
        )

    return result


def validate_observation(
    observation: Sequence[float],
    dimension: int,
) -> np.ndarray:
    try:
        array = np.asarray(
            observation,
            dtype=np.float32,
        )
    except (TypeError, ValueError) as exc:
        raise PolicyValidationError(
            "Invalid observation"
        ) from exc

    if array.shape != (dimension,):
        raise PolicyValidationError(
            f"Expected observation shape ({dimension},)"
        )

    if not np.isfinite(array).all():
        raise PolicyValidationError(
            "Observation contains non-finite values"
        )

    return array


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

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


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PolicyConfig:
    observation_dimension: int
    action_dimension: int

    policy_type: PolicyType

    hidden_dimensions: tuple[int, ...] = (256, 256)

    learning_rate: float = 0.0003

    minimum_log_std: float = -5.0
    maximum_log_std: float = 2.0

    observation_clip: float = 10.0

    random_seed: int = 42

    device: str = "cpu"

    maximum_audit_events: int = 20_000

    simulation_only: bool = True
    human_approval_required: bool = True

    def __post_init__(self) -> None:
        for name in (
            "observation_dimension",
            "action_dimension",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise PolicyValidationError(
                    f"{name} must be a positive integer"
                )

        if not isinstance(self.policy_type, PolicyType):
            raise PolicyValidationError(
                "Invalid policy type"
            )

        if not self.hidden_dimensions:
            raise PolicyValidationError(
                "At least one hidden layer is required"
            )

        for dimension in self.hidden_dimensions:
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or not 1 <= dimension <= 8192
            ):
                raise PolicyValidationError(
                    "Invalid hidden dimension"
                )

        if finite_number(
            self.learning_rate,
            "learning_rate",
        ) <= 0:
            raise PolicyValidationError(
                "Learning rate must be positive"
            )

        if (
            finite_number(
                self.minimum_log_std,
                "minimum_log_std",
            )
            >= finite_number(
                self.maximum_log_std,
                "maximum_log_std",
            )
        ):
            raise PolicyValidationError(
                "Invalid log standard deviation bounds"
            )

        if finite_number(
            self.observation_clip,
            "observation_clip",
        ) <= 0:
            raise PolicyValidationError(
                "Observation clip must be positive"
            )

        if self.device not in ("cpu", "cuda"):
            raise PolicyValidationError(
                "Supported devices: cpu and cuda"
            )

        if (
            self.device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise PolicyValidationError(
                "CUDA is unavailable"
            )

        if self.simulation_only is not True:
            raise PolicyValidationError(
                "Physical execution is unsupported"
            )

        if self.human_approval_required is not True:
            raise PolicyValidationError(
                "Human approval cannot be disabled"
            )


# ============================================================
# OBSERVATION NORMALIZER
# ============================================================

@dataclass
class PolicyObservationNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def identity(
        cls,
        dimension: int,
    ) -> "PolicyObservationNormalizer":
        return cls(
            mean=np.zeros(
                dimension,
                dtype=np.float32,
            ),
            std=np.ones(
                dimension,
                dtype=np.float32,
            ),
        )

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
    ) -> "PolicyObservationNormalizer":
        array = np.asarray(
            observations,
            dtype=np.float32,
        )

        if array.ndim != 2 or len(array) == 0:
            raise PolicyValidationError(
                "Expected nonempty 2D observation array"
            )

        if not np.isfinite(array).all():
            raise PolicyValidationError(
                "Non-finite observations"
            )

        mean = array.mean(axis=0)

        std = array.std(axis=0)

        std = np.where(
            std < 1e-6,
            1.0,
            std,
        )

        return cls(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
        )

    def transform(
        self,
        observations: np.ndarray,
        clip: float = 10.0,
    ) -> np.ndarray:
        result = (
            observations - self.mean
        ) / self.std

        return np.clip(
            result,
            -clip,
            clip,
        ).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


# ============================================================
# NEURAL NETWORK UTILITIES
# ============================================================

def build_mlp(
    input_dimension: int,
    output_dimension: int,
    hidden_dimensions: tuple[int, ...],
) -> nn.Sequential:
    dimensions = (
        input_dimension,
        *hidden_dimensions,
        output_dimension,
    )

    layers: list[nn.Module] = []

    for index in range(len(dimensions) - 1):
        layers.append(
            nn.Linear(
                dimensions[index],
                dimensions[index + 1],
            )
        )

        if index < len(dimensions) - 2:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


# ============================================================
# DISCRETE ACTOR-CRITIC
# ============================================================

class DiscreteActorCritic(nn.Module):

    def __init__(
        self,
        config: PolicyConfig,
    ) -> None:
        super().__init__()

        self.actor = build_mlp(
            config.observation_dimension,
            config.action_dimension,
            config.hidden_dimensions,
        )

        self.critic = build_mlp(
            config.observation_dimension,
            1,
            config.hidden_dimensions,
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(observations)

        values = self.critic(
            observations
        ).squeeze(-1)

        return logits, values


# ============================================================
# CONTINUOUS ACTOR-CRITIC
# ============================================================

class ContinuousActorCritic(nn.Module):

    def __init__(
        self,
        config: PolicyConfig,
    ) -> None:
        super().__init__()

        self.actor_mean = build_mlp(
            config.observation_dimension,
            config.action_dimension,
            config.hidden_dimensions,
        )

        self.actor_log_std = nn.Parameter(
            torch.zeros(
                config.action_dimension
            )
        )

        self.critic = build_mlp(
            config.observation_dimension,
            1,
            config.hidden_dimensions,
        )

        self.minimum_log_std = (
            config.minimum_log_std
        )

        self.maximum_log_std = (
            config.maximum_log_std
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        means = self.actor_mean(
            observations
        )

        log_std = torch.clamp(
            self.actor_log_std,
            self.minimum_log_std,
            self.maximum_log_std,
        )

        std = log_std.exp().expand_as(
            means
        )

        values = self.critic(
            observations
        ).squeeze(-1)

        return means, std, values


# ============================================================
# POLICY PROFILE
# ============================================================

@dataclass
class PolicyProfile:
    policy_id: str
    tenant_id: str
    robot_id: str

    name: str
    version: str

    config: PolicyConfig

    status: PolicyStatus = PolicyStatus.REGISTERED

    reviewer_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "policy_id",
            "tenant_id",
            "robot_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if not self.name.strip():
            raise PolicyValidationError(
                "Policy name cannot be empty"
            )

        if not self.version.strip():
            raise PolicyValidationError(
                "Policy version cannot be empty"
            )

        if not isinstance(self.config, PolicyConfig):
            raise PolicyValidationError(
                "Invalid policy configuration"
            )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "policy_id": self.policy_id,
                "tenant_id": self.tenant_id,
                "robot_id": self.robot_id,
                "name": self.name,
                "version": self.version,
                "policy_type": self.config.policy_type,
                "observation_dimension": (
                    self.config.observation_dimension
                ),
                "action_dimension": (
                    self.config.action_dimension
                ),
                "hidden_dimensions": (
                    self.config.hidden_dimensions
                ),
                "status": self.status,
                "reviewer_id": self.reviewer_id,
                "metadata": self.metadata,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "simulation_only": True,
                "physical_deployment_approved": False,
            }
        )


# ============================================================
# POLICY ACTION RESULT
# ============================================================

@dataclass(frozen=True)
class PolicyActionResult:
    request_id: str

    tenant_id: str
    robot_id: str
    policy_id: str

    action: int | tuple[float, ...]

    selection_mode: ActionSelectionMode
    execution_mode: ExecutionMode

    log_probability: float
    entropy: float
    state_value: float

    action_probabilities: tuple[float, ...] | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                **self.__dict__,
                "simulation_only": True,
                "physical_execution_authorized": False,
            }
        )


# ============================================================
# POLICY ENGINE
# ============================================================

class RoboticsPolicyEngine:

    def __init__(self) -> None:
        self._policies: dict[
            tuple[str, str],
            PolicyProfile,
        ] = {}

        self._networks: dict[
            tuple[str, str],
            nn.Module,
        ] = {}

        self._normalizers: dict[
            tuple[str, str],
            PolicyObservationNormalizer,
        ] = {}

        self._audit: deque[
            dict[str, Any]
        ] = deque(
            maxlen=20_000
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _log(
        self,
        *,
        tenant_id: str,
        action: str,
        entity_id: str,
        actor_id: str = "system",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._audit.append(
            {
                "event_id": new_id(),
                "timestamp": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "action": action,
                "entity_id": entity_id,
                "actor_id": actor_id,
                "details": json_safe(
                    dict(details or {})
                ),
            }
        )

    # --------------------------------------------------------
    # REGISTER POLICY
    # --------------------------------------------------------

    def register_policy(
        self,
        profile: PolicyProfile,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PolicyPermissionDenied(
                "Authorized policy registration required"
            )

        if not isinstance(profile, PolicyProfile):
            raise PolicyValidationError(
                "Invalid policy profile"
            )

        validate_id(actor_id, "actor_id")

        key = (
            profile.tenant_id,
            profile.policy_id,
        )

        with self._lock:
            if key in self._policies:
                raise PolicyConflict(
                    "Policy already exists"
                )

            torch.manual_seed(
                profile.config.random_seed
            )

            if (
                profile.config.policy_type
                == PolicyType.DISCRETE_ACTOR_CRITIC
            ):
                network = DiscreteActorCritic(
                    profile.config
                )

            else:
                network = ContinuousActorCritic(
                    profile.config
                )

            network = network.to(
                profile.config.device
            )

            network.eval()

            self._policies[key] = copy.deepcopy(
                profile
            )

            self._networks[key] = network

            self._normalizers[key] = (
                PolicyObservationNormalizer.identity(
                    profile.config.observation_dimension
                )
            )

            self._log(
                tenant_id=profile.tenant_id,
                action="policy_registered",
                entity_id=profile.policy_id,
                actor_id=actor_id,
                details={
                    "policy_type": (
                        profile.config.policy_type.value
                    ),
                    "version": profile.version,
                },
            )

            return self._policies[key].to_dict()

    # --------------------------------------------------------
    # GET POLICY
    # --------------------------------------------------------

    def _get_profile(
        self,
        tenant_id: str,
        policy_id: str,
    ) -> PolicyProfile:
        validate_id(tenant_id, "tenant_id")
        validate_id(policy_id, "policy_id")

        profile = self._policies.get(
            (tenant_id, policy_id)
        )

        if profile is None:
            raise PolicyNotFound(
                "Policy not found"
            )

        return profile

    def get_policy(
        self,
        *,
        tenant_id: str,
        policy_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get_profile(
                tenant_id,
                policy_id,
            ).to_dict()

    # --------------------------------------------------------
    # UPDATE OBSERVATION NORMALIZER
    # --------------------------------------------------------

    def fit_observation_normalizer(
        self,
        *,
        tenant_id: str,
        policy_id: str,
        observations: Sequence[Sequence[float]],
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PolicyPermissionDenied(
                "Authorized normalizer update required"
            )

        with self._lock:
            profile = self._get_profile(
                tenant_id,
                policy_id,
            )

            if profile.status in (
                PolicyStatus.APPROVED_FOR_SIMULATION,
                PolicyStatus.ARCHIVED,
            ):
                raise PolicyConflict(
                    "Approved or archived policy cannot "
                    "be modified"
                )

            try:
                array = np.asarray(
                    observations,
                    dtype=np.float32,
                )
            except (TypeError, ValueError) as exc:
                raise PolicyValidationError(
                    "Invalid observation dataset"
                ) from exc

            if (
                array.ndim != 2
                or array.shape[1]
                != profile.config.observation_dimension
                or len(array) == 0
            ):
                raise PolicyValidationError(
                    "Observation dataset shape mismatch"
                )

            normalizer = (
                PolicyObservationNormalizer.fit(
                    array
                )
            )

            self._normalizers[
                (tenant_id, policy_id)
            ] = normalizer

            profile.status = PolicyStatus.REGISTERED
            profile.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                action="policy_normalizer_updated",
                entity_id=policy_id,
                actor_id=actor_id,
            )

            return normalizer.to_dict()

    # --------------------------------------------------------
    # LOAD TRAINED WEIGHTS
    # --------------------------------------------------------

    def load_state_dict(
        self,
        *,
        tenant_id: str,
        policy_id: str,
        state_dict: Mapping[str, torch.Tensor],
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PolicyPermissionDenied(
                "Authorized model update required"
            )

        with self._lock:
            profile = self._get_profile(
                tenant_id,
                policy_id,
            )

            if profile.status in (
                PolicyStatus.APPROVED_FOR_SIMULATION,
                PolicyStatus.ARCHIVED,
            ):
                raise PolicyConflict(
                    "Cannot modify approved or archived policy"
                )

            key = (tenant_id, policy_id)

            candidate = copy.deepcopy(
                self._networks[key]
            )

            candidate.load_state_dict(
                state_dict,
                strict=True,
            )

            for parameter in candidate.parameters():
                if not torch.isfinite(parameter).all():
                    raise PolicyValidationError(
                        "Model contains non-finite weights"
                    )

            candidate.eval()

            self._networks[key] = candidate

            profile.status = PolicyStatus.TRAINED
            profile.reviewer_id = None
            profile.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                action="policy_weights_loaded",
                entity_id=policy_id,
                actor_id=actor_id,
            )

            return profile.to_dict()

    # --------------------------------------------------------
    # MARK EVALUATED
    # --------------------------------------------------------

    def mark_evaluated(
        self,
        *,
        tenant_id: str,
        policy_id: str,
        evaluation_id: str,
        passed: bool,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PolicyPermissionDenied(
                "Authorized evaluation update required"
            )

        validate_id(
            evaluation_id,
            "evaluation_id",
        )

        with self._lock:
            profile = self._get_profile(
                tenant_id,
                policy_id,
            )

            if profile.status != PolicyStatus.TRAINED:
                raise PolicyConflict(
                    "Policy must be trained before evaluation"
                )

            if passed is not True:
                self._log(
                    tenant_id=tenant_id,
                    action="policy_evaluation_failed",
                    entity_id=policy_id,
                    actor_id=actor_id,
                    details={
                        "evaluation_id": evaluation_id,
                    },
                )

                return {
                    "policy_id": policy_id,
                    "status": profile.status.value,
                    "evaluation_passed": False,
                }

            profile.status = PolicyStatus.EVALUATED
            profile.updated_at = utc_now()

            profile.metadata[
                "evaluation_id"
            ] = evaluation_id

            self._log(
                tenant_id=tenant_id,
                action="policy_evaluation_passed",
                entity_id=policy_id,
                actor_id=actor_id,
                details={
                    "evaluation_id": evaluation_id,
                },
            )

            return profile.to_dict()

    # --------------------------------------------------------
    # HUMAN APPROVAL
    # --------------------------------------------------------

    def approve_for_simulation(
        self,
        *,
        tenant_id: str,
        policy_id: str,
        reviewer_id: str,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PolicyPermissionDenied(
                "Human approval required"
            )

        validate_id(
            reviewer_id,
            "reviewer_id",
        )

        with self._lock:
            profile = self._get_profile(
                tenant_id,
                policy_id,
            )

            if profile.status != PolicyStatus.EVALUATED:
                raise PolicyConflict(
                    "Policy must pass evaluation first"
                )

            profile.status = (
                PolicyStatus.APPROVED_FOR_SIMULATION
            )

            profile.reviewer_id = reviewer_id
            profile.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                action="policy_approved_for_simulation",
                entity_id=policy_id,
                actor_id=reviewer_id,
            )

            return profile.to_dict()

    # --------------------------------------------------------
    # ACTION MASK
    # --------------------------------------------------------

    @staticmethod
    def _validate_action_mask(
        mask: Sequence[bool] | None,
        action_dimension: int,
    ) -> torch.Tensor | None:
        if mask is None:
            return None

        values = list(mask)

        if len(values) != action_dimension:
            raise PolicyValidationError(
                "Action mask dimension mismatch"
            )

        if not all(
            isinstance(value, bool)
            for value in values
        ):
            raise PolicyValidationError(
                "Action mask must contain booleans"
            )

        if not any(values):
            raise PolicyValidationError(
                "At least one action must be permitted"
            )

        return torch.tensor(
            values,
            dtype=torch.bool,
        )

    # --------------------------------------------------------
    # ACTION SELECTION
    # --------------------------------------------------------

    def select_action(
        self,
        *,
        tenant_id: str,
        policy_id: str,
        observation: Sequence[float],
        selection_mode: ActionSelectionMode = (
            ActionSelectionMode.DETERMINISTIC
        ),
        execution_mode: ExecutionMode = (
            ExecutionMode.OFFLINE_ANALYSIS
        ),
        action_mask: Sequence[bool] | None = None,
        action_bounds: tuple[
            Sequence[float],
            Sequence[float],
        ] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(
            selection_mode,
            ActionSelectionMode,
        ):
            raise PolicyValidationError(
                "Invalid action selection mode"
            )

        if not isinstance(
            execution_mode,
            ExecutionMode,
        ):
            raise PolicyValidationError(
                "Invalid execution mode"
            )

        with self._lock:
            profile = self._get_profile(
                tenant_id,
                policy_id,
            )

            if profile.status == PolicyStatus.ARCHIVED:
                raise PolicyConflict(
                    "Archived policy cannot generate actions"
                )

            if (
                execution_mode == ExecutionMode.SIMULATION
                and profile.status
                != PolicyStatus.APPROVED_FOR_SIMULATION
            ):
                raise PolicyPermissionDenied(
                    "Policy is not approved for simulation"
                )

            config = profile.config
            key = (tenant_id, policy_id)

            vector = validate_observation(
                observation,
                config.observation_dimension,
            )

            normalized = self._normalizers[
                key
            ].transform(
                vector,
                clip=config.observation_clip,
            )

            tensor = torch.from_numpy(
                normalized
            ).unsqueeze(0).to(
                config.device
            )

            network = self._networks[key]
            network.eval()

            with torch.no_grad():
                if (
                    config.policy_type
                    == PolicyType.DISCRETE_ACTOR_CRITIC
                ):
                    if action_bounds is not None:
                        raise PolicyValidationError(
                            "Action bounds are only supported "
                            "for continuous policies"
                        )

                    logits, values = network(
                        tensor
                    )

                    mask = self._validate_action_mask(
                        action_mask,
                        config.action_dimension,
                    )

                    if mask is not None:
                        logits = logits.masked_fill(
                            ~mask.to(config.device),
                            float("-inf"),
                        )

                    distribution = Categorical(
                        logits=logits
                    )

                    if (
                        selection_mode
                        == ActionSelectionMode.DETERMINISTIC
                    ):
                        action_tensor = torch.argmax(
                            logits,
                            dim=-1,
                        )

                    else:
                        action_tensor = (
                            distribution.sample()
                        )

                    action = int(
                        action_tensor.item()
                    )

                    log_probability = float(
                        distribution.log_prob(
                            action_tensor
                        ).item()
                    )

                    entropy = float(
                        distribution.entropy().item()
                    )

                    action_probabilities = tuple(
                        float(value)
                        for value in distribution.probs[
                            0
                        ].cpu().tolist()
                    )

                    state_value = float(
                        values.item()
                    )

                else:
                    if action_mask is not None:
                        raise PolicyValidationError(
                            "Action masks are only supported "
                            "for discrete policies"
                        )

                    means, std, values = network(
                        tensor
                    )

                    distribution = Normal(
                        means,
                        std,
                    )

                    if (
                        selection_mode
                        == ActionSelectionMode.DETERMINISTIC
                    ):
                        action_tensor = means

                    else:
                        action_tensor = (
                            distribution.sample()
                        )

                    if action_bounds is not None:
                        lower = np.asarray(
                            action_bounds[0],
                            dtype=np.float32,
                        )

                        upper = np.asarray(
                            action_bounds[1],
                            dtype=np.float32,
                        )

                        expected_shape = (
                            config.action_dimension,
                        )

                        if (
                            lower.shape != expected_shape
                            or upper.shape != expected_shape
                            or not np.isfinite(lower).all()
                            or not np.isfinite(upper).all()
                            or np.any(lower >= upper)
                        ):
                            raise PolicyValidationError(
                                "Invalid continuous action bounds"
                            )

                        # Squashed Gaussian policy:
                        # tanh maps the unbounded action to
                        # (-1, 1), then affine scaling maps
                        # it into the configured bounds.
                        #
                        # Log probability includes the
                        # Jacobian correction.

                        bounded = torch.tanh(
                            action_tensor
                        )

                        scale = torch.from_numpy(
                            (upper - lower) / 2.0
                        ).to(config.device)

                        offset = torch.from_numpy(
                            (upper + lower) / 2.0
                        ).to(config.device)

                        executed_action = (
                            offset + scale * bounded
                        )

                        correction = torch.log(
                            scale * (
                                1.0 - bounded.square()
                            ) + 1e-6
                        )

                        log_probability = float(
                            (
                                distribution.log_prob(
                                    action_tensor
                                )
                                - correction
                            ).sum(dim=-1).item()
                        )

                        action_tensor = executed_action

                    else:
                        log_probability = float(
                            distribution.log_prob(
                                action_tensor
                            ).sum(dim=-1).item()
                        )

                    action = tuple(
                        float(value)
                        for value in action_tensor[
                            0
                        ].cpu().tolist()
                    )

                    # Gaussian entropy is computed before
                    # optional tanh squashing.
                    entropy = float(
                        distribution.entropy().sum(
                            dim=-1
                        ).item()
                    )

                    state_value = float(
                        values.item()
                    )

                    action_probabilities = None

            result = PolicyActionResult(
                request_id=new_id(),
                tenant_id=tenant_id,
                robot_id=profile.robot_id,
                policy_id=policy_id,
                action=action,
                selection_mode=selection_mode,
                execution_mode=execution_mode,
                log_probability=log_probability,
                entropy=entropy,
                state_value=state_value,
                action_probabilities=action_probabilities,
            )

            self._log(
                tenant_id=tenant_id,
                action="policy_action_generated",
                entity_id=result.request_id,
                details={
                    "policy_id": policy_id,
                    "execution_mode": (
                        execution_mode.value
                    ),
                },
            )

            return result.to_dict()

    # --------------------------------------------------------
    # BATCH ACTION SELECTION
    # --------------------------------------------------------

    def select_actions_batch(
        self,
        *,
        tenant_id: str,
        policy_id: str,
        observations: Sequence[Sequence[float]],
        selection_mode: ActionSelectionMode = (
            ActionSelectionMode.DETERMINISTIC
        ),
        execution_mode: ExecutionMode = (
            ExecutionMode.OFFLINE_ANALYSIS
        ),
    ) -> dict[str, Any]:
        if not observations:
            raise PolicyValidationError(
                "Batch cannot be empty"
            )

        results = [
            self.select_action(
                tenant_id=tenant_id,
                policy_id=policy_id,
                observation=observation,
                selection_mode=selection_mode,
                execution_mode=execution_mode,
            )
            for observation in observations
        ]

        return {
            "tenant_id": tenant_id,
            "policy_id": policy_id,
            "batch_size": len(results),
            "results": results,
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # POLICY SUMMARY
    # --------------------------------------------------------

    def policy_summary(
        self,
        *,
        tenant_id: str,
        policy_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            profile = self._get_profile(
                tenant_id,
                policy_id,
            )

            network = self._networks[
                (tenant_id, policy_id)
            ]

            parameter_count = sum(
                parameter.numel()
                for parameter in network.parameters()
            )

            trainable_count = sum(
                parameter.numel()
                for parameter in network.parameters()
                if parameter.requires_grad
            )

            return {
                **profile.to_dict(),
                "total_parameters": parameter_count,
                "trainable_parameters": trainable_count,
                "device": profile.config.device,
                "normalizer": self._normalizers[
                    (tenant_id, policy_id)
                ].to_dict(),
            }

    # --------------------------------------------------------
    # LIST POLICIES
    # --------------------------------------------------------

    def list_policies(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        with self._lock:
            return [
                profile.to_dict()
                for (tenant, _), profile
                in self._policies.items()
                if (
                    tenant == tenant_id
                    and (
                        robot_id is None
                        or profile.robot_id == robot_id
                    )
                )
            ]

    # --------------------------------------------------------
    # ARCHIVE POLICY
    # --------------------------------------------------------

    def archive_policy(
        self,
        *,
        tenant_id: str,
        policy_id: str,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PolicyPermissionDenied(
                "Authorized archive operation required"
            )

        with self._lock:
            profile = self._get_profile(
                tenant_id,
                policy_id,
            )

            profile.status = PolicyStatus.ARCHIVED
            profile.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                action="policy_archived",
                entity_id=policy_id,
                actor_id=actor_id,
            )

            return profile.to_dict()

    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    def save_checkpoint(
        self,
        *,
        tenant_id: str,
        policy_id: str,
        directory: str | Path,
        authorized: bool = False,
    ) -> Path:
        if authorized is not True:
            raise PolicyPermissionDenied(
                "Authorized checkpoint export required"
            )

        with self._lock:
            profile = self._get_profile(
                tenant_id,
                policy_id,
            )

            key = (tenant_id, policy_id)

            output_directory = Path(directory)

            output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            output = output_directory / (
                f"{tenant_id}_{policy_id}.pt"
            )

            checkpoint = {
                "format_version": 1,
                "tenant_id": tenant_id,
                "policy_id": policy_id,
                "robot_id": profile.robot_id,
                "policy_type": (
                    profile.config.policy_type.value
                ),
                "observation_dimension": (
                    profile.config.observation_dimension
                ),
                "action_dimension": (
                    profile.config.action_dimension
                ),
                "hidden_dimensions": (
                    profile.config.hidden_dimensions
                ),
                "model_state_dict": (
                    self._networks[key].state_dict()
                ),
                "normalizer_mean": (
                    torch.from_numpy(
                        self._normalizers[key].mean.copy()
                    )
                ),
                "normalizer_std": (
                    torch.from_numpy(
                        self._normalizers[key].std.copy()
                    )
                ),
                "profile": profile.to_dict(),
                "simulation_only": True,
            }

            with tempfile.NamedTemporaryFile(
                dir=output_directory,
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(
                    temporary.name
                )

            try:
                torch.save(
                    checkpoint,
                    temporary_path,
                )

                os.replace(
                    temporary_path,
                    output,
                )

            finally:
                temporary_path.unlink(
                    missing_ok=True
                )

            self._log(
                tenant_id=tenant_id,
                action="policy_checkpoint_saved",
                entity_id=policy_id,
                details={
                    "path": str(output),
                },
            )

            return output

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    def load_checkpoint(
        self,
        *,
        tenant_id: str,
        policy_id: str,
        checkpoint_path: str | Path,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PolicyPermissionDenied(
                "Authorized checkpoint loading required"
            )

        path = Path(checkpoint_path)

        if not path.is_file():
            raise PolicyNotFound(
                "Checkpoint not found"
            )

        with self._lock:
            profile = self._get_profile(
                tenant_id,
                policy_id,
            )

            if profile.status in (
                PolicyStatus.APPROVED_FOR_SIMULATION,
                PolicyStatus.ARCHIVED,
            ):
                raise PolicyConflict(
                    "Cannot replace approved or archived policy"
                )

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            if not isinstance(checkpoint, dict):
                raise PolicyValidationError(
                    "Invalid checkpoint format"
                )

            config = profile.config

            if (
                checkpoint.get("tenant_id") != tenant_id
                or checkpoint.get("policy_id") != policy_id
                or checkpoint.get("robot_id")
                != profile.robot_id
            ):
                raise PolicyPermissionDenied(
                    "Checkpoint identity mismatch"
                )

            if (
                checkpoint.get("policy_type")
                != config.policy_type.value
                or checkpoint.get("observation_dimension")
                != config.observation_dimension
                or checkpoint.get("action_dimension")
                != config.action_dimension
                or tuple(
                    checkpoint.get("hidden_dimensions", ())
                ) != config.hidden_dimensions
            ):
                raise PolicyValidationError(
                    "Checkpoint architecture mismatch"
                )

            key = (tenant_id, policy_id)

            candidate = copy.deepcopy(
                self._networks[key]
            )

            candidate.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )

            for parameter in candidate.parameters():
                if not torch.isfinite(parameter).all():
                    raise PolicyValidationError(
                        "Non-finite checkpoint weights"
                    )

            mean = checkpoint[
                "normalizer_mean"
            ].cpu().numpy().astype(np.float32)

            std = checkpoint[
                "normalizer_std"
            ].cpu().numpy().astype(np.float32)

            expected_shape = (
                config.observation_dimension,
            )

            if (
                mean.shape != expected_shape
                or std.shape != expected_shape
                or not np.isfinite(mean).all()
                or not np.isfinite(std).all()
                or np.any(std <= 0)
            ):
                raise PolicyValidationError(
                    "Invalid checkpoint normalizer"
                )

            candidate.eval()

            self._networks[key] = candidate

            self._normalizers[key] = (
                PolicyObservationNormalizer(
                    mean=mean,
                    std=std,
                )
            )

            profile.status = PolicyStatus.TRAINED
            profile.reviewer_id = None
            profile.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                action="policy_checkpoint_loaded",
                entity_id=policy_id,
                details={
                    "path": str(path),
                },
            )

            return profile.to_dict()

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
        ):
            raise PolicyValidationError(
                "Limit must be positive"
            )

        with self._lock:
            return [
                copy.deepcopy(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ][-limit:]

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
            policies = [
                self.policy_summary(
                    tenant_id=tenant_id,
                    policy_id=policy_id,
                )
                for (tenant, policy_id)
                in self._policies
                if tenant == tenant_id
            ]

            audit = [
                copy.deepcopy(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ]

        payload = {
            "generated_at": utc_now().isoformat(),
            "tenant_id": tenant_id,
            "policies": policies,
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
# DEMONSTRATION
# ============================================================

def main() -> None:
    engine = RoboticsPolicyEngine()

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"

    # --------------------------------------------------------
    # DISCRETE POLICY
    # --------------------------------------------------------

    discrete_config = PolicyConfig(
        observation_dimension=4,
        action_dimension=4,
        policy_type=(
            PolicyType.DISCRETE_ACTOR_CRITIC
        ),
        hidden_dimensions=(128, 128),
        device="cpu",
    )

    discrete_profile = PolicyProfile(
        policy_id="navigation_policy_v1",
        tenant_id=tenant_id,
        robot_id=robot_id,
        name="Warehouse Navigation Policy",
        version="1.0.0",
        config=discrete_config,
    )

    print("\nREGISTER DISCRETE POLICY")

    print(
        json.dumps(
            engine.register_policy(
                discrete_profile,
                authorized=True,
                actor_id="training_admin",
            ),
            indent=2,
        )
    )

    print("\nDISCRETE ACTION")

    result = engine.select_action(
        tenant_id=tenant_id,
        policy_id="navigation_policy_v1",
        observation=(1.0, 2.0, 8.0, 9.0),
        selection_mode=(
            ActionSelectionMode.DETERMINISTIC
        ),
        execution_mode=(
            ExecutionMode.OFFLINE_ANALYSIS
        ),
        action_mask=(
            True,
            True,
            False,
            True,
        ),
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # CONTINUOUS POLICY
    # --------------------------------------------------------

    continuous_config = PolicyConfig(
        observation_dimension=6,
        action_dimension=2,
        policy_type=(
            PolicyType.CONTINUOUS_ACTOR_CRITIC
        ),
        hidden_dimensions=(128, 128),
        device="cpu",
    )

    continuous_profile = PolicyProfile(
        policy_id="mobile_robot_velocity_v1",
        tenant_id=tenant_id,
        robot_id=robot_id,
        name="Mobile Robot Velocity Policy",
        version="1.0.0",
        config=continuous_config,
    )

    print("\nREGISTER CONTINUOUS POLICY")

    engine.register_policy(
        continuous_profile,
        authorized=True,
        actor_id="training_admin",
    )

    print("\nCONTINUOUS ACTION")

    continuous_result = engine.select_action(
        tenant_id=tenant_id,
        policy_id="mobile_robot_velocity_v1",
        observation=(
            1.0,
            2.0,
            0.0,
            0.5,
            8.0,
            9.0,
        ),
        selection_mode=(
            ActionSelectionMode.DETERMINISTIC
        ),
        action_bounds=(
            (-1.0, -0.5),
            (1.0, 0.5),
        ),
    )

    print(
        json.dumps(
            continuous_result,
            indent=2,
        )
    )

    print("\nPOLICY SUMMARY")

    print(
        json.dumps(
            engine.policy_summary(
                tenant_id=tenant_id,
                policy_id="navigation_policy_v1",
            ),
            indent=2,
        )
    )

    print("\nSAVE CHECKPOINT")

    checkpoint = engine.save_checkpoint(
        tenant_id=tenant_id,
        policy_id="navigation_policy_v1",
        directory="data/robotics/checkpoints",
        authorized=True,
    )

    print(checkpoint)

    print("\nEXPORT REPORT")

    output = engine.export_json(
        "data/robotics/policy_report.json",
        tenant_id=tenant_id,
    )

    print(output)


if __name__ == "__main__":
    main()
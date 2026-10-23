"""Domain Randomization module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/domain_randomization.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Domain Randomization Engine

Python 3.11+
Dependencies: Standard library only

Features:
    - Physics parameter randomization
    - Robot mass, friction, damping and inertia randomization
    - Motor and actuator uncertainty
    - Camera, LiDAR and IMU sensor randomization
    - Sensor noise, latency, bias and dropout
    - Lighting, textures and environmental variation
    - Obstacles and terrain variation
    - Domain randomization difficulty levels
    - Uniform, normal, log-uniform and discrete sampling
    - Correlated parameter groups
    - Progressive curriculum randomization
    - Reproducible seeded scenarios
    - Parameter bounds and constraints
    - Batch scenario generation
    - Simulation configuration adapters
    - Scenario evaluation metadata
    - Training experiment tracking
    - Human-reviewed configuration changes
    - Tenant isolation
    - Thread-safe storage
    - Audit logging
    - JSON export

Integrations:
    curriculum.py
    checkpoints.py
    environment_generator.py
    scenario_manager.py
    physics_adapter.py
    sensor_simulation.py
    obstacle_generator.py
    simulation_adapter.py
    simulator.py
    sim_to_real.py
    synthetic_data.py
    evaluation_service.py

Safety:
    Simulation-only parameter generation.

    This module does not modify physical robot hardware,
    calibrate real sensors, change motor controllers,
    execute robot commands or authorize deployment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import threading
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class DomainRandomizationError(Exception):
    pass


class RandomizationValidationError(DomainRandomizationError):
    pass


class RandomizationNotFound(DomainRandomizationError):
    pass


class RandomizationConflict(DomainRandomizationError):
    pass


class RandomizationPermissionDenied(DomainRandomizationError):
    pass


# ============================================================
# ENUMS
# ============================================================

class DistributionType(str, Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOG_UNIFORM = "log_uniform"
    DISCRETE = "discrete"
    CONSTANT = "constant"


class ParameterCategory(str, Enum):
    PHYSICS = "physics"
    ACTUATOR = "actuator"
    SENSOR = "sensor"
    CAMERA = "camera"
    LIDAR = "lidar"
    IMU = "imu"
    ENVIRONMENT = "environment"
    LIGHTING = "lighting"
    TERRAIN = "terrain"
    OBSTACLE = "obstacle"
    MATERIAL = "material"
    NETWORK = "network"
    CUSTOM = "custom"


class RandomizationDifficulty(str, Enum):
    BEGINNER = "beginner"
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    EXPERT = "expert"


class ProfileStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class SamplingStrategy(str, Enum):
    INDEPENDENT = "independent"
    CORRELATED = "correlated"


class ConstraintOperator(str, Enum):
    LESS_EQUAL = "less_equal"
    GREATER_EQUAL = "greater_equal"
    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than"


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
        raise RandomizationValidationError(
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
        raise RandomizationValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if not allow_empty and not result:
        raise RandomizationValidationError(
            f"{name} cannot be empty"
        )

    if len(result) > maximum:
        raise RandomizationValidationError(
            f"{name} exceeds maximum length"
        )

    return result


def finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise RandomizationValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise RandomizationValidationError(
            f"Invalid {name}"
        ) from exc

    if not math.isfinite(result):
        raise RandomizationValidationError(
            f"{name} must be finite"
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
        raise RandomizationValidationError(
            f"Invalid {name}"
        )

    return value


def validate_probability(
    value: Any,
    name: str,
) -> float:
    result = finite_number(value, name)

    if not 0 <= result <= 1:
        raise RandomizationValidationError(
            f"{name} must be between 0 and 1"
        )

    return result


def validate_json_mapping(
    value: Mapping[str, Any],
    name: str,
    *,
    maximum_bytes: int = 1048576,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RandomizationValidationError(
            f"{name} must be a mapping"
        )

    if not all(isinstance(k, str) for k in value):
        raise RandomizationValidationError(
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
        raise RandomizationValidationError(
            f"{name} must contain JSON-compatible data"
        ) from exc

    if len(encoded.encode("utf-8")) > maximum_bytes:
        raise RandomizationValidationError(
            f"{name} exceeds maximum size"
        )

    return result


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_seed(*parts: Any) -> int:
    payload = canonical_json(parts)

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DomainRandomizationConfig:
    maximum_profiles: int = 10000
    maximum_parameters_per_profile: int = 200
    maximum_constraints_per_profile: int = 100

    maximum_scenarios: int = 100000
    maximum_batch_size: int = 1000
    maximum_sampling_attempts: int = 1000

    maximum_audit_events: int = 20000

    default_seed: int = 42

    simulation_only: bool = True
    require_authorization: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_profiles",
            "maximum_parameters_per_profile",
            "maximum_constraints_per_profile",
            "maximum_scenarios",
            "maximum_batch_size",
            "maximum_sampling_attempts",
            "maximum_audit_events",
        ):
            validate_integer(
                getattr(self, name),
                name,
                minimum=1,
            )

        validate_integer(
            self.default_seed,
            "default_seed",
            maximum=2**63 - 1,
        )

        if self.simulation_only is not True:
            raise RandomizationValidationError(
                "Only simulation mode is supported"
            )

        if self.require_authorization is not True:
            raise RandomizationValidationError(
                "Authorization cannot be disabled"
            )


# ============================================================
# RANDOMIZATION PARAMETER
# ============================================================

@dataclass(frozen=True)
class RandomizationParameter:
    name: str
    category: ParameterCategory

    distribution: DistributionType

    minimum: float = 0.0
    maximum: float = 1.0

    mean: float | None = None
    standard_deviation: float | None = None

    discrete_values: tuple[
        float | int | str | bool,
        ...
    ] = ()

    constant_value: (
        float | int | str | bool | None
    ) = None

    nominal_value: float | None = None

    enabled: bool = True

    unit: str = ""

    description: str = ""

    def __post_init__(self) -> None:
        validate_id(self.name, "name")

        if not isinstance(
            self.category,
            ParameterCategory,
        ):
            raise RandomizationValidationError(
                "Invalid parameter category"
            )

        if not isinstance(
            self.distribution,
            DistributionType,
        ):
            raise RandomizationValidationError(
                "Invalid distribution"
            )

        minimum = finite_number(
            self.minimum,
            "minimum",
        )

        maximum = finite_number(
            self.maximum,
            "maximum",
        )

        if minimum > maximum:
            raise RandomizationValidationError(
                "Minimum exceeds maximum"
            )

        if self.distribution == DistributionType.LOG_UNIFORM:
            if minimum <= 0:
                raise RandomizationValidationError(
                    "Log-uniform minimum must be positive"
                )

        if self.distribution == DistributionType.NORMAL:
            if self.mean is None:
                raise RandomizationValidationError(
                    "Normal distribution requires mean"
                )

            if self.standard_deviation is None:
                raise RandomizationValidationError(
                    "Normal distribution requires "
                    "standard deviation"
                )

            finite_number(self.mean, "mean")

            std = finite_number(
                self.standard_deviation,
                "standard_deviation",
            )

            if std <= 0:
                raise RandomizationValidationError(
                    "Standard deviation must be positive"
                )

        if self.distribution == DistributionType.DISCRETE:
            if not self.discrete_values:
                raise RandomizationValidationError(
                    "Discrete distribution requires values"
                )

            for value in self.discrete_values:
                if not isinstance(
                    value,
                    (str, int, float, bool),
                ):
                    raise RandomizationValidationError(
                        "Unsupported discrete value"
                    )

                if isinstance(value, float):
                    finite_number(
                        value,
                        "discrete value",
                    )

        if self.distribution == DistributionType.CONSTANT:
            if self.constant_value is None:
                raise RandomizationValidationError(
                    "Constant distribution requires a value"
                )

            if not isinstance(
                self.constant_value,
                (str, int, float, bool),
            ):
                raise RandomizationValidationError(
                    "Unsupported constant value"
                )

            if isinstance(self.constant_value, float):
                finite_number(
                    self.constant_value,
                    "constant_value",
                )

        if self.nominal_value is not None:
            finite_number(
                self.nominal_value,
                "nominal_value",
            )

        if not isinstance(self.enabled, bool):
            raise RandomizationValidationError(
                "enabled must be boolean"
            )

        validate_text(
            self.unit,
            "unit",
            allow_empty=True,
            maximum=64,
        )

        validate_text(
            self.description,
            "description",
            allow_empty=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "distribution": self.distribution.value,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "discrete_values": list(
                self.discrete_values
            ),
            "constant_value": self.constant_value,
            "nominal_value": self.nominal_value,
            "enabled": self.enabled,
            "unit": self.unit,
            "description": self.description,
        }


# ============================================================
# PARAMETER CONSTRAINT
# ============================================================

@dataclass(frozen=True)
class ParameterConstraint:
    left_parameter: str
    operator: ConstraintOperator

    right_parameter: str | None = None
    right_constant: float | None = None

    offset: float = 0.0

    def __post_init__(self) -> None:
        validate_id(
            self.left_parameter,
            "left_parameter",
        )

        if not isinstance(
            self.operator,
            ConstraintOperator,
        ):
            raise RandomizationValidationError(
                "Invalid constraint operator"
            )

        if (
            self.right_parameter is None
        ) == (
            self.right_constant is None
        ):
            raise RandomizationValidationError(
                "Specify exactly one right operand"
            )

        if self.right_parameter is not None:
            validate_id(
                self.right_parameter,
                "right_parameter",
            )

        if self.right_constant is not None:
            finite_number(
                self.right_constant,
                "right_constant",
            )

        finite_number(self.offset, "offset")

    def evaluate(
        self,
        values: Mapping[str, Any],
    ) -> bool:
        if self.left_parameter not in values:
            return False

        left = finite_number(
            values[self.left_parameter],
            self.left_parameter,
        )

        if self.right_parameter is not None:
            if self.right_parameter not in values:
                return False

            right = finite_number(
                values[self.right_parameter],
                self.right_parameter,
            )

        else:
            right = float(self.right_constant)

        right += self.offset

        if self.operator == ConstraintOperator.LESS_EQUAL:
            return left <= right

        if self.operator == ConstraintOperator.GREATER_EQUAL:
            return left >= right

        if self.operator == ConstraintOperator.LESS_THAN:
            return left < right

        if self.operator == ConstraintOperator.GREATER_THAN:
            return left > right

        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_parameter": self.left_parameter,
            "operator": self.operator.value,
            "right_parameter": self.right_parameter,
            "right_constant": self.right_constant,
            "offset": self.offset,
        }


# ============================================================
# CORRELATED PARAMETER GROUP
# ============================================================

@dataclass(frozen=True)
class CorrelatedParameterGroup:
    group_id: str

    parameter_names: tuple[str, ...]

    correlation_strength: float = 0.8

    def __post_init__(self) -> None:
        validate_id(self.group_id, "group_id")

        if len(self.parameter_names) < 2:
            raise RandomizationValidationError(
                "Correlation group requires two parameters"
            )

        if (
            len(set(self.parameter_names))
            != len(self.parameter_names)
        ):
            raise RandomizationValidationError(
                "Duplicate correlated parameters"
            )

        for name in self.parameter_names:
            validate_id(name, "parameter_name")

        strength = finite_number(
            self.correlation_strength,
            "correlation_strength",
        )

        if not 0 <= strength <= 1:
            raise RandomizationValidationError(
                "Correlation strength must be in [0, 1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "parameter_names": list(
                self.parameter_names
            ),
            "correlation_strength": (
                self.correlation_strength
            ),
        }


# ============================================================
# RANDOMIZATION PROFILE
# ============================================================

@dataclass(frozen=True)
class DomainRandomizationProfile:
    profile_id: str

    tenant_id: str
    robot_id: str

    name: str

    parameters: tuple[
        RandomizationParameter,
        ...
    ]

    difficulty: RandomizationDifficulty = (
        RandomizationDifficulty.MODERATE
    )

    sampling_strategy: SamplingStrategy = (
        SamplingStrategy.INDEPENDENT
    )

    correlated_groups: tuple[
        CorrelatedParameterGroup,
        ...
    ] = ()

    constraints: tuple[
        ParameterConstraint,
        ...
    ] = ()

    base_seed: int = 42

    curriculum_stage_id: str | None = None

    description: str = ""

    version: int = 1

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "tenant_id",
            "robot_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        validate_text(self.name, "name")

        if not self.parameters:
            raise RandomizationValidationError(
                "Profile requires parameters"
            )

        parameter_names = set()

        for parameter in self.parameters:
            if not isinstance(
                parameter,
                RandomizationParameter,
            ):
                raise RandomizationValidationError(
                    "Invalid parameter"
                )

            if parameter.name in parameter_names:
                raise RandomizationValidationError(
                    "Duplicate parameter name"
                )

            parameter_names.add(parameter.name)

        if not isinstance(
            self.difficulty,
            RandomizationDifficulty,
        ):
            raise RandomizationValidationError(
                "Invalid difficulty"
            )

        if not isinstance(
            self.sampling_strategy,
            SamplingStrategy,
        ):
            raise RandomizationValidationError(
                "Invalid sampling strategy"
            )

        used_group_parameters = set()

        for group in self.correlated_groups:
            if not isinstance(
                group,
                CorrelatedParameterGroup,
            ):
                raise RandomizationValidationError(
                    "Invalid correlated group"
                )

            for name in group.parameter_names:
                if name not in parameter_names:
                    raise RandomizationValidationError(
                        "Unknown correlated parameter"
                    )

                if name in used_group_parameters:
                    raise RandomizationValidationError(
                        "Parameter belongs to multiple "
                        "correlation groups"
                    )

                used_group_parameters.add(name)

        if (
            self.correlated_groups
            and self.sampling_strategy
            != SamplingStrategy.CORRELATED
        ):
            raise RandomizationValidationError(
                "Correlated groups require correlated strategy"
            )

        for constraint in self.constraints:
            if not isinstance(
                constraint,
                ParameterConstraint,
            ):
                raise RandomizationValidationError(
                    "Invalid constraint"
                )

            if constraint.left_parameter not in parameter_names:
                raise RandomizationValidationError(
                    "Unknown constraint parameter"
                )

            if (
                constraint.right_parameter is not None
                and constraint.right_parameter
                not in parameter_names
            ):
                raise RandomizationValidationError(
                    "Unknown right constraint parameter"
                )

        validate_integer(
            self.base_seed,
            "base_seed",
            maximum=2**63 - 1,
        )

        if self.curriculum_stage_id is not None:
            validate_id(
                self.curriculum_stage_id,
                "curriculum_stage_id",
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

        if (
            self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise RandomizationValidationError(
                "created_at must be timezone-aware"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "name": self.name,
            "parameters": [
                parameter.to_dict()
                for parameter in self.parameters
            ],
            "difficulty": self.difficulty.value,
            "sampling_strategy": (
                self.sampling_strategy.value
            ),
            "correlated_groups": [
                group.to_dict()
                for group in self.correlated_groups
            ],
            "constraints": [
                constraint.to_dict()
                for constraint in self.constraints
            ],
            "base_seed": self.base_seed,
            "curriculum_stage_id": (
                self.curriculum_stage_id
            ),
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "simulation_only": True,
        }


# ============================================================
# GENERATED SCENARIO
# ============================================================

@dataclass(frozen=True)
class RandomizedScenario:
    scenario_id: str

    tenant_id: str
    robot_id: str

    profile_id: str
    profile_version: int

    seed: int

    difficulty: RandomizationDifficulty

    parameters: Mapping[str, Any]

    curriculum_stage_id: str | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "seed": self.seed,
            "difficulty": self.difficulty.value,
            "parameters": copy.deepcopy(
                dict(self.parameters)
            ),
            "curriculum_stage_id": (
                self.curriculum_stage_id
            ),
            "created_at": self.created_at.isoformat(),
            "simulation_only": True,
            "physical_execution_authorized": False,
        }


# ============================================================
# SAMPLING ENGINE
# ============================================================

class ParameterSampler:

    @staticmethod
    def sample(
        parameter: RandomizationParameter,
        rng: random.Random,
        *,
        difficulty_factor: float = 1.0,
        shared_quantile: float | None = None,
        correlation_strength: float = 0.0,
    ) -> Any:
        if not parameter.enabled:
            if parameter.nominal_value is not None:
                return parameter.nominal_value

            if parameter.distribution == DistributionType.CONSTANT:
                return parameter.constant_value

            if parameter.distribution == DistributionType.DISCRETE:
                return parameter.discrete_values[0]

            return (
                parameter.minimum + parameter.maximum
            ) / 2

        if parameter.distribution == DistributionType.CONSTANT:
            return parameter.constant_value

        if parameter.distribution == DistributionType.DISCRETE:
            values = parameter.discrete_values

            if shared_quantile is None:
                return rng.choice(values)

            local_quantile = rng.random()

            quantile = (
                correlation_strength * shared_quantile
                + (1 - correlation_strength)
                * local_quantile
            )

            index = min(
                len(values) - 1,
                int(quantile * len(values)),
            )

            return values[index]

        minimum = parameter.minimum
        maximum = parameter.maximum

        factor = validate_probability(
            difficulty_factor,
            "difficulty_factor",
        )

        nominal = (
            parameter.nominal_value
            if parameter.nominal_value is not None
            else (minimum + maximum) / 2
        )

        nominal = max(
            minimum,
            min(maximum, nominal),
        )

        low = nominal - (
            nominal - minimum
        ) * factor

        high = nominal + (
            maximum - nominal
        ) * factor

        if parameter.distribution == DistributionType.UNIFORM:
            quantile = (
                rng.random()
                if shared_quantile is None
                else (
                    correlation_strength * shared_quantile
                    + (1 - correlation_strength)
                    * rng.random()
                )
            )

            return low + (high - low) * quantile

        if parameter.distribution == DistributionType.LOG_UNIFORM:
            if low <= 0 or high <= 0:
                raise RandomizationValidationError(
                    "Invalid log-uniform bounds"
                )

            quantile = (
                rng.random()
                if shared_quantile is None
                else (
                    correlation_strength * shared_quantile
                    + (1 - correlation_strength)
                    * rng.random()
                )
            )

            return math.exp(
                math.log(low)
                + (
                    math.log(high) - math.log(low)
                ) * quantile
            )

        if parameter.distribution == DistributionType.NORMAL:
            mean = float(parameter.mean)

            std = (
                float(parameter.standard_deviation)
                * max(factor, 1e-9)
            )

            if shared_quantile is None:
                value = rng.gauss(mean, std)

            else:
                # Approximate positive correlation through
                # a shared standard-normal latent variable.
                # shared_quantile is converted into a
                # normal variate by a bounded inverse CDF.
                from statistics import NormalDist

                normal = NormalDist()

                q = max(
                    1e-12,
                    min(1 - 1e-12, shared_quantile),
                )

                common = normal.inv_cdf(q)
                independent = rng.gauss(0, 1)

                rho = correlation_strength

                latent = (
                    math.sqrt(rho) * common
                    + math.sqrt(1 - rho) * independent
                )

                value = mean + std * latent

            return max(
                low,
                min(high, value),
            )

        raise RandomizationValidationError(
            "Unsupported distribution"
        )


# ============================================================
# DOMAIN RANDOMIZATION ENGINE
# ============================================================

class RoboticsDomainRandomizationEngine:

    DIFFICULTY_FACTORS = {
        RandomizationDifficulty.BEGINNER: 0.10,
        RandomizationDifficulty.EASY: 0.25,
        RandomizationDifficulty.MODERATE: 0.50,
        RandomizationDifficulty.HARD: 0.75,
        RandomizationDifficulty.EXPERT: 1.00,
    }

    def __init__(
        self,
        config: DomainRandomizationConfig | None = None,
    ) -> None:
        self.config = (
            config or DomainRandomizationConfig()
        )

        self._profiles: dict[
            tuple[str, str],
            DomainRandomizationProfile,
        ] = {}

        self._profile_statuses: dict[
            tuple[str, str],
            ProfileStatus,
        ] = {}

        self._scenarios: dict[
            tuple[str, str],
            RandomizedScenario,
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
                "details": copy.deepcopy(
                    dict(details or {})
                ),
            }
        )

    # --------------------------------------------------------
    # REGISTER PROFILE
    # --------------------------------------------------------

    def register_profile(
        self,
        profile: DomainRandomizationProfile,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise RandomizationPermissionDenied(
                "Authorized profile registration required"
            )

        if not isinstance(
            profile,
            DomainRandomizationProfile,
        ):
            raise RandomizationValidationError(
                "Invalid profile"
            )

        validate_id(actor_id, "actor_id")

        if (
            len(profile.parameters)
            > self.config.maximum_parameters_per_profile
        ):
            raise RandomizationValidationError(
                "Too many parameters"
            )

        if (
            len(profile.constraints)
            > self.config.maximum_constraints_per_profile
        ):
            raise RandomizationValidationError(
                "Too many constraints"
            )

        key = (
            profile.tenant_id,
            profile.profile_id,
        )

        with self._lock:
            if key in self._profiles:
                raise RandomizationConflict(
                    "Profile already exists"
                )

            if (
                len(self._profiles)
                >= self.config.maximum_profiles
            ):
                raise RandomizationConflict(
                    "Profile capacity reached"
                )

            self._profiles[key] = profile

            self._profile_statuses[
                key
            ] = ProfileStatus.DRAFT

            self._log(
                tenant_id=profile.tenant_id,
                action="profile_registered",
                entity_id=profile.profile_id,
                actor_id=actor_id,
                details={
                    "robot_id": profile.robot_id,
                    "parameter_count": len(
                        profile.parameters
                    ),
                },
            )

            return {
                **profile.to_dict(),
                "status": ProfileStatus.DRAFT.value,
            }

    # --------------------------------------------------------
    # ACTIVATE PROFILE
    # --------------------------------------------------------

    def activate_profile(
        self,
        *,
        tenant_id: str,
        profile_id: str,
        authorized: bool = False,
        reviewer_id: str | None = None,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise RandomizationPermissionDenied(
                "Authorized activation required"
            )

        if reviewer_id is None:
            raise RandomizationPermissionDenied(
                "Human reviewer required"
            )

        validate_id(reviewer_id, "reviewer_id")

        with self._lock:
            key = (tenant_id, profile_id)

            profile = self._get_profile(
                tenant_id,
                profile_id,
            )

            if (
                self._profile_statuses[key]
                != ProfileStatus.DRAFT
            ):
                raise RandomizationConflict(
                    "Only draft profiles can activate"
                )

            self._profile_statuses[
                key
            ] = ProfileStatus.ACTIVE

            self._log(
                tenant_id=tenant_id,
                action="profile_activated",
                entity_id=profile_id,
                actor_id=reviewer_id,
            )

            return {
                "profile_id": profile.profile_id,
                "status": ProfileStatus.ACTIVE.value,
                "reviewed_by": reviewer_id,
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # GET PROFILE
    # --------------------------------------------------------

    def _get_profile(
        self,
        tenant_id: str,
        profile_id: str,
    ) -> DomainRandomizationProfile:
        validate_id(tenant_id, "tenant_id")
        validate_id(profile_id, "profile_id")

        profile = self._profiles.get(
            (tenant_id, profile_id)
        )

        if profile is None:
            raise RandomizationNotFound(
                "Randomization profile not found"
            )

        return profile

    def get_profile(
        self,
        *,
        tenant_id: str,
        profile_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            profile = self._get_profile(
                tenant_id,
                profile_id,
            )

            return {
                **profile.to_dict(),
                "status": self._profile_statuses[
                    (tenant_id, profile_id)
                ].value,
            }

    # --------------------------------------------------------
    # LIST PROFILES
    # --------------------------------------------------------

    def list_profiles(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            profiles = [
                {
                    **profile.to_dict(),
                    "status": self._profile_statuses[
                        (tenant, profile_id)
                    ].value,
                }
                for (tenant, profile_id), profile
                in self._profiles.items()
                if (
                    tenant == tenant_id
                    and (
                        robot_id is None
                        or profile.robot_id == robot_id
                    )
                )
            ]

            return profiles[:limit]

    # --------------------------------------------------------
    # SAMPLE PARAMETER VALUES
    # --------------------------------------------------------

    def _sample_values(
        self,
        profile: DomainRandomizationProfile,
        rng: random.Random,
        difficulty: RandomizationDifficulty,
    ) -> dict[str, Any]:
        factor = self.DIFFICULTY_FACTORS[
            difficulty
        ]

        group_map: dict[
            str,
            tuple[float, float],
        ] = {}

        if (
            profile.sampling_strategy
            == SamplingStrategy.CORRELATED
        ):
            for group in profile.correlated_groups:
                shared_quantile = rng.random()

                for name in group.parameter_names:
                    group_map[name] = (
                        shared_quantile,
                        group.correlation_strength,
                    )

        values = {}

        for parameter in profile.parameters:
            group = group_map.get(
                parameter.name
            )

            if group is None:
                shared_quantile = None
                strength = 0.0

            else:
                shared_quantile, strength = group

            values[parameter.name] = (
                ParameterSampler.sample(
                    parameter,
                    rng,
                    difficulty_factor=factor,
                    shared_quantile=shared_quantile,
                    correlation_strength=strength,
                )
            )

        return values

    # --------------------------------------------------------
    # VALIDATE CONSTRAINTS
    # --------------------------------------------------------

    def _constraints_satisfied(
        self,
        profile: DomainRandomizationProfile,
        values: Mapping[str, Any],
    ) -> bool:
        return all(
            constraint.evaluate(values)
            for constraint in profile.constraints
        )

    # --------------------------------------------------------
    # GENERATE SCENARIO
    # --------------------------------------------------------

    def generate_scenario(
        self,
        *,
        tenant_id: str,
        profile_id: str,
        scenario_id: str | None = None,
        seed: int | None = None,
        difficulty: RandomizationDifficulty | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            profile = self._get_profile(
                tenant_id,
                profile_id,
            )

            if (
                self._profile_statuses[
                    (tenant_id, profile_id)
                ] != ProfileStatus.ACTIVE
            ):
                raise RandomizationConflict(
                    "Profile must be active"
                )

            if (
                len(self._scenarios)
                >= self.config.maximum_scenarios
            ):
                raise RandomizationConflict(
                    "Scenario capacity reached"
                )

            chosen_difficulty = (
                difficulty or profile.difficulty
            )

            if not isinstance(
                chosen_difficulty,
                RandomizationDifficulty,
            ):
                raise RandomizationValidationError(
                    "Invalid difficulty"
                )

            chosen_id = scenario_id or new_id()

            validate_id(chosen_id, "scenario_id")

            key = (tenant_id, chosen_id)

            if key in self._scenarios:
                raise RandomizationConflict(
                    "Scenario already exists"
                )

            if seed is None:
                chosen_seed = stable_seed(
                    tenant_id,
                    profile_id,
                    profile.version,
                    profile.base_seed,
                    chosen_id,
                )

            else:
                chosen_seed = validate_integer(
                    seed,
                    "seed",
                    maximum=2**64 - 1,
                )

            rng = random.Random(chosen_seed)

            sampled_values = None

            for _ in range(
                self.config.maximum_sampling_attempts
            ):
                candidate = self._sample_values(
                    profile,
                    rng,
                    chosen_difficulty,
                )

                if self._constraints_satisfied(
                    profile,
                    candidate,
                ):
                    sampled_values = candidate
                    break

            if sampled_values is None:
                raise RandomizationConflict(
                    "Unable to satisfy parameter constraints"
                )

            scenario = RandomizedScenario(
                scenario_id=chosen_id,
                tenant_id=tenant_id,
                robot_id=profile.robot_id,
                profile_id=profile_id,
                profile_version=profile.version,
                seed=chosen_seed,
                difficulty=chosen_difficulty,
                parameters=sampled_values,
                curriculum_stage_id=(
                    profile.curriculum_stage_id
                ),
            )

            self._scenarios[key] = scenario

            self._log(
                tenant_id=tenant_id,
                action="scenario_generated",
                entity_id=chosen_id,
                details={
                    "profile_id": profile_id,
                    "difficulty": chosen_difficulty.value,
                    "seed": chosen_seed,
                },
            )

            return scenario.to_dict()

    # --------------------------------------------------------
    # GENERATE BATCH
    # --------------------------------------------------------

    def generate_batch(
        self,
        *,
        tenant_id: str,
        profile_id: str,
        count: int,
        seed: int | None = None,
        difficulty: RandomizationDifficulty | None = None,
    ) -> list[dict[str, Any]]:
        validate_integer(
            count,
            "count",
            minimum=1,
            maximum=self.config.maximum_batch_size,
        )

        with self._lock:
            self._get_profile(
                tenant_id,
                profile_id,
            )

            if (
                len(self._scenarios) + count
                > self.config.maximum_scenarios
            ):
                raise RandomizationConflict(
                    "Insufficient scenario capacity"
                )

            batch_id = new_id()

            results = []

            for index in range(count):
                scenario_seed = (
                    None
                    if seed is None
                    else stable_seed(seed, index)
                )

                scenario_id = (
                    f"scenario_{batch_id}_{index:04d}"
                )

                result = self.generate_scenario(
                    tenant_id=tenant_id,
                    profile_id=profile_id,
                    scenario_id=scenario_id,
                    seed=scenario_seed,
                    difficulty=difficulty,
                )

                results.append(result)

            return results

    # --------------------------------------------------------
    # REPRODUCE SCENARIO
    # --------------------------------------------------------

    def reproduce_scenario(
        self,
        *,
        tenant_id: str,
        scenario_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            scenario = self._scenarios.get(
                (tenant_id, scenario_id)
            )

            if scenario is None:
                raise RandomizationNotFound(
                    "Scenario not found"
                )

            profile = self._get_profile(
                tenant_id,
                scenario.profile_id,
            )

            if (
                profile.version
                != scenario.profile_version
            ):
                raise RandomizationConflict(
                    "Profile version changed; exact "
                    "reproduction requires original profile"
                )

            rng = random.Random(
                scenario.seed
            )

            sampled_values = None

            for _ in range(
                self.config.maximum_sampling_attempts
            ):
                candidate = self._sample_values(
                    profile,
                    rng,
                    scenario.difficulty,
                )

                if self._constraints_satisfied(
                    profile,
                    candidate,
                ):
                    sampled_values = candidate
                    break

            reproduced = (
                sampled_values is not None
                and sampled_values
                == dict(scenario.parameters)
            )

            return {
                "scenario_id": scenario_id,
                "profile_id": scenario.profile_id,
                "seed": scenario.seed,
                "original_parameters": copy.deepcopy(
                    dict(scenario.parameters)
                ),
                "reproduced_parameters": sampled_values,
                "exact_match": reproduced,
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # GET SCENARIO
    # --------------------------------------------------------

    def get_scenario(
        self,
        *,
        tenant_id: str,
        scenario_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(scenario_id, "scenario_id")

        with self._lock:
            scenario = self._scenarios.get(
                (tenant_id, scenario_id)
            )

            if scenario is None:
                raise RandomizationNotFound(
                    "Scenario not found"
                )

            return scenario.to_dict()

    # --------------------------------------------------------
    # LIST SCENARIOS
    # --------------------------------------------------------

    def list_scenarios(
        self,
        *,
        tenant_id: str,
        profile_id: str | None = None,
        robot_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if profile_id is not None:
            validate_id(profile_id, "profile_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            scenarios = [
                scenario
                for (tenant, _), scenario
                in self._scenarios.items()
                if (
                    tenant == tenant_id
                    and (
                        profile_id is None
                        or scenario.profile_id == profile_id
                    )
                    and (
                        robot_id is None
                        or scenario.robot_id == robot_id
                    )
                )
            ]

            scenarios.sort(
                key=lambda item: item.created_at,
                reverse=True,
            )

            return [
                scenario.to_dict()
                for scenario in scenarios[:limit]
            ]

    # --------------------------------------------------------
    # CURRICULUM DIFFICULTY ADAPTATION
    # --------------------------------------------------------

    def recommend_difficulty(
        self,
        *,
        current_difficulty: RandomizationDifficulty,
        success_rate: float,
        safety_violation_rate: float,
        minimum_success_rate: float = 0.85,
    ) -> dict[str, Any]:
        if not isinstance(
            current_difficulty,
            RandomizationDifficulty,
        ):
            raise RandomizationValidationError(
                "Invalid current difficulty"
            )

        success = validate_probability(
            success_rate,
            "success_rate",
        )

        safety = validate_probability(
            safety_violation_rate,
            "safety_violation_rate",
        )

        threshold = validate_probability(
            minimum_success_rate,
            "minimum_success_rate",
        )

        levels = list(
            RandomizationDifficulty
        )

        index = levels.index(
            current_difficulty
        )

        if safety > 0:
            next_index = max(
                0,
                index - 1,
            )

            reason = (
                "Safety violations detected. "
                "Recommend lower simulation difficulty "
                "and independent review."
            )

            review_required = True

        elif success >= threshold:
            next_index = min(
                len(levels) - 1,
                index + 1,
            )

            reason = (
                "Performance threshold met. "
                "Higher simulation difficulty may be evaluated."
            )

            review_required = True

        elif success < 0.40:
            next_index = max(
                0,
                index - 1,
            )

            reason = (
                "Low success rate. "
                "Recommend easier simulation scenarios."
            )

            review_required = False

        else:
            next_index = index

            reason = (
                "Continue training at current difficulty."
            )

            review_required = False

        return {
            "current_difficulty": (
                current_difficulty.value
            ),
            "recommended_difficulty": (
                levels[next_index].value
            ),
            "success_rate": success,
            "safety_violation_rate": safety,
            "reason": reason,
            "human_review_required": review_required,
            "automatically_applied": False,
            "physical_deployment_approved": False,
        }

    # --------------------------------------------------------
    # SIMULATION ADAPTER CONFIGURATION
    # --------------------------------------------------------

    def simulation_configuration(
        self,
        *,
        tenant_id: str,
        scenario_id: str,
    ) -> dict[str, Any]:
        scenario = self.get_scenario(
            tenant_id=tenant_id,
            scenario_id=scenario_id,
        )

        profile = self._get_profile(
            tenant_id,
            scenario["profile_id"],
        )

        parameter_categories: dict[
            str,
            dict[str, Any],
        ] = {}

        for parameter in profile.parameters:
            category = parameter.category.value

            parameter_categories.setdefault(
                category,
                {},
            )

            parameter_categories[
                category
            ][parameter.name] = scenario[
                "parameters"
            ][parameter.name]

        return {
            "scenario_id": scenario_id,
            "tenant_id": tenant_id,
            "robot_id": scenario["robot_id"],
            "seed": scenario["seed"],
            "difficulty": scenario["difficulty"],
            "physics": parameter_categories.get(
                "physics",
                {},
            ),
            "actuators": parameter_categories.get(
                "actuator",
                {},
            ),
            "sensors": parameter_categories.get(
                "sensor",
                {},
            ),
            "camera": parameter_categories.get(
                "camera",
                {},
            ),
            "lidar": parameter_categories.get(
                "lidar",
                {},
            ),
            "imu": parameter_categories.get(
                "imu",
                {},
            ),
            "environment": parameter_categories.get(
                "environment",
                {},
            ),
            "lighting": parameter_categories.get(
                "lighting",
                {},
            ),
            "terrain": parameter_categories.get(
                "terrain",
                {},
            ),
            "obstacles": parameter_categories.get(
                "obstacle",
                {},
            ),
            "materials": parameter_categories.get(
                "material",
                {},
            ),
            "network": parameter_categories.get(
                "network",
                {},
            ),
            "custom": parameter_categories.get(
                "custom",
                {},
            ),
            "simulation_only": True,
            "apply_to_physical_robot": False,
        }

    # --------------------------------------------------------
    # RANDOMIZATION COVERAGE
    # --------------------------------------------------------

    def randomization_coverage(
        self,
        *,
        tenant_id: str,
        profile_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            profile = self._get_profile(
                tenant_id,
                profile_id,
            )

            scenarios = [
                scenario
                for (tenant, _), scenario
                in self._scenarios.items()
                if (
                    tenant == tenant_id
                    and scenario.profile_id == profile_id
                )
            ]

            statistics_by_parameter = {}

            for parameter in profile.parameters:
                values = [
                    scenario.parameters.get(
                        parameter.name
                    )
                    for scenario in scenarios
                ]

                numeric_values = [
                    float(value)
                    for value in values
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    )
                ]

                if numeric_values:
                    statistics_by_parameter[
                        parameter.name
                    ] = {
                        "sample_count": len(
                            numeric_values
                        ),
                        "observed_minimum": min(
                            numeric_values
                        ),
                        "observed_maximum": max(
                            numeric_values
                        ),
                        "observed_mean": (
                            sum(numeric_values)
                            / len(numeric_values)
                        ),
                        "configured_minimum": (
                            parameter.minimum
                        ),
                        "configured_maximum": (
                            parameter.maximum
                        ),
                    }

                else:
                    counts = Counter(
                        str(value)
                        for value in values
                    )

                    statistics_by_parameter[
                        parameter.name
                    ] = {
                        "sample_count": len(values),
                        "value_counts": dict(counts),
                    }

            return {
                "profile_id": profile_id,
                "tenant_id": tenant_id,
                "scenario_count": len(scenarios),
                "parameter_statistics": (
                    statistics_by_parameter
                ),
                "note": (
                    "Observed sample ranges are descriptive "
                    "and do not establish sufficient "
                    "sim-to-real coverage."
                ),
            }

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
                if event["tenant_id"] == tenant_id
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
            profiles = [
                {
                    **profile.to_dict(),
                    "status": self._profile_statuses[
                        (tenant, profile_id)
                    ].value,
                }
                for (tenant, profile_id), profile
                in self._profiles.items()
                if tenant == tenant_id
            ]

            scenarios = [
                scenario.to_dict()
                for (tenant, _), scenario
                in self._scenarios.items()
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
            "profiles": profiles,
            "scenarios": scenarios,
            "audit_events": audit,
            "simulation_only": True,
        }

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                payload,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output


# ============================================================
# DEFAULT ROBOT DOMAIN RANDOMIZATION PROFILE
# ============================================================

def create_default_robot_profile(
    *,
    tenant_id: str,
    robot_id: str,
    profile_id: str = "robot_domain_randomization",
) -> DomainRandomizationProfile:

    parameters = (
        RandomizationParameter(
            name="robot_mass_kg",
            category=ParameterCategory.PHYSICS,
            distribution=DistributionType.UNIFORM,
            minimum=18.0,
            maximum=22.0,
            nominal_value=20.0,
            unit="kg",
        ),
        RandomizationParameter(
            name="friction_coefficient",
            category=ParameterCategory.PHYSICS,
            distribution=DistributionType.UNIFORM,
            minimum=0.4,
            maximum=1.0,
            nominal_value=0.7,
        ),
        RandomizationParameter(
            name="joint_damping",
            category=ParameterCategory.PHYSICS,
            distribution=DistributionType.UNIFORM,
            minimum=0.05,
            maximum=0.30,
            nominal_value=0.15,
        ),
        RandomizationParameter(
            name="motor_efficiency",
            category=ParameterCategory.ACTUATOR,
            distribution=DistributionType.UNIFORM,
            minimum=0.75,
            maximum=0.98,
            nominal_value=0.90,
        ),
        RandomizationParameter(
            name="actuator_latency_ms",
            category=ParameterCategory.ACTUATOR,
            distribution=DistributionType.UNIFORM,
            minimum=1.0,
            maximum=40.0,
            nominal_value=10.0,
            unit="ms",
        ),
        RandomizationParameter(
            name="camera_noise_std",
            category=ParameterCategory.CAMERA,
            distribution=DistributionType.UNIFORM,
            minimum=0.0,
            maximum=0.12,
            nominal_value=0.02,
        ),
        RandomizationParameter(
            name="camera_brightness",
            category=ParameterCategory.CAMERA,
            distribution=DistributionType.UNIFORM,
            minimum=0.5,
            maximum=1.5,
            nominal_value=1.0,
        ),
        RandomizationParameter(
            name="lidar_range_noise_m",
            category=ParameterCategory.LIDAR,
            distribution=DistributionType.UNIFORM,
            minimum=0.0,
            maximum=0.15,
            nominal_value=0.02,
            unit="m",
        ),
        RandomizationParameter(
            name="lidar_dropout_rate",
            category=ParameterCategory.LIDAR,
            distribution=DistributionType.UNIFORM,
            minimum=0.0,
            maximum=0.15,
            nominal_value=0.01,
        ),
        RandomizationParameter(
            name="imu_bias",
            category=ParameterCategory.IMU,
            distribution=DistributionType.NORMAL,
            minimum=-0.05,
            maximum=0.05,
            mean=0.0,
            standard_deviation=0.01,
            nominal_value=0.0,
        ),
        RandomizationParameter(
            name="lighting_intensity",
            category=ParameterCategory.LIGHTING,
            distribution=DistributionType.UNIFORM,
            minimum=0.3,
            maximum=1.8,
            nominal_value=1.0,
        ),
        RandomizationParameter(
            name="terrain_roughness",
            category=ParameterCategory.TERRAIN,
            distribution=DistributionType.UNIFORM,
            minimum=0.0,
            maximum=0.7,
            nominal_value=0.1,
        ),
        RandomizationParameter(
            name="obstacle_count",
            category=ParameterCategory.OBSTACLE,
            distribution=DistributionType.DISCRETE,
            discrete_values=(
                0, 2, 5, 8, 12, 16, 20,
            ),
        ),
        RandomizationParameter(
            name="obstacle_speed_mps",
            category=ParameterCategory.OBSTACLE,
            distribution=DistributionType.UNIFORM,
            minimum=0.0,
            maximum=1.5,
            nominal_value=0.3,
            unit="m/s",
        ),
        RandomizationParameter(
            name="network_latency_ms",
            category=ParameterCategory.NETWORK,
            distribution=DistributionType.UNIFORM,
            minimum=0.0,
            maximum=100.0,
            nominal_value=20.0,
            unit="ms",
        ),
    )

    constraints = (
        ParameterConstraint(
            left_parameter="motor_efficiency",
            operator=ConstraintOperator.LESS_EQUAL,
            right_constant=1.0,
        ),
        ParameterConstraint(
            left_parameter="lidar_dropout_rate",
            operator=ConstraintOperator.LESS_EQUAL,
            right_constant=1.0,
        ),
    )

    correlated_groups = (
        CorrelatedParameterGroup(
            group_id="vision_environment",
            parameter_names=(
                "camera_brightness",
                "lighting_intensity",
            ),
            correlation_strength=0.85,
        ),
    )

    return DomainRandomizationProfile(
        profile_id=profile_id,
        tenant_id=tenant_id,
        robot_id=robot_id,
        name="Robot Physics and Sensor Randomization",
        parameters=parameters,
        difficulty=RandomizationDifficulty.MODERATE,
        sampling_strategy=SamplingStrategy.CORRELATED,
        correlated_groups=correlated_groups,
        constraints=constraints,
        base_seed=42,
        curriculum_stage_id="stage_003",
        description=(
            "Simulation domain randomization for "
            "robot physics, sensors, actuators, "
            "lighting, terrain and obstacles."
        ),
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    engine = RoboticsDomainRandomizationEngine()

    profile = create_default_robot_profile(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
    )

    engine.register_profile(
        profile,
        authorized=True,
        actor_id="simulation_admin",
    )

    engine.activate_profile(
        tenant_id="tenant_alpha",
        profile_id="robot_domain_randomization",
        authorized=True,
        reviewer_id="simulation_reviewer",
    )

    scenario = engine.generate_scenario(
        tenant_id="tenant_alpha",
        profile_id="robot_domain_randomization",
        scenario_id="scenario_001",
        seed=12345,
        difficulty=RandomizationDifficulty.HARD,
    )

    print("\nGENERATED SCENARIO")

    print(
        json.dumps(
            scenario,
            indent=2,
        )
    )

    print("\nREPRODUCIBILITY CHECK")

    print(
        json.dumps(
            engine.reproduce_scenario(
                tenant_id="tenant_alpha",
                scenario_id="scenario_001",
            ),
            indent=2,
        )
    )

    print("\nSIMULATION CONFIGURATION")

    print(
        json.dumps(
            engine.simulation_configuration(
                tenant_id="tenant_alpha",
                scenario_id="scenario_001",
            ),
            indent=2,
        )
    )

    print("\nGENERATE SCENARIO BATCH")

    batch = engine.generate_batch(
        tenant_id="tenant_alpha",
        profile_id="robot_domain_randomization",
        count=5,
        seed=2026,
    )

    print(
        json.dumps(
            {
                "generated_count": len(batch),
                "scenario_ids": [
                    item["scenario_id"]
                    for item in batch
                ],
            },
            indent=2,
        )
    )

    print("\nCURRICULUM DIFFICULTY ADVISOR")

    print(
        json.dumps(
            engine.recommend_difficulty(
                current_difficulty=(
                    RandomizationDifficulty.MODERATE
                ),
                success_rate=0.92,
                safety_violation_rate=0.0,
            ),
            indent=2,
        )
    )

    print("\nRANDOMIZATION COVERAGE")

    print(
        json.dumps(
            engine.randomization_coverage(
                tenant_id="tenant_alpha",
                profile_id="robot_domain_randomization",
            ),
            indent=2,
        )
    )

    output = engine.export_json(
        "data/robotics/domain_randomization_report.json",
        tenant_id="tenant_alpha",
    )

    print("\nExported:", output)


if __name__ == "__main__":
    main()
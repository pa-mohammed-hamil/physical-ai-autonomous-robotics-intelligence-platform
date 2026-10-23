"""Feature Flags module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Core Feature Flags

Features:
- Centralized feature flag management
- Environment-based feature configuration
- Global and tenant-specific overrides
- User, organization, fleet, and robot targeting
- Deterministic percentage rollouts
- Feature dependencies
- Time-based activation
- Kill switches
- Optional Redis-backed persistence
- Feature evaluation and audit hooks
- Safe defaults for physical robotics operations

Safety:
- Feature flags are not authorization controls.
- Enabling a feature never authorizes physical actuation.
- Hardware safety interlocks must operate independently.
- Dangerous capabilities default to disabled.
- Missing or invalid safety-critical flags fail closed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Protocol

logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes", "on", "enabled"}:
            return True

        if normalized in {"false", "0", "no", "off", "disabled"}:
            return False

    raise ValueError(f"Invalid boolean value: {value!r}")


def validate_flag_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("Feature flag name must be a string")

    name = name.strip().lower()

    if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", name):
        raise ValueError(
            "Feature flag name must start with a letter and "
            "contain only lowercase letters, digits, dots, "
            "underscores, or hyphens"
        )

    return name


def validate_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")

    value = value.strip()

    if len(value) > 255:
        raise ValueError(f"{field_name} is too long")

    return value


def serialize_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


# ============================================================
# Enums
# ============================================================


class FeatureCategory(str, Enum):
    CORE = "core"
    ROBOTICS = "robotics"
    FLEET = "fleet"
    TELEMETRY = "telemetry"
    SENSOR = "sensor"
    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PHYSICAL_AI = "physical_ai"
    WORLD_MODEL = "world_model"
    VLA = "vla"
    AGENTS = "agents"
    RAG = "rag"
    SIMULATION = "simulation"
    TRAINING = "training"
    EVALUATION = "evaluation"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    SAFETY = "safety"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    EXPERIMENTAL = "experimental"


class FeatureRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FeatureEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class FeatureDecisionReason(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    NOT_FOUND = "not_found"
    KILL_SWITCH = "kill_switch"
    OUTSIDE_SCHEDULE = "outside_schedule"
    ENVIRONMENT_DISABLED = "environment_disabled"
    TENANT_DISABLED = "tenant_disabled"
    TENANT_ENABLED = "tenant_enabled"
    USER_DISABLED = "user_disabled"
    USER_ENABLED = "user_enabled"
    FLEET_DISABLED = "fleet_disabled"
    FLEET_ENABLED = "fleet_enabled"
    ROBOT_DISABLED = "robot_disabled"
    ROBOT_ENABLED = "robot_enabled"
    DEPENDENCY_DISABLED = "dependency_disabled"
    ROLLOUT_EXCLUDED = "rollout_excluded"
    TARGETING_EXCLUDED = "targeting_excluded"
    CONFIGURATION_ERROR = "configuration_error"
    SAFETY_RESTRICTED = "safety_restricted"


# ============================================================
# Feature Flag Definitions
# ============================================================


class FeatureFlag(str, Enum):
    # Core
    API_V1 = "api_v1"
    WEBSOCKET_STREAMING = "websocket_streaming"
    MULTI_TENANCY = "multi_tenancy"
    AUDIT_LOGGING = "audit_logging"
    BACKGROUND_JOBS = "background_jobs"
    EVENT_BUS = "event_bus"
    REDIS_CACHE = "redis_cache"

    # Robotics
    ROBOT_REGISTRATION = "robot_registration"
    ROBOT_MONITORING = "robot_monitoring"
    ROBOT_HEALTH_MONITORING = "robot_health_monitoring"
    ROBOT_STATE_TRACKING = "robot_state_tracking"
    ROBOT_REMOTE_CONTROL = "robot_remote_control"
    ROBOT_AUTONOMOUS_EXECUTION = "robot_autonomous_execution"
    ROBOT_TELEOPERATION = "robot_teleoperation"
    ROBOT_TASK_SCHEDULING = "robot_task_scheduling"
    ROBOT_DIGITAL_TWIN = "robot_digital_twin"

    # Fleet
    FLEET_MANAGEMENT = "fleet_management"
    FLEET_ANALYTICS = "fleet_analytics"
    FLEET_OPTIMIZATION = "fleet_optimization"
    MULTI_ROBOT_COORDINATION = "multi_robot_coordination"

    # Telemetry
    TELEMETRY_INGESTION = "telemetry_ingestion"
    LIVE_TELEMETRY = "live_telemetry"
    TELEMETRY_ANOMALY_DETECTION = "telemetry_anomaly_detection"
    TELEMETRY_AGGREGATION = "telemetry_aggregation"
    TELEMETRY_EXPORT = "telemetry_export"

    # Sensors
    SENSOR_MONITORING = "sensor_monitoring"
    SENSOR_HEALTH_ANALYTICS = "sensor_health_analytics"
    SENSOR_CALIBRATION_TRACKING = "sensor_calibration_tracking"
    SENSOR_FUSION = "sensor_fusion"

    # Perception
    COMPUTER_VISION = "computer_vision"
    OBJECT_DETECTION = "object_detection"
    OBJECT_TRACKING = "object_tracking"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    DEPTH_ESTIMATION = "depth_estimation"
    POSE_ESTIMATION = "pose_estimation"
    MULTIMODAL_PERCEPTION = "multimodal_perception"
    LIVE_CAMERA_ANALYTICS = "live_camera_analytics"

    # Navigation
    NAVIGATION_PLANNING = "navigation_planning"
    AUTONOMOUS_NAVIGATION = "autonomous_navigation"
    SLAM = "slam"
    OBSTACLE_DETECTION = "obstacle_detection"
    PATH_OPTIMIZATION = "path_optimization"
    MAP_MANAGEMENT = "map_management"

    # Manipulation
    MANIPULATION_PLANNING = "manipulation_planning"
    ROBOT_GRASP_PREDICTION = "robot_grasp_prediction"
    PICK_AND_PLACE = "pick_and_place"
    AUTONOMOUS_MANIPULATION = "autonomous_manipulation"
    COLLISION_CHECKING = "collision_checking"

    # Physical AI
    PHYSICAL_AI_INFERENCE = "physical_ai_inference"
    PHYSICAL_AI_POLICY_TRAINING = "physical_ai_policy_training"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    IMITATION_LEARNING = "imitation_learning"
    SIM_TO_REAL = "sim_to_real"
    EMBODIED_AI = "embodied_ai"

    # World Models
    WORLD_MODEL_INFERENCE = "world_model_inference"
    WORLD_MODEL_TRAINING = "world_model_training"
    FUTURE_STATE_PREDICTION = "future_state_prediction"
    WORLD_MODEL_ROLLOUTS = "world_model_rollouts"
    IMAGINATION_PLANNING = "imagination_planning"

    # VLA
    VLA_INFERENCE = "vla_inference"
    VLA_TRAINING = "vla_training"
    LANGUAGE_CONDITIONED_PLANNING = (
        "language_conditioned_planning"
    )
    VLA_ACTION_EXECUTION = "vla_action_execution"

    # Agents
    AGENT_ORCHESTRATION = "agent_orchestration"
    MULTI_AGENT_SYSTEM = "multi_agent_system"
    ROBOT_DIAGNOSTIC_AGENT = "robot_diagnostic_agent"
    MAINTENANCE_AGENT = "maintenance_agent"
    REPAIR_AGENT = "repair_agent"
    SAFETY_REVIEW_AGENT = "safety_review_agent"

    # RAG
    ROBOTICS_RAG = "robotics_rag"
    HYBRID_SEARCH = "hybrid_search"
    GRAPH_RAG = "graph_rag"
    RAG_RERANKING = "rag_reranking"
    RAG_CITATIONS = "rag_citations"
    DOCUMENT_INTELLIGENCE = "document_intelligence"

    # Simulation
    SIMULATION_ENGINE = "simulation_engine"
    SIMULATION_SCENARIOS = "simulation_scenarios"
    DIGITAL_TWIN_SIMULATION = "digital_twin_simulation"
    DOMAIN_RANDOMIZATION = "domain_randomization"
    SIMULATION_BENCHMARKS = "simulation_benchmarks"

    # Evaluation
    MODEL_EVALUATION = "model_evaluation"
    ROBOTICS_BENCHMARKS = "robotics_benchmarks"
    SAFETY_EVALUATION = "safety_evaluation"
    VLA_EVALUATION = "vla_evaluation"
    WORLD_MODEL_EVALUATION = "world_model_evaluation"
    AGENT_EVALUATION = "agent_evaluation"
    RAG_EVALUATION = "rag_evaluation"

    # Maintenance
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    RUL_PREDICTION = "rul_prediction"
    FAILURE_PREDICTION = "failure_prediction"
    VIBRATION_ANALYSIS = "vibration_analysis"
    THERMAL_MONITORING = "thermal_monitoring"
    CONDITION_MONITORING = "condition_monitoring"

    # Repair
    ROBOT_PART_IMAGE_INSPECTION = "robot_part_image_inspection"
    ROBOT_DAMAGE_DETECTION = "robot_damage_detection"
    REPAIR_COST_ESTIMATION = "repair_cost_estimation"
    REPAIR_VS_REPLACE = "repair_vs_replace"
    PART_COMPATIBILITY = "part_compatibility"
    PART_PRICE_COMPARISON = "part_price_comparison"
    PART_INVENTORY = "part_inventory"

    # Safety
    SAFETY_MONITORING = "safety_monitoring"
    SAFETY_GUARDRAILS = "safety_guardrails"
    COLLISION_RISK_DETECTION = "collision_risk_detection"
    HUMAN_IN_THE_LOOP = "human_in_the_loop"
    SAFETY_INCIDENT_REPORTING = "safety_incident_reporting"
    EMERGENCY_ALERTING = "emergency_alerting"
    REMOTE_EMERGENCY_STOP_REQUEST = (
        "remote_emergency_stop_request"
    )
    EMERGENCY_STOP_RESET_REQUEST = (
        "emergency_stop_reset_request"
    )
    SAFETY_OVERRIDE = "safety_override"

    # Security
    RBAC = "rbac"
    MFA = "mfa"
    TENANT_ISOLATION = "tenant_isolation"
    PROMPT_INJECTION_DETECTION = "prompt_injection_detection"
    PII_REDACTION = "pii_redaction"
    FILE_MALWARE_SCANNING = "file_malware_scanning"
    MODEL_SECURITY_SCANNING = "model_security_scanning"

    # Experimental
    EXPERIMENTAL_POLICY_DEPLOYMENT = (
        "experimental_policy_deployment"
    )
    UNSUPERVISED_HARDWARE_EXPERIMENTS = (
        "unsupervised_hardware_experiments"
    )


# ============================================================
# Safety-Critical Flags
# ============================================================


SAFETY_CRITICAL_FLAGS: frozenset[str] = frozenset({
    FeatureFlag.ROBOT_REMOTE_CONTROL.value,
    FeatureFlag.ROBOT_AUTONOMOUS_EXECUTION.value,
    FeatureFlag.ROBOT_TELEOPERATION.value,
    FeatureFlag.AUTONOMOUS_NAVIGATION.value,
    FeatureFlag.AUTONOMOUS_MANIPULATION.value,
    FeatureFlag.VLA_ACTION_EXECUTION.value,
    FeatureFlag.REMOTE_EMERGENCY_STOP_REQUEST.value,
    FeatureFlag.EMERGENCY_STOP_RESET_REQUEST.value,
    FeatureFlag.SAFETY_OVERRIDE.value,
    FeatureFlag.EXPERIMENTAL_POLICY_DEPLOYMENT.value,
    FeatureFlag.UNSUPERVISED_HARDWARE_EXPERIMENTS.value,
})

PROTECTED_SAFETY_FLAGS: frozenset[str] = frozenset({
    FeatureFlag.SAFETY_MONITORING.value,
    FeatureFlag.SAFETY_GUARDRAILS.value,
    FeatureFlag.HUMAN_IN_THE_LOOP.value,
    FeatureFlag.RBAC.value,
    FeatureFlag.TENANT_ISOLATION.value,
    FeatureFlag.AUDIT_LOGGING.value,
})

# These capabilities must never be activated by feature
# flags alone. Separate certified safety and authorization
# systems are required.
PERMANENTLY_RESTRICTED_FLAGS: frozenset[str] = frozenset({
    FeatureFlag.SAFETY_OVERRIDE.value,
    FeatureFlag.UNSUPERVISED_HARDWARE_EXPERIMENTS.value,
})


# ============================================================
# Evaluation Context
# ============================================================


@dataclass(frozen=True, slots=True)
class FeatureContext:
    environment: FeatureEnvironment = (
        FeatureEnvironment.DEVELOPMENT
    )

    tenant_id: str | None = None
    organization_id: str | None = None
    user_id: str | None = None
    fleet_id: str | None = None
    robot_id: str | None = None

    roles: frozenset[str] = field(default_factory=frozenset)
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "tenant_id",
            "organization_id",
            "user_id",
            "fleet_id",
            "robot_id",
        ):
            value = getattr(self, field_name)

            if value is not None:
                validate_identifier(value, field_name)

    @property
    def rollout_identity(self) -> str | None:
        if self.user_id and self.tenant_id:
            return f"{self.tenant_id}:user:{self.user_id}"

        if self.robot_id and self.tenant_id:
            return f"{self.tenant_id}:robot:{self.robot_id}"

        if self.fleet_id and self.tenant_id:
            return f"{self.tenant_id}:fleet:{self.fleet_id}"

        if self.tenant_id:
            return f"tenant:{self.tenant_id}"

        return None


# ============================================================
# Feature Configuration
# ============================================================


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    enabled: bool = False

    category: FeatureCategory = FeatureCategory.CORE
    risk_level: FeatureRiskLevel = FeatureRiskLevel.LOW

    description: str = ""

    environments: frozenset[str] = field(
        default_factory=lambda: frozenset(
            environment.value
            for environment in FeatureEnvironment
        )
    )

    dependencies: tuple[str, ...] = ()

    rollout_percentage: float = 100.0

    allowed_tenants: frozenset[str] = field(
        default_factory=frozenset
    )
    denied_tenants: frozenset[str] = field(
        default_factory=frozenset
    )

    allowed_users: frozenset[str] = field(
        default_factory=frozenset
    )
    denied_users: frozenset[str] = field(
        default_factory=frozenset
    )

    allowed_fleets: frozenset[str] = field(
        default_factory=frozenset
    )
    denied_fleets: frozenset[str] = field(
        default_factory=frozenset
    )

    allowed_robots: frozenset[str] = field(
        default_factory=frozenset
    )
    denied_robots: frozenset[str] = field(
        default_factory=frozenset
    )

    required_roles: frozenset[str] = field(
        default_factory=frozenset
    )

    starts_at: datetime | None = None
    ends_at: datetime | None = None

    kill_switch: bool = False

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            validate_flag_name(self.name),
        )

        if not 0.0 <= self.rollout_percentage <= 100.0:
            raise ValueError(
                "rollout_percentage must be between 0 and 100"
            )

        for timestamp in (self.starts_at, self.ends_at):
            if timestamp is not None and timestamp.tzinfo is None:
                raise ValueError(
                    "Feature schedule timestamps must be timezone-aware"
                )

        if (
            self.starts_at is not None
            and self.ends_at is not None
            and self.starts_at >= self.ends_at
        ):
            raise ValueError(
                "starts_at must be earlier than ends_at"
            )

        dependencies = tuple(
            validate_flag_name(name)
            for name in self.dependencies
        )

        if self.name in dependencies:
            raise ValueError(
                "Feature cannot depend on itself"
            )

        object.__setattr__(
            self,
            "dependencies",
            dependencies,
        )

        try:
            json.dumps(dict(self.metadata), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Feature metadata must be JSON serializable"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "category": self.category.value,
            "risk_level": self.risk_level.value,
            "description": self.description,
            "environments": sorted(self.environments),
            "dependencies": list(self.dependencies),
            "rollout_percentage": self.rollout_percentage,
            "allowed_tenants": sorted(self.allowed_tenants),
            "denied_tenants": sorted(self.denied_tenants),
            "allowed_users": sorted(self.allowed_users),
            "denied_users": sorted(self.denied_users),
            "allowed_fleets": sorted(self.allowed_fleets),
            "denied_fleets": sorted(self.denied_fleets),
            "allowed_robots": sorted(self.allowed_robots),
            "denied_robots": sorted(self.denied_robots),
            "required_roles": sorted(self.required_roles),
            "starts_at": serialize_datetime(self.starts_at),
            "ends_at": serialize_datetime(self.ends_at),
            "kill_switch": self.kill_switch,
            "metadata": dict(self.metadata),
        }


# ============================================================
# Feature Evaluation Result
# ============================================================


@dataclass(frozen=True, slots=True)
class FeatureDecision:
    name: str
    enabled: bool
    reason: FeatureDecisionReason
    evaluated_at: datetime = field(default_factory=utc_now)
    rollout_bucket: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "reason": self.reason.value,
            "rollout_bucket": self.rollout_bucket,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


# ============================================================
# Feature Flag Exceptions
# ============================================================


class FeatureFlagError(RuntimeError):
    pass


class FeatureFlagNotFoundError(FeatureFlagError):
    pass


class FeatureFlagConfigurationError(FeatureFlagError):
    pass


class FeatureFlagPermissionError(FeatureFlagError):
    pass


# ============================================================
# Deterministic Rollout
# ============================================================


def calculate_rollout_bucket(
    flag_name: str,
    identity: str,
) -> float:
    """
    Stable rollout bucket in [0, 100).

    Uses SHA-256 instead of Python's randomized hash().
    """

    name = validate_flag_name(flag_name)

    if not identity:
        raise ValueError("Rollout identity is required")

    digest = hashlib.sha256(
        f"{name}:{identity}".encode("utf-8")
    ).digest()

    integer = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return integer / (2**64) * 100.0


# ============================================================
# Feature Registry
# ============================================================


class FeatureFlagRegistry:
    """
    In-process feature registry.

    Dynamic changes affect the current process only.
    Use an external store and explicit synchronization
    for multi-worker deployments.
    """

    def __init__(
        self,
        definitions: list[FeatureDefinition] | None = None,
    ) -> None:
        self._definitions: dict[
            str, FeatureDefinition
        ] = {}

        self._tenant_overrides: dict[
            tuple[str, str], bool
        ] = {}

        self._user_overrides: dict[
            tuple[str, str, str], bool
        ] = {}

        self._fleet_overrides: dict[
            tuple[str, str, str], bool
        ] = {}

        self._robot_overrides: dict[
            tuple[str, str, str], bool
        ] = {}

        self._lock = asyncio.Lock()

        for definition in definitions or []:
            self.register(definition)

    def register(
        self,
        definition: FeatureDefinition,
        *,
        replace_existing: bool = False,
    ) -> None:
        if (
            definition.name in self._definitions
            and not replace_existing
        ):
            raise FeatureFlagConfigurationError(
                f"Feature already registered: {definition.name}"
            )

        if (
            definition.name in PERMANENTLY_RESTRICTED_FLAGS
            and definition.enabled
        ):
            raise FeatureFlagPermissionError(
                "Restricted feature cannot be enabled"
            )

        if (
            definition.name in PROTECTED_SAFETY_FLAGS
            and not definition.enabled
        ):
            raise FeatureFlagPermissionError(
                "Protected safety feature cannot be disabled"
            )

        self._definitions[definition.name] = definition

    def get(
        self,
        name: str | FeatureFlag,
    ) -> FeatureDefinition:
        key = (
            name.value
            if isinstance(name, FeatureFlag)
            else validate_flag_name(name)
        )

        definition = self._definitions.get(key)

        if definition is None:
            raise FeatureFlagNotFoundError(
                f"Feature flag not found: {key}"
            )

        return definition

    def list_definitions(self) -> list[FeatureDefinition]:
        return list(self._definitions.values())

    async def set_enabled(
        self,
        name: str | FeatureFlag,
        enabled: bool,
    ) -> None:
        definition = self.get(name)

        if (
            definition.name in PERMANENTLY_RESTRICTED_FLAGS
            and enabled
        ):
            raise FeatureFlagPermissionError(
                "Restricted feature cannot be enabled"
            )

        if (
            definition.name in PROTECTED_SAFETY_FLAGS
            and not enabled
        ):
            raise FeatureFlagPermissionError(
                "Protected safety feature cannot be disabled"
            )

        async with self._lock:
            self._definitions[definition.name] = replace(
                definition,
                enabled=enabled,
            )

    async def set_kill_switch(
        self,
        name: str | FeatureFlag,
        active: bool,
    ) -> None:
        definition = self.get(name)

        if (
            definition.name in PROTECTED_SAFETY_FLAGS
            and active
        ):
            raise FeatureFlagPermissionError(
                "Cannot kill-switch protected safety controls"
            )

        async with self._lock:
            self._definitions[definition.name] = replace(
                definition,
                kill_switch=active,
            )

    async def set_tenant_override(
        self,
        name: str | FeatureFlag,
        tenant_id: str,
        enabled: bool,
    ) -> None:
        definition = self.get(name)
        tenant_id = validate_identifier(
            tenant_id,
            "tenant_id",
        )

        self._validate_override(definition.name, enabled)

        async with self._lock:
            self._tenant_overrides[
                (definition.name, tenant_id)
            ] = enabled

    async def set_user_override(
        self,
        name: str | FeatureFlag,
        tenant_id: str,
        user_id: str,
        enabled: bool,
    ) -> None:
        definition = self.get(name)

        tenant_id = validate_identifier(
            tenant_id,
            "tenant_id",
        )
        user_id = validate_identifier(
            user_id,
            "user_id",
        )

        self._validate_override(definition.name, enabled)

        async with self._lock:
            self._user_overrides[
                (definition.name, tenant_id, user_id)
            ] = enabled

    async def set_fleet_override(
        self,
        name: str | FeatureFlag,
        tenant_id: str,
        fleet_id: str,
        enabled: bool,
    ) -> None:
        definition = self.get(name)

        tenant_id = validate_identifier(
            tenant_id,
            "tenant_id",
        )
        fleet_id = validate_identifier(
            fleet_id,
            "fleet_id",
        )

        self._validate_override(definition.name, enabled)

        async with self._lock:
            self._fleet_overrides[
                (definition.name, tenant_id, fleet_id)
            ] = enabled

    async def set_robot_override(
        self,
        name: str | FeatureFlag,
        tenant_id: str,
        robot_id: str,
        enabled: bool,
    ) -> None:
        definition = self.get(name)

        tenant_id = validate_identifier(
            tenant_id,
            "tenant_id",
        )
        robot_id = validate_identifier(
            robot_id,
            "robot_id",
        )

        self._validate_override(definition.name, enabled)

        async with self._lock:
            self._robot_overrides[
                (definition.name, tenant_id, robot_id)
            ] = enabled

    @staticmethod
    def _validate_override(
        name: str,
        enabled: bool,
    ) -> None:
        if (
            name in PERMANENTLY_RESTRICTED_FLAGS
            and enabled
        ):
            raise FeatureFlagPermissionError(
                "Restricted feature cannot be enabled"
            )

        if (
            name in PROTECTED_SAFETY_FLAGS
            and not enabled
        ):
            raise FeatureFlagPermissionError(
                "Protected safety feature cannot be disabled"
            )

    def evaluate(
        self,
        name: str | FeatureFlag,
        context: FeatureContext | None = None,
    ) -> FeatureDecision:
        context = context or FeatureContext()

        key = (
            name.value
            if isinstance(name, FeatureFlag)
            else validate_flag_name(name)
        )

        return self._evaluate_recursive(
            key,
            context,
            visited=frozenset(),
        )

    def _evaluate_recursive(
        self,
        name: str,
        context: FeatureContext,
        visited: frozenset[str],
    ) -> FeatureDecision:
        if name in visited:
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.CONFIGURATION_ERROR,
            )

        definition = self._definitions.get(name)

        if definition is None:
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.NOT_FOUND,
            )

        if name in PERMANENTLY_RESTRICTED_FLAGS:
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.SAFETY_RESTRICTED,
            )

        if definition.kill_switch:
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.KILL_SWITCH,
            )

        now = utc_now()

        if definition.starts_at and now < definition.starts_at:
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.OUTSIDE_SCHEDULE,
            )

        if definition.ends_at and now >= definition.ends_at:
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.OUTSIDE_SCHEDULE,
            )

        if context.environment.value not in definition.environments:
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.ENVIRONMENT_DISABLED,
            )

        if not definition.enabled:
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.DISABLED,
            )

        # Safety-critical features require an explicit tenant
        # and cannot be activated in an anonymous context.
        if (
            name in SAFETY_CRITICAL_FLAGS
            and not context.tenant_id
        ):
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.SAFETY_RESTRICTED,
            )

        if (
            context.tenant_id
            and context.tenant_id in definition.denied_tenants
        ):
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.TENANT_DISABLED,
            )

        if (
            context.user_id
            and context.user_id in definition.denied_users
        ):
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.USER_DISABLED,
            )

        if (
            context.fleet_id
            and context.fleet_id in definition.denied_fleets
        ):
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.FLEET_DISABLED,
            )

        if (
            context.robot_id
            and context.robot_id in definition.denied_robots
        ):
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.ROBOT_DISABLED,
            )

        if (
            definition.allowed_tenants
            and context.tenant_id not in definition.allowed_tenants
        ):
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.TARGETING_EXCLUDED,
            )

        if (
            definition.allowed_users
            and context.user_id not in definition.allowed_users
        ):
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.TARGETING_EXCLUDED,
            )

        if (
            definition.allowed_fleets
            and context.fleet_id not in definition.allowed_fleets
        ):
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.TARGETING_EXCLUDED,
            )

        if (
            definition.allowed_robots
            and context.robot_id not in definition.allowed_robots
        ):
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.TARGETING_EXCLUDED,
            )

        if (
            definition.required_roles
            and not definition.required_roles.issubset(
                context.roles
            )
        ):
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.TARGETING_EXCLUDED,
            )

        next_visited = visited | {name}

        for dependency in definition.dependencies:
            dependency_decision = self._evaluate_recursive(
                dependency,
                context,
                next_visited,
            )

            if not dependency_decision.enabled:
                return FeatureDecision(
                    name=name,
                    enabled=False,
                    reason=FeatureDecisionReason.DEPENDENCY_DISABLED,
                )

        # Explicit false overrides are authoritative.
        # True overrides can enable targeting, but never
        # bypass global disablement or safety restrictions.
        override_reason: FeatureDecisionReason | None = None

        if context.tenant_id:
            tenant_override = self._tenant_overrides.get(
                (name, context.tenant_id)
            )

            if tenant_override is False:
                return FeatureDecision(
                    name=name,
                    enabled=False,
                    reason=FeatureDecisionReason.TENANT_DISABLED,
                )

            if tenant_override is True:
                override_reason = (
                    FeatureDecisionReason.TENANT_ENABLED
                )

        if context.tenant_id and context.user_id:
            user_override = self._user_overrides.get(
                (name, context.tenant_id, context.user_id)
            )

            if user_override is False:
                return FeatureDecision(
                    name=name,
                    enabled=False,
                    reason=FeatureDecisionReason.USER_DISABLED,
                )

            if user_override is True:
                override_reason = (
                    FeatureDecisionReason.USER_ENABLED
                )

        if context.tenant_id and context.fleet_id:
            fleet_override = self._fleet_overrides.get(
                (name, context.tenant_id, context.fleet_id)
            )

            if fleet_override is False:
                return FeatureDecision(
                    name=name,
                    enabled=False,
                    reason=FeatureDecisionReason.FLEET_DISABLED,
                )

            if fleet_override is True:
                override_reason = (
                    FeatureDecisionReason.FLEET_ENABLED
                )

        if context.tenant_id and context.robot_id:
            robot_override = self._robot_overrides.get(
                (name, context.tenant_id, context.robot_id)
            )

            if robot_override is False:
                return FeatureDecision(
                    name=name,
                    enabled=False,
                    reason=FeatureDecisionReason.ROBOT_DISABLED,
                )

            if robot_override is True:
                override_reason = (
                    FeatureDecisionReason.ROBOT_ENABLED
                )

        if override_reason is not None:
            return FeatureDecision(
                name=name,
                enabled=True,
                reason=override_reason,
            )

        if definition.rollout_percentage >= 100:
            return FeatureDecision(
                name=name,
                enabled=True,
                reason=FeatureDecisionReason.ENABLED,
            )

        identity = context.rollout_identity

        if identity is None:
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.ROLLOUT_EXCLUDED,
            )

        bucket = calculate_rollout_bucket(name, identity)

        if bucket >= definition.rollout_percentage:
            return FeatureDecision(
                name=name,
                enabled=False,
                reason=FeatureDecisionReason.ROLLOUT_EXCLUDED,
                rollout_bucket=bucket,
            )

        return FeatureDecision(
            name=name,
            enabled=True,
            reason=FeatureDecisionReason.ENABLED,
            rollout_bucket=bucket,
        )

    def is_enabled(
        self,
        name: str | FeatureFlag,
        context: FeatureContext | None = None,
    ) -> bool:
        return self.evaluate(name, context).enabled

    def evaluate_all(
        self,
        context: FeatureContext | None = None,
    ) -> dict[str, FeatureDecision]:
        return {
            name: self.evaluate(name, context)
            for name in sorted(self._definitions)
        }

    def snapshot(
        self,
        context: FeatureContext | None = None,
    ) -> dict[str, bool]:
        return {
            name: decision.enabled
            for name, decision in self.evaluate_all(
                context
            ).items()
        }


# ============================================================
# Default Feature Definitions
# ============================================================


DEFAULT_ENABLED_FLAGS: frozenset[str] = frozenset({
    FeatureFlag.API_V1.value,
    FeatureFlag.MULTI_TENANCY.value,
    FeatureFlag.AUDIT_LOGGING.value,
    FeatureFlag.ROBOT_REGISTRATION.value,
    FeatureFlag.ROBOT_MONITORING.value,
    FeatureFlag.ROBOT_HEALTH_MONITORING.value,
    FeatureFlag.ROBOT_STATE_TRACKING.value,
    FeatureFlag.FLEET_MANAGEMENT.value,
    FeatureFlag.FLEET_ANALYTICS.value,
    FeatureFlag.TELEMETRY_INGESTION.value,
    FeatureFlag.LIVE_TELEMETRY.value,
    FeatureFlag.SENSOR_MONITORING.value,
    FeatureFlag.SENSOR_HEALTH_ANALYTICS.value,
    FeatureFlag.SENSOR_CALIBRATION_TRACKING.value,
    FeatureFlag.PREDICTIVE_MAINTENANCE.value,
    FeatureFlag.RUL_PREDICTION.value,
    FeatureFlag.FAILURE_PREDICTION.value,
    FeatureFlag.REPAIR_COST_ESTIMATION.value,
    FeatureFlag.REPAIR_VS_REPLACE.value,
    FeatureFlag.PART_COMPATIBILITY.value,
    FeatureFlag.PART_INVENTORY.value,
    FeatureFlag.SAFETY_MONITORING.value,
    FeatureFlag.SAFETY_GUARDRAILS.value,
    FeatureFlag.HUMAN_IN_THE_LOOP.value,
    FeatureFlag.SAFETY_INCIDENT_REPORTING.value,
    FeatureFlag.RBAC.value,
    FeatureFlag.TENANT_ISOLATION.value,
    FeatureFlag.PII_REDACTION.value,
    FeatureFlag.MODEL_EVALUATION.value,
    FeatureFlag.SAFETY_EVALUATION.value,
    FeatureFlag.ROBOTICS_BENCHMARKS.value,
})


FEATURE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    FeatureFlag.ROBOT_HEALTH_MONITORING.value: (
        FeatureFlag.ROBOT_MONITORING.value,
    ),
    FeatureFlag.LIVE_TELEMETRY.value: (
        FeatureFlag.TELEMETRY_INGESTION.value,
    ),
    FeatureFlag.TELEMETRY_ANOMALY_DETECTION.value: (
        FeatureFlag.TELEMETRY_INGESTION.value,
    ),
    FeatureFlag.SENSOR_HEALTH_ANALYTICS.value: (
        FeatureFlag.SENSOR_MONITORING.value,
    ),
    FeatureFlag.AUTONOMOUS_NAVIGATION.value: (
        FeatureFlag.NAVIGATION_PLANNING.value,
        FeatureFlag.SAFETY_GUARDRAILS.value,
    ),
    FeatureFlag.AUTONOMOUS_MANIPULATION.value: (
        FeatureFlag.MANIPULATION_PLANNING.value,
        FeatureFlag.COLLISION_CHECKING.value,
        FeatureFlag.SAFETY_GUARDRAILS.value,
    ),
    FeatureFlag.VLA_ACTION_EXECUTION.value: (
        FeatureFlag.VLA_INFERENCE.value,
        FeatureFlag.SAFETY_GUARDRAILS.value,
        FeatureFlag.HUMAN_IN_THE_LOOP.value,
    ),
    FeatureFlag.WORLD_MODEL_ROLLOUTS.value: (
        FeatureFlag.WORLD_MODEL_INFERENCE.value,
    ),
    FeatureFlag.IMAGINATION_PLANNING.value: (
        FeatureFlag.WORLD_MODEL_INFERENCE.value,
        FeatureFlag.WORLD_MODEL_ROLLOUTS.value,
    ),
    FeatureFlag.SIM_TO_REAL.value: (
        FeatureFlag.SIMULATION_ENGINE.value,
        FeatureFlag.SAFETY_EVALUATION.value,
    ),
    FeatureFlag.ROBOT_DAMAGE_DETECTION.value: (
        FeatureFlag.COMPUTER_VISION.value,
    ),
    FeatureFlag.ROBOT_PART_IMAGE_INSPECTION.value: (
        FeatureFlag.COMPUTER_VISION.value,
    ),
    FeatureFlag.REPAIR_VS_REPLACE.value: (
        FeatureFlag.REPAIR_COST_ESTIMATION.value,
    ),
    FeatureFlag.RUL_PREDICTION.value: (
        FeatureFlag.PREDICTIVE_MAINTENANCE.value,
    ),
    FeatureFlag.FAILURE_PREDICTION.value: (
        FeatureFlag.PREDICTIVE_MAINTENANCE.value,
    ),
    FeatureFlag.RAG_CITATIONS.value: (
        FeatureFlag.ROBOTICS_RAG.value,
    ),
    FeatureFlag.GRAPH_RAG.value: (
        FeatureFlag.ROBOTICS_RAG.value,
    ),
}


def create_default_definitions() -> list[FeatureDefinition]:
    definitions: list[FeatureDefinition] = []

    for flag in FeatureFlag:
        name = flag.value

        if name in SAFETY_CRITICAL_FLAGS:
            risk = FeatureRiskLevel.CRITICAL
        elif name in PROTECTED_SAFETY_FLAGS:
            risk = FeatureRiskLevel.HIGH
        else:
            risk = FeatureRiskLevel.LOW

        category = FeatureCategory.CORE

        for candidate in FeatureCategory:
            if name.startswith(candidate.value + "_"):
                category = candidate
                break

        definitions.append(
            FeatureDefinition(
                name=name,
                enabled=name in DEFAULT_ENABLED_FLAGS,
                category=category,
                risk_level=risk,
                dependencies=FEATURE_DEPENDENCIES.get(
                    name,
                    (),
                ),
                description=(
                    flag.name.replace("_", " ").title()
                ),
            )
        )

    return definitions


# ============================================================
# Environment Overrides
# ============================================================


def load_environment_overrides(
    registry: FeatureFlagRegistry,
    *,
    prefix: str = "FEATURE_",
) -> None:
    """
    Load startup-only environment overrides.

    Example:
        FEATURE_COMPUTER_VISION=true
        FEATURE_GRAPH_RAG=false

    Protected and permanently restricted flags cannot
    be disabled or enabled contrary to safety policy.
    """

    for definition in registry.list_definitions():
        variable = (
            prefix
            + definition.name.upper().replace(".", "_")
        )

        raw = os.getenv(variable)

        if raw is None:
            continue

        try:
            enabled = parse_bool(raw)

            if (
                definition.name in PERMANENTLY_RESTRICTED_FLAGS
                and enabled
            ):
                raise FeatureFlagPermissionError(
                    "Restricted feature cannot be enabled"
                )

            if (
                definition.name in PROTECTED_SAFETY_FLAGS
                and not enabled
            ):
                raise FeatureFlagPermissionError(
                    "Protected safety feature cannot be disabled"
                )

            registry.register(
                replace(
                    definition,
                    enabled=enabled,
                ),
                replace_existing=True,
            )

        except (ValueError, FeatureFlagPermissionError) as exc:
            logger.error(
                "Invalid feature flag environment setting %s: %s",
                variable,
                exc,
            )

            if definition.name in SAFETY_CRITICAL_FLAGS:
                registry.register(
                    replace(
                        definition,
                        enabled=False,
                    ),
                    replace_existing=True,
                )


# ============================================================
# Feature Flag Service
# ============================================================


class FeatureFlagService:
    def __init__(
        self,
        registry: FeatureFlagRegistry,
    ) -> None:
        self.registry = registry

    def is_enabled(
        self,
        flag: str | FeatureFlag,
        *,
        environment: FeatureEnvironment = (
            FeatureEnvironment.DEVELOPMENT
        ),
        tenant_id: str | None = None,
        organization_id: str | None = None,
        user_id: str | None = None,
        fleet_id: str | None = None,
        robot_id: str | None = None,
        roles: frozenset[str] | None = None,
    ) -> bool:
        context = FeatureContext(
            environment=environment,
            tenant_id=tenant_id,
            organization_id=organization_id,
            user_id=user_id,
            fleet_id=fleet_id,
            robot_id=robot_id,
            roles=roles or frozenset(),
        )

        return self.registry.is_enabled(flag, context)

    def require_enabled(
        self,
        flag: str | FeatureFlag,
        context: FeatureContext,
    ) -> None:
        decision = self.registry.evaluate(flag, context)

        if not decision.enabled:
            raise FeatureFlagPermissionError(
                f"Feature unavailable: {decision.name} "
                f"({decision.reason.value})"
            )

    def get_available_features(
        self,
        context: FeatureContext,
    ) -> list[str]:
        return [
            name
            for name, enabled in self.registry.snapshot(
                context
            ).items()
            if enabled
        ]

    def get_feature_report(
        self,
        context: FeatureContext,
    ) -> dict[str, Any]:
        decisions = self.registry.evaluate_all(context)

        enabled = [
            name
            for name, decision in decisions.items()
            if decision.enabled
        ]

        disabled = [
            name
            for name, decision in decisions.items()
            if not decision.enabled
        ]

        return {
            "environment": context.environment.value,
            "tenant_id": context.tenant_id,
            "total_features": len(decisions),
            "enabled_count": len(enabled),
            "disabled_count": len(disabled),
            "enabled_features": enabled,
            "disabled_features": disabled,
            "evaluated_at": utc_now().isoformat(),
        }


# ============================================================
# Application Singleton
# ============================================================


_feature_registry: FeatureFlagRegistry | None = None
_feature_service: FeatureFlagService | None = None


def get_feature_registry() -> FeatureFlagRegistry:
    global _feature_registry

    if _feature_registry is None:
        _feature_registry = FeatureFlagRegistry(
            create_default_definitions()
        )

        load_environment_overrides(_feature_registry)

    return _feature_registry


def get_feature_flag_service() -> FeatureFlagService:
    global _feature_service

    if _feature_service is None:
        _feature_service = FeatureFlagService(
            get_feature_registry()
        )

    return _feature_service


def is_feature_enabled(
    flag: str | FeatureFlag,
    context: FeatureContext | None = None,
) -> bool:
    return get_feature_registry().is_enabled(
        flag,
        context,
    )


def require_feature(
    flag: str | FeatureFlag,
    context: FeatureContext,
) -> None:
    get_feature_flag_service().require_enabled(
        flag,
        context,
    )


# ============================================================
# Exports
# ============================================================


__all__ = [
    "FeatureCategory",
    "FeatureRiskLevel",
    "FeatureEnvironment",
    "FeatureDecisionReason",
    "FeatureFlag",
    "FeatureContext",
    "FeatureDefinition",
    "FeatureDecision",
    "FeatureFlagError",
    "FeatureFlagNotFoundError",
    "FeatureFlagConfigurationError",
    "FeatureFlagPermissionError",
    "FeatureFlagRegistry",
    "FeatureFlagService",
    "SAFETY_CRITICAL_FLAGS",
    "PROTECTED_SAFETY_FLAGS",
    "PERMANENTLY_RESTRICTED_FLAGS",
    "DEFAULT_ENABLED_FLAGS",
    "FEATURE_DEPENDENCIES",
    "calculate_rollout_bucket",
    "create_default_definitions",
    "load_environment_overrides",
    "get_feature_registry",
    "get_feature_flag_service",
    "is_feature_enabled",
    "require_feature",
    "parse_bool",
    "utc_now",
]
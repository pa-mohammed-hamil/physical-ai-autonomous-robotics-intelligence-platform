"""Agent Registry module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/agent_registry.py

Physical AI & Autonomous Robotics Intelligence Platform
Agent Registry & Lifecycle Management

Python 3.11+

Features:
- Agent registration and discovery
- Multi-agent management
- Agent capabilities and skills
- Agent lifecycle management
- Agent health monitoring
- Agent availability tracking
- Agent workload and capacity
- Agent selection and routing
- Task-to-agent matching
- Agent heartbeat monitoring
- Agent performance metrics
- Tenant isolation
- Concurrent registry operations
- Safe registration and deregistration

This module manages agent metadata and availability.
Actual agent execution, authorization, persistence,
and robot actuation are implemented separately.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import re
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

MAX_AGENTS_PER_TENANT = 1000
MAX_CAPABILITIES = 100
MAX_SKILLS = 100
MAX_METADATA_FIELDS = 50
MAX_SEARCH_RESULTS = 100

DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 120

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)


# ============================================================
# Utilities
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def validate_identifier(value: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            "Invalid identifier"
        )

    return value


def validate_score(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(
            "Score must be finite"
        )

    if not 0 <= value <= 1:
        raise ValueError(
            "Score must be between 0 and 1"
        )

    return value


def validate_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError(
            "Datetime must be timezone-aware"
        )

    return value


# ============================================================
# Enums
# ============================================================

class AgentType(str, Enum):
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    RESEARCH = "research"
    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    ROBOT_CONTROL = "robot_control"
    WORLD_MODEL = "world_model"
    IMAGINATION = "imagination"
    VLA = "vla"
    SAFETY = "safety"
    MAINTENANCE = "maintenance"
    DIAGNOSTICS = "diagnostics"
    REPAIR = "repair"
    COST_ESTIMATION = "cost_estimation"
    FLEET = "fleet"
    SIMULATION = "simulation"
    TRAINING = "training"
    EVALUATION = "evaluation"
    RAG = "rag"
    MEMORY = "memory"
    ANALYTICS = "analytics"
    REPORTING = "reporting"
    CUSTOM = "custom"


class AgentStatus(str, Enum):
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    FAILED = "failed"
    DISABLED = "disabled"


class AgentHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AgentCapability(str, Enum):
    TASK_PLANNING = "task_planning"
    TASK_DECOMPOSITION = "task_decomposition"
    TOOL_CALLING = "tool_calling"
    MULTI_AGENT_COORDINATION = "multi_agent_coordination"
    COMPUTER_VISION = "computer_vision"
    OBJECT_DETECTION = "object_detection"
    OBJECT_TRACKING = "object_tracking"
    SENSOR_FUSION = "sensor_fusion"
    LIDAR_PROCESSING = "lidar_processing"
    SLAM = "slam"
    PATH_PLANNING = "path_planning"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    MOTION_PLANNING = "motion_planning"
    ROBOT_MANIPULATION = "robot_manipulation"
    GRASP_PLANNING = "grasp_planning"
    VLA_INFERENCE = "vla_inference"
    WORLD_STATE_ESTIMATION = "world_state_estimation"
    FUTURE_PREDICTION = "future_prediction"
    IMAGINATION = "imagination"
    SIMULATION = "simulation"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ANOMALY_DETECTION = "anomaly_detection"
    FAILURE_PREDICTION = "failure_prediction"
    REMAINING_USEFUL_LIFE = "remaining_useful_life"
    HEALTH_MONITORING = "health_monitoring"
    DAMAGE_DETECTION = "damage_detection"
    REPAIR_ESTIMATION = "repair_estimation"
    COST_ANALYSIS = "cost_analysis"
    SAFETY_VALIDATION = "safety_validation"
    RISK_ASSESSMENT = "risk_assessment"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    REPORT_GENERATION = "report_generation"
    DATA_ANALYSIS = "data_analysis"
    MODEL_EVALUATION = "model_evaluation"
    CUSTOM = "custom"


class SelectionStrategy(str, Enum):
    LEAST_BUSY = "least_busy"
    HIGHEST_PRIORITY = "highest_priority"
    BEST_PERFORMANCE = "best_performance"
    BALANCED = "balanced"


# ============================================================
# Exceptions
# ============================================================

class AgentRegistryError(Exception):
    pass


class AgentNotFoundError(AgentRegistryError):
    pass


class AgentAlreadyExistsError(AgentRegistryError):
    pass


class AgentCapacityError(AgentRegistryError):
    pass


class AgentUnavailableError(AgentRegistryError):
    pass


class AgentValidationError(AgentRegistryError):
    pass


class AgentStateConflictError(AgentRegistryError):
    pass


# ============================================================
# Models
# ============================================================

@dataclass(frozen=True, slots=True)
class AgentScope:
    tenant_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)


@dataclass(slots=True)
class AgentPerformance:
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0

    average_latency_ms: float = 0.0
    success_rate: float = 0.0

    total_tokens: int = 0
    total_cost_usd: float = 0.0

    last_task_at: datetime | None = None

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.total_tasks,
                self.completed_tasks,
                self.failed_tasks,
                self.total_tokens,
            )
        ):
            raise AgentValidationError(
                "Performance counters cannot be negative"
            )

        if (
            not math.isfinite(self.average_latency_ms)
            or self.average_latency_ms < 0
        ):
            raise AgentValidationError(
                "Invalid average latency"
            )

        if (
            not math.isfinite(self.total_cost_usd)
            or self.total_cost_usd < 0
        ):
            raise AgentValidationError(
                "Invalid total cost"
            )

        validate_score(self.success_rate)


@dataclass(slots=True)
class AgentDefinition:
    agent_id: str
    tenant_id: str

    name: str
    agent_type: AgentType

    description: str = ""

    capabilities: list[AgentCapability] = field(
        default_factory=list
    )

    skills: list[str] = field(
        default_factory=list
    )

    supported_robot_types: list[str] = field(
        default_factory=list
    )

    status: AgentStatus = AgentStatus.REGISTERED
    health: AgentHealth = AgentHealth.UNKNOWN

    priority: int = 50
    max_concurrent_tasks: int = 1
    active_tasks: int = 0

    enabled: bool = True

    model_id: str | None = None

    version: str = "1.0.0"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    performance: AgentPerformance = field(
        default_factory=AgentPerformance
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    last_heartbeat_at: datetime | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.agent_id)
        validate_identifier(self.tenant_id)

        if self.model_id is not None:
            validate_identifier(self.model_id)

        if not self.name.strip():
            raise AgentValidationError(
                "Agent name cannot be empty"
            )

        if len(self.name) > 200:
            raise AgentValidationError(
                "Agent name is too long"
            )

        if len(self.capabilities) > MAX_CAPABILITIES:
            raise AgentValidationError(
                "Too many agent capabilities"
            )

        if len(self.skills) > MAX_SKILLS:
            raise AgentValidationError(
                "Too many agent skills"
            )

        if len(self.metadata) > MAX_METADATA_FIELDS:
            raise AgentValidationError(
                "Too many metadata fields"
            )

        if not 0 <= self.priority <= 100:
            raise AgentValidationError(
                "Priority must be between 0 and 100"
            )

        if self.max_concurrent_tasks < 1:
            raise AgentValidationError(
                "Maximum concurrent tasks must be positive"
            )

        if not (
            0 <= self.active_tasks <= self.max_concurrent_tasks
        ):
            raise AgentValidationError(
                "Invalid active task count"
            )

        validate_datetime(self.created_at)
        validate_datetime(self.updated_at)

        if self.last_heartbeat_at is not None:
            validate_datetime(self.last_heartbeat_at)

    @property
    def available_capacity(self) -> int:
        return max(
            0,
            self.max_concurrent_tasks - self.active_tasks,
        )

    @property
    def utilization(self) -> float:
        return (
            self.active_tasks
            / self.max_concurrent_tasks
        )


@dataclass(slots=True)
class AgentRegistrationRequest:
    name: str
    agent_type: AgentType

    description: str = ""

    capabilities: list[AgentCapability] = field(
        default_factory=list
    )

    skills: list[str] = field(
        default_factory=list
    )

    supported_robot_types: list[str] = field(
        default_factory=list
    )

    priority: int = 50
    max_concurrent_tasks: int = 1

    model_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class AgentSearchFilters:
    agent_type: AgentType | None = None
    capability: AgentCapability | None = None

    status: AgentStatus | None = None
    health: AgentHealth | None = None

    robot_type: str | None = None

    enabled_only: bool = True
    available_only: bool = False


@dataclass(slots=True)
class AgentSelectionRequest:
    required_capabilities: list[AgentCapability] = field(
        default_factory=list
    )

    required_skills: list[str] = field(
        default_factory=list
    )

    preferred_agent_type: AgentType | None = None

    robot_type: str | None = None

    strategy: SelectionStrategy = (
        SelectionStrategy.BALANCED
    )


@dataclass(slots=True)
class AgentSelectionResult:
    agent: AgentDefinition
    score: float
    reason: str


@dataclass(slots=True)
class AgentRegistryStatistics:
    tenant_id: str

    total_agents: int
    enabled_agents: int
    available_agents: int

    busy_agents: int
    offline_agents: int
    failed_agents: int

    total_active_tasks: int
    total_capacity: int

    average_utilization: float

    generated_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# Repository Protocol
# ============================================================

class AgentRegistryRepository(Protocol):

    async def save(
        self,
        agent: AgentDefinition,
    ) -> AgentDefinition:
        ...

    async def get(
        self,
        *,
        scope: AgentScope,
        agent_id: str,
    ) -> AgentDefinition | None:
        ...

    async def list(
        self,
        *,
        scope: AgentScope,
    ) -> list[AgentDefinition]:
        ...

    async def delete(
        self,
        *,
        scope: AgentScope,
        agent_id: str,
    ) -> bool:
        ...


# ============================================================
# In-Memory Repository
# ============================================================

class InMemoryAgentRegistryRepository:

    def __init__(self) -> None:
        self._agents: dict[
            tuple[str, str],
            AgentDefinition,
        ] = {}

        self._lock = asyncio.Lock()

    async def save(
        self,
        agent: AgentDefinition,
    ) -> AgentDefinition:

        key = (
            agent.tenant_id,
            agent.agent_id,
        )

        async with self._lock:
            self._agents[key] = copy.deepcopy(agent)

        return copy.deepcopy(agent)

    async def get(
        self,
        *,
        scope: AgentScope,
        agent_id: str,
    ) -> AgentDefinition | None:

        key = (
            scope.tenant_id,
            agent_id,
        )

        async with self._lock:
            agent = self._agents.get(key)

            return (
                copy.deepcopy(agent)
                if agent is not None
                else None
            )

    async def list(
        self,
        *,
        scope: AgentScope,
    ) -> list[AgentDefinition]:

        async with self._lock:
            return [
                copy.deepcopy(agent)
                for (tenant_id, _), agent
                in self._agents.items()
                if tenant_id == scope.tenant_id
            ]

    async def delete(
        self,
        *,
        scope: AgentScope,
        agent_id: str,
    ) -> bool:

        key = (
            scope.tenant_id,
            agent_id,
        )

        async with self._lock:
            return self._agents.pop(key, None) is not None


# ============================================================
# Agent Registry
# ============================================================

class AgentRegistry:

    def __init__(
        self,
        *,
        repository: AgentRegistryRepository,
        heartbeat_timeout_seconds: int = (
            DEFAULT_HEARTBEAT_TIMEOUT_SECONDS
        ),
    ) -> None:

        if heartbeat_timeout_seconds <= 0:
            raise ValueError(
                "Heartbeat timeout must be positive"
            )

        self.repository = repository

        self.heartbeat_timeout_seconds = (
            heartbeat_timeout_seconds
        )

        self._lock = asyncio.Lock()

    # --------------------------------------------------------
    # Registration
    # --------------------------------------------------------

    async def register_agent(
        self,
        *,
        scope: AgentScope,
        payload: AgentRegistrationRequest,
        agent_id: str | None = None,
    ) -> AgentDefinition:

        async with self._lock:
            agents = await self.repository.list(
                scope=scope
            )

            if len(agents) >= MAX_AGENTS_PER_TENANT:
                raise AgentCapacityError(
                    "Maximum agents per tenant reached"
                )

            assigned_id = agent_id or generate_id()

            validate_identifier(assigned_id)

            existing = await self.repository.get(
                scope=scope,
                agent_id=assigned_id,
            )

            if existing is not None:
                raise AgentAlreadyExistsError(
                    "Agent already registered"
                )

            agent = AgentDefinition(
                agent_id=assigned_id,
                tenant_id=scope.tenant_id,
                name=payload.name,
                agent_type=payload.agent_type,
                description=payload.description,
                capabilities=list(payload.capabilities),
                skills=list(payload.skills),
                supported_robot_types=list(
                    payload.supported_robot_types
                ),
                priority=payload.priority,
                max_concurrent_tasks=(
                    payload.max_concurrent_tasks
                ),
                model_id=payload.model_id,
                metadata=dict(payload.metadata),
            )

            return await self.repository.save(agent)

    # --------------------------------------------------------
    # Get Agent
    # --------------------------------------------------------

    async def get_agent(
        self,
        *,
        scope: AgentScope,
        agent_id: str,
    ) -> AgentDefinition:

        validate_identifier(agent_id)

        agent = await self.repository.get(
            scope=scope,
            agent_id=agent_id,
        )

        if agent is None:
            raise AgentNotFoundError(
                "Agent not found"
            )

        return agent

    # --------------------------------------------------------
    # List Agents
    # --------------------------------------------------------

    async def list_agents(
        self,
        *,
        scope: AgentScope,
        filters: AgentSearchFilters | None = None,
        limit: int = MAX_SEARCH_RESULTS,
    ) -> list[AgentDefinition]:

        if not 1 <= limit <= MAX_SEARCH_RESULTS:
            raise AgentValidationError(
                "Invalid result limit"
            )

        filters = filters or AgentSearchFilters()

        agents = await self.repository.list(
            scope=scope
        )

        results: list[AgentDefinition] = []

        for agent in agents:

            if (
                filters.agent_type is not None
                and agent.agent_type != filters.agent_type
            ):
                continue

            if (
                filters.capability is not None
                and filters.capability not in agent.capabilities
            ):
                continue

            if (
                filters.status is not None
                and agent.status != filters.status
            ):
                continue

            if (
                filters.health is not None
                and agent.health != filters.health
            ):
                continue

            if (
                filters.robot_type is not None
                and filters.robot_type
                not in agent.supported_robot_types
            ):
                continue

            if filters.enabled_only and not agent.enabled:
                continue

            if filters.available_only:
                if not self._is_available(agent):
                    continue

            results.append(agent)

        results.sort(
            key=lambda item: (
                item.priority,
                -item.utilization,
            ),
            reverse=True,
        )

        return results[:limit]

    # --------------------------------------------------------
    # Availability
    # --------------------------------------------------------

    def _is_heartbeat_fresh(
        self,
        agent: AgentDefinition,
    ) -> bool:

        if agent.last_heartbeat_at is None:
            return False

        elapsed = (
            utc_now() - agent.last_heartbeat_at
        ).total_seconds()

        return (
            0 <= elapsed
            <= self.heartbeat_timeout_seconds
        )

    def _is_available(
        self,
        agent: AgentDefinition,
    ) -> bool:

        if not agent.enabled:
            return False

        if agent.status not in (
            AgentStatus.IDLE,
            AgentStatus.BUSY,
        ):
            return False

        if agent.health != AgentHealth.HEALTHY:
            return False

        if not self._is_heartbeat_fresh(agent):
            return False

        if agent.available_capacity <= 0:
            return False

        return True

    # --------------------------------------------------------
    # Update Status
    # --------------------------------------------------------

    async def update_status(
        self,
        *,
        scope: AgentScope,
        agent_id: str,
        status: AgentStatus,
    ) -> AgentDefinition:

        async with self._lock:
            agent = await self.get_agent(
                scope=scope,
                agent_id=agent_id,
            )

            if (
                status in (
                    AgentStatus.OFFLINE,
                    AgentStatus.FAILED,
                    AgentStatus.DISABLED,
                )
                and agent.active_tasks > 0
            ):
                raise AgentStateConflictError(
                    "Agent has active tasks"
                )

            agent.status = status
            agent.updated_at = utc_now()

            return await self.repository.save(agent)

    # --------------------------------------------------------
    # Update Health
    # --------------------------------------------------------

    async def update_health(
        self,
        *,
        scope: AgentScope,
        agent_id: str,
        health: AgentHealth,
    ) -> AgentDefinition:

        async with self._lock:
            agent = await self.get_agent(
                scope=scope,
                agent_id=agent_id,
            )

            agent.health = health
            agent.updated_at = utc_now()

            if health == AgentHealth.UNHEALTHY:
                agent.status = AgentStatus.DEGRADED

            return await self.repository.save(agent)

    # --------------------------------------------------------
    # Heartbeat
    # --------------------------------------------------------

    async def heartbeat(
        self,
        *,
        scope: AgentScope,
        agent_id: str,
        health: AgentHealth = AgentHealth.HEALTHY,
    ) -> AgentDefinition:

        async with self._lock:
            agent = await self.get_agent(
                scope=scope,
                agent_id=agent_id,
            )

            if not agent.enabled:
                raise AgentUnavailableError(
                    "Agent is disabled"
                )

            if agent.status == AgentStatus.FAILED:
                raise AgentStateConflictError(
                    "Failed agent requires recovery"
                )

            agent.last_heartbeat_at = utc_now()
            agent.health = health

            if health == AgentHealth.HEALTHY:
                agent.status = (
                    AgentStatus.BUSY
                    if agent.active_tasks > 0
                    else AgentStatus.IDLE
                )

            else:
                agent.status = AgentStatus.DEGRADED

            agent.updated_at = utc_now()

            return await self.repository.save(agent)

    # --------------------------------------------------------
    # Reserve Capacity
    # --------------------------------------------------------

    async def reserve_agent(
        self,
        *,
        scope: AgentScope,
        agent_id: str,
    ) -> AgentDefinition:

        async with self._lock:
            agent = await self.get_agent(
                scope=scope,
                agent_id=agent_id,
            )

            if not self._is_available(agent):
                raise AgentUnavailableError(
                    "Agent is not available"
                )

            agent.active_tasks += 1
            agent.status = AgentStatus.BUSY
            agent.updated_at = utc_now()

            return await self.repository.save(agent)

    # --------------------------------------------------------
    # Release Capacity
    # --------------------------------------------------------

    async def release_agent(
        self,
        *,
        scope: AgentScope,
        agent_id: str,
    ) -> AgentDefinition:

        async with self._lock:
            agent = await self.get_agent(
                scope=scope,
                agent_id=agent_id,
            )

            if agent.active_tasks <= 0:
                raise AgentStateConflictError(
                    "Agent has no reserved tasks"
                )

            agent.active_tasks -= 1

            if agent.enabled:
                agent.status = (
                    AgentStatus.BUSY
                    if agent.active_tasks > 0
                    else AgentStatus.IDLE
                )

            agent.updated_at = utc_now()

            return await self.repository.save(agent)

    # --------------------------------------------------------
    # Agent Selection
    # --------------------------------------------------------

    async def select_agent(
        self,
        *,
        scope: AgentScope,
        request: AgentSelectionRequest,
    ) -> AgentSelectionResult:

        agents = await self.repository.list(
            scope=scope
        )

        candidates: list[AgentSelectionResult] = []

        required_capabilities = set(
            request.required_capabilities
        )

        required_skills = set(
            request.required_skills
        )

        for agent in agents:

            if not self._is_available(agent):
                continue

            if not required_capabilities.issubset(
                set(agent.capabilities)
            ):
                continue

            if not required_skills.issubset(
                set(agent.skills)
            ):
                continue

            if (
                request.robot_type is not None
                and request.robot_type
                not in agent.supported_robot_types
            ):
                continue

            priority_score = agent.priority / 100

            capacity_score = (
                agent.available_capacity
                / agent.max_concurrent_tasks
            )

            performance_score = (
                agent.performance.success_rate
                if agent.performance.total_tasks > 0
                else 0.5
            )

            type_score = (
                1.0
                if request.preferred_agent_type is None
                or agent.agent_type
                == request.preferred_agent_type
                else 0.0
            )

            if request.strategy == SelectionStrategy.LEAST_BUSY:
                score = capacity_score

            elif (
                request.strategy
                == SelectionStrategy.HIGHEST_PRIORITY
            ):
                score = priority_score

            elif (
                request.strategy
                == SelectionStrategy.BEST_PERFORMANCE
            ):
                score = performance_score

            else:
                score = (
                    0.35 * capacity_score
                    + 0.25 * performance_score
                    + 0.25 * priority_score
                    + 0.15 * type_score
                )

            candidates.append(
                AgentSelectionResult(
                    agent=agent,
                    score=round(score, 6),
                    reason=(
                        "Matched required capabilities, "
                        "skills, health and capacity"
                    ),
                )
            )

        if not candidates:
            raise AgentUnavailableError(
                "No suitable agent available"
            )

        candidates.sort(
            key=lambda item: (
                item.score,
                item.agent.priority,
            ),
            reverse=True,
        )

        return candidates[0]

    # --------------------------------------------------------
    # Record Task Result
    # --------------------------------------------------------

    async def record_task_result(
        self,
        *,
        scope: AgentScope,
        agent_id: str,
        success: bool,
        latency_ms: float,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
    ) -> AgentDefinition:

        if (
            not math.isfinite(latency_ms)
            or latency_ms < 0
        ):
            raise AgentValidationError(
                "Invalid task latency"
            )

        if tokens_used < 0:
            raise AgentValidationError(
                "Tokens used cannot be negative"
            )

        if (
            not math.isfinite(cost_usd)
            or cost_usd < 0
        ):
            raise AgentValidationError(
                "Invalid task cost"
            )

        async with self._lock:
            agent = await self.get_agent(
                scope=scope,
                agent_id=agent_id,
            )

            metrics = agent.performance

            previous_tasks = metrics.total_tasks

            metrics.total_tasks += 1

            if success:
                metrics.completed_tasks += 1
            else:
                metrics.failed_tasks += 1

            metrics.average_latency_ms = (
                (
                    metrics.average_latency_ms
                    * previous_tasks
                )
                + latency_ms
            ) / metrics.total_tasks

            metrics.success_rate = (
                metrics.completed_tasks
                / metrics.total_tasks
            )

            metrics.total_tokens += tokens_used
            metrics.total_cost_usd += cost_usd
            metrics.last_task_at = utc_now()

            agent.updated_at = utc_now()

            return await self.repository.save(agent)

    # --------------------------------------------------------
    # Disable Agent
    # --------------------------------------------------------

    async def disable_agent(
        self,
        *,
        scope: AgentScope,
        agent_id: str,
    ) -> AgentDefinition:

        async with self._lock:
            agent = await self.get_agent(
                scope=scope,
                agent_id=agent_id,
            )

            if agent.active_tasks > 0:
                raise AgentStateConflictError(
                    "Cannot disable agent with active tasks"
                )

            agent.enabled = False
            agent.status = AgentStatus.DISABLED
            agent.updated_at = utc_now()

            return await self.repository.save(agent)

    # --------------------------------------------------------
    # Enable Agent
    # --------------------------------------------------------

    async def enable_agent(
        self,
        *,
        scope: AgentScope,
        agent_id: str,
    ) -> AgentDefinition:

        async with self._lock:
            agent = await self.get_agent(
                scope=scope,
                agent_id=agent_id,
            )

            agent.enabled = True
            agent.status = AgentStatus.REGISTERED
            agent.health = AgentHealth.UNKNOWN
            agent.last_heartbeat_at = None
            agent.updated_at = utc_now()

            return await self.repository.save(agent)

    # --------------------------------------------------------
    # Deregister Agent
    # --------------------------------------------------------

    async def deregister_agent(
        self,
        *,
        scope: AgentScope,
        agent_id: str,
    ) -> bool:

        async with self._lock:
            agent = await self.get_agent(
                scope=scope,
                agent_id=agent_id,
            )

            if agent.active_tasks > 0:
                raise AgentStateConflictError(
                    "Cannot deregister agent with active tasks"
                )

            return await self.repository.delete(
                scope=scope,
                agent_id=agent_id,
            )

    # --------------------------------------------------------
    # Expired Heartbeats
    # --------------------------------------------------------

    async def mark_stale_agents_offline(
        self,
        *,
        scope: AgentScope,
    ) -> int:

        async with self._lock:
            agents = await self.repository.list(
                scope=scope
            )

            updated = 0

            for agent in agents:

                if not agent.enabled:
                    continue

                if agent.status in (
                    AgentStatus.REGISTERED,
                    AgentStatus.INITIALIZING,
                    AgentStatus.DISABLED,
                    AgentStatus.FAILED,
                ):
                    continue

                if self._is_heartbeat_fresh(agent):
                    continue

                agent.status = AgentStatus.OFFLINE
                agent.health = AgentHealth.UNKNOWN
                agent.updated_at = utc_now()

                await self.repository.save(agent)

                updated += 1

            return updated

    # --------------------------------------------------------
    # Registry Statistics
    # --------------------------------------------------------

    async def get_statistics(
        self,
        *,
        scope: AgentScope,
    ) -> AgentRegistryStatistics:

        agents = await self.repository.list(
            scope=scope
        )

        total_agents = len(agents)

        enabled_agents = sum(
            agent.enabled
            for agent in agents
        )

        available_agents = sum(
            self._is_available(agent)
            for agent in agents
        )

        busy_agents = sum(
            agent.status == AgentStatus.BUSY
            for agent in agents
        )

        offline_agents = sum(
            agent.status == AgentStatus.OFFLINE
            for agent in agents
        )

        failed_agents = sum(
            agent.status == AgentStatus.FAILED
            for agent in agents
        )

        total_active_tasks = sum(
            agent.active_tasks
            for agent in agents
        )

        total_capacity = sum(
            agent.max_concurrent_tasks
            for agent in agents
            if agent.enabled
        )

        average_utilization = (
            sum(
                agent.utilization
                for agent in agents
                if agent.enabled
            ) / enabled_agents
            if enabled_agents > 0
            else 0.0
        )

        return AgentRegistryStatistics(
            tenant_id=scope.tenant_id,
            total_agents=total_agents,
            enabled_agents=enabled_agents,
            available_agents=available_agents,
            busy_agents=busy_agents,
            offline_agents=offline_agents,
            failed_agents=failed_agents,
            total_active_tasks=total_active_tasks,
            total_capacity=total_capacity,
            average_utilization=round(
                average_utilization,
                6,
            ),
        )


# ============================================================
# Factory
# ============================================================

def create_agent_registry(
    *,
    repository: AgentRegistryRepository | None = None,
    heartbeat_timeout_seconds: int = (
        DEFAULT_HEARTBEAT_TIMEOUT_SECONDS
    ),
) -> AgentRegistry:

    return AgentRegistry(
        repository=(
            repository
            if repository is not None
            else InMemoryAgentRegistryRepository()
        ),
        heartbeat_timeout_seconds=(
            heartbeat_timeout_seconds
        ),
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "AgentType",
    "AgentStatus",
    "AgentHealth",
    "AgentCapability",
    "SelectionStrategy",
    "AgentScope",
    "AgentPerformance",
    "AgentDefinition",
    "AgentRegistrationRequest",
    "AgentSearchFilters",
    "AgentSelectionRequest",
    "AgentSelectionResult",
    "AgentRegistryStatistics",
    "AgentRegistryRepository",
    "InMemoryAgentRegistryRepository",
    "AgentRegistry",
    "AgentRegistryError",
    "AgentNotFoundError",
    "AgentAlreadyExistsError",
    "AgentCapacityError",
    "AgentUnavailableError",
    "AgentValidationError",
    "AgentStateConflictError",
    "create_agent_registry",
]
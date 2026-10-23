"""Robot Capabilities module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/robot_capabilities.py

Python 3.11+
Standard library only.

ROBOT CAPABILITIES — SIMULATION ONLY

Features:
- Robot capability profiles
- Built-in robotics capability catalog
- Sensors, actuators, AI models and software dependencies
- Capability registration and versioning
- Component health observations
- Capability dependency graphs
- Cycle and missing-dependency detection
- Capability readiness evaluation
- Capability enable/disable controls
- Capability task matching
- Multi-robot and tenant isolation
- Health observation expiration
- Audit logging
- JSON export

Safety:
    This module does not execute robot commands.
    Readiness means simulation eligibility only.
    It does not establish physical safety.

    Component health observations are supplied
    by the caller. Production systems must use
    authenticated, independently verified sources.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


# ============================================================
# EXCEPTIONS
# ============================================================

class RobotCapabilityError(Exception):
    pass


class CapabilityValidationError(RobotCapabilityError):
    pass


class CapabilityNotFound(RobotCapabilityError):
    pass


class CapabilityConflict(RobotCapabilityError):
    pass


class CapabilityPermissionDenied(RobotCapabilityError):
    pass


# ============================================================
# ENUMS
# ============================================================

class CapabilityCategory(str, Enum):
    NAVIGATION = "navigation"
    LOCALIZATION = "localization"
    PERCEPTION = "perception"
    OBJECT_DETECTION = "object_detection"
    OBJECT_TRACKING = "object_tracking"
    MAPPING = "mapping"
    SENSOR_FUSION = "sensor_fusion"
    MANIPULATION = "manipulation"
    GRASPING = "grasping"
    INSPECTION = "inspection"
    DIAGNOSTICS = "diagnostics"
    MAINTENANCE = "maintenance"
    REPAIR_ASSISTANCE = "repair_assistance"
    CHARGING = "charging"
    HUMAN_INTERACTION = "human_interaction"
    VOICE_INTERACTION = "voice_interaction"
    VLA_INFERENCE = "vla_inference"
    TASK_PLANNING = "task_planning"
    TELEMETRY = "telemetry"
    SIMULATION = "simulation"
    CUSTOM = "custom"


class ComponentType(str, Enum):
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    MODEL = "model"
    SOFTWARE = "software"


class ComponentStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULTED = "faulted"
    UNAVAILABLE = "unavailable"


class CapabilityStatus(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


class CapabilityRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CapabilityEventType(str, Enum):
    CAPABILITY_REGISTERED = "capability_registered"
    CAPABILITY_UPDATED = "capability_updated"
    COMPONENT_UPDATED = "component_updated"
    CAPABILITY_ENABLED = "capability_enabled"
    CAPABILITY_DISABLED = "capability_disabled"
    CAPABILITY_EVALUATED = "capability_evaluated"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}",
            value,
        )
    ):
        raise CapabilityValidationError(
            f"Invalid {label}"
        )

    return value


def validate_version(value: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"\d+\.\d+\.\d+",
            value,
        )
    ):
        raise CapabilityValidationError(
            "Version must use MAJOR.MINOR.PATCH"
        )

    return value


def positive(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise CapabilityValidationError(
            f"{label} must be finite and positive"
        )

    return float(value)


def validate_ids(
    values: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise CapabilityValidationError(
            f"{label} must be a tuple"
        )

    for value in values:
        validate_id(value, label)

    if len(values) != len(set(values)):
        raise CapabilityValidationError(
            f"Duplicate {label}"
        )

    return values


def validate_timestamp(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise CapabilityValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RobotCapabilitiesConfig:
    maximum_capabilities: int = 10000
    maximum_components: int = 50000
    maximum_audit_events: int = 20000

    component_timeout_seconds: float = 5.0

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_capabilities",
            "maximum_components",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise CapabilityValidationError(
                    f"{name} must be positive"
                )

        positive(
            self.component_timeout_seconds,
            "component_timeout_seconds",
        )

        if not self.simulation_only:
            raise CapabilityValidationError(
                "Physical robot capability control unsupported"
            )


# ============================================================
# CAPABILITY SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class RobotCapabilitySpec:
    tenant_id: str
    robot_id: str
    capability_id: str

    name: str
    category: CapabilityCategory

    version: str = "1.0.0"
    description: str = ""

    required_sensors: tuple[str, ...] = ()
    required_actuators: tuple[str, ...] = ()
    required_models: tuple[str, ...] = ()
    required_software: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()

    risk_level: CapabilityRiskLevel = (
        CapabilityRiskLevel.LOW
    )

    enabled: bool = True

    def __post_init__(self) -> None:
        for label in (
            "tenant_id",
            "robot_id",
            "capability_id",
        ):
            validate_id(
                getattr(self, label),
                label,
            )

        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or len(self.name) > 256
        ):
            raise CapabilityValidationError(
                "Invalid capability name"
            )

        if not isinstance(
            self.category,
            CapabilityCategory,
        ):
            raise CapabilityValidationError(
                "Invalid capability category"
            )

        if not isinstance(
            self.risk_level,
            CapabilityRiskLevel,
        ):
            raise CapabilityValidationError(
                "Invalid risk level"
            )

        if not isinstance(self.enabled, bool):
            raise CapabilityValidationError(
                "enabled must be boolean"
            )

        validate_version(self.version)

        for label in (
            "required_sensors",
            "required_actuators",
            "required_models",
            "required_software",
            "required_capabilities",
        ):
            validate_ids(
                getattr(self, label),
                label,
            )

        if (
            self.capability_id
            in self.required_capabilities
        ):
            raise CapabilityValidationError(
                "Capability cannot depend on itself"
            )


# ============================================================
# COMPONENT HEALTH
# ============================================================

@dataclass(frozen=True)
class RobotComponentObservation:
    tenant_id: str
    robot_id: str

    component_id: str
    component_type: ComponentType

    status: ComponentStatus

    sequence_number: int

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for label in (
            "tenant_id",
            "robot_id",
            "component_id",
        ):
            validate_id(
                getattr(self, label),
                label,
            )

        if not isinstance(
            self.component_type,
            ComponentType,
        ):
            raise CapabilityValidationError(
                "Invalid component type"
            )

        if not isinstance(
            self.status,
            ComponentStatus,
        ):
            raise CapabilityValidationError(
                "Invalid component status"
            )

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise CapabilityValidationError(
                "Invalid sequence number"
            )

        validate_timestamp(self.timestamp)


@dataclass
class ComponentRecord:
    observation: RobotComponentObservation
    received_monotonic: float

    def to_dict(self) -> dict[str, Any]:
        observation = self.observation

        return {
            "tenant_id": observation.tenant_id,
            "robot_id": observation.robot_id,
            "component_id": observation.component_id,
            "component_type": (
                observation.component_type.value
            ),
            "status": observation.status.value,
            "sequence_number": (
                observation.sequence_number
            ),
            "timestamp": (
                observation.timestamp.isoformat()
            ),
        }


# ============================================================
# CAPABILITY RECORD
# ============================================================

@dataclass
class CapabilityRecord:
    specification: RobotCapabilitySpec

    enabled: bool = True

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    evaluation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        spec = self.specification

        return {
            "tenant_id": spec.tenant_id,
            "robot_id": spec.robot_id,
            "capability_id": spec.capability_id,
            "name": spec.name,
            "category": spec.category.value,
            "version": spec.version,
            "description": spec.description,
            "risk_level": spec.risk_level.value,
            "enabled": self.enabled,
            "required_sensors": list(
                spec.required_sensors
            ),
            "required_actuators": list(
                spec.required_actuators
            ),
            "required_models": list(
                spec.required_models
            ),
            "required_software": list(
                spec.required_software
            ),
            "required_capabilities": list(
                spec.required_capabilities
            ),
            "evaluation_count": (
                self.evaluation_count
            ),
            "created_at": (
                self.created_at.isoformat()
            ),
            "updated_at": (
                self.updated_at.isoformat()
            ),
        }


# ============================================================
# CAPABILITY EVALUATION
# ============================================================

@dataclass(frozen=True)
class CapabilityEvaluation:
    tenant_id: str
    robot_id: str
    capability_id: str

    status: CapabilityStatus

    missing_components: tuple[str, ...]
    degraded_components: tuple[str, ...]
    faulted_components: tuple[str, ...]

    unavailable_dependencies: tuple[str, ...]

    ready_for_simulation: bool

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "capability_id": self.capability_id,
            "status": self.status.value,
            "missing_components": list(
                self.missing_components
            ),
            "degraded_components": list(
                self.degraded_components
            ),
            "faulted_components": list(
                self.faulted_components
            ),
            "unavailable_dependencies": list(
                self.unavailable_dependencies
            ),
            "ready_for_simulation": (
                self.ready_for_simulation
            ),
            "timestamp": (
                self.timestamp.isoformat()
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class CapabilityAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    resource_id: str

    event_type: CapabilityEventType

    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "resource_id": self.resource_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# CAPABILITY MANAGER
# ============================================================

class RoboticsCapabilityManager:

    """
    Robot capability definition and readiness manager.

    A capability is ready only when:
    - It is enabled.
    - All required components are healthy.
    - Component observations are fresh.
    - All required capabilities are ready.
    - No dependency cycle exists.

    Readiness is simulation-only.
    """

    def __init__(
        self,
        config: RobotCapabilitiesConfig | None = None,
    ) -> None:
        self.config = (
            config or RobotCapabilitiesConfig()
        )

        self._capabilities: dict[
            tuple[str, str, str],
            CapabilityRecord
        ] = {}

        self._components: dict[
            tuple[str, str, ComponentType, str],
            ComponentRecord
        ] = {}

        self._audit_events: deque[
            CapabilityAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # KEYS
    # --------------------------------------------------------

    def _capability_key(
        self,
        tenant_id: str,
        robot_id: str,
        capability_id: str,
    ) -> tuple[str, str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
            validate_id(capability_id, "capability_id"),
        )

    def _component_key(
        self,
        tenant_id: str,
        robot_id: str,
        component_type: ComponentType,
        component_id: str,
    ) -> tuple[
        str,
        str,
        ComponentType,
        str,
    ]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
            component_type,
            validate_id(component_id, "component_id"),
        )

    def _get_capability(
        self,
        tenant_id: str,
        robot_id: str,
        capability_id: str,
    ) -> CapabilityRecord:
        record = self._capabilities.get(
            self._capability_key(
                tenant_id,
                robot_id,
                capability_id,
            )
        )

        if record is None:
            raise CapabilityNotFound(
                f"Capability not found: {capability_id}"
            )

        return record

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        resource_id: str,
        event_type: CapabilityEventType,
        details: str,
    ) -> None:
        self._audit_events.append(
            CapabilityAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=resource_id,
                event_type=event_type,
                details=details,
            )
        )

    # --------------------------------------------------------
    # REGISTER CAPABILITY
    # --------------------------------------------------------

    def register_capability(
        self,
        specification: RobotCapabilitySpec,
    ) -> dict[str, Any]:
        if not isinstance(
            specification,
            RobotCapabilitySpec,
        ):
            raise CapabilityValidationError(
                "Invalid capability specification"
            )

        with self._lock:
            key = self._capability_key(
                specification.tenant_id,
                specification.robot_id,
                specification.capability_id,
            )

            if key in self._capabilities:
                raise CapabilityConflict(
                    "Capability already registered"
                )

            if (
                len(self._capabilities)
                >= self.config.maximum_capabilities
            ):
                raise RobotCapabilityError(
                    "Maximum capability capacity reached"
                )

            record = CapabilityRecord(
                specification=specification,
                enabled=specification.enabled,
            )

            self._capabilities[key] = record

            self._audit(
                tenant_id=specification.tenant_id,
                robot_id=specification.robot_id,
                resource_id=specification.capability_id,
                event_type=(
                    CapabilityEventType.CAPABILITY_REGISTERED
                ),
                details="Capability registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # INGEST COMPONENT HEALTH
    # --------------------------------------------------------

    def ingest_component_health(
        self,
        observation: RobotComponentObservation,
    ) -> dict[str, Any]:
        if not isinstance(
            observation,
            RobotComponentObservation,
        ):
            raise CapabilityValidationError(
                "Invalid component observation"
            )

        with self._lock:
            key = self._component_key(
                observation.tenant_id,
                observation.robot_id,
                observation.component_type,
                observation.component_id,
            )

            previous = self._components.get(key)

            if (
                previous is not None
                and observation.sequence_number
                <= previous.observation.sequence_number
            ):
                raise CapabilityValidationError(
                    "Duplicate or out-of-order observation"
                )

            now = utc_now()

            age = (
                now
                - validate_timestamp(
                    observation.timestamp
                )
            ).total_seconds()

            if age < 0:
                raise CapabilityValidationError(
                    "Future observation rejected"
                )

            if (
                age
                > self.config.component_timeout_seconds
            ):
                raise CapabilityValidationError(
                    "Stale observation rejected"
                )

            if (
                previous is None
                and len(self._components)
                >= self.config.maximum_components
            ):
                raise RobotCapabilityError(
                    "Maximum component capacity reached"
                )

            record = ComponentRecord(
                observation=observation,
                received_monotonic=time.monotonic(),
            )

            self._components[key] = record

            self._audit(
                tenant_id=observation.tenant_id,
                robot_id=observation.robot_id,
                resource_id=observation.component_id,
                event_type=(
                    CapabilityEventType.COMPONENT_UPDATED
                ),
                details=(
                    f"{observation.component_type.value}:"
                    f"{observation.status.value}"
                ),
            )

            return record.to_dict()

    # --------------------------------------------------------
    # COMPONENT STATUS
    # --------------------------------------------------------

    def _component_status(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        component_type: ComponentType,
        component_id: str,
    ) -> ComponentStatus | None:
        record = self._components.get(
            self._component_key(
                tenant_id,
                robot_id,
                component_type,
                component_id,
            )
        )

        if record is None:
            return None

        age = (
            time.monotonic()
            - record.received_monotonic
        )

        if (
            age
            > self.config.component_timeout_seconds
        ):
            return None

        return record.observation.status

    # --------------------------------------------------------
    # DEPENDENCY GRAPH
    # --------------------------------------------------------

    def validate_dependency_graph(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            graph = {
                record.specification.capability_id: (
                    record.specification.required_capabilities
                )
                for record in self._capabilities.values()
                if (
                    record.specification.tenant_id
                    == tenant_id
                    and record.specification.robot_id
                    == robot_id
                )
            }

            visiting: set[str] = set()
            visited: set[str] = set()

            missing: set[str] = set()
            cycles: set[str] = set()

            def visit(node: str) -> None:
                if node in visiting:
                    cycles.add(node)
                    return

                if node in visited:
                    return

                visiting.add(node)

                for dependency in graph.get(node, ()):
                    if dependency not in graph:
                        missing.add(dependency)
                    else:
                        visit(dependency)

                visiting.remove(node)
                visited.add(node)

            for node in graph:
                visit(node)

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "valid": (
                    not missing
                    and not cycles
                ),
                "missing_dependencies": sorted(missing),
                "cycles": sorted(cycles),
            }

    # --------------------------------------------------------
    # INTERNAL EVALUATION
    # --------------------------------------------------------

    def _evaluate(
        self,
        record: CapabilityRecord,
        visiting: frozenset[str],
    ) -> CapabilityEvaluation:
        spec = record.specification

        if not record.enabled:
            return CapabilityEvaluation(
                tenant_id=spec.tenant_id,
                robot_id=spec.robot_id,
                capability_id=spec.capability_id,
                status=CapabilityStatus.DISABLED,
                missing_components=(),
                degraded_components=(),
                faulted_components=(),
                unavailable_dependencies=(),
                ready_for_simulation=False,
            )

        if spec.capability_id in visiting:
            return CapabilityEvaluation(
                tenant_id=spec.tenant_id,
                robot_id=spec.robot_id,
                capability_id=spec.capability_id,
                status=CapabilityStatus.UNAVAILABLE,
                missing_components=(),
                degraded_components=(),
                faulted_components=(),
                unavailable_dependencies=(
                    "dependency_cycle",
                ),
                ready_for_simulation=False,
            )

        visited = visiting | {
            spec.capability_id
        }

        missing: list[str] = []
        degraded: list[str] = []
        faulted: list[str] = []

        requirements = (
            (
                ComponentType.SENSOR,
                spec.required_sensors,
            ),
            (
                ComponentType.ACTUATOR,
                spec.required_actuators,
            ),
            (
                ComponentType.MODEL,
                spec.required_models,
            ),
            (
                ComponentType.SOFTWARE,
                spec.required_software,
            ),
        )

        for component_type, component_ids in requirements:
            for component_id in component_ids:
                status = self._component_status(
                    tenant_id=spec.tenant_id,
                    robot_id=spec.robot_id,
                    component_type=component_type,
                    component_id=component_id,
                )

                label = (
                    f"{component_type.value}:"
                    f"{component_id}"
                )

                if status is None:
                    missing.append(label)

                elif status == ComponentStatus.DEGRADED:
                    degraded.append(label)

                elif status in (
                    ComponentStatus.FAULTED,
                    ComponentStatus.UNAVAILABLE,
                ):
                    faulted.append(label)

        unavailable_dependencies: list[str] = []

        for dependency_id in spec.required_capabilities:
            key = self._capability_key(
                spec.tenant_id,
                spec.robot_id,
                dependency_id,
            )

            dependency = self._capabilities.get(key)

            if dependency is None:
                unavailable_dependencies.append(
                    dependency_id
                )
                continue

            evaluation = self._evaluate(
                dependency,
                visited,
            )

            if not evaluation.ready_for_simulation:
                unavailable_dependencies.append(
                    dependency_id
                )

        if (
            missing
            or faulted
            or unavailable_dependencies
        ):
            status = CapabilityStatus.UNAVAILABLE

        elif degraded:
            status = CapabilityStatus.DEGRADED

        else:
            status = CapabilityStatus.READY

        return CapabilityEvaluation(
            tenant_id=spec.tenant_id,
            robot_id=spec.robot_id,
            capability_id=spec.capability_id,
            status=status,
            missing_components=tuple(missing),
            degraded_components=tuple(degraded),
            faulted_components=tuple(faulted),
            unavailable_dependencies=tuple(
                unavailable_dependencies
            ),
            ready_for_simulation=(
                status == CapabilityStatus.READY
            ),
        )

    # --------------------------------------------------------
    # PUBLIC EVALUATION
    # --------------------------------------------------------

    def evaluate_capability(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        capability_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get_capability(
                tenant_id,
                robot_id,
                capability_id,
            )

            evaluation = self._evaluate(
                record,
                frozenset(),
            )

            record.evaluation_count += 1

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=capability_id,
                event_type=(
                    CapabilityEventType.CAPABILITY_EVALUATED
                ),
                details=evaluation.status.value,
            )

            return evaluation.to_dict()

    # --------------------------------------------------------
    # ENABLE / DISABLE
    # --------------------------------------------------------

    def set_capability_enabled(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        capability_id: str,
        enabled: bool,
        authorized: bool,
    ) -> dict[str, Any]:
        if not isinstance(enabled, bool):
            raise CapabilityValidationError(
                "enabled must be boolean"
            )

        if authorized is not True:
            raise CapabilityPermissionDenied(
                "Authorization required"
            )

        with self._lock:
            record = self._get_capability(
                tenant_id,
                robot_id,
                capability_id,
            )

            record.enabled = enabled
            record.updated_at = utc_now()

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=capability_id,
                event_type=(
                    CapabilityEventType.CAPABILITY_ENABLED
                    if enabled
                    else CapabilityEventType.CAPABILITY_DISABLED
                ),
                details=(
                    "Capability enabled"
                    if enabled
                    else "Capability disabled"
                ),
            )

            return record.to_dict()

    # --------------------------------------------------------
    # MATCH TASK REQUIREMENTS
    # --------------------------------------------------------

    def match_task(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        required_capabilities: tuple[str, ...],
    ) -> dict[str, Any]:
        validate_ids(
            required_capabilities,
            "required_capabilities",
        )

        with self._lock:
            evaluations = []
            missing = []

            for capability_id in required_capabilities:
                key = self._capability_key(
                    tenant_id,
                    robot_id,
                    capability_id,
                )

                record = self._capabilities.get(key)

                if record is None:
                    missing.append(capability_id)
                    continue

                evaluations.append(
                    self._evaluate(
                        record,
                        frozenset(),
                    ).to_dict()
                )

            ready = (
                not missing
                and all(
                    item["ready_for_simulation"]
                    for item in evaluations
                )
            )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "required_capabilities": list(
                    required_capabilities
                ),
                "eligible_for_simulation": ready,
                "missing_capabilities": missing,
                "evaluations": evaluations,
                "timestamp": utc_now().isoformat(),
            }

    # --------------------------------------------------------
    # GET CAPABILITY
    # --------------------------------------------------------

    def get_capability(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        capability_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get_capability(
                tenant_id,
                robot_id,
                capability_id,
            )

            return {
                "definition": record.to_dict(),
                "evaluation": self._evaluate(
                    record,
                    frozenset(),
                ).to_dict(),
            }

    # --------------------------------------------------------
    # LIST CAPABILITIES
    # --------------------------------------------------------

    def list_capabilities(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(
                [
                    self.get_capability(
                        tenant_id=tenant_id,
                        robot_id=robot_id,
                        capability_id=(
                            record.specification.capability_id
                        ),
                    )
                    for record in self._capabilities.values()
                    if (
                        record.specification.tenant_id
                        == tenant_id
                        and record.specification.robot_id
                        == robot_id
                    )
                ],
                key=lambda item: (
                    item["definition"]["capability_id"]
                ),
            )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        capabilities = self.list_capabilities(
            tenant_id=tenant_id,
            robot_id=robot_id,
        )

        counts = {
            status.value: 0
            for status in CapabilityStatus
        }

        for item in capabilities:
            counts[
                item["evaluation"]["status"]
            ] += 1

        return {
            "total_capabilities": len(capabilities),
            "status_counts": counts,
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id,
                    robot_id=robot_id,
                ),
                "capabilities": self.list_capabilities(
                    tenant_id=tenant_id,
                    robot_id=robot_id,
                ),
                "components": [
                    record.to_dict()
                    for key, record in self._components.items()
                    if (
                        key[0] == tenant_id
                        and key[1] == robot_id
                    )
                ],
                "audit_events": [
                    event.to_dict()
                    for event in self._audit_events
                    if (
                        event.tenant_id == tenant_id
                        and event.robot_id == robot_id
                    )
                ],
            }

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
# BUILT-IN CAPABILITY CATALOG
# ============================================================

def build_default_capabilities(
    *,
    tenant_id: str,
    robot_id: str,
) -> tuple[RobotCapabilitySpec, ...]:
    """
    Default capability definitions.

    Component identifiers must match the IDs
    used by your simulation and model services.
    """

    common = {
        "tenant_id": tenant_id,
        "robot_id": robot_id,
    }

    return (
        RobotCapabilitySpec(
            **common,
            capability_id="object_detection",
            name="Object Detection",
            category=CapabilityCategory.OBJECT_DETECTION,
            required_sensors=("camera",),
            required_models=("object_detector",),
            risk_level=CapabilityRiskLevel.LOW,
        ),

        RobotCapabilitySpec(
            **common,
            capability_id="localization",
            name="Robot Localization",
            category=CapabilityCategory.LOCALIZATION,
            required_sensors=("lidar", "imu"),
            required_models=("localization_model",),
            risk_level=CapabilityRiskLevel.MEDIUM,
        ),

        RobotCapabilitySpec(
            **common,
            capability_id="navigation",
            name="Autonomous Navigation",
            category=CapabilityCategory.NAVIGATION,
            required_sensors=("lidar", "imu"),
            required_actuators=("drive_base",),
            required_software=("navigation_stack",),
            required_capabilities=("localization",),
            risk_level=CapabilityRiskLevel.HIGH,
        ),

        RobotCapabilitySpec(
            **common,
            capability_id="manipulation",
            name="Robot Arm Manipulation",
            category=CapabilityCategory.MANIPULATION,
            required_actuators=("robot_arm",),
            required_software=("motion_planner",),
            risk_level=CapabilityRiskLevel.HIGH,
        ),

        RobotCapabilitySpec(
            **common,
            capability_id="grasping",
            name="Object Grasping",
            category=CapabilityCategory.GRASPING,
            required_sensors=("camera",),
            required_actuators=("gripper",),
            required_models=("grasp_model",),
            required_capabilities=(
                "object_detection",
                "manipulation",
            ),
            risk_level=CapabilityRiskLevel.HIGH,
        ),

        RobotCapabilitySpec(
            **common,
            capability_id="inspection",
            name="Visual Inspection",
            category=CapabilityCategory.INSPECTION,
            required_sensors=("camera",),
            required_models=("inspection_model",),
            required_capabilities=("object_detection",),
            risk_level=CapabilityRiskLevel.MEDIUM,
        ),

        RobotCapabilitySpec(
            **common,
            capability_id="diagnostics",
            name="Robot Diagnostics",
            category=CapabilityCategory.DIAGNOSTICS,
            required_sensors=("telemetry",),
            required_models=("diagnostics_model",),
            risk_level=CapabilityRiskLevel.LOW,
        ),

        RobotCapabilitySpec(
            **common,
            capability_id="vla_inference",
            name="Vision Language Action Inference",
            category=CapabilityCategory.VLA_INFERENCE,
            required_sensors=("camera",),
            required_models=("vla_policy",),
            required_software=("vla_runtime",),
            risk_level=CapabilityRiskLevel.HIGH,
        ),
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    manager = RoboticsCapabilityManager()

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"

    definitions = build_default_capabilities(
        tenant_id=tenant_id,
        robot_id=robot_id,
    )

    for definition in definitions:
        manager.register_capability(
            definition
        )

    components = (
        (ComponentType.SENSOR, "camera"),
        (ComponentType.SENSOR, "lidar"),
        (ComponentType.SENSOR, "imu"),
        (ComponentType.ACTUATOR, "drive_base"),
        (ComponentType.MODEL, "object_detector"),
        (ComponentType.MODEL, "localization_model"),
        (ComponentType.SOFTWARE, "navigation_stack"),
    )

    for component_type, component_id in components:
        manager.ingest_component_health(
            RobotComponentObservation(
                tenant_id=tenant_id,
                robot_id=robot_id,
                component_id=component_id,
                component_type=component_type,
                status=ComponentStatus.HEALTHY,
                sequence_number=1,
            )
        )

    print(
        "Navigation capability:",
        json.dumps(
            manager.evaluate_capability(
                tenant_id=tenant_id,
                robot_id=robot_id,
                capability_id="navigation",
            ),
            indent=2,
        ),
    )

    print(
        "\nTask matching:",
        json.dumps(
            manager.match_task(
                tenant_id=tenant_id,
                robot_id=robot_id,
                required_capabilities=(
                    "navigation",
                    "object_detection",
                ),
            ),
            indent=2,
        ),
    )

    report = manager.export_json(
        "data/robotics/robot_capabilities_report.json",
        tenant_id=tenant_id,
        robot_id=robot_id,
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()

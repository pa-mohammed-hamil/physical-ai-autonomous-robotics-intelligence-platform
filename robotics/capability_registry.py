"""Capability Registry module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/capability_registry.py

Python 3.11+
Standard library only.

ROBOT CAPABILITY REGISTRY — SIMULATION ONLY

Features:
- Robot capability registration
- Capability discovery and filtering
- Capability availability monitoring
- Required sensors, actuators and models
- Capability dependency validation
- Capability health and readiness
- Task requirement matching
- Simulation-only task eligibility
- Multi-robot capability discovery
- Tenant and robot isolation
- Versioned capability definitions
- Thread-safe registry
- Audit logging and JSON export

Safety:
    This module does not execute robot tasks,
    move actuators, or control physical hardware.

    Capability readiness is informational and
    does not replace independent safety checks.

    Health and authorization must be supplied
    by trusted external systems.
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
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class CapabilityRegistryError(Exception):
    pass


class CapabilityValidationError(CapabilityRegistryError):
    pass


class CapabilityNotFound(CapabilityRegistryError):
    pass


class CapabilityConflict(CapabilityRegistryError):
    pass


class CapabilityPermissionDenied(CapabilityRegistryError):
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
    MANIPULATION = "manipulation"
    GRASPING = "grasping"
    INSPECTION = "inspection"
    MAINTENANCE = "maintenance"
    REPAIR_ASSISTANCE = "repair_assistance"
    AUTONOMOUS_CHARGING = "autonomous_charging"
    HUMAN_INTERACTION = "human_interaction"
    VOICE_INTERACTION = "voice_interaction"
    VLA_INFERENCE = "vla_inference"
    PLANNING = "planning"
    TELEMETRY = "telemetry"
    DIAGNOSTICS = "diagnostics"
    SIMULATION = "simulation"
    OTHER = "other"


class CapabilityStatus(str, Enum):
    REGISTERED = "registered"
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    FAULTED = "faulted"
    STALE = "stale"


class CapabilityRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CapabilityEventType(str, Enum):
    REGISTERED = "registered"
    UPDATED = "updated"
    HEALTH_UPDATED = "health_updated"
    BECAME_STALE = "became_stale"
    FAULT_LATCHED = "fault_latched"
    RECOVERED = "recovered"
    TASK_MATCHED = "task_matched"
    TASK_REJECTED = "task_rejected"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not re.fullmatch(
            r"[A-Za-z0-9_.-]+",
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

    if len(set(values)) != len(values):
        raise CapabilityValidationError(
            f"Duplicate {label}"
        )

    return values


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CapabilityRegistryConfig:
    maximum_capabilities: int = 10000
    maximum_audit_events: int = 20000

    default_health_timeout_seconds: float = 5.0

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for label in (
            "maximum_capabilities",
            "maximum_audit_events",
        ):
            value = getattr(self, label)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise CapabilityValidationError(
                    f"{label} must be positive"
                )

        positive(
            self.default_health_timeout_seconds,
            "default_health_timeout_seconds",
        )

        if not self.simulation_only:
            raise CapabilityValidationError(
                "Physical capability execution unsupported"
            )


# ============================================================
# CAPABILITY DEFINITION
# ============================================================

@dataclass(frozen=True)
class RobotCapability:
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
    required_capabilities: tuple[str, ...] = ()

    risk_level: CapabilityRiskLevel = (
        CapabilityRiskLevel.LOW
    )

    health_timeout_seconds: float = 5.0

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
                "Invalid capability risk level"
            )

        validate_version(self.version)

        positive(
            self.health_timeout_seconds,
            "health_timeout_seconds",
        )

        for label in (
            "required_sensors",
            "required_actuators",
            "required_models",
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
# HEALTH OBSERVATION
# ============================================================

@dataclass(frozen=True)
class CapabilityHealthObservation:
    tenant_id: str
    robot_id: str
    capability_id: str

    sequence_number: int

    operational: bool

    healthy_sensors: tuple[str, ...] = ()
    healthy_actuators: tuple[str, ...] = ()
    available_models: tuple[str, ...] = ()

    degraded: bool = False

    timestamp: datetime = field(
        default_factory=utc_now
    )

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
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise CapabilityValidationError(
                "Invalid sequence number"
            )

        if (
            not isinstance(self.operational, bool)
            or not isinstance(self.degraded, bool)
        ):
            raise CapabilityValidationError(
                "Invalid health flags"
            )

        for label in (
            "healthy_sensors",
            "healthy_actuators",
            "available_models",
        ):
            validate_ids(
                getattr(self, label),
                label,
            )

        validate_timestamp(self.timestamp)


# ============================================================
# TASK REQUIREMENTS
# ============================================================

@dataclass(frozen=True)
class CapabilityTaskRequirement:
    tenant_id: str
    task_id: str

    required_capabilities: tuple[str, ...]

    preferred_robot_id: str | None = None

    maximum_risk_level: CapabilityRiskLevel = (
        CapabilityRiskLevel.MEDIUM
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.task_id, "task_id")

        validate_ids(
            self.required_capabilities,
            "required_capabilities",
        )

        if not self.required_capabilities:
            raise CapabilityValidationError(
                "At least one capability is required"
            )

        if self.preferred_robot_id is not None:
            validate_id(
                self.preferred_robot_id,
                "preferred_robot_id",
            )

        if not isinstance(
            self.maximum_risk_level,
            CapabilityRiskLevel,
        ):
            raise CapabilityValidationError(
                "Invalid maximum risk level"
            )


# ============================================================
# CAPABILITY RECORD
# ============================================================

@dataclass
class CapabilityRecord:
    definition: RobotCapability

    status: CapabilityStatus = (
        CapabilityStatus.REGISTERED
    )

    last_health: CapabilityHealthObservation | None = None

    last_received_monotonic: float | None = None

    last_sequence_number: int = -1

    fault_latched: bool = False

    fault_reasons: list[str] = field(
        default_factory=list
    )

    health_update_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        definition = self.definition

        return {
            "tenant_id": definition.tenant_id,
            "robot_id": definition.robot_id,
            "capability_id": definition.capability_id,
            "name": definition.name,
            "category": definition.category.value,
            "version": definition.version,
            "description": definition.description,
            "risk_level": definition.risk_level.value,
            "status": self.status.value,
            "required_sensors": list(
                definition.required_sensors
            ),
            "required_actuators": list(
                definition.required_actuators
            ),
            "required_models": list(
                definition.required_models
            ),
            "required_capabilities": list(
                definition.required_capabilities
            ),
            "fault_latched": self.fault_latched,
            "fault_reasons": list(
                self.fault_reasons
            ),
            "health_update_count": (
                self.health_update_count
            ),
            "last_health_at": (
                self.last_health.timestamp.isoformat()
                if self.last_health
                else None
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
    capability_id: str

    event_type: CapabilityEventType

    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "capability_id": self.capability_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# REGISTRY
# ============================================================

class RoboticsCapabilityRegistry:

    RISK_ORDER = {
        CapabilityRiskLevel.LOW: 0,
        CapabilityRiskLevel.MEDIUM: 1,
        CapabilityRiskLevel.HIGH: 2,
        CapabilityRiskLevel.CRITICAL: 3,
    }

    def __init__(
        self,
        config: CapabilityRegistryConfig | None = None,
    ) -> None:
        self.config = (
            config or CapabilityRegistryConfig()
        )

        self.capabilities: dict[
            tuple[str, str, str],
            CapabilityRecord
        ] = {}

        self.audit_events: deque[
            CapabilityAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def _key(
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

    def _get(
        self,
        tenant_id: str,
        robot_id: str,
        capability_id: str,
    ) -> CapabilityRecord:
        record = self.capabilities.get(
            self._key(
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

    def _audit(
        self,
        record: CapabilityRecord,
        event_type: CapabilityEventType,
        details: str,
    ) -> None:
        definition = record.definition

        self.audit_events.append(
            CapabilityAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=definition.tenant_id,
                robot_id=definition.robot_id,
                capability_id=definition.capability_id,
                event_type=event_type,
                details=details,
            )
        )

    # --------------------------------------------------------
    # REGISTER
    # --------------------------------------------------------

    def register_capability(
        self,
        definition: RobotCapability,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.robot_id,
                definition.capability_id,
            )

            if key in self.capabilities:
                raise CapabilityConflict(
                    "Capability already registered"
                )

            if (
                len(self.capabilities)
                >= self.config.maximum_capabilities
            ):
                raise CapabilityRegistryError(
                    "Maximum capability capacity reached"
                )

            record = CapabilityRecord(
                definition=definition
            )

            self.capabilities[key] = record

            self._audit(
                record,
                CapabilityEventType.REGISTERED,
                "Capability registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # DEPENDENCY VALIDATION
    # --------------------------------------------------------

    def validate_dependency_graph(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            graph = {
                record.definition.capability_id: (
                    record.definition.required_capabilities
                )
                for record in self.capabilities.values()
                if (
                    record.definition.tenant_id == tenant_id
                    and record.definition.robot_id == robot_id
                )
            }

            visiting: set[str] = set()
            visited: set[str] = set()

            missing: set[str] = set()
            cycles: list[str] = []

            def visit(node: str) -> None:
                if node in visiting:
                    cycles.append(node)
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
                "missing_dependencies": sorted(
                    missing
                ),
                "cycles": sorted(
                    set(cycles)
                ),
            }

    # --------------------------------------------------------
    # UPDATE HEALTH
    # --------------------------------------------------------

    def update_health(
        self,
        observation: CapabilityHealthObservation,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                observation.tenant_id,
                observation.robot_id,
                observation.capability_id,
            )

            if (
                observation.sequence_number
                <= record.last_sequence_number
            ):
                raise CapabilityValidationError(
                    "Duplicate or out-of-order health update"
                )

            timestamp = validate_timestamp(
                observation.timestamp
            )

            now = utc_now()

            if timestamp > now:
                raise CapabilityValidationError(
                    "Future health observation rejected"
                )

            age = (
                now - timestamp
            ).total_seconds()

            if (
                age
                > record.definition.health_timeout_seconds
            ):
                raise CapabilityValidationError(
                    "Stale health observation rejected"
                )

            record.last_sequence_number = (
                observation.sequence_number
            )

            record.last_health = observation

            record.last_received_monotonic = (
                time.monotonic()
            )

            record.health_update_count += 1

            definition = record.definition

            missing_sensors = (
                set(definition.required_sensors)
                - set(observation.healthy_sensors)
            )

            missing_actuators = (
                set(definition.required_actuators)
                - set(observation.healthy_actuators)
            )

            missing_models = (
                set(definition.required_models)
                - set(observation.available_models)
            )

            if not observation.operational:
                record.fault_latched = True

                if (
                    "not_operational"
                    not in record.fault_reasons
                ):
                    record.fault_reasons.append(
                        "not_operational"
                    )

                record.status = CapabilityStatus.FAULTED

                self._audit(
                    record,
                    CapabilityEventType.FAULT_LATCHED,
                    "Capability operational failure",
                )

            elif record.fault_latched:
                record.status = CapabilityStatus.FAULTED

            elif (
                missing_sensors
                or missing_actuators
                or missing_models
            ):
                record.status = CapabilityStatus.UNAVAILABLE

            elif observation.degraded:
                record.status = CapabilityStatus.DEGRADED

            else:
                record.status = CapabilityStatus.AVAILABLE

            self._audit(
                record,
                CapabilityEventType.HEALTH_UPDATED,
                "Capability health updated",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # REFRESH
    # --------------------------------------------------------

    def _refresh(
        self,
        record: CapabilityRecord,
    ) -> None:
        last = record.last_received_monotonic

        if last is None:
            return

        age = time.monotonic() - last

        if (
            age
            > record.definition.health_timeout_seconds
            and record.status
            not in (
                CapabilityStatus.STALE,
                CapabilityStatus.FAULTED,
            )
        ):
            record.status = CapabilityStatus.STALE

            self._audit(
                record,
                CapabilityEventType.BECAME_STALE,
                "Capability health observation expired",
            )

    # --------------------------------------------------------
    # CAPABILITY READINESS
    # --------------------------------------------------------

    def _ready(
        self,
        record: CapabilityRecord,
        visited: set[str] | None = None,
    ) -> bool:
        self._refresh(record)

        if record.status != CapabilityStatus.AVAILABLE:
            return False

        visited = set(visited or ())

        capability_id = (
            record.definition.capability_id
        )

        if capability_id in visited:
            return False

        visited.add(capability_id)

        definition = record.definition

        for dependency_id in (
            definition.required_capabilities
        ):
            try:
                dependency = self._get(
                    definition.tenant_id,
                    definition.robot_id,
                    dependency_id,
                )
            except CapabilityNotFound:
                return False

            if not self._ready(
                dependency,
                visited,
            ):
                return False

        return True

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
            record = self._get(
                tenant_id,
                robot_id,
                capability_id,
            )

            result = record.to_dict()

            result["ready_for_simulation"] = (
                self._ready(record)
            )

            return result

    # --------------------------------------------------------
    # DISCOVER CAPABILITIES
    # --------------------------------------------------------

    def discover(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        category: CapabilityCategory | None = None,
        available_only: bool = False,
    ) -> list[dict[str, Any]]:
        with self._lock:
            results = []

            for record in self.capabilities.values():
                definition = record.definition

                if definition.tenant_id != tenant_id:
                    continue

                if (
                    robot_id is not None
                    and definition.robot_id != robot_id
                ):
                    continue

                if (
                    category is not None
                    and definition.category != category
                ):
                    continue

                ready = self._ready(record)

                if available_only and not ready:
                    continue

                result = record.to_dict()

                result["ready_for_simulation"] = ready

                results.append(result)

            return sorted(
                results,
                key=lambda item: (
                    item["robot_id"],
                    item["capability_id"],
                ),
            )

    # --------------------------------------------------------
    # MATCH TASK
    # --------------------------------------------------------

    def match_task(
        self,
        requirement: CapabilityTaskRequirement,
    ) -> dict[str, Any]:
        with self._lock:
            robots = sorted({
                record.definition.robot_id
                for record in self.capabilities.values()
                if record.definition.tenant_id
                == requirement.tenant_id
            })

            if requirement.preferred_robot_id:
                robots = [
                    robot_id
                    for robot_id in robots
                    if robot_id
                    == requirement.preferred_robot_id
                ]

            matches = []

            for robot_id in robots:
                missing = []
                unavailable = []
                risk_exceeded = []

                for capability_id in (
                    requirement.required_capabilities
                ):
                    try:
                        record = self._get(
                            requirement.tenant_id,
                            robot_id,
                            capability_id,
                        )
                    except CapabilityNotFound:
                        missing.append(capability_id)
                        continue

                    if (
                        self.RISK_ORDER[
                            record.definition.risk_level
                        ]
                        > self.RISK_ORDER[
                            requirement.maximum_risk_level
                        ]
                    ):
                        risk_exceeded.append(
                            capability_id
                        )

                    if not self._ready(record):
                        unavailable.append(
                            capability_id
                        )

                eligible = not (
                    missing
                    or unavailable
                    or risk_exceeded
                )

                matches.append({
                    "robot_id": robot_id,
                    "eligible_for_simulation": eligible,
                    "missing_capabilities": missing,
                    "unavailable_capabilities": unavailable,
                    "risk_exceeded_capabilities": risk_exceeded,
                })

            eligible_robots = [
                item["robot_id"]
                for item in matches
                if item["eligible_for_simulation"]
            ]

            return {
                "tenant_id": requirement.tenant_id,
                "task_id": requirement.task_id,
                "required_capabilities": list(
                    requirement.required_capabilities
                ),
                "eligible_robots": eligible_robots,
                "matches": matches,
                "simulation_only": True,
                "evaluated_at": utc_now().isoformat(),
            }

    # --------------------------------------------------------
    # RECOVER CAPABILITY
    # --------------------------------------------------------

    def recover_capability(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        capability_id: str,
        authorized: bool,
        independently_verified: bool,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                capability_id,
            )

            if (
                authorized is not True
                or independently_verified is not True
            ):
                raise CapabilityPermissionDenied(
                    "Verified recovery authorization required"
                )

            if record.last_health is None:
                raise CapabilityValidationError(
                    "Fresh health observation required"
                )

            if (
                record.last_received_monotonic is None
                or (
                    time.monotonic()
                    - record.last_received_monotonic
                )
                > record.definition.health_timeout_seconds
            ):
                raise CapabilityValidationError(
                    "Health observation is stale"
                )

            observation = record.last_health

            if (
                not observation.operational
                or observation.degraded
                or not set(
                    record.definition.required_sensors
                ).issubset(
                    observation.healthy_sensors
                )
                or not set(
                    record.definition.required_actuators
                ).issubset(
                    observation.healthy_actuators
                )
                or not set(
                    record.definition.required_models
                ).issubset(
                    observation.available_models
                )
            ):
                raise CapabilityValidationError(
                    "Capability is not healthy"
                )

            record.fault_latched = False
            record.fault_reasons.clear()

            record.status = CapabilityStatus.AVAILABLE

            self._audit(
                record,
                CapabilityEventType.RECOVERED,
                "Capability recovered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            records = [
                record
                for record in self.capabilities.values()
                if (
                    tenant_id is None
                    or record.definition.tenant_id
                    == tenant_id
                )
            ]

            for record in records:
                self._refresh(record)

            return {
                "total_capabilities": len(records),
                "available": sum(
                    record.status
                    == CapabilityStatus.AVAILABLE
                    for record in records
                ),
                "degraded": sum(
                    record.status
                    == CapabilityStatus.DEGRADED
                    for record in records
                ),
                "unavailable": sum(
                    record.status
                    == CapabilityStatus.UNAVAILABLE
                    for record in records
                ),
                "faulted": sum(
                    record.status
                    == CapabilityStatus.FAULTED
                    for record in records
                ),
                "stale": sum(
                    record.status
                    == CapabilityStatus.STALE
                    for record in records
                ),
            }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
    ) -> Path:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "simulation_only": True,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "capabilities": self.discover(
                    tenant_id=tenant_id
                ),
                "audit_events": [
                    event.to_dict()
                    for event in self.audit_events
                    if event.tenant_id == tenant_id
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
# DEMONSTRATION
# ============================================================

def main() -> None:
    registry = RoboticsCapabilityRegistry()

    registry.register_capability(
        RobotCapability(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            capability_id="object_detection",
            name="Object Detection",
            category=CapabilityCategory.OBJECT_DETECTION,
            required_sensors=("camera",),
            required_models=("detector_v1",),
            risk_level=CapabilityRiskLevel.LOW,
        )
    )

    registry.register_capability(
        RobotCapability(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            capability_id="inspection",
            name="Visual Inspection",
            category=CapabilityCategory.INSPECTION,
            required_sensors=("camera",),
            required_models=("inspection_v1",),
            required_capabilities=("object_detection",),
            risk_level=CapabilityRiskLevel.MEDIUM,
        )
    )

    registry.update_health(
        CapabilityHealthObservation(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            capability_id="object_detection",
            sequence_number=1,
            operational=True,
            healthy_sensors=("camera",),
            available_models=("detector_v1",),
        )
    )

    registry.update_health(
        CapabilityHealthObservation(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            capability_id="inspection",
            sequence_number=1,
            operational=True,
            healthy_sensors=("camera",),
            available_models=("inspection_v1",),
        )
    )

    result = registry.match_task(
        CapabilityTaskRequirement(
            tenant_id="tenant_alpha",
            task_id="inspection_task_001",
            required_capabilities=("inspection",),
            maximum_risk_level=CapabilityRiskLevel.MEDIUM,
        )
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    registry.export_json(
        "data/robotics/capability_registry_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

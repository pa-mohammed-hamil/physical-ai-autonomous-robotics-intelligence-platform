"""Parameter Manager module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/robotics/control/parameter_manager.py

Python 3.11+
Standard library only.

Features:
- Robot and node-specific parameters
- Strong parameter type validation
- Numeric range and enum constraints
- Parameter registration
- Version-controlled updates
- Atomic batch updates
- Optimistic concurrency control
- Safety-critical parameter protection
- Authenticated-approval integration interface
- Immutable revision snapshots
- Parameter rollback
- Tenant isolation
- Thread-safe operations
- Audit logging
- JSON export

SAFETY:
    Simulation configuration management only.

    No physical robot parameters are modified.
    No ROS 2 parameter service is called.

    Safety-critical configuration changes require
    an external trusted authorization decision.

    This module is not a safety-rated controller.
"""

from __future__ import annotations

import copy
import json
import math
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class ParameterManagerError(Exception):
    pass


class ParameterValidationError(ParameterManagerError):
    pass


class ParameterNotFound(ParameterManagerError):
    pass


class ParameterConflict(ParameterManagerError):
    pass


class ParameterPermissionDenied(ParameterManagerError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING_LIST = "string_list"
    INTEGER_LIST = "integer_list"
    FLOAT_LIST = "float_list"


class ParameterCategory(str, Enum):
    GENERAL = "general"
    NAVIGATION = "navigation"
    PERCEPTION = "perception"
    MANIPULATION = "manipulation"
    VLA_MODEL = "vla_model"
    SIMULATION = "simulation"
    TELEMETRY = "telemetry"
    SAFETY = "safety"
    MLOPS = "mlops"


class ParameterEventType(str, Enum):
    REGISTERED = "registered"
    UPDATED = "updated"
    BATCH_UPDATED = "batch_updated"
    ROLLED_BACK = "rolled_back"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_identifier(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            char.isalnum() or char in "_.-"
            for char in value
        )
    ):
        raise ParameterValidationError(
            f"Invalid {name}"
        )

    return value


def finite_number(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ParameterValidationError(
            f"{name} must be a finite number"
        )

    return float(value)


def json_copy(value: Any) -> Any:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
        )
        return json.loads(encoded)

    except (TypeError, ValueError) as exc:
        raise ParameterValidationError(
            "Value must be JSON serializable"
        ) from exc


# ============================================================
# PARAMETER DEFINITION
# ============================================================

@dataclass(frozen=True)
class ParameterDefinition:
    name: str
    parameter_type: ParameterType
    default_value: Any

    category: ParameterCategory = (
        ParameterCategory.GENERAL
    )

    description: str = ""

    minimum: float | None = None
    maximum: float | None = None

    allowed_values: tuple[Any, ...] = ()

    read_only: bool = False
    safety_critical: bool = False

    def __post_init__(self) -> None:
        validate_identifier(
            self.name,
            "parameter name",
        )

        if not isinstance(
            self.parameter_type,
            ParameterType,
        ):
            raise ParameterValidationError(
                "Invalid parameter type"
            )

        if not isinstance(
            self.category,
            ParameterCategory,
        ):
            raise ParameterValidationError(
                "Invalid parameter category"
            )

        if self.minimum is not None:
            finite_number(
                self.minimum,
                "minimum",
            )

        if self.maximum is not None:
            finite_number(
                self.maximum,
                "maximum",
            )

        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ParameterValidationError(
                "Minimum exceeds maximum"
            )

        self.validate_value(
            self.default_value
        )

    def validate_value(
        self,
        value: Any,
    ) -> Any:
        kind = self.parameter_type

        if kind == ParameterType.STRING:
            valid = isinstance(value, str)

        elif kind == ParameterType.INTEGER:
            valid = (
                isinstance(value, int)
                and not isinstance(value, bool)
            )

        elif kind == ParameterType.FLOAT:
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )

        elif kind == ParameterType.BOOLEAN:
            valid = isinstance(value, bool)

        elif kind == ParameterType.STRING_LIST:
            valid = (
                isinstance(value, list)
                and all(
                    isinstance(item, str)
                    for item in value
                )
            )

        elif kind == ParameterType.INTEGER_LIST:
            valid = (
                isinstance(value, list)
                and all(
                    isinstance(item, int)
                    and not isinstance(item, bool)
                    for item in value
                )
            )

        elif kind == ParameterType.FLOAT_LIST:
            valid = (
                isinstance(value, list)
                and all(
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and math.isfinite(item)
                    for item in value
                )
            )

        else:
            valid = False

        if not valid:
            raise ParameterValidationError(
                f"Invalid value type for {self.name}"
            )

        numeric_values = (
            value
            if isinstance(value, list)
            else [value]
        )

        if kind in {
            ParameterType.INTEGER,
            ParameterType.FLOAT,
            ParameterType.INTEGER_LIST,
            ParameterType.FLOAT_LIST,
        }:
            for item in numeric_values:
                number = finite_number(
                    item,
                    self.name,
                )

                if (
                    self.minimum is not None
                    and number < self.minimum
                ):
                    raise ParameterValidationError(
                        f"{self.name} below minimum"
                    )

                if (
                    self.maximum is not None
                    and number > self.maximum
                ):
                    raise ParameterValidationError(
                        f"{self.name} above maximum"
                    )

        if (
            self.allowed_values
            and value not in self.allowed_values
        ):
            raise ParameterValidationError(
                f"{self.name} value not allowed"
            )

        return json_copy(value)


# ============================================================
# PARAMETER SCOPE
# ============================================================

@dataclass(frozen=True)
class ParameterScope:
    tenant_id: str
    robot_id: str
    node_id: str

    def __post_init__(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.robot_id,
            "robot_id",
        )

        validate_identifier(
            self.node_id,
            "node_id",
        )

    def key(self) -> tuple[str, str, str]:
        return (
            self.tenant_id,
            self.robot_id,
            self.node_id,
        )


# ============================================================
# CHANGE APPROVAL
# ============================================================

@dataclass(frozen=True)
class ParameterChangeContext:
    actor_id: str
    reason: str

    approval_reference: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(
            self.actor_id,
            "actor_id",
        )

        if not self.reason.strip():
            raise ParameterValidationError(
                "Change reason required"
            )

        if self.approval_reference is not None:
            validate_identifier(
                self.approval_reference,
                "approval_reference",
            )


# External trusted approval validator.
#
# This callback must independently verify:
# - actor identity
# - permission to modify this parameter
# - approval binding to the exact change
# - approval expiry and one-time consumption
#
# Merely supplying an approval reference is
# not sufficient authorization.

ApprovalValidator = Callable[
    [
        ParameterScope,
        ParameterDefinition,
        Any,
        ParameterChangeContext,
    ],
    bool,
]


# ============================================================
# PARAMETER REVISION
# ============================================================

@dataclass(frozen=True)
class ParameterRevision:
    revision_id: str
    version: int

    value: Any

    changed_by: str
    reason: str
    changed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "version": self.version,
            "value": json_copy(self.value),
            "changed_by": self.changed_by,
            "reason": self.reason,
            "changed_at": (
                self.changed_at.isoformat()
            ),
        }


# ============================================================
# PARAMETER RECORD
# ============================================================

@dataclass
class ParameterRecord:
    definition: ParameterDefinition
    current_value: Any

    version: int = 1

    history: list[
        ParameterRevision
    ] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.definition.name,
            "type": (
                self.definition.parameter_type.value
            ),
            "category": (
                self.definition.category.value
            ),
            "description": (
                self.definition.description
            ),
            "value": json_copy(
                self.current_value
            ),
            "default_value": json_copy(
                self.definition.default_value
            ),
            "version": self.version,
            "read_only": self.definition.read_only,
            "safety_critical": (
                self.definition.safety_critical
            ),
            "minimum": self.definition.minimum,
            "maximum": self.definition.maximum,
            "allowed_values": json_copy(
                self.definition.allowed_values
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class ParameterAuditEvent:
    event_id: str
    timestamp: datetime

    scope: ParameterScope
    parameter_name: str

    event_type: ParameterEventType

    actor_id: str
    version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": (
                self.timestamp.isoformat()
            ),
            "tenant_id": self.scope.tenant_id,
            "robot_id": self.scope.robot_id,
            "node_id": self.scope.node_id,
            "parameter_name": (
                self.parameter_name
            ),
            "event_type": (
                self.event_type.value
            ),
            "actor_id": self.actor_id,
            "version": self.version,
        }


# ============================================================
# PARAMETER MANAGER
# ============================================================

class RoboticsParameterManager:

    def __init__(
        self,
        *,
        approval_validator: ApprovalValidator | None = None,
        maximum_audit_events: int = 20000,
    ) -> None:
        self.approval_validator = (
            approval_validator
        )

        self.maximum_audit_events = (
            maximum_audit_events
        )

        self.parameters: dict[
            tuple[str, str, str, str],
            ParameterRecord
        ] = {}

        self.audit_events: list[
            ParameterAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def _key(
        self,
        scope: ParameterScope,
        name: str,
    ) -> tuple[str, str, str, str]:
        return (
            *scope.key(),
            validate_identifier(
                name,
                "parameter name",
            ),
        )

    def _get(
        self,
        scope: ParameterScope,
        name: str,
    ) -> ParameterRecord:
        record = self.parameters.get(
            self._key(scope, name)
        )

        if record is None:
            raise ParameterNotFound(
                f"Parameter not found: {name}"
            )

        return record

    def _audit(
        self,
        scope: ParameterScope,
        record: ParameterRecord,
        event_type: ParameterEventType,
        actor_id: str,
    ) -> None:
        self.audit_events.append(
            ParameterAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                scope=scope,
                parameter_name=(
                    record.definition.name
                ),
                event_type=event_type,
                actor_id=actor_id,
                version=record.version,
            )
        )

        if (
            len(self.audit_events)
            > self.maximum_audit_events
        ):
            self.audit_events.pop(0)

    def _authorize(
        self,
        scope: ParameterScope,
        record: ParameterRecord,
        new_value: Any,
        context: ParameterChangeContext,
    ) -> None:
        if record.definition.read_only:
            raise ParameterPermissionDenied(
                "Read-only parameter"
            )

        if record.definition.safety_critical:
            if self.approval_validator is None:
                raise ParameterPermissionDenied(
                    "Trusted approval validator required"
                )

            try:
                approved = self.approval_validator(
                    scope,
                    record.definition,
                    new_value,
                    context,
                )
            except Exception as exc:
                raise ParameterPermissionDenied(
                    "Approval validation failed"
                ) from exc

            if approved is not True:
                raise ParameterPermissionDenied(
                    "Safety-critical change denied"
                )

    def _commit(
        self,
        scope: ParameterScope,
        record: ParameterRecord,
        new_value: Any,
        context: ParameterChangeContext,
        event_type: ParameterEventType,
    ) -> None:
        record.version += 1

        record.current_value = json_copy(
            new_value
        )

        record.history.append(
            ParameterRevision(
                revision_id=str(uuid.uuid4()),
                version=record.version,
                value=json_copy(new_value),
                changed_by=context.actor_id,
                reason=context.reason,
                changed_at=utc_now(),
            )
        )

        self._audit(
            scope,
            record,
            event_type,
            context.actor_id,
        )

    # --------------------------------------------------------
    # REGISTER PARAMETER
    # --------------------------------------------------------

    def register(
        self,
        *,
        scope: ParameterScope,
        definition: ParameterDefinition,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        validate_identifier(
            actor_id,
            "actor_id",
        )

        with self._lock:
            key = self._key(
                scope,
                definition.name,
            )

            if key in self.parameters:
                raise ParameterConflict(
                    "Parameter already registered"
                )

            initial_value = (
                definition.validate_value(
                    definition.default_value
                )
            )

            record = ParameterRecord(
                definition=definition,
                current_value=initial_value,
            )

            record.history.append(
                ParameterRevision(
                    revision_id=str(uuid.uuid4()),
                    version=1,
                    value=json_copy(initial_value),
                    changed_by=actor_id,
                    reason="Initial registration",
                    changed_at=utc_now(),
                )
            )

            self.parameters[key] = record

            self._audit(
                scope,
                record,
                ParameterEventType.REGISTERED,
                actor_id,
            )

            return record.to_dict()

    # --------------------------------------------------------
    # GET PARAMETER
    # --------------------------------------------------------

    def get(
        self,
        *,
        scope: ParameterScope,
        name: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get(
                scope,
                name,
            ).to_dict()

    # --------------------------------------------------------
    # UPDATE PARAMETER
    # --------------------------------------------------------

    def update(
        self,
        *,
        scope: ParameterScope,
        name: str,
        value: Any,
        expected_version: int,
        context: ParameterChangeContext,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                scope,
                name,
            )

            if record.version != expected_version:
                raise ParameterConflict(
                    "Parameter version conflict"
                )

            validated = (
                record.definition.validate_value(
                    value
                )
            )

            self._authorize(
                scope,
                record,
                validated,
                context,
            )

            self._commit(
                scope,
                record,
                validated,
                context,
                ParameterEventType.UPDATED,
            )

            return record.to_dict()

    # --------------------------------------------------------
    # ATOMIC BATCH UPDATE
    # --------------------------------------------------------

    def update_batch(
        self,
        *,
        scope: ParameterScope,
        changes: Mapping[str, Any],
        expected_versions: Mapping[str, int],
        context: ParameterChangeContext,
    ) -> list[dict[str, Any]]:
        if not changes:
            raise ParameterValidationError(
                "Batch changes cannot be empty"
            )

        with self._lock:
            prepared = []

            # Validate all changes before mutation.
            for name, value in changes.items():
                record = self._get(
                    scope,
                    name,
                )

                expected = expected_versions.get(
                    name
                )

                if record.version != expected:
                    raise ParameterConflict(
                        f"Version conflict: {name}"
                    )

                validated = (
                    record.definition.validate_value(
                        value
                    )
                )

                self._authorize(
                    scope,
                    record,
                    validated,
                    context,
                )

                prepared.append(
                    (
                        record,
                        validated,
                    )
                )

            # All validations passed.
            for record, validated in prepared:
                self._commit(
                    scope,
                    record,
                    validated,
                    context,
                    ParameterEventType.BATCH_UPDATED,
                )

            return [
                record.to_dict()
                for record, _ in prepared
            ]

    # --------------------------------------------------------
    # ROLLBACK
    # --------------------------------------------------------

    def rollback(
        self,
        *,
        scope: ParameterScope,
        name: str,
        target_version: int,
        expected_version: int,
        context: ParameterChangeContext,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                scope,
                name,
            )

            if record.version != expected_version:
                raise ParameterConflict(
                    "Parameter version conflict"
                )

            revision = next(
                (
                    item
                    for item in record.history
                    if item.version == target_version
                ),
                None,
            )

            if revision is None:
                raise ParameterNotFound(
                    "Target revision not found"
                )

            restored_value = (
                record.definition.validate_value(
                    revision.value
                )
            )

            self._authorize(
                scope,
                record,
                restored_value,
                context,
            )

            # Rollback creates a new revision.
            # It never rewrites history.
            self._commit(
                scope,
                record,
                restored_value,
                context,
                ParameterEventType.ROLLED_BACK,
            )

            return record.to_dict()

    # --------------------------------------------------------
    # LIST PARAMETERS
    # --------------------------------------------------------

    def list_parameters(
        self,
        *,
        scope: ParameterScope,
        category: ParameterCategory | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            results = []

            for key, record in self.parameters.items():
                if key[:3] != scope.key():
                    continue

                if (
                    category is not None
                    and record.definition.category
                    != category
                ):
                    continue

                results.append(
                    record.to_dict()
                )

            return sorted(
                results,
                key=lambda item: item["name"],
            )

    # --------------------------------------------------------
    # PARAMETER HISTORY
    # --------------------------------------------------------

    def history(
        self,
        *,
        scope: ParameterScope,
        name: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            record = self._get(
                scope,
                name,
            )

            return [
                revision.to_dict()
                for revision in record.history
            ]

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
    ) -> Path:
        validate_identifier(
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
                "generated_at": (
                    utc_now().isoformat()
                ),
                "tenant_id": tenant_id,
                "parameters": [
                    {
                        "robot_id": key[1],
                        "node_id": key[2],
                        **record.to_dict(),
                    }
                    for key, record
                    in self.parameters.items()
                    if key[0] == tenant_id
                ],
                "audit_events": [
                    event.to_dict()
                    for event in self.audit_events
                    if event.scope.tenant_id
                    == tenant_id
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
    manager = RoboticsParameterManager()

    scope = ParameterScope(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        node_id="navigation",
    )

    manager.register(
        scope=scope,
        definition=ParameterDefinition(
            name="maximum_simulation_speed_mps",
            parameter_type=ParameterType.FLOAT,
            default_value=1.0,
            category=ParameterCategory.NAVIGATION,
            minimum=0.0,
            maximum=2.0,
            description=(
                "Maximum navigation speed "
                "for the robot simulator"
            ),
        ),
    )

    manager.register(
        scope=scope,
        definition=ParameterDefinition(
            name="planner_mode",
            parameter_type=ParameterType.STRING,
            default_value="astar",
            category=ParameterCategory.NAVIGATION,
            allowed_values=(
                "astar",
                "dijkstra",
                "rrt",
            ),
        ),
    )

    updated = manager.update(
        scope=scope,
        name="maximum_simulation_speed_mps",
        value=0.8,
        expected_version=1,
        context=ParameterChangeContext(
            actor_id="simulation_operator",
            reason="Reduce navigation test speed",
        ),
    )

    print(
        "Updated parameter:"
    )

    print(
        json.dumps(
            updated,
            indent=2,
        )
    )

    print(
        "\nParameter history:"
    )

    print(
        json.dumps(
            manager.history(
                scope=scope,
                name="maximum_simulation_speed_mps",
            ),
            indent=2,
        )
    )

    manager.export_json(
        "data/robotics/parameter_manager_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

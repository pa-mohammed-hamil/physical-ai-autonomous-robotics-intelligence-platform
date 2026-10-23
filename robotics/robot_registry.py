"""Robot Registry module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/robot_registry.py

Python 3.11+
Standard library only.

CENTRAL ROBOT REGISTRY — SIMULATION ONLY

Features:
    - Multi-tenant robot registry
    - Fleet and robot identity management
    - Robot manufacturer/model metadata
    - Capability inventory
    - Software and firmware version tracking
    - Robot lifecycle status references
    - Registration and metadata updates
    - Robot activation and retirement
    - Registry search and filtering
    - Optimistic revision control
    - Duplicate identity prevention
    - Robot inventory statistics
    - Audit logging
    - JSON export/import
    - Simulation-only operation

IMPORTANT:
    Registry records do not establish robot identity,
    hardware authenticity, operational readiness,
    physical safety or network availability.

    Production integrations must use authenticated
    device identities, trusted authorization,
    persistent storage and independent safety checks.
"""

from __future__ import annotations

import json
import re
import threading
import uuid

from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class RobotRegistryError(Exception):
    pass


class RegistryValidationError(RobotRegistryError):
    pass


class RegistryRobotNotFound(RobotRegistryError):
    pass


class RegistryConflict(RobotRegistryError):
    pass


class RegistryPermissionDenied(RobotRegistryError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RegisteredRobotType(str, Enum):
    AMR = "amr"
    AGV = "agv"
    ROBOT_ARM = "robot_arm"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    HUMANOID = "humanoid"
    INSPECTION_ROBOT = "inspection_robot"
    SERVICE_ROBOT = "service_robot"
    RESEARCH_ROBOT = "research_robot"
    DRONE_SIMULATION = "drone_simulation"
    OTHER = "other"


class RegistryRobotStatus(str, Enum):
    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    SUSPENDED = "suspended"
    RETIRED = "retired"


class RegistryEventType(str, Enum):
    ROBOT_REGISTERED = "robot_registered"
    METADATA_UPDATED = "metadata_updated"
    CAPABILITIES_UPDATED = "capabilities_updated"
    SOFTWARE_UPDATED = "software_updated"
    STATUS_CHANGED = "status_changed"
    ROBOT_RETIRED = "robot_retired"
    REGISTRY_IMPORTED = "registry_imported"


# ============================================================
# HELPERS
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
        raise RegistryValidationError(
            f"Invalid {label}"
        )

    return value


def validate_text(
    value: str,
    label: str,
    *,
    maximum_length: int = 256,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum_length
    ):
        raise RegistryValidationError(
            f"Invalid {label}"
        )

    return value.strip()


def validate_version(value: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"\d+\.\d+\.\d+",
            value,
        )
    ):
        raise RegistryValidationError(
            "Version must use MAJOR.MINOR.PATCH"
        )

    return value


def validate_capabilities(
    capabilities: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(capabilities, tuple):
        raise RegistryValidationError(
            "Capabilities must be a tuple"
        )

    for capability in capabilities:
        validate_id(
            capability,
            "capability",
        )

    if (
        len(capabilities)
        != len(set(capabilities))
    ):
        raise RegistryValidationError(
            "Duplicate capabilities"
        )

    return capabilities


def validate_revision(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise RegistryValidationError(
            "Revision must be a positive integer"
        )

    return value


def require_authorized(
    authorized: bool,
) -> None:
    if authorized is not True:
        raise RegistryPermissionDenied(
            "Registry write authorization required"
        )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RobotRegistryConfig:
    maximum_robots: int = 100000
    maximum_audit_events: int = 50000

    maximum_capabilities_per_robot: int = 256

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_robots",
            "maximum_audit_events",
            "maximum_capabilities_per_robot",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise RegistryValidationError(
                    f"{name} must be a positive integer"
                )

        if not self.simulation_only:
            raise RegistryValidationError(
                "Physical robot registry operation unsupported"
            )


# ============================================================
# ROBOT IDENTITY
# ============================================================

@dataclass(frozen=True)
class RobotRegistryDefinition:
    tenant_id: str
    fleet_id: str
    robot_id: str

    name: str
    robot_type: RegisteredRobotType

    manufacturer: str
    model: str

    serial_number: str

    software_version: str = "1.0.0"
    firmware_version: str = "1.0.0"

    capabilities: tuple[str, ...] = ()

    location_zone: str = "default"

    description: str = ""

    def __post_init__(self) -> None:
        for label in (
            "tenant_id",
            "fleet_id",
            "robot_id",
            "serial_number",
            "location_zone",
        ):
            validate_id(
                getattr(self, label),
                label,
            )

        for label in (
            "name",
            "manufacturer",
            "model",
        ):
            validate_text(
                getattr(self, label),
                label,
            )

        if not isinstance(
            self.robot_type,
            RegisteredRobotType,
        ):
            raise RegistryValidationError(
                "Invalid robot type"
            )

        validate_version(
            self.software_version
        )

        validate_version(
            self.firmware_version
        )

        validate_capabilities(
            self.capabilities
        )

        if (
            not isinstance(self.description, str)
            or len(self.description) > 2048
        ):
            raise RegistryValidationError(
                "Invalid description"
            )


# ============================================================
# ROBOT RECORD
# ============================================================

@dataclass
class RobotRegistryRecord:
    definition: RobotRegistryDefinition

    status: RegistryRobotStatus = (
        RegistryRobotStatus.REGISTERED
    )

    revision: int = 1

    registered_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    retired_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        definition = self.definition

        return {
            "tenant_id": definition.tenant_id,
            "fleet_id": definition.fleet_id,
            "robot_id": definition.robot_id,
            "name": definition.name,
            "robot_type": definition.robot_type.value,
            "manufacturer": definition.manufacturer,
            "model": definition.model,
            "serial_number": definition.serial_number,
            "software_version": (
                definition.software_version
            ),
            "firmware_version": (
                definition.firmware_version
            ),
            "capabilities": list(
                definition.capabilities
            ),
            "location_zone": (
                definition.location_zone
            ),
            "description": definition.description,
            "status": self.status.value,
            "revision": self.revision,
            "registered_at": (
                self.registered_at.isoformat()
            ),
            "updated_at": (
                self.updated_at.isoformat()
            ),
            "retired_at": (
                self.retired_at.isoformat()
                if self.retired_at
                else None
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class RobotRegistryAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    fleet_id: str
    robot_id: str

    event_type: RegistryEventType

    revision: int
    actor_id: str
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "fleet_id": self.fleet_id,
            "robot_id": self.robot_id,
            "event_type": self.event_type.value,
            "revision": self.revision,
            "actor_id": self.actor_id,
            "details": self.details,
        }


# ============================================================
# ROBOT REGISTRY
# ============================================================

class RoboticsRobotRegistry:

    """
    Thread-safe in-memory robot inventory registry.

    Robot identity:
        tenant_id + fleet_id + robot_id

    Serial numbers must be unique within each tenant.

    Registry status is administrative metadata.
    It is not equivalent to live lifecycle,
    discovery or safety status.
    """

    def __init__(
        self,
        config: RobotRegistryConfig | None = None,
    ) -> None:
        self.config = (
            config or RobotRegistryConfig()
        )

        self._robots: dict[
            tuple[str, str, str],
            RobotRegistryRecord
        ] = {}

        self._serial_index: dict[
            tuple[str, str],
            tuple[str, str, str]
        ] = {}

        self._audit_events: deque[
            RobotRegistryAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # INTERNAL KEY
    # --------------------------------------------------------

    def _key(
        self,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
    ) -> tuple[str, str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(fleet_id, "fleet_id"),
            validate_id(robot_id, "robot_id"),
        )

    def _get(
        self,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
    ) -> RobotRegistryRecord:
        record = self._robots.get(
            self._key(
                tenant_id,
                fleet_id,
                robot_id,
            )
        )

        if record is None:
            raise RegistryRobotNotFound(
                "Robot not registered"
            )

        return record

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        record: RobotRegistryRecord,
        event_type: RegistryEventType,
        actor_id: str,
        details: str,
    ) -> None:
        definition = record.definition

        self._audit_events.append(
            RobotRegistryAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=definition.tenant_id,
                fleet_id=definition.fleet_id,
                robot_id=definition.robot_id,
                event_type=event_type,
                revision=record.revision,
                actor_id=actor_id,
                details=details,
            )
        )

    # --------------------------------------------------------
    # REVISION CONTROL
    # --------------------------------------------------------

    @staticmethod
    def _check_revision(
        record: RobotRegistryRecord,
        expected_revision: int,
    ) -> None:
        validate_revision(
            expected_revision
        )

        if record.revision != expected_revision:
            raise RegistryConflict(
                "Robot revision conflict"
            )

    @staticmethod
    def _touch(
        record: RobotRegistryRecord,
    ) -> None:
        record.revision += 1
        record.updated_at = utc_now()

    # --------------------------------------------------------
    # REGISTER ROBOT
    # --------------------------------------------------------

    def register_robot(
        self,
        definition: RobotRegistryDefinition,
        *,
        actor_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        require_authorized(authorized)

        validate_id(actor_id, "actor_id")

        if not isinstance(
            definition,
            RobotRegistryDefinition,
        ):
            raise RegistryValidationError(
                "Invalid robot definition"
            )

        if (
            len(definition.capabilities)
            > self.config.maximum_capabilities_per_robot
        ):
            raise RegistryValidationError(
                "Too many robot capabilities"
            )

        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.fleet_id,
                definition.robot_id,
            )

            if key in self._robots:
                raise RegistryConflict(
                    "Robot identity already registered"
                )

            serial_key = (
                definition.tenant_id,
                definition.serial_number,
            )

            if serial_key in self._serial_index:
                raise RegistryConflict(
                    "Serial number already registered"
                )

            if (
                len(self._robots)
                >= self.config.maximum_robots
            ):
                raise RobotRegistryError(
                    "Registry capacity reached"
                )

            record = RobotRegistryRecord(
                definition=definition
            )

            self._robots[key] = record

            self._serial_index[
                serial_key
            ] = key

            self._audit(
                record,
                RegistryEventType.ROBOT_REGISTERED,
                actor_id,
                "Robot registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # UPDATE METADATA
    # --------------------------------------------------------

    def update_metadata(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
        expected_revision: int,
        actor_id: str,
        authorized: bool,
        name: str | None = None,
        location_zone: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        require_authorized(authorized)
        validate_id(actor_id, "actor_id")

        if (
            name is None
            and location_zone is None
            and description is None
        ):
            raise RegistryValidationError(
                "No metadata updates supplied"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                fleet_id,
                robot_id,
            )

            self._check_revision(
                record,
                expected_revision,
            )

            if (
                record.status
                == RegistryRobotStatus.RETIRED
            ):
                raise RegistryConflict(
                    "Retired robot cannot be modified"
                )

            updates: dict[str, Any] = {}

            if name is not None:
                updates["name"] = validate_text(
                    name,
                    "name",
                )

            if location_zone is not None:
                updates["location_zone"] = (
                    validate_id(
                        location_zone,
                        "location_zone",
                    )
                )

            if description is not None:
                if (
                    not isinstance(description, str)
                    or len(description) > 2048
                ):
                    raise RegistryValidationError(
                        "Invalid description"
                    )

                updates["description"] = description

            record.definition = replace(
                record.definition,
                **updates,
            )

            self._touch(record)

            self._audit(
                record,
                RegistryEventType.METADATA_UPDATED,
                actor_id,
                "Robot metadata updated",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # UPDATE CAPABILITIES
    # --------------------------------------------------------

    def update_capabilities(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
        capabilities: tuple[str, ...],
        expected_revision: int,
        actor_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        require_authorized(authorized)
        validate_id(actor_id, "actor_id")

        validate_capabilities(
            capabilities
        )

        if (
            len(capabilities)
            > self.config.maximum_capabilities_per_robot
        ):
            raise RegistryValidationError(
                "Too many capabilities"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                fleet_id,
                robot_id,
            )

            self._check_revision(
                record,
                expected_revision,
            )

            if (
                record.status
                == RegistryRobotStatus.RETIRED
            ):
                raise RegistryConflict(
                    "Retired robot cannot be modified"
                )

            record.definition = replace(
                record.definition,
                capabilities=capabilities,
            )

            self._touch(record)

            self._audit(
                record,
                RegistryEventType.CAPABILITIES_UPDATED,
                actor_id,
                "Robot capability inventory updated",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # UPDATE SOFTWARE
    # --------------------------------------------------------

    def update_software(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
        software_version: str,
        firmware_version: str,
        expected_revision: int,
        actor_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        require_authorized(authorized)
        validate_id(actor_id, "actor_id")

        validate_version(
            software_version
        )

        validate_version(
            firmware_version
        )

        with self._lock:
            record = self._get(
                tenant_id,
                fleet_id,
                robot_id,
            )

            self._check_revision(
                record,
                expected_revision,
            )

            if (
                record.status
                == RegistryRobotStatus.RETIRED
            ):
                raise RegistryConflict(
                    "Retired robot cannot be modified"
                )

            record.definition = replace(
                record.definition,
                software_version=software_version,
                firmware_version=firmware_version,
            )

            self._touch(record)

            self._audit(
                record,
                RegistryEventType.SOFTWARE_UPDATED,
                actor_id,
                "Software and firmware metadata updated",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # CHANGE ADMINISTRATIVE STATUS
    # --------------------------------------------------------

    def change_status(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
        new_status: RegistryRobotStatus,
        expected_revision: int,
        actor_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        require_authorized(authorized)
        validate_id(actor_id, "actor_id")

        if not isinstance(
            new_status,
            RegistryRobotStatus,
        ):
            raise RegistryValidationError(
                "Invalid registry status"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                fleet_id,
                robot_id,
            )

            self._check_revision(
                record,
                expected_revision,
            )

            if (
                record.status
                == RegistryRobotStatus.RETIRED
            ):
                raise RegistryConflict(
                    "Retired robot status is immutable"
                )

            if new_status == RegistryRobotStatus.RETIRED:
                raise RegistryValidationError(
                    "Use retire_robot()"
                )

            if new_status == record.status:
                return record.to_dict()

            record.status = new_status

            self._touch(record)

            self._audit(
                record,
                RegistryEventType.STATUS_CHANGED,
                actor_id,
                f"Administrative status: {new_status.value}",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # RETIRE ROBOT
    # --------------------------------------------------------

    def retire_robot(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
        expected_revision: int,
        actor_id: str,
        authorized: bool,
        independently_verified: bool,
    ) -> dict[str, Any]:
        require_authorized(authorized)
        validate_id(actor_id, "actor_id")

        if independently_verified is not True:
            raise RegistryPermissionDenied(
                "Independent retirement verification required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                fleet_id,
                robot_id,
            )

            self._check_revision(
                record,
                expected_revision,
            )

            if (
                record.status
                == RegistryRobotStatus.RETIRED
            ):
                return record.to_dict()

            record.status = (
                RegistryRobotStatus.RETIRED
            )

            record.retired_at = utc_now()

            self._touch(record)

            self._audit(
                record,
                RegistryEventType.ROBOT_RETIRED,
                actor_id,
                "Robot administratively retired",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # GET ROBOT
    # --------------------------------------------------------

    def get_robot(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get(
                tenant_id,
                fleet_id,
                robot_id,
            ).to_dict()

    # --------------------------------------------------------
    # LOOKUP BY SERIAL NUMBER
    # --------------------------------------------------------

    def get_by_serial_number(
        self,
        *,
        tenant_id: str,
        serial_number: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(serial_number, "serial_number")

        with self._lock:
            key = self._serial_index.get(
                (tenant_id, serial_number)
            )

            if key is None:
                raise RegistryRobotNotFound(
                    "Serial number not found"
                )

            return self._robots[key].to_dict()

    # --------------------------------------------------------
    # LIST / SEARCH
    # --------------------------------------------------------

    def list_robots(
        self,
        *,
        tenant_id: str,
        fleet_id: str | None = None,
        status: RegistryRobotStatus | None = None,
        robot_type: RegisteredRobotType | None = None,
        required_capabilities: tuple[str, ...] = (),
        location_zone: str | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if fleet_id is not None:
            validate_id(fleet_id, "fleet_id")

        if location_zone is not None:
            validate_id(location_zone, "location_zone")

        validate_capabilities(
            required_capabilities
        )

        if (
            status is not None
            and not isinstance(status, RegistryRobotStatus)
        ):
            raise RegistryValidationError(
                "Invalid status filter"
            )

        if (
            robot_type is not None
            and not isinstance(robot_type, RegisteredRobotType)
        ):
            raise RegistryValidationError(
                "Invalid robot type filter"
            )

        required = set(required_capabilities)

        with self._lock:
            result = []

            for record in self._robots.values():
                definition = record.definition

                if definition.tenant_id != tenant_id:
                    continue

                if (
                    fleet_id is not None
                    and definition.fleet_id != fleet_id
                ):
                    continue

                if (
                    status is not None
                    and record.status != status
                ):
                    continue

                if (
                    robot_type is not None
                    and definition.robot_type != robot_type
                ):
                    continue

                if (
                    location_zone is not None
                    and definition.location_zone
                    != location_zone
                ):
                    continue

                if not required.issubset(
                    definition.capabilities
                ):
                    continue

                result.append(
                    record.to_dict()
                )

            return sorted(
                result,
                key=lambda item: (
                    item["fleet_id"],
                    item["robot_id"],
                ),
            )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str,
        fleet_id: str | None = None,
    ) -> dict[str, Any]:
        robots = self.list_robots(
            tenant_id=tenant_id,
            fleet_id=fleet_id,
        )

        status_counts = {
            status.value: 0
            for status in RegistryRobotStatus
        }

        type_counts = {
            robot_type.value: 0
            for robot_type in RegisteredRobotType
        }

        for robot in robots:
            status_counts[
                robot["status"]
            ] += 1

            type_counts[
                robot["robot_type"]
            ] += 1

        return {
            "tenant_id": tenant_id,
            "fleet_id": fleet_id,
            "total_robots": len(robots),
            "status_counts": status_counts,
            "type_counts": type_counts,
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
        fleet_id: str | None = None,
    ) -> Path:
        validate_id(tenant_id, "tenant_id")

        if fleet_id is not None:
            validate_id(fleet_id, "fleet_id")

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "schema_version": "1.0.0",
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "fleet_id": fleet_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id,
                    fleet_id=fleet_id,
                ),
                "robots": self.list_robots(
                    tenant_id=tenant_id,
                    fleet_id=fleet_id,
                ),
                "audit_events": [
                    event.to_dict()
                    for event in self._audit_events
                    if (
                        event.tenant_id == tenant_id
                        and (
                            fleet_id is None
                            or event.fleet_id == fleet_id
                        )
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

    # --------------------------------------------------------
    # IMPORT JSON
    # --------------------------------------------------------

    def import_json(
        self,
        path: str | Path,
        *,
        actor_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        """
        Imports new robot inventory records.

        Existing robots are never overwritten.
        Import is validated before mutation.
        """

        require_authorized(authorized)
        validate_id(actor_id, "actor_id")

        source = Path(path)

        try:
            payload = json.loads(
                source.read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError) as exc:
            raise RegistryValidationError(
                "Cannot read registry import"
            ) from exc

        if not isinstance(payload, dict):
            raise RegistryValidationError(
                "Invalid import format"
            )

        if payload.get("schema_version") != "1.0.0":
            raise RegistryValidationError(
                "Unsupported registry schema"
            )

        rows = payload.get("robots")

        if not isinstance(rows, list):
            raise RegistryValidationError(
                "Import must contain robots list"
            )

        definitions = []

        for row in rows:
            if not isinstance(row, dict):
                raise RegistryValidationError(
                    "Invalid robot import row"
                )

            try:
                definition = RobotRegistryDefinition(
                    tenant_id=row["tenant_id"],
                    fleet_id=row["fleet_id"],
                    robot_id=row["robot_id"],
                    name=row["name"],
                    robot_type=RegisteredRobotType(
                        row["robot_type"]
                    ),
                    manufacturer=row["manufacturer"],
                    model=row["model"],
                    serial_number=row["serial_number"],
                    software_version=(
                        row["software_version"]
                    ),
                    firmware_version=(
                        row["firmware_version"]
                    ),
                    capabilities=tuple(
                        row["capabilities"]
                    ),
                    location_zone=(
                        row["location_zone"]
                    ),
                    description=(
                        row.get("description", "")
                    ),
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise RegistryValidationError(
                    "Malformed robot import record"
                ) from exc

            definitions.append(definition)

        with self._lock:
            incoming_keys: set[
                tuple[str, str, str]
            ] = set()

            incoming_serials: set[
                tuple[str, str]
            ] = set()

            if (
                len(self._robots) + len(definitions)
                > self.config.maximum_robots
            ):
                raise RegistryConflict(
                    "Import exceeds registry capacity"
                )

            for definition in definitions:
                key = self._key(
                    definition.tenant_id,
                    definition.fleet_id,
                    definition.robot_id,
                )

                serial_key = (
                    definition.tenant_id,
                    definition.serial_number,
                )

                if (
                    key in self._robots
                    or key in incoming_keys
                ):
                    raise RegistryConflict(
                        "Duplicate robot identity in import"
                    )

                if (
                    serial_key in self._serial_index
                    or serial_key in incoming_serials
                ):
                    raise RegistryConflict(
                        "Duplicate serial number in import"
                    )

                if (
                    len(definition.capabilities)
                    > self.config.maximum_capabilities_per_robot
                ):
                    raise RegistryValidationError(
                        "Imported capability limit exceeded"
                    )

                incoming_keys.add(key)
                incoming_serials.add(serial_key)

            for definition in definitions:
                key = self._key(
                    definition.tenant_id,
                    definition.fleet_id,
                    definition.robot_id,
                )

                record = RobotRegistryRecord(
                    definition=definition
                )

                self._robots[key] = record

                self._serial_index[
                    (
                        definition.tenant_id,
                        definition.serial_number,
                    )
                ] = key

                self._audit(
                    record,
                    RegistryEventType.REGISTRY_IMPORTED,
                    actor_id,
                    "Robot imported into registry",
                )

            return {
                "imported_robots": len(definitions),
                "simulation_only": True,
            }


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    registry = RoboticsRobotRegistry()

    tenant_id = "tenant_alpha"
    fleet_id = "factory_fleet"

    robots = (
        RobotRegistryDefinition(
            tenant_id=tenant_id,
            fleet_id=fleet_id,
            robot_id="robot_001",
            name="Inspection AMR",
            robot_type=RegisteredRobotType.AMR,
            manufacturer="Simulation Robotics",
            model="AMR-SIM-100",
            serial_number="SIM-AMR-001",
            capabilities=(
                "navigation",
                "inspection",
                "object_detection",
            ),
            location_zone="production_a",
        ),
        RobotRegistryDefinition(
            tenant_id=tenant_id,
            fleet_id=fleet_id,
            robot_id="robot_002",
            name="Mobile Manipulator",
            robot_type=(
                RegisteredRobotType.MOBILE_MANIPULATOR
            ),
            manufacturer="Simulation Robotics",
            model="MM-SIM-200",
            serial_number="SIM-MM-002",
            capabilities=(
                "navigation",
                "manipulation",
                "grasping",
            ),
            location_zone="warehouse_b",
        ),
    )

    for definition in robots:
        registry.register_robot(
            definition,
            actor_id="simulation_admin",
            authorized=True,
        )

    registry.change_status(
        tenant_id=tenant_id,
        fleet_id=fleet_id,
        robot_id="robot_001",
        new_status=RegistryRobotStatus.ACTIVE,
        expected_revision=1,
        actor_id="simulation_admin",
        authorized=True,
    )

    print(
        "Navigation robots:",
        json.dumps(
            registry.list_robots(
                tenant_id=tenant_id,
                fleet_id=fleet_id,
                required_capabilities=(
                    "navigation",
                ),
            ),
            indent=2,
        ),
    )

    print(
        "\nRegistry statistics:",
        json.dumps(
            registry.statistics(
                tenant_id=tenant_id,
                fleet_id=fleet_id,
            ),
            indent=2,
        ),
    )

    report = registry.export_json(
        "data/robotics/robot_registry_report.json",
        tenant_id=tenant_id,
        fleet_id=fleet_id,
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()

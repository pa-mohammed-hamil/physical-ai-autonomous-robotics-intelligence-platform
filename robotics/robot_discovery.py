"""Robot Discovery module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/robot_discovery.py

Python 3.11+
Standard library only.

ROBOT DISCOVERY SERVICE — SIMULATION ONLY

Features:
    - Multi-robot discovery and registration
    - Robot identity and endpoint metadata
    - Simulated discovery announcements
    - Capability-based discovery
    - Robot heartbeat monitoring
    - Robot online/offline detection
    - Robot availability checks
    - Discovery TTL and expiration
    - Duplicate robot and endpoint detection
    - Tenant and fleet isolation
    - Sequence number validation
    - Robot filtering and searching
    - Discovery statistics
    - Audit logging
    - JSON report export

Security:
    Discovery announcements are not proof of
    robot identity or endpoint ownership.

    The service does not scan networks,
    broadcast packets, connect to ROS 2,
    authenticate devices or control hardware.

    Production integrations require authenticated
    device identities, trusted discovery sources,
    TLS/mTLS, replay protection, access control
    and network segmentation.
"""

from __future__ import annotations

import ipaddress
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
from typing import Any, Callable


# ============================================================
# EXCEPTIONS
# ============================================================

class RobotDiscoveryError(Exception):
    pass


class DiscoveryValidationError(RobotDiscoveryError):
    pass


class DiscoveredRobotNotFound(RobotDiscoveryError):
    pass


class DiscoveryConflict(RobotDiscoveryError):
    pass


class DiscoveryPermissionDenied(RobotDiscoveryError):
    pass


# ============================================================
# ENUMS
# ============================================================

class DiscoveryProtocol(str, Enum):
    STATIC = "static"
    SIMULATION = "simulation"
    ROS2_GRAPH = "ros2_graph"
    DDS = "dds"
    MDNS = "mdns"
    MQTT = "mqtt"
    CUSTOM = "custom"


class RobotDiscoveryStatus(str, Enum):
    DISCOVERED = "discovered"
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    EXPIRED = "expired"
    BLOCKED = "blocked"


class RobotDiscoveryEventType(str, Enum):
    ROBOT_DISCOVERED = "robot_discovered"
    ROBOT_UPDATED = "robot_updated"
    HEARTBEAT_RECEIVED = "heartbeat_received"
    ROBOT_ONLINE = "robot_online"
    ROBOT_OFFLINE = "robot_offline"
    ROBOT_EXPIRED = "robot_expired"
    ROBOT_BLOCKED = "robot_blocked"
    ROBOT_UNBLOCKED = "robot_unblocked"


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
        raise DiscoveryValidationError(
            f"Invalid {label}"
        )

    return value


def validate_timestamp(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise DiscoveryValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


def positive(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise DiscoveryValidationError(
            f"{label} must be finite and positive"
        )

    return float(value)


def validate_port(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= 65535
    ):
        raise DiscoveryValidationError(
            "Port must be between 1 and 65535"
        )

    return value


def validate_host(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 253
    ):
        raise DiscoveryValidationError(
            "Invalid endpoint host"
        )

    try:
        ipaddress.ip_address(value)
        return value
    except ValueError:
        pass

    if not re.fullmatch(
        r"[A-Za-z0-9.-]+",
        value,
    ):
        raise DiscoveryValidationError(
            "Invalid hostname"
        )

    labels = value.split(".")

    if any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        for label in labels
    ):
        raise DiscoveryValidationError(
            "Invalid hostname labels"
        )

    return value.lower()


def validate_capabilities(
    values: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise DiscoveryValidationError(
            "Capabilities must be a tuple"
        )

    for value in values:
        validate_id(value, "capability")

    if len(values) != len(set(values)):
        raise DiscoveryValidationError(
            "Duplicate capabilities"
        )

    return values


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RobotDiscoveryConfig:
    maximum_robots: int = 10000
    maximum_audit_events: int = 20000

    heartbeat_timeout_seconds: float = 5.0
    discovery_ttl_seconds: float = 60.0

    maximum_clock_skew_seconds: float = 2.0

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_robots",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise DiscoveryValidationError(
                    f"{name} must be positive"
                )

        for name in (
            "heartbeat_timeout_seconds",
            "discovery_ttl_seconds",
            "maximum_clock_skew_seconds",
        ):
            positive(
                getattr(self, name),
                name,
            )

        if (
            self.discovery_ttl_seconds
            <= self.heartbeat_timeout_seconds
        ):
            raise DiscoveryValidationError(
                "Discovery TTL must exceed heartbeat timeout"
            )

        if not self.simulation_only:
            raise DiscoveryValidationError(
                "Physical network discovery unsupported"
            )


# ============================================================
# ROBOT ENDPOINT
# ============================================================

@dataclass(frozen=True)
class RobotEndpoint:
    host: str
    port: int

    protocol: str = "simulation"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "host",
            validate_host(self.host),
        )

        validate_port(self.port)

        validate_id(
            self.protocol,
            "endpoint_protocol",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
        }


# ============================================================
# ROBOT ANNOUNCEMENT
# ============================================================

@dataclass(frozen=True)
class RobotDiscoveryAnnouncement:
    tenant_id: str
    fleet_id: str
    robot_id: str

    robot_name: str
    robot_type: str

    endpoint: RobotEndpoint

    discovery_protocol: DiscoveryProtocol

    capabilities: tuple[str, ...] = ()

    software_version: str = "1.0.0"

    sequence_number: int = 0

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "fleet_id",
            "robot_id",
            "robot_type",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if (
            not isinstance(self.robot_name, str)
            or not self.robot_name.strip()
            or len(self.robot_name) > 256
        ):
            raise DiscoveryValidationError(
                "Invalid robot name"
            )

        if not isinstance(
            self.endpoint,
            RobotEndpoint,
        ):
            raise DiscoveryValidationError(
                "Invalid robot endpoint"
            )

        if not isinstance(
            self.discovery_protocol,
            DiscoveryProtocol,
        ):
            raise DiscoveryValidationError(
                "Invalid discovery protocol"
            )

        validate_capabilities(
            self.capabilities
        )

        if (
            not isinstance(self.software_version, str)
            or not re.fullmatch(
                r"\d+\.\d+\.\d+",
                self.software_version,
            )
        ):
            raise DiscoveryValidationError(
                "Invalid software version"
            )

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise DiscoveryValidationError(
                "Invalid sequence number"
            )

        validate_timestamp(
            self.timestamp
        )


# ============================================================
# ROBOT HEARTBEAT
# ============================================================

@dataclass(frozen=True)
class RobotDiscoveryHeartbeat:
    tenant_id: str
    fleet_id: str
    robot_id: str

    sequence_number: int

    operational: bool = True
    degraded: bool = False

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "fleet_id",
            "robot_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise DiscoveryValidationError(
                "Invalid heartbeat sequence"
            )

        if (
            not isinstance(self.operational, bool)
            or not isinstance(self.degraded, bool)
        ):
            raise DiscoveryValidationError(
                "Heartbeat flags must be boolean"
            )

        validate_timestamp(
            self.timestamp
        )


# ============================================================
# ROBOT RECORD
# ============================================================

@dataclass
class DiscoveredRobotRecord:
    announcement: RobotDiscoveryAnnouncement

    status: RobotDiscoveryStatus = (
        RobotDiscoveryStatus.DISCOVERED
    )

    discovered_at: datetime = field(
        default_factory=utc_now
    )

    last_announcement_monotonic: float = field(
        default_factory=time.monotonic
    )

    last_heartbeat_monotonic: float | None = None

    last_heartbeat_sequence: int = -1

    blocked: bool = False

    heartbeat_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        announcement = self.announcement

        return {
            "tenant_id": announcement.tenant_id,
            "fleet_id": announcement.fleet_id,
            "robot_id": announcement.robot_id,
            "robot_name": announcement.robot_name,
            "robot_type": announcement.robot_type,
            "endpoint": announcement.endpoint.to_dict(),
            "discovery_protocol": (
                announcement.discovery_protocol.value
            ),
            "capabilities": list(
                announcement.capabilities
            ),
            "software_version": (
                announcement.software_version
            ),
            "status": self.status.value,
            "blocked": self.blocked,
            "heartbeat_count": self.heartbeat_count,
            "discovered_at": (
                self.discovered_at.isoformat()
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class RobotDiscoveryAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    fleet_id: str
    robot_id: str

    event_type: RobotDiscoveryEventType

    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "fleet_id": self.fleet_id,
            "robot_id": self.robot_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# AUTHORIZATION CALLBACK
# ============================================================

DiscoveryAuthorizer = Callable[
    [RobotDiscoveryAnnouncement],
    bool,
]


# ============================================================
# ROBOT DISCOVERY MANAGER
# ============================================================

class RoboticsRobotDiscovery:

    """
    In-memory simulation robot discovery service.

    Discovery workflow:
        1. Receive trusted simulation announcement.
        2. Validate metadata and timestamp.
        3. Check discovery authorization.
        4. Register robot.
        5. Receive periodic heartbeat.
        6. Evaluate freshness.
        7. Discover available robots.

    Endpoint metadata is never contacted.
    """

    def __init__(
        self,
        config: RobotDiscoveryConfig | None = None,
        *,
        announcement_authorizer: (
            DiscoveryAuthorizer | None
        ) = None,
    ) -> None:
        self.config = (
            config or RobotDiscoveryConfig()
        )

        self.announcement_authorizer = (
            announcement_authorizer
        )

        self._robots: dict[
            tuple[str, str, str],
            DiscoveredRobotRecord
        ] = {}

        self._audit_events: deque[
            RobotDiscoveryAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # KEY
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
    ) -> DiscoveredRobotRecord:
        record = self._robots.get(
            self._key(
                tenant_id,
                fleet_id,
                robot_id,
            )
        )

        if record is None:
            raise DiscoveredRobotNotFound(
                "Robot not discovered"
            )

        return record

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
        event_type: RobotDiscoveryEventType,
        details: str,
    ) -> None:
        self._audit_events.append(
            RobotDiscoveryAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=tenant_id,
                fleet_id=fleet_id,
                robot_id=robot_id,
                event_type=event_type,
                details=details,
            )
        )

    # --------------------------------------------------------
    # TIMESTAMP VALIDATION
    # --------------------------------------------------------

    def _validate_fresh_timestamp(
        self,
        timestamp: datetime,
    ) -> None:
        age = (
            utc_now()
            - validate_timestamp(timestamp)
        ).total_seconds()

        if (
            age
            < -self.config.maximum_clock_skew_seconds
        ):
            raise DiscoveryValidationError(
                "Timestamp too far in future"
            )

        if (
            age
            > self.config.discovery_ttl_seconds
        ):
            raise DiscoveryValidationError(
                "Discovery message expired"
            )

    # --------------------------------------------------------
    # AUTHORIZE
    # --------------------------------------------------------

    def _authorized(
        self,
        announcement: RobotDiscoveryAnnouncement,
    ) -> bool:
        callback = self.announcement_authorizer

        if callback is None:
            return False

        try:
            return callback(announcement) is True
        except Exception:
            return False

    # --------------------------------------------------------
    # DISCOVER ROBOT
    # --------------------------------------------------------

    def discover_robot(
        self,
        announcement: RobotDiscoveryAnnouncement,
    ) -> dict[str, Any]:
        if not isinstance(
            announcement,
            RobotDiscoveryAnnouncement,
        ):
            raise DiscoveryValidationError(
                "Invalid announcement"
            )

        self._validate_fresh_timestamp(
            announcement.timestamp
        )

        if not self._authorized(announcement):
            raise DiscoveryPermissionDenied(
                "Discovery authorization denied"
            )

        with self._lock:
            key = self._key(
                announcement.tenant_id,
                announcement.fleet_id,
                announcement.robot_id,
            )

            existing = self._robots.get(key)

            for other_key, other in self._robots.items():
                if (
                    other_key != key
                    and other_key[:2] == key[:2]
                    and other.announcement.endpoint
                    == announcement.endpoint
                    and other.status
                    != RobotDiscoveryStatus.EXPIRED
                ):
                    raise DiscoveryConflict(
                        "Endpoint already registered "
                        "to another robot"
                    )

            if existing is not None:
                if existing.blocked:
                    raise DiscoveryPermissionDenied(
                        "Robot discovery is blocked"
                    )

                if (
                    announcement.sequence_number
                    <= existing.announcement.sequence_number
                ):
                    raise DiscoveryValidationError(
                        "Duplicate or out-of-order announcement"
                    )

                existing.announcement = announcement

                existing.last_announcement_monotonic = (
                    time.monotonic()
                )

                self._audit(
                    tenant_id=announcement.tenant_id,
                    fleet_id=announcement.fleet_id,
                    robot_id=announcement.robot_id,
                    event_type=(
                        RobotDiscoveryEventType.ROBOT_UPDATED
                    ),
                    details="Robot announcement updated",
                )

                return existing.to_dict()

            if (
                len(self._robots)
                >= self.config.maximum_robots
            ):
                raise RobotDiscoveryError(
                    "Maximum discovery capacity reached"
                )

            record = DiscoveredRobotRecord(
                announcement=announcement
            )

            self._robots[key] = record

            self._audit(
                tenant_id=announcement.tenant_id,
                fleet_id=announcement.fleet_id,
                robot_id=announcement.robot_id,
                event_type=(
                    RobotDiscoveryEventType.ROBOT_DISCOVERED
                ),
                details="Robot discovered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # HEARTBEAT
    # --------------------------------------------------------

    def receive_heartbeat(
        self,
        heartbeat: RobotDiscoveryHeartbeat,
    ) -> dict[str, Any]:
        if not isinstance(
            heartbeat,
            RobotDiscoveryHeartbeat,
        ):
            raise DiscoveryValidationError(
                "Invalid heartbeat"
            )

        self._validate_fresh_timestamp(
            heartbeat.timestamp
        )

        age = (
            utc_now()
            - validate_timestamp(
                heartbeat.timestamp
            )
        ).total_seconds()

        if (
            age
            > self.config.heartbeat_timeout_seconds
        ):
            raise DiscoveryValidationError(
                "Heartbeat too old"
            )

        with self._lock:
            record = self._get(
                heartbeat.tenant_id,
                heartbeat.fleet_id,
                heartbeat.robot_id,
            )

            self._refresh(record)

            if record.blocked:
                raise DiscoveryPermissionDenied(
                    "Robot is blocked"
                )

            if (
                record.status
                == RobotDiscoveryStatus.EXPIRED
            ):
                raise DiscoveryConflict(
                    "Expired robot must be rediscovered"
                )

            if (
                heartbeat.sequence_number
                <= record.last_heartbeat_sequence
            ):
                raise DiscoveryValidationError(
                    "Duplicate or out-of-order heartbeat"
                )

            record.last_heartbeat_sequence = (
                heartbeat.sequence_number
            )

            record.last_heartbeat_monotonic = (
                time.monotonic()
            )

            record.heartbeat_count += 1

            if not heartbeat.operational:
                record.status = (
                    RobotDiscoveryStatus.OFFLINE
                )

            elif heartbeat.degraded:
                record.status = (
                    RobotDiscoveryStatus.DEGRADED
                )

            else:
                record.status = (
                    RobotDiscoveryStatus.ONLINE
                )

            self._audit(
                tenant_id=heartbeat.tenant_id,
                fleet_id=heartbeat.fleet_id,
                robot_id=heartbeat.robot_id,
                event_type=(
                    RobotDiscoveryEventType.HEARTBEAT_RECEIVED
                ),
                details=record.status.value,
            )

            return record.to_dict()

    # --------------------------------------------------------
    # REFRESH DISCOVERY STATUS
    # --------------------------------------------------------

    def _refresh(
        self,
        record: DiscoveredRobotRecord,
    ) -> None:
        if record.blocked:
            record.status = (
                RobotDiscoveryStatus.BLOCKED
            )
            return

        now = time.monotonic()

        announcement_age = (
            now - record.last_announcement_monotonic
        )

        if (
            announcement_age
            > self.config.discovery_ttl_seconds
        ):
            if (
                record.status
                != RobotDiscoveryStatus.EXPIRED
            ):
                record.status = (
                    RobotDiscoveryStatus.EXPIRED
                )

                self._audit(
                    tenant_id=record.announcement.tenant_id,
                    fleet_id=record.announcement.fleet_id,
                    robot_id=record.announcement.robot_id,
                    event_type=(
                        RobotDiscoveryEventType.ROBOT_EXPIRED
                    ),
                    details="Discovery TTL expired",
                )

            return

        heartbeat_time = (
            record.last_heartbeat_monotonic
        )

        if heartbeat_time is None:
            record.status = (
                RobotDiscoveryStatus.DISCOVERED
            )
            return

        heartbeat_age = (
            now - heartbeat_time
        )

        if (
            heartbeat_age
            > self.config.heartbeat_timeout_seconds
        ):
            if (
                record.status
                != RobotDiscoveryStatus.OFFLINE
            ):
                record.status = (
                    RobotDiscoveryStatus.OFFLINE
                )

                self._audit(
                    tenant_id=record.announcement.tenant_id,
                    fleet_id=record.announcement.fleet_id,
                    robot_id=record.announcement.robot_id,
                    event_type=(
                        RobotDiscoveryEventType.ROBOT_OFFLINE
                    ),
                    details="Heartbeat timeout",
                )

    # --------------------------------------------------------
    # BLOCK ROBOT
    # --------------------------------------------------------

    def block_robot(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise DiscoveryPermissionDenied(
                "Authorization required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                fleet_id,
                robot_id,
            )

            record.blocked = True

            record.status = (
                RobotDiscoveryStatus.BLOCKED
            )

            self._audit(
                tenant_id=tenant_id,
                fleet_id=fleet_id,
                robot_id=robot_id,
                event_type=(
                    RobotDiscoveryEventType.ROBOT_BLOCKED
                ),
                details="Robot blocked",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # UNBLOCK ROBOT
    # --------------------------------------------------------

    def unblock_robot(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise DiscoveryPermissionDenied(
                "Authorization required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                fleet_id,
                robot_id,
            )

            record.blocked = False

            # A fresh announcement and heartbeat
            # are required before ONLINE status.
            record.status = (
                RobotDiscoveryStatus.EXPIRED
            )

            record.last_announcement_monotonic = (
                0.0
            )

            record.last_heartbeat_monotonic = None

            self._audit(
                tenant_id=tenant_id,
                fleet_id=fleet_id,
                robot_id=robot_id,
                event_type=(
                    RobotDiscoveryEventType.ROBOT_UNBLOCKED
                ),
                details="Robot unblocked; rediscovery required",
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
            record = self._get(
                tenant_id,
                fleet_id,
                robot_id,
            )

            self._refresh(record)

            return record.to_dict()

    # --------------------------------------------------------
    # DISCOVER ROBOTS
    # --------------------------------------------------------

    def list_robots(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        required_capabilities: tuple[str, ...] = (),
        online_only: bool = False,
        robot_type: str | None = None,
    ) -> list[dict[str, Any]]:
        validate_capabilities(
            required_capabilities
        )

        if robot_type is not None:
            validate_id(
                robot_type,
                "robot_type",
            )

        with self._lock:
            results = []

            for record in self._robots.values():
                announcement = record.announcement

                if (
                    announcement.tenant_id != tenant_id
                    or announcement.fleet_id != fleet_id
                ):
                    continue

                self._refresh(record)

                if (
                    online_only
                    and record.status
                    != RobotDiscoveryStatus.ONLINE
                ):
                    continue

                if (
                    robot_type is not None
                    and announcement.robot_type != robot_type
                ):
                    continue

                if not set(
                    required_capabilities
                ).issubset(
                    announcement.capabilities
                ):
                    continue

                results.append(
                    record.to_dict()
                )

            return sorted(
                results,
                key=lambda item: item["robot_id"],
            )

    # --------------------------------------------------------
    # CHECK AVAILABILITY
    # --------------------------------------------------------

    def check_availability(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
        required_capabilities: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        validate_capabilities(
            required_capabilities
        )

        with self._lock:
            record = self._get(
                tenant_id,
                fleet_id,
                robot_id,
            )

            self._refresh(record)

            available = (
                record.status
                == RobotDiscoveryStatus.ONLINE
                and set(
                    required_capabilities
                ).issubset(
                    record.announcement.capabilities
                )
            )

            return {
                "tenant_id": tenant_id,
                "fleet_id": fleet_id,
                "robot_id": robot_id,
                "available_for_simulation": available,
                "status": record.status.value,
                "required_capabilities": list(
                    required_capabilities
                ),
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            records = [
                record
                for record in self._robots.values()
                if (
                    record.announcement.tenant_id
                    == tenant_id
                    and record.announcement.fleet_id
                    == fleet_id
                )
            ]

            for record in records:
                self._refresh(record)

            return {
                "total_robots": len(records),
                "online": sum(
                    record.status
                    == RobotDiscoveryStatus.ONLINE
                    for record in records
                ),
                "offline": sum(
                    record.status
                    == RobotDiscoveryStatus.OFFLINE
                    for record in records
                ),
                "degraded": sum(
                    record.status
                    == RobotDiscoveryStatus.DEGRADED
                    for record in records
                ),
                "expired": sum(
                    record.status
                    == RobotDiscoveryStatus.EXPIRED
                    for record in records
                ),
                "blocked": sum(
                    record.status
                    == RobotDiscoveryStatus.BLOCKED
                    for record in records
                ),
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
        fleet_id: str,
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
                        and event.fleet_id == fleet_id
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
# DEMONSTRATION
# ============================================================

def main() -> None:
    discovery = RoboticsRobotDiscovery(
        announcement_authorizer=lambda announcement: (
            announcement.tenant_id == "tenant_alpha"
            and announcement.discovery_protocol
            == DiscoveryProtocol.SIMULATION
        )
    )

    scope = {
        "tenant_id": "tenant_alpha",
        "fleet_id": "factory_fleet",
    }

    robots = (
        (
            "robot_001",
            "Inspection AMR",
            "amr",
            "127.0.0.1",
            9001,
            ("navigation", "inspection"),
        ),
        (
            "robot_002",
            "Mobile Manipulator",
            "mobile_manipulator",
            "127.0.0.2",
            9002,
            ("navigation", "manipulation"),
        ),
    )

    for (
        robot_id,
        name,
        robot_type,
        host,
        port,
        capabilities,
    ) in robots:
        discovery.discover_robot(
            RobotDiscoveryAnnouncement(
                **scope,
                robot_id=robot_id,
                robot_name=name,
                robot_type=robot_type,
                endpoint=RobotEndpoint(
                    host=host,
                    port=port,
                ),
                discovery_protocol=(
                    DiscoveryProtocol.SIMULATION
                ),
                capabilities=capabilities,
                sequence_number=1,
            )
        )

        discovery.receive_heartbeat(
            RobotDiscoveryHeartbeat(
                **scope,
                robot_id=robot_id,
                sequence_number=1,
                operational=True,
            )
        )

    print(
        "Available navigation robots:",
        json.dumps(
            discovery.list_robots(
                **scope,
                required_capabilities=(
                    "navigation",
                ),
                online_only=True,
            ),
            indent=2,
        ),
    )

    print(
        "\nFleet discovery statistics:",
        json.dumps(
            discovery.statistics(**scope),
            indent=2,
        ),
    )

    report = discovery.export_json(
        "data/robotics/robot_discovery_report.json",
        **scope,
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()

"""Geofencing module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/security/geofencing.py

Python 3.11+
Standard library only.

Features:
    - GPS and indoor Cartesian geofencing
    - Circular and polygonal zones
    - Allowed operating zones
    - Restricted/no-go zones
    - Tenant-scoped zone management
    - Robot-specific zone assignments
    - Boundary breach detection
    - Movement destination validation
    - Route segment validation
    - Telemetry freshness checks
    - Safety alerts
    - Audit logging
    - JSON export

IMPORTANT:
    Simulation and supervisory checks only.
    This module does not control physical hardware.

    Real robots require independent, validated
    safety controllers and emergency-stop systems.
"""

from __future__ import annotations

import json
import math
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


# ============================================================
# EXCEPTIONS
# ============================================================

class GeofencingError(Exception):
    pass


class GeofenceValidationError(GeofencingError):
    pass


class GeofenceNotFoundError(GeofencingError):
    pass


class GeofenceAccessDenied(GeofencingError):
    pass


# ============================================================
# ENUMS
# ============================================================

class CoordinateSystem(str, Enum):
    GPS = "gps"
    CARTESIAN = "cartesian"


class GeofenceZoneType(str, Enum):
    ALLOWED = "allowed"
    RESTRICTED = "restricted"


class GeofenceShape(str, Enum):
    CIRCLE = "circle"
    POLYGON = "polygon"


class GeofenceDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class GeofenceEventType(str, Enum):
    ZONE_CREATED = "zone_created"
    ZONE_REMOVED = "zone_removed"
    ROBOT_ASSIGNED = "robot_assigned"
    POSITION_ALLOWED = "position_allowed"
    BOUNDARY_BREACH = "boundary_breach"
    RESTRICTED_ZONE_ENTERED = (
        "restricted_zone_entered"
    )
    MOVEMENT_DENIED = "movement_denied"
    STALE_TELEMETRY = "stale_telemetry"


class GeofenceSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def validate_id(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            character.isalnum()
            or character in "_-"
            for character in value
        )
    ):
        raise GeofenceValidationError(
            f"Invalid {name}"
        )

    return value


def finite_number(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise GeofenceValidationError(
            f"{name} must be finite"
        )

    return float(value)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class GeofencingConfig:
    maximum_zones: int = 10000
    maximum_polygon_vertices: int = 1000
    maximum_audit_events: int = 20000

    maximum_telemetry_age_seconds: float = 5.0

    maximum_route_step_meters: float = 1.0

    require_allowed_zone: bool = True

    deny_stale_telemetry: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_zones",
            "maximum_polygon_vertices",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise GeofenceValidationError(
                    f"{name} must be positive"
                )

        for name in (
            "maximum_telemetry_age_seconds",
            "maximum_route_step_meters",
        ):
            value = finite_number(
                getattr(self, name),
                name,
            )

            if value <= 0:
                raise GeofenceValidationError(
                    f"{name} must be positive"
                )


# ============================================================
# POSITION
# ============================================================

@dataclass(frozen=True)
class GeoPosition:
    x: float
    y: float

    coordinate_system: CoordinateSystem

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        x = finite_number(self.x, "x")
        y = finite_number(self.y, "y")

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

        system = CoordinateSystem(
            self.coordinate_system
        )

        object.__setattr__(
            self,
            "coordinate_system",
            system,
        )

        if system == CoordinateSystem.GPS:
            # x = longitude, y = latitude
            if not (-180 <= x <= 180):
                raise GeofenceValidationError(
                    "Longitude out of range"
                )

            if not (-90 <= y <= 90):
                raise GeofenceValidationError(
                    "Latitude out of range"
                )

        if (
            self.timestamp.tzinfo is None
            or self.timestamp.utcoffset() is None
        ):
            raise GeofenceValidationError(
                "Timestamp must be timezone-aware"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "coordinate_system": (
                self.coordinate_system.value
            ),
            "timestamp": (
                self.timestamp.isoformat()
            ),
        }


# ============================================================
# GEOMETRY
# ============================================================

EARTH_RADIUS_METERS = 6371008.8


def distance_meters(
    first: GeoPosition,
    second: GeoPosition,
) -> float:
    if (
        first.coordinate_system
        != second.coordinate_system
    ):
        raise GeofenceValidationError(
            "Coordinate systems do not match"
        )

    if (
        first.coordinate_system
        == CoordinateSystem.CARTESIAN
    ):
        return math.hypot(
            second.x - first.x,
            second.y - first.y,
        )

    latitude_1 = math.radians(
        first.y
    )

    latitude_2 = math.radians(
        second.y
    )

    latitude_delta = (
        latitude_2 - latitude_1
    )

    longitude_delta = math.radians(
        second.x - first.x
    )

    a = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_1)
        * math.cos(latitude_2)
        * math.sin(longitude_delta / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    return (
        2
        * EARTH_RADIUS_METERS
        * math.asin(math.sqrt(a))
    )


def point_on_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    epsilon: float = 1e-9,
) -> bool:
    cross = (
        (px - ax) * (by - ay)
        - (py - ay) * (bx - ax)
    )

    if abs(cross) > epsilon:
        return False

    return (
        min(ax, bx) - epsilon
        <= px
        <= max(ax, bx) + epsilon
        and min(ay, by) - epsilon
        <= py
        <= max(ay, by) + epsilon
    )


def point_in_polygon(
    point: GeoPosition,
    vertices: tuple[GeoPosition, ...],
) -> bool:
    if len(vertices) < 3:
        return False

    inside = False

    px = point.x
    py = point.y

    for index in range(len(vertices)):
        first = vertices[index]
        second = vertices[
            (index + 1) % len(vertices)
        ]

        if point_on_segment(
            px,
            py,
            first.x,
            first.y,
            second.x,
            second.y,
        ):
            return True

        if (first.y > py) != (second.y > py):
            intersection_x = (
                first.x
                + (py - first.y)
                * (second.x - first.x)
                / (second.y - first.y)
            )

            if px < intersection_x:
                inside = not inside

    return inside


# ============================================================
# GEOFENCE ZONE
# ============================================================

@dataclass(frozen=True)
class GeofenceZone:
    zone_id: str
    tenant_id: str
    name: str

    zone_type: GeofenceZoneType
    shape: GeofenceShape
    coordinate_system: CoordinateSystem

    center: GeoPosition | None = None
    radius_meters: float | None = None

    vertices: tuple[
        GeoPosition,
        ...
    ] = ()

    active: bool = True

    created_at: str = field(
        default_factory=utc_iso
    )

    def __post_init__(self) -> None:
        validate_id(
            self.zone_id,
            "zone_id",
        )

        validate_id(
            self.tenant_id,
            "tenant_id",
        )

        if not self.name.strip():
            raise GeofenceValidationError(
                "Zone name is required"
            )

        shape = GeofenceShape(self.shape)
        system = CoordinateSystem(
            self.coordinate_system
        )

        object.__setattr__(
            self,
            "shape",
            shape,
        )

        object.__setattr__(
            self,
            "coordinate_system",
            system,
        )

        if shape == GeofenceShape.CIRCLE:
            if self.center is None:
                raise GeofenceValidationError(
                    "Circle requires center"
                )

            if (
                self.center.coordinate_system
                != system
            ):
                raise GeofenceValidationError(
                    "Circle coordinate mismatch"
                )

            if (
                self.radius_meters is None
                or finite_number(
                    self.radius_meters,
                    "radius_meters",
                ) <= 0
            ):
                raise GeofenceValidationError(
                    "Circle requires positive radius"
                )

        else:
            if len(self.vertices) < 3:
                raise GeofenceValidationError(
                    "Polygon requires 3+ vertices"
                )

            for vertex in self.vertices:
                if vertex.coordinate_system != system:
                    raise GeofenceValidationError(
                        "Polygon coordinate mismatch"
                    )

    def contains(
        self,
        position: GeoPosition,
    ) -> bool:
        if (
            position.coordinate_system
            != self.coordinate_system
        ):
            raise GeofenceValidationError(
                "Position coordinate mismatch"
            )

        if self.shape == GeofenceShape.CIRCLE:
            assert self.center is not None
            assert self.radius_meters is not None

            return (
                distance_meters(
                    self.center,
                    position,
                )
                <= self.radius_meters
            )

        return point_in_polygon(
            position,
            self.vertices,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "zone_type": self.zone_type.value,
            "shape": self.shape.value,
            "coordinate_system": (
                self.coordinate_system.value
            ),
            "center": (
                self.center.to_dict()
                if self.center
                else None
            ),
            "radius_meters": self.radius_meters,
            "vertices": [
                vertex.to_dict()
                for vertex in self.vertices
            ],
            "active": self.active,
            "created_at": self.created_at,
        }


# ============================================================
# ROBOT GEOFENCE ASSIGNMENT
# ============================================================

@dataclass
class RobotGeofenceAssignment:
    robot_id: str
    tenant_id: str

    allowed_zone_ids: set[str] = field(
        default_factory=set
    )

    restricted_zone_ids: set[str] = field(
        default_factory=set
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "tenant_id": self.tenant_id,
            "allowed_zone_ids": sorted(
                self.allowed_zone_ids
            ),
            "restricted_zone_ids": sorted(
                self.restricted_zone_ids
            ),
        }


# ============================================================
# DECISION
# ============================================================

@dataclass(frozen=True)
class GeofenceEvaluation:
    evaluation_id: str

    tenant_id: str
    robot_id: str

    decision: GeofenceDecision
    reason: str

    matched_allowed_zones: tuple[str, ...]
    matched_restricted_zones: tuple[str, ...]

    evaluated_at: str = field(
        default_factory=utc_iso
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "decision": self.decision.value,
            "reason": self.reason,
            "matched_allowed_zones": list(
                self.matched_allowed_zones
            ),
            "matched_restricted_zones": list(
                self.matched_restricted_zones
            ),
            "evaluated_at": self.evaluated_at,
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class GeofenceAuditEvent:
    event_id: str
    timestamp: str

    tenant_id: str
    robot_id: str | None

    event_type: GeofenceEventType
    severity: GeofenceSeverity

    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "message": self.message,
        }


# ============================================================
# GEOFENCING SERVICE
# ============================================================

class RoboticsGeofencingService:

    def __init__(
        self,
        config: GeofencingConfig | None = None,
    ) -> None:
        self.config = (
            config or GeofencingConfig()
        )

        self.zones: dict[
            str,
            GeofenceZone
        ] = {}

        self.assignments: dict[
            tuple[str, str],
            RobotGeofenceAssignment
        ] = {}

        self.audit_events: list[
            GeofenceAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        *,
        tenant_id: str,
        robot_id: str | None,
        event_type: GeofenceEventType,
        severity: GeofenceSeverity,
        message: str,
    ) -> None:
        self.audit_events.append(
            GeofenceAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_iso(),
                tenant_id=tenant_id,
                robot_id=robot_id,
                event_type=event_type,
                severity=severity,
                message=message,
            )
        )

        if (
            len(self.audit_events)
            > self.config.maximum_audit_events
        ):
            self.audit_events.pop(0)

    # --------------------------------------------------------
    # REGISTER ZONE
    # --------------------------------------------------------

    def register_zone(
        self,
        zone: GeofenceZone,
    ) -> None:
        with self._lock:
            if zone.zone_id in self.zones:
                raise GeofenceValidationError(
                    "Zone already exists"
                )

            if (
                len(self.zones)
                >= self.config.maximum_zones
            ):
                raise GeofenceValidationError(
                    "Zone capacity reached"
                )

            if (
                len(zone.vertices)
                > self.config.maximum_polygon_vertices
            ):
                raise GeofenceValidationError(
                    "Polygon has too many vertices"
                )

            self.zones[
                zone.zone_id
            ] = zone

            self._audit(
                tenant_id=zone.tenant_id,
                robot_id=None,
                event_type=(
                    GeofenceEventType.ZONE_CREATED
                ),
                severity=GeofenceSeverity.INFO,
                message=(
                    f"Geofence zone created: "
                    f"{zone.zone_id}"
                ),
            )

    # --------------------------------------------------------
    # ASSIGN ZONES TO ROBOT
    # --------------------------------------------------------

    def assign_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        allowed_zone_ids: Iterable[str],
        restricted_zone_ids: Iterable[str] = (),
    ) -> RobotGeofenceAssignment:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            robot_id,
            "robot_id",
        )

        allowed = set(
            allowed_zone_ids
        )

        restricted = set(
            restricted_zone_ids
        )

        with self._lock:
            for zone_id in allowed:
                zone = self.zones.get(
                    zone_id
                )

                if (
                    zone is None
                    or zone.tenant_id != tenant_id
                    or zone.zone_type
                    != GeofenceZoneType.ALLOWED
                ):
                    raise GeofenceAccessDenied(
                        "Invalid allowed zone assignment"
                    )

            for zone_id in restricted:
                zone = self.zones.get(
                    zone_id
                )

                if (
                    zone is None
                    or zone.tenant_id != tenant_id
                    or zone.zone_type
                    != GeofenceZoneType.RESTRICTED
                ):
                    raise GeofenceAccessDenied(
                        "Invalid restricted zone assignment"
                    )

            assignment = RobotGeofenceAssignment(
                robot_id=robot_id,
                tenant_id=tenant_id,
                allowed_zone_ids=allowed,
                restricted_zone_ids=restricted,
            )

            self.assignments[
                (tenant_id, robot_id)
            ] = assignment

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                event_type=(
                    GeofenceEventType.ROBOT_ASSIGNED
                ),
                severity=GeofenceSeverity.INFO,
                message="Robot geofence assignment updated",
            )

            return assignment

    # --------------------------------------------------------
    # EVALUATE POSITION
    # --------------------------------------------------------

    def evaluate_position(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        position: GeoPosition,
    ) -> GeofenceEvaluation:
        with self._lock:
            assignment = self.assignments.get(
                (tenant_id, robot_id)
            )

            if assignment is None:
                raise GeofenceAccessDenied(
                    "Robot has no geofence assignment"
                )

            age = (
                utc_now()
                - position.timestamp.astimezone(
                    timezone.utc
                )
            ).total_seconds()

            if (
                self.config.deny_stale_telemetry
                and (
                    age < -1
                    or age
                    > self.config.maximum_telemetry_age_seconds
                )
            ):
                self._audit(
                    tenant_id=tenant_id,
                    robot_id=robot_id,
                    event_type=(
                        GeofenceEventType.STALE_TELEMETRY
                    ),
                    severity=GeofenceSeverity.HIGH,
                    message="Position telemetry is stale",
                )

                return self._decision(
                    tenant_id,
                    robot_id,
                    GeofenceDecision.DENY,
                    "Position telemetry is stale",
                    (),
                    (),
                )

            allowed_matches = []
            restricted_matches = []

            try:
                for zone_id in assignment.allowed_zone_ids:
                    zone = self.zones.get(zone_id)

                    if zone is None or not zone.active:
                        continue

                    if zone.contains(position):
                        allowed_matches.append(zone_id)

                for zone_id in assignment.restricted_zone_ids:
                    zone = self.zones.get(zone_id)

                    if zone is None or not zone.active:
                        continue

                    if zone.contains(position):
                        restricted_matches.append(zone_id)

            except GeofenceValidationError:
                return self._decision(
                    tenant_id,
                    robot_id,
                    GeofenceDecision.DENY,
                    "Coordinate system mismatch",
                    (),
                    (),
                )

            if restricted_matches:
                self._audit(
                    tenant_id=tenant_id,
                    robot_id=robot_id,
                    event_type=(
                        GeofenceEventType.RESTRICTED_ZONE_ENTERED
                    ),
                    severity=GeofenceSeverity.CRITICAL,
                    message="Restricted zone detected",
                )

                return self._decision(
                    tenant_id,
                    robot_id,
                    GeofenceDecision.DENY,
                    "Position is inside restricted zone",
                    allowed_matches,
                    restricted_matches,
                )

            if (
                self.config.require_allowed_zone
                and not allowed_matches
            ):
                self._audit(
                    tenant_id=tenant_id,
                    robot_id=robot_id,
                    event_type=(
                        GeofenceEventType.BOUNDARY_BREACH
                    ),
                    severity=GeofenceSeverity.HIGH,
                    message="Robot outside allowed area",
                )

                return self._decision(
                    tenant_id,
                    robot_id,
                    GeofenceDecision.DENY,
                    "Position outside allowed operating zones",
                    allowed_matches,
                    restricted_matches,
                )

            return self._decision(
                tenant_id,
                robot_id,
                GeofenceDecision.ALLOW,
                "Position satisfies geofence policy",
                allowed_matches,
                restricted_matches,
            )

    def _decision(
        self,
        tenant_id: str,
        robot_id: str,
        decision: GeofenceDecision,
        reason: str,
        allowed: Iterable[str],
        restricted: Iterable[str],
    ) -> GeofenceEvaluation:
        return GeofenceEvaluation(
            evaluation_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            robot_id=robot_id,
            decision=decision,
            reason=reason,
            matched_allowed_zones=tuple(
                sorted(allowed)
            ),
            matched_restricted_zones=tuple(
                sorted(restricted)
            ),
        )

    # --------------------------------------------------------
    # VALIDATE MOVEMENT DESTINATION
    # --------------------------------------------------------

    def validate_destination(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        destination: GeoPosition,
    ) -> GeofenceEvaluation:
        return self.evaluate_position(
            tenant_id=tenant_id,
            robot_id=robot_id,
            position=destination,
        )

    # --------------------------------------------------------
    # VALIDATE ROUTE
    # --------------------------------------------------------

    def validate_route(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        waypoints: Iterable[GeoPosition],
    ) -> list[GeofenceEvaluation]:
        points = list(waypoints)

        if not points:
            raise GeofenceValidationError(
                "Route must contain waypoints"
            )

        results: list[
            GeofenceEvaluation
        ] = []

        for point in points:
            result = self.evaluate_position(
                tenant_id=tenant_id,
                robot_id=robot_id,
                position=point,
            )

            results.append(result)

            if result.decision == GeofenceDecision.DENY:
                break

        return results

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            return {
                "zones": len(self.zones),
                "robot_assignments": len(
                    self.assignments
                ),
                "audit_events": len(
                    self.audit_events
                ),
                "allowed_zones": sum(
                    zone.zone_type
                    == GeofenceZoneType.ALLOWED
                    for zone in self.zones.values()
                ),
                "restricted_zones": sum(
                    zone.zone_type
                    == GeofenceZoneType.RESTRICTED
                    for zone in self.zones.values()
                ),
            }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_iso(),
                "statistics": self.statistics(),
                "zones": [
                    zone.to_dict()
                    for zone in self.zones.values()
                ],
                "assignments": [
                    assignment.to_dict()
                    for assignment in self.assignments.values()
                ],
                "audit_events": [
                    event.to_dict()
                    for event in self.audit_events
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
    service = RoboticsGeofencingService()

    service.register_zone(
        GeofenceZone(
            zone_id="warehouse_allowed",
            tenant_id="tenant_alpha",
            name="Warehouse Operating Area",
            zone_type=GeofenceZoneType.ALLOWED,
            shape=GeofenceShape.POLYGON,
            coordinate_system=CoordinateSystem.CARTESIAN,
            vertices=(
                GeoPosition(0, 0, CoordinateSystem.CARTESIAN),
                GeoPosition(100, 0, CoordinateSystem.CARTESIAN),
                GeoPosition(100, 100, CoordinateSystem.CARTESIAN),
                GeoPosition(0, 100, CoordinateSystem.CARTESIAN),
            ),
        )
    )

    service.register_zone(
        GeofenceZone(
            zone_id="restricted_machine_area",
            tenant_id="tenant_alpha",
            name="Restricted Machinery Zone",
            zone_type=GeofenceZoneType.RESTRICTED,
            shape=GeofenceShape.CIRCLE,
            coordinate_system=CoordinateSystem.CARTESIAN,
            center=GeoPosition(
                50,
                50,
                CoordinateSystem.CARTESIAN,
            ),
            radius_meters=10,
        )
    )

    service.assign_robot(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        allowed_zone_ids={
            "warehouse_allowed"
        },
        restricted_zone_ids={
            "restricted_machine_area"
        },
    )

    safe = service.evaluate_position(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        position=GeoPosition(
            20,
            20,
            CoordinateSystem.CARTESIAN,
        ),
    )

    unsafe = service.evaluate_position(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        position=GeoPosition(
            50,
            50,
            CoordinateSystem.CARTESIAN,
        ),
    )

    print(
        json.dumps(
            {
                "safe_position": safe.to_dict(),
                "unsafe_position": unsafe.to_dict(),
                "statistics": service.statistics(),
            },
            indent=2,
        )
    )

    service.export_json(
        "data/security/geofencing_report.json"
    )


if __name__ == "__main__":
    main()

"""Speed Limits module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/security/speed_limits.py

Python 3.11+
Standard library only.

Features:
    - Robot-specific speed limits
    - Global and zone speed limits
    - Dynamic human proximity limits
    - Collision risk speed restrictions
    - Battery and thermal restrictions
    - Emergency and interlock restrictions
    - Telemetry freshness checks
    - Overspeed detection
    - Speed command validation
    - Fleet-wide assessments
    - Tenant isolation
    - Audit logging
    - JSON report export

SAFETY:
    Simulation and supervisory monitoring ONLY.

    No motor commands or physical braking
    commands are issued by this module.

    A speed advisory is not a safety-rated
    speed monitoring or limiting function.

    Real robots require independently validated
    speed supervision and protective stopping.
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
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class SpeedLimitError(Exception):
    pass


class SpeedLimitValidationError(SpeedLimitError):
    pass


class SpeedLimitNotFound(SpeedLimitError):
    pass


# ============================================================
# ENUMS
# ============================================================

class SpeedRiskLevel(str, Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class SpeedDecision(str, Enum):
    ALLOW_SIMULATION = "allow_simulation"
    REDUCE_SPEED = "reduce_speed"
    STOP_RECOMMENDED = "stop_recommended"
    DENY = "deny"


class SpeedRestrictionReason(str, Enum):
    GLOBAL_LIMIT = "global_limit"
    ROBOT_LIMIT = "robot_limit"
    ZONE_LIMIT = "zone_limit"
    HUMAN_PROXIMITY = "human_proximity"
    COLLISION_RISK = "collision_risk"
    LOW_BATTERY = "low_battery"
    OVERHEATING = "overheating"
    EMERGENCY_STOP = "emergency_stop"
    COMMAND_INTERLOCK = "command_interlock"
    SENSOR_FAILURE = "sensor_failure"
    COMMUNICATION_FAILURE = "communication_failure"
    STALE_TELEMETRY = "stale_telemetry"
    GEOFENCE_BREACH = "geofence_breach"
    HAZARD_DETECTED = "hazard_detected"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            char.isalnum() or char in "_-"
            for char in value
        )
    ):
        raise SpeedLimitValidationError(
            f"Invalid {name}"
        )

    return value


def finite(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise SpeedLimitValidationError(
            f"{name} must be a finite number"
        )

    return float(value)


def nonnegative(
    value: float,
    name: str,
) -> float:
    result = finite(value, name)

    if result < 0:
        raise SpeedLimitValidationError(
            f"{name} cannot be negative"
        )

    return result


def validate_time(
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise SpeedLimitValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SpeedLimitConfig:
    global_max_speed_mps: float = 2.0

    human_caution_distance_m: float = 3.0
    human_critical_distance_m: float = 1.0
    human_caution_max_speed_mps: float = 0.5

    collision_caution_score: float = 30.0
    collision_high_score: float = 60.0
    collision_critical_score: float = 85.0
    collision_caution_max_speed_mps: float = 0.5

    minimum_battery_percent: float = 20.0
    low_battery_max_speed_mps: float = 0.5

    maximum_temperature_c: float = 70.0
    thermal_caution_temperature_c: float = 60.0
    thermal_caution_max_speed_mps: float = 0.5

    maximum_telemetry_age_seconds: float = 2.0

    speed_tolerance_mps: float = 0.02

    maximum_audit_events: int = 20000

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "global_max_speed_mps",
            "human_caution_distance_m",
            "human_critical_distance_m",
            "human_caution_max_speed_mps",
            "collision_caution_max_speed_mps",
            "low_battery_max_speed_mps",
            "thermal_caution_max_speed_mps",
            "maximum_telemetry_age_seconds",
            "speed_tolerance_mps",
        ):
            nonnegative(getattr(self, name), name)

        if self.global_max_speed_mps <= 0:
            raise SpeedLimitValidationError(
                "Global maximum speed must be positive"
            )

        if not (
            0 < self.human_critical_distance_m
            < self.human_caution_distance_m
        ):
            raise SpeedLimitValidationError(
                "Human distance thresholds must be ordered"
            )

        if not (
            0 <= self.collision_caution_score
            < self.collision_high_score
            < self.collision_critical_score
            <= 100
        ):
            raise SpeedLimitValidationError(
                "Collision risk thresholds must be ordered"
            )

        if not (
            self.thermal_caution_temperature_c
            < self.maximum_temperature_c
        ):
            raise SpeedLimitValidationError(
                "Thermal thresholds must be ordered"
            )

        if self.maximum_telemetry_age_seconds <= 0:
            raise SpeedLimitValidationError(
                "Telemetry age limit must be positive"
            )

        if (
            isinstance(self.maximum_audit_events, bool)
            or not isinstance(self.maximum_audit_events, int)
            or self.maximum_audit_events <= 0
        ):
            raise SpeedLimitValidationError(
                "Audit capacity must be positive"
            )

        if not self.simulation_only:
            raise SpeedLimitValidationError(
                "Physical hardware control unsupported"
            )


# ============================================================
# ROBOT SPEED POLICY
# ============================================================

@dataclass(frozen=True)
class RobotSpeedPolicy:
    tenant_id: str
    robot_id: str

    maximum_speed_mps: float

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        if nonnegative(
            self.maximum_speed_mps,
            "maximum_speed_mps",
        ) <= 0:
            raise SpeedLimitValidationError(
                "Robot speed limit must be positive"
            )


# ============================================================
# SPEED ZONE
# ============================================================

@dataclass(frozen=True)
class SpeedZone:
    tenant_id: str
    zone_id: str

    maximum_speed_mps: float

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.zone_id, "zone_id")

        nonnegative(
            self.maximum_speed_mps,
            "maximum_speed_mps",
        )


# ============================================================
# ROBOT SPEED SNAPSHOT
# ============================================================

@dataclass(frozen=True)
class RobotSpeedSnapshot:
    tenant_id: str
    robot_id: str

    current_speed_mps: float
    requested_speed_mps: float

    zone_id: str | None

    nearest_human_distance_m: float | None
    collision_risk_score: float

    battery_percent: float
    temperature_c: float

    geofence_clear: bool
    hazards_clear: bool
    emergency_stop_active: bool
    command_interlocks_clear: bool

    sensors_healthy: bool
    communication_healthy: bool

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        if self.zone_id is not None:
            validate_id(self.zone_id, "zone_id")

        for name in (
            "current_speed_mps",
            "requested_speed_mps",
        ):
            nonnegative(
                getattr(self, name),
                name,
            )

        if self.nearest_human_distance_m is not None:
            nonnegative(
                self.nearest_human_distance_m,
                "nearest_human_distance_m",
            )

        collision = finite(
            self.collision_risk_score,
            "collision_risk_score",
        )

        battery = finite(
            self.battery_percent,
            "battery_percent",
        )

        if not 0 <= collision <= 100:
            raise SpeedLimitValidationError(
                "Collision score out of range"
            )

        if not 0 <= battery <= 100:
            raise SpeedLimitValidationError(
                "Battery percent out of range"
            )

        finite(
            self.temperature_c,
            "temperature_c",
        )

        for name in (
            "geofence_clear",
            "hazards_clear",
            "emergency_stop_active",
            "command_interlocks_clear",
            "sensors_healthy",
            "communication_healthy",
        ):
            if not isinstance(
                getattr(self, name),
                bool,
            ):
                raise SpeedLimitValidationError(
                    f"{name} must be boolean"
                )

        validate_time(self.timestamp)


# ============================================================
# SPEED RESTRICTION
# ============================================================

@dataclass(frozen=True)
class SpeedRestriction:
    reason: SpeedRestrictionReason
    maximum_speed_mps: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "maximum_speed_mps": (
                self.maximum_speed_mps
            ),
            "description": self.description,
        }


# ============================================================
# SPEED ASSESSMENT
# ============================================================

@dataclass(frozen=True)
class SpeedLimitAssessment:
    assessment_id: str

    tenant_id: str
    robot_id: str

    current_speed_mps: float
    requested_speed_mps: float

    effective_max_speed_mps: float

    risk_level: SpeedRiskLevel
    decision: SpeedDecision

    overspeed_detected: bool

    restrictions: tuple[
        SpeedRestriction,
        ...
    ]

    evaluated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "current_speed_mps": (
                self.current_speed_mps
            ),
            "requested_speed_mps": (
                self.requested_speed_mps
            ),
            "effective_max_speed_mps": (
                self.effective_max_speed_mps
            ),
            "risk_level": self.risk_level.value,
            "decision": self.decision.value,
            "overspeed_detected": (
                self.overspeed_detected
            ),
            "restrictions": [
                item.to_dict()
                for item in self.restrictions
            ],
            "evaluated_at": (
                self.evaluated_at.isoformat()
            ),
        }


# ============================================================
# SPEED LIMIT ENGINE
# ============================================================

class RoboticsSpeedLimitEngine:

    def __init__(
        self,
        config: SpeedLimitConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        snapshot: RobotSpeedSnapshot,
        *,
        robot_limit: float | None,
        zone_limit: float | None,
    ) -> SpeedLimitAssessment:
        config = self.config

        restrictions: list[
            SpeedRestriction
        ] = []

        def add(
            reason: SpeedRestrictionReason,
            maximum_speed: float,
            description: str,
        ) -> None:
            restrictions.append(
                SpeedRestriction(
                    reason=reason,
                    maximum_speed_mps=maximum_speed,
                    description=description,
                )
            )

        add(
            SpeedRestrictionReason.GLOBAL_LIMIT,
            config.global_max_speed_mps,
            "Global maximum speed",
        )

        if robot_limit is not None:
            add(
                SpeedRestrictionReason.ROBOT_LIMIT,
                robot_limit,
                "Robot-specific maximum speed",
            )

        if zone_limit is not None:
            add(
                SpeedRestrictionReason.ZONE_LIMIT,
                zone_limit,
                "Operating-zone maximum speed",
            )

        age = (
            utc_now()
            - validate_time(
                snapshot.timestamp
            )
        ).total_seconds()

        unreliable = False

        if (
            age < -1.0
            or age
            > config.maximum_telemetry_age_seconds
        ):
            unreliable = True

            add(
                SpeedRestrictionReason.STALE_TELEMETRY,
                0.0,
                "Speed telemetry is stale",
            )

        if not snapshot.sensors_healthy:
            unreliable = True

            add(
                SpeedRestrictionReason.SENSOR_FAILURE,
                0.0,
                "Sensor health check failed",
            )

        if not snapshot.communication_healthy:
            unreliable = True

            add(
                SpeedRestrictionReason.COMMUNICATION_FAILURE,
                0.0,
                "Communication health check failed",
            )

        if not snapshot.geofence_clear:
            add(
                SpeedRestrictionReason.GEOFENCE_BREACH,
                0.0,
                "Geofence restriction detected",
            )

        if not snapshot.hazards_clear:
            add(
                SpeedRestrictionReason.HAZARD_DETECTED,
                0.0,
                "Unresolved hazard detected",
            )

        if snapshot.emergency_stop_active:
            add(
                SpeedRestrictionReason.EMERGENCY_STOP,
                0.0,
                "Emergency stop is active",
            )

        if not snapshot.command_interlocks_clear:
            add(
                SpeedRestrictionReason.COMMAND_INTERLOCK,
                0.0,
                "Command interlock is active",
            )

        human_distance = (
            snapshot.nearest_human_distance_m
        )

        if human_distance is None:
            unreliable = True

            add(
                SpeedRestrictionReason.HUMAN_PROXIMITY,
                0.0,
                "Human proximity evidence unavailable",
            )

        elif human_distance <= (
            config.human_critical_distance_m
        ):
            add(
                SpeedRestrictionReason.HUMAN_PROXIMITY,
                0.0,
                "Human inside critical proximity zone",
            )

        elif human_distance <= (
            config.human_caution_distance_m
        ):
            add(
                SpeedRestrictionReason.HUMAN_PROXIMITY,
                config.human_caution_max_speed_mps,
                "Human near robot operating zone",
            )

        collision = snapshot.collision_risk_score

        if collision >= (
            config.collision_high_score
        ):
            add(
                SpeedRestrictionReason.COLLISION_RISK,
                0.0,
                "High collision risk",
            )

        elif collision >= (
            config.collision_caution_score
        ):
            add(
                SpeedRestrictionReason.COLLISION_RISK,
                config.collision_caution_max_speed_mps,
                "Elevated collision risk",
            )

        if (
            snapshot.battery_percent
            <= config.minimum_battery_percent
        ):
            add(
                SpeedRestrictionReason.LOW_BATTERY,
                config.low_battery_max_speed_mps,
                "Low battery speed restriction",
            )

        if (
            snapshot.temperature_c
            >= config.maximum_temperature_c
        ):
            add(
                SpeedRestrictionReason.OVERHEATING,
                0.0,
                "Critical thermal restriction",
            )

        elif (
            snapshot.temperature_c
            >= config.thermal_caution_temperature_c
        ):
            add(
                SpeedRestrictionReason.OVERHEATING,
                config.thermal_caution_max_speed_mps,
                "Thermal caution speed restriction",
            )

        effective_limit = min(
            restriction.maximum_speed_mps
            for restriction in restrictions
        )

        overspeed = (
            snapshot.current_speed_mps
            > effective_limit
            + config.speed_tolerance_mps
        )

        requested_exceeds = (
            snapshot.requested_speed_mps
            > effective_limit
            + config.speed_tolerance_mps
        )

        if unreliable:
            risk = SpeedRiskLevel.UNKNOWN
            decision = SpeedDecision.DENY

        elif effective_limit <= 0:
            risk = SpeedRiskLevel.CRITICAL
            decision = SpeedDecision.DENY

        elif overspeed:
            risk = SpeedRiskLevel.HIGH
            decision = SpeedDecision.STOP_RECOMMENDED

        elif requested_exceeds:
            risk = SpeedRiskLevel.CAUTION
            decision = SpeedDecision.REDUCE_SPEED

        else:
            risk = SpeedRiskLevel.NORMAL
            decision = SpeedDecision.ALLOW_SIMULATION

        return SpeedLimitAssessment(
            assessment_id=str(uuid.uuid4()),
            tenant_id=snapshot.tenant_id,
            robot_id=snapshot.robot_id,
            current_speed_mps=(
                snapshot.current_speed_mps
            ),
            requested_speed_mps=(
                snapshot.requested_speed_mps
            ),
            effective_max_speed_mps=(
                effective_limit
            ),
            risk_level=risk,
            decision=decision,
            overspeed_detected=overspeed,
            restrictions=tuple(restrictions),
            evaluated_at=utc_now(),
        )


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class SpeedLimitAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str

    decision: SpeedDecision
    effective_max_speed_mps: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "decision": self.decision.value,
            "effective_max_speed_mps": (
                self.effective_max_speed_mps
            ),
        }


# ============================================================
# SPEED LIMIT SERVICE
# ============================================================

class RoboticsSpeedLimitService:

    def __init__(
        self,
        config: SpeedLimitConfig | None = None,
    ) -> None:
        self.config = config or SpeedLimitConfig()

        self.engine = RoboticsSpeedLimitEngine(
            self.config
        )

        self.robot_policies: dict[
            tuple[str, str],
            RobotSpeedPolicy
        ] = {}

        self.zones: dict[
            tuple[str, str],
            SpeedZone
        ] = {}

        self.latest_assessments: dict[
            tuple[str, str],
            SpeedLimitAssessment
        ] = {}

        self.audit_events: list[
            SpeedLimitAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # REGISTER ROBOT POLICY
    # --------------------------------------------------------

    def register_robot_policy(
        self,
        policy: RobotSpeedPolicy,
    ) -> None:
        with self._lock:
            self.robot_policies[
                (
                    policy.tenant_id,
                    policy.robot_id,
                )
            ] = policy

    # --------------------------------------------------------
    # REGISTER ZONE
    # --------------------------------------------------------

    def register_zone(
        self,
        zone: SpeedZone,
    ) -> None:
        with self._lock:
            self.zones[
                (
                    zone.tenant_id,
                    zone.zone_id,
                )
            ] = zone

    # --------------------------------------------------------
    # ASSESS
    # --------------------------------------------------------

    def assess(
        self,
        snapshot: RobotSpeedSnapshot,
    ) -> SpeedLimitAssessment:
        key = (
            snapshot.tenant_id,
            snapshot.robot_id,
        )

        with self._lock:
            policy = self.robot_policies.get(
                key
            )

            robot_limit = (
                policy.maximum_speed_mps
                if policy
                else None
            )

            zone_limit = None

            if snapshot.zone_id is not None:
                zone = self.zones.get(
                    (
                        snapshot.tenant_id,
                        snapshot.zone_id,
                    )
                )

                # Fail closed for an unknown zone.
                zone_limit = (
                    zone.maximum_speed_mps
                    if zone
                    else 0.0
                )

            assessment = self.engine.evaluate(
                snapshot,
                robot_limit=robot_limit,
                zone_limit=zone_limit,
            )

            self.latest_assessments[
                key
            ] = assessment

            self.audit_events.append(
                SpeedLimitAuditEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=utc_now(),
                    tenant_id=snapshot.tenant_id,
                    robot_id=snapshot.robot_id,
                    decision=assessment.decision,
                    effective_max_speed_mps=(
                        assessment.effective_max_speed_mps
                    ),
                )
            )

            if (
                len(self.audit_events)
                > self.config.maximum_audit_events
            ):
                self.audit_events.pop(0)

            return assessment

    # --------------------------------------------------------
    # GET LATEST
    # --------------------------------------------------------

    def get_latest(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> SpeedLimitAssessment:
        with self._lock:
            assessment = (
                self.latest_assessments.get(
                    (tenant_id, robot_id)
                )
            )

            if assessment is None:
                raise SpeedLimitNotFound(
                    "Speed assessment not found"
                )

            return assessment

    # --------------------------------------------------------
    # COMMAND VALIDATION
    # --------------------------------------------------------

    def validate_speed_command(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        requested_speed_mps: float,
    ) -> dict[str, Any]:
        requested = nonnegative(
            requested_speed_mps,
            "requested_speed_mps",
        )

        with self._lock:
            assessment = self.get_latest(
                tenant_id=tenant_id,
                robot_id=robot_id,
            )

            # Cached assessments are not authorization.
            # The caller must provide fresh telemetry
            # and perform a new safety assessment.
            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "decision": "deny",
                "reason": (
                    "Fresh speed and safety assessment "
                    "required before command validation"
                ),
                "requested_speed_mps": requested,
                "last_effective_limit_mps": (
                    assessment.effective_max_speed_mps
                ),
            }

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            assessments = [
                assessment
                for (record_tenant, _), assessment
                in self.latest_assessments.items()
                if (
                    tenant_id is None
                    or record_tenant == tenant_id
                )
            ]

            return {
                "monitored_robots": len(assessments),
                "overspeed_robots": sum(
                    item.overspeed_detected
                    for item in assessments
                ),
                "denied": sum(
                    item.decision == SpeedDecision.DENY
                    for item in assessments
                ),
                "speed_reduction_required": sum(
                    item.decision
                    == SpeedDecision.REDUCE_SPEED
                    for item in assessments
                ),
            }

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
    ) -> Path:
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
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "assessments": [
                    assessment.to_dict()
                    for (record_tenant, _), assessment
                    in self.latest_assessments.items()
                    if record_tenant == tenant_id
                ],
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
    service = RoboticsSpeedLimitService()

    service.register_robot_policy(
        RobotSpeedPolicy(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            maximum_speed_mps=1.5,
        )
    )

    service.register_zone(
        SpeedZone(
            tenant_id="tenant_alpha",
            zone_id="warehouse_zone_a",
            maximum_speed_mps=1.0,
        )
    )

    snapshot = RobotSpeedSnapshot(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        current_speed_mps=1.2,
        requested_speed_mps=1.4,
        zone_id="warehouse_zone_a",
        nearest_human_distance_m=2.5,
        collision_risk_score=20.0,
        battery_percent=80.0,
        temperature_c=40.0,
        geofence_clear=True,
        hazards_clear=True,
        emergency_stop_active=False,
        command_interlocks_clear=True,
        sensors_healthy=True,
        communication_healthy=True,
    )

    assessment = service.assess(
        snapshot
    )

    print(
        json.dumps(
            assessment.to_dict(),
            indent=2,
        )
    )

    service.export_json(
        "data/security/speed_limits_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

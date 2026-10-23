"""Human Proximity module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/security/human_proximity.py

Python 3.11+
Standard library only.

Features:
    - Human-to-robot proximity monitoring
    - Multiple human tracking
    - Relative motion prediction
    - Time to closest approach
    - Personal safety zones
    - Configurable separation thresholds
    - Localization uncertainty margins
    - Stale observation detection
    - Risk classification
    - Human proximity advisories
    - Tenant-scoped assessments
    - Audit logging
    - JSON report export

SAFETY:
    Simulation and supervisory monitoring only.

    This is NOT a safety-rated protective
    separation monitoring system.

    No physical robot commands are executed.
    Independent protective-stop and emergency-stop
    systems are required for real robots.
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

class HumanProximityError(Exception):
    pass


class HumanProximityValidationError(
    HumanProximityError
):
    pass


# ============================================================
# ENUMS
# ============================================================

class ProximityRiskLevel(str, Enum):
    CLEAR = "clear"
    CAUTION = "caution"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ProximityDecision(str, Enum):
    ALLOW_SIMULATION = "allow_simulation"
    SLOW_DOWN = "slow_down"
    STOP_RECOMMENDED = "stop_recommended"
    DENY = "deny"


class HumanTrackingStatus(str, Enum):
    TRACKED = "tracked"
    OCCLUDED = "occluded"
    LOST = "lost"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def finite(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise HumanProximityValidationError(
            f"{name} must be finite"
        )

    return float(value)


def nonnegative(
    value: float,
    name: str,
) -> float:
    number = finite(value, name)

    if number < 0:
        raise HumanProximityValidationError(
            f"{name} must be nonnegative"
        )

    return number


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
        raise HumanProximityValidationError(
            f"Invalid {name}"
        )

    return value


def validate_timestamp(
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise HumanProximityValidationError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# VECTOR
# ============================================================

@dataclass(frozen=True)
class Vector2D:
    x: float
    y: float

    def __post_init__(self) -> None:
        finite(self.x, "x")
        finite(self.y, "y")

    def __add__(
        self,
        other: "Vector2D",
    ) -> "Vector2D":
        return Vector2D(
            self.x + other.x,
            self.y + other.y,
        )

    def __sub__(
        self,
        other: "Vector2D",
    ) -> "Vector2D":
        return Vector2D(
            self.x - other.x,
            self.y - other.y,
        )

    def scale(
        self,
        factor: float,
    ) -> "Vector2D":
        return Vector2D(
            self.x * factor,
            self.y * factor,
        )

    def dot(
        self,
        other: "Vector2D",
    ) -> float:
        return (
            self.x * other.x
            + self.y * other.y
        )

    def norm(self) -> float:
        return math.hypot(
            self.x,
            self.y,
        )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class HumanProximityConfig:
    caution_distance_meters: float = 3.0
    high_risk_distance_meters: float = 1.5
    critical_distance_meters: float = 0.75

    prediction_horizon_seconds: float = 5.0

    robot_safety_margin_meters: float = 0.3
    human_safety_margin_meters: float = 0.3

    minimum_tracking_confidence: float = 0.6
    maximum_observation_age_seconds: float = 2.0

    maximum_humans: int = 500
    maximum_audit_events: int = 20000

    require_perception_healthy: bool = True
    require_coverage_confirmed: bool = True

    def __post_init__(self) -> None:
        distances = (
            self.critical_distance_meters,
            self.high_risk_distance_meters,
            self.caution_distance_meters,
        )

        if any(
            finite(value, "distance") <= 0
            for value in distances
        ):
            raise HumanProximityValidationError(
                "Distances must be positive"
            )

        if not (
            distances[0]
            < distances[1]
            < distances[2]
        ):
            raise HumanProximityValidationError(
                "Proximity thresholds must be ordered"
            )

        for name in (
            "prediction_horizon_seconds",
            "maximum_observation_age_seconds",
        ):
            if finite(getattr(self, name), name) <= 0:
                raise HumanProximityValidationError(
                    f"{name} must be positive"
                )

        for name in (
            "robot_safety_margin_meters",
            "human_safety_margin_meters",
        ):
            nonnegative(getattr(self, name), name)

        confidence = finite(
            self.minimum_tracking_confidence,
            "minimum_tracking_confidence",
        )

        if not (0 <= confidence <= 1):
            raise HumanProximityValidationError(
                "Confidence must be in [0, 1]"
            )

        for name in (
            "maximum_humans",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise HumanProximityValidationError(
                    f"{name} must be positive"
                )


# ============================================================
# ROBOT STATE
# ============================================================

@dataclass(frozen=True)
class RobotProximityState:
    tenant_id: str
    robot_id: str

    position: Vector2D
    velocity: Vector2D

    radius_meters: float

    localization_uncertainty_meters: float = 0.0

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        if nonnegative(
            self.radius_meters,
            "radius_meters",
        ) <= 0:
            raise HumanProximityValidationError(
                "Robot radius must be positive"
            )

        nonnegative(
            self.localization_uncertainty_meters,
            "localization_uncertainty_meters",
        )

        validate_timestamp(self.timestamp)


# ============================================================
# HUMAN STATE
# ============================================================

@dataclass(frozen=True)
class HumanProximityState:
    human_id: str

    position: Vector2D
    velocity: Vector2D

    radius_meters: float = 0.3

    tracking_confidence: float = 1.0

    localization_uncertainty_meters: float = 0.0

    tracking_status: HumanTrackingStatus = (
        HumanTrackingStatus.TRACKED
    )

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.human_id, "human_id")

        nonnegative(
            self.radius_meters,
            "radius_meters",
        )

        nonnegative(
            self.localization_uncertainty_meters,
            "localization_uncertainty_meters",
        )

        confidence = finite(
            self.tracking_confidence,
            "tracking_confidence",
        )

        if not (0 <= confidence <= 1):
            raise HumanProximityValidationError(
                "Tracking confidence must be in [0, 1]"
            )

        validate_timestamp(self.timestamp)


# ============================================================
# PERCEPTION STATUS
# ============================================================

@dataclass(frozen=True)
class ProximityPerceptionStatus:
    healthy: bool
    coverage_confirmed: bool

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_timestamp(self.timestamp)


# ============================================================
# INDIVIDUAL ASSESSMENT
# ============================================================

@dataclass(frozen=True)
class HumanProximityAssessment:
    human_id: str

    current_center_distance_meters: float
    current_clearance_meters: float

    closest_approach_seconds: float
    predicted_minimum_clearance_meters: float

    risk_level: ProximityRiskLevel
    risk_score: float

    tracking_confidence: float

    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "human_id": self.human_id,
            "current_center_distance_meters": (
                self.current_center_distance_meters
            ),
            "current_clearance_meters": (
                self.current_clearance_meters
            ),
            "closest_approach_seconds": (
                self.closest_approach_seconds
            ),
            "predicted_minimum_clearance_meters": (
                self.predicted_minimum_clearance_meters
            ),
            "risk_level": self.risk_level.value,
            "risk_score": self.risk_score,
            "tracking_confidence": (
                self.tracking_confidence
            ),
            "reason": self.reason,
        }


# ============================================================
# REPORT
# ============================================================

@dataclass(frozen=True)
class HumanProximityReport:
    report_id: str

    tenant_id: str
    robot_id: str

    decision: ProximityDecision
    overall_risk: ProximityRiskLevel
    overall_risk_score: float

    assessments: tuple[
        HumanProximityAssessment,
        ...
    ]

    reason: str

    evaluated_at: str = field(
        default_factory=utc_iso
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "decision": self.decision.value,
            "overall_risk": self.overall_risk.value,
            "overall_risk_score": (
                self.overall_risk_score
            ),
            "assessments": [
                item.to_dict()
                for item in self.assessments
            ],
            "reason": self.reason,
            "evaluated_at": self.evaluated_at,
        }


# ============================================================
# GEOMETRY
# ============================================================

def predict_closest_approach(
    relative_position: Vector2D,
    relative_velocity: Vector2D,
    horizon_seconds: float,
) -> tuple[float, float]:
    speed_squared = relative_velocity.dot(
        relative_velocity
    )

    if speed_squared <= 1e-12:
        time_seconds = 0.0
    else:
        time_seconds = (
            -relative_position.dot(
                relative_velocity
            )
            / speed_squared
        )

    time_seconds = max(
        0.0,
        min(horizon_seconds, time_seconds),
    )

    predicted = (
        relative_position
        + relative_velocity.scale(
            time_seconds
        )
    )

    return time_seconds, predicted.norm()


# ============================================================
# PROXIMITY ENGINE
# ============================================================

class HumanProximityEngine:

    def __init__(
        self,
        config: HumanProximityConfig,
    ) -> None:
        self.config = config

    def assess_human(
        self,
        robot: RobotProximityState,
        human: HumanProximityState,
    ) -> HumanProximityAssessment:
        relative_position = (
            human.position - robot.position
        )

        relative_velocity = (
            human.velocity - robot.velocity
        )

        center_distance = relative_position.norm()

        physical_radius = (
            robot.radius_meters
            + human.radius_meters
        )

        uncertainty_margin = (
            robot.localization_uncertainty_meters
            + human.localization_uncertainty_meters
            + self.config.robot_safety_margin_meters
            + self.config.human_safety_margin_meters
        )

        clearance = (
            center_distance - physical_radius
        )

        closest_time, closest_distance = (
            predict_closest_approach(
                relative_position,
                relative_velocity,
                self.config.prediction_horizon_seconds,
            )
        )

        predicted_clearance = (
            closest_distance
            - physical_radius
            - uncertainty_margin
        )

        if (
            predicted_clearance
            <= self.config.critical_distance_meters
        ):
            risk = ProximityRiskLevel.CRITICAL
            score = 100.0

        elif (
            predicted_clearance
            <= self.config.high_risk_distance_meters
        ):
            risk = ProximityRiskLevel.HIGH
            score = 85.0

        elif (
            predicted_clearance
            <= self.config.caution_distance_meters
        ):
            risk = ProximityRiskLevel.CAUTION
            score = 55.0

        else:
            risk = ProximityRiskLevel.CLEAR
            score = 0.0

        return HumanProximityAssessment(
            human_id=human.human_id,
            current_center_distance_meters=(
                center_distance
            ),
            current_clearance_meters=clearance,
            closest_approach_seconds=closest_time,
            predicted_minimum_clearance_meters=(
                predicted_clearance
            ),
            risk_level=risk,
            risk_score=score,
            tracking_confidence=(
                human.tracking_confidence
            ),
            reason=(
                "Constant-velocity proximity "
                "prediction completed"
            ),
        )


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class HumanProximityAuditEvent:
    event_id: str
    timestamp: str

    tenant_id: str
    robot_id: str

    risk_level: ProximityRiskLevel
    decision: ProximityDecision

    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "risk_level": self.risk_level.value,
            "decision": self.decision.value,
            "reason": self.reason,
        }


# ============================================================
# CENTRAL HUMAN PROXIMITY SERVICE
# ============================================================

class RoboticsHumanProximityService:

    def __init__(
        self,
        config: HumanProximityConfig | None = None,
    ) -> None:
        self.config = (
            config or HumanProximityConfig()
        )

        self.engine = HumanProximityEngine(
            self.config
        )

        self.audit_events: list[
            HumanProximityAuditEvent
        ] = []

        self._lock = threading.RLock()

    def _fresh(
        self,
        timestamp: datetime,
    ) -> bool:
        age = (
            utc_now()
            - validate_timestamp(timestamp)
        ).total_seconds()

        return (
            -1.0 <= age
            <= self.config.maximum_observation_age_seconds
        )

    def _report(
        self,
        robot: RobotProximityState,
        decision: ProximityDecision,
        risk: ProximityRiskLevel,
        score: float,
        assessments: tuple[
            HumanProximityAssessment,
            ...
        ],
        reason: str,
    ) -> HumanProximityReport:
        report = HumanProximityReport(
            report_id=str(uuid.uuid4()),
            tenant_id=robot.tenant_id,
            robot_id=robot.robot_id,
            decision=decision,
            overall_risk=risk,
            overall_risk_score=score,
            assessments=assessments,
            reason=reason,
        )

        with self._lock:
            self.audit_events.append(
                HumanProximityAuditEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=utc_iso(),
                    tenant_id=robot.tenant_id,
                    robot_id=robot.robot_id,
                    risk_level=risk,
                    decision=decision,
                    reason=reason,
                )
            )

            if (
                len(self.audit_events)
                > self.config.maximum_audit_events
            ):
                self.audit_events.pop(0)

        return report

    # --------------------------------------------------------
    # ASSESS
    # --------------------------------------------------------

    def assess(
        self,
        *,
        robot: RobotProximityState,
        humans: Iterable[HumanProximityState],
        perception: ProximityPerceptionStatus,
    ) -> HumanProximityReport:
        human_list = list(humans)

        if (
            len(human_list)
            > self.config.maximum_humans
        ):
            raise HumanProximityValidationError(
                "Human observation capacity exceeded"
            )

        if (
            not self._fresh(robot.timestamp)
            or not self._fresh(perception.timestamp)
        ):
            return self._report(
                robot,
                ProximityDecision.DENY,
                ProximityRiskLevel.UNKNOWN,
                100.0,
                (),
                "Robot or perception state is stale",
            )

        if (
            self.config.require_perception_healthy
            and not perception.healthy
        ):
            return self._report(
                robot,
                ProximityDecision.DENY,
                ProximityRiskLevel.UNKNOWN,
                100.0,
                (),
                "Human perception system is unhealthy",
            )

        if (
            self.config.require_coverage_confirmed
            and not perception.coverage_confirmed
        ):
            return self._report(
                robot,
                ProximityDecision.DENY,
                ProximityRiskLevel.UNKNOWN,
                100.0,
                (),
                "Human detection coverage is unconfirmed",
            )

        for human in human_list:
            if (
                not self._fresh(human.timestamp)
                or human.tracking_status
                != HumanTrackingStatus.TRACKED
                or human.tracking_confidence
                < self.config.minimum_tracking_confidence
            ):
                return self._report(
                    robot,
                    ProximityDecision.DENY,
                    ProximityRiskLevel.UNKNOWN,
                    100.0,
                    (),
                    "Human tracking is stale or uncertain",
                )

        assessments = tuple(
            self.engine.assess_human(
                robot,
                human,
            )
            for human in human_list
        )

        if assessments:
            worst = max(
                assessments,
                key=lambda item: item.risk_score,
            )

            risk = worst.risk_level
            score = worst.risk_score

        else:
            risk = ProximityRiskLevel.CLEAR
            score = 0.0

        decisions = {
            ProximityRiskLevel.CLEAR:
                ProximityDecision.ALLOW_SIMULATION,

            ProximityRiskLevel.CAUTION:
                ProximityDecision.SLOW_DOWN,

            ProximityRiskLevel.HIGH:
                ProximityDecision.STOP_RECOMMENDED,

            ProximityRiskLevel.CRITICAL:
                ProximityDecision.DENY,

            ProximityRiskLevel.UNKNOWN:
                ProximityDecision.DENY,
        }

        return self._report(
            robot,
            decisions[risk],
            risk,
            score,
            assessments,
            "Human proximity assessment completed",
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            events = [
                event
                for event in self.audit_events
                if (
                    tenant_id is None
                    or event.tenant_id == tenant_id
                )
            ]

            return {
                "total_assessments": len(events),
                "critical_assessments": sum(
                    event.risk_level
                    == ProximityRiskLevel.CRITICAL
                    for event in events
                ),
                "high_assessments": sum(
                    event.risk_level
                    == ProximityRiskLevel.HIGH
                    for event in events
                ),
                "denied_assessments": sum(
                    event.decision
                    == ProximityDecision.DENY
                    for event in events
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
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_iso(),
                "tenant_id": tenant_id,
                "statistics": self.statistics(
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
    service = RoboticsHumanProximityService()

    robot = RobotProximityState(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        position=Vector2D(0.0, 0.0),
        velocity=Vector2D(1.0, 0.0),
        radius_meters=0.45,
        localization_uncertainty_meters=0.1,
    )

    humans = [
        HumanProximityState(
            human_id="human_001",
            position=Vector2D(4.0, 0.0),
            velocity=Vector2D(0.0, 0.0),
            tracking_confidence=0.98,
        ),
        HumanProximityState(
            human_id="human_002",
            position=Vector2D(8.0, 2.0),
            velocity=Vector2D(-0.2, 0.0),
            tracking_confidence=0.95,
        ),
    ]

    perception = ProximityPerceptionStatus(
        healthy=True,
        coverage_confirmed=True,
    )

    report = service.assess(
        robot=robot,
        humans=humans,
        perception=perception,
    )

    print(
        json.dumps(
            report.to_dict(),
            indent=2,
        )
    )

    service.export_json(
        "data/security/human_proximity_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

"""Collision Risk module.

TODO: implement and test; no real robot control is enabled.
"""
"""Command Interlock module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/security/collision_risk.py

Python 3.11+
Standard library only.

Features:
    - 2D robot and obstacle state validation
    - Relative velocity and closest approach
    - Time-to-collision estimation
    - Collision risk scoring
    - Robot footprint and safety margins
    - Static and moving obstacle assessment
    - Trajectory sampling
    - Risk aggregation
    - Movement authorization recommendations
    - Audit logging and JSON export

IMPORTANT:
    This is a simulation/supervisory risk estimator.
    It is NOT a certified collision avoidance system,
    safety-rated controller, or emergency-stop device.

    Never use its ALLOW result as the sole permission
    to move a physical robot.
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

class CollisionRiskError(Exception):
    pass


class CollisionValidationError(CollisionRiskError):
    pass


# ============================================================
# ENUMS
# ============================================================

class CollisionRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class CollisionDecision(str, Enum):
    ALLOW = "allow"
    SLOW_DOWN = "slow_down"
    STOP_RECOMMENDED = "stop_recommended"
    DENY = "deny"


class ObstacleType(str, Enum):
    STATIC = "static"
    MOVING = "moving"
    HUMAN = "human"
    ROBOT = "robot"
    VEHICLE = "vehicle"
    UNKNOWN = "unknown"


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
        raise CollisionValidationError(
            f"{name} must be a finite number"
        )

    return float(value)


def nonnegative(
    value: float,
    name: str,
) -> float:
    value = finite(value, name)

    if value < 0:
        raise CollisionValidationError(
            f"{name} must be nonnegative"
        )

    return value


def validate_id(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            c.isalnum() or c in "_-"
            for c in value
        )
    ):
        raise CollisionValidationError(
            f"Invalid {name}"
        )

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CollisionRiskConfig:
    prediction_horizon_seconds: float = 10.0

    safety_margin_meters: float = 0.5

    minimum_separation_meters: float = 0.25

    moderate_ttc_seconds: float = 8.0
    high_ttc_seconds: float = 4.0
    critical_ttc_seconds: float = 1.5

    maximum_state_age_seconds: float = 2.0

    maximum_obstacles: int = 1000
    maximum_audit_events: int = 20000

    def __post_init__(self) -> None:
        for name in (
            "prediction_horizon_seconds",
            "moderate_ttc_seconds",
            "high_ttc_seconds",
            "critical_ttc_seconds",
            "maximum_state_age_seconds",
        ):
            if finite(getattr(self, name), name) <= 0:
                raise CollisionValidationError(
                    f"{name} must be positive"
                )

        for name in (
            "safety_margin_meters",
            "minimum_separation_meters",
        ):
            nonnegative(getattr(self, name), name)

        if not (
            self.critical_ttc_seconds
            < self.high_ttc_seconds
            < self.moderate_ttc_seconds
        ):
            raise CollisionValidationError(
                "TTC thresholds must be ordered"
            )

        for name in (
            "maximum_obstacles",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise CollisionValidationError(
                    f"{name} must be positive"
                )


# ============================================================
# GEOMETRY
# ============================================================

@dataclass(frozen=True)
class Vector2D:
    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "x", finite(self.x, "x")
        )
        object.__setattr__(
            self, "y", finite(self.y, "y")
        )

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

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
        }


# ============================================================
# ROBOT STATE
# ============================================================

@dataclass(frozen=True)
class RobotMotionState:
    robot_id: str
    tenant_id: str

    position: Vector2D
    velocity: Vector2D

    radius_meters: float

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(
            self.robot_id,
            "robot_id",
        )
        validate_id(
            self.tenant_id,
            "tenant_id",
        )
        if nonnegative(
            self.radius_meters,
            "radius_meters",
        ) <= 0:
            raise CollisionValidationError(
                "Robot radius must be positive"
            )

        validate_timestamp(self.timestamp)


# ============================================================
# OBSTACLE STATE
# ============================================================

@dataclass(frozen=True)
class ObstacleMotionState:
    obstacle_id: str

    position: Vector2D
    velocity: Vector2D

    radius_meters: float

    obstacle_type: ObstacleType = (
        ObstacleType.UNKNOWN
    )

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(
            self.obstacle_id,
            "obstacle_id",
        )

        nonnegative(
            self.radius_meters,
            "radius_meters",
        )

        validate_timestamp(self.timestamp)


def validate_timestamp(
    timestamp: datetime,
) -> None:
    if (
        not isinstance(timestamp, datetime)
        or timestamp.tzinfo is None
        or timestamp.utcoffset() is None
    ):
        raise CollisionValidationError(
            "Timestamp must be timezone-aware"
        )


# ============================================================
# COLLISION RESULT
# ============================================================

@dataclass(frozen=True)
class ObstacleCollisionAssessment:
    obstacle_id: str
    obstacle_type: ObstacleType

    current_distance_meters: float
    current_clearance_meters: float

    closest_approach_seconds: float
    minimum_clearance_meters: float

    time_to_collision_seconds: float | None

    risk_score: float
    risk_level: CollisionRiskLevel

    def to_dict(self) -> dict[str, Any]:
        return {
            "obstacle_id": self.obstacle_id,
            "obstacle_type": self.obstacle_type.value,
            "current_distance_meters": (
                self.current_distance_meters
            ),
            "current_clearance_meters": (
                self.current_clearance_meters
            ),
            "closest_approach_seconds": (
                self.closest_approach_seconds
            ),
            "minimum_clearance_meters": (
                self.minimum_clearance_meters
            ),
            "time_to_collision_seconds": (
                self.time_to_collision_seconds
            ),
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
        }


@dataclass(frozen=True)
class CollisionRiskReport:
    report_id: str

    tenant_id: str
    robot_id: str

    decision: CollisionDecision
    overall_risk_level: CollisionRiskLevel
    overall_risk_score: float

    assessments: tuple[
        ObstacleCollisionAssessment,
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
            "overall_risk_level": (
                self.overall_risk_level.value
            ),
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
# COLLISION GEOMETRY
# ============================================================

def closest_approach(
    relative_position: Vector2D,
    relative_velocity: Vector2D,
    horizon_seconds: float,
) -> tuple[float, float]:
    """
    Returns:
        time of closest approach
        center-to-center distance at that time
    """
    velocity_squared = (
        relative_velocity.dot(
            relative_velocity
        )
    )

    if velocity_squared <= 1e-12:
        time_seconds = 0.0
    else:
        time_seconds = (
            -relative_position.dot(
                relative_velocity
            )
            / velocity_squared
        )

    time_seconds = max(
        0.0,
        min(
            horizon_seconds,
            time_seconds,
        ),
    )

    closest_position = (
        relative_position
        + relative_velocity.scale(
            time_seconds
        )
    )

    return (
        time_seconds,
        closest_position.norm(),
    )


def calculate_time_to_collision(
    relative_position: Vector2D,
    relative_velocity: Vector2D,
    collision_radius: float,
    horizon_seconds: float,
) -> float | None:
    """
    Constant-velocity circle-circle collision model.

    Solve:
        |relative_position + relative_velocity * t|^2
        = collision_radius^2
    """
    a = relative_velocity.dot(
        relative_velocity
    )

    b = (
        2.0
        * relative_position.dot(
            relative_velocity
        )
    )

    c = (
        relative_position.dot(
            relative_position
        )
        - collision_radius**2
    )

    if c <= 0:
        return 0.0

    if a <= 1e-12:
        return None

    discriminant = (
        b * b - 4.0 * a * c
    )

    if discriminant < 0:
        return None

    root = math.sqrt(
        discriminant
    )

    first = (
        -b - root
    ) / (2.0 * a)

    second = (
        -b + root
    ) / (2.0 * a)

    candidates = [
        value
        for value in (first, second)
        if (
            value >= 0
            and value <= horizon_seconds
        )
    ]

    return (
        min(candidates)
        if candidates
        else None
    )


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(
    *,
    time_to_collision: float | None,
    minimum_clearance: float,
    config: CollisionRiskConfig,
) -> tuple[CollisionRiskLevel, float]:
    if minimum_clearance <= 0:
        return (
            CollisionRiskLevel.CRITICAL,
            100.0,
        )

    if time_to_collision is not None:
        if (
            time_to_collision
            <= config.critical_ttc_seconds
        ):
            return (
                CollisionRiskLevel.CRITICAL,
                100.0,
            )

        if (
            time_to_collision
            <= config.high_ttc_seconds
        ):
            return (
                CollisionRiskLevel.HIGH,
                85.0,
            )

        if (
            time_to_collision
            <= config.moderate_ttc_seconds
        ):
            return (
                CollisionRiskLevel.MODERATE,
                60.0,
            )

        return (
            CollisionRiskLevel.MODERATE,
            40.0,
        )

    if (
        minimum_clearance
        <= config.minimum_separation_meters
    ):
        return (
            CollisionRiskLevel.HIGH,
            80.0,
        )

    if minimum_clearance <= 1.0:
        return (
            CollisionRiskLevel.MODERATE,
            50.0,
        )

    return (
        CollisionRiskLevel.LOW,
        10.0,
    )


# ============================================================
# COLLISION RISK ENGINE
# ============================================================

class CollisionRiskEngine:

    def __init__(
        self,
        config: CollisionRiskConfig,
    ) -> None:
        self.config = config

    def assess_obstacle(
        self,
        robot: RobotMotionState,
        obstacle: ObstacleMotionState,
    ) -> ObstacleCollisionAssessment:
        relative_position = (
            obstacle.position
            - robot.position
        )

        relative_velocity = (
            obstacle.velocity
            - robot.velocity
        )

        center_distance = (
            relative_position.norm()
        )

        physical_radius = (
            robot.radius_meters
            + obstacle.radius_meters
        )

        collision_radius = (
            physical_radius
            + self.config.safety_margin_meters
        )

        current_clearance = (
            center_distance
            - physical_radius
        )

        closest_time, closest_distance = (
            closest_approach(
                relative_position,
                relative_velocity,
                self.config.prediction_horizon_seconds,
            )
        )

        minimum_clearance = (
            closest_distance
            - collision_radius
        )

        ttc = calculate_time_to_collision(
            relative_position,
            relative_velocity,
            collision_radius,
            self.config.prediction_horizon_seconds,
        )

        level, score = classify_risk(
            time_to_collision=ttc,
            minimum_clearance=minimum_clearance,
            config=self.config,
        )

        return ObstacleCollisionAssessment(
            obstacle_id=obstacle.obstacle_id,
            obstacle_type=obstacle.obstacle_type,
            current_distance_meters=(
                center_distance
            ),
            current_clearance_meters=(
                current_clearance
            ),
            closest_approach_seconds=(
                closest_time
            ),
            minimum_clearance_meters=(
                minimum_clearance
            ),
            time_to_collision_seconds=ttc,
            risk_score=score,
            risk_level=level,
        )


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class CollisionAuditEvent:
    event_id: str
    timestamp: str

    tenant_id: str
    robot_id: str

    risk_level: CollisionRiskLevel
    decision: CollisionDecision

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
# CENTRAL COLLISION RISK SERVICE
# ============================================================

class RoboticsCollisionRiskService:

    def __init__(
        self,
        config: CollisionRiskConfig | None = None,
    ) -> None:
        self.config = (
            config or CollisionRiskConfig()
        )

        self.engine = CollisionRiskEngine(
            self.config
        )

        self.audit_events: list[
            CollisionAuditEvent
        ] = []

        self._lock = threading.RLock()

    def _audit(
        self,
        report: CollisionRiskReport,
    ) -> None:
        with self._lock:
            self.audit_events.append(
                CollisionAuditEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=utc_iso(),
                    tenant_id=report.tenant_id,
                    robot_id=report.robot_id,
                    risk_level=report.overall_risk_level,
                    decision=report.decision,
                    reason=report.reason,
                )
            )

            if (
                len(self.audit_events)
                > self.config.maximum_audit_events
            ):
                self.audit_events.pop(0)

    def assess(
        self,
        *,
        robot: RobotMotionState,
        obstacles: Iterable[
            ObstacleMotionState
        ],
    ) -> CollisionRiskReport:
        obstacle_list = list(
            obstacles
        )

        if (
            len(obstacle_list)
            > self.config.maximum_obstacles
        ):
            raise CollisionValidationError(
                "Too many obstacles"
            )

        now = utc_now()

        states = [
            robot.timestamp,
            *(
                obstacle.timestamp
                for obstacle in obstacle_list
            ),
        ]

        for timestamp in states:
            age = (
                now - timestamp.astimezone(
                    timezone.utc
                )
            ).total_seconds()

            if (
                age < -1.0
                or age
                > self.config.maximum_state_age_seconds
            ):
                return self._blocked_report(
                    robot,
                    "Motion state is stale or in the future",
                )

        assessments = tuple(
            self.engine.assess_obstacle(
                robot,
                obstacle,
            )
            for obstacle in obstacle_list
        )

        if not assessments:
            risk_level = CollisionRiskLevel.LOW
            risk_score = 0.0
        else:
            worst = max(
                assessments,
                key=lambda item: item.risk_score,
            )

            risk_level = worst.risk_level
            risk_score = worst.risk_score

        decision_map = {
            CollisionRiskLevel.LOW:
                CollisionDecision.ALLOW,

            CollisionRiskLevel.MODERATE:
                CollisionDecision.SLOW_DOWN,

            CollisionRiskLevel.HIGH:
                CollisionDecision.STOP_RECOMMENDED,

            CollisionRiskLevel.CRITICAL:
                CollisionDecision.DENY,
        }

        decision = decision_map[
            risk_level
        ]

        report = CollisionRiskReport(
            report_id=str(uuid.uuid4()),
            tenant_id=robot.tenant_id,
            robot_id=robot.robot_id,
            decision=decision,
            overall_risk_level=risk_level,
            overall_risk_score=risk_score,
            assessments=assessments,
            reason=(
                "Constant-velocity collision risk "
                "assessment completed"
            ),
        )

        self._audit(report)

        return report

    def _blocked_report(
        self,
        robot: RobotMotionState,
        reason: str,
    ) -> CollisionRiskReport:
        report = CollisionRiskReport(
            report_id=str(uuid.uuid4()),
            tenant_id=robot.tenant_id,
            robot_id=robot.robot_id,
            decision=CollisionDecision.DENY,
            overall_risk_level=(
                CollisionRiskLevel.CRITICAL
            ),
            overall_risk_score=100.0,
            assessments=(),
            reason=reason,
        )

        self._audit(report)

        return report

    def statistics(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            return {
                "audit_events": len(
                    self.audit_events
                ),
                "critical_events": sum(
                    event.risk_level
                    == CollisionRiskLevel.CRITICAL
                    for event in self.audit_events
                ),
                "high_events": sum(
                    event.risk_level
                    == CollisionRiskLevel.HIGH
                    for event in self.audit_events
                ),
                "denied_events": sum(
                    event.decision
                    == CollisionDecision.DENY
                    for event in self.audit_events
                ),
            }

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
    service = RoboticsCollisionRiskService()

    robot = RobotMotionState(
        robot_id="robot_001",
        tenant_id="tenant_alpha",
        position=Vector2D(0.0, 0.0),
        velocity=Vector2D(1.0, 0.0),
        radius_meters=0.4,
    )

    obstacles = [
        ObstacleMotionState(
            obstacle_id="obstacle_001",
            position=Vector2D(5.0, 0.0),
            velocity=Vector2D(0.0, 0.0),
            radius_meters=0.5,
            obstacle_type=ObstacleType.STATIC,
        ),
        ObstacleMotionState(
            obstacle_id="worker_001",
            position=Vector2D(3.0, 1.0),
            velocity=Vector2D(0.0, 0.0),
            radius_meters=0.3,
            obstacle_type=ObstacleType.HUMAN,
        ),
    ]

    report = service.assess(
        robot=robot,
        obstacles=obstacles,
    )

    print(
        json.dumps(
            report.to_dict(),
            indent=2,
        )
    )

    service.export_json(
        "data/security/collision_risk_report.json"
    )


if __name__ == "__main__":
    main()

"""Workspace Limits module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/security/workspace_limits.py

Python 3.11+
Standard library only.

Features:
    - 3D rectangular robot workspace boundaries
    - Tenant-specific robot workspaces
    - Restricted-zone detection
    - Safety-margin monitoring
    - Altitude / height restrictions
    - Workspace speed restrictions
    - Current and target position validation
    - Telemetry freshness checks
    - Out-of-order telemetry rejection
    - Fail-closed simulation decisions
    - Workspace violation events
    - Thread-safe operations
    - Audit logging
    - JSON report export

SAFETY:
    Simulation and supervisory decision support ONLY.

    This module does not send physical robot
    movement commands or enforce hardware limits.

    Real robots require independently validated
    workspace monitoring, motion constraints,
    protective stops and safety-rated controls.
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

class WorkspaceLimitError(Exception):
    pass


class WorkspaceValidationError(WorkspaceLimitError):
    pass


class WorkspaceNotFound(WorkspaceLimitError):
    pass


# ============================================================
# ENUMS
# ============================================================

class WorkspaceRiskLevel(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class WorkspaceDecision(str, Enum):
    ALLOW_SIMULATION = "allow_simulation"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    DENY = "deny"


class WorkspaceViolationType(str, Enum):
    OUTSIDE_WORKSPACE = "outside_workspace"
    NEAR_BOUNDARY = "near_boundary"
    RESTRICTED_ZONE = "restricted_zone"
    HEIGHT_LIMIT = "height_limit"
    SPEED_LIMIT = "speed_limit"
    STALE_TELEMETRY = "stale_telemetry"
    INVALID_TARGET = "invalid_target"
    UNKNOWN_WORKSPACE = "unknown_workspace"


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
        raise WorkspaceValidationError(
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
        raise WorkspaceValidationError(
            f"{name} must be finite"
        )

    return float(value)


def nonnegative(
    value: float,
    name: str,
) -> float:
    result = finite(value, name)

    if result < 0:
        raise WorkspaceValidationError(
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
        raise WorkspaceValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# GEOMETRY
# ============================================================

@dataclass(frozen=True)
class Vector3D:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        finite(self.x, "x")
        finite(self.y, "y")
        finite(self.z, "z")

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


@dataclass(frozen=True)
class Box3D:
    minimum: Vector3D
    maximum: Vector3D

    def __post_init__(self) -> None:
        if not (
            self.minimum.x < self.maximum.x
            and self.minimum.y < self.maximum.y
            and self.minimum.z < self.maximum.z
        ):
            raise WorkspaceValidationError(
                "Box minimum must be below maximum"
            )

    def contains(
        self,
        point: Vector3D,
    ) -> bool:
        return (
            self.minimum.x <= point.x <= self.maximum.x
            and self.minimum.y <= point.y <= self.maximum.y
            and self.minimum.z <= point.z <= self.maximum.z
        )

    def boundary_clearance(
        self,
        point: Vector3D,
    ) -> float:
        if not self.contains(point):
            return 0.0

        return min(
            point.x - self.minimum.x,
            self.maximum.x - point.x,
            point.y - self.minimum.y,
            self.maximum.y - point.y,
            point.z - self.minimum.z,
            self.maximum.z - point.z,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum": self.minimum.to_dict(),
            "maximum": self.maximum.to_dict(),
        }


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class WorkspaceLimitConfig:
    maximum_telemetry_age_seconds: float = 2.0

    maximum_audit_events: int = 20000
    maximum_workspaces: int = 10000

    simulation_only: bool = True

    def __post_init__(self) -> None:
        if (
            nonnegative(
                self.maximum_telemetry_age_seconds,
                "maximum_telemetry_age_seconds",
            ) <= 0
        ):
            raise WorkspaceValidationError(
                "Telemetry age must be positive"
            )

        for name in (
            "maximum_audit_events",
            "maximum_workspaces",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise WorkspaceValidationError(
                    f"{name} must be positive"
                )

        if not self.simulation_only:
            raise WorkspaceValidationError(
                "Hardware control unsupported"
            )


# ============================================================
# RESTRICTED ZONE
# ============================================================

@dataclass(frozen=True)
class RestrictedWorkspaceZone:
    zone_id: str
    bounds: Box3D

    description: str = ""

    def __post_init__(self) -> None:
        validate_id(self.zone_id, "zone_id")


# ============================================================
# WORKSPACE POLICY
# ============================================================

@dataclass(frozen=True)
class RobotWorkspacePolicy:
    tenant_id: str
    robot_id: str
    workspace_id: str

    bounds: Box3D

    safety_margin_m: float = 0.5
    maximum_speed_mps: float = 1.5

    restricted_zones: tuple[
        RestrictedWorkspaceZone,
        ...
    ] = ()

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.workspace_id, "workspace_id")

        nonnegative(
            self.safety_margin_m,
            "safety_margin_m",
        )

        nonnegative(
            self.maximum_speed_mps,
            "maximum_speed_mps",
        )

        zone_ids = [
            zone.zone_id
            for zone in self.restricted_zones
        ]

        if len(zone_ids) != len(set(zone_ids)):
            raise WorkspaceValidationError(
                "Duplicate restricted zone IDs"
            )


# ============================================================
# WORKSPACE SNAPSHOT
# ============================================================

@dataclass(frozen=True)
class RobotWorkspaceSnapshot:
    tenant_id: str
    robot_id: str

    position: Vector3D
    target_position: Vector3D | None

    current_speed_mps: float
    requested_speed_mps: float

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        nonnegative(
            self.current_speed_mps,
            "current_speed_mps",
        )

        nonnegative(
            self.requested_speed_mps,
            "requested_speed_mps",
        )

        validate_time(self.timestamp)


# ============================================================
# WORKSPACE VIOLATION
# ============================================================

@dataclass(frozen=True)
class WorkspaceViolation:
    violation_type: WorkspaceViolationType
    description: str
    risk_level: WorkspaceRiskLevel

    def to_dict(self) -> dict[str, Any]:
        return {
            "violation_type": (
                self.violation_type.value
            ),
            "description": self.description,
            "risk_level": self.risk_level.value,
        }


# ============================================================
# WORKSPACE ASSESSMENT
# ============================================================

@dataclass(frozen=True)
class WorkspaceAssessment:
    assessment_id: str

    tenant_id: str
    robot_id: str
    workspace_id: str | None

    risk_level: WorkspaceRiskLevel
    decision: WorkspaceDecision

    current_position: Vector3D
    target_position: Vector3D | None

    boundary_clearance_m: float | None

    violations: tuple[
        WorkspaceViolation,
        ...
    ]

    evaluated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "workspace_id": self.workspace_id,
            "risk_level": self.risk_level.value,
            "decision": self.decision.value,
            "current_position": (
                self.current_position.to_dict()
            ),
            "target_position": (
                self.target_position.to_dict()
                if self.target_position
                else None
            ),
            "boundary_clearance_m": (
                self.boundary_clearance_m
            ),
            "violations": [
                violation.to_dict()
                for violation in self.violations
            ],
            "evaluated_at": (
                self.evaluated_at.isoformat()
            ),
        }


# ============================================================
# WORKSPACE EVALUATION ENGINE
# ============================================================

class RoboticsWorkspaceLimitEngine:

    def __init__(
        self,
        config: WorkspaceLimitConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        snapshot: RobotWorkspaceSnapshot,
        policy: RobotWorkspacePolicy | None,
    ) -> WorkspaceAssessment:
        violations: list[
            WorkspaceViolation
        ] = []

        def add(
            violation_type: WorkspaceViolationType,
            description: str,
            risk: WorkspaceRiskLevel,
        ) -> None:
            violations.append(
                WorkspaceViolation(
                    violation_type=violation_type,
                    description=description,
                    risk_level=risk,
                )
            )

        now = utc_now()

        age = (
            now - validate_time(snapshot.timestamp)
        ).total_seconds()

        if (
            age < -1.0
            or age
            > self.config.maximum_telemetry_age_seconds
        ):
            add(
                WorkspaceViolationType.STALE_TELEMETRY,
                "Workspace telemetry is stale",
                WorkspaceRiskLevel.UNKNOWN,
            )

        if policy is None:
            add(
                WorkspaceViolationType.UNKNOWN_WORKSPACE,
                "Robot workspace policy is unavailable",
                WorkspaceRiskLevel.UNKNOWN,
            )

            return WorkspaceAssessment(
                assessment_id=str(uuid.uuid4()),
                tenant_id=snapshot.tenant_id,
                robot_id=snapshot.robot_id,
                workspace_id=None,
                risk_level=WorkspaceRiskLevel.UNKNOWN,
                decision=WorkspaceDecision.DENY,
                current_position=snapshot.position,
                target_position=snapshot.target_position,
                boundary_clearance_m=None,
                violations=tuple(violations),
                evaluated_at=now,
            )

        bounds = policy.bounds

        clearance = (
            bounds.boundary_clearance(
                snapshot.position
            )
        )

        if not bounds.contains(snapshot.position):
            add(
                WorkspaceViolationType.OUTSIDE_WORKSPACE,
                "Robot position is outside workspace",
                WorkspaceRiskLevel.CRITICAL,
            )

        elif clearance < policy.safety_margin_m:
            add(
                WorkspaceViolationType.NEAR_BOUNDARY,
                "Robot is near workspace boundary",
                WorkspaceRiskLevel.CAUTION,
            )

        if snapshot.target_position is not None:
            if not bounds.contains(
                snapshot.target_position
            ):
                add(
                    WorkspaceViolationType.INVALID_TARGET,
                    "Requested target is outside workspace",
                    WorkspaceRiskLevel.CRITICAL,
                )

            elif (
                bounds.boundary_clearance(
                    snapshot.target_position
                )
                < policy.safety_margin_m
            ):
                add(
                    WorkspaceViolationType.INVALID_TARGET,
                    "Target violates workspace safety margin",
                    WorkspaceRiskLevel.HIGH,
                )

        for zone in policy.restricted_zones:
            if zone.bounds.contains(
                snapshot.position
            ):
                add(
                    WorkspaceViolationType.RESTRICTED_ZONE,
                    (
                        "Robot entered restricted zone: "
                        f"{zone.zone_id}"
                    ),
                    WorkspaceRiskLevel.CRITICAL,
                )

            if (
                snapshot.target_position is not None
                and zone.bounds.contains(
                    snapshot.target_position
                )
            ):
                add(
                    WorkspaceViolationType.INVALID_TARGET,
                    (
                        "Target enters restricted zone: "
                        f"{zone.zone_id}"
                    ),
                    WorkspaceRiskLevel.CRITICAL,
                )

        if (
            snapshot.current_speed_mps
            > policy.maximum_speed_mps
        ):
            add(
                WorkspaceViolationType.SPEED_LIMIT,
                "Current speed exceeds workspace limit",
                WorkspaceRiskLevel.HIGH,
            )

        if (
            snapshot.requested_speed_mps
            > policy.maximum_speed_mps
        ):
            add(
                WorkspaceViolationType.SPEED_LIMIT,
                "Requested speed exceeds workspace limit",
                WorkspaceRiskLevel.HIGH,
            )

        if not violations:
            risk = WorkspaceRiskLevel.SAFE
            decision = WorkspaceDecision.ALLOW_SIMULATION

        elif any(
            item.risk_level
            in {
                WorkspaceRiskLevel.CRITICAL,
                WorkspaceRiskLevel.HIGH,
                WorkspaceRiskLevel.UNKNOWN,
            }
            for item in violations
        ):
            risk = (
                WorkspaceRiskLevel.CRITICAL
                if any(
                    item.risk_level
                    == WorkspaceRiskLevel.CRITICAL
                    for item in violations
                )
                else WorkspaceRiskLevel.HIGH
            )

            decision = WorkspaceDecision.DENY

        else:
            risk = WorkspaceRiskLevel.CAUTION
            decision = (
                WorkspaceDecision.HUMAN_REVIEW_REQUIRED
            )

        return WorkspaceAssessment(
            assessment_id=str(uuid.uuid4()),
            tenant_id=snapshot.tenant_id,
            robot_id=snapshot.robot_id,
            workspace_id=policy.workspace_id,
            risk_level=risk,
            decision=decision,
            current_position=snapshot.position,
            target_position=snapshot.target_position,
            boundary_clearance_m=clearance,
            violations=tuple(violations),
            evaluated_at=now,
        )


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class WorkspaceAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str

    decision: WorkspaceDecision
    violation_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "decision": self.decision.value,
            "violation_count": self.violation_count,
        }


# ============================================================
# WORKSPACE LIMIT SERVICE
# ============================================================

class RoboticsWorkspaceLimitService:

    def __init__(
        self,
        config: WorkspaceLimitConfig | None = None,
    ) -> None:
        self.config = (
            config or WorkspaceLimitConfig()
        )

        self.engine = RoboticsWorkspaceLimitEngine(
            self.config
        )

        self.policies: dict[
            tuple[str, str],
            RobotWorkspacePolicy
        ] = {}

        self.latest_snapshots: dict[
            tuple[str, str],
            RobotWorkspaceSnapshot
        ] = {}

        self.latest_assessments: dict[
            tuple[str, str],
            WorkspaceAssessment
        ] = {}

        self.audit_events: list[
            WorkspaceAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # REGISTER POLICY
    # --------------------------------------------------------

    def register_policy(
        self,
        policy: RobotWorkspacePolicy,
    ) -> None:
        key = (
            policy.tenant_id,
            policy.robot_id,
        )

        with self._lock:
            if (
                key not in self.policies
                and len(self.policies)
                >= self.config.maximum_workspaces
            ):
                raise WorkspaceLimitError(
                    "Workspace capacity reached"
                )

            self.policies[key] = policy

            # Invalidate assessments when policy changes.
            self.latest_assessments.pop(
                key,
                None,
            )

    # --------------------------------------------------------
    # ASSESS WORKSPACE
    # --------------------------------------------------------

    def assess(
        self,
        snapshot: RobotWorkspaceSnapshot,
    ) -> WorkspaceAssessment:
        key = (
            snapshot.tenant_id,
            snapshot.robot_id,
        )

        with self._lock:
            previous = (
                self.latest_snapshots.get(key)
            )

            if (
                previous is not None
                and snapshot.timestamp
                <= previous.timestamp
            ):
                raise WorkspaceValidationError(
                    "Duplicate or out-of-order telemetry"
                )

            policy = self.policies.get(
                key
            )

            assessment = self.engine.evaluate(
                snapshot,
                policy,
            )

            self.latest_snapshots[
                key
            ] = snapshot

            self.latest_assessments[
                key
            ] = assessment

            self.audit_events.append(
                WorkspaceAuditEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=utc_now(),
                    tenant_id=snapshot.tenant_id,
                    robot_id=snapshot.robot_id,
                    decision=assessment.decision,
                    violation_count=len(
                        assessment.violations
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
    # COMMAND VALIDATION
    # --------------------------------------------------------

    def validate_target(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        target_position: Vector3D,
        requested_speed_mps: float,
    ) -> WorkspaceAssessment:
        nonnegative(
            requested_speed_mps,
            "requested_speed_mps",
        )

        key = (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
        )

        with self._lock:
            snapshot = (
                self.latest_snapshots.get(key)
            )

            if snapshot is None:
                raise WorkspaceNotFound(
                    "No robot position telemetry"
                )

            # Reassess against current policy and
            # fresh telemetry. This checks the
            # endpoint only, not the full trajectory.
            command_snapshot = RobotWorkspaceSnapshot(
                tenant_id=tenant_id,
                robot_id=robot_id,
                position=snapshot.position,
                target_position=target_position,
                current_speed_mps=(
                    snapshot.current_speed_mps
                ),
                requested_speed_mps=requested_speed_mps,
                timestamp=snapshot.timestamp,
            )

            return self.engine.evaluate(
                command_snapshot,
                self.policies.get(key),
            )

    # --------------------------------------------------------
    # GET LATEST
    # --------------------------------------------------------

    def get_latest(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> WorkspaceAssessment:
        with self._lock:
            assessment = (
                self.latest_assessments.get(
                    (tenant_id, robot_id)
                )
            )

            if assessment is None:
                raise WorkspaceNotFound(
                    "No workspace assessment"
                )

            return assessment

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
                item
                for (record_tenant, _), item
                in self.latest_assessments.items()
                if (
                    tenant_id is None
                    or record_tenant == tenant_id
                )
            ]

            return {
                "monitored_robots": len(assessments),
                "safe": sum(
                    item.risk_level
                    == WorkspaceRiskLevel.SAFE
                    for item in assessments
                ),
                "caution": sum(
                    item.risk_level
                    == WorkspaceRiskLevel.CAUTION
                    for item in assessments
                ),
                "denied": sum(
                    item.decision
                    == WorkspaceDecision.DENY
                    for item in assessments
                ),
                "total_violations": sum(
                    len(item.violations)
                    for item in assessments
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
                "generated_at": (
                    utc_now().isoformat()
                ),
                "tenant_id": tenant_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "assessments": [
                    item.to_dict()
                    for (record_tenant, _), item
                    in self.latest_assessments.items()
                    if record_tenant == tenant_id
                ],
                "audit_events": [
                    item.to_dict()
                    for item in self.audit_events
                    if item.tenant_id == tenant_id
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
    service = RoboticsWorkspaceLimitService()

    policy = RobotWorkspacePolicy(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        workspace_id="warehouse_workspace",
        bounds=Box3D(
            minimum=Vector3D(0.0, 0.0, 0.0),
            maximum=Vector3D(20.0, 15.0, 5.0),
        ),
        safety_margin_m=0.5,
        maximum_speed_mps=1.5,
        restricted_zones=(
            RestrictedWorkspaceZone(
                zone_id="hazard_zone_a",
                bounds=Box3D(
                    minimum=Vector3D(8.0, 5.0, 0.0),
                    maximum=Vector3D(12.0, 9.0, 4.0),
                ),
                description="Restricted work area",
            ),
        ),
    )

    service.register_policy(
        policy
    )

    snapshot = RobotWorkspaceSnapshot(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        position=Vector3D(4.0, 4.0, 1.0),
        target_position=Vector3D(10.0, 7.0, 1.0),
        current_speed_mps=0.8,
        requested_speed_mps=1.0,
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
        "data/security/workspace_limits_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

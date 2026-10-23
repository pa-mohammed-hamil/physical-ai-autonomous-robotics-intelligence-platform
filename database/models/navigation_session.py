"""Navigation Session module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/db/models/navigation_session.py

Physical AI & Autonomous Robotics Intelligence Platform

SQLAlchemy 2.x ORM models for:
- Autonomous robot navigation sessions
- Navigation goals and waypoints
- Robot position and trajectory history
- Navigation planning and execution status
- Localization and path tracking
- Obstacle and collision-risk events
- Navigation performance metrics
- Human supervision and safety monitoring
- Multi-tenant data isolation

Navigation records do not directly command robot hardware.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    return str(uuid.uuid4())


def serialize_datetime(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def finite_number(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class NavigationStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    PLANNING = "planning"
    READY = "ready"
    NAVIGATING = "navigating"
    PAUSED = "paused"
    REPLANNING = "replanning"
    GOAL_REACHED = "goal_reached"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EMERGENCY_STOPPED = "emergency_stopped"


class NavigationMode(str, Enum):
    AUTONOMOUS = "autonomous"
    SUPERVISED = "supervised"
    TELEOPERATION = "teleoperation"
    ASSISTED = "assisted"
    SIMULATION = "simulation"
    DIGITAL_TWIN = "digital_twin"


class NavigationEnvironment(str, Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    WAREHOUSE = "warehouse"
    FACTORY = "factory"
    HOSPITAL = "hospital"
    ROAD = "road"
    OFF_ROAD = "off_road"
    LABORATORY = "laboratory"
    SIMULATION = "simulation"
    UNKNOWN = "unknown"


class NavigationEventType(str, Enum):
    CREATED = "created"
    PLANNING_STARTED = "planning_started"
    PATH_PLANNED = "path_planned"
    NAVIGATION_STARTED = "navigation_started"
    WAYPOINT_REACHED = "waypoint_reached"
    GOAL_REACHED = "goal_reached"
    PAUSED = "paused"
    RESUMED = "resumed"
    REPLANNING_STARTED = "replanning_started"
    PATH_UPDATED = "path_updated"
    OBSTACLE_DETECTED = "obstacle_detected"
    COLLISION_RISK = "collision_risk"
    COLLISION_DETECTED = "collision_detected"
    LOCALIZATION_LOST = "localization_lost"
    LOCALIZATION_RECOVERED = "localization_recovered"
    GEOFENCE_VIOLATION = "geofence_violation"
    HUMAN_DETECTED = "human_detected"
    SAFETY_WARNING = "safety_warning"
    EMERGENCY_STOP = "emergency_stop"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NOTE_ADDED = "note_added"


class NavigationEventSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WaypointStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    REACHED = "reached"
    SKIPPED = "skipped"
    FAILED = "failed"


class NavigationSession(Base):
    __tablename__ = "navigation_sessions"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "id",
            name="uq_navigation_session_tenant_id",
        ),
        UniqueConstraint(
            "tenant_id", "session_number",
            name="uq_navigation_session_number",
        ),
        Index(
            "ix_navigation_session_robot_status",
            "tenant_id", "robot_id", "status",
        ),
        Index(
            "ix_navigation_session_created",
            "tenant_id", "created_at",
        ),
        Index(
            "ix_navigation_session_task",
            "tenant_id", "task_id",
        ),
        CheckConstraint(
            "progress_percent >= 0 AND progress_percent <= 100",
            name="ck_navigation_progress",
        ),
        CheckConstraint(
            "planned_distance_m IS NULL OR planned_distance_m >= 0",
            name="ck_navigation_planned_distance",
        ),
        CheckConstraint(
            "traveled_distance_m >= 0",
            name="ck_navigation_traveled_distance",
        ),
        CheckConstraint(
            "replan_count >= 0",
            name="ck_navigation_replan_count",
        ),
        CheckConstraint(
            "collision_count >= 0",
            name="ck_navigation_collision_count",
        ),
        CheckConstraint(
            "obstacle_count >= 0",
            name="ck_navigation_obstacle_count",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    organization_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )
    session_number: Mapped[str] = mapped_column(
        String(100), nullable=False,
    )

    robot_id: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    fleet_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )
    task_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )
    mission_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )
    agent_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )
    model_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )
    simulation_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )

    name: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False,
        default=NavigationStatus.CREATED.value,
    )
    mode: Mapped[str] = mapped_column(
        String(32), nullable=False,
        default=NavigationMode.SUPERVISED.value,
    )
    environment: Mapped[str] = mapped_column(
        String(64), nullable=False,
        default=NavigationEnvironment.UNKNOWN.value,
    )

    map_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )
    map_version: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True,
    )
    coordinate_frame: Mapped[str] = mapped_column(
        String(100), nullable=False, default="map",
    )
    planner_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )
    controller_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )
    localization_method: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True,
    )

    start_pose_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    goal_pose_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    current_pose_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    planned_path_json: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list,
    )

    progress_percent: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
    )
    planned_distance_m: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    traveled_distance_m: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
    )
    remaining_distance_m: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    average_speed_mps: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    maximum_speed_mps: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    localization_confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    path_efficiency: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )

    replan_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    obstacle_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    collision_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )

    requires_human_supervision: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
    )
    safety_approved: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    is_simulation: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )

    assigned_operator_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )
    created_by: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )

    failure_reason: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )
    completion_notes: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
    )

    configuration_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    safety_constraints_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    metrics_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )

    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    last_position_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=utc_now, onupdate=utc_now,
    )

    waypoints: Mapped[list["NavigationWaypoint"]] = relationship(
        "NavigationWaypoint",
        back_populates="session",
        lazy="selectin",
        cascade="save-update, merge",
        order_by="NavigationWaypoint.sequence_number",
    )
    positions: Mapped[list["NavigationPosition"]] = relationship(
        "NavigationPosition",
        back_populates="session",
        lazy="select",
        cascade="save-update, merge",
    )
    events: Mapped[list["NavigationEvent"]] = relationship(
        "NavigationEvent",
        back_populates="session",
        lazy="selectin",
        cascade="save-update, merge",
    )

    @validates(
        "progress_percent",
        "planned_distance_m",
        "traveled_distance_m",
        "remaining_distance_m",
        "average_speed_mps",
        "maximum_speed_mps",
        "localization_confidence",
        "path_efficiency",
    )
    def validate_number(
        self, key: str, value: Optional[float],
    ) -> Optional[float]:
        number = finite_number(value, key)
        if number is None:
            return None
        if key in {"progress_percent"} and not 0 <= number <= 100:
            raise ValueError("progress_percent must be between 0 and 100")
        if key in {"localization_confidence", "path_efficiency"}:
            if not 0 <= number <= 1:
                raise ValueError(f"{key} must be between 0 and 1")
        if key in {
            "planned_distance_m", "traveled_distance_m",
            "remaining_distance_m", "average_speed_mps",
            "maximum_speed_mps",
        } and number < 0:
            raise ValueError(f"{key} cannot be negative")
        return number

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            NavigationStatus.GOAL_REACHED.value,
            NavigationStatus.FAILED.value,
            NavigationStatus.CANCELLED.value,
            NavigationStatus.EMERGENCY_STOPPED.value,
        }

    @property
    def is_active(self) -> bool:
        return self.status in {
            NavigationStatus.PLANNING.value,
            NavigationStatus.NAVIGATING.value,
            NavigationStatus.REPLANNING.value,
        }

    def start(self) -> None:
        if self.status not in {
            NavigationStatus.CREATED.value,
            NavigationStatus.READY.value,
        }:
            raise ValueError("Session cannot start from current status")
        now = utc_now()
        self.status = NavigationStatus.NAVIGATING.value
        self.started_at = self.started_at or now
        self.updated_at = now

    def pause(self) -> None:
        if not self.is_active:
            raise ValueError("Only active sessions can be paused")
        self.status = NavigationStatus.PAUSED.value
        self.updated_at = utc_now()

    def resume(self) -> None:
        if self.status != NavigationStatus.PAUSED.value:
            raise ValueError("Only paused sessions can resume")
        self.status = NavigationStatus.NAVIGATING.value
        self.updated_at = utc_now()

    def complete(self) -> None:
        if self.status not in {
            NavigationStatus.NAVIGATING.value,
            NavigationStatus.REPLANNING.value,
        }:
            raise ValueError("Session is not navigating")
        now = utc_now()
        self.status = NavigationStatus.GOAL_REACHED.value
        self.progress_percent = 100.0
        self.remaining_distance_m = 0.0
        self.completed_at = now
        self.updated_at = now

    def fail(self, reason: str) -> None:
        if self.is_terminal:
            raise ValueError("Session is already terminal")
        if not reason.strip():
            raise ValueError("Failure reason is required")
        now = utc_now()
        self.status = NavigationStatus.FAILED.value
        self.failure_reason = reason.strip()
        self.completed_at = now
        self.updated_at = now

    def record_emergency_stop(self, reason: str) -> None:
        if self.is_terminal:
            raise ValueError("Session is already terminal")
        if not reason.strip():
            raise ValueError("Emergency stop reason is required")
        now = utc_now()
        self.status = NavigationStatus.EMERGENCY_STOPPED.value
        self.failure_reason = reason.strip()
        self.completed_at = now
        self.updated_at = now

    def update_progress(
        self,
        *,
        progress_percent: float,
        current_pose: Optional[dict[str, Any]] = None,
        remaining_distance_m: Optional[float] = None,
    ) -> None:
        if not self.is_active:
            raise ValueError("Session is not active")
        self.progress_percent = progress_percent
        if current_pose is not None:
            self.current_pose_json = dict(current_pose)
            self.last_position_at = utc_now()
        if remaining_distance_m is not None:
            self.remaining_distance_m = remaining_distance_m
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "session_number": self.session_number,
            "robot_id": self.robot_id,
            "fleet_id": self.fleet_id,
            "task_id": self.task_id,
            "mission_id": self.mission_id,
            "agent_id": self.agent_id,
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "mode": self.mode,
            "environment": self.environment,
            "map_id": self.map_id,
            "coordinate_frame": self.coordinate_frame,
            "planner_name": self.planner_name,
            "controller_name": self.controller_name,
            "start_pose": self.start_pose_json,
            "goal_pose": self.goal_pose_json,
            "current_pose": self.current_pose_json,
            "progress_percent": self.progress_percent,
            "planned_distance_m": self.planned_distance_m,
            "traveled_distance_m": self.traveled_distance_m,
            "remaining_distance_m": self.remaining_distance_m,
            "average_speed_mps": self.average_speed_mps,
            "localization_confidence": self.localization_confidence,
            "path_efficiency": self.path_efficiency,
            "replan_count": self.replan_count,
            "obstacle_count": self.obstacle_count,
            "collision_count": self.collision_count,
            "requires_human_supervision": self.requires_human_supervision,
            "safety_approved": self.safety_approved,
            "is_simulation": self.is_simulation,
            "failure_reason": self.failure_reason,
            "metrics": self.metrics_json,
            "metadata": self.metadata_json,
            "started_at": serialize_datetime(self.started_at),
            "completed_at": serialize_datetime(self.completed_at),
            "created_at": serialize_datetime(self.created_at),
            "updated_at": serialize_datetime(self.updated_at),
        }

    def __repr__(self) -> str:
        return (
            f"<NavigationSession(id={self.id!r}, "
            f"robot_id={self.robot_id!r}, status={self.status!r})>"
        )


class NavigationWaypoint(Base):
    __tablename__ = "navigation_waypoints"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "navigation_session_id"],
            ["navigation_sessions.tenant_id", "navigation_sessions.id"],
            name="fk_navigation_waypoint_session",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "tenant_id", "navigation_session_id", "sequence_number",
            name="uq_navigation_waypoint_sequence",
        ),
        Index(
            "ix_navigation_waypoint_session_status",
            "tenant_id", "navigation_session_id", "status",
        ),
        CheckConstraint(
            "sequence_number >= 0",
            name="ck_navigation_waypoint_sequence",
        ),
        CheckConstraint(
            "tolerance_m > 0",
            name="ck_navigation_waypoint_tolerance",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    navigation_session_id: Mapped[str] = mapped_column(
        String(36), nullable=False,
    )
    sequence_number: Mapped[int] = mapped_column(
        Integer, nullable=False,
    )
    name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False,
        default=WaypointStatus.PENDING.value,
    )
    position_x: Mapped[float] = mapped_column(Float, nullable=False)
    position_y: Mapped[float] = mapped_column(Float, nullable=False)
    position_z: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
    )
    orientation_yaw: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    coordinate_frame: Mapped[str] = mapped_column(
        String(100), nullable=False, default="map",
    )
    tolerance_m: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.25,
    )
    pose_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    reached_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )

    session: Mapped["NavigationSession"] = relationship(
        "NavigationSession", back_populates="waypoints",
    )

    @validates(
        "position_x", "position_y", "position_z",
        "orientation_yaw", "tolerance_m",
    )
    def validate_number(
        self, key: str, value: Optional[float],
    ) -> Optional[float]:
        number = finite_number(value, key)
        if key == "tolerance_m" and number is not None and number <= 0:
            raise ValueError("tolerance_m must be positive")
        return number

    def mark_reached(self) -> None:
        if self.status not in {
            WaypointStatus.PENDING.value,
            WaypointStatus.ACTIVE.value,
        }:
            raise ValueError("Waypoint cannot be marked reached")
        self.status = WaypointStatus.REACHED.value
        self.reached_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "navigation_session_id": self.navigation_session_id,
            "sequence_number": self.sequence_number,
            "name": self.name,
            "status": self.status,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "position_z": self.position_z,
            "orientation_yaw": self.orientation_yaw,
            "coordinate_frame": self.coordinate_frame,
            "tolerance_m": self.tolerance_m,
            "pose": self.pose_json,
            "reached_at": serialize_datetime(self.reached_at),
        }


class NavigationPosition(Base):
    """
    Timestamped robot pose samples.

    High-frequency raw telemetry should be stored in
    a time-series store when appropriate.
    """

    __tablename__ = "navigation_positions"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "navigation_session_id"],
            ["navigation_sessions.tenant_id", "navigation_sessions.id"],
            name="fk_navigation_position_session",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_navigation_position_session_time",
            "tenant_id", "navigation_session_id", "recorded_at",
        ),
        CheckConstraint(
            "confidence IS NULL OR "
            "(confidence >= 0 AND confidence <= 1)",
            name="ck_navigation_position_confidence",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    navigation_session_id: Mapped[str] = mapped_column(
        String(36), nullable=False,
    )
    position_x: Mapped[float] = mapped_column(Float, nullable=False)
    position_y: Mapped[float] = mapped_column(Float, nullable=False)
    position_z: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
    )
    orientation_roll: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    orientation_pitch: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    orientation_yaw: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    linear_speed_mps: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    angular_speed_radps: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
    )
    coordinate_frame: Mapped[str] = mapped_column(
        String(100), nullable=False, default="map",
    )
    source: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )

    session: Mapped["NavigationSession"] = relationship(
        "NavigationSession", back_populates="positions",
    )

    @validates(
        "position_x", "position_y", "position_z",
        "orientation_roll", "orientation_pitch", "orientation_yaw",
        "linear_speed_mps", "angular_speed_radps", "confidence",
    )
    def validate_number(
        self, key: str, value: Optional[float],
    ) -> Optional[float]:
        number = finite_number(value, key)
        if key == "confidence" and number is not None:
            if not 0 <= number <= 1:
                raise ValueError("confidence must be between 0 and 1")
        return number

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "navigation_session_id": self.navigation_session_id,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "position_z": self.position_z,
            "orientation_roll": self.orientation_roll,
            "orientation_pitch": self.orientation_pitch,
            "orientation_yaw": self.orientation_yaw,
            "linear_speed_mps": self.linear_speed_mps,
            "angular_speed_radps": self.angular_speed_radps,
            "confidence": self.confidence,
            "coordinate_frame": self.coordinate_frame,
            "source": self.source,
            "recorded_at": serialize_datetime(self.recorded_at),
        }


class NavigationEvent(Base):
    """
    Navigation lifecycle, obstacle and safety event history.
    """

    __tablename__ = "navigation_events"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "navigation_session_id"],
            ["navigation_sessions.tenant_id", "navigation_sessions.id"],
            name="fk_navigation_event_session",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_navigation_event_session_time",
            "tenant_id", "navigation_session_id", "created_at",
        ),
        Index(
            "ix_navigation_event_tenant_type",
            "tenant_id", "event_type",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    navigation_session_id: Mapped[str] = mapped_column(
        String(36), nullable=False,
    )
    event_type: Mapped[str] = mapped_column(
        String(100), nullable=False,
    )
    severity: Mapped[str] = mapped_column(
        String(32), nullable=False,
        default=NavigationEventSeverity.INFO.value,
    )
    message: Mapped[str] = mapped_column(
        Text, nullable=False,
    )
    actor_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True,
    )
    previous_status: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True,
    )
    new_status: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True,
    )
    position_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )

    session: Mapped["NavigationSession"] = relationship(
        "NavigationSession", back_populates="events",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "navigation_session_id": self.navigation_session_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "actor_id": self.actor_id,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "position": self.position_json,
            "metadata": self.metadata_json,
            "created_at": serialize_datetime(self.created_at),
        }


__all__ = [
    "NavigationSession",
    "NavigationWaypoint",
    "NavigationPosition",
    "NavigationEvent",
    "NavigationStatus",
    "NavigationMode",
    "NavigationEnvironment",
    "NavigationEventType",
    "NavigationEventSeverity",
    "WaypointStatus",
    "utc_now",
    "generate_uuid",
]
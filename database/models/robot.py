"""Robot module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/db/models/robot.py

Physical AI & Autonomous Robotics Intelligence Platform

SQLAlchemy 2.x ORM models for:
- Robot registration and lifecycle management
- Robot operational and health status
- Robot telemetry
- Robot location history
- Robot capabilities
- Robot lifecycle events
- Multi-tenant isolation

This module stores robot information and observations.
It does not directly control physical robot hardware.
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
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    validates,
)

from app.db.base import Base


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    return str(uuid.uuid4())


def serialize_datetime(
    value: Optional[datetime],
) -> Optional[str]:
    return value.isoformat() if value is not None else None


def validate_finite(
    value: Optional[float],
    field_name: str,
) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")

    return result


def validate_percentage(
    value: Optional[float],
    field_name: str,
) -> Optional[float]:
    result = validate_finite(value, field_name)

    if result is not None and not 0 <= result <= 100:
        raise ValueError(
            f"{field_name} must be between 0 and 100"
        )

    return result


# ============================================================
# Enumerations
# ============================================================


class RobotType(str, Enum):
    MOBILE = "mobile"
    HUMANOID = "humanoid"
    INDUSTRIAL_ARM = "industrial_arm"
    COLLABORATIVE = "collaborative"
    WAREHOUSE = "warehouse"
    DELIVERY = "delivery"
    SERVICE = "service"
    MEDICAL = "medical"
    AGRICULTURAL = "agricultural"
    CONSTRUCTION = "construction"
    INSPECTION = "inspection"
    SECURITY = "security"
    DRONE = "drone"
    AUTONOMOUS_VEHICLE = "autonomous_vehicle"
    UNDERWATER = "underwater"
    QUADRUPED = "quadruped"
    MANIPULATOR = "manipulator"
    RESEARCH = "research"
    SIMULATED = "simulated"
    CUSTOM = "custom"
    OTHER = "other"


class RobotStatus(str, Enum):
    REGISTERED = "registered"
    OFFLINE = "offline"
    ONLINE = "online"
    IDLE = "idle"
    READY = "ready"
    INITIALIZING = "initializing"
    CALIBRATING = "calibrating"
    WORKING = "working"
    NAVIGATING = "navigating"
    MANIPULATING = "manipulating"
    CHARGING = "charging"
    PAUSED = "paused"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    FAULT = "fault"
    EMERGENCY_STOPPED = "emergency_stopped"
    SHUTDOWN = "shutdown"
    RETIRED = "retired"


class RobotHealthStatus(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    GOOD = "good"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    MAINTENANCE_REQUIRED = "maintenance_required"


class RobotCapabilityType(str, Enum):
    NAVIGATION = "navigation"
    LOCALIZATION = "localization"
    MAPPING = "mapping"
    SLAM = "slam"
    OBJECT_DETECTION = "object_detection"
    OBJECT_TRACKING = "object_tracking"
    SEGMENTATION = "segmentation"
    POSE_ESTIMATION = "pose_estimation"
    DEPTH_ESTIMATION = "depth_estimation"
    SENSOR_FUSION = "sensor_fusion"
    MANIPULATION = "manipulation"
    GRASPING = "grasping"
    PICK_AND_PLACE = "pick_and_place"
    ASSEMBLY = "assembly"
    VOICE_INTERACTION = "voice_interaction"
    VISION_LANGUAGE_ACTION = "vision_language_action"
    AUTONOMOUS_PLANNING = "autonomous_planning"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    HUMAN_ROBOT_INTERACTION = "human_robot_interaction"
    TELEOPERATION = "teleoperation"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    SELF_DIAGNOSTICS = "self_diagnostics"
    CHARGING = "charging"
    DOCKING = "docking"
    OTHER = "other"


class RobotEventType(str, Enum):
    REGISTERED = "registered"
    UPDATED = "updated"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    ONLINE = "online"
    OFFLINE = "offline"
    STATUS_CHANGED = "status_changed"
    HEALTH_CHANGED = "health_changed"
    FLEET_ASSIGNED = "fleet_assigned"
    FLEET_REMOVED = "fleet_removed"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    CHARGING_STARTED = "charging_started"
    CHARGING_COMPLETED = "charging_completed"
    BATTERY_LOW = "battery_low"
    BATTERY_CRITICAL = "battery_critical"
    SENSOR_FAILURE = "sensor_failure"
    ACTUATOR_FAILURE = "actuator_failure"
    MOTOR_FAILURE = "motor_failure"
    THERMAL_WARNING = "thermal_warning"
    COLLISION_DETECTED = "collision_detected"
    SAFETY_WARNING = "safety_warning"
    EMERGENCY_STOP = "emergency_stop"
    FAULT_DETECTED = "fault_detected"
    FAULT_CLEARED = "fault_cleared"
    MAINTENANCE_STARTED = "maintenance_started"
    MAINTENANCE_COMPLETED = "maintenance_completed"
    REPAIR_STARTED = "repair_started"
    REPAIR_COMPLETED = "repair_completed"
    FIRMWARE_UPDATED = "firmware_updated"
    SOFTWARE_UPDATED = "software_updated"
    RETIRED = "retired"
    DELETED = "deleted"
    RESTORED = "restored"
    NOTE_ADDED = "note_added"


class RobotEventSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RobotTelemetrySource(str, Enum):
    ROBOT = "robot"
    ROS = "ros"
    ROS2 = "ros2"
    MQTT = "mqtt"
    KAFKA = "kafka"
    IOT_GATEWAY = "iot_gateway"
    SENSOR_FUSION = "sensor_fusion"
    SIMULATION = "simulation"
    DIGITAL_TWIN = "digital_twin"
    API = "api"
    OTHER = "other"


# ============================================================
# Robot
# ============================================================


class Robot(Base):
    """
    Master robot registration model.

    A robot belongs to one tenant and can optionally
    be assigned to an organization and fleet.
    """

    __tablename__ = "robots"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_robot_tenant_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "serial_number",
            name="uq_robot_tenant_serial",
        ),
        Index(
            "ix_robot_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_robot_tenant_health",
            "tenant_id",
            "health_status",
        ),
        Index(
            "ix_robot_tenant_fleet",
            "tenant_id",
            "fleet_id",
        ),
        Index(
            "ix_robot_tenant_organization",
            "tenant_id",
            "organization_id",
        ),
        Index(
            "ix_robot_tenant_type",
            "tenant_id",
            "robot_type",
        ),
        Index(
            "ix_robot_tenant_last_seen",
            "tenant_id",
            "last_seen_at",
        ),
        CheckConstraint(
            "battery_percent IS NULL OR "
            "(battery_percent >= 0 AND battery_percent <= 100)",
            name="ck_robot_battery_percent",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    organization_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    fleet_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    serial_number: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    manufacturer: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    robot_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=RobotType.OTHER.value,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=RobotStatus.REGISTERED.value,
    )

    health_status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=RobotHealthStatus.UNKNOWN.value,
    )

    battery_percent: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    firmware_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    software_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    hardware_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    operating_system: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    ros_distribution: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45),
        nullable=True,
    )

    mac_address: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
    )

    hostname: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    location_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    assigned_operator_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    registered_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    image_uri: Mapped[Optional[str]] = mapped_column(
        String(2048),
        nullable=True,
    )

    documentation_uri: Mapped[Optional[str]] = mapped_column(
        String(2048),
        nullable=True,
    )

    configuration_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    specifications_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    safety_configuration_json: Mapped[
        dict[str, Any]
    ] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_simulated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    requires_human_supervision: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    last_seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_heartbeat_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    activated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    retired_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    telemetry: Mapped[list["RobotTelemetry"]] = relationship(
        "RobotTelemetry",
        back_populates="robot",
        cascade="save-update, merge",
        lazy="select",
    )

    locations: Mapped[list["RobotLocation"]] = relationship(
        "RobotLocation",
        back_populates="robot",
        cascade="save-update, merge",
        lazy="select",
    )

    capabilities: Mapped[list["RobotCapability"]] = relationship(
        "RobotCapability",
        back_populates="robot",
        cascade="save-update, merge",
        lazy="selectin",
    )

    events: Mapped[list["RobotEvent"]] = relationship(
        "RobotEvent",
        back_populates="robot",
        cascade="save-update, merge",
        lazy="selectin",
    )

    @validates("name", "serial_number")
    def validate_required_string(
        self,
        key: str,
        value: str,
    ) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} is required")

        return value.strip()

    @validates("battery_percent")
    def validate_battery(
        self,
        key: str,
        value: Optional[float],
    ) -> Optional[float]:
        return validate_percentage(value, key)

    @property
    def is_online(self) -> bool:
        return (
            self.is_active
            and not self.is_deleted
            and self.status not in {
                RobotStatus.OFFLINE.value,
                RobotStatus.SHUTDOWN.value,
                RobotStatus.RETIRED.value,
                RobotStatus.REGISTERED.value,
            }
        )

    @property
    def is_healthy(self) -> bool:
        return (
            self.is_online
            and self.health_status in {
                RobotHealthStatus.HEALTHY.value,
                RobotHealthStatus.GOOD.value,
            }
            and self.status not in {
                RobotStatus.FAULT.value,
                RobotStatus.EMERGENCY_STOPPED.value,
            }
        )

    @property
    def requires_attention(self) -> bool:
        return (
            self.health_status in {
                RobotHealthStatus.WARNING.value,
                RobotHealthStatus.DEGRADED.value,
                RobotHealthStatus.CRITICAL.value,
                RobotHealthStatus.FAILED.value,
                RobotHealthStatus.MAINTENANCE_REQUIRED.value,
            }
            or self.status in {
                RobotStatus.FAULT.value,
                RobotStatus.EMERGENCY_STOPPED.value,
                RobotStatus.DEGRADED.value,
            }
        )

    def activate(self) -> None:
        if self.is_deleted:
            raise ValueError(
                "Deleted robots cannot be activated"
            )

        if self.status == RobotStatus.RETIRED.value:
            raise ValueError(
                "Retired robots cannot be activated"
            )

        now = utc_now()

        self.is_active = True
        self.status = RobotStatus.IDLE.value
        self.activated_at = now
        self.updated_at = now

    def deactivate(self) -> None:
        if self.is_deleted:
            raise ValueError(
                "Deleted robots cannot be deactivated"
            )

        self.is_active = False
        self.status = RobotStatus.OFFLINE.value
        self.updated_at = utc_now()

    def update_health(
        self,
        health_status: RobotHealthStatus | str,
        battery_percent: Optional[float] = None,
    ) -> None:
        value = (
            health_status.value
            if isinstance(health_status, RobotHealthStatus)
            else health_status
        )

        if value not in {
            item.value for item in RobotHealthStatus
        }:
            raise ValueError("Invalid robot health status")

        self.health_status = value

        if battery_percent is not None:
            self.battery_percent = battery_percent

        self.updated_at = utc_now()

    def record_heartbeat(
        self,
        timestamp: Optional[datetime] = None,
    ) -> None:
        now = timestamp or utc_now()

        if now.tzinfo is None:
            raise ValueError(
                "Heartbeat timestamp must be timezone-aware"
            )

        if (
            self.last_heartbeat_at is not None
            and now < self.last_heartbeat_at
        ):
            raise ValueError("Out-of-order heartbeat")

        self.last_heartbeat_at = now
        self.last_seen_at = now
        self.updated_at = utc_now()

    def retire(self) -> None:
        if self.is_deleted:
            raise ValueError("Deleted robots cannot be retired")

        now = utc_now()

        self.status = RobotStatus.RETIRED.value
        self.is_active = False
        self.retired_at = now
        self.updated_at = now

    def soft_delete(self) -> None:
        if self.is_deleted:
            raise ValueError("Robot is already deleted")

        now = utc_now()

        self.is_deleted = True
        self.is_active = False
        self.status = RobotStatus.OFFLINE.value
        self.deleted_at = now
        self.updated_at = now

    def restore(self) -> None:
        if not self.is_deleted:
            raise ValueError("Robot is not deleted")

        self.is_deleted = False
        self.is_active = False
        self.status = RobotStatus.REGISTERED.value
        self.deleted_at = None
        self.updated_at = utc_now()

    def to_dict(
        self,
        *,
        include_configuration: bool = False,
    ) -> dict[str, Any]:
        result = {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "fleet_id": self.fleet_id,
            "name": self.name,
            "serial_number": self.serial_number,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "robot_type": self.robot_type,
            "description": self.description,
            "status": self.status,
            "health_status": self.health_status,
            "battery_percent": self.battery_percent,
            "firmware_version": self.firmware_version,
            "software_version": self.software_version,
            "hardware_version": self.hardware_version,
            "operating_system": self.operating_system,
            "ros_distribution": self.ros_distribution,
            "location_name": self.location_name,
            "assigned_operator_id": self.assigned_operator_id,
            "image_uri": self.image_uri,
            "documentation_uri": self.documentation_uri,
            "is_active": self.is_active,
            "is_deleted": self.is_deleted,
            "is_simulated": self.is_simulated,
            "requires_human_supervision": (
                self.requires_human_supervision
            ),
            "is_online": self.is_online,
            "is_healthy": self.is_healthy,
            "requires_attention": self.requires_attention,
            "last_seen_at": serialize_datetime(
                self.last_seen_at
            ),
            "last_heartbeat_at": serialize_datetime(
                self.last_heartbeat_at
            ),
            "activated_at": serialize_datetime(
                self.activated_at
            ),
            "retired_at": serialize_datetime(
                self.retired_at
            ),
            "created_at": serialize_datetime(
                self.created_at
            ),
            "updated_at": serialize_datetime(
                self.updated_at
            ),
        }

        if include_configuration:
            result["configuration"] = self.configuration_json
            result["specifications"] = self.specifications_json
            result["safety_configuration"] = (
                self.safety_configuration_json
            )
            result["metadata"] = self.metadata_json

        return result

    def __repr__(self) -> str:
        return (
            f"<Robot("
            f"id={self.id!r}, "
            f"name={self.name!r}, "
            f"serial_number={self.serial_number!r}, "
            f"status={self.status!r}"
            f")>"
        )


# ============================================================
# Robot Telemetry
# ============================================================


class RobotTelemetry(Base):
    """
    Timestamped robot telemetry measurements.

    For very high-frequency streams, use a time-series
    database or partitioned PostgreSQL tables.
    """

    __tablename__ = "robot_telemetry"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "robot_id"],
            ["robots.tenant_id", "robots.id"],
            name="fk_robot_telemetry_tenant_robot",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_robot_telemetry_robot_time",
            "tenant_id",
            "robot_id",
            "recorded_at",
        ),
        Index(
            "ix_robot_telemetry_source",
            "tenant_id",
            "source",
        ),
        CheckConstraint(
            "battery_percent IS NULL OR "
            "(battery_percent >= 0 AND battery_percent <= 100)",
            name="ck_robot_telemetry_battery",
        ),
        CheckConstraint(
            "cpu_percent IS NULL OR "
            "(cpu_percent >= 0 AND cpu_percent <= 100)",
            name="ck_robot_telemetry_cpu",
        ),
        CheckConstraint(
            "memory_percent IS NULL OR "
            "(memory_percent >= 0 AND memory_percent <= 100)",
            name="ck_robot_telemetry_memory",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    robot_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=RobotTelemetrySource.ROBOT.value,
    )

    battery_percent: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    battery_voltage_v: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    battery_current_a: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    battery_temperature_c: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    cpu_percent: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    memory_percent: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    cpu_temperature_c: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    temperature_c: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    motor_temperature_c: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    vibration_level: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    power_consumption_w: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    network_latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    linear_velocity_mps: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    angular_velocity_radps: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    joint_states_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    sensor_readings_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    diagnostics_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    robot: Mapped["Robot"] = relationship(
        "Robot",
        back_populates="telemetry",
    )

    @validates(
        "battery_percent",
        "cpu_percent",
        "memory_percent",
    )
    def validate_percentages(
        self,
        key: str,
        value: Optional[float],
    ) -> Optional[float]:
        return validate_percentage(value, key)

    @validates(
        "battery_voltage_v",
        "battery_current_a",
        "battery_temperature_c",
        "cpu_temperature_c",
        "temperature_c",
        "motor_temperature_c",
        "vibration_level",
        "power_consumption_w",
        "network_latency_ms",
        "linear_velocity_mps",
        "angular_velocity_radps",
    )
    def validate_measurement(
        self,
        key: str,
        value: Optional[float],
    ) -> Optional[float]:
        return validate_finite(value, key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "source": self.source,
            "battery_percent": self.battery_percent,
            "battery_voltage_v": self.battery_voltage_v,
            "battery_current_a": self.battery_current_a,
            "battery_temperature_c": self.battery_temperature_c,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "cpu_temperature_c": self.cpu_temperature_c,
            "temperature_c": self.temperature_c,
            "motor_temperature_c": self.motor_temperature_c,
            "vibration_level": self.vibration_level,
            "power_consumption_w": self.power_consumption_w,
            "network_latency_ms": self.network_latency_ms,
            "linear_velocity_mps": self.linear_velocity_mps,
            "angular_velocity_radps": self.angular_velocity_radps,
            "joint_states": self.joint_states_json,
            "sensor_readings": self.sensor_readings_json,
            "diagnostics": self.diagnostics_json,
            "metadata": self.metadata_json,
            "recorded_at": serialize_datetime(
                self.recorded_at
            ),
        }


# ============================================================
# Robot Location
# ============================================================


class RobotLocation(Base):
    """
    Robot location and pose history.

    Supports GPS, indoor mapping, SLAM, localization
    and simulation coordinate systems.
    """

    __tablename__ = "robot_locations"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "robot_id"],
            ["robots.tenant_id", "robots.id"],
            name="fk_robot_location_tenant_robot",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_robot_location_robot_time",
            "tenant_id",
            "robot_id",
            "recorded_at",
        ),
        CheckConstraint(
            "latitude IS NULL OR "
            "(latitude >= -90 AND latitude <= 90)",
            name="ck_robot_location_latitude",
        ),
        CheckConstraint(
            "longitude IS NULL OR "
            "(longitude >= -180 AND longitude <= 180)",
            name="ck_robot_location_longitude",
        ),
        CheckConstraint(
            "confidence IS NULL OR "
            "(confidence >= 0 AND confidence <= 1)",
            name="ck_robot_location_confidence",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    robot_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    latitude: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    longitude: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    altitude_m: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    position_x: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    position_y: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    position_z: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    orientation_roll: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    orientation_pitch: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    orientation_yaw: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    coordinate_frame: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="map",
    )

    localization_method: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    map_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    location_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    pose_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    covariance_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    robot: Mapped["Robot"] = relationship(
        "Robot",
        back_populates="locations",
    )

    @validates(
        "latitude",
        "longitude",
        "altitude_m",
        "position_x",
        "position_y",
        "position_z",
        "orientation_roll",
        "orientation_pitch",
        "orientation_yaw",
        "confidence",
    )
    def validate_coordinates(
        self,
        key: str,
        value: Optional[float],
    ) -> Optional[float]:
        result = validate_finite(value, key)

        if result is None:
            return None

        if key == "latitude" and not -90 <= result <= 90:
            raise ValueError("Invalid latitude")

        if key == "longitude" and not -180 <= result <= 180:
            raise ValueError("Invalid longitude")

        if key == "confidence" and not 0 <= result <= 1:
            raise ValueError("Confidence must be between 0 and 1")

        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_m": self.altitude_m,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "position_z": self.position_z,
            "orientation_roll": self.orientation_roll,
            "orientation_pitch": self.orientation_pitch,
            "orientation_yaw": self.orientation_yaw,
            "confidence": self.confidence,
            "coordinate_frame": self.coordinate_frame,
            "localization_method": self.localization_method,
            "map_id": self.map_id,
            "location_name": self.location_name,
            "pose": self.pose_json,
            "covariance": self.covariance_json,
            "metadata": self.metadata_json,
            "recorded_at": serialize_datetime(
                self.recorded_at
            ),
        }


# ============================================================
# Robot Capability
# ============================================================


class RobotCapability(Base):
    """
    Registered robot capabilities.

    A capability record does not grant permission
    to execute a physical action.
    """

    __tablename__ = "robot_capabilities"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "robot_id"],
            ["robots.tenant_id", "robots.id"],
            name="fk_robot_capability_tenant_robot",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "tenant_id",
            "robot_id",
            "capability_name",
            name="uq_robot_capability_name",
        ),
        Index(
            "ix_robot_capability_tenant_type",
            "tenant_id",
            "capability_type",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    robot_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    capability_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    capability_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=RobotCapabilityType.OTHER.value,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    model_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    is_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    requires_human_approval: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    verified_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    configuration_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    requirements_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    robot: Mapped["Robot"] = relationship(
        "Robot",
        back_populates="capabilities",
    )

    @property
    def is_available(self) -> bool:
        return self.is_enabled and self.is_verified

    def verify(self, user_id: str) -> None:
        if not user_id.strip():
            raise ValueError("Verifier ID is required")

        now = utc_now()

        self.is_verified = True
        self.verified_by = user_id
        self.verified_at = now
        self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "capability_name": self.capability_name,
            "capability_type": self.capability_type,
            "description": self.description,
            "version": self.version,
            "model_id": self.model_id,
            "is_enabled": self.is_enabled,
            "requires_human_approval": (
                self.requires_human_approval
            ),
            "is_verified": self.is_verified,
            "is_available": self.is_available,
            "verified_by": self.verified_by,
            "configuration": self.configuration_json,
            "requirements": self.requirements_json,
            "metadata": self.metadata_json,
            "verified_at": serialize_datetime(
                self.verified_at
            ),
            "created_at": serialize_datetime(
                self.created_at
            ),
            "updated_at": serialize_datetime(
                self.updated_at
            ),
        }


# ============================================================
# Robot Event
# ============================================================


class RobotEvent(Base):
    """
    Robot lifecycle and operational event history.

    Production systems should enforce append-only
    audit behavior at the database level.
    """

    __tablename__ = "robot_events"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "robot_id"],
            ["robots.tenant_id", "robots.id"],
            name="fk_robot_event_tenant_robot",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_robot_event_robot_time",
            "tenant_id",
            "robot_id",
            "created_at",
        ),
        Index(
            "ix_robot_event_type",
            "tenant_id",
            "event_type",
        ),
        Index(
            "ix_robot_event_severity",
            "tenant_id",
            "severity",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    robot_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=RobotEventSeverity.INFO.value,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    actor_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    previous_status: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    new_status: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    correlation_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    requires_human_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    robot: Mapped["Robot"] = relationship(
        "Robot",
        back_populates="events",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "actor_id": self.actor_id,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "correlation_id": self.correlation_id,
            "requires_human_review": (
                self.requires_human_review
            ),
            "metadata": self.metadata_json,
            "created_at": serialize_datetime(
                self.created_at
            ),
        }


# ============================================================
# Exports
# ============================================================


__all__ = [
    "Robot",
    "RobotTelemetry",
    "RobotLocation",
    "RobotCapability",
    "RobotEvent",
    "RobotType",
    "RobotStatus",
    "RobotHealthStatus",
    "RobotCapabilityType",
    "RobotEventType",
    "RobotEventSeverity",
    "RobotTelemetrySource",
    "utc_now",
    "generate_uuid",
    "serialize_datetime",
    "validate_finite",
    "validate_percentage",
]
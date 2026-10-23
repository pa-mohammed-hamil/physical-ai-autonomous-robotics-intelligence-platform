"""Sensor Manager module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/sensor_manager.py

Python 3.11+
Standard library only.

ROBOT SENSOR MANAGER — SIMULATION ONLY

Features:
- Multi-robot, multi-tenant sensor registry
- Camera, LiDAR, IMU, GPS, encoder, force/torque, etc.
- Sensor configuration and capabilities
- Sensor observation ingestion
- Data type and numeric validation
- Sequence number validation
- Telemetry freshness monitoring
- Sensor health and anomaly detection
- Sensor calibration metadata
- Sensor activation/deactivation
- Sensor history and audit events
- Robot-wide sensor health summaries
- JSON report export

This is a telemetry/state-management component.
It does not connect to real hardware or ROS 2.

Production sensor readings must be authenticated,
time-synchronized and independently validated.
"""

from __future__ import annotations

import copy
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
from typing import Any, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class SensorManagerError(Exception):
    pass


class SensorValidationError(SensorManagerError):
    pass


class SensorNotFound(SensorManagerError):
    pass


class SensorConflict(SensorManagerError):
    pass


class SensorPermissionDenied(SensorManagerError):
    pass


# ============================================================
# ENUMS
# ============================================================

class SensorType(str, Enum):
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    THERMAL_CAMERA = "thermal_camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    ULTRASONIC = "ultrasonic"
    PROXIMITY = "proximity"
    FORCE_TORQUE = "force_torque"
    ENCODER = "encoder"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    VIBRATION = "vibration"
    BATTERY = "battery"
    MICROPHONE = "microphone"
    TACTILE = "tactile"
    OTHER = "other"


class SensorStatus(str, Enum):
    REGISTERED = "registered"
    ACTIVE = "active"
    DEGRADED = "degraded"
    STALE = "stale"
    FAULTED = "faulted"
    DISABLED = "disabled"


class SensorHealth(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    STALE = "stale"
    DISABLED = "disabled"


class SensorEventType(str, Enum):
    REGISTERED = "registered"
    ACTIVATED = "activated"
    DISABLED = "disabled"
    OBSERVATION_RECEIVED = "observation_received"
    HEALTH_CHANGED = "health_changed"
    STALE = "stale"
    FAULT_DETECTED = "fault_detected"
    CALIBRATION_UPDATED = "calibration_updated"
    FAULT_CLEARED = "fault_cleared"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}",
            value,
        )
    ):
        raise SensorValidationError(
            f"Invalid {name}"
        )
    return value


def finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise SensorValidationError(
            f"{name} must be a finite number"
        )
    return float(value)


def positive(value: Any, name: str) -> float:
    result = finite(value, name)
    if result <= 0:
        raise SensorValidationError(
            f"{name} must be positive"
        )
    return result


def timestamp_utc(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise SensorValidationError(
            "Timezone-aware timestamp required"
        )
    return value.astimezone(timezone.utc)


def validate_json(
    data: Mapping[str, Any],
    maximum_bytes: int,
) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise SensorValidationError(
            "Sensor data must be a mapping"
        )

    try:
        encoded = json.dumps(
            dict(data),
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise SensorValidationError(
            "Invalid JSON sensor data"
        ) from exc

    if len(encoded) > maximum_bytes:
        raise SensorValidationError(
            "Sensor payload size exceeded"
        )

    return json.loads(encoded)


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class SensorManagerConfig:
    maximum_sensors: int = 50000
    maximum_history_per_sensor: int = 500
    maximum_audit_events: int = 20000
    maximum_payload_bytes: int = 65536

    default_timeout_seconds: float = 5.0
    maximum_observation_age_seconds: float = 5.0

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_sensors",
            "maximum_history_per_sensor",
            "maximum_audit_events",
            "maximum_payload_bytes",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise SensorValidationError(
                    f"{name} must be positive"
                )

        positive(
            self.default_timeout_seconds,
            "default_timeout_seconds",
        )
        positive(
            self.maximum_observation_age_seconds,
            "maximum_observation_age_seconds",
        )

        if not self.simulation_only:
            raise SensorValidationError(
                "Real sensor hardware access unsupported"
            )


# ============================================================
# SENSOR DEFINITION
# ============================================================

@dataclass(frozen=True)
class SensorDefinition:
    tenant_id: str
    robot_id: str
    sensor_id: str

    name: str
    sensor_type: SensorType

    frame_id: str = "base_link"
    unit: str = "unitless"

    minimum_value: float | None = None
    maximum_value: float | None = None

    warning_minimum: float | None = None
    warning_maximum: float | None = None

    timeout_seconds: float = 5.0
    calibration_required: bool = False

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "sensor_id",
            "frame_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or len(self.name) > 256
        ):
            raise SensorValidationError(
                "Invalid sensor name"
            )

        if not isinstance(
            self.sensor_type,
            SensorType,
        ):
            raise SensorValidationError(
                "Invalid sensor type"
            )

        if (
            not isinstance(self.unit, str)
            or not self.unit.strip()
            or len(self.unit) > 64
        ):
            raise SensorValidationError(
                "Invalid sensor unit"
            )

        positive(
            self.timeout_seconds,
            "timeout_seconds",
        )

        if not isinstance(
            self.calibration_required,
            bool,
        ):
            raise SensorValidationError(
                "calibration_required must be boolean"
            )

        for name in (
            "minimum_value",
            "maximum_value",
            "warning_minimum",
            "warning_maximum",
        ):
            value = getattr(self, name)
            if value is not None:
                finite(value, name)

        if (
            self.minimum_value is not None
            and self.maximum_value is not None
            and self.minimum_value >= self.maximum_value
        ):
            raise SensorValidationError(
                "Invalid sensor range"
            )

        if (
            self.warning_minimum is not None
            and self.warning_maximum is not None
            and self.warning_minimum
            >= self.warning_maximum
        ):
            raise SensorValidationError(
                "Invalid warning range"
            )


# ============================================================
# SENSOR OBSERVATION
# ============================================================

@dataclass(frozen=True)
class SensorObservation:
    tenant_id: str
    robot_id: str
    sensor_id: str

    sequence_number: int

    data: Mapping[str, Any]

    healthy: bool = True
    fault_detected: bool = False

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "sensor_id",
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
            raise SensorValidationError(
                "Invalid observation sequence"
            )

        if not isinstance(self.data, Mapping):
            raise SensorValidationError(
                "Observation data must be a mapping"
            )

        if (
            not isinstance(self.healthy, bool)
            or not isinstance(self.fault_detected, bool)
        ):
            raise SensorValidationError(
                "Health flags must be boolean"
            )

        timestamp_utc(self.timestamp)


# ============================================================
# CALIBRATION METADATA
# ============================================================

@dataclass(frozen=True)
class SensorCalibration:
    calibration_id: str
    version: str
    approved: bool
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        validate_id(
            self.calibration_id,
            "calibration_id",
        )

        if (
            not isinstance(self.version, str)
            or not re.fullmatch(
                r"\d+\.\d+\.\d+",
                self.version,
            )
        ):
            raise SensorValidationError(
                "Invalid calibration version"
            )

        if not isinstance(self.approved, bool):
            raise SensorValidationError(
                "approved must be boolean"
            )

        if self.expires_at is not None:
            timestamp_utc(
                self.expires_at
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "version": self.version,
            "approved": self.approved,
            "expires_at": (
                self.expires_at.isoformat()
                if self.expires_at
                else None
            ),
        }


# ============================================================
# SENSOR RECORD
# ============================================================

@dataclass
class SensorRecord:
    definition: SensorDefinition

    enabled: bool = False
    status: SensorStatus = SensorStatus.REGISTERED
    health: SensorHealth = SensorHealth.UNKNOWN

    last_sequence: int = -1
    last_received_monotonic: float | None = None

    last_data: dict[str, Any] | None = None
    last_timestamp: datetime | None = None

    calibration: SensorCalibration | None = None

    fault_latched: bool = False
    fault_reasons: list[str] = field(
        default_factory=list
    )

    history: deque[dict[str, Any]] = field(
        default_factory=deque
    )

    registered_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        d = self.definition

        return {
            "tenant_id": d.tenant_id,
            "robot_id": d.robot_id,
            "sensor_id": d.sensor_id,
            "name": d.name,
            "sensor_type": d.sensor_type.value,
            "frame_id": d.frame_id,
            "unit": d.unit,
            "enabled": self.enabled,
            "status": self.status.value,
            "health": self.health.value,
            "last_sequence": self.last_sequence,
            "last_timestamp": (
                self.last_timestamp.isoformat()
                if self.last_timestamp
                else None
            ),
            "last_data": copy.deepcopy(
                self.last_data
            ),
            "calibration": (
                self.calibration.to_dict()
                if self.calibration
                else None
            ),
            "fault_latched": self.fault_latched,
            "fault_reasons": list(
                self.fault_reasons
            ),
            "history_count": len(self.history),
            "registered_at": (
                self.registered_at.isoformat()
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class SensorAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    sensor_id: str

    event_type: SensorEventType
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "sensor_id": self.sensor_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# SENSOR MANAGER
# ============================================================

class RoboticsSensorManager:

    """
    Thread-safe simulation sensor registry.

    Expected flow:

        register_sensor()
             |
        update_calibration()
             |
        activate_sensor()
             |
        ingest_observation()
             |
        evaluate_sensor()
             |
        evaluate_robot()
    """

    def __init__(
        self,
        config: SensorManagerConfig | None = None,
    ) -> None:
        self.config = (
            config or SensorManagerConfig()
        )

        self._sensors: dict[
            tuple[str, str, str],
            SensorRecord
        ] = {}

        self._audit_events: deque[
            SensorAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------

    def _key(
        self,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
    ) -> tuple[str, str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
            validate_id(sensor_id, "sensor_id"),
        )

    def _get(
        self,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
    ) -> SensorRecord:
        record = self._sensors.get(
            self._key(
                tenant_id,
                robot_id,
                sensor_id,
            )
        )

        if record is None:
            raise SensorNotFound(
                "Sensor not registered"
            )

        return record

    def _audit(
        self,
        record: SensorRecord,
        event_type: SensorEventType,
        details: str,
    ) -> None:
        d = record.definition

        self._audit_events.append(
            SensorAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=d.tenant_id,
                robot_id=d.robot_id,
                sensor_id=d.sensor_id,
                event_type=event_type,
                details=details,
            )
        )

    # --------------------------------------------------------
    # CALIBRATION CHECK
    # --------------------------------------------------------

    @staticmethod
    def _calibration_valid(
        record: SensorRecord,
    ) -> bool:
        if not record.definition.calibration_required:
            return True

        calibration = record.calibration

        if (
            calibration is None
            or not calibration.approved
        ):
            return False

        if (
            calibration.expires_at is not None
            and utc_now()
            >= calibration.expires_at
        ):
            return False

        return True

    # --------------------------------------------------------
    # REGISTER SENSOR
    # --------------------------------------------------------

    def register_sensor(
        self,
        definition: SensorDefinition,
    ) -> dict[str, Any]:
        if not isinstance(
            definition,
            SensorDefinition,
        ):
            raise SensorValidationError(
                "Invalid sensor definition"
            )

        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.robot_id,
                definition.sensor_id,
            )

            if key in self._sensors:
                raise SensorConflict(
                    "Sensor already registered"
                )

            if (
                len(self._sensors)
                >= self.config.maximum_sensors
            ):
                raise SensorManagerError(
                    "Sensor registry capacity reached"
                )

            record = SensorRecord(
                definition=definition,
                history=deque(
                    maxlen=(
                        self.config.maximum_history_per_sensor
                    )
                ),
            )

            self._sensors[key] = record

            self._audit(
                record,
                SensorEventType.REGISTERED,
                "Sensor registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # UPDATE CALIBRATION
    # --------------------------------------------------------

    def update_calibration(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
        calibration: SensorCalibration,
        authorized: bool,
        independently_verified: bool,
    ) -> dict[str, Any]:
        if (
            authorized is not True
            or independently_verified is not True
        ):
            raise SensorPermissionDenied(
                "Verified calibration update required"
            )

        if not isinstance(
            calibration,
            SensorCalibration,
        ):
            raise SensorValidationError(
                "Invalid calibration"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                sensor_id,
            )

            if record.enabled:
                raise SensorConflict(
                    "Disable sensor before calibration update"
                )

            record.calibration = calibration

            self._audit(
                record,
                SensorEventType.CALIBRATION_UPDATED,
                "Calibration metadata updated",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # ACTIVATE SENSOR
    # --------------------------------------------------------

    def activate_sensor(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise SensorPermissionDenied(
                "Activation authorization required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                sensor_id,
            )

            if record.fault_latched:
                raise SensorConflict(
                    "Sensor fault must be cleared"
                )

            if not self._calibration_valid(record):
                raise SensorConflict(
                    "Valid calibration required"
                )

            record.enabled = True

            # No data received yet.
            record.status = SensorStatus.STALE
            record.health = SensorHealth.STALE

            self._audit(
                record,
                SensorEventType.ACTIVATED,
                "Sensor activated in simulation",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # DISABLE SENSOR
    # --------------------------------------------------------

    def disable_sensor(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise SensorPermissionDenied(
                "Disable authorization required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                sensor_id,
            )

            record.enabled = False
            record.status = SensorStatus.DISABLED
            record.health = SensorHealth.DISABLED

            self._audit(
                record,
                SensorEventType.DISABLED,
                "Sensor disabled",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # EVALUATE SENSOR
    # --------------------------------------------------------

    def _evaluate(
        self,
        record: SensorRecord,
    ) -> None:
        previous_health = record.health

        if not record.enabled:
            record.status = SensorStatus.DISABLED
            record.health = SensorHealth.DISABLED

        elif record.fault_latched:
            record.status = SensorStatus.FAULTED
            record.health = SensorHealth.CRITICAL

        elif not self._calibration_valid(record):
            record.status = SensorStatus.DEGRADED
            record.health = SensorHealth.WARNING

        elif (
            record.last_received_monotonic is None
            or time.monotonic()
            - record.last_received_monotonic
            > record.definition.timeout_seconds
        ):
            record.status = SensorStatus.STALE
            record.health = SensorHealth.STALE

        elif record.health == SensorHealth.WARNING:
            record.status = SensorStatus.DEGRADED

        else:
            record.status = SensorStatus.ACTIVE
            record.health = SensorHealth.HEALTHY

        if previous_health != record.health:
            self._audit(
                record,
                SensorEventType.HEALTH_CHANGED,
                (
                    f"{previous_health.value} -> "
                    f"{record.health.value}"
                ),
            )

            if record.health == SensorHealth.STALE:
                self._audit(
                    record,
                    SensorEventType.STALE,
                    "Sensor telemetry stale",
                )

    # --------------------------------------------------------
    # INGEST OBSERVATION
    # --------------------------------------------------------

    def ingest_observation(
        self,
        observation: SensorObservation,
    ) -> dict[str, Any]:
        if not isinstance(
            observation,
            SensorObservation,
        ):
            raise SensorValidationError(
                "Invalid sensor observation"
            )

        age = (
            utc_now()
            - timestamp_utc(
                observation.timestamp
            )
        ).total_seconds()

        if (
            age < 0
            or age
            > self.config.maximum_observation_age_seconds
        ):
            raise SensorValidationError(
                "Stale or future observation"
            )

        safe_data = validate_json(
            observation.data,
            self.config.maximum_payload_bytes,
        )

        with self._lock:
            record = self._get(
                observation.tenant_id,
                observation.robot_id,
                observation.sensor_id,
            )

            if not record.enabled:
                raise SensorConflict(
                    "Sensor is disabled"
                )

            if (
                observation.sequence_number
                <= record.last_sequence
            ):
                raise SensorValidationError(
                    "Out-of-order observation"
                )

            record.last_sequence = (
                observation.sequence_number
            )

            record.last_received_monotonic = (
                time.monotonic()
            )

            record.last_timestamp = (
                observation.timestamp
            )

            record.last_data = safe_data

            record.history.append(
                {
                    "sequence_number": (
                        observation.sequence_number
                    ),
                    "timestamp": (
                        observation.timestamp.isoformat()
                    ),
                    "data": copy.deepcopy(
                        safe_data
                    ),
                }
            )

            warning = not observation.healthy

            if observation.fault_detected:
                record.fault_latched = True

                if (
                    "sensor_reported_fault"
                    not in record.fault_reasons
                ):
                    record.fault_reasons.append(
                        "sensor_reported_fault"
                    )

                self._audit(
                    record,
                    SensorEventType.FAULT_DETECTED,
                    "Sensor reported fault",
                )

            value = safe_data.get("value")

            if (
                value is not None
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                measured = finite(
                    value,
                    "sensor value",
                )

                d = record.definition

                if (
                    d.minimum_value is not None
                    and measured < d.minimum_value
                ):
                    record.fault_latched = True
                    record.fault_reasons.append(
                        "below_minimum"
                    )

                if (
                    d.maximum_value is not None
                    and measured > d.maximum_value
                ):
                    record.fault_latched = True
                    record.fault_reasons.append(
                        "above_maximum"
                    )

                if (
                    d.warning_minimum is not None
                    and measured < d.warning_minimum
                ):
                    warning = True

                if (
                    d.warning_maximum is not None
                    and measured > d.warning_maximum
                ):
                    warning = True

            # Recompute current warning for this observation.
            record.health = (
                SensorHealth.WARNING
                if warning
                else SensorHealth.HEALTHY
            )

            self._evaluate(record)

            self._audit(
                record,
                SensorEventType.OBSERVATION_RECEIVED,
                "Sensor observation accepted",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # CLEAR FAULT
    # --------------------------------------------------------

    def clear_fault(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
        authorized: bool,
        independently_verified: bool,
    ) -> dict[str, Any]:
        if (
            authorized is not True
            or independently_verified is not True
        ):
            raise SensorPermissionDenied(
                "Verified fault clearance required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                sensor_id,
            )

            if record.enabled:
                raise SensorConflict(
                    "Disable sensor before fault clearance"
                )

            record.fault_latched = False
            record.fault_reasons.clear()

            self._audit(
                record,
                SensorEventType.FAULT_CLEARED,
                "Sensor fault cleared in simulation",
            )

            self._evaluate(record)

            return record.to_dict()

    # --------------------------------------------------------
    # GET SENSOR
    # --------------------------------------------------------

    def get_sensor(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                sensor_id,
            )

            self._evaluate(record)

            return record.to_dict()

    # --------------------------------------------------------
    # SENSOR HISTORY
    # --------------------------------------------------------

    def get_history(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
        ):
            raise SensorValidationError(
                "Invalid history limit"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                sensor_id,
            )

            return copy.deepcopy(
                list(record.history)[-limit:]
            )

    # --------------------------------------------------------
    # LIST SENSORS
    # --------------------------------------------------------

    def list_sensors(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        sensor_type: SensorType | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        if robot_id is not None:
            validate_id(
                robot_id,
                "robot_id",
            )

        with self._lock:
            result = []

            for record in self._sensors.values():
                d = record.definition

                if d.tenant_id != tenant_id:
                    continue

                if (
                    robot_id is not None
                    and d.robot_id != robot_id
                ):
                    continue

                if (
                    sensor_type is not None
                    and d.sensor_type != sensor_type
                ):
                    continue

                self._evaluate(record)

                result.append(
                    record.to_dict()
                )

            return sorted(
                result,
                key=lambda item: (
                    item["robot_id"],
                    item["sensor_id"],
                ),
            )

    # --------------------------------------------------------
    # ROBOT SENSOR HEALTH
    # --------------------------------------------------------

    def evaluate_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        sensors = self.list_sensors(
            tenant_id=tenant_id,
            robot_id=robot_id,
        )

        counts = {
            status.value: 0
            for status in SensorHealth
        }

        for sensor in sensors:
            counts[
                sensor["health"]
            ] += 1

        critical = counts[
            SensorHealth.CRITICAL.value
        ]

        stale = counts[
            SensorHealth.STALE.value
        ]

        warning = counts[
            SensorHealth.WARNING.value
        ]

        disabled = counts[
            SensorHealth.DISABLED.value
        ]

        if not sensors:
            overall = "unknown"
        elif critical:
            overall = "critical"
        elif stale or warning or disabled:
            overall = "degraded"
        else:
            overall = "healthy"

        return {
            "tenant_id": tenant_id,
            "robot_id": robot_id,
            "total_sensors": len(sensors),
            "overall_health": overall,
            "health_counts": counts,
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str,
    ) -> dict[str, Any]:
        sensors = self.list_sensors(
            tenant_id=tenant_id
        )

        type_counts = {
            sensor_type.value: 0
            for sensor_type in SensorType
        }

        for sensor in sensors:
            type_counts[
                sensor["sensor_type"]
            ] += 1

        return {
            "tenant_id": tenant_id,
            "total_sensors": len(sensors),
            "active_sensors": sum(
                sensor["status"]
                == SensorStatus.ACTIVE.value
                for sensor in sensors
            ),
            "faulted_sensors": sum(
                sensor["fault_latched"]
                for sensor in sensors
            ),
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
    ) -> Path:
        validate_id(
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
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "sensors": self.list_sensors(
                    tenant_id=tenant_id
                ),
                "audit_events": [
                    event.to_dict()
                    for event in self._audit_events
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
    manager = RoboticsSensorManager()

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"

    manager.register_sensor(
        SensorDefinition(
            tenant_id=tenant_id,
            robot_id=robot_id,
            sensor_id="temperature_001",
            name="Motor Temperature Sensor",
            sensor_type=SensorType.TEMPERATURE,
            unit="celsius",
            minimum_value=-20.0,
            maximum_value=100.0,
            warning_maximum=70.0,
        )
    )

    manager.register_sensor(
        SensorDefinition(
            tenant_id=tenant_id,
            robot_id=robot_id,
            sensor_id="imu_001",
            name="Main IMU",
            sensor_type=SensorType.IMU,
            unit="m_per_s2",
            calibration_required=True,
        )
    )

    manager.update_calibration(
        tenant_id=tenant_id,
        robot_id=robot_id,
        sensor_id="imu_001",
        calibration=SensorCalibration(
            calibration_id="imu_calibration_001",
            version="1.0.0",
            approved=True,
        ),
        authorized=True,
        independently_verified=True,
    )

    for sensor_id in (
        "temperature_001",
        "imu_001",
    ):
        manager.activate_sensor(
            tenant_id=tenant_id,
            robot_id=robot_id,
            sensor_id=sensor_id,
            authorized=True,
        )

    manager.ingest_observation(
        SensorObservation(
            tenant_id=tenant_id,
            robot_id=robot_id,
            sensor_id="temperature_001",
            sequence_number=1,
            data={
                "value": 38.5,
            },
        )
    )

    manager.ingest_observation(
        SensorObservation(
            tenant_id=tenant_id,
            robot_id=robot_id,
            sensor_id="imu_001",
            sequence_number=1,
            data={
                "acceleration": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 9.81,
                },
                "angular_velocity": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.02,
                },
            },
        )
    )

    print(
        "Robot sensor health:",
        json.dumps(
            manager.evaluate_robot(
                tenant_id=tenant_id,
                robot_id=robot_id,
            ),
            indent=2,
        ),
    )

    report = manager.export_json(
        "data/robotics/sensor_manager_report.json",
        tenant_id=tenant_id,
    )

    print("Report:", report)


if __name__ == "__main__":
    main()

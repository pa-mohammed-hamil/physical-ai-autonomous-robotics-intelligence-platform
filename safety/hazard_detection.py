"""Hazard Detection module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/security/hazard_detection.py

Python 3.11+
Standard library only.

Features:
    - Multi-sensor hazard detection
    - Fire and smoke detection
    - Gas and thermal hazard assessment
    - Electrical and mechanical hazards
    - Human proximity risk
    - Falling-object detection
    - Sensor-health validation
    - Confidence-based classification
    - Hazard severity and risk scores
    - Emergency advisory integration payloads
    - Tenant-scoped hazard records
    - Audit events and JSON export

SAFETY:
    Simulation / supervisory monitoring only.
    Not a certified fire, gas, machine-guarding,
    or personnel protection system.

    No physical robot commands are executed.
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

class HazardDetectionError(Exception):
    pass


class HazardValidationError(HazardDetectionError):
    pass


class HazardNotFoundError(HazardDetectionError):
    pass


# ============================================================
# ENUMS
# ============================================================

class HazardType(str, Enum):
    FIRE = "fire"
    SMOKE = "smoke"
    GAS_LEAK = "gas_leak"
    OVERHEATING = "overheating"
    BATTERY_THERMAL = "battery_thermal"
    ELECTRICAL_FAULT = "electrical_fault"
    MECHANICAL_FAULT = "mechanical_fault"
    HUMAN_PROXIMITY = "human_proximity"
    FALLING_OBJECT = "falling_object"
    COLLISION_RISK = "collision_risk"
    FLOODING = "flooding"
    SENSOR_FAILURE = "sensor_failure"


class HazardSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class HazardStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class SensorType(str, Enum):
    TEMPERATURE = "temperature"
    BATTERY_TEMPERATURE = "battery_temperature"
    SMOKE = "smoke"
    FLAME = "flame"
    GAS = "gas"
    CURRENT = "current"
    VIBRATION = "vibration"
    HUMAN_DISTANCE = "human_distance"
    COLLISION_RISK = "collision_risk"
    WATER_LEVEL = "water_level"
    FALLING_OBJECT = "falling_object"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def validate_identifier(
    value: str,
    field_name: str,
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
        raise HazardValidationError(
            f"Invalid {field_name}"
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
        raise HazardValidationError(
            f"{name} must be finite"
        )

    return float(value)


def validate_timestamp(
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise HazardValidationError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class HazardDetectionConfig:
    maximum_temperature_c: float = 75.0
    maximum_battery_temperature_c: float = 60.0

    smoke_threshold: float = 0.7
    flame_threshold: float = 0.7
    gas_threshold: float = 0.7

    maximum_current_amperes: float = 100.0
    maximum_vibration_mm_s: float = 15.0

    minimum_human_distance_meters: float = 1.5

    maximum_collision_risk_score: float = 60.0
    maximum_water_level_meters: float = 0.1

    falling_object_threshold: float = 0.7

    minimum_confidence: float = 0.6

    maximum_sensor_age_seconds: float = 3.0

    maximum_records: int = 100000
    maximum_audit_events: int = 20000

    def __post_init__(self) -> None:
        numeric_fields = (
            "maximum_temperature_c",
            "maximum_battery_temperature_c",
            "smoke_threshold",
            "flame_threshold",
            "gas_threshold",
            "maximum_current_amperes",
            "maximum_vibration_mm_s",
            "minimum_human_distance_meters",
            "maximum_collision_risk_score",
            "maximum_water_level_meters",
            "falling_object_threshold",
            "minimum_confidence",
            "maximum_sensor_age_seconds",
        )

        for name in numeric_fields:
            finite_number(
                getattr(self, name),
                name,
            )

        for name in (
            "smoke_threshold",
            "flame_threshold",
            "gas_threshold",
            "falling_object_threshold",
            "minimum_confidence",
        ):
            value = getattr(self, name)

            if not (0 <= value <= 1):
                raise HazardValidationError(
                    f"{name} must be between 0 and 1"
                )

        if not (
            0 <= self.maximum_collision_risk_score <= 100
        ):
            raise HazardValidationError(
                "Collision risk threshold out of range"
            )

        if self.maximum_sensor_age_seconds <= 0:
            raise HazardValidationError(
                "Sensor age limit must be positive"
            )

        for name in (
            "maximum_records",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise HazardValidationError(
                    f"{name} must be positive"
                )


# ============================================================
# SENSOR READING
# ============================================================

@dataclass(frozen=True)
class HazardSensorReading:
    sensor_id: str
    sensor_type: SensorType

    value: float
    confidence: float = 1.0

    healthy: bool = True

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.sensor_id,
            "sensor_id",
        )

        finite_number(
            self.value,
            "value",
        )

        confidence = finite_number(
            self.confidence,
            "confidence",
        )

        if not (0 <= confidence <= 1):
            raise HazardValidationError(
                "Confidence must be between 0 and 1"
            )

        validate_timestamp(
            self.timestamp
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "value": self.value,
            "confidence": self.confidence,
            "healthy": self.healthy,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================
# HAZARD OBSERVATION
# ============================================================

@dataclass(frozen=True)
class HazardObservation:
    tenant_id: str
    robot_id: str

    readings: tuple[
        HazardSensorReading,
        ...
    ]

    location: str | None = None

    observed_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.robot_id,
            "robot_id",
        )

        if not self.readings:
            raise HazardValidationError(
                "At least one sensor reading is required"
            )

        if (
            self.location is not None
            and len(self.location) > 256
        ):
            raise HazardValidationError(
                "Location description too long"
            )

        validate_timestamp(
            self.observed_at
        )


# ============================================================
# HAZARD FINDING
# ============================================================

@dataclass(frozen=True)
class HazardFinding:
    hazard_type: HazardType
    severity: HazardSeverity

    risk_score: float
    confidence: float

    description: str

    sensor_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hazard_type": self.hazard_type.value,
            "severity": self.severity.value,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "description": self.description,
            "sensor_ids": list(self.sensor_ids),
        }


# ============================================================
# HAZARD RECORD
# ============================================================

@dataclass
class HazardRecord:
    hazard_id: str

    tenant_id: str
    robot_id: str

    finding: HazardFinding

    status: HazardStatus = HazardStatus.ACTIVE

    detected_at: str = field(
        default_factory=utc_iso
    )

    acknowledged_by: str | None = None
    acknowledged_at: str | None = None

    resolved_by: str | None = None
    resolved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hazard_id": self.hazard_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "finding": self.finding.to_dict(),
            "status": self.status.value,
            "detected_at": self.detected_at,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at,
        }


# ============================================================
# HAZARD DETECTION ENGINE
# ============================================================

class HazardDetectionEngine:

    def __init__(
        self,
        config: HazardDetectionConfig,
    ) -> None:
        self.config = config

    @staticmethod
    def severity_from_score(
        score: float,
    ) -> HazardSeverity:
        if score >= 90:
            return HazardSeverity.CRITICAL

        if score >= 70:
            return HazardSeverity.HIGH

        if score >= 40:
            return HazardSeverity.MODERATE

        return HazardSeverity.LOW

    def detect(
        self,
        observation: HazardObservation,
    ) -> list[HazardFinding]:
        findings: list[HazardFinding] = []

        now = utc_now()

        for reading in observation.readings:
            age = (
                now
                - validate_timestamp(
                    reading.timestamp
                )
            ).total_seconds()

            if (
                not reading.healthy
                or age < -1.0
                or age
                > self.config.maximum_sensor_age_seconds
            ):
                findings.append(
                    HazardFinding(
                        hazard_type=HazardType.SENSOR_FAILURE,
                        severity=HazardSeverity.HIGH,
                        risk_score=85.0,
                        confidence=1.0,
                        description=(
                            "Sensor unavailable, unhealthy "
                            "or telemetry stale"
                        ),
                        sensor_ids=(
                            reading.sensor_id,
                        ),
                    )
                )

                continue

            if (
                reading.confidence
                < self.config.minimum_confidence
            ):
                continue

            hazard: HazardType | None = None
            score = 0.0
            description = ""

            value = reading.value
            sensor = reading.sensor_type

            if (
                sensor == SensorType.TEMPERATURE
                and value
                >= self.config.maximum_temperature_c
            ):
                hazard = HazardType.OVERHEATING
                score = min(
                    100.0,
                    75.0
                    + (
                        value
                        - self.config.maximum_temperature_c
                    ),
                )
                description = (
                    "Temperature exceeds configured limit"
                )

            elif (
                sensor == SensorType.BATTERY_TEMPERATURE
                and value
                >= self.config.maximum_battery_temperature_c
            ):
                hazard = HazardType.BATTERY_THERMAL
                score = min(
                    100.0,
                    90.0
                    + (
                        value
                        - self.config.maximum_battery_temperature_c
                    ),
                )
                description = (
                    "Battery thermal threshold exceeded"
                )

            elif (
                sensor == SensorType.SMOKE
                and value
                >= self.config.smoke_threshold
            ):
                hazard = HazardType.SMOKE
                score = min(
                    100.0,
                    value * 100.0,
                )
                description = "Smoke signal detected"

            elif (
                sensor == SensorType.FLAME
                and value
                >= self.config.flame_threshold
            ):
                hazard = HazardType.FIRE
                score = min(
                    100.0,
                    value * 100.0,
                )
                description = "Flame signal detected"

            elif (
                sensor == SensorType.GAS
                and value
                >= self.config.gas_threshold
            ):
                hazard = HazardType.GAS_LEAK
                score = min(
                    100.0,
                    value * 100.0,
                )
                description = "Gas sensor threshold exceeded"

            elif (
                sensor == SensorType.CURRENT
                and value
                >= self.config.maximum_current_amperes
            ):
                hazard = HazardType.ELECTRICAL_FAULT
                score = 85.0
                description = (
                    "Electrical current exceeds limit"
                )

            elif (
                sensor == SensorType.VIBRATION
                and value
                >= self.config.maximum_vibration_mm_s
            ):
                hazard = HazardType.MECHANICAL_FAULT
                score = 75.0
                description = (
                    "Abnormal mechanical vibration"
                )

            elif (
                sensor == SensorType.HUMAN_DISTANCE
                and value
                <= self.config.minimum_human_distance_meters
            ):
                hazard = HazardType.HUMAN_PROXIMITY
                score = 90.0
                description = (
                    "Human detected within configured "
                    "proximity threshold"
                )

            elif (
                sensor == SensorType.COLLISION_RISK
                and value
                >= self.config.maximum_collision_risk_score
            ):
                hazard = HazardType.COLLISION_RISK
                score = min(
                    100.0,
                    value,
                )
                description = (
                    "Collision risk threshold exceeded"
                )

            elif (
                sensor == SensorType.WATER_LEVEL
                and value
                >= self.config.maximum_water_level_meters
            ):
                hazard = HazardType.FLOODING
                score = 85.0
                description = (
                    "Water level exceeds configured limit"
                )

            elif (
                sensor == SensorType.FALLING_OBJECT
                and value
                >= self.config.falling_object_threshold
            ):
                hazard = HazardType.FALLING_OBJECT
                score = 95.0
                description = (
                    "Potential falling object detected"
                )

            if hazard is not None:
                findings.append(
                    HazardFinding(
                        hazard_type=hazard,
                        severity=self.severity_from_score(
                            score
                        ),
                        risk_score=score,
                        confidence=reading.confidence,
                        description=description,
                        sensor_ids=(
                            reading.sensor_id,
                        ),
                    )
                )

        return findings


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class HazardAuditEvent:
    event_id: str
    timestamp: str

    tenant_id: str
    robot_id: str

    action: str
    hazard_id: str | None

    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "action": self.action,
            "hazard_id": self.hazard_id,
            "details": self.details,
        }


# ============================================================
# CENTRAL HAZARD DETECTION SERVICE
# ============================================================

class RoboticsHazardDetectionService:

    def __init__(
        self,
        config: HazardDetectionConfig | None = None,
    ) -> None:
        self.config = (
            config or HazardDetectionConfig()
        )

        self.engine = HazardDetectionEngine(
            self.config
        )

        self.records: dict[
            str,
            HazardRecord
        ] = {}

        self.audit_events: list[
            HazardAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        action: str,
        hazard_id: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.audit_events.append(
            HazardAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_iso(),
                tenant_id=tenant_id,
                robot_id=robot_id,
                action=action,
                hazard_id=hazard_id,
                details=dict(details or {}),
            )
        )

        if (
            len(self.audit_events)
            > self.config.maximum_audit_events
        ):
            self.audit_events.pop(0)

    # --------------------------------------------------------
    # PROCESS OBSERVATION
    # --------------------------------------------------------

    def process_observation(
        self,
        observation: HazardObservation,
    ) -> list[HazardRecord]:
        findings = self.engine.detect(
            observation
        )

        created: list[HazardRecord] = []

        with self._lock:
            if (
                len(self.records) + len(findings)
                > self.config.maximum_records
            ):
                raise HazardDetectionError(
                    "Hazard record capacity reached"
                )

            for finding in findings:
                record = HazardRecord(
                    hazard_id=str(uuid.uuid4()),
                    tenant_id=observation.tenant_id,
                    robot_id=observation.robot_id,
                    finding=finding,
                )

                self.records[
                    record.hazard_id
                ] = record

                created.append(
                    record
                )

                self._audit(
                    tenant_id=observation.tenant_id,
                    robot_id=observation.robot_id,
                    action="hazard_detected",
                    hazard_id=record.hazard_id,
                    details={
                        "hazard_type": (
                            finding.hazard_type.value
                        ),
                        "severity": (
                            finding.severity.value
                        ),
                        "risk_score": (
                            finding.risk_score
                        ),
                    },
                )

        return created

    # --------------------------------------------------------
    # GET RECORD
    # --------------------------------------------------------

    def get_record(
        self,
        *,
        tenant_id: str,
        hazard_id: str,
    ) -> HazardRecord:
        with self._lock:
            record = self.records.get(
                hazard_id
            )

            if (
                record is None
                or record.tenant_id != tenant_id
            ):
                raise HazardNotFoundError(
                    "Hazard record not found"
                )

            return record

    # --------------------------------------------------------
    # ACKNOWLEDGE
    # --------------------------------------------------------

    def acknowledge(
        self,
        *,
        tenant_id: str,
        hazard_id: str,
        operator_id: str,
    ) -> HazardRecord:
        validate_identifier(
            operator_id,
            "operator_id",
        )

        with self._lock:
            record = self.get_record(
                tenant_id=tenant_id,
                hazard_id=hazard_id,
            )

            if record.status != HazardStatus.ACTIVE:
                raise HazardValidationError(
                    "Only active hazards can be acknowledged"
                )

            record.status = (
                HazardStatus.ACKNOWLEDGED
            )

            record.acknowledged_by = operator_id
            record.acknowledged_at = utc_iso()

            self._audit(
                tenant_id=record.tenant_id,
                robot_id=record.robot_id,
                action="hazard_acknowledged",
                hazard_id=hazard_id,
                details={
                    "operator_id": operator_id
                },
            )

            return record

    # --------------------------------------------------------
    # RESOLVE
    # --------------------------------------------------------

    def resolve(
        self,
        *,
        tenant_id: str,
        hazard_id: str,
        reviewer_id: str,
        verified_clear: bool,
    ) -> HazardRecord:
        validate_identifier(
            reviewer_id,
            "reviewer_id",
        )

        if not verified_clear:
            raise HazardValidationError(
                "Independent hazard-clear verification required"
            )

        with self._lock:
            record = self.get_record(
                tenant_id=tenant_id,
                hazard_id=hazard_id,
            )

            if record.status != (
                HazardStatus.ACKNOWLEDGED
            ):
                raise HazardValidationError(
                    "Hazard must be acknowledged first"
                )

            record.status = HazardStatus.RESOLVED
            record.resolved_by = reviewer_id
            record.resolved_at = utc_iso()

            self._audit(
                tenant_id=record.tenant_id,
                robot_id=record.robot_id,
                action="hazard_resolved",
                hazard_id=hazard_id,
                details={
                    "reviewer_id": reviewer_id
                },
            )

            return record

    # --------------------------------------------------------
    # LIST RECORDS
    # --------------------------------------------------------

    def list_records(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        status: HazardStatus | None = None,
    ) -> list[HazardRecord]:
        with self._lock:
            results = [
                record
                for record in self.records.values()
                if record.tenant_id == tenant_id
            ]

            if robot_id is not None:
                results = [
                    record
                    for record in results
                    if record.robot_id == robot_id
                ]

            if status is not None:
                results = [
                    record
                    for record in results
                    if record.status == status
                ]

            return results

    # --------------------------------------------------------
    # EMERGENCY ADVISORY PAYLOAD
    # --------------------------------------------------------

    def emergency_advisory_payload(
        self,
        *,
        tenant_id: str,
        hazard_id: str,
    ) -> dict[str, Any]:
        record = self.get_record(
            tenant_id=tenant_id,
            hazard_id=hazard_id,
        )

        return {
            "tenant_id": record.tenant_id,
            "robot_id": record.robot_id,
            "emergency_type": (
                record.finding.hazard_type.value
            ),
            "risk_score": (
                record.finding.risk_score
            ),
            "description": (
                record.finding.description
            ),
            "source": "hazard_detection",
            "hazard_id": record.hazard_id,
        }

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            return {
                "total_hazards": len(
                    self.records
                ),
                "active_hazards": sum(
                    record.status == HazardStatus.ACTIVE
                    for record in self.records.values()
                ),
                "critical_hazards": sum(
                    record.finding.severity
                    == HazardSeverity.CRITICAL
                    for record in self.records.values()
                ),
                "resolved_hazards": sum(
                    record.status == HazardStatus.RESOLVED
                    for record in self.records.values()
                ),
                "audit_events": len(
                    self.audit_events
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
                "hazards": [
                    record.to_dict()
                    for record in self.records.values()
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
    service = RoboticsHazardDetectionService()

    observation = HazardObservation(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        location="Warehouse Zone A",
        readings=(
            HazardSensorReading(
                sensor_id="temperature_001",
                sensor_type=SensorType.TEMPERATURE,
                value=92.0,
                confidence=0.99,
            ),
            HazardSensorReading(
                sensor_id="smoke_001",
                sensor_type=SensorType.SMOKE,
                value=0.91,
                confidence=0.96,
            ),
            HazardSensorReading(
                sensor_id="human_distance_001",
                sensor_type=SensorType.HUMAN_DISTANCE,
                value=0.8,
                confidence=0.98,
            ),
            HazardSensorReading(
                sensor_id="battery_temp_001",
                sensor_type=SensorType.BATTERY_TEMPERATURE,
                value=72.0,
                confidence=0.97,
            ),
        ),
    )

    records = service.process_observation(
        observation
    )

    print(
        json.dumps(
            [
                record.to_dict()
                for record in records
            ],
            indent=2,
        )
    )

    print(
        json.dumps(
            service.statistics(),
            indent=2,
        )
    )

    service.export_json(
        "data/security/hazard_detection_report.json"
    )


if __name__ == "__main__":
    main()

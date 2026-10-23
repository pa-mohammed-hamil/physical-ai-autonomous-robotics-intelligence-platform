"""Issue Diagnosis module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/issue_diagnosis.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robot Issue Diagnosis Engine

Python 3.11+
Standard library only.

Features:
    - Robot fault and issue diagnosis
    - Mechanical, electrical, sensor, battery and software faults
    - Telemetry-based diagnostic rules
    - Symptom-based fault identification
    - Multiple possible root causes
    - Evidence-based diagnostic scoring
    - Damage inspection integration
    - Fault severity classification
    - Maintenance recommendations
    - Human review requirements
    - Multi-tenant isolation
    - Thread-safe diagnostic history
    - JSON report export
    - Runnable demonstration

SAFETY:
    This is a simulation and diagnostic decision-support engine.
    It does not issue robot movement or repair commands.
    Diagnostic scores are heuristic, not calibrated probabilities.
    Physical faults and safety-critical conditions require
    independent verification by qualified personnel.
"""

from __future__ import annotations

import json
import math
import re
import threading
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class IssueDiagnosisError(Exception):
    pass


class DiagnosisValidationError(IssueDiagnosisError):
    pass


class DiagnosisNotFound(IssueDiagnosisError):
    pass


class DiagnosisConflict(IssueDiagnosisError):
    pass


class DiagnosisPermissionDenied(IssueDiagnosisError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RobotIssueCategory(str, Enum):
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    SENSOR = "sensor"
    BATTERY = "battery"
    THERMAL = "thermal"
    SOFTWARE = "software"
    NETWORK = "network"
    CONTROL = "control"
    STRUCTURAL = "structural"
    UNKNOWN = "unknown"


class RobotFaultType(str, Enum):
    MOTOR_OVERHEATING = "motor_overheating"
    MOTOR_OVERLOAD = "motor_overload"
    BEARING_WEAR = "bearing_wear"
    GEARBOX_DAMAGE = "gearbox_damage"
    JOINT_MISALIGNMENT = "joint_misalignment"
    STRUCTURAL_DAMAGE = "structural_damage"
    BATTERY_LOW = "battery_low"
    BATTERY_OVERHEATING = "battery_overheating"
    VOLTAGE_ANOMALY = "voltage_anomaly"
    SENSOR_FAILURE = "sensor_failure"
    SENSOR_COMMUNICATION_LOSS = "sensor_communication_loss"
    NETWORK_FAILURE = "network_failure"
    CONTROLLER_FAULT = "controller_fault"
    SOFTWARE_EXCEPTION = "software_exception"
    UNKNOWN = "unknown"


class IssueSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class DiagnosisStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


class MaintenanceAction(str, Enum):
    MONITOR = "monitor"
    INSPECT = "inspect"
    SCHEDULE_MAINTENANCE = "schedule_maintenance"
    PRIORITY_REPAIR = "priority_repair"
    STOP_AND_REVIEW = "stop_and_review"


class DiagnosticEventType(str, Enum):
    DIAGNOSIS_CREATED = "diagnosis_created"
    DIAGNOSIS_REVIEWED = "diagnosis_reviewed"
    DIAGNOSIS_REJECTED = "diagnosis_rejected"


# ============================================================
# UTILITIES
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
        raise DiagnosisValidationError(
            f"Invalid {name}"
        )

    return value


def validate_text(
    value: str,
    name: str,
    maximum_length: int = 2048,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) > maximum_length
    ):
        raise DiagnosisValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise DiagnosisValidationError(
            f"{name} must be finite"
        )

    return float(value)


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, value))


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class IssueDiagnosisConfig:
    maximum_diagnoses: int = 100000
    maximum_audit_events: int = 20000
    maximum_symptoms: int = 100
    maximum_candidates: int = 20

    motor_warning_temperature_c: float = 70.0
    motor_critical_temperature_c: float = 90.0

    battery_warning_temperature_c: float = 45.0
    battery_critical_temperature_c: float = 60.0

    battery_low_soc_percent: float = 20.0
    battery_critical_soc_percent: float = 10.0

    motor_warning_current_a: float = 20.0
    motor_critical_current_a: float = 35.0

    vibration_warning_mm_s: float = 5.0
    vibration_critical_mm_s: float = 12.0

    minimum_diagnostic_score: float = 0.25

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_diagnoses",
            "maximum_audit_events",
            "maximum_symptoms",
            "maximum_candidates",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise DiagnosisValidationError(
                    f"{name} must be positive"
                )

        threshold_pairs = (
            (
                self.motor_warning_temperature_c,
                self.motor_critical_temperature_c,
            ),
            (
                self.battery_warning_temperature_c,
                self.battery_critical_temperature_c,
            ),
            (
                self.motor_warning_current_a,
                self.motor_critical_current_a,
            ),
            (
                self.vibration_warning_mm_s,
                self.vibration_critical_mm_s,
            ),
        )

        for warning, critical in threshold_pairs:
            if not (
                finite(warning, "warning") >= 0
                and finite(critical, "critical") > warning
            ):
                raise DiagnosisValidationError(
                    "Invalid warning/critical thresholds"
                )

        if not (
            0 <= self.battery_critical_soc_percent
            < self.battery_low_soc_percent
            <= 100
        ):
            raise DiagnosisValidationError(
                "Invalid battery SOC thresholds"
            )

        if not (
            0 <= self.minimum_diagnostic_score <= 1
        ):
            raise DiagnosisValidationError(
                "Invalid minimum diagnostic score"
            )

        if self.simulation_only is not True:
            raise DiagnosisValidationError(
                "Only simulation mode is supported"
            )


# ============================================================
# TELEMETRY
# ============================================================

@dataclass(frozen=True)
class RobotDiagnosticTelemetry:
    motor_temperature_c: float | None = None
    motor_current_a: float | None = None
    vibration_mm_s: float | None = None

    battery_soc_percent: float | None = None
    battery_temperature_c: float | None = None
    battery_voltage_v: float | None = None

    sensor_healthy: bool | None = None
    network_connected: bool | None = None
    controller_healthy: bool | None = None

    software_error_count: int = 0

    def __post_init__(self) -> None:
        numeric_fields = (
            "motor_temperature_c",
            "motor_current_a",
            "vibration_mm_s",
            "battery_soc_percent",
            "battery_temperature_c",
            "battery_voltage_v",
        )

        for name in numeric_fields:
            value = getattr(self, name)

            if value is not None:
                finite(value, name)

        for name in (
            "motor_current_a",
            "vibration_mm_s",
            "battery_voltage_v",
        ):
            value = getattr(self, name)

            if value is not None and value < 0:
                raise DiagnosisValidationError(
                    f"{name} cannot be negative"
                )

        if (
            self.battery_soc_percent is not None
            and not 0 <= self.battery_soc_percent <= 100
        ):
            raise DiagnosisValidationError(
                "Battery SOC must be between 0 and 100"
            )

        for name in (
            "sensor_healthy",
            "network_connected",
            "controller_healthy",
        ):
            value = getattr(self, name)

            if (
                value is not None
                and not isinstance(value, bool)
            ):
                raise DiagnosisValidationError(
                    f"{name} must be boolean or None"
                )

        if (
            isinstance(self.software_error_count, bool)
            or not isinstance(
                self.software_error_count,
                int,
            )
            or self.software_error_count < 0
        ):
            raise DiagnosisValidationError(
                "Invalid software error count"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


# ============================================================
# SYMPTOM
# ============================================================

@dataclass(frozen=True)
class RobotSymptom:
    symptom_id: str
    name: str
    description: str = ""
    confidence: float = 1.0

    def __post_init__(self) -> None:
        validate_id(self.symptom_id, "symptom_id")
        validate_id(self.name, "name")

        validate_text(
            self.description,
            "description",
        )

        if not 0 <= finite(
            self.confidence,
            "confidence",
        ) <= 1:
            raise DiagnosisValidationError(
                "Confidence must be between 0 and 1"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symptom_id": self.symptom_id,
            "name": self.name,
            "description": self.description,
            "confidence": self.confidence,
        }


# ============================================================
# DAMAGE EVIDENCE
# ============================================================

@dataclass(frozen=True)
class RobotDamageEvidence:
    damage_type: str
    severity: str
    confidence: float

    component_id: str = ""

    def __post_init__(self) -> None:
        validate_id(
            self.damage_type,
            "damage_type",
        )

        validate_id(
            self.severity,
            "severity",
        )

        if self.component_id:
            validate_id(
                self.component_id,
                "component_id",
            )

        if not 0 <= finite(
            self.confidence,
            "confidence",
        ) <= 1:
            raise DiagnosisValidationError(
                "Invalid damage confidence"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "damage_type": self.damage_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "component_id": self.component_id,
        }


# ============================================================
# DIAGNOSIS REQUEST
# ============================================================

@dataclass(frozen=True)
class IssueDiagnosisRequest:
    tenant_id: str
    robot_id: str

    telemetry: RobotDiagnosticTelemetry

    symptoms: tuple[RobotSymptom, ...] = ()
    damage_evidence: tuple[RobotDamageEvidence, ...] = ()

    inspection_reference: str = ""

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        if not isinstance(
            self.telemetry,
            RobotDiagnosticTelemetry,
        ):
            raise DiagnosisValidationError(
                "Invalid telemetry"
            )

        if not isinstance(self.symptoms, tuple):
            raise DiagnosisValidationError(
                "Symptoms must be a tuple"
            )

        if not isinstance(
            self.damage_evidence,
            tuple,
        ):
            raise DiagnosisValidationError(
                "Damage evidence must be a tuple"
            )

        ids = set()

        for symptom in self.symptoms:
            if not isinstance(
                symptom,
                RobotSymptom,
            ):
                raise DiagnosisValidationError(
                    "Invalid symptom"
                )

            if symptom.symptom_id in ids:
                raise DiagnosisValidationError(
                    "Duplicate symptom ID"
                )

            ids.add(symptom.symptom_id)

        for evidence in self.damage_evidence:
            if not isinstance(
                evidence,
                RobotDamageEvidence,
            ):
                raise DiagnosisValidationError(
                    "Invalid damage evidence"
                )

        validate_text(
            self.inspection_reference,
            "inspection_reference",
            1024,
        )


# ============================================================
# FAULT KNOWLEDGE BASE
# ============================================================

@dataclass(frozen=True)
class FaultDefinition:
    fault_type: RobotFaultType
    category: RobotIssueCategory
    severity: IssueSeverity
    recommendation: MaintenanceAction

    description: str
    possible_causes: tuple[str, ...]


FAULT_KNOWLEDGE_BASE: dict[
    RobotFaultType,
    FaultDefinition,
] = {
    RobotFaultType.MOTOR_OVERHEATING: FaultDefinition(
        fault_type=RobotFaultType.MOTOR_OVERHEATING,
        category=RobotIssueCategory.THERMAL,
        severity=IssueSeverity.HIGH,
        recommendation=MaintenanceAction.PRIORITY_REPAIR,
        description="Motor temperature above configured threshold",
        possible_causes=(
            "Excessive mechanical load",
            "Insufficient cooling",
            "Bearing friction",
            "Motor electrical fault",
        ),
    ),
    RobotFaultType.MOTOR_OVERLOAD: FaultDefinition(
        fault_type=RobotFaultType.MOTOR_OVERLOAD,
        category=RobotIssueCategory.ELECTRICAL,
        severity=IssueSeverity.HIGH,
        recommendation=MaintenanceAction.PRIORITY_REPAIR,
        description="Elevated motor current",
        possible_causes=(
            "Excessive load",
            "Mechanical obstruction",
            "Actuator friction",
            "Electrical malfunction",
        ),
    ),
    RobotFaultType.BEARING_WEAR: FaultDefinition(
        fault_type=RobotFaultType.BEARING_WEAR,
        category=RobotIssueCategory.MECHANICAL,
        severity=IssueSeverity.MODERATE,
        recommendation=MaintenanceAction.SCHEDULE_MAINTENANCE,
        description="Vibration or symptoms consistent with bearing wear",
        possible_causes=(
            "Bearing degradation",
            "Lubrication deficiency",
            "Mechanical imbalance",
        ),
    ),
    RobotFaultType.GEARBOX_DAMAGE: FaultDefinition(
        fault_type=RobotFaultType.GEARBOX_DAMAGE,
        category=RobotIssueCategory.MECHANICAL,
        severity=IssueSeverity.HIGH,
        recommendation=MaintenanceAction.PRIORITY_REPAIR,
        description="Possible gearbox damage",
        possible_causes=(
            "Gear tooth wear",
            "Lubrication problems",
            "Mechanical overload",
        ),
    ),
    RobotFaultType.JOINT_MISALIGNMENT: FaultDefinition(
        fault_type=RobotFaultType.JOINT_MISALIGNMENT,
        category=RobotIssueCategory.MECHANICAL,
        severity=IssueSeverity.HIGH,
        recommendation=MaintenanceAction.INSPECT,
        description="Possible joint alignment problem",
        possible_causes=(
            "Mechanical deformation",
            "Calibration error",
            "Loose mounting hardware",
        ),
    ),
    RobotFaultType.STRUCTURAL_DAMAGE: FaultDefinition(
        fault_type=RobotFaultType.STRUCTURAL_DAMAGE,
        category=RobotIssueCategory.STRUCTURAL,
        severity=IssueSeverity.CRITICAL,
        recommendation=MaintenanceAction.STOP_AND_REVIEW,
        description="Potential structural damage",
        possible_causes=(
            "Impact",
            "Fatigue",
            "Corrosion",
            "Overloading",
        ),
    ),
    RobotFaultType.BATTERY_LOW: FaultDefinition(
        fault_type=RobotFaultType.BATTERY_LOW,
        category=RobotIssueCategory.BATTERY,
        severity=IssueSeverity.MODERATE,
        recommendation=MaintenanceAction.INSPECT,
        description="Battery state of charge is low",
        possible_causes=(
            "Normal discharge",
            "Charging interruption",
            "Battery degradation",
        ),
    ),
    RobotFaultType.BATTERY_OVERHEATING: FaultDefinition(
        fault_type=RobotFaultType.BATTERY_OVERHEATING,
        category=RobotIssueCategory.THERMAL,
        severity=IssueSeverity.CRITICAL,
        recommendation=MaintenanceAction.STOP_AND_REVIEW,
        description="Battery temperature above threshold",
        possible_causes=(
            "Thermal management fault",
            "Excessive charging/discharging",
            "Internal battery fault",
        ),
    ),
    RobotFaultType.VOLTAGE_ANOMALY: FaultDefinition(
        fault_type=RobotFaultType.VOLTAGE_ANOMALY,
        category=RobotIssueCategory.ELECTRICAL,
        severity=IssueSeverity.HIGH,
        recommendation=MaintenanceAction.INSPECT,
        description="Possible battery voltage anomaly",
        possible_causes=(
            "Power supply problem",
            "Battery degradation",
            "Loose electrical connection",
        ),
    ),
    RobotFaultType.SENSOR_FAILURE: FaultDefinition(
        fault_type=RobotFaultType.SENSOR_FAILURE,
        category=RobotIssueCategory.SENSOR,
        severity=IssueSeverity.HIGH,
        recommendation=MaintenanceAction.INSPECT,
        description="Sensor health check failed",
        possible_causes=(
            "Sensor hardware fault",
            "Power failure",
            "Calibration problem",
        ),
    ),
    RobotFaultType.SENSOR_COMMUNICATION_LOSS: FaultDefinition(
        fault_type=RobotFaultType.SENSOR_COMMUNICATION_LOSS,
        category=RobotIssueCategory.NETWORK,
        severity=IssueSeverity.HIGH,
        recommendation=MaintenanceAction.INSPECT,
        description="Sensor communication interruption",
        possible_causes=(
            "Bus communication failure",
            "Network issue",
            "Disconnected cable",
        ),
    ),
    RobotFaultType.NETWORK_FAILURE: FaultDefinition(
        fault_type=RobotFaultType.NETWORK_FAILURE,
        category=RobotIssueCategory.NETWORK,
        severity=IssueSeverity.HIGH,
        recommendation=MaintenanceAction.INSPECT,
        description="Robot network disconnected",
        possible_causes=(
            "Network outage",
            "Router or switch fault",
            "Wireless interference",
        ),
    ),
    RobotFaultType.CONTROLLER_FAULT: FaultDefinition(
        fault_type=RobotFaultType.CONTROLLER_FAULT,
        category=RobotIssueCategory.CONTROL,
        severity=IssueSeverity.CRITICAL,
        recommendation=MaintenanceAction.STOP_AND_REVIEW,
        description="Robot controller health check failed",
        possible_causes=(
            "Controller hardware failure",
            "Firmware malfunction",
            "Power supply problem",
        ),
    ),
    RobotFaultType.SOFTWARE_EXCEPTION: FaultDefinition(
        fault_type=RobotFaultType.SOFTWARE_EXCEPTION,
        category=RobotIssueCategory.SOFTWARE,
        severity=IssueSeverity.MODERATE,
        recommendation=MaintenanceAction.INSPECT,
        description="Software errors detected",
        possible_causes=(
            "Application defect",
            "Resource exhaustion",
            "Configuration error",
        ),
    ),
}


# ============================================================
# DIAGNOSTIC CANDIDATE
# ============================================================

@dataclass(frozen=True)
class DiagnosticCandidate:
    fault_type: RobotFaultType
    category: RobotIssueCategory
    severity: IssueSeverity

    diagnostic_score: float

    description: str
    evidence: tuple[str, ...]
    possible_causes: tuple[str, ...]

    recommendation: MaintenanceAction

    def to_dict(self) -> dict[str, Any]:
        return {
            "fault_type": self.fault_type.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "diagnostic_score": round(
                self.diagnostic_score,
                4,
            ),
            "description": self.description,
            "evidence": list(self.evidence),
            "possible_causes": list(
                self.possible_causes
            ),
            "recommendation": (
                self.recommendation.value
            ),
        }


# ============================================================
# DIAGNOSIS RECORD
# ============================================================

@dataclass
class IssueDiagnosisRecord:
    diagnosis_id: str
    request: IssueDiagnosisRequest

    candidates: list[DiagnosticCandidate]

    overall_severity: IssueSeverity
    recommendation: MaintenanceAction

    human_review_required: bool

    status: DiagnosisStatus = (
        DiagnosisStatus.PENDING_REVIEW
    )

    review_notes: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    reviewed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis_id": self.diagnosis_id,
            "tenant_id": self.request.tenant_id,
            "robot_id": self.request.robot_id,
            "inspection_reference": (
                self.request.inspection_reference
            ),
            "telemetry": (
                self.request.telemetry.to_dict()
            ),
            "symptoms": [
                symptom.to_dict()
                for symptom in self.request.symptoms
            ],
            "damage_evidence": [
                evidence.to_dict()
                for evidence in self.request.damage_evidence
            ],
            "candidate_count": len(self.candidates),
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "overall_severity": (
                self.overall_severity.value
            ),
            "recommendation": (
                self.recommendation.value
            ),
            "human_review_required": (
                self.human_review_required
            ),
            "status": self.status.value,
            "review_notes": self.review_notes,
            "created_at": (
                self.created_at.isoformat()
            ),
            "reviewed_at": (
                self.reviewed_at.isoformat()
                if self.reviewed_at
                else None
            ),
            "simulation_only": True,
        }


# ============================================================
# DIAGNOSIS ENGINE
# ============================================================

class RoboticsIssueDiagnosisEngine:

    SEVERITY_ORDER = {
        IssueSeverity.UNKNOWN: -1,
        IssueSeverity.INFO: 0,
        IssueSeverity.LOW: 1,
        IssueSeverity.MODERATE: 2,
        IssueSeverity.HIGH: 3,
        IssueSeverity.CRITICAL: 4,
    }

    def __init__(
        self,
        config: IssueDiagnosisConfig | None = None,
    ) -> None:
        self.config = (
            config or IssueDiagnosisConfig()
        )

        self._diagnoses: dict[
            str,
            IssueDiagnosisRecord,
        ] = {}

        self._audit_events: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # EVIDENCE COLLECTION
    # --------------------------------------------------------

    def _collect_evidence(
        self,
        request: IssueDiagnosisRequest,
    ) -> dict[
        RobotFaultType,
        list[tuple[float, str]],
    ]:
        telemetry = request.telemetry

        evidence: dict[
            RobotFaultType,
            list[tuple[float, str]],
        ] = {}

        def add(
            fault: RobotFaultType,
            score: float,
            reason: str,
        ) -> None:
            evidence.setdefault(
                fault,
                [],
            ).append(
                (
                    clamp(score),
                    reason,
                )
            )

        # ----------------------------------------------------
        # MOTOR TEMPERATURE
        # ----------------------------------------------------

        temperature = (
            telemetry.motor_temperature_c
        )

        if temperature is not None:
            if (
                temperature
                >= self.config.motor_critical_temperature_c
            ):
                add(
                    RobotFaultType.MOTOR_OVERHEATING,
                    1.0,
                    "Critical motor temperature threshold exceeded",
                )

            elif (
                temperature
                >= self.config.motor_warning_temperature_c
            ):
                add(
                    RobotFaultType.MOTOR_OVERHEATING,
                    0.75,
                    "Motor temperature warning threshold exceeded",
                )

        # ----------------------------------------------------
        # MOTOR CURRENT
        # ----------------------------------------------------

        current = telemetry.motor_current_a

        if current is not None:
            if (
                current
                >= self.config.motor_critical_current_a
            ):
                add(
                    RobotFaultType.MOTOR_OVERLOAD,
                    1.0,
                    "Critical motor current threshold exceeded",
                )

            elif (
                current
                >= self.config.motor_warning_current_a
            ):
                add(
                    RobotFaultType.MOTOR_OVERLOAD,
                    0.75,
                    "Motor current warning threshold exceeded",
                )

        # ----------------------------------------------------
        # VIBRATION
        # ----------------------------------------------------

        vibration = telemetry.vibration_mm_s

        if vibration is not None:
            if (
                vibration
                >= self.config.vibration_critical_mm_s
            ):
                add(
                    RobotFaultType.BEARING_WEAR,
                    0.90,
                    "Critical vibration threshold exceeded",
                )

                add(
                    RobotFaultType.GEARBOX_DAMAGE,
                    0.70,
                    "High vibration may indicate gearbox damage",
                )

            elif (
                vibration
                >= self.config.vibration_warning_mm_s
            ):
                add(
                    RobotFaultType.BEARING_WEAR,
                    0.65,
                    "Elevated vibration detected",
                )

        # ----------------------------------------------------
        # BATTERY SOC
        # ----------------------------------------------------

        soc = telemetry.battery_soc_percent

        if soc is not None:
            if (
                soc
                <= self.config.battery_critical_soc_percent
            ):
                add(
                    RobotFaultType.BATTERY_LOW,
                    0.95,
                    "Battery SOC below critical threshold",
                )

            elif (
                soc
                <= self.config.battery_low_soc_percent
            ):
                add(
                    RobotFaultType.BATTERY_LOW,
                    0.70,
                    "Battery SOC below warning threshold",
                )

        # ----------------------------------------------------
        # BATTERY TEMPERATURE
        # ----------------------------------------------------

        battery_temp = (
            telemetry.battery_temperature_c
        )

        if battery_temp is not None:
            if (
                battery_temp
                >= self.config.battery_critical_temperature_c
            ):
                add(
                    RobotFaultType.BATTERY_OVERHEATING,
                    1.0,
                    "Critical battery temperature threshold exceeded",
                )

            elif (
                battery_temp
                >= self.config.battery_warning_temperature_c
            ):
                add(
                    RobotFaultType.BATTERY_OVERHEATING,
                    0.80,
                    "Battery temperature warning threshold exceeded",
                )

        # ----------------------------------------------------
        # SENSOR HEALTH
        # ----------------------------------------------------

        if telemetry.sensor_healthy is False:
            add(
                RobotFaultType.SENSOR_FAILURE,
                0.90,
                "Sensor health status indicates failure",
            )

        # ----------------------------------------------------
        # NETWORK
        # ----------------------------------------------------

        if telemetry.network_connected is False:
            add(
                RobotFaultType.NETWORK_FAILURE,
                0.95,
                "Robot network disconnected",
            )

        # ----------------------------------------------------
        # CONTROLLER
        # ----------------------------------------------------

        if telemetry.controller_healthy is False:
            add(
                RobotFaultType.CONTROLLER_FAULT,
                1.0,
                "Controller health status indicates failure",
            )

        # ----------------------------------------------------
        # SOFTWARE
        # ----------------------------------------------------

        if telemetry.software_error_count > 0:
            score = min(
                1.0,
                0.35
                + telemetry.software_error_count * 0.10,
            )

            add(
                RobotFaultType.SOFTWARE_EXCEPTION,
                score,
                (
                    f"{telemetry.software_error_count} "
                    "software errors reported"
                ),
            )

        # ----------------------------------------------------
        # SYMPTOM RULES
        # ----------------------------------------------------

        symptom_rules = {
            "abnormal_vibration": (
                RobotFaultType.BEARING_WEAR,
                RobotFaultType.GEARBOX_DAMAGE,
            ),
            "grinding_noise": (
                RobotFaultType.BEARING_WEAR,
                RobotFaultType.GEARBOX_DAMAGE,
            ),
            "joint_stiffness": (
                RobotFaultType.JOINT_MISALIGNMENT,
                RobotFaultType.MOTOR_OVERLOAD,
            ),
            "motor_stall": (
                RobotFaultType.MOTOR_OVERLOAD,
            ),
            "unexpected_shutdown": (
                RobotFaultType.CONTROLLER_FAULT,
                RobotFaultType.BATTERY_LOW,
            ),
            "sensor_dropout": (
                RobotFaultType.SENSOR_COMMUNICATION_LOSS,
            ),
            "communication_timeout": (
                RobotFaultType.NETWORK_FAILURE,
                RobotFaultType.SENSOR_COMMUNICATION_LOSS,
            ),
            "visible_crack": (
                RobotFaultType.STRUCTURAL_DAMAGE,
            ),
            "joint_misalignment": (
                RobotFaultType.JOINT_MISALIGNMENT,
            ),
        }

        for symptom in request.symptoms:
            matched_faults = symptom_rules.get(
                symptom.name,
                (),
            )

            for fault in matched_faults:
                add(
                    fault,
                    symptom.confidence * 0.75,
                    (
                        f"Symptom reported: "
                        f"{symptom.name}"
                    ),
                )

        # ----------------------------------------------------
        # DAMAGE INSPECTION EVIDENCE
        # ----------------------------------------------------

        damage_rules = {
            "crack": RobotFaultType.STRUCTURAL_DAMAGE,
            "fracture": RobotFaultType.STRUCTURAL_DAMAGE,
            "deformation": RobotFaultType.STRUCTURAL_DAMAGE,
            "wear": RobotFaultType.BEARING_WEAR,
            "misalignment": RobotFaultType.JOINT_MISALIGNMENT,
            "overheating": RobotFaultType.MOTOR_OVERHEATING,
            "electrical": RobotFaultType.CONTROLLER_FAULT,
        }

        severity_factors = {
            "none": 0.0,
            "low": 0.25,
            "moderate": 0.50,
            "high": 0.80,
            "critical": 1.0,
            "unknown": 0.20,
        }

        for damage in request.damage_evidence:
            fault = damage_rules.get(
                damage.damage_type
            )

            if fault is None:
                continue

            factor = severity_factors.get(
                damage.severity,
                0.20,
            )

            score = (
                damage.confidence
                * factor
            )

            add(
                fault,
                score,
                (
                    f"Damage inspection: "
                    f"{damage.damage_type} "
                    f"({damage.severity})"
                ),
            )

        return evidence

    # --------------------------------------------------------
    # SCORE CANDIDATES
    # --------------------------------------------------------

    def _build_candidates(
        self,
        evidence: dict[
            RobotFaultType,
            list[tuple[float, str]],
        ],
    ) -> list[DiagnosticCandidate]:

        candidates = []

        for fault_type, observations in evidence.items():
            definition = FAULT_KNOWLEDGE_BASE.get(
                fault_type
            )

            if definition is None:
                continue

            # Noisy-OR evidence aggregation.
            # This is a heuristic, not probability.
            remaining = 1.0

            for score, _ in observations:
                remaining *= 1.0 - clamp(score)

            diagnostic_score = clamp(
                1.0 - remaining
            )

            if (
                diagnostic_score
                < self.config.minimum_diagnostic_score
            ):
                continue

            candidates.append(
                DiagnosticCandidate(
                    fault_type=fault_type,
                    category=definition.category,
                    severity=definition.severity,
                    diagnostic_score=diagnostic_score,
                    description=definition.description,
                    evidence=tuple(
                        reason
                        for _, reason in observations
                    ),
                    possible_causes=(
                        definition.possible_causes
                    ),
                    recommendation=(
                        definition.recommendation
                    ),
                )
            )

        candidates.sort(
            key=lambda item: (
                self.SEVERITY_ORDER[
                    item.severity
                ],
                item.diagnostic_score,
            ),
            reverse=True,
        )

        return candidates[
            :self.config.maximum_candidates
        ]

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    def diagnose(
        self,
        request: IssueDiagnosisRequest,
    ) -> dict[str, Any]:
        if not isinstance(
            request,
            IssueDiagnosisRequest,
        ):
            raise DiagnosisValidationError(
                "Invalid diagnosis request"
            )

        if (
            len(request.symptoms)
            > self.config.maximum_symptoms
        ):
            raise DiagnosisValidationError(
                "Too many symptoms"
            )

        evidence = self._collect_evidence(
            request
        )

        candidates = self._build_candidates(
            evidence
        )

        if candidates:
            overall_severity = max(
                (
                    candidate.severity
                    for candidate in candidates
                ),
                key=lambda severity: (
                    self.SEVERITY_ORDER[severity]
                ),
            )

            if overall_severity == IssueSeverity.CRITICAL:
                recommendation = (
                    MaintenanceAction.STOP_AND_REVIEW
                )

            elif overall_severity == IssueSeverity.HIGH:
                recommendation = (
                    MaintenanceAction.PRIORITY_REPAIR
                )

            elif overall_severity == IssueSeverity.MODERATE:
                recommendation = (
                    MaintenanceAction.SCHEDULE_MAINTENANCE
                )

            else:
                recommendation = (
                    MaintenanceAction.INSPECT
                )

        else:
            overall_severity = IssueSeverity.UNKNOWN
            recommendation = MaintenanceAction.INSPECT

        human_review_required = (
            overall_severity in (
                IssueSeverity.HIGH,
                IssueSeverity.CRITICAL,
                IssueSeverity.UNKNOWN,
            )
            or any(
                candidate.recommendation
                == MaintenanceAction.STOP_AND_REVIEW
                for candidate in candidates
            )
        )

        record = IssueDiagnosisRecord(
            diagnosis_id=str(uuid.uuid4()),
            request=request,
            candidates=candidates,
            overall_severity=overall_severity,
            recommendation=recommendation,
            human_review_required=(
                human_review_required
            ),
        )

        with self._lock:
            if (
                len(self._diagnoses)
                >= self.config.maximum_diagnoses
            ):
                raise IssueDiagnosisError(
                    "Diagnosis capacity reached"
                )

            self._diagnoses[
                record.diagnosis_id
            ] = record

            self._audit_events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "timestamp": utc_now().isoformat(),
                    "tenant_id": request.tenant_id,
                    "robot_id": request.robot_id,
                    "diagnosis_id": record.diagnosis_id,
                    "event": (
                        DiagnosticEventType
                        .DIAGNOSIS_CREATED.value
                    ),
                }
            )

            return record.to_dict()

    # --------------------------------------------------------
    # GET DIAGNOSIS
    # --------------------------------------------------------

    def get_diagnosis(
        self,
        *,
        tenant_id: str,
        diagnosis_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(diagnosis_id, "diagnosis_id")

        with self._lock:
            record = self._diagnoses.get(
                diagnosis_id
            )

            if (
                record is None
                or record.request.tenant_id != tenant_id
            ):
                raise DiagnosisNotFound(
                    "Diagnosis not found"
                )

            return record.to_dict()

    # --------------------------------------------------------
    # REVIEW DIAGNOSIS
    # --------------------------------------------------------

    def review_diagnosis(
        self,
        *,
        tenant_id: str,
        diagnosis_id: str,
        approved: bool,
        notes: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise DiagnosisPermissionDenied(
                "Review authorization required"
            )

        if not isinstance(approved, bool):
            raise DiagnosisValidationError(
                "approved must be boolean"
            )

        notes = validate_text(
            notes,
            "notes",
        )

        with self._lock:
            record = self._diagnoses.get(
                diagnosis_id
            )

            if (
                record is None
                or record.request.tenant_id != tenant_id
            ):
                raise DiagnosisNotFound(
                    "Diagnosis not found"
                )

            if (
                record.status
                != DiagnosisStatus.PENDING_REVIEW
            ):
                raise DiagnosisConflict(
                    "Diagnosis already reviewed"
                )

            record.status = (
                DiagnosisStatus.REVIEWED
                if approved
                else DiagnosisStatus.REJECTED
            )

            record.review_notes = notes
            record.reviewed_at = utc_now()

            self._audit_events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "timestamp": utc_now().isoformat(),
                    "tenant_id": tenant_id,
                    "robot_id": record.request.robot_id,
                    "diagnosis_id": diagnosis_id,
                    "event": (
                        DiagnosticEventType
                        .DIAGNOSIS_REVIEWED.value
                        if approved
                        else DiagnosticEventType
                        .DIAGNOSIS_REJECTED.value
                    ),
                }
            )

            return record.to_dict()

    # --------------------------------------------------------
    # LIST DIAGNOSES
    # --------------------------------------------------------

    def list_diagnoses(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        with self._lock:
            return [
                record.to_dict()
                for record in self._diagnoses.values()
                if (
                    record.request.tenant_id == tenant_id
                    and (
                        robot_id is None
                        or record.request.robot_id == robot_id
                    )
                )
            ]

    # --------------------------------------------------------
    # ROBOT ISSUE SUMMARY
    # --------------------------------------------------------

    def robot_issue_summary(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        diagnoses = self.list_diagnoses(
            tenant_id=tenant_id,
            robot_id=robot_id,
        )

        severity_counts = Counter(
            diagnosis["overall_severity"]
            for diagnosis in diagnoses
        )

        fault_counts = Counter()

        for diagnosis in diagnoses:
            for candidate in diagnosis["candidates"]:
                fault_counts[
                    candidate["fault_type"]
                ] += 1

        return {
            "tenant_id": tenant_id,
            "robot_id": robot_id,
            "total_diagnoses": len(diagnoses),
            "severity_counts": dict(
                severity_counts
            ),
            "fault_candidate_counts": dict(
                fault_counts
            ),
            "pending_reviews": sum(
                diagnosis["status"]
                == DiagnosisStatus.PENDING_REVIEW.value
                for diagnosis in diagnoses
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
    ) -> Path:
        validate_id(tenant_id, "tenant_id")

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "diagnoses": self.list_diagnoses(
                    tenant_id=tenant_id
                ),
                "audit_events": [
                    event
                    for event in self._audit_events
                    if event["tenant_id"] == tenant_id
                ],
                "simulation_only": True,
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
    engine = RoboticsIssueDiagnosisEngine()

    request = IssueDiagnosisRequest(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        telemetry=RobotDiagnosticTelemetry(
            motor_temperature_c=94.0,
            motor_current_a=28.0,
            vibration_mm_s=13.5,
            battery_soc_percent=68.0,
            battery_temperature_c=34.0,
            battery_voltage_v=48.2,
            sensor_healthy=True,
            network_connected=True,
            controller_healthy=True,
            software_error_count=0,
        ),
        symptoms=(
            RobotSymptom(
                symptom_id="symptom_001",
                name="abnormal_vibration",
                description="Unusual joint vibration",
                confidence=0.95,
            ),
            RobotSymptom(
                symptom_id="symptom_002",
                name="grinding_noise",
                description="Grinding sound near gearbox",
                confidence=0.90,
            ),
        ),
        damage_evidence=(
            RobotDamageEvidence(
                damage_type="wear",
                severity="high",
                confidence=0.92,
                component_id="joint_001",
            ),
        ),
        inspection_reference="inspection_001",
    )

    diagnosis = engine.diagnose(request)

    print(
        json.dumps(
            diagnosis,
            indent=2,
        )
    )

    summary = engine.robot_issue_summary(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
    )

    print(
        "\nRobot issue summary:"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    report = engine.export_json(
        "data/robotics/issue_diagnosis_report.json",
        tenant_id="tenant_alpha",
    )

    print(
        "\nReport:",
        report,
    )


if __name__ == "__main__":
    main()
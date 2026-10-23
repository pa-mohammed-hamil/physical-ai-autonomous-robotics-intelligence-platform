"""Health Agent module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/health_agent.py

Physical AI & Autonomous Robotics Intelligence Platform
Robot Health Intelligence Agent

Python 3.11+

Features:
- Real-time robot health assessment
- Sensor health monitoring
- Battery health analysis
- Motor and actuator diagnostics
- CPU, memory, and thermal monitoring
- Network and connectivity diagnostics
- Robot component health scoring
- Anomaly detection
- Failure-risk assessment
- Remaining useful life (RUL) integration
- Predictive maintenance recommendations
- Fault correlation
- Robot readiness evaluation
- Fleet health summaries
- Explainable health reports
- Multi-tenant isolation
- Role-based authorization integration
- Audit logging

Safety:
- This agent provides diagnostic recommendations only.
- It cannot directly actuate robots.
- It cannot reset emergency stops.
- It cannot override hardware safety interlocks.
- Critical faults require qualified human review.
- Threshold-based findings are not certified diagnoses.

Production integrations:
- Telemetry database
- Sensor monitoring service
- ML anomaly detection service
- Failure prediction models
- RUL prediction models
- Maintenance management system
- Alert notification service
- Audit database
"""

from __future__ import annotations

import asyncio
import copy
import math
import re
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


# ============================================================
# Constants
# ============================================================

MAX_COMPONENTS = 500
MAX_SENSORS = 1000
MAX_FINDINGS = 1000
MAX_METADATA_FIELDS = 50

DEFAULT_TELEMETRY_TIMEOUT_SECONDS = 120

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)


# ============================================================
# Utilities
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def validate_identifier(value: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError("Invalid identifier")
    return value


def validate_percentage(value: float) -> float:
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError(
            "Percentage must be finite and between 0 and 100"
        )
    return value


def validate_probability(value: float) -> float:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(
            "Probability must be finite and between 0 and 1"
        )
    return value


def validate_nonnegative(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "Value must be finite and nonnegative"
        )
    return value


def validate_finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Value must be finite")
    return value


def validate_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError(
            "Datetime must be timezone-aware"
        )
    return value


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, value))


# ============================================================
# Enumerations
# ============================================================

class RobotHealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    BATTERY = "battery"
    MOTOR = "motor"
    ACTUATOR = "actuator"
    GEARBOX = "gearbox"
    BEARING = "bearing"
    CONTROLLER = "controller"
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    ENCODER = "encoder"
    NETWORK = "network"
    POWER_SYSTEM = "power_system"
    COOLING_SYSTEM = "cooling_system"
    SAFETY_SYSTEM = "safety_system"
    MANIPULATOR = "manipulator"
    END_EFFECTOR = "end_effector"
    CUSTOM = "custom"


class SensorStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    FAILED = "failed"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class FaultSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class HealthFindingType(str, Enum):
    LOW_BATTERY = "low_battery"
    BATTERY_DEGRADATION = "battery_degradation"
    HIGH_TEMPERATURE = "high_temperature"
    HIGH_CPU_USAGE = "high_cpu_usage"
    HIGH_MEMORY_USAGE = "high_memory_usage"
    MOTOR_OVERLOAD = "motor_overload"
    MOTOR_OVERHEATING = "motor_overheating"
    EXCESSIVE_VIBRATION = "excessive_vibration"
    SENSOR_FAILURE = "sensor_failure"
    SENSOR_DEGRADATION = "sensor_degradation"
    SENSOR_OFFLINE = "sensor_offline"
    NETWORK_DEGRADATION = "network_degradation"
    TELEMETRY_STALE = "telemetry_stale"
    FAILURE_RISK = "failure_risk"
    LOW_REMAINING_USEFUL_LIFE = "low_remaining_useful_life"
    MAINTENANCE_REQUIRED = "maintenance_required"
    SAFETY_FAULT = "safety_fault"
    EMERGENCY_STOP = "emergency_stop"
    CUSTOM = "custom"


class MaintenancePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class RobotReadiness(str, Enum):
    READY = "ready"
    RESTRICTED = "restricted"
    NOT_READY = "not_ready"
    UNKNOWN = "unknown"


class HealthAssessmentSource(str, Enum):
    RULE_BASED = "rule_based"
    ML_MODEL = "ml_model"
    SENSOR_DIAGNOSTICS = "sensor_diagnostics"
    TELEMETRY = "telemetry"
    FUSED = "fused"


# ============================================================
# Exceptions
# ============================================================

class HealthAgentError(Exception):
    pass


class HealthValidationError(HealthAgentError):
    pass


class RobotHealthDataNotFoundError(HealthAgentError):
    pass


class HealthPermissionError(HealthAgentError):
    pass


class HealthServiceUnavailableError(HealthAgentError):
    pass


class HealthConflictError(HealthAgentError):
    pass


# ============================================================
# Security Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class HealthAgentScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Threshold Configuration
# ============================================================

@dataclass(frozen=True, slots=True)
class HealthThresholds:
    battery_warning_percentage: float = 25.0
    battery_critical_percentage: float = 10.0

    battery_soh_warning_percentage: float = 80.0
    battery_soh_critical_percentage: float = 60.0

    cpu_warning_percentage: float = 85.0
    cpu_critical_percentage: float = 95.0

    memory_warning_percentage: float = 85.0
    memory_critical_percentage: float = 95.0

    controller_warning_celsius: float = 70.0
    controller_critical_celsius: float = 85.0

    motor_warning_celsius: float = 75.0
    motor_critical_celsius: float = 95.0

    motor_load_warning_percentage: float = 85.0
    motor_load_critical_percentage: float = 98.0

    vibration_warning_mm_s: float = 7.0
    vibration_critical_mm_s: float = 12.0

    network_latency_warning_ms: float = 150.0
    network_latency_critical_ms: float = 500.0

    packet_loss_warning_percentage: float = 2.0
    packet_loss_critical_percentage: float = 10.0

    failure_probability_warning: float = 0.30
    failure_probability_critical: float = 0.70

    rul_warning_hours: float = 200.0
    rul_critical_hours: float = 48.0

    telemetry_timeout_seconds: int = (
        DEFAULT_TELEMETRY_TIMEOUT_SECONDS
    )

    def __post_init__(self) -> None:
        percentage_fields = (
            "battery_warning_percentage",
            "battery_critical_percentage",
            "battery_soh_warning_percentage",
            "battery_soh_critical_percentage",
            "cpu_warning_percentage",
            "cpu_critical_percentage",
            "memory_warning_percentage",
            "memory_critical_percentage",
            "motor_load_warning_percentage",
            "motor_load_critical_percentage",
            "packet_loss_warning_percentage",
            "packet_loss_critical_percentage",
        )

        for name in percentage_fields:
            validate_percentage(getattr(self, name))

        numeric_fields = (
            "controller_warning_celsius",
            "controller_critical_celsius",
            "motor_warning_celsius",
            "motor_critical_celsius",
            "vibration_warning_mm_s",
            "vibration_critical_mm_s",
            "network_latency_warning_ms",
            "network_latency_critical_ms",
            "rul_warning_hours",
            "rul_critical_hours",
        )

        for name in numeric_fields:
            validate_finite(getattr(self, name))

        validate_probability(
            self.failure_probability_warning
        )

        validate_probability(
            self.failure_probability_critical
        )

        threshold_pairs = (
            (
                self.battery_critical_percentage,
                self.battery_warning_percentage,
            ),
            (
                self.battery_soh_critical_percentage,
                self.battery_soh_warning_percentage,
            ),
            (
                self.cpu_warning_percentage,
                self.cpu_critical_percentage,
            ),
            (
                self.memory_warning_percentage,
                self.memory_critical_percentage,
            ),
            (
                self.controller_warning_celsius,
                self.controller_critical_celsius,
            ),
            (
                self.motor_warning_celsius,
                self.motor_critical_celsius,
            ),
            (
                self.motor_load_warning_percentage,
                self.motor_load_critical_percentage,
            ),
            (
                self.vibration_warning_mm_s,
                self.vibration_critical_mm_s,
            ),
            (
                self.network_latency_warning_ms,
                self.network_latency_critical_ms,
            ),
            (
                self.packet_loss_warning_percentage,
                self.packet_loss_critical_percentage,
            ),
            (
                self.failure_probability_warning,
                self.failure_probability_critical,
            ),
        )

        for lower, upper in threshold_pairs:
            if lower >= upper:
                raise HealthValidationError(
                    "Warning and critical thresholds "
                    "must be correctly ordered"
                )

        if self.rul_critical_hours >= self.rul_warning_hours:
            raise HealthValidationError(
                "Critical RUL must be below warning RUL"
            )

        if self.telemetry_timeout_seconds <= 0:
            raise HealthValidationError(
                "Telemetry timeout must be positive"
            )


# ============================================================
# Telemetry Models
# ============================================================

@dataclass(slots=True)
class BatteryHealthTelemetry:
    charge_percentage: float
    state_of_health_percentage: float | None = None

    voltage_v: float | None = None
    current_a: float | None = None
    temperature_c: float | None = None

    cycle_count: int | None = None

    charging: bool = False

    def __post_init__(self) -> None:
        validate_percentage(self.charge_percentage)

        if self.state_of_health_percentage is not None:
            validate_percentage(
                self.state_of_health_percentage
            )

        for value in (
            self.voltage_v,
            self.current_a,
            self.temperature_c,
        ):
            if value is not None:
                validate_finite(value)

        if self.cycle_count is not None:
            if self.cycle_count < 0:
                raise HealthValidationError(
                    "Cycle count cannot be negative"
                )


@dataclass(slots=True)
class ComputeHealthTelemetry:
    cpu_usage_percentage: float
    memory_usage_percentage: float

    controller_temperature_c: float | None = None

    gpu_usage_percentage: float | None = None
    disk_usage_percentage: float | None = None

    def __post_init__(self) -> None:
        validate_percentage(
            self.cpu_usage_percentage
        )

        validate_percentage(
            self.memory_usage_percentage
        )

        if self.controller_temperature_c is not None:
            validate_finite(
                self.controller_temperature_c
            )

        if self.gpu_usage_percentage is not None:
            validate_percentage(
                self.gpu_usage_percentage
            )

        if self.disk_usage_percentage is not None:
            validate_percentage(
                self.disk_usage_percentage
            )


@dataclass(slots=True)
class MotorHealthTelemetry:
    motor_id: str

    temperature_c: float | None = None
    load_percentage: float | None = None
    vibration_mm_s: float | None = None

    current_a: float | None = None

    fault_active: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.motor_id)

        for value in (
            self.temperature_c,
            self.vibration_mm_s,
            self.current_a,
        ):
            if value is not None:
                validate_finite(value)

        if self.load_percentage is not None:
            validate_percentage(
                self.load_percentage
            )


@dataclass(slots=True)
class NetworkHealthTelemetry:
    connected: bool

    latency_ms: float | None = None
    packet_loss_percentage: float | None = None

    signal_strength_percentage: float | None = None

    def __post_init__(self) -> None:
        if self.latency_ms is not None:
            validate_nonnegative(self.latency_ms)

        if self.packet_loss_percentage is not None:
            validate_percentage(
                self.packet_loss_percentage
            )

        if self.signal_strength_percentage is not None:
            validate_percentage(
                self.signal_strength_percentage
            )


@dataclass(slots=True)
class SensorHealthTelemetry:
    sensor_id: str
    sensor_type: ComponentType

    status: SensorStatus

    confidence: float | None = None

    last_seen_at: datetime | None = None

    error_count: int = 0

    def __post_init__(self) -> None:
        validate_identifier(self.sensor_id)

        if self.confidence is not None:
            validate_probability(self.confidence)

        if self.last_seen_at is not None:
            validate_timestamp(self.last_seen_at)

        if self.error_count < 0:
            raise HealthValidationError(
                "Error count cannot be negative"
            )


@dataclass(slots=True)
class ComponentHealthTelemetry:
    component_id: str
    component_type: ComponentType

    health_score: float | None = None

    failure_probability: float | None = None
    remaining_useful_life_hours: float | None = None

    maintenance_required: bool = False
    fault_active: bool = False

    model_id: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.component_id)

        if self.model_id is not None:
            validate_identifier(self.model_id)

        if self.health_score is not None:
            validate_probability(self.health_score)

        if self.failure_probability is not None:
            validate_probability(
                self.failure_probability
            )

        if self.remaining_useful_life_hours is not None:
            validate_nonnegative(
                self.remaining_useful_life_hours
            )


@dataclass(slots=True)
class RobotHealthSnapshot:
    robot_id: str
    tenant_id: str

    timestamp: datetime

    battery: BatteryHealthTelemetry | None = None
    compute: ComputeHealthTelemetry | None = None
    network: NetworkHealthTelemetry | None = None

    motors: list[MotorHealthTelemetry] = field(
        default_factory=list
    )

    sensors: list[SensorHealthTelemetry] = field(
        default_factory=list
    )

    components: list[ComponentHealthTelemetry] = field(
        default_factory=list
    )

    emergency_stop_active: bool = False
    safety_fault_active: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(self.robot_id)
        validate_identifier(self.tenant_id)
        validate_timestamp(self.timestamp)

        if len(self.motors) > MAX_COMPONENTS:
            raise HealthValidationError(
                "Too many motors"
            )

        if len(self.sensors) > MAX_SENSORS:
            raise HealthValidationError(
                "Too many sensors"
            )

        if len(self.components) > MAX_COMPONENTS:
            raise HealthValidationError(
                "Too many components"
            )

        if len(self.metadata) > MAX_METADATA_FIELDS:
            raise HealthValidationError(
                "Too many metadata fields"
            )

        for collection, attr in (
            (self.motors, "motor_id"),
            (self.sensors, "sensor_id"),
            (self.components, "component_id"),
        ):
            identifiers = [
                getattr(item, attr)
                for item in collection
            ]

            if len(identifiers) != len(set(identifiers)):
                raise HealthValidationError(
                    f"Duplicate {attr} values"
                )


# ============================================================
# Findings and Recommendations
# ============================================================

@dataclass(frozen=True, slots=True)
class HealthFinding:
    finding_id: str

    robot_id: str

    finding_type: HealthFindingType
    severity: FaultSeverity

    title: str
    description: str

    component_id: str | None = None

    observed_value: float | None = None
    threshold_value: float | None = None

    confidence: float = 1.0

    source: HealthAssessmentSource = (
        HealthAssessmentSource.RULE_BASED
    )

    detected_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True, slots=True)
class HealthRecommendation:
    recommendation_id: str

    robot_id: str

    title: str
    description: str

    priority: MaintenancePriority

    component_id: str | None = None

    requires_human_review: bool = True

    suggested_actions: tuple[str, ...] = ()

    created_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(slots=True)
class ComponentHealthAssessment:
    component_id: str
    component_type: ComponentType

    health_score: float | None
    health_status: RobotHealthStatus

    failure_probability: float | None
    remaining_useful_life_hours: float | None

    maintenance_required: bool

    findings: list[HealthFinding] = field(
        default_factory=list
    )


@dataclass(slots=True)
class RobotHealthAssessment:
    assessment_id: str

    tenant_id: str
    robot_id: str

    health_status: RobotHealthStatus
    readiness: RobotReadiness

    overall_health_score: float | None

    battery_health_score: float | None
    compute_health_score: float | None
    network_health_score: float | None
    sensor_health_score: float | None
    motor_health_score: float | None

    component_assessments: list[
        ComponentHealthAssessment
    ]

    findings: list[HealthFinding]
    recommendations: list[HealthRecommendation]

    critical_finding_count: int
    high_finding_count: int

    requires_human_review: bool

    telemetry_timestamp: datetime

    generated_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(slots=True)
class FleetHealthAssessment:
    assessment_id: str
    tenant_id: str

    robot_count: int

    healthy_count: int
    warning_count: int
    degraded_count: int
    critical_count: int
    unknown_count: int

    ready_count: int
    restricted_count: int
    not_ready_count: int
    unknown_readiness_count: int

    average_health_score: float | None

    critical_robot_ids: list[str]

    generated_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# Integration Protocols
# ============================================================

class HealthTelemetryProvider(Protocol):

    async def get_latest_snapshot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> RobotHealthSnapshot | None:
        ...


class HealthAssessmentRepository(Protocol):

    async def save_assessment(
        self,
        *,
        assessment: RobotHealthAssessment,
    ) -> None:
        ...


class HealthAuthorizer(Protocol):

    async def require_access(
        self,
        *,
        tenant_id: str,
        user_id: str,
        robot_id: str,
        permission: str,
    ) -> None:
        ...


class HealthAuditSink(Protocol):

    async def record(
        self,
        *,
        tenant_id: str,
        user_id: str,
        action: str,
        resource_id: str,
        metadata: Mapping[str, Any],
    ) -> None:
        ...


class HealthAlertSink(Protocol):

    async def publish(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        assessment: RobotHealthAssessment,
    ) -> None:
        ...


# ============================================================
# Health Agent
# ============================================================

class HealthAgent:

    def __init__(
        self,
        *,
        telemetry_provider: HealthTelemetryProvider,
        authorizer: HealthAuthorizer,
        repository: HealthAssessmentRepository | None = None,
        audit_sink: HealthAuditSink | None = None,
        alert_sink: HealthAlertSink | None = None,
        thresholds: HealthThresholds | None = None,
    ) -> None:

        self.telemetry_provider = telemetry_provider
        self.authorizer = authorizer
        self.repository = repository
        self.audit_sink = audit_sink
        self.alert_sink = alert_sink

        self.thresholds = (
            thresholds or HealthThresholds()
        )

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    async def _authorize(
        self,
        *,
        scope: HealthAgentScope,
        robot_id: str,
        permission: str,
    ) -> None:

        await self.authorizer.require_access(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            robot_id=robot_id,
            permission=permission,
        )

    # --------------------------------------------------------
    # Finding Creation
    # --------------------------------------------------------

    @staticmethod
    def _finding(
        *,
        robot_id: str,
        finding_type: HealthFindingType,
        severity: FaultSeverity,
        title: str,
        description: str,
        component_id: str | None = None,
        observed_value: float | None = None,
        threshold_value: float | None = None,
        confidence: float = 1.0,
        source: HealthAssessmentSource = (
            HealthAssessmentSource.RULE_BASED
        ),
    ) -> HealthFinding:

        return HealthFinding(
            finding_id=generate_id(),
            robot_id=robot_id,
            finding_type=finding_type,
            severity=severity,
            title=title,
            description=description,
            component_id=component_id,
            observed_value=observed_value,
            threshold_value=threshold_value,
            confidence=validate_probability(confidence),
            source=source,
        )

    # --------------------------------------------------------
    # Threshold Checking
    # --------------------------------------------------------

    @staticmethod
    def _high_threshold_status(
        value: float,
        warning: float,
        critical: float,
    ) -> RobotHealthStatus:

        if value >= critical:
            return RobotHealthStatus.CRITICAL

        if value >= warning:
            return RobotHealthStatus.WARNING

        return RobotHealthStatus.HEALTHY

    @staticmethod
    def _low_threshold_status(
        value: float,
        warning: float,
        critical: float,
    ) -> RobotHealthStatus:

        if value <= critical:
            return RobotHealthStatus.CRITICAL

        if value <= warning:
            return RobotHealthStatus.WARNING

        return RobotHealthStatus.HEALTHY

    @staticmethod
    def _status_score(
        status: RobotHealthStatus,
    ) -> float | None:

        scores = {
            RobotHealthStatus.HEALTHY: 1.0,
            RobotHealthStatus.WARNING: 0.75,
            RobotHealthStatus.DEGRADED: 0.40,
            RobotHealthStatus.CRITICAL: 0.0,
            RobotHealthStatus.UNKNOWN: None,
        }

        return scores[status]

    # --------------------------------------------------------
    # Battery Analysis
    # --------------------------------------------------------

    def analyze_battery(
        self,
        *,
        robot_id: str,
        battery: BatteryHealthTelemetry | None,
    ) -> tuple[float | None, list[HealthFinding]]:

        if battery is None:
            return None, []

        findings: list[HealthFinding] = []

        charge_status = self._low_threshold_status(
            battery.charge_percentage,
            self.thresholds.battery_warning_percentage,
            self.thresholds.battery_critical_percentage,
        )

        if charge_status != RobotHealthStatus.HEALTHY:
            severity = (
                FaultSeverity.CRITICAL
                if charge_status == RobotHealthStatus.CRITICAL
                else FaultSeverity.MODERATE
            )

            findings.append(
                self._finding(
                    robot_id=robot_id,
                    finding_type=HealthFindingType.LOW_BATTERY,
                    severity=severity,
                    title="Low robot battery",
                    description=(
                        "Battery charge is below the "
                        "configured operating threshold."
                    ),
                    observed_value=battery.charge_percentage,
                    threshold_value=(
                        self.thresholds.battery_warning_percentage
                    ),
                )
            )

        scores = [
            self._status_score(charge_status)
        ]

        if battery.state_of_health_percentage is not None:
            soh_status = self._low_threshold_status(
                battery.state_of_health_percentage,
                self.thresholds.battery_soh_warning_percentage,
                self.thresholds.battery_soh_critical_percentage,
            )

            scores.append(
                self._status_score(soh_status)
            )

            if soh_status != RobotHealthStatus.HEALTHY:
                findings.append(
                    self._finding(
                        robot_id=robot_id,
                        finding_type=(
                            HealthFindingType.BATTERY_DEGRADATION
                        ),
                        severity=(
                            FaultSeverity.HIGH
                            if soh_status
                            == RobotHealthStatus.CRITICAL
                            else FaultSeverity.MODERATE
                        ),
                        title="Battery degradation detected",
                        description=(
                            "Battery state of health is below "
                            "the configured threshold."
                        ),
                        observed_value=(
                            battery.state_of_health_percentage
                        ),
                        threshold_value=(
                            self.thresholds
                            .battery_soh_warning_percentage
                        ),
                    )
                )

        valid_scores = [
            score
            for score in scores
            if score is not None
        ]

        return (
            min(valid_scores)
            if valid_scores
            else None,
            findings,
        )

    # --------------------------------------------------------
    # Compute Analysis
    # --------------------------------------------------------

    def analyze_compute(
        self,
        *,
        robot_id: str,
        compute: ComputeHealthTelemetry | None,
    ) -> tuple[float | None, list[HealthFinding]]:

        if compute is None:
            return None, []

        findings: list[HealthFinding] = []
        scores: list[float] = []

        metrics = (
            (
                compute.cpu_usage_percentage,
                self.thresholds.cpu_warning_percentage,
                self.thresholds.cpu_critical_percentage,
                HealthFindingType.HIGH_CPU_USAGE,
                "High CPU usage",
            ),
            (
                compute.memory_usage_percentage,
                self.thresholds.memory_warning_percentage,
                self.thresholds.memory_critical_percentage,
                HealthFindingType.HIGH_MEMORY_USAGE,
                "High memory usage",
            ),
        )

        for (
            value,
            warning,
            critical,
            finding_type,
            title,
        ) in metrics:

            status = self._high_threshold_status(
                value,
                warning,
                critical,
            )

            score = self._status_score(status)

            if score is not None:
                scores.append(score)

            if status != RobotHealthStatus.HEALTHY:
                findings.append(
                    self._finding(
                        robot_id=robot_id,
                        finding_type=finding_type,
                        severity=(
                            FaultSeverity.HIGH
                            if status == RobotHealthStatus.CRITICAL
                            else FaultSeverity.MODERATE
                        ),
                        title=title,
                        description=(
                            "Compute utilization exceeds "
                            "the configured threshold."
                        ),
                        observed_value=value,
                        threshold_value=warning,
                    )
                )

        if compute.controller_temperature_c is not None:
            status = self._high_threshold_status(
                compute.controller_temperature_c,
                self.thresholds.controller_warning_celsius,
                self.thresholds.controller_critical_celsius,
            )

            score = self._status_score(status)

            if score is not None:
                scores.append(score)

            if status != RobotHealthStatus.HEALTHY:
                findings.append(
                    self._finding(
                        robot_id=robot_id,
                        finding_type=(
                            HealthFindingType.HIGH_TEMPERATURE
                        ),
                        severity=(
                            FaultSeverity.CRITICAL
                            if status == RobotHealthStatus.CRITICAL
                            else FaultSeverity.MODERATE
                        ),
                        title="Controller overheating",
                        description=(
                            "Controller temperature exceeds "
                            "the configured threshold."
                        ),
                        observed_value=(
                            compute.controller_temperature_c
                        ),
                        threshold_value=(
                            self.thresholds
                            .controller_warning_celsius
                        ),
                    )
                )

        return (
            min(scores) if scores else None,
            findings,
        )

    # --------------------------------------------------------
    # Motor Analysis
    # --------------------------------------------------------

    def analyze_motors(
        self,
        *,
        robot_id: str,
        motors: Sequence[MotorHealthTelemetry],
    ) -> tuple[float | None, list[HealthFinding]]:

        if not motors:
            return None, []

        findings: list[HealthFinding] = []
        scores: list[float] = []

        for motor in motors:
            motor_scores: list[float] = []

            if motor.fault_active:
                motor_scores.append(0.0)

                findings.append(
                    self._finding(
                        robot_id=robot_id,
                        component_id=motor.motor_id,
                        finding_type=(
                            HealthFindingType.MOTOR_OVERLOAD
                        ),
                        severity=FaultSeverity.CRITICAL,
                        title="Motor fault active",
                        description=(
                            "Motor controller reports an active fault."
                        ),
                    )
                )

            checks = (
                (
                    motor.temperature_c,
                    self.thresholds.motor_warning_celsius,
                    self.thresholds.motor_critical_celsius,
                    HealthFindingType.MOTOR_OVERHEATING,
                    "Motor overheating",
                ),
                (
                    motor.load_percentage,
                    self.thresholds.motor_load_warning_percentage,
                    self.thresholds.motor_load_critical_percentage,
                    HealthFindingType.MOTOR_OVERLOAD,
                    "Motor overload",
                ),
                (
                    motor.vibration_mm_s,
                    self.thresholds.vibration_warning_mm_s,
                    self.thresholds.vibration_critical_mm_s,
                    HealthFindingType.EXCESSIVE_VIBRATION,
                    "Excessive motor vibration",
                ),
            )

            for (
                value,
                warning,
                critical,
                finding_type,
                title,
            ) in checks:

                if value is None:
                    continue

                status = self._high_threshold_status(
                    value,
                    warning,
                    critical,
                )

                score = self._status_score(status)

                if score is not None:
                    motor_scores.append(score)

                if status != RobotHealthStatus.HEALTHY:
                    findings.append(
                        self._finding(
                            robot_id=robot_id,
                            component_id=motor.motor_id,
                            finding_type=finding_type,
                            severity=(
                                FaultSeverity.CRITICAL
                                if status
                                == RobotHealthStatus.CRITICAL
                                else FaultSeverity.MODERATE
                            ),
                            title=title,
                            description=(
                                "Motor telemetry exceeds "
                                "the configured threshold."
                            ),
                            observed_value=value,
                            threshold_value=warning,
                        )
                    )

            if motor_scores:
                scores.append(min(motor_scores))

        return (
            min(scores) if scores else None,
            findings,
        )

    # --------------------------------------------------------
    # Network Analysis
    # --------------------------------------------------------

    def analyze_network(
        self,
        *,
        robot_id: str,
        network: NetworkHealthTelemetry | None,
    ) -> tuple[float | None, list[HealthFinding]]:

        if network is None:
            return None, []

        findings: list[HealthFinding] = []

        if not network.connected:
            findings.append(
                self._finding(
                    robot_id=robot_id,
                    finding_type=(
                        HealthFindingType.NETWORK_DEGRADATION
                    ),
                    severity=FaultSeverity.HIGH,
                    title="Robot network disconnected",
                    description=(
                        "Robot network connectivity is unavailable."
                    ),
                )
            )

            return 0.0, findings

        scores = [1.0]

        checks = (
            (
                network.latency_ms,
                self.thresholds.network_latency_warning_ms,
                self.thresholds.network_latency_critical_ms,
            ),
            (
                network.packet_loss_percentage,
                self.thresholds.packet_loss_warning_percentage,
                self.thresholds.packet_loss_critical_percentage,
            ),
        )

        for value, warning, critical in checks:
            if value is None:
                continue

            status = self._high_threshold_status(
                value,
                warning,
                critical,
            )

            score = self._status_score(status)

            if score is not None:
                scores.append(score)

            if status != RobotHealthStatus.HEALTHY:
                findings.append(
                    self._finding(
                        robot_id=robot_id,
                        finding_type=(
                            HealthFindingType.NETWORK_DEGRADATION
                        ),
                        severity=(
                            FaultSeverity.HIGH
                            if status == RobotHealthStatus.CRITICAL
                            else FaultSeverity.MODERATE
                        ),
                        title="Network performance degraded",
                        description=(
                            "Network telemetry exceeds "
                            "the configured threshold."
                        ),
                        observed_value=value,
                        threshold_value=warning,
                    )
                )

        return min(scores), findings

    # --------------------------------------------------------
    # Sensor Analysis
    # --------------------------------------------------------

    def analyze_sensors(
        self,
        *,
        robot_id: str,
        sensors: Sequence[SensorHealthTelemetry],
    ) -> tuple[float | None, list[HealthFinding]]:

        if not sensors:
            return None, []

        findings: list[HealthFinding] = []
        scores: list[float] = []

        sensor_scores = {
            SensorStatus.HEALTHY: 1.0,
            SensorStatus.WARNING: 0.75,
            SensorStatus.DEGRADED: 0.40,
            SensorStatus.FAILED: 0.0,
            SensorStatus.OFFLINE: 0.0,
            SensorStatus.UNKNOWN: 0.0,
        }

        for sensor in sensors:
            scores.append(
                sensor_scores[sensor.status]
            )

            if sensor.status == SensorStatus.HEALTHY:
                continue

            if sensor.status in (
                SensorStatus.FAILED,
                SensorStatus.OFFLINE,
            ):
                severity = FaultSeverity.HIGH

                finding_type = (
                    HealthFindingType.SENSOR_FAILURE
                    if sensor.status == SensorStatus.FAILED
                    else HealthFindingType.SENSOR_OFFLINE
                )

            else:
                severity = FaultSeverity.MODERATE
                finding_type = (
                    HealthFindingType.SENSOR_DEGRADATION
                )

            findings.append(
                self._finding(
                    robot_id=robot_id,
                    component_id=sensor.sensor_id,
                    finding_type=finding_type,
                    severity=severity,
                    title="Sensor health issue",
                    description=(
                        f"Sensor {sensor.sensor_id} reports "
                        f"status {sensor.status.value}."
                    ),
                    confidence=(
                        sensor.confidence
                        if sensor.confidence is not None
                        else 1.0
                    ),
                    source=(
                        HealthAssessmentSource.SENSOR_DIAGNOSTICS
                    ),
                )
            )

        return (
            min(scores) if scores else None,
            findings,
        )

    # --------------------------------------------------------
    # Component Analysis
    # --------------------------------------------------------

    def analyze_components(
        self,
        *,
        robot_id: str,
        components: Sequence[ComponentHealthTelemetry],
    ) -> tuple[
        list[ComponentHealthAssessment],
        list[HealthFinding],
    ]:

        assessments: list[
            ComponentHealthAssessment
        ] = []

        all_findings: list[HealthFinding] = []

        for component in components:
            findings: list[HealthFinding] = []

            if component.fault_active:
                findings.append(
                    self._finding(
                        robot_id=robot_id,
                        component_id=component.component_id,
                        finding_type=(
                            HealthFindingType.FAILURE_RISK
                        ),
                        severity=FaultSeverity.CRITICAL,
                        title="Component fault active",
                        description=(
                            "Component reports an active fault."
                        ),
                    )
                )

            if component.failure_probability is not None:
                probability_value = (
                    component.failure_probability
                )

                if (
                    probability_value
                    >= self.thresholds.failure_probability_warning
                ):
                    critical = (
                        probability_value
                        >= self.thresholds.failure_probability_critical
                    )

                    findings.append(
                        self._finding(
                            robot_id=robot_id,
                            component_id=component.component_id,
                            finding_type=(
                                HealthFindingType.FAILURE_RISK
                            ),
                            severity=(
                                FaultSeverity.CRITICAL
                                if critical
                                else FaultSeverity.MODERATE
                            ),
                            title="Predicted component failure risk",
                            description=(
                                "Failure prediction exceeds "
                                "the configured threshold."
                            ),
                            observed_value=probability_value,
                            threshold_value=(
                                self.thresholds
                                .failure_probability_warning
                            ),
                            source=(
                                HealthAssessmentSource.ML_MODEL
                            ),
                        )
                    )

            if (
                component.remaining_useful_life_hours
                is not None
            ):
                rul = component.remaining_useful_life_hours

                if rul <= self.thresholds.rul_warning_hours:
                    critical = (
                        rul <= self.thresholds.rul_critical_hours
                    )

                    findings.append(
                        self._finding(
                            robot_id=robot_id,
                            component_id=component.component_id,
                            finding_type=(
                                HealthFindingType
                                .LOW_REMAINING_USEFUL_LIFE
                            ),
                            severity=(
                                FaultSeverity.HIGH
                                if critical
                                else FaultSeverity.MODERATE
                            ),
                            title="Low remaining useful life",
                            description=(
                                "Predicted remaining useful life "
                                "is below the configured threshold."
                            ),
                            observed_value=rul,
                            threshold_value=(
                                self.thresholds.rul_warning_hours
                            ),
                            source=(
                                HealthAssessmentSource.ML_MODEL
                            ),
                        )
                    )

            if component.maintenance_required:
                findings.append(
                    self._finding(
                        robot_id=robot_id,
                        component_id=component.component_id,
                        finding_type=(
                            HealthFindingType.MAINTENANCE_REQUIRED
                        ),
                        severity=FaultSeverity.MODERATE,
                        title="Component maintenance required",
                        description=(
                            "Component has a pending "
                            "maintenance requirement."
                        ),
                    )
                )

            score = component.health_score

            if component.fault_active:
                status = RobotHealthStatus.CRITICAL

            elif any(
                finding.severity == FaultSeverity.CRITICAL
                for finding in findings
            ):
                status = RobotHealthStatus.CRITICAL

            elif any(
                finding.severity == FaultSeverity.HIGH
                for finding in findings
            ):
                status = RobotHealthStatus.DEGRADED

            elif findings:
                status = RobotHealthStatus.WARNING

            elif score is None:
                status = RobotHealthStatus.UNKNOWN

            elif score >= 0.85:
                status = RobotHealthStatus.HEALTHY

            elif score >= 0.65:
                status = RobotHealthStatus.WARNING

            elif score >= 0.35:
                status = RobotHealthStatus.DEGRADED

            else:
                status = RobotHealthStatus.CRITICAL

            assessments.append(
                ComponentHealthAssessment(
                    component_id=component.component_id,
                    component_type=component.component_type,
                    health_score=score,
                    health_status=status,
                    failure_probability=(
                        component.failure_probability
                    ),
                    remaining_useful_life_hours=(
                        component.remaining_useful_life_hours
                    ),
                    maintenance_required=(
                        component.maintenance_required
                    ),
                    findings=findings,
                )
            )

            all_findings.extend(findings)

        return assessments, all_findings

    # --------------------------------------------------------
    # Overall Health Score
    # --------------------------------------------------------

    @staticmethod
    def calculate_overall_health_score(
        scores: Sequence[float | None],
    ) -> float | None:

        valid = [
            score
            for score in scores
            if score is not None
        ]

        if not valid:
            return None

        # Conservative aggregation:
        # A weak subsystem should significantly affect
        # the overall health assessment.
        average = sum(valid) / len(valid)
        minimum = min(valid)

        result = (
            0.60 * minimum
            + 0.40 * average
        )

        return round(
            clamp(result),
            6,
        )

    # --------------------------------------------------------
    # Overall Status
    # --------------------------------------------------------

    @staticmethod
    def determine_health_status(
        *,
        score: float | None,
        findings: Sequence[HealthFinding],
    ) -> RobotHealthStatus:

        if any(
            finding.severity == FaultSeverity.CRITICAL
            for finding in findings
        ):
            return RobotHealthStatus.CRITICAL

        if any(
            finding.severity == FaultSeverity.HIGH
            for finding in findings
        ):
            return RobotHealthStatus.DEGRADED

        if score is None:
            return RobotHealthStatus.UNKNOWN

        if score < 0.35:
            return RobotHealthStatus.CRITICAL

        if score < 0.65:
            return RobotHealthStatus.DEGRADED

        if (
            score < 0.85
            or any(
                finding.severity == FaultSeverity.MODERATE
                for finding in findings
            )
        ):
            return RobotHealthStatus.WARNING

        return RobotHealthStatus.HEALTHY

    # --------------------------------------------------------
    # Robot Readiness
    # --------------------------------------------------------

    @staticmethod
    def determine_readiness(
        *,
        health_status: RobotHealthStatus,
        findings: Sequence[HealthFinding],
    ) -> RobotReadiness:

        if health_status == RobotHealthStatus.UNKNOWN:
            return RobotReadiness.UNKNOWN

        if health_status in (
            RobotHealthStatus.CRITICAL,
            RobotHealthStatus.DEGRADED,
        ):
            return RobotReadiness.NOT_READY

        if any(
            finding.finding_type in (
                HealthFindingType.EMERGENCY_STOP,
                HealthFindingType.SAFETY_FAULT,
                HealthFindingType.TELEMETRY_STALE,
            )
            for finding in findings
        ):
            return RobotReadiness.NOT_READY

        if health_status == RobotHealthStatus.WARNING:
            return RobotReadiness.RESTRICTED

        return RobotReadiness.READY

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    @staticmethod
    def generate_recommendations(
        *,
        robot_id: str,
        findings: Sequence[HealthFinding],
    ) -> list[HealthRecommendation]:

        recommendations: list[
            HealthRecommendation
        ] = []

        seen: set[
            tuple[HealthFindingType, str | None]
        ] = set()

        for finding in findings:

            key = (
                finding.finding_type,
                finding.component_id,
            )

            if key in seen:
                continue

            seen.add(key)

            if finding.severity == FaultSeverity.CRITICAL:
                priority = MaintenancePriority.URGENT

            elif finding.severity == FaultSeverity.HIGH:
                priority = MaintenancePriority.HIGH

            elif finding.severity == FaultSeverity.MODERATE:
                priority = MaintenancePriority.NORMAL

            else:
                priority = MaintenancePriority.LOW

            action_map = {
                HealthFindingType.LOW_BATTERY: (
                    "Verify battery telemetry",
                    "Schedule charging according to "
                    "approved operating procedures",
                ),
                HealthFindingType.BATTERY_DEGRADATION: (
                    "Inspect battery health records",
                    "Request qualified battery inspection",
                ),
                HealthFindingType.HIGH_TEMPERATURE: (
                    "Inspect thermal telemetry",
                    "Check cooling and ventilation",
                    "Follow approved overheating procedures",
                ),
                HealthFindingType.HIGH_CPU_USAGE: (
                    "Inspect active workloads",
                    "Review compute resource utilization",
                ),
                HealthFindingType.HIGH_MEMORY_USAGE: (
                    "Inspect memory usage",
                    "Investigate memory-intensive processes",
                ),
                HealthFindingType.MOTOR_OVERLOAD: (
                    "Review motor load history",
                    "Inspect motor and mechanical components",
                ),
                HealthFindingType.MOTOR_OVERHEATING: (
                    "Inspect motor temperature trends",
                    "Request motor cooling inspection",
                ),
                HealthFindingType.EXCESSIVE_VIBRATION: (
                    "Inspect vibration history",
                    "Check bearings and mechanical alignment",
                ),
                HealthFindingType.SENSOR_FAILURE: (
                    "Inspect sensor diagnostics",
                    "Verify sensor connection and calibration",
                ),
                HealthFindingType.SENSOR_OFFLINE: (
                    "Check sensor connectivity",
                    "Inspect sensor power and communication",
                ),
                HealthFindingType.SENSOR_DEGRADATION: (
                    "Review sensor quality metrics",
                    "Schedule calibration verification",
                ),
                HealthFindingType.NETWORK_DEGRADATION: (
                    "Inspect network connectivity",
                    "Review latency and packet loss",
                ),
                HealthFindingType.TELEMETRY_STALE: (
                    "Verify robot connectivity",
                    "Check telemetry ingestion pipeline",
                    "Do not rely on stale status for control",
                ),
                HealthFindingType.FAILURE_RISK: (
                    "Review failure prediction evidence",
                    "Schedule component inspection",
                ),
                HealthFindingType.LOW_REMAINING_USEFUL_LIFE: (
                    "Review RUL model confidence",
                    "Plan preventive maintenance",
                ),
                HealthFindingType.MAINTENANCE_REQUIRED: (
                    "Review maintenance history",
                    "Create technician review request",
                ),
                HealthFindingType.SAFETY_FAULT: (
                    "Follow approved safety procedures",
                    "Request qualified safety officer review",
                ),
                HealthFindingType.EMERGENCY_STOP: (
                    "Keep emergency-stop safety controls active",
                    "Request authorized safety inspection",
                    "Do not remotely reset the emergency stop",
                ),
            }

            actions = action_map.get(
                finding.finding_type,
                (
                    "Inspect the reported condition",
                    "Request qualified human review",
                ),
            )

            recommendations.append(
                HealthRecommendation(
                    recommendation_id=generate_id(),
                    robot_id=robot_id,
                    title=finding.title,
                    description=finding.description,
                    priority=priority,
                    component_id=finding.component_id,
                    requires_human_review=True,
                    suggested_actions=actions,
                )
            )

        return recommendations

    # --------------------------------------------------------
    # Main Health Assessment
    # --------------------------------------------------------

    async def assess_robot_health(
        self,
        *,
        scope: HealthAgentScope,
        robot_id: str,
        persist: bool = True,
        publish_alerts: bool = True,
    ) -> RobotHealthAssessment:

        validate_identifier(robot_id)

        await self._authorize(
            scope=scope,
            robot_id=robot_id,
            permission="robot:health:read",
        )

        snapshot = await self.telemetry_provider.get_latest_snapshot(
            tenant_id=scope.tenant_id,
            robot_id=robot_id,
        )

        if snapshot is None:
            raise RobotHealthDataNotFoundError(
                "Robot health telemetry is unavailable"
            )

        if (
            snapshot.tenant_id != scope.tenant_id
            or snapshot.robot_id != robot_id
        ):
            raise HealthPermissionError(
                "Telemetry resource scope mismatch"
            )

        now = utc_now()

        age_seconds = (
            now - snapshot.timestamp
        ).total_seconds()

        findings: list[HealthFinding] = []

        telemetry_stale = (
            age_seconds < 0
            or age_seconds
            > self.thresholds.telemetry_timeout_seconds
        )

        if telemetry_stale:
            findings.append(
                self._finding(
                    robot_id=robot_id,
                    finding_type=(
                        HealthFindingType.TELEMETRY_STALE
                    ),
                    severity=FaultSeverity.HIGH,
                    title="Robot telemetry is stale",
                    description=(
                        "The latest robot telemetry cannot "
                        "be treated as current."
                    ),
                    observed_value=age_seconds,
                    threshold_value=float(
                        self.thresholds
                        .telemetry_timeout_seconds
                    ),
                )
            )

        if snapshot.emergency_stop_active:
            findings.append(
                self._finding(
                    robot_id=robot_id,
                    finding_type=(
                        HealthFindingType.EMERGENCY_STOP
                    ),
                    severity=FaultSeverity.CRITICAL,
                    title="Emergency stop active",
                    description=(
                        "Robot reports an active emergency stop."
                    ),
                )
            )

        if snapshot.safety_fault_active:
            findings.append(
                self._finding(
                    robot_id=robot_id,
                    finding_type=(
                        HealthFindingType.SAFETY_FAULT
                    ),
                    severity=FaultSeverity.CRITICAL,
                    title="Robot safety fault",
                    description=(
                        "Robot safety subsystem reports a fault."
                    ),
                )
            )

        battery_score, battery_findings = (
            self.analyze_battery(
                robot_id=robot_id,
                battery=snapshot.battery,
            )
        )

        compute_score, compute_findings = (
            self.analyze_compute(
                robot_id=robot_id,
                compute=snapshot.compute,
            )
        )

        motor_score, motor_findings = (
            self.analyze_motors(
                robot_id=robot_id,
                motors=snapshot.motors,
            )
        )

        network_score, network_findings = (
            self.analyze_network(
                robot_id=robot_id,
                network=snapshot.network,
            )
        )

        sensor_score, sensor_findings = (
            self.analyze_sensors(
                robot_id=robot_id,
                sensors=snapshot.sensors,
            )
        )

        component_assessments, component_findings = (
            self.analyze_components(
                robot_id=robot_id,
                components=snapshot.components,
            )
        )

        findings.extend(battery_findings)
        findings.extend(compute_findings)
        findings.extend(motor_findings)
        findings.extend(network_findings)
        findings.extend(sensor_findings)
        findings.extend(component_findings)

        if len(findings) > MAX_FINDINGS:
            raise HealthValidationError(
                "Too many health findings"
            )

        component_scores = [
            assessment.health_score
            for assessment in component_assessments
            if assessment.health_score is not None
        ]

        component_score = (
            min(component_scores)
            if component_scores
            else None
        )

        overall_score = (
            self.calculate_overall_health_score(
                (
                    battery_score,
                    compute_score,
                    motor_score,
                    network_score,
                    sensor_score,
                    component_score,
                )
            )
        )

        if snapshot.emergency_stop_active:
            overall_score = 0.0

        if snapshot.safety_fault_active:
            overall_score = 0.0

        if telemetry_stale:
            # Stale telemetry must not yield an apparently
            # healthy score.
            overall_score = None

        status = self.determine_health_status(
            score=overall_score,
            findings=findings,
        )

        readiness = self.determine_readiness(
            health_status=status,
            findings=findings,
        )

        recommendations = self.generate_recommendations(
            robot_id=robot_id,
            findings=findings,
        )

        critical_count = sum(
            finding.severity == FaultSeverity.CRITICAL
            for finding in findings
        )

        high_count = sum(
            finding.severity == FaultSeverity.HIGH
            for finding in findings
        )

        requires_review = (
            critical_count > 0
            or high_count > 0
            or readiness != RobotReadiness.READY
        )

        assessment = RobotHealthAssessment(
            assessment_id=generate_id(),
            tenant_id=scope.tenant_id,
            robot_id=robot_id,
            health_status=status,
            readiness=readiness,
            overall_health_score=overall_score,
            battery_health_score=battery_score,
            compute_health_score=compute_score,
            network_health_score=network_score,
            sensor_health_score=sensor_score,
            motor_health_score=motor_score,
            component_assessments=component_assessments,
            findings=findings,
            recommendations=recommendations,
            critical_finding_count=critical_count,
            high_finding_count=high_count,
            requires_human_review=requires_review,
            telemetry_timestamp=snapshot.timestamp,
        )

        if persist and self.repository is not None:
            await self.repository.save_assessment(
                assessment=assessment
            )

        if self.audit_sink is not None:
            await self.audit_sink.record(
                tenant_id=scope.tenant_id,
                user_id=scope.user_id,
                action="health_agent.assess_robot",
                resource_id=assessment.assessment_id,
                metadata={
                    "robot_id": robot_id,
                    "health_status": status.value,
                    "readiness": readiness.value,
                    "critical_findings": critical_count,
                    "high_findings": high_count,
                },
            )

        if (
            publish_alerts
            and self.alert_sink is not None
            and (
                critical_count > 0
                or high_count > 0
            )
        ):
            await self.alert_sink.publish(
                tenant_id=scope.tenant_id,
                robot_id=robot_id,
                assessment=assessment,
            )

        return assessment

    # --------------------------------------------------------
    # Fleet Health Assessment
    # --------------------------------------------------------

    async def assess_fleet_health(
        self,
        *,
        scope: HealthAgentScope,
        robot_ids: Sequence[str],
    ) -> FleetHealthAssessment:

        if len(robot_ids) > MAX_COMPONENTS:
            raise HealthValidationError(
                "Too many robots in one fleet assessment"
            )

        if len(set(robot_ids)) != len(robot_ids):
            raise HealthValidationError(
                "Duplicate robot IDs"
            )

        assessments: list[RobotHealthAssessment] = []

        # Bounded concurrency avoids overwhelming
        # telemetry and authorization backends.
        semaphore = asyncio.Semaphore(20)

        async def assess_one(
            robot_id: str,
        ) -> RobotHealthAssessment:
            async with semaphore:
                return await self.assess_robot_health(
                    scope=scope,
                    robot_id=robot_id,
                    persist=False,
                    publish_alerts=False,
                )

        results = await asyncio.gather(
            *(
                assess_one(robot_id)
                for robot_id in robot_ids
            ),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, BaseException):
                # Never hide authorization or backend errors
                # by silently treating them as healthy.
                raise result

            assessments.append(result)

        def count_status(
            status: RobotHealthStatus,
        ) -> int:
            return sum(
                assessment.health_status == status
                for assessment in assessments
            )

        def count_readiness(
            readiness: RobotReadiness,
        ) -> int:
            return sum(
                assessment.readiness == readiness
                for assessment in assessments
            )

        valid_scores = [
            assessment.overall_health_score
            for assessment in assessments
            if assessment.overall_health_score is not None
        ]

        average_score = (
            round(
                sum(valid_scores) / len(valid_scores),
                6,
            )
            if valid_scores
            else None
        )

        return FleetHealthAssessment(
            assessment_id=generate_id(),
            tenant_id=scope.tenant_id,
            robot_count=len(assessments),
            healthy_count=count_status(
                RobotHealthStatus.HEALTHY
            ),
            warning_count=count_status(
                RobotHealthStatus.WARNING
            ),
            degraded_count=count_status(
                RobotHealthStatus.DEGRADED
            ),
            critical_count=count_status(
                RobotHealthStatus.CRITICAL
            ),
            unknown_count=count_status(
                RobotHealthStatus.UNKNOWN
            ),
            ready_count=count_readiness(
                RobotReadiness.READY
            ),
            restricted_count=count_readiness(
                RobotReadiness.RESTRICTED
            ),
            not_ready_count=count_readiness(
                RobotReadiness.NOT_READY
            ),
            unknown_readiness_count=count_readiness(
                RobotReadiness.UNKNOWN
            ),
            average_health_score=average_score,
            critical_robot_ids=[
                assessment.robot_id
                for assessment in assessments
                if assessment.health_status
                == RobotHealthStatus.CRITICAL
            ],
        )

    # --------------------------------------------------------
    # Explainable Report
    # --------------------------------------------------------

    @staticmethod
    def generate_health_report(
        assessment: RobotHealthAssessment,
    ) -> dict[str, Any]:

        return {
            "assessment_id": assessment.assessment_id,
            "tenant_id": assessment.tenant_id,
            "robot_id": assessment.robot_id,
            "generated_at": (
                assessment.generated_at.isoformat()
            ),
            "telemetry_timestamp": (
                assessment.telemetry_timestamp.isoformat()
            ),
            "health": {
                "status": assessment.health_status.value,
                "readiness": assessment.readiness.value,
                "overall_score": (
                    assessment.overall_health_score
                ),
                "subsystem_scores": {
                    "battery": (
                        assessment.battery_health_score
                    ),
                    "compute": (
                        assessment.compute_health_score
                    ),
                    "network": (
                        assessment.network_health_score
                    ),
                    "sensors": (
                        assessment.sensor_health_score
                    ),
                    "motors": (
                        assessment.motor_health_score
                    ),
                },
            },
            "findings": [
                {
                    "finding_id": finding.finding_id,
                    "type": finding.finding_type.value,
                    "severity": finding.severity.value,
                    "title": finding.title,
                    "description": finding.description,
                    "component_id": finding.component_id,
                    "observed_value": finding.observed_value,
                    "threshold_value": finding.threshold_value,
                    "confidence": finding.confidence,
                    "source": finding.source.value,
                }
                for finding in assessment.findings
            ],
            "recommendations": [
                {
                    "recommendation_id": (
                        recommendation.recommendation_id
                    ),
                    "title": recommendation.title,
                    "description": recommendation.description,
                    "priority": recommendation.priority.value,
                    "component_id": recommendation.component_id,
                    "requires_human_review": (
                        recommendation.requires_human_review
                    ),
                    "suggested_actions": list(
                        recommendation.suggested_actions
                    ),
                }
                for recommendation
                in assessment.recommendations
            ],
            "summary": {
                "critical_findings": (
                    assessment.critical_finding_count
                ),
                "high_findings": (
                    assessment.high_finding_count
                ),
                "requires_human_review": (
                    assessment.requires_human_review
                ),
            },
            "safety": {
                "physical_robot_control": False,
                "emergency_stop_reset": False,
                "hardware_interlock_override": False,
                "diagnostic_recommendations_only": True,
                "readiness_is_advisory": True,
            },
        }


# ============================================================
# Factory
# ============================================================

def create_health_agent(
    *,
    telemetry_provider: HealthTelemetryProvider,
    authorizer: HealthAuthorizer,
    repository: HealthAssessmentRepository | None = None,
    audit_sink: HealthAuditSink | None = None,
    alert_sink: HealthAlertSink | None = None,
    thresholds: HealthThresholds | None = None,
) -> HealthAgent:

    return HealthAgent(
        telemetry_provider=telemetry_provider,
        authorizer=authorizer,
        repository=repository,
        audit_sink=audit_sink,
        alert_sink=alert_sink,
        thresholds=thresholds,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "RobotHealthStatus",
    "ComponentType",
    "SensorStatus",
    "FaultSeverity",
    "HealthFindingType",
    "MaintenancePriority",
    "RobotReadiness",
    "HealthAssessmentSource",
    "HealthAgentError",
    "HealthValidationError",
    "RobotHealthDataNotFoundError",
    "HealthPermissionError",
    "HealthServiceUnavailableError",
    "HealthConflictError",
    "HealthAgentScope",
    "HealthThresholds",
    "BatteryHealthTelemetry",
    "ComputeHealthTelemetry",
    "MotorHealthTelemetry",
    "NetworkHealthTelemetry",
    "SensorHealthTelemetry",
    "ComponentHealthTelemetry",
    "RobotHealthSnapshot",
    "HealthFinding",
    "HealthRecommendation",
    "ComponentHealthAssessment",
    "RobotHealthAssessment",
    "FleetHealthAssessment",
    "HealthTelemetryProvider",
    "HealthAssessmentRepository",
    "HealthAuthorizer",
    "HealthAuditSink",
    "HealthAlertSink",
    "HealthAgent",
    "create_health_agent",
]
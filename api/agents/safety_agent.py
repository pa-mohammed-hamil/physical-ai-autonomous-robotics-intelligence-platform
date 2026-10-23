"""Safety Agent module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/safety_agent.py

Physical AI & Autonomous Robotics Intelligence Platform
Robot Safety Intelligence & Risk Assessment Agent

Python 3.11+

Features:
- Robot operational safety assessment
- Multi-sensor safety evidence analysis
- Human proximity and collision-risk assessment
- Emergency-stop and safety-interlock monitoring
- Robot health and fault risk assessment
- Navigation and manipulation safety review
- Battery, thermal and electrical risk assessment
- Safety policy evaluation
- Human-in-the-loop review requirements
- Safety incident detection and reporting
- Fleet-level safety assessment
- Tenant-aware authorization
- Audit logging
- Safety evaluation metrics

Important:
- This module is advisory and read-only.
- It does not control robots or actuators.
- It cannot reset emergency stops.
- It cannot override safety interlocks.
- AI approval is not a safety certification.
- Unknown or stale sensor data must never be
  interpreted as proof of a safe environment.
- Independent certified safety controllers remain
  responsible for real-world protective functions.
"""

from __future__ import annotations

import asyncio
import math
import re
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


# ============================================================
# Constants
# ============================================================

MAX_SAFETY_OBSERVATIONS = 500
MAX_SAFETY_FINDINGS = 500
MAX_BATCH_ASSESSMENTS = 50

DEFAULT_SENSOR_STALE_SECONDS = 2.0
DEFAULT_HUMAN_WARNING_DISTANCE_M = 2.0
DEFAULT_HUMAN_CRITICAL_DISTANCE_M = 1.0

DEFAULT_BATTERY_WARNING_SOC = 0.20
DEFAULT_BATTERY_CRITICAL_SOC = 0.10

DEFAULT_TEMPERATURE_WARNING_C = 65.0
DEFAULT_TEMPERATURE_CRITICAL_C = 80.0

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)


# ============================================================
# Exceptions
# ============================================================

class SafetyAgentError(Exception):
    """Base safety agent exception."""


class SafetyValidationError(SafetyAgentError):
    """Invalid safety input or configuration."""


class SafetyPermissionError(SafetyAgentError):
    """Safety assessment access denied."""


class SafetyNotFoundError(SafetyAgentError):
    """Robot safety data not found."""


class SafetyProviderError(SafetyAgentError):
    """Safety data provider failed."""


class SafetyPolicyError(SafetyAgentError):
    """Invalid safety policy."""


class SafetyServiceUnavailableError(SafetyAgentError):
    """Required safety service is unavailable."""


# ============================================================
# Utilities
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def validate_identifier(value: str) -> str:
    if (
        not isinstance(value, str)
        or not IDENTIFIER_PATTERN.fullmatch(value)
    ):
        raise SafetyValidationError(
            f"Invalid identifier: {value!r}"
        )
    return value


def validate_finite(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise SafetyValidationError(
            "Value must be finite"
        )
    return float(value)


def validate_nonnegative(value: float) -> float:
    validate_finite(value)

    if value < 0:
        raise SafetyValidationError(
            "Value must be nonnegative"
        )

    return float(value)


def validate_positive(value: float) -> float:
    validate_finite(value)

    if value <= 0:
        raise SafetyValidationError(
            "Value must be positive"
        )

    return float(value)


def validate_probability(value: float) -> float:
    validate_finite(value)

    if not 0.0 <= value <= 1.0:
        raise SafetyValidationError(
            "Probability must be between 0 and 1"
        )

    return float(value)


def validate_timestamp(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
    ):
        raise SafetyValidationError(
            "Timestamp must be timezone-aware"
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

class SafetyRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class SafetyAssessmentStatus(str, Enum):
    COMPLETED = "completed"
    REVIEW_REQUIRED = "review_required"
    INSUFFICIENT_DATA = "insufficient_data"
    CRITICAL_FINDINGS = "critical_findings"


class SafetyFindingType(str, Enum):
    EMERGENCY_STOP = "emergency_stop"
    SAFETY_INTERLOCK = "safety_interlock"
    HUMAN_PROXIMITY = "human_proximity"
    COLLISION_RISK = "collision_risk"
    SENSOR_FAILURE = "sensor_failure"
    STALE_SENSOR_DATA = "stale_sensor_data"
    LOCALIZATION_FAILURE = "localization_failure"
    ROBOT_HEALTH = "robot_health"
    BATTERY = "battery"
    THERMAL = "thermal"
    ELECTRICAL = "electrical"
    MOTOR = "motor"
    ACTUATOR = "actuator"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    COMMUNICATION = "communication"
    CYBERSECURITY = "cybersecurity"
    POLICY_VIOLATION = "policy_violation"
    UNKNOWN = "unknown"


class SafetyObservationType(str, Enum):
    HUMAN_DISTANCE = "human_distance"
    OBSTACLE_DISTANCE = "obstacle_distance"
    BATTERY_SOC = "battery_soc"
    TEMPERATURE_C = "temperature_c"
    COLLISION_PROBABILITY = "collision_probability"
    SENSOR_CONFIDENCE = "sensor_confidence"
    LOCALIZATION_CONFIDENCE = "localization_confidence"
    NETWORK_LATENCY_MS = "network_latency_ms"
    CUSTOM = "custom"


class SafetySensorStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULT = "fault"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class SafetyDecision(str, Enum):
    ADVISORY_LOW_RISK = "advisory_low_risk"
    REVIEW_REQUIRED = "review_required"
    REJECT_PROPOSED_PLAN = "reject_proposed_plan"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class SafetyIncidentSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class RobotOperationalState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class LocalizationStatus(str, Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    LOST = "lost"
    UNKNOWN = "unknown"


# ============================================================
# Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class SafetyScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Safety Policy
# ============================================================

@dataclass(frozen=True, slots=True)
class SafetyPolicy:
    policy_id: str

    sensor_stale_seconds: float = (
        DEFAULT_SENSOR_STALE_SECONDS
    )

    human_warning_distance_m: float = (
        DEFAULT_HUMAN_WARNING_DISTANCE_M
    )

    human_critical_distance_m: float = (
        DEFAULT_HUMAN_CRITICAL_DISTANCE_M
    )

    battery_warning_soc: float = (
        DEFAULT_BATTERY_WARNING_SOC
    )

    battery_critical_soc: float = (
        DEFAULT_BATTERY_CRITICAL_SOC
    )

    temperature_warning_c: float = (
        DEFAULT_TEMPERATURE_WARNING_C
    )

    temperature_critical_c: float = (
        DEFAULT_TEMPERATURE_CRITICAL_C
    )

    collision_warning_probability: float = 0.10
    collision_critical_probability: float = 0.50

    minimum_sensor_confidence: float = 0.80
    minimum_localization_confidence: float = 0.80

    require_human_review_for_high_risk: bool = True

    def __post_init__(self) -> None:
        validate_identifier(self.policy_id)

        validate_positive(
            self.sensor_stale_seconds
        )

        validate_positive(
            self.human_warning_distance_m
        )

        validate_positive(
            self.human_critical_distance_m
        )

        if (
            self.human_critical_distance_m
            >= self.human_warning_distance_m
        ):
            raise SafetyPolicyError(
                "Critical human distance must be "
                "less than warning distance"
            )

        validate_probability(
            self.battery_warning_soc
        )

        validate_probability(
            self.battery_critical_soc
        )

        if (
            self.battery_critical_soc
            >= self.battery_warning_soc
        ):
            raise SafetyPolicyError(
                "Critical battery threshold must be "
                "less than warning threshold"
            )

        validate_finite(
            self.temperature_warning_c
        )

        validate_finite(
            self.temperature_critical_c
        )

        if (
            self.temperature_critical_c
            <= self.temperature_warning_c
        ):
            raise SafetyPolicyError(
                "Critical temperature threshold must "
                "exceed warning threshold"
            )

        validate_probability(
            self.collision_warning_probability
        )

        validate_probability(
            self.collision_critical_probability
        )

        if (
            self.collision_critical_probability
            <= self.collision_warning_probability
        ):
            raise SafetyPolicyError(
                "Critical collision probability must "
                "exceed warning probability"
            )

        validate_probability(
            self.minimum_sensor_confidence
        )

        validate_probability(
            self.minimum_localization_confidence
        )


# ============================================================
# Sensor Observations
# ============================================================

@dataclass(frozen=True, slots=True)
class SafetyObservation:
    observation_id: str

    sensor_id: str
    observation_type: SafetyObservationType

    value: float
    timestamp: datetime

    sensor_status: SafetySensorStatus = (
        SafetySensorStatus.UNKNOWN
    )

    confidence: float = 1.0

    unit: str = ""

    def __post_init__(self) -> None:
        validate_identifier(self.observation_id)
        validate_identifier(self.sensor_id)

        validate_finite(self.value)
        validate_timestamp(self.timestamp)
        validate_probability(self.confidence)


# ============================================================
# Robot Safety Snapshot
# ============================================================

@dataclass(slots=True)
class RobotSafetySnapshot:
    tenant_id: str
    robot_id: str

    timestamp: datetime

    operational_state: RobotOperationalState

    emergency_stop_active: bool
    safety_interlock_active: bool

    safety_controller_healthy: bool | None

    localization_status: LocalizationStatus

    battery_soc: float | None = None
    maximum_motor_temperature_c: float | None = None

    human_distance_m: float | None = None
    nearest_obstacle_distance_m: float | None = None

    collision_probability: float | None = None

    observations: list[SafetyObservation] = field(
        default_factory=list
    )

    active_fault_codes: list[str] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.robot_id)

        validate_timestamp(self.timestamp)

        if self.battery_soc is not None:
            validate_probability(self.battery_soc)

        if self.maximum_motor_temperature_c is not None:
            validate_finite(
                self.maximum_motor_temperature_c
            )

        if self.human_distance_m is not None:
            validate_nonnegative(
                self.human_distance_m
            )

        if self.nearest_obstacle_distance_m is not None:
            validate_nonnegative(
                self.nearest_obstacle_distance_m
            )

        if self.collision_probability is not None:
            validate_probability(
                self.collision_probability
            )

        if len(self.observations) > MAX_SAFETY_OBSERVATIONS:
            raise SafetyValidationError(
                "Too many safety observations"
            )


# ============================================================
# Safety Finding
# ============================================================

@dataclass(frozen=True, slots=True)
class SafetyFinding:
    finding_id: str

    finding_type: SafetyFindingType

    risk_level: SafetyRiskLevel

    title: str
    description: str

    source: str

    confidence: float

    requires_human_review: bool

    recommendation: str

    def __post_init__(self) -> None:
        validate_identifier(self.finding_id)
        validate_probability(self.confidence)


# ============================================================
# Safety Incident
# ============================================================

@dataclass(frozen=True, slots=True)
class SafetyIncident:
    incident_id: str

    tenant_id: str
    robot_id: str

    severity: SafetyIncidentSeverity

    title: str
    description: str

    related_finding_ids: tuple[str, ...]

    detected_at: datetime

    requires_human_review: bool = True

    def __post_init__(self) -> None:
        validate_identifier(self.incident_id)
        validate_identifier(self.tenant_id)
        validate_identifier(self.robot_id)

        validate_timestamp(self.detected_at)


# ============================================================
# Safety Assessment
# ============================================================

@dataclass(frozen=True, slots=True)
class SafetyAssessment:
    assessment_id: str

    tenant_id: str
    robot_id: str

    status: SafetyAssessmentStatus

    risk_level: SafetyRiskLevel

    decision: SafetyDecision

    safety_score: float

    findings: tuple[SafetyFinding, ...]

    incidents: tuple[SafetyIncident, ...]

    requires_human_review: bool

    sensor_data_reliable: bool

    physical_execution_authorized: bool

    recommendations: tuple[str, ...]

    assessed_at: datetime

    def __post_init__(self) -> None:
        validate_identifier(self.assessment_id)
        validate_identifier(self.tenant_id)
        validate_identifier(self.robot_id)

        validate_probability(self.safety_score)
        validate_timestamp(self.assessed_at)

        if self.physical_execution_authorized:
            raise SafetyValidationError(
                "Advisory safety agent cannot authorize "
                "physical robot execution"
            )


# ============================================================
# Fleet Safety
# ============================================================

@dataclass(frozen=True, slots=True)
class FleetSafetyAssessment:
    tenant_id: str

    robot_count: int

    low_risk_count: int
    moderate_risk_count: int
    high_risk_count: int
    critical_risk_count: int
    unknown_risk_count: int

    average_safety_score: float

    robot_assessments: tuple[SafetyAssessment, ...]

    assessed_at: datetime


# ============================================================
# Provider Protocols
# ============================================================

class SafetyDataProvider(Protocol):

    async def get_robot_safety_snapshot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> RobotSafetySnapshot | None:
        ...


class SafetyPolicyProvider(Protocol):

    async def get_policy(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> SafetyPolicy:
        ...


class SafetyAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        robot_id: str,
        permission: str,
    ) -> None:
        ...


class SafetyAssessmentRepository(Protocol):

    async def save_assessment(
        self,
        *,
        assessment: SafetyAssessment,
    ) -> None:
        ...


class SafetyIncidentSink(Protocol):

    async def record_incident(
        self,
        *,
        incident: SafetyIncident,
    ) -> None:
        ...


class SafetyAuditSink(Protocol):

    async def record(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        action: str,
        resource_id: str,
        metadata: Mapping[str, Any],
    ) -> None:
        ...


# ============================================================
# Safety Agent
# ============================================================

class SafetyAgent:

    RISK_RANK = {
        SafetyRiskLevel.LOW: 1,
        SafetyRiskLevel.MODERATE: 2,
        SafetyRiskLevel.HIGH: 3,
        SafetyRiskLevel.CRITICAL: 4,
        SafetyRiskLevel.UNKNOWN: 5,
    }

    RISK_SCORE = {
        SafetyRiskLevel.LOW: 1.0,
        SafetyRiskLevel.MODERATE: 0.70,
        SafetyRiskLevel.HIGH: 0.35,
        SafetyRiskLevel.CRITICAL: 0.0,
        SafetyRiskLevel.UNKNOWN: 0.0,
    }

    def __init__(
        self,
        *,
        data_provider: SafetyDataProvider,
        authorizer: SafetyAuthorizer,
        policy_provider: SafetyPolicyProvider | None = None,
        repository: SafetyAssessmentRepository | None = None,
        incident_sink: SafetyIncidentSink | None = None,
        audit_sink: SafetyAuditSink | None = None,
    ) -> None:

        self.data_provider = data_provider
        self.authorizer = authorizer

        self.policy_provider = policy_provider
        self.repository = repository

        self.incident_sink = incident_sink
        self.audit_sink = audit_sink

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    async def _authorize(
        self,
        *,
        scope: SafetyScope,
        robot_id: str,
    ) -> None:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            robot_id=robot_id,
            permission="safety:assess",
        )

    # --------------------------------------------------------
    # Finding Builder
    # --------------------------------------------------------

    @staticmethod
    def _finding(
        *,
        finding_type: SafetyFindingType,
        risk_level: SafetyRiskLevel,
        title: str,
        description: str,
        source: str,
        recommendation: str,
        confidence: float = 1.0,
    ) -> SafetyFinding:

        return SafetyFinding(
            finding_id=generate_id(),
            finding_type=finding_type,
            risk_level=risk_level,
            title=title,
            description=description,
            source=source,
            confidence=confidence,
            requires_human_review=(
                risk_level
                in (
                    SafetyRiskLevel.HIGH,
                    SafetyRiskLevel.CRITICAL,
                    SafetyRiskLevel.UNKNOWN,
                )
            ),
            recommendation=recommendation,
        )

    # --------------------------------------------------------
    # Emergency Stop
    # --------------------------------------------------------

    def analyze_emergency_stop(
        self,
        snapshot: RobotSafetySnapshot,
    ) -> list[SafetyFinding]:

        findings: list[SafetyFinding] = []

        if snapshot.emergency_stop_active:
            findings.append(
                self._finding(
                    finding_type=(
                        SafetyFindingType.EMERGENCY_STOP
                    ),
                    risk_level=SafetyRiskLevel.CRITICAL,
                    title="Emergency stop active",
                    description=(
                        "The robot reports an active "
                        "emergency-stop condition."
                    ),
                    source="robot_safety_controller",
                    recommendation=(
                        "Keep the robot out of service. "
                        "A qualified operator must investigate "
                        "the cause and follow approved "
                        "recovery procedures."
                    ),
                )
            )

        if snapshot.safety_controller_healthy is False:
            findings.append(
                self._finding(
                    finding_type=(
                        SafetyFindingType.SAFETY_INTERLOCK
                    ),
                    risk_level=SafetyRiskLevel.CRITICAL,
                    title="Safety controller fault",
                    description=(
                        "The robot safety controller "
                        "reports an unhealthy state."
                    ),
                    source="robot_safety_controller",
                    recommendation=(
                        "Refer the robot to qualified "
                        "maintenance personnel. Do not "
                        "bypass the safety controller."
                    ),
                )
            )

        elif snapshot.safety_controller_healthy is None:
            findings.append(
                self._finding(
                    finding_type=(
                        SafetyFindingType.SAFETY_INTERLOCK
                    ),
                    risk_level=SafetyRiskLevel.UNKNOWN,
                    title="Safety controller status unknown",
                    description=(
                        "Safety controller health could "
                        "not be verified."
                    ),
                    source="robot_safety_controller",
                    recommendation=(
                        "Verify controller health through "
                        "an independent safety-rated system."
                    ),
                )
            )

        if snapshot.safety_interlock_active:
            findings.append(
                self._finding(
                    finding_type=(
                        SafetyFindingType.SAFETY_INTERLOCK
                    ),
                    risk_level=SafetyRiskLevel.HIGH,
                    title="Safety interlock active",
                    description=(
                        "A protective safety interlock "
                        "is currently active."
                    ),
                    source="robot_safety_controller",
                    recommendation=(
                        "Investigate the protective condition "
                        "without overriding the interlock."
                    ),
                )
            )

        return findings

    # --------------------------------------------------------
    # Human Proximity
    # --------------------------------------------------------

    def analyze_human_proximity(
        self,
        *,
        snapshot: RobotSafetySnapshot,
        policy: SafetyPolicy,
    ) -> list[SafetyFinding]:

        distance = snapshot.human_distance_m

        if distance is None:
            return [
                self._finding(
                    finding_type=(
                        SafetyFindingType.HUMAN_PROXIMITY
                    ),
                    risk_level=SafetyRiskLevel.UNKNOWN,
                    title="Human proximity unknown",
                    description=(
                        "Human proximity information "
                        "is unavailable."
                    ),
                    source="perception",
                    recommendation=(
                        "Verify human presence using "
                        "approved protective sensing."
                    ),
                )
            ]

        if distance <= policy.human_critical_distance_m:
            return [
                self._finding(
                    finding_type=(
                        SafetyFindingType.HUMAN_PROXIMITY
                    ),
                    risk_level=SafetyRiskLevel.CRITICAL,
                    title="Human within critical distance",
                    description=(
                        f"Reported human distance is "
                        f"{distance:.2f} meters."
                    ),
                    source="perception",
                    recommendation=(
                        "Treat the situation as hazardous. "
                        "Protective actions must be handled "
                        "by independent safety-rated systems."
                    ),
                )
            ]

        if distance <= policy.human_warning_distance_m:
            return [
                self._finding(
                    finding_type=(
                        SafetyFindingType.HUMAN_PROXIMITY
                    ),
                    risk_level=SafetyRiskLevel.HIGH,
                    title="Human near robot workspace",
                    description=(
                        f"Reported human distance is "
                        f"{distance:.2f} meters."
                    ),
                    source="perception",
                    recommendation=(
                        "Review the separation distance "
                        "and applicable safety procedures."
                    ),
                )
            ]

        return []

    # --------------------------------------------------------
    # Collision Risk
    # --------------------------------------------------------

    def analyze_collision_risk(
        self,
        *,
        snapshot: RobotSafetySnapshot,
        policy: SafetyPolicy,
    ) -> list[SafetyFinding]:

        probability = snapshot.collision_probability

        if probability is None:
            return [
                self._finding(
                    finding_type=(
                        SafetyFindingType.COLLISION_RISK
                    ),
                    risk_level=SafetyRiskLevel.UNKNOWN,
                    title="Collision risk unknown",
                    description=(
                        "Collision probability is unavailable."
                    ),
                    source="collision_prediction",
                    recommendation=(
                        "Verify the environment and planned "
                        "trajectory with independent safeguards."
                    ),
                )
            ]

        if (
            probability
            >= policy.collision_critical_probability
        ):
            risk = SafetyRiskLevel.CRITICAL

        elif (
            probability
            >= policy.collision_warning_probability
        ):
            risk = SafetyRiskLevel.HIGH

        else:
            return []

        return [
            self._finding(
                finding_type=(
                    SafetyFindingType.COLLISION_RISK
                ),
                risk_level=risk,
                title="Elevated collision probability",
                description=(
                    f"Estimated collision probability: "
                    f"{probability:.1%}."
                ),
                source="collision_prediction",
                recommendation=(
                    "Reject the proposed trajectory for "
                    "advisory planning and request independent "
                    "collision-safety verification."
                ),
                confidence=probability,
            )
        ]

    # --------------------------------------------------------
    # Battery Safety
    # --------------------------------------------------------

    def analyze_battery(
        self,
        *,
        snapshot: RobotSafetySnapshot,
        policy: SafetyPolicy,
    ) -> list[SafetyFinding]:

        soc = snapshot.battery_soc

        if soc is None:
            return [
                self._finding(
                    finding_type=SafetyFindingType.BATTERY,
                    risk_level=SafetyRiskLevel.UNKNOWN,
                    title="Battery state unavailable",
                    description=(
                        "Battery state of charge is unknown."
                    ),
                    source="battery_management",
                    recommendation=(
                        "Verify battery telemetry before "
                        "planning energy-dependent operations."
                    ),
                )
            ]

        if soc <= policy.battery_critical_soc:
            risk = SafetyRiskLevel.CRITICAL

        elif soc <= policy.battery_warning_soc:
            risk = SafetyRiskLevel.MODERATE

        else:
            return []

        return [
            self._finding(
                finding_type=SafetyFindingType.BATTERY,
                risk_level=risk,
                title="Low battery state of charge",
                description=(
                    f"Battery state of charge: {soc:.1%}."
                ),
                source="battery_management",
                recommendation=(
                    "Review charging requirements and "
                    "battery condition before operation."
                ),
            )
        ]

    # --------------------------------------------------------
    # Thermal Safety
    # --------------------------------------------------------

    def analyze_temperature(
        self,
        *,
        snapshot: RobotSafetySnapshot,
        policy: SafetyPolicy,
    ) -> list[SafetyFinding]:

        temperature = (
            snapshot.maximum_motor_temperature_c
        )

        if temperature is None:
            return [
                self._finding(
                    finding_type=SafetyFindingType.THERMAL,
                    risk_level=SafetyRiskLevel.UNKNOWN,
                    title="Motor temperature unavailable",
                    description=(
                        "Motor thermal condition cannot "
                        "be verified."
                    ),
                    source="motor_telemetry",
                    recommendation=(
                        "Verify motor temperature and "
                        "thermal sensor health."
                    ),
                )
            ]

        if temperature >= policy.temperature_critical_c:
            risk = SafetyRiskLevel.CRITICAL

        elif temperature >= policy.temperature_warning_c:
            risk = SafetyRiskLevel.HIGH

        else:
            return []

        return [
            self._finding(
                finding_type=SafetyFindingType.THERMAL,
                risk_level=risk,
                title="Elevated motor temperature",
                description=(
                    f"Reported maximum motor temperature: "
                    f"{temperature:.1f} °C."
                ),
                source="motor_telemetry",
                recommendation=(
                    "Request qualified inspection of "
                    "motor loading, cooling and thermal "
                    "protection systems."
                ),
            )
        ]

    # --------------------------------------------------------
    # Localization
    # --------------------------------------------------------

    def analyze_localization(
        self,
        snapshot: RobotSafetySnapshot,
    ) -> list[SafetyFinding]:

        if snapshot.localization_status == (
            LocalizationStatus.LOST
        ):
            risk = SafetyRiskLevel.CRITICAL

        elif snapshot.localization_status == (
            LocalizationStatus.DEGRADED
        ):
            risk = SafetyRiskLevel.HIGH

        elif snapshot.localization_status == (
            LocalizationStatus.UNKNOWN
        ):
            risk = SafetyRiskLevel.UNKNOWN

        else:
            return []

        return [
            self._finding(
                finding_type=(
                    SafetyFindingType.LOCALIZATION_FAILURE
                ),
                risk_level=risk,
                title="Localization integrity concern",
                description=(
                    "Robot localization status: "
                    f"{snapshot.localization_status.value}."
                ),
                source="localization",
                recommendation=(
                    "Do not rely on unverified localization "
                    "for autonomous navigation planning."
                ),
            )
        ]

    # --------------------------------------------------------
    # Sensor Reliability
    # --------------------------------------------------------

    def analyze_sensor_reliability(
        self,
        *,
        snapshot: RobotSafetySnapshot,
        policy: SafetyPolicy,
    ) -> tuple[list[SafetyFinding], bool]:

        findings: list[SafetyFinding] = []

        now = utc_now()
        reliable = True

        if not snapshot.observations:
            reliable = False

            findings.append(
                self._finding(
                    finding_type=(
                        SafetyFindingType.SENSOR_FAILURE
                    ),
                    risk_level=SafetyRiskLevel.UNKNOWN,
                    title="No safety sensor observations",
                    description=(
                        "No independent safety observations "
                        "were supplied."
                    ),
                    source="sensor_monitoring",
                    recommendation=(
                        "Verify sensor availability and "
                        "independent protective systems."
                    ),
                )
            )

        for observation in snapshot.observations:
            age = (
                now - observation.timestamp
            ).total_seconds()

            if (
                age > policy.sensor_stale_seconds
                or age < -policy.sensor_stale_seconds
            ):
                reliable = False

                findings.append(
                    self._finding(
                        finding_type=(
                            SafetyFindingType.STALE_SENSOR_DATA
                        ),
                        risk_level=SafetyRiskLevel.UNKNOWN,
                        title="Stale or invalid sensor timestamp",
                        description=(
                            f"Sensor {observation.sensor_id} "
                            "has an unacceptable timestamp."
                        ),
                        source=observation.sensor_id,
                        recommendation=(
                            "Verify sensor synchronization "
                            "and data freshness."
                        ),
                    )
                )

            if observation.sensor_status != (
                SafetySensorStatus.HEALTHY
            ):
                reliable = False

                findings.append(
                    self._finding(
                        finding_type=(
                            SafetyFindingType.SENSOR_FAILURE
                        ),
                        risk_level=SafetyRiskLevel.UNKNOWN,
                        title="Sensor health concern",
                        description=(
                            f"Sensor {observation.sensor_id} "
                            f"status: "
                            f"{observation.sensor_status.value}."
                        ),
                        source=observation.sensor_id,
                        recommendation=(
                            "Investigate sensor health "
                            "before relying on its measurements."
                        ),
                    )
                )

            if (
                observation.confidence
                < policy.minimum_sensor_confidence
            ):
                reliable = False

                findings.append(
                    self._finding(
                        finding_type=(
                            SafetyFindingType.SENSOR_FAILURE
                        ),
                        risk_level=SafetyRiskLevel.UNKNOWN,
                        title="Low-confidence sensor data",
                        description=(
                            f"Sensor {observation.sensor_id} "
                            f"confidence: "
                            f"{observation.confidence:.1%}."
                        ),
                        source=observation.sensor_id,
                        recommendation=(
                            "Verify measurements using "
                            "independent sensing."
                        ),
                        confidence=observation.confidence,
                    )
                )

        return findings, reliable

    # --------------------------------------------------------
    # Fault Analysis
    # --------------------------------------------------------

    def analyze_faults(
        self,
        snapshot: RobotSafetySnapshot,
    ) -> list[SafetyFinding]:

        if not snapshot.active_fault_codes:
            return []

        return [
            self._finding(
                finding_type=SafetyFindingType.ROBOT_HEALTH,
                risk_level=SafetyRiskLevel.HIGH,
                title="Active robot fault codes",
                description=(
                    f"{len(snapshot.active_fault_codes)} "
                    "active fault code(s) reported."
                ),
                source="robot_diagnostics",
                recommendation=(
                    "Have qualified personnel investigate "
                    "faults using manufacturer-approved "
                    "diagnostic procedures."
                ),
            )
        ]

    # --------------------------------------------------------
    # Overall Risk
    # --------------------------------------------------------

    @classmethod
    def determine_overall_risk(
        cls,
        findings: Sequence[SafetyFinding],
        sensor_reliable: bool,
    ) -> SafetyRiskLevel:

        if any(
            finding.risk_level
            == SafetyRiskLevel.CRITICAL
            for finding in findings
        ):
            return SafetyRiskLevel.CRITICAL

        if not sensor_reliable or any(
            finding.risk_level
            == SafetyRiskLevel.UNKNOWN
            for finding in findings
        ):
            return SafetyRiskLevel.UNKNOWN

        if any(
            finding.risk_level == SafetyRiskLevel.HIGH
            for finding in findings
        ):
            return SafetyRiskLevel.HIGH

        if any(
            finding.risk_level
            == SafetyRiskLevel.MODERATE
            for finding in findings
        ):
            return SafetyRiskLevel.MODERATE

        return SafetyRiskLevel.LOW

    # --------------------------------------------------------
    # Incident Generation
    # --------------------------------------------------------

    @staticmethod
    def generate_incidents(
        *,
        snapshot: RobotSafetySnapshot,
        findings: Sequence[SafetyFinding],
    ) -> list[SafetyIncident]:

        incidents: list[SafetyIncident] = []

        for finding in findings:
            if finding.risk_level not in (
                SafetyRiskLevel.HIGH,
                SafetyRiskLevel.CRITICAL,
            ):
                continue

            severity = (
                SafetyIncidentSeverity.CRITICAL
                if finding.risk_level
                == SafetyRiskLevel.CRITICAL
                else SafetyIncidentSeverity.MAJOR
            )

            incidents.append(
                SafetyIncident(
                    incident_id=generate_id(),
                    tenant_id=snapshot.tenant_id,
                    robot_id=snapshot.robot_id,
                    severity=severity,
                    title=finding.title,
                    description=finding.description,
                    related_finding_ids=(
                        finding.finding_id,
                    ),
                    detected_at=utc_now(),
                )
            )

        return incidents

    # --------------------------------------------------------
    # Main Safety Assessment
    # --------------------------------------------------------

    async def assess_robot_safety(
        self,
        *,
        scope: SafetyScope,
        robot_id: str,
    ) -> SafetyAssessment:

        validate_identifier(robot_id)

        await self._authorize(
            scope=scope,
            robot_id=robot_id,
        )

        snapshot = (
            await self.data_provider.get_robot_safety_snapshot(
                tenant_id=scope.tenant_id,
                robot_id=robot_id,
            )
        )

        if snapshot is None:
            raise SafetyNotFoundError(
                "Robot safety snapshot not found"
            )

        if (
            snapshot.tenant_id != scope.tenant_id
            or snapshot.robot_id != robot_id
        ):
            raise SafetyPermissionError(
                "Safety snapshot scope mismatch"
            )

        if self.policy_provider is not None:
            policy = await self.policy_provider.get_policy(
                tenant_id=scope.tenant_id,
                robot_id=robot_id,
            )
        else:
            policy = SafetyPolicy(
                policy_id="default"
            )

        findings: list[SafetyFinding] = []

        findings.extend(
            self.analyze_emergency_stop(snapshot)
        )

        findings.extend(
            self.analyze_human_proximity(
                snapshot=snapshot,
                policy=policy,
            )
        )

        findings.extend(
            self.analyze_collision_risk(
                snapshot=snapshot,
                policy=policy,
            )
        )

        findings.extend(
            self.analyze_battery(
                snapshot=snapshot,
                policy=policy,
            )
        )

        findings.extend(
            self.analyze_temperature(
                snapshot=snapshot,
                policy=policy,
            )
        )

        findings.extend(
            self.analyze_localization(snapshot)
        )

        sensor_findings, sensor_reliable = (
            self.analyze_sensor_reliability(
                snapshot=snapshot,
                policy=policy,
            )
        )

        findings.extend(sensor_findings)
        findings.extend(self.analyze_faults(snapshot))

        snapshot_age = (
            utc_now() - snapshot.timestamp
        ).total_seconds()

        if (
            snapshot_age > policy.sensor_stale_seconds
            or snapshot_age < -policy.sensor_stale_seconds
        ):
            sensor_reliable = False

            findings.append(
                self._finding(
                    finding_type=(
                        SafetyFindingType.STALE_SENSOR_DATA
                    ),
                    risk_level=SafetyRiskLevel.UNKNOWN,
                    title="Safety snapshot is stale",
                    description=(
                        "Robot safety snapshot is outside "
                        "the permitted freshness window."
                    ),
                    source="robot_safety_snapshot",
                    recommendation=(
                        "Acquire fresh safety telemetry."
                    ),
                )
            )

        if len(findings) > MAX_SAFETY_FINDINGS:
            raise SafetyValidationError(
                "Safety finding limit exceeded"
            )

        risk = self.determine_overall_risk(
            findings=findings,
            sensor_reliable=sensor_reliable,
        )

        requires_review = (
            risk != SafetyRiskLevel.LOW
        )

        if risk == SafetyRiskLevel.CRITICAL:
            status = (
                SafetyAssessmentStatus.CRITICAL_FINDINGS
            )
            decision = SafetyDecision.REJECT_PROPOSED_PLAN

        elif risk == SafetyRiskLevel.UNKNOWN:
            status = (
                SafetyAssessmentStatus.INSUFFICIENT_DATA
            )
            decision = (
                SafetyDecision.INSUFFICIENT_EVIDENCE
            )

        elif requires_review:
            status = (
                SafetyAssessmentStatus.REVIEW_REQUIRED
            )
            decision = SafetyDecision.REVIEW_REQUIRED

        else:
            status = SafetyAssessmentStatus.COMPLETED
            decision = SafetyDecision.ADVISORY_LOW_RISK

        incidents = self.generate_incidents(
            snapshot=snapshot,
            findings=findings,
        )

        recommendations = tuple(
            dict.fromkeys(
                finding.recommendation
                for finding in findings
            )
        )

        assessment = SafetyAssessment(
            assessment_id=generate_id(),
            tenant_id=scope.tenant_id,
            robot_id=robot_id,
            status=status,
            risk_level=risk,
            decision=decision,
            safety_score=self.RISK_SCORE[risk],
            findings=tuple(findings),
            incidents=tuple(incidents),
            requires_human_review=requires_review,
            sensor_data_reliable=sensor_reliable,
            physical_execution_authorized=False,
            recommendations=recommendations,
            assessed_at=utc_now(),
        )

        if self.repository is not None:
            await self.repository.save_assessment(
                assessment=assessment
            )

        if self.incident_sink is not None:
            for incident in incidents:
                await self.incident_sink.record_incident(
                    incident=incident
                )

        if self.audit_sink is not None:
            await self.audit_sink.record(
                tenant_id=scope.tenant_id,
                actor_id=scope.user_id,
                action="safety_agent.robot_assessed",
                resource_id=assessment.assessment_id,
                metadata={
                    "robot_id": robot_id,
                    "risk_level": risk.value,
                    "decision": decision.value,
                    "finding_count": len(findings),
                    "incident_count": len(incidents),
                    "physical_execution_authorized": False,
                },
            )

        return assessment

    # --------------------------------------------------------
    # Fleet Safety
    # --------------------------------------------------------

    async def assess_fleet_safety(
        self,
        *,
        scope: SafetyScope,
        robot_ids: Sequence[str],
    ) -> FleetSafetyAssessment:

        if len(robot_ids) > MAX_BATCH_ASSESSMENTS:
            raise SafetyValidationError(
                "Fleet assessment limit exceeded"
            )

        if len(set(robot_ids)) != len(robot_ids):
            raise SafetyValidationError(
                "Duplicate robot identifiers"
            )

        semaphore = asyncio.Semaphore(5)

        async def assess(
            robot_id: str,
        ) -> SafetyAssessment:

            async with semaphore:
                return await self.assess_robot_safety(
                    scope=scope,
                    robot_id=robot_id,
                )

        assessments = list(
            await asyncio.gather(
                *(assess(robot_id) for robot_id in robot_ids)
            )
        )

        def count_risk(
            risk: SafetyRiskLevel,
        ) -> int:
            return sum(
                assessment.risk_level == risk
                for assessment in assessments
            )

        average_score = (
            sum(
                assessment.safety_score
                for assessment in assessments
            ) / len(assessments)
            if assessments
            else 0.0
        )

        return FleetSafetyAssessment(
            tenant_id=scope.tenant_id,
            robot_count=len(assessments),
            low_risk_count=count_risk(
                SafetyRiskLevel.LOW
            ),
            moderate_risk_count=count_risk(
                SafetyRiskLevel.MODERATE
            ),
            high_risk_count=count_risk(
                SafetyRiskLevel.HIGH
            ),
            critical_risk_count=count_risk(
                SafetyRiskLevel.CRITICAL
            ),
            unknown_risk_count=count_risk(
                SafetyRiskLevel.UNKNOWN
            ),
            average_safety_score=round(
                average_score, 4
            ),
            robot_assessments=tuple(assessments),
            assessed_at=utc_now(),
        )

    # --------------------------------------------------------
    # Safety Report
    # --------------------------------------------------------

    @staticmethod
    def generate_safety_report(
        assessment: SafetyAssessment,
    ) -> dict[str, Any]:

        return {
            "assessment_id": assessment.assessment_id,
            "tenant_id": assessment.tenant_id,
            "robot_id": assessment.robot_id,
            "status": assessment.status.value,
            "risk_level": assessment.risk_level.value,
            "decision": assessment.decision.value,
            "safety_score": assessment.safety_score,
            "assessed_at": (
                assessment.assessed_at.isoformat()
            ),
            "sensor_data_reliable": (
                assessment.sensor_data_reliable
            ),
            "requires_human_review": (
                assessment.requires_human_review
            ),
            "findings": [
                {
                    "finding_id": finding.finding_id,
                    "finding_type": (
                        finding.finding_type.value
                    ),
                    "risk_level": (
                        finding.risk_level.value
                    ),
                    "title": finding.title,
                    "description": finding.description,
                    "source": finding.source,
                    "confidence": finding.confidence,
                    "requires_human_review": (
                        finding.requires_human_review
                    ),
                    "recommendation": (
                        finding.recommendation
                    ),
                }
                for finding in assessment.findings
            ],
            "incidents": [
                {
                    "incident_id": incident.incident_id,
                    "severity": incident.severity.value,
                    "title": incident.title,
                    "description": incident.description,
                    "related_finding_ids": list(
                        incident.related_finding_ids
                    ),
                    "detected_at": (
                        incident.detected_at.isoformat()
                    ),
                }
                for incident in assessment.incidents
            ],
            "recommendations": list(
                assessment.recommendations
            ),
            "safety_controls": {
                "physical_execution_authorized": False,
                "emergency_stop_reset_allowed": False,
                "safety_interlock_override_allowed": False,
                "automatic_robot_actuation": False,
                "independent_safety_validation_required": True,
                "safety_certification": False,
            },
        }


# ============================================================
# Factory
# ============================================================

def create_safety_agent(
    *,
    data_provider: SafetyDataProvider,
    authorizer: SafetyAuthorizer,
    policy_provider: SafetyPolicyProvider | None = None,
    repository: SafetyAssessmentRepository | None = None,
    incident_sink: SafetyIncidentSink | None = None,
    audit_sink: SafetyAuditSink | None = None,
) -> SafetyAgent:

    return SafetyAgent(
        data_provider=data_provider,
        authorizer=authorizer,
        policy_provider=policy_provider,
        repository=repository,
        incident_sink=incident_sink,
        audit_sink=audit_sink,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "SafetyAgentError",
    "SafetyValidationError",
    "SafetyPermissionError",
    "SafetyNotFoundError",
    "SafetyProviderError",
    "SafetyPolicyError",
    "SafetyServiceUnavailableError",
    "SafetyRiskLevel",
    "SafetyAssessmentStatus",
    "SafetyFindingType",
    "SafetyObservationType",
    "SafetySensorStatus",
    "SafetyDecision",
    "SafetyIncidentSeverity",
    "RobotOperationalState",
    "LocalizationStatus",
    "SafetyScope",
    "SafetyPolicy",
    "SafetyObservation",
    "RobotSafetySnapshot",
    "SafetyFinding",
    "SafetyIncident",
    "SafetyAssessment",
    "FleetSafetyAssessment",
    "SafetyDataProvider",
    "SafetyPolicyProvider",
    "SafetyAuthorizer",
    "SafetyAssessmentRepository",
    "SafetyIncidentSink",
    "SafetyAuditSink",
    "SafetyAgent",
    "create_safety_agent",
]
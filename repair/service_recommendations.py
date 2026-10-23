"""Service Recommendations module.

TODO: implement and test; no real robot control is enabled.
"""


APP_ENV=development
ALLOW_PHYSICAL_ACTUATION=false
# backend/app/robotics/control/service_recommendations.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robotics Service Recommendations Engine

Python 3.11+
Dependencies: Standard library only

Features:
    - Robot health-based service recommendations
    - Predictive and preventive maintenance recommendations
    - Fault and damage-based service planning
    - Emergency inspection recommendations
    - Component repair and replacement recommendations
    - Battery, motor, gearbox, sensor and actuator servicing
    - Remaining useful life (RUL) analysis
    - Maintenance scheduling recommendations
    - Service urgency and priority classification
    - Service cost and downtime estimation
    - Technician skill recommendations
    - Spare parts and replacement planning
    - Warranty and service provider considerations
    - Service history and recurring fault analysis
    - Human review and approval workflow
    - Multi-tenant isolation
    - Thread-safe recommendation storage
    - Audit logging
    - JSON export

Integrations:
    issue_diagnosis.py
    damage_analysis.py
    damage_severity.py
    image_inspection.py
    labor_cost.py
    downtime_cost.py
    parts_catalog.py
    parts_pricing.py
    repair_vs_replace.py
    repair_report.py
    replacement_advisor.py
    maintenance_service.py

Safety:
    This module provides advisory recommendations only.
    It does not authorize repairs, robot movement,
    equipment restart, or safety-critical operations.

    Production authentication and authorization must
    be enforced by the calling application.
"""

from __future__ import annotations

import json
import threading
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class ServiceRecommendationError(Exception):
    pass


class ServiceValidationError(ServiceRecommendationError):
    pass


class ServiceRecommendationNotFound(ServiceRecommendationError):
    pass


class ServiceConflict(ServiceRecommendationError):
    pass


class ServicePermissionDenied(ServiceRecommendationError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RobotComponentType(str, Enum):
    MOTOR = "motor"
    GEARBOX = "gearbox"
    BEARING = "bearing"
    ACTUATOR = "actuator"
    BATTERY = "battery"
    SENSOR = "sensor"
    CAMERA = "camera"
    LIDAR = "lidar"
    CONTROLLER = "controller"
    POWER_SYSTEM = "power_system"
    COOLING_SYSTEM = "cooling_system"
    BRAKING_SYSTEM = "braking_system"
    MANIPULATOR = "manipulator"
    END_EFFECTOR = "end_effector"
    WHEEL = "wheel"
    JOINT = "joint"
    STRUCTURE = "structure"
    NETWORK = "network"
    SOFTWARE = "software"
    OTHER = "other"


class ComponentCondition(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"
    UNKNOWN = "unknown"


class FaultSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentCriticality(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SAFETY_CRITICAL = "safety_critical"


class ServiceType(str, Enum):
    INSPECTION = "inspection"
    PREVENTIVE_MAINTENANCE = "preventive_maintenance"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    CORRECTIVE_MAINTENANCE = "corrective_maintenance"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    CALIBRATION = "calibration"
    LUBRICATION = "lubrication"
    CLEANING = "cleaning"
    DIAGNOSTIC_TEST = "diagnostic_test"
    SAFETY_INSPECTION = "safety_inspection"
    SOFTWARE_DIAGNOSTICS = "software_diagnostics"
    FIRMWARE_REVIEW = "firmware_review"
    BATTERY_SERVICE = "battery_service"
    EMERGENCY_ASSESSMENT = "emergency_assessment"


class ServicePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class RecommendationStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class TechnicianSpecialty(str, Enum):
    GENERAL_MAINTENANCE = "general_maintenance"
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    ROBOTICS = "robotics"
    BATTERY = "battery"
    CONTROL_SYSTEMS = "control_systems"
    SOFTWARE = "software"
    SAFETY_ENGINEERING = "safety_engineering"


class EvidenceSource(str, Enum):
    TELEMETRY = "telemetry"
    DIAGNOSTICS = "diagnostics"
    IMAGE_INSPECTION = "image_inspection"
    MAINTENANCE_HISTORY = "maintenance_history"
    HUMAN_INSPECTION = "human_inspection"
    PREDICTIVE_MODEL = "predictive_model"
    OTHER = "other"


# ============================================================
# UTILITIES
# ============================================================

MONEY_QUANTUM = Decimal("0.01")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or not all(
            char.isalnum() or char in "._-"
            for char in value
        )
    ):
        raise ServiceValidationError(
            f"Invalid {name}"
        )

    return value


def validate_text(
    value: str,
    name: str,
    *,
    allow_empty: bool = False,
    maximum: int = 4000,
) -> str:
    if not isinstance(value, str):
        raise ServiceValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if len(result) > maximum:
        raise ServiceValidationError(
            f"{name} exceeds maximum length"
        )

    if not allow_empty and not result:
        raise ServiceValidationError(
            f"{name} cannot be empty"
        )

    return result


def decimal_value(
    value: Any,
    name: str,
    *,
    minimum: Decimal = Decimal("0"),
    maximum: Decimal = Decimal("1000000000000"),
) -> Decimal:
    if isinstance(value, bool):
        raise ServiceValidationError(
            f"Invalid {name}"
        )

    try:
        result = Decimal(str(value))
    except (
        InvalidOperation,
        TypeError,
        ValueError,
    ) as exc:
        raise ServiceValidationError(
            f"Invalid {name}"
        ) from exc

    if (
        not result.is_finite()
        or result < minimum
        or result > maximum
    ):
        raise ServiceValidationError(
            f"Invalid {name}"
        )

    return result


def money(value: Any, name: str = "amount") -> Decimal:
    return decimal_value(value, name).quantize(
        MONEY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def money_string(value: Decimal) -> str:
    return str(
        value.quantize(
            MONEY_QUANTUM,
            rounding=ROUND_HALF_UP,
        )
    )


def validate_currency(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 3
        or not value.isascii()
        or not value.isalpha()
        or value != value.upper()
    ):
        raise ServiceValidationError(
            "Currency must be a three-letter uppercase code"
        )

    return value


def validate_integer(
    value: int,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 1000000,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ServiceValidationError(
            f"Invalid {name}"
        )

    return value


def validate_datetime(
    value: datetime,
    name: str,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ServiceValidationError(
            f"{name} must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def optional_decimal(
    value: Any | None,
    name: str,
    *,
    maximum: Decimal,
) -> Decimal | None:
    if value is None:
        return None

    return decimal_value(
        value,
        name,
        maximum=maximum,
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ServiceRecommendationConfig:
    default_currency: str = "INR"

    low_health_threshold: Decimal = Decimal("40")
    critical_health_threshold: Decimal = Decimal("20")

    low_rul_days: Decimal = Decimal("90")
    critical_rul_days: Decimal = Decimal("14")

    high_failure_probability: Decimal = Decimal("0.25")
    critical_failure_probability: Decimal = Decimal("0.50")

    maximum_recommendations: int = 100000
    maximum_audit_events: int = 20000
    maximum_evidence_per_request: int = 100
    maximum_recommendations_per_request: int = 50

    simulation_only: bool = True

    def __post_init__(self) -> None:
        validate_currency(self.default_currency)

        for name in (
            "low_health_threshold",
            "critical_health_threshold",
        ):
            decimal_value(
                getattr(self, name),
                name,
                maximum=Decimal("100"),
            )

        if (
            self.critical_health_threshold
            > self.low_health_threshold
        ):
            raise ServiceValidationError(
                "Critical health threshold cannot "
                "exceed low health threshold"
            )

        for name in (
            "low_rul_days",
            "critical_rul_days",
        ):
            decimal_value(
                getattr(self, name),
                name,
                maximum=Decimal("36500"),
            )

        if self.critical_rul_days > self.low_rul_days:
            raise ServiceValidationError(
                "Critical RUL threshold cannot "
                "exceed low RUL threshold"
            )

        for name in (
            "high_failure_probability",
            "critical_failure_probability",
        ):
            decimal_value(
                getattr(self, name),
                name,
                maximum=Decimal("1"),
            )

        if (
            self.critical_failure_probability
            < self.high_failure_probability
        ):
            raise ServiceValidationError(
                "Critical failure threshold must be "
                "at least the high failure threshold"
            )

        for name in (
            "maximum_recommendations",
            "maximum_audit_events",
            "maximum_evidence_per_request",
            "maximum_recommendations_per_request",
        ):
            validate_integer(
                getattr(self, name),
                name,
                minimum=1,
            )

        if self.simulation_only is not True:
            raise ServiceValidationError(
                "Only advisory mode is supported"
            )


# ============================================================
# ROBOT COMPONENT HEALTH
# ============================================================

@dataclass(frozen=True)
class ServiceComponentProfile:
    tenant_id: str
    robot_id: str
    component_id: str
    component_name: str

    component_type: RobotComponentType

    condition: ComponentCondition = (
        ComponentCondition.UNKNOWN
    )

    criticality: ComponentCriticality = (
        ComponentCriticality.MODERATE
    )

    fault_severity: FaultSeverity = FaultSeverity.UNKNOWN

    health_score: Decimal | None = None
    remaining_useful_life_days: Decimal | None = None
    failure_probability: Decimal | None = None

    operating_hours: Decimal = Decimal("0")
    hours_since_last_service: Decimal | None = None
    recommended_service_interval_hours: Decimal | None = None

    recurring_fault_count: int = 0

    fault_codes: tuple[str, ...] = ()

    manufacturer: str = ""
    model_number: str = ""

    warranty_active: bool = False
    safety_fault_active: bool = False

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "component_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        validate_text(
            self.component_name,
            "component_name",
        )

        for name, enum_type in (
            ("component_type", RobotComponentType),
            ("condition", ComponentCondition),
            ("criticality", ComponentCriticality),
            ("fault_severity", FaultSeverity),
        ):
            if not isinstance(
                getattr(self, name),
                enum_type,
            ):
                raise ServiceValidationError(
                    f"Invalid {name}"
                )

        if self.health_score is not None:
            decimal_value(
                self.health_score,
                "health_score",
                maximum=Decimal("100"),
            )

        if self.remaining_useful_life_days is not None:
            decimal_value(
                self.remaining_useful_life_days,
                "remaining_useful_life_days",
                maximum=Decimal("36500"),
            )

        if self.failure_probability is not None:
            decimal_value(
                self.failure_probability,
                "failure_probability",
                maximum=Decimal("1"),
            )

        decimal_value(
            self.operating_hours,
            "operating_hours",
            maximum=Decimal("10000000"),
        )

        if self.hours_since_last_service is not None:
            decimal_value(
                self.hours_since_last_service,
                "hours_since_last_service",
                maximum=Decimal("10000000"),
            )

        if self.recommended_service_interval_hours is not None:
            decimal_value(
                self.recommended_service_interval_hours,
                "recommended_service_interval_hours",
                minimum=Decimal("0.01"),
                maximum=Decimal("10000000"),
            )

        validate_integer(
            self.recurring_fault_count,
            "recurring_fault_count",
        )

        if not isinstance(self.fault_codes, tuple):
            raise ServiceValidationError(
                "fault_codes must be a tuple"
            )

        for code in self.fault_codes:
            validate_text(
                code,
                "fault_code",
                maximum=128,
            )

        for name in (
            "manufacturer",
            "model_number",
        ):
            validate_text(
                getattr(self, name),
                name,
                allow_empty=True,
            )

        if not isinstance(self.warranty_active, bool):
            raise ServiceValidationError(
                "warranty_active must be boolean"
            )

        if not isinstance(self.safety_fault_active, bool):
            raise ServiceValidationError(
                "safety_fault_active must be boolean"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "component_id": self.component_id,
            "component_name": self.component_name,
            "component_type": self.component_type.value,
            "condition": self.condition.value,
            "criticality": self.criticality.value,
            "fault_severity": self.fault_severity.value,
            "health_score": (
                str(self.health_score)
                if self.health_score is not None
                else None
            ),
            "remaining_useful_life_days": (
                str(self.remaining_useful_life_days)
                if self.remaining_useful_life_days is not None
                else None
            ),
            "failure_probability": (
                str(self.failure_probability)
                if self.failure_probability is not None
                else None
            ),
            "operating_hours": str(self.operating_hours),
            "hours_since_last_service": (
                str(self.hours_since_last_service)
                if self.hours_since_last_service is not None
                else None
            ),
            "recommended_service_interval_hours": (
                str(self.recommended_service_interval_hours)
                if self.recommended_service_interval_hours
                is not None
                else None
            ),
            "recurring_fault_count": (
                self.recurring_fault_count
            ),
            "fault_codes": list(self.fault_codes),
            "manufacturer": self.manufacturer,
            "model_number": self.model_number,
            "warranty_active": self.warranty_active,
            "safety_fault_active": self.safety_fault_active,
        }


# ============================================================
# SERVICE EVIDENCE
# ============================================================

@dataclass(frozen=True)
class ServiceEvidence:
    evidence_id: str
    source: EvidenceSource
    description: str

    reference_id: str = ""
    confidence: Decimal | None = None

    captured_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.evidence_id, "evidence_id")

        if not isinstance(self.source, EvidenceSource):
            raise ServiceValidationError(
                "Invalid evidence source"
            )

        validate_text(
            self.description,
            "description",
        )

        if self.reference_id:
            validate_id(
                self.reference_id,
                "reference_id",
            )

        if self.confidence is not None:
            decimal_value(
                self.confidence,
                "confidence",
                maximum=Decimal("1"),
            )

        validate_datetime(
            self.captured_at,
            "captured_at",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source.value,
            "description": self.description,
            "reference_id": self.reference_id,
            "confidence": (
                str(self.confidence)
                if self.confidence is not None
                else None
            ),
            "captured_at": self.captured_at.isoformat(),
        }


# ============================================================
# SERVICE COST ESTIMATE
# ============================================================

@dataclass(frozen=True)
class ServiceCostEstimate:
    currency: str = "INR"

    labor_cost: Decimal = Decimal("0")
    parts_cost: Decimal = Decimal("0")
    downtime_cost: Decimal = Decimal("0")
    travel_cost: Decimal = Decimal("0")
    inspection_cost: Decimal = Decimal("0")
    other_cost: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        validate_currency(self.currency)

        for name in (
            "labor_cost",
            "parts_cost",
            "downtime_cost",
            "travel_cost",
            "inspection_cost",
            "other_cost",
        ):
            object.__setattr__(
                self,
                name,
                money(
                    getattr(self, name),
                    name,
                ),
            )

    @property
    def total(self) -> Decimal:
        return sum(
            (
                self.labor_cost,
                self.parts_cost,
                self.downtime_cost,
                self.travel_cost,
                self.inspection_cost,
                self.other_cost,
            ),
            Decimal("0"),
        ).quantize(
            MONEY_QUANTUM,
            rounding=ROUND_HALF_UP,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "labor_cost": money_string(self.labor_cost),
            "parts_cost": money_string(self.parts_cost),
            "downtime_cost": money_string(
                self.downtime_cost
            ),
            "travel_cost": money_string(self.travel_cost),
            "inspection_cost": money_string(
                self.inspection_cost
            ),
            "other_cost": money_string(self.other_cost),
            "total": money_string(self.total),
            "estimate_only": True,
        }


# ============================================================
# SERVICE REQUEST
# ============================================================

@dataclass(frozen=True)
class ServiceRecommendationRequest:
    component: ServiceComponentProfile

    evidence: tuple[ServiceEvidence, ...] = ()

    cost_estimate: ServiceCostEstimate | None = None

    replacement_advice_id: str = ""
    repair_report_id: str = ""
    diagnosis_id: str = ""
    maintenance_record_id: str = ""

    preferred_service_date: datetime | None = None

    budget: Decimal | None = None
    currency: str = "INR"

    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(
            self.component,
            ServiceComponentProfile,
        ):
            raise ServiceValidationError(
                "Invalid component profile"
            )

        if not isinstance(self.evidence, tuple):
            raise ServiceValidationError(
                "evidence must be a tuple"
            )

        seen = set()

        for item in self.evidence:
            if not isinstance(item, ServiceEvidence):
                raise ServiceValidationError(
                    "Invalid service evidence"
                )

            if item.evidence_id in seen:
                raise ServiceValidationError(
                    "Duplicate evidence ID"
                )

            seen.add(item.evidence_id)

        validate_currency(self.currency)

        if self.cost_estimate is not None:
            if not isinstance(
                self.cost_estimate,
                ServiceCostEstimate,
            ):
                raise ServiceValidationError(
                    "Invalid cost estimate"
                )

            if self.cost_estimate.currency != self.currency:
                raise ServiceValidationError(
                    "Cost estimate currency mismatch"
                )

        if self.budget is not None:
            object.__setattr__(
                self,
                "budget",
                money(self.budget, "budget"),
            )

        for name in (
            "replacement_advice_id",
            "repair_report_id",
            "diagnosis_id",
            "maintenance_record_id",
        ):
            value = getattr(self, name)

            if value:
                validate_id(value, name)

        if self.preferred_service_date is not None:
            validate_datetime(
                self.preferred_service_date,
                "preferred_service_date",
            )

        validate_text(
            self.notes,
            "notes",
            allow_empty=True,
        )


# ============================================================
# SERVICE RECOMMENDATION
# ============================================================

@dataclass(frozen=True)
class ServiceActionRecommendation:
    action_id: str

    service_type: ServiceType
    priority: ServicePriority

    title: str
    description: str

    technician_specialty: TechnicianSpecialty

    recommended_within_days: int

    estimated_duration_hours: Decimal

    required_parts: tuple[str, ...] = ()

    reasons: tuple[str, ...] = ()
    safety_notes: tuple[str, ...] = ()

    human_approval_required: bool = True

    def __post_init__(self) -> None:
        validate_id(self.action_id, "action_id")

        if not isinstance(self.service_type, ServiceType):
            raise ServiceValidationError(
                "Invalid service type"
            )

        if not isinstance(self.priority, ServicePriority):
            raise ServiceValidationError(
                "Invalid service priority"
            )

        if not isinstance(
            self.technician_specialty,
            TechnicianSpecialty,
        ):
            raise ServiceValidationError(
                "Invalid technician specialty"
            )

        validate_text(self.title, "title")
        validate_text(self.description, "description")

        validate_integer(
            self.recommended_within_days,
            "recommended_within_days",
            maximum=36500,
        )

        decimal_value(
            self.estimated_duration_hours,
            "estimated_duration_hours",
            maximum=Decimal("100000"),
        )

        for collection_name in (
            "required_parts",
            "reasons",
            "safety_notes",
        ):
            collection = getattr(self, collection_name)

            if not isinstance(collection, tuple):
                raise ServiceValidationError(
                    f"{collection_name} must be a tuple"
                )

            for value in collection:
                validate_text(
                    value,
                    collection_name,
                )

        if self.human_approval_required is not True:
            raise ServiceValidationError(
                "Human approval is mandatory"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "service_type": self.service_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "technician_specialty": (
                self.technician_specialty.value
            ),
            "recommended_within_days": (
                self.recommended_within_days
            ),
            "estimated_duration_hours": str(
                self.estimated_duration_hours
            ),
            "required_parts": list(self.required_parts),
            "reasons": list(self.reasons),
            "safety_notes": list(self.safety_notes),
            "human_approval_required": True,
        }


# ============================================================
# RECOMMENDATION RECORD
# ============================================================

@dataclass
class ServiceRecommendationRecord:
    recommendation_id: str

    tenant_id: str
    robot_id: str
    component_id: str

    overall_priority: ServicePriority

    risk_score: Decimal

    recommendations: list[
        ServiceActionRecommendation
    ]

    evidence: list[ServiceEvidence]

    cost_estimate: ServiceCostEstimate | None

    warnings: list[str]
    reasons: list[str]

    replacement_advice_id: str = ""
    repair_report_id: str = ""
    diagnosis_id: str = ""
    maintenance_record_id: str = ""

    status: RecommendationStatus = (
        RecommendationStatus.PENDING_REVIEW
    )

    reviewer_id: str | None = None
    review_notes: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    reviewed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": (
                self.recommendation_id
            ),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "component_id": self.component_id,
            "overall_priority": (
                self.overall_priority.value
            ),
            "risk_score": str(self.risk_score),
            "recommendations": [
                item.to_dict()
                for item in self.recommendations
            ],
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
            "cost_estimate": (
                self.cost_estimate.to_dict()
                if self.cost_estimate
                else None
            ),
            "warnings": list(self.warnings),
            "reasons": list(self.reasons),
            "replacement_advice_id": (
                self.replacement_advice_id
            ),
            "repair_report_id": self.repair_report_id,
            "diagnosis_id": self.diagnosis_id,
            "maintenance_record_id": (
                self.maintenance_record_id
            ),
            "status": self.status.value,
            "reviewer_id": self.reviewer_id,
            "review_notes": self.review_notes,
            "created_at": self.created_at.isoformat(),
            "reviewed_at": (
                self.reviewed_at.isoformat()
                if self.reviewed_at
                else None
            ),
            "advisory_only": True,
            "service_execution_authorized": False,
            "robot_operation_authorized": False,
        }


# ============================================================
# SERVICE RECOMMENDATIONS ENGINE
# ============================================================

class RoboticsServiceRecommendationsEngine:

    def __init__(
        self,
        config: ServiceRecommendationConfig | None = None,
    ) -> None:
        self.config = (
            config or ServiceRecommendationConfig()
        )

        self._records: dict[
            tuple[str, str],
            ServiceRecommendationRecord,
        ] = {}

        self._audit: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _log(
        self,
        *,
        tenant_id: str,
        recommendation_id: str,
        action: str,
        actor_id: str = "system",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._audit.append(
            {
                "event_id": new_id(),
                "timestamp": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "recommendation_id": recommendation_id,
                "action": action,
                "actor_id": actor_id,
                "details": dict(details or {}),
            }
        )

    # --------------------------------------------------------
    # RISK ASSESSMENT
    # --------------------------------------------------------

    def assess_service_risk(
        self,
        component: ServiceComponentProfile,
    ) -> tuple[
        Decimal,
        list[str],
        int,
    ]:
        """
        Calculate an advisory heuristic score.

        The score is not a calibrated failure
        probability or certified safety metric.
        """
        score = Decimal("0")
        reasons = []
        observations = 0

        if component.health_score is not None:
            observations += 1

            score += (
                Decimal("100")
                - component.health_score
            ) * Decimal("0.30")

            if (
                component.health_score
                <= self.config.critical_health_threshold
            ):
                reasons.append(
                    "Component health is critically low."
                )

            elif (
                component.health_score
                <= self.config.low_health_threshold
            ):
                reasons.append(
                    "Component health is below "
                    "the configured threshold."
                )

        if component.remaining_useful_life_days is not None:
            observations += 1

            rul = component.remaining_useful_life_days

            if rul <= self.config.critical_rul_days:
                score += Decimal("25")

                reasons.append(
                    "Remaining useful life is critically low."
                )

            elif rul <= self.config.low_rul_days:
                score += Decimal("15")

                reasons.append(
                    "Remaining useful life is low."
                )

        if component.failure_probability is not None:
            observations += 1

            score += (
                component.failure_probability
                * Decimal("25")
            )

            if (
                component.failure_probability
                >= self.config.critical_failure_probability
            ):
                reasons.append(
                    "Predicted failure risk is critical."
                )

            elif (
                component.failure_probability
                >= self.config.high_failure_probability
            ):
                reasons.append(
                    "Predicted failure risk is elevated."
                )

        severity_points = {
            FaultSeverity.NONE: Decimal("0"),
            FaultSeverity.LOW: Decimal("3"),
            FaultSeverity.MODERATE: Decimal("8"),
            FaultSeverity.HIGH: Decimal("18"),
            FaultSeverity.CRITICAL: Decimal("30"),
            FaultSeverity.UNKNOWN: Decimal("0"),
        }

        if component.fault_severity != FaultSeverity.UNKNOWN:
            observations += 1

        score += severity_points[
            component.fault_severity
        ]

        condition_points = {
            ComponentCondition.EXCELLENT: Decimal("0"),
            ComponentCondition.GOOD: Decimal("2"),
            ComponentCondition.FAIR: Decimal("7"),
            ComponentCondition.POOR: Decimal("15"),
            ComponentCondition.FAILED: Decimal("30"),
            ComponentCondition.UNKNOWN: Decimal("0"),
        }

        if component.condition != ComponentCondition.UNKNOWN:
            observations += 1

        score += condition_points[
            component.condition
        ]

        if component.recurring_fault_count:
            observations += 1

            score += min(
                Decimal("15"),
                Decimal(component.recurring_fault_count)
                * Decimal("3"),
            )

            reasons.append(
                "Recurring faults have been reported."
            )

        if (
            component.hours_since_last_service is not None
            and component.recommended_service_interval_hours
            is not None
        ):
            observations += 1

            if (
                component.hours_since_last_service
                >= component.recommended_service_interval_hours
            ):
                score += Decimal("12")

                reasons.append(
                    "Recommended service interval has "
                    "been reached or exceeded."
                )

        if component.safety_fault_active:
            score += Decimal("35")

            reasons.append(
                "An active safety-related fault is reported."
            )

        if (
            component.criticality
            == ComponentCriticality.SAFETY_CRITICAL
            and component.fault_severity
            in (
                FaultSeverity.HIGH,
                FaultSeverity.CRITICAL,
            )
        ):
            score += Decimal("20")

            reasons.append(
                "High-severity fault affects "
                "a safety-critical component."
            )

        score = min(
            Decimal("100"),
            score,
        ).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

        return score, reasons, observations

    # --------------------------------------------------------
    # PRIORITY CLASSIFICATION
    # --------------------------------------------------------

    def classify_priority(
        self,
        component: ServiceComponentProfile,
        risk_score: Decimal,
    ) -> ServicePriority:
        if (
            component.safety_fault_active
            or component.condition
            == ComponentCondition.FAILED
            or component.fault_severity
            == FaultSeverity.CRITICAL
        ):
            return ServicePriority.CRITICAL

        if (
            component.criticality
            == ComponentCriticality.SAFETY_CRITICAL
            and (
                component.fault_severity
                == FaultSeverity.HIGH
                or risk_score >= Decimal("60")
            )
        ):
            return ServicePriority.CRITICAL

        if risk_score >= Decimal("75"):
            return ServicePriority.URGENT

        if risk_score >= Decimal("45"):
            return ServicePriority.HIGH

        if risk_score >= Decimal("20"):
            return ServicePriority.NORMAL

        return ServicePriority.LOW

    # --------------------------------------------------------
    # TECHNICIAN SPECIALTY
    # --------------------------------------------------------

    @staticmethod
    def technician_for_component(
        component_type: RobotComponentType,
    ) -> TechnicianSpecialty:
        mapping = {
            RobotComponentType.MOTOR: (
                TechnicianSpecialty.ELECTRICAL
            ),
            RobotComponentType.GEARBOX: (
                TechnicianSpecialty.MECHANICAL
            ),
            RobotComponentType.BEARING: (
                TechnicianSpecialty.MECHANICAL
            ),
            RobotComponentType.ACTUATOR: (
                TechnicianSpecialty.ROBOTICS
            ),
            RobotComponentType.BATTERY: (
                TechnicianSpecialty.BATTERY
            ),
            RobotComponentType.SENSOR: (
                TechnicianSpecialty.CONTROL_SYSTEMS
            ),
            RobotComponentType.CAMERA: (
                TechnicianSpecialty.ROBOTICS
            ),
            RobotComponentType.LIDAR: (
                TechnicianSpecialty.ROBOTICS
            ),
            RobotComponentType.CONTROLLER: (
                TechnicianSpecialty.CONTROL_SYSTEMS
            ),
            RobotComponentType.POWER_SYSTEM: (
                TechnicianSpecialty.ELECTRICAL
            ),
            RobotComponentType.COOLING_SYSTEM: (
                TechnicianSpecialty.MECHANICAL
            ),
            RobotComponentType.BRAKING_SYSTEM: (
                TechnicianSpecialty.SAFETY_ENGINEERING
            ),
            RobotComponentType.MANIPULATOR: (
                TechnicianSpecialty.ROBOTICS
            ),
            RobotComponentType.END_EFFECTOR: (
                TechnicianSpecialty.ROBOTICS
            ),
            RobotComponentType.WHEEL: (
                TechnicianSpecialty.MECHANICAL
            ),
            RobotComponentType.JOINT: (
                TechnicianSpecialty.ROBOTICS
            ),
            RobotComponentType.STRUCTURE: (
                TechnicianSpecialty.MECHANICAL
            ),
            RobotComponentType.NETWORK: (
                TechnicianSpecialty.CONTROL_SYSTEMS
            ),
            RobotComponentType.SOFTWARE: (
                TechnicianSpecialty.SOFTWARE
            ),
        }

        return mapping.get(
            component_type,
            TechnicianSpecialty.GENERAL_MAINTENANCE,
        )

    # --------------------------------------------------------
    # SERVICE ACTION CREATION
    # --------------------------------------------------------

    @staticmethod
    def _make_action(
        *,
        service_type: ServiceType,
        priority: ServicePriority,
        title: str,
        description: str,
        specialty: TechnicianSpecialty,
        within_days: int,
        duration_hours: Any,
        reasons: Sequence[str] = (),
        required_parts: Sequence[str] = (),
        safety_notes: Sequence[str] = (),
    ) -> ServiceActionRecommendation:
        return ServiceActionRecommendation(
            action_id=new_id(),
            service_type=service_type,
            priority=priority,
            title=title,
            description=description,
            technician_specialty=specialty,
            recommended_within_days=within_days,
            estimated_duration_hours=decimal_value(
                duration_hours,
                "duration_hours",
                maximum=Decimal("100000"),
            ),
            reasons=tuple(reasons),
            required_parts=tuple(required_parts),
            safety_notes=tuple(safety_notes),
            human_approval_required=True,
        )

    # --------------------------------------------------------
    # GENERATE ACTIONS
    # --------------------------------------------------------

    def generate_service_actions(
        self,
        component: ServiceComponentProfile,
        priority: ServicePriority,
        risk_score: Decimal,
        observations: int,
    ) -> list[ServiceActionRecommendation]:
        actions = []

        specialty = self.technician_for_component(
            component.component_type
        )

        if observations == 0:
            actions.append(
                self._make_action(
                    service_type=ServiceType.INSPECTION,
                    priority=ServicePriority.NORMAL,
                    title="Establish component condition",
                    description=(
                        "Collect telemetry, inspect the "
                        "component and confirm operating "
                        "condition before deciding on service."
                    ),
                    specialty=specialty,
                    within_days=7,
                    duration_hours="1.5",
                    reasons=(
                        "Insufficient condition data.",
                    ),
                )
            )

            return actions

        if priority == ServicePriority.CRITICAL:
            actions.append(
                self._make_action(
                    service_type=(
                        ServiceType.EMERGENCY_ASSESSMENT
                    ),
                    priority=ServicePriority.CRITICAL,
                    title="Urgent engineering safety assessment",
                    description=(
                        "Escalate the component to qualified "
                        "engineering personnel. Assess whether "
                        "the robot must remain isolated and "
                        "out of service pending inspection."
                    ),
                    specialty=(
                        TechnicianSpecialty.SAFETY_ENGINEERING
                    ),
                    within_days=0,
                    duration_hours="2",
                    reasons=(
                        "Critical condition or safety fault.",
                    ),
                    safety_notes=(
                        "Do not rely on this advisory "
                        "as an operational clearance.",
                        "Follow approved isolation and "
                        "lockout/tagout procedures.",
                    ),
                )
            )

        if (
            component.condition
            in (
                ComponentCondition.POOR,
                ComponentCondition.FAILED,
            )
            or component.fault_severity
            in (
                FaultSeverity.HIGH,
                FaultSeverity.CRITICAL,
            )
        ):
            actions.append(
                self._make_action(
                    service_type=(
                        ServiceType.CORRECTIVE_MAINTENANCE
                    ),
                    priority=priority,
                    title="Corrective maintenance assessment",
                    description=(
                        "Investigate the confirmed faults, "
                        "inspect affected assemblies and "
                        "prepare a qualified repair plan."
                    ),
                    specialty=specialty,
                    within_days=(
                        0 if priority == ServicePriority.CRITICAL
                        else 1 if priority == ServicePriority.URGENT
                        else 7
                    ),
                    duration_hours="3",
                    reasons=(
                        "Poor condition or high-severity fault.",
                    ),
                )
            )

        if (
            component.remaining_useful_life_days
            is not None
            and component.remaining_useful_life_days
            <= self.config.low_rul_days
        ):
            actions.append(
                self._make_action(
                    service_type=(
                        ServiceType.PREDICTIVE_MAINTENANCE
                    ),
                    priority=priority,
                    title="Plan service based on remaining useful life",
                    description=(
                        "Validate the RUL estimate against "
                        "recent telemetry and plan maintenance "
                        "before the projected end of useful life."
                    ),
                    specialty=specialty,
                    within_days=(
                        0
                        if component.remaining_useful_life_days
                        <= self.config.critical_rul_days
                        else 14
                    ),
                    duration_hours="2",
                    reasons=(
                        "Predicted remaining useful life "
                        "is below the configured threshold.",
                    ),
                )
            )

        if (
            component.hours_since_last_service is not None
            and component.recommended_service_interval_hours
            is not None
            and component.hours_since_last_service
            >= component.recommended_service_interval_hours
        ):
            actions.append(
                self._make_action(
                    service_type=(
                        ServiceType.PREVENTIVE_MAINTENANCE
                    ),
                    priority=(
                        ServicePriority.HIGH
                        if priority in (
                            ServicePriority.HIGH,
                            ServicePriority.URGENT,
                            ServicePriority.CRITICAL,
                        )
                        else ServicePriority.NORMAL
                    ),
                    title="Perform scheduled preventive maintenance",
                    description=(
                        "The recorded service interval "
                        "has been reached. Perform "
                        "manufacturer-approved preventive "
                        "maintenance procedures."
                    ),
                    specialty=specialty,
                    within_days=7,
                    duration_hours="2",
                    reasons=(
                        "Service interval reached.",
                    ),
                )
            )

        if component.recurring_fault_count >= 2:
            actions.append(
                self._make_action(
                    service_type=ServiceType.DIAGNOSTIC_TEST,
                    priority=priority,
                    title="Investigate recurring faults",
                    description=(
                        "Perform root-cause analysis "
                        "of repeated faults and review "
                        "maintenance history."
                    ),
                    specialty=specialty,
                    within_days=7,
                    duration_hours="2",
                    reasons=(
                        "Multiple recurring faults recorded.",
                    ),
                )
            )

        component_actions = {
            RobotComponentType.MOTOR: (
                ServiceType.DIAGNOSTIC_TEST,
                "Motor electrical and thermal inspection",
                "Inspect current draw, temperature, "
                "vibration and bearing condition.",
                "2",
            ),
            RobotComponentType.GEARBOX: (
                ServiceType.INSPECTION,
                "Gearbox inspection",
                "Inspect lubrication, backlash, "
                "wear and abnormal vibration.",
                "2",
            ),
            RobotComponentType.BEARING: (
                ServiceType.LUBRICATION,
                "Bearing lubrication and wear assessment",
                "Inspect bearing wear and apply only "
                "manufacturer-approved lubrication.",
                "1.5",
            ),
            RobotComponentType.BATTERY: (
                ServiceType.BATTERY_SERVICE,
                "Battery health and thermal assessment",
                "Review battery temperature, cell "
                "imbalance, state of health and "
                "manufacturer safety requirements.",
                "2",
            ),
            RobotComponentType.SENSOR: (
                ServiceType.CALIBRATION,
                "Sensor inspection and calibration",
                "Verify sensor measurements, mounting "
                "and calibration using approved procedures.",
                "1.5",
            ),
            RobotComponentType.CAMERA: (
                ServiceType.CALIBRATION,
                "Camera calibration and inspection",
                "Inspect optics, mounting and image "
                "calibration against reference targets.",
                "1.5",
            ),
            RobotComponentType.LIDAR: (
                ServiceType.CALIBRATION,
                "LiDAR inspection and calibration",
                "Verify LiDAR alignment, cleanliness "
                "and range measurements.",
                "2",
            ),
            RobotComponentType.ACTUATOR: (
                ServiceType.INSPECTION,
                "Actuator inspection",
                "Inspect mechanical movement, "
                "position feedback and wear.",
                "2",
            ),
            RobotComponentType.CONTROLLER: (
                ServiceType.DIAGNOSTIC_TEST,
                "Controller diagnostic review",
                "Review controller logs, communication "
                "errors and thermal conditions.",
                "2",
            ),
            RobotComponentType.COOLING_SYSTEM: (
                ServiceType.CLEANING,
                "Cooling system inspection",
                "Inspect filters, airflow, cooling "
                "performance and obstructions.",
                "1.5",
            ),
            RobotComponentType.SOFTWARE: (
                ServiceType.SOFTWARE_DIAGNOSTICS,
                "Robot software diagnostics",
                "Review application logs, software "
                "faults, version compatibility "
                "and regression-test requirements.",
                "2",
            ),
            RobotComponentType.NETWORK: (
                ServiceType.DIAGNOSTIC_TEST,
                "Robot communication diagnostics",
                "Inspect network connectivity, "
                "packet loss and interface errors.",
                "1.5",
            ),
            RobotComponentType.BRAKING_SYSTEM: (
                ServiceType.SAFETY_INSPECTION,
                "Braking system safety inspection",
                "Have qualified personnel inspect "
                "brake integrity and safety functions.",
                "2",
            ),
        }

        if (
            priority
            in (
                ServicePriority.HIGH,
                ServicePriority.URGENT,
                ServicePriority.CRITICAL,
            )
            and component.component_type in component_actions
        ):
            (
                service_type,
                title,
                description,
                duration,
            ) = component_actions[
                component.component_type
            ]

            actions.append(
                self._make_action(
                    service_type=service_type,
                    priority=priority,
                    title=title,
                    description=description,
                    specialty=specialty,
                    within_days=(
                        0 if priority == ServicePriority.CRITICAL
                        else 2
                    ),
                    duration_hours=duration,
                    reasons=(
                        "Component-specific high-priority "
                        "service assessment.",
                    ),
                )
            )

        if not actions:
            actions.append(
                self._make_action(
                    service_type=ServiceType.INSPECTION,
                    priority=ServicePriority.LOW,
                    title="Routine component health inspection",
                    description=(
                        "Continue condition monitoring "
                        "and follow manufacturer-approved "
                        "inspection intervals."
                    ),
                    specialty=specialty,
                    within_days=30,
                    duration_hours="1",
                    reasons=(
                        "No immediate service trigger detected.",
                    ),
                )
            )

        return actions

    # --------------------------------------------------------
    # GENERATE RECOMMENDATIONS
    # --------------------------------------------------------

    def recommend(
        self,
        request: ServiceRecommendationRequest,
    ) -> dict[str, Any]:
        if not isinstance(
            request,
            ServiceRecommendationRequest,
        ):
            raise ServiceValidationError(
                "Invalid service recommendation request"
            )

        if (
            len(request.evidence)
            > self.config.maximum_evidence_per_request
        ):
            raise ServiceValidationError(
                "Too many evidence records"
            )

        component = request.component

        risk_score, reasons, observations = (
            self.assess_service_risk(component)
        )

        priority = self.classify_priority(
            component,
            risk_score,
        )

        recommendations = self.generate_service_actions(
            component,
            priority,
            risk_score,
            observations,
        )

        if (
            len(recommendations)
            > self.config.maximum_recommendations_per_request
        ):
            raise ServiceConflict(
                "Recommendation count exceeds limit"
            )

        warnings = []

        if observations == 0:
            warnings.append(
                "Insufficient diagnostic data; "
                "recommendations are preliminary."
            )

        if component.safety_fault_active:
            warnings.append(
                "Active safety fault requires "
                "qualified personnel assessment."
            )

        if (
            component.criticality
            == ComponentCriticality.SAFETY_CRITICAL
        ):
            warnings.append(
                "Safety-critical component: service "
                "approval does not authorize operation."
            )

        if component.warranty_active:
            warnings.append(
                "Review warranty conditions and "
                "authorized service requirements."
            )

        if (
            request.budget is not None
            and request.cost_estimate is not None
            and request.cost_estimate.total
            > request.budget
        ):
            warnings.append(
                "Estimated service cost exceeds budget."
            )

        if request.preferred_service_date is not None:
            preferred = validate_datetime(
                request.preferred_service_date,
                "preferred_service_date",
            )

            earliest_deadline = min(
                item.recommended_within_days
                for item in recommendations
            )

            if preferred > (
                utc_now()
                + timedelta(days=earliest_deadline)
            ):
                warnings.append(
                    "Preferred service date is later "
                    "than the advisory service deadline."
                )

        if request.replacement_advice_id:
            reasons.append(
                "A replacement advisory reference "
                "is available for engineering review."
            )

        record = ServiceRecommendationRecord(
            recommendation_id=new_id(),
            tenant_id=component.tenant_id,
            robot_id=component.robot_id,
            component_id=component.component_id,
            overall_priority=priority,
            risk_score=risk_score,
            recommendations=recommendations,
            evidence=list(request.evidence),
            cost_estimate=request.cost_estimate,
            warnings=warnings,
            reasons=reasons,
            replacement_advice_id=(
                request.replacement_advice_id
            ),
            repair_report_id=request.repair_report_id,
            diagnosis_id=request.diagnosis_id,
            maintenance_record_id=(
                request.maintenance_record_id
            ),
        )

        with self._lock:
            if (
                len(self._records)
                >= self.config.maximum_recommendations
            ):
                raise ServiceConflict(
                    "Recommendation storage capacity reached"
                )

            self._records[
                (
                    record.tenant_id,
                    record.recommendation_id,
                )
            ] = record

            self._log(
                tenant_id=record.tenant_id,
                recommendation_id=(
                    record.recommendation_id
                ),
                action="service_recommendation_created",
                details={
                    "priority": priority.value,
                    "risk_score": str(risk_score),
                },
            )

            return record.to_dict()

    # --------------------------------------------------------
    # GET RECOMMENDATION
    # --------------------------------------------------------

    def get_recommendation(
        self,
        *,
        tenant_id: str,
        recommendation_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(
            recommendation_id,
            "recommendation_id",
        )

        with self._lock:
            record = self._records.get(
                (tenant_id, recommendation_id)
            )

            if record is None:
                raise ServiceRecommendationNotFound(
                    "Service recommendation not found"
                )

            return record.to_dict()

    # --------------------------------------------------------
    # REVIEW RECOMMENDATION
    # --------------------------------------------------------

    def review_recommendation(
        self,
        *,
        tenant_id: str,
        recommendation_id: str,
        reviewer_id: str,
        approve: bool,
        notes: str = "",
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ServicePermissionDenied(
                "Authorized engineering reviewer required"
            )

        validate_id(tenant_id, "tenant_id")
        validate_id(
            recommendation_id,
            "recommendation_id",
        )
        validate_id(reviewer_id, "reviewer_id")

        if not isinstance(approve, bool):
            raise ServiceValidationError(
                "approve must be boolean"
            )

        validate_text(
            notes,
            "notes",
            allow_empty=True,
        )

        with self._lock:
            record = self._records.get(
                (tenant_id, recommendation_id)
            )

            if record is None:
                raise ServiceRecommendationNotFound(
                    "Service recommendation not found"
                )

            if (
                record.status
                != RecommendationStatus.PENDING_REVIEW
            ):
                raise ServiceConflict(
                    "Recommendation already reviewed"
                )

            record.status = (
                RecommendationStatus.APPROVED
                if approve
                else RecommendationStatus.REJECTED
            )

            record.reviewer_id = reviewer_id
            record.review_notes = notes
            record.reviewed_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                recommendation_id=recommendation_id,
                action="service_recommendation_reviewed",
                actor_id=reviewer_id,
                details={
                    "approved": approve,
                },
            )

            return record.to_dict()

    # --------------------------------------------------------
    # LIST RECOMMENDATIONS
    # --------------------------------------------------------

    def list_recommendations(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        component_id: str | None = None,
        status: RecommendationStatus | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        if component_id is not None:
            validate_id(component_id, "component_id")

        if status is not None and not isinstance(
            status,
            RecommendationStatus,
        ):
            raise ServiceValidationError(
                "Invalid recommendation status"
            )

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            records = [
                record
                for (tenant, _), record
                in self._records.items()
                if (
                    tenant == tenant_id
                    and (
                        robot_id is None
                        or record.robot_id == robot_id
                    )
                    and (
                        component_id is None
                        or record.component_id == component_id
                    )
                    and (
                        status is None
                        or record.status == status
                    )
                )
            ]

            records.sort(
                key=lambda item: item.created_at,
                reverse=True,
            )

            return [
                record.to_dict()
                for record in records[:limit]
            ]

    # --------------------------------------------------------
    # ROBOT SERVICE SUMMARY
    # --------------------------------------------------------

    def robot_service_summary(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        records = self.list_recommendations(
            tenant_id=tenant_id,
            robot_id=robot_id,
            limit=100000,
        )

        priorities = Counter(
            item["overall_priority"]
            for item in records
        )

        statuses = Counter(
            item["status"]
            for item in records
        )

        service_types = Counter(
            action["service_type"]
            for record in records
            for action in record["recommendations"]
        )

        return {
            "tenant_id": tenant_id,
            "robot_id": robot_id,
            "total_recommendations": len(records),
            "priorities": dict(priorities),
            "statuses": dict(statuses),
            "service_types": dict(service_types),
            "pending_reviews": statuses.get(
                RecommendationStatus.PENDING_REVIEW.value,
                0,
            ),
            "critical_recommendations": priorities.get(
                ServicePriority.CRITICAL.value,
                0,
            ),
            "advisory_only": True,
        }

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        *,
        tenant_id: str,
        recommendation_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if recommendation_id is not None:
            validate_id(
                recommendation_id,
                "recommendation_id",
            )

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            events = [
                dict(event)
                for event in self._audit
                if (
                    event["tenant_id"] == tenant_id
                    and (
                        recommendation_id is None
                        or event["recommendation_id"]
                        == recommendation_id
                    )
                )
            ]

            return events[-limit:]

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
            records = [
                record.to_dict()
                for (tenant, _), record
                in self._records.items()
                if tenant == tenant_id
            ]

            audit = [
                dict(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ]

        payload = {
            "generated_at": utc_now().isoformat(),
            "tenant_id": tenant_id,
            "recommendations": records,
            "audit_events": audit,
            "advisory_only": True,
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
    engine = RoboticsServiceRecommendationsEngine()

    component = ServiceComponentProfile(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        component_id="joint_motor_001",
        component_name="Industrial Robot Joint Motor",
        component_type=RobotComponentType.MOTOR,
        condition=ComponentCondition.POOR,
        criticality=ComponentCriticality.HIGH,
        fault_severity=FaultSeverity.HIGH,
        health_score=Decimal("32"),
        remaining_useful_life_days=Decimal("45"),
        failure_probability=Decimal("0.35"),
        operating_hours=Decimal("6200"),
        hours_since_last_service=Decimal("1300"),
        recommended_service_interval_hours=Decimal("1000"),
        recurring_fault_count=3,
        fault_codes=("MOTOR_OVERHEAT", "VIBRATION_HIGH"),
        manufacturer="Demo Robotics",
        model_number="MTR-100",
        warranty_active=False,
        safety_fault_active=False,
    )

    evidence = (
        ServiceEvidence(
            evidence_id=new_id(),
            source=EvidenceSource.TELEMETRY,
            description=(
                "Elevated motor temperature and vibration "
                "reported by monitoring sensors."
            ),
            reference_id="telemetry_001",
            confidence=Decimal("0.95"),
        ),
        ServiceEvidence(
            evidence_id=new_id(),
            source=EvidenceSource.DIAGNOSTICS,
            description=(
                "Motor bearing wear identified as "
                "a possible diagnostic cause."
            ),
            reference_id="diagnosis_001",
            confidence=Decimal("0.86"),
        ),
    )

    cost = ServiceCostEstimate(
        currency="INR",
        labor_cost=Decimal("3500"),
        parts_cost=Decimal("2400"),
        downtime_cost=Decimal("5000"),
        travel_cost=Decimal("500"),
        inspection_cost=Decimal("800"),
        other_cost=Decimal("0"),
    )

    request = ServiceRecommendationRequest(
        component=component,
        evidence=evidence,
        cost_estimate=cost,
        replacement_advice_id="replacement_001",
        repair_report_id="repair_report_001",
        diagnosis_id="diagnosis_001",
        maintenance_record_id="maintenance_001",
        budget=Decimal("15000"),
        currency="INR",
        notes="Demonstration service assessment",
    )

    result = engine.recommend(request)

    print("\nSERVICE RECOMMENDATIONS")

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print("\nROBOT SERVICE SUMMARY")

    print(
        json.dumps(
            engine.robot_service_summary(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
            ),
            indent=2,
        )
    )

    report_path = engine.export_json(
        "data/robotics/service_recommendations_report.json",
        tenant_id="tenant_alpha",
    )

    print("\nExported:", report_path)


if __name__ == "__main__":
    main()
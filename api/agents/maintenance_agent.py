"""Maintenance Agent module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/maintenance_agent.py

Physical AI & Autonomous Robotics Intelligence Platform
Predictive Maintenance & Maintenance Planning Agent

Python 3.11+

Features:
- Robot preventive and predictive maintenance
- Health-agent assessment integration
- Failure prediction and RUL integration
- Maintenance recommendation generation
- Maintenance work-order planning
- Risk-based task prioritization
- Technician skill requirements
- Spare-parts requirement estimation
- Repair and replacement cost integration
- Maintenance scheduling proposals
- Fleet maintenance analysis
- Human-in-the-loop approval integration
- Maintenance history and audit support
- Tenant isolation and authorization
- Duplicate work-order prevention
- No autonomous physical maintenance execution

IMPORTANT:
This module generates maintenance recommendations and plans.
It does not physically control robots, bypass safety interlocks,
reset emergency stops, or automatically authorize repairs.

Production integrations are supplied through Protocol interfaces.
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

MAX_ROBOTS_PER_BATCH = 200
MAX_COMPONENTS_PER_ROBOT = 500
MAX_RECOMMENDATIONS = 1000
MAX_REQUIRED_PARTS = 100
MAX_METADATA_FIELDS = 50

DEFAULT_MAINTENANCE_WINDOW_HOURS = 24
DEFAULT_WORK_ORDER_DURATION_MINUTES = 120

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
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def validate_probability(value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
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


def validate_positive(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "Value must be finite and positive"
        )
    return value


def validate_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Datetime must be timezone-aware")
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

class MaintenanceType(str, Enum):
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    CONDITION_BASED = "condition_based"
    INSPECTION = "inspection"
    CALIBRATION = "calibration"
    SOFTWARE_UPDATE = "software_update"
    PART_REPLACEMENT = "part_replacement"
    EMERGENCY = "emergency"


class MaintenancePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class MaintenanceStatus(str, Enum):
    PROPOSED = "proposed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class ComponentCategory(str, Enum):
    BATTERY = "battery"
    MOTOR = "motor"
    ACTUATOR = "actuator"
    GEARBOX = "gearbox"
    BEARING = "bearing"
    CONTROLLER = "controller"
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    ENCODER = "encoder"
    MANIPULATOR = "manipulator"
    END_EFFECTOR = "end_effector"
    POWER_SYSTEM = "power_system"
    COOLING_SYSTEM = "cooling_system"
    NETWORK = "network"
    SAFETY_SYSTEM = "safety_system"
    SOFTWARE = "software"
    OTHER = "other"


class FailureRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MaintenanceTrigger(str, Enum):
    HEALTH_ASSESSMENT = "health_assessment"
    FAILURE_PREDICTION = "failure_prediction"
    RUL_PREDICTION = "rul_prediction"
    SENSOR_ANOMALY = "sensor_anomaly"
    SCHEDULED_INTERVAL = "scheduled_interval"
    FAULT_CODE = "fault_code"
    MANUAL_REQUEST = "manual_request"


class TechnicianSkill(str, Enum):
    ROBOTICS = "robotics"
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    ELECTRONICS = "electronics"
    BATTERY_SYSTEMS = "battery_systems"
    SENSOR_CALIBRATION = "sensor_calibration"
    NETWORKING = "networking"
    SOFTWARE = "software"
    SAFETY = "safety"
    GENERAL_MAINTENANCE = "general_maintenance"


class WorkOrderAction(str, Enum):
    INSPECT = "inspect"
    DIAGNOSE = "diagnose"
    CALIBRATE = "calibrate"
    SERVICE = "service"
    REPAIR = "repair"
    REPLACE = "replace"
    TEST = "test"
    VERIFY = "verify"


# ============================================================
# Exceptions
# ============================================================

class MaintenanceAgentError(Exception):
    pass


class MaintenanceValidationError(MaintenanceAgentError):
    pass


class MaintenanceNotFoundError(MaintenanceAgentError):
    pass


class MaintenancePermissionError(MaintenanceAgentError):
    pass


class MaintenanceConflictError(MaintenanceAgentError):
    pass


class MaintenanceServiceUnavailableError(MaintenanceAgentError):
    pass


# ============================================================
# Security Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class MaintenanceScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True, slots=True)
class MaintenancePolicy:
    failure_probability_warning: float = 0.30
    failure_probability_high: float = 0.60
    failure_probability_critical: float = 0.85

    rul_warning_hours: float = 250.0
    rul_high_hours: float = 100.0
    rul_critical_hours: float = 24.0

    health_score_warning: float = 0.75
    health_score_high: float = 0.50
    health_score_critical: float = 0.25

    maximum_plan_horizon_days: int = 90

    require_human_approval: bool = True
    require_safety_review_for_critical: bool = True

    prevent_duplicate_open_orders: bool = True

    def __post_init__(self) -> None:
        probabilities = (
            self.failure_probability_warning,
            self.failure_probability_high,
            self.failure_probability_critical,
            self.health_score_warning,
            self.health_score_high,
            self.health_score_critical,
        )

        for value in probabilities:
            validate_probability(value)

        if not (
            self.failure_probability_warning
            < self.failure_probability_high
            < self.failure_probability_critical
        ):
            raise MaintenanceValidationError(
                "Failure probability thresholds are invalid"
            )

        if not (
            self.health_score_critical
            < self.health_score_high
            < self.health_score_warning
        ):
            raise MaintenanceValidationError(
                "Health score thresholds are invalid"
            )

        if not (
            0 <= self.rul_critical_hours
            < self.rul_high_hours
            < self.rul_warning_hours
        ):
            raise MaintenanceValidationError(
                "RUL thresholds are invalid"
            )

        if self.maximum_plan_horizon_days < 1:
            raise MaintenanceValidationError(
                "Plan horizon must be positive"
            )


# ============================================================
# Input Models
# ============================================================

@dataclass(slots=True)
class ComponentMaintenanceSignal:
    component_id: str
    category: ComponentCategory

    health_score: float | None = None
    failure_probability: float | None = None
    remaining_useful_life_hours: float | None = None

    fault_active: bool = False
    anomaly_detected: bool = False
    maintenance_due: bool = False

    operating_hours: float | None = None

    last_maintenance_at: datetime | None = None

    fault_codes: list[str] = field(default_factory=list)

    model_id: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.component_id)

        if self.model_id is not None:
            validate_identifier(self.model_id)

        if self.health_score is not None:
            validate_probability(self.health_score)

        if self.failure_probability is not None:
            validate_probability(self.failure_probability)

        for value in (
            self.remaining_useful_life_hours,
            self.operating_hours,
        ):
            if value is not None:
                validate_nonnegative(value)

        if self.last_maintenance_at is not None:
            validate_timestamp(self.last_maintenance_at)

        if len(self.fault_codes) > 100:
            raise MaintenanceValidationError(
                "Too many fault codes"
            )


@dataclass(slots=True)
class RobotMaintenanceContext:
    tenant_id: str
    robot_id: str

    timestamp: datetime

    components: list[ComponentMaintenanceSignal]

    robot_health_score: float | None = None

    emergency_stop_active: bool = False
    safety_fault_active: bool = False

    active_task: bool = False

    robot_model: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.robot_id)
        validate_timestamp(self.timestamp)

        if self.robot_health_score is not None:
            validate_probability(self.robot_health_score)

        if len(self.components) > MAX_COMPONENTS_PER_ROBOT:
            raise MaintenanceValidationError(
                "Too many components"
            )

        identifiers = [
            component.component_id
            for component in self.components
        ]

        if len(identifiers) != len(set(identifiers)):
            raise MaintenanceValidationError(
                "Duplicate component identifiers"
            )

        if len(self.metadata) > MAX_METADATA_FIELDS:
            raise MaintenanceValidationError(
                "Too many metadata fields"
            )


# ============================================================
# Spare Parts and Cost Models
# ============================================================

@dataclass(frozen=True, slots=True)
class RequiredPart:
    part_id: str
    quantity: int

    description: str = ""

    estimated_unit_cost: float | None = None

    currency: str = "INR"

    def __post_init__(self) -> None:
        validate_identifier(self.part_id)

        if self.quantity < 1:
            raise MaintenanceValidationError(
                "Part quantity must be positive"
            )

        if self.estimated_unit_cost is not None:
            validate_nonnegative(
                self.estimated_unit_cost
            )

        if not re.fullmatch(r"[A-Z]{3}", self.currency):
            raise MaintenanceValidationError(
                "Currency must be a three-letter ISO code"
            )


@dataclass(frozen=True, slots=True)
class MaintenanceCostEstimate:
    labor_cost: float
    parts_cost: float
    downtime_cost: float

    total_cost: float

    currency: str = "INR"

    estimated_downtime_hours: float = 0.0

    def __post_init__(self) -> None:
        for value in (
            self.labor_cost,
            self.parts_cost,
            self.downtime_cost,
            self.total_cost,
            self.estimated_downtime_hours,
        ):
            validate_nonnegative(value)

        if not re.fullmatch(r"[A-Z]{3}", self.currency):
            raise MaintenanceValidationError(
                "Invalid currency"
            )


# ============================================================
# Recommendations
# ============================================================

@dataclass(slots=True)
class MaintenanceRecommendation:
    recommendation_id: str

    tenant_id: str
    robot_id: str
    component_id: str

    category: ComponentCategory

    maintenance_type: MaintenanceType
    priority: MaintenancePriority

    title: str
    description: str

    risk_level: FailureRiskLevel

    trigger: MaintenanceTrigger

    recommended_actions: list[WorkOrderAction]

    required_skills: list[TechnicianSkill]

    failure_probability: float | None = None
    remaining_useful_life_hours: float | None = None

    recommended_by: datetime | None = None

    estimated_duration_minutes: int = (
        DEFAULT_WORK_ORDER_DURATION_MINUTES
    )

    required_parts: list[RequiredPart] = field(
        default_factory=list
    )

    cost_estimate: MaintenanceCostEstimate | None = None

    requires_human_approval: bool = True
    requires_safety_review: bool = False

    reasoning: list[str] = field(
        default_factory=list
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for value in (
            self.recommendation_id,
            self.tenant_id,
            self.robot_id,
            self.component_id,
        ):
            validate_identifier(value)

        if self.estimated_duration_minutes <= 0:
            raise MaintenanceValidationError(
                "Maintenance duration must be positive"
            )

        if len(self.required_parts) > MAX_REQUIRED_PARTS:
            raise MaintenanceValidationError(
                "Too many required parts"
            )

        if self.recommended_by is not None:
            validate_timestamp(self.recommended_by)


@dataclass(slots=True)
class MaintenancePlan:
    plan_id: str

    tenant_id: str
    robot_id: str

    recommendations: list[MaintenanceRecommendation]

    overall_priority: MaintenancePriority

    requires_human_approval: bool
    requires_safety_review: bool

    total_estimated_duration_minutes: int

    total_estimated_cost: float | None
    currency: str | None

    generated_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# Work Order Models
# ============================================================

@dataclass(slots=True)
class MaintenanceWorkOrder:
    work_order_id: str

    tenant_id: str
    robot_id: str
    component_id: str

    recommendation_id: str

    title: str
    description: str

    maintenance_type: MaintenanceType
    priority: MaintenancePriority

    status: MaintenanceStatus

    required_skills: list[TechnicianSkill]
    required_parts: list[RequiredPart]

    estimated_duration_minutes: int

    scheduled_start_at: datetime | None = None
    scheduled_end_at: datetime | None = None

    assigned_technician_id: str | None = None

    approval_id: str | None = None

    created_by: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    version: int = 1

    def __post_init__(self) -> None:
        for value in (
            self.work_order_id,
            self.tenant_id,
            self.robot_id,
            self.component_id,
            self.recommendation_id,
        ):
            validate_identifier(value)

        if self.created_by:
            validate_identifier(self.created_by)

        if self.assigned_technician_id is not None:
            validate_identifier(
                self.assigned_technician_id
            )

        if self.approval_id is not None:
            validate_identifier(self.approval_id)

        if self.estimated_duration_minutes <= 0:
            raise MaintenanceValidationError(
                "Invalid estimated duration"
            )

        if self.scheduled_start_at is not None:
            validate_timestamp(self.scheduled_start_at)

        if self.scheduled_end_at is not None:
            validate_timestamp(self.scheduled_end_at)

        if (
            self.scheduled_start_at is not None
            and self.scheduled_end_at is not None
            and self.scheduled_end_at
            <= self.scheduled_start_at
        ):
            raise MaintenanceValidationError(
                "Schedule end must follow start"
            )


@dataclass(frozen=True, slots=True)
class WorkOrderCreateResult:
    work_order: MaintenanceWorkOrder

    duplicate_detected: bool

    approval_required: bool

    approval_id: str | None


# ============================================================
# Fleet Summary
# ============================================================

@dataclass(slots=True)
class FleetMaintenanceSummary:
    tenant_id: str

    robot_count: int
    total_recommendations: int

    critical_count: int
    urgent_count: int
    high_count: int
    normal_count: int
    low_count: int

    robots_requiring_maintenance: list[str]

    generated_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# Integration Protocols
# ============================================================

class MaintenanceDataProvider(Protocol):

    async def get_robot_context(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> RobotMaintenanceContext | None:
        ...


class MaintenanceAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        robot_id: str,
        permission: str,
    ) -> None:
        ...


class MaintenanceWorkOrderRepository(Protocol):

    async def create_if_absent(
        self,
        *,
        work_order: MaintenanceWorkOrder,
        open_statuses: Sequence[MaintenanceStatus],
    ) -> tuple[MaintenanceWorkOrder, bool]:
        """
        Atomically create a work order if no matching open
        order exists for the tenant, robot, component,
        and maintenance type.

        Return (record, created).
        """
        ...

    async def get(
        self,
        *,
        tenant_id: str,
        work_order_id: str,
    ) -> MaintenanceWorkOrder | None:
        ...

    async def list_robot_orders(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> list[MaintenanceWorkOrder]:
        ...

    async def save(
        self,
        *,
        work_order: MaintenanceWorkOrder,
        expected_version: int,
    ) -> MaintenanceWorkOrder:
        ...


class MaintenanceCostProvider(Protocol):

    async def estimate_cost(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        component_id: str,
        maintenance_type: MaintenanceType,
        estimated_duration_minutes: int,
    ) -> MaintenanceCostEstimate | None:
        ...


class MaintenancePartsProvider(Protocol):

    async def recommend_parts(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        component_id: str,
        maintenance_type: MaintenanceType,
    ) -> list[RequiredPart]:
        ...


class MaintenanceApprovalProvider(Protocol):

    async def request_approval(
        self,
        *,
        tenant_id: str,
        requested_by: str,
        work_order: MaintenanceWorkOrder,
        requires_safety_review: bool,
    ) -> str:
        ...


class MaintenanceAuditSink(Protocol):

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
# Maintenance Agent
# ============================================================

class MaintenanceAgent:

    OPEN_WORK_ORDER_STATUSES = (
        MaintenanceStatus.PROPOSED,
        MaintenanceStatus.PENDING_APPROVAL,
        MaintenanceStatus.APPROVED,
        MaintenanceStatus.SCHEDULED,
        MaintenanceStatus.IN_PROGRESS,
    )

    PRIORITY_RANK = {
        MaintenancePriority.LOW: 1,
        MaintenancePriority.NORMAL: 2,
        MaintenancePriority.HIGH: 3,
        MaintenancePriority.URGENT: 4,
        MaintenancePriority.CRITICAL: 5,
    }

    def __init__(
        self,
        *,
        data_provider: MaintenanceDataProvider,
        authorizer: MaintenanceAuthorizer,
        work_order_repository: (
            MaintenanceWorkOrderRepository | None
        ) = None,
        cost_provider: MaintenanceCostProvider | None = None,
        parts_provider: MaintenancePartsProvider | None = None,
        approval_provider: (
            MaintenanceApprovalProvider | None
        ) = None,
        audit_sink: MaintenanceAuditSink | None = None,
        policy: MaintenancePolicy | None = None,
    ) -> None:

        self.data_provider = data_provider
        self.authorizer = authorizer

        self.work_order_repository = (
            work_order_repository
        )

        self.cost_provider = cost_provider
        self.parts_provider = parts_provider
        self.approval_provider = approval_provider
        self.audit_sink = audit_sink

        self.policy = policy or MaintenancePolicy()

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    async def _authorize(
        self,
        *,
        scope: MaintenanceScope,
        robot_id: str,
        permission: str,
    ) -> None:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            robot_id=robot_id,
            permission=permission,
        )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    async def _audit(
        self,
        *,
        scope: MaintenanceScope,
        action: str,
        resource_id: str,
        metadata: Mapping[str, Any],
    ) -> None:

        if self.audit_sink is None:
            return

        await self.audit_sink.record(
            tenant_id=scope.tenant_id,
            actor_id=scope.user_id,
            action=action,
            resource_id=resource_id,
            metadata=metadata,
        )

    # --------------------------------------------------------
    # Risk Assessment
    # --------------------------------------------------------

    def assess_component_risk(
        self,
        component: ComponentMaintenanceSignal,
    ) -> FailureRiskLevel:

        if component.fault_active:
            return FailureRiskLevel.CRITICAL

        if component.failure_probability is not None:
            probability = component.failure_probability

            if probability >= (
                self.policy.failure_probability_critical
            ):
                return FailureRiskLevel.CRITICAL

            if probability >= (
                self.policy.failure_probability_high
            ):
                return FailureRiskLevel.HIGH

            if probability >= (
                self.policy.failure_probability_warning
            ):
                return FailureRiskLevel.MODERATE

        if component.remaining_useful_life_hours is not None:
            rul = component.remaining_useful_life_hours

            if rul <= self.policy.rul_critical_hours:
                return FailureRiskLevel.CRITICAL

            if rul <= self.policy.rul_high_hours:
                return FailureRiskLevel.HIGH

            if rul <= self.policy.rul_warning_hours:
                return FailureRiskLevel.MODERATE

        if component.health_score is not None:
            score = component.health_score

            if score <= self.policy.health_score_critical:
                return FailureRiskLevel.CRITICAL

            if score <= self.policy.health_score_high:
                return FailureRiskLevel.HIGH

            if score <= self.policy.health_score_warning:
                return FailureRiskLevel.MODERATE

        if component.anomaly_detected:
            return FailureRiskLevel.MODERATE

        if component.maintenance_due:
            return FailureRiskLevel.MODERATE

        if (
            component.failure_probability is None
            and component.remaining_useful_life_hours is None
            and component.health_score is None
        ):
            return FailureRiskLevel.UNKNOWN

        return FailureRiskLevel.LOW

    # --------------------------------------------------------
    # Priority Mapping
    # --------------------------------------------------------

    @staticmethod
    def priority_from_risk(
        risk: FailureRiskLevel,
    ) -> MaintenancePriority:

        mapping = {
            FailureRiskLevel.LOW: MaintenancePriority.LOW,
            FailureRiskLevel.MODERATE: (
                MaintenancePriority.NORMAL
            ),
            FailureRiskLevel.HIGH: MaintenancePriority.HIGH,
            FailureRiskLevel.CRITICAL: (
                MaintenancePriority.CRITICAL
            ),
            FailureRiskLevel.UNKNOWN: (
                MaintenancePriority.NORMAL
            ),
        }

        return mapping[risk]

    # --------------------------------------------------------
    # Maintenance Type
    # --------------------------------------------------------

    @staticmethod
    def determine_maintenance_type(
        component: ComponentMaintenanceSignal,
    ) -> MaintenanceType:

        if component.fault_active:
            return MaintenanceType.CORRECTIVE

        if (
            component.failure_probability is not None
            or component.remaining_useful_life_hours is not None
        ):
            return MaintenanceType.PREDICTIVE

        if component.anomaly_detected:
            return MaintenanceType.CONDITION_BASED

        if component.maintenance_due:
            return MaintenanceType.PREVENTIVE

        return MaintenanceType.INSPECTION

    # --------------------------------------------------------
    # Trigger
    # --------------------------------------------------------

    @staticmethod
    def determine_trigger(
        component: ComponentMaintenanceSignal,
    ) -> MaintenanceTrigger:

        if component.fault_active or component.fault_codes:
            return MaintenanceTrigger.FAULT_CODE

        if component.failure_probability is not None:
            return MaintenanceTrigger.FAILURE_PREDICTION

        if component.remaining_useful_life_hours is not None:
            return MaintenanceTrigger.RUL_PREDICTION

        if component.anomaly_detected:
            return MaintenanceTrigger.SENSOR_ANOMALY

        if component.maintenance_due:
            return MaintenanceTrigger.SCHEDULED_INTERVAL

        return MaintenanceTrigger.HEALTH_ASSESSMENT

    # --------------------------------------------------------
    # Technician Skills
    # --------------------------------------------------------

    @staticmethod
    def required_skills(
        category: ComponentCategory,
    ) -> list[TechnicianSkill]:

        mapping = {
            ComponentCategory.BATTERY: [
                TechnicianSkill.BATTERY_SYSTEMS,
                TechnicianSkill.ELECTRICAL,
            ],
            ComponentCategory.MOTOR: [
                TechnicianSkill.MECHANICAL,
                TechnicianSkill.ELECTRICAL,
            ],
            ComponentCategory.ACTUATOR: [
                TechnicianSkill.ROBOTICS,
                TechnicianSkill.MECHANICAL,
            ],
            ComponentCategory.GEARBOX: [
                TechnicianSkill.MECHANICAL,
            ],
            ComponentCategory.BEARING: [
                TechnicianSkill.MECHANICAL,
            ],
            ComponentCategory.CONTROLLER: [
                TechnicianSkill.ELECTRONICS,
                TechnicianSkill.SOFTWARE,
            ],
            ComponentCategory.CAMERA: [
                TechnicianSkill.SENSOR_CALIBRATION,
            ],
            ComponentCategory.LIDAR: [
                TechnicianSkill.SENSOR_CALIBRATION,
            ],
            ComponentCategory.RADAR: [
                TechnicianSkill.SENSOR_CALIBRATION,
            ],
            ComponentCategory.IMU: [
                TechnicianSkill.SENSOR_CALIBRATION,
            ],
            ComponentCategory.GPS: [
                TechnicianSkill.SENSOR_CALIBRATION,
            ],
            ComponentCategory.ENCODER: [
                TechnicianSkill.SENSOR_CALIBRATION,
            ],
            ComponentCategory.MANIPULATOR: [
                TechnicianSkill.ROBOTICS,
                TechnicianSkill.MECHANICAL,
            ],
            ComponentCategory.END_EFFECTOR: [
                TechnicianSkill.ROBOTICS,
                TechnicianSkill.MECHANICAL,
            ],
            ComponentCategory.POWER_SYSTEM: [
                TechnicianSkill.ELECTRICAL,
            ],
            ComponentCategory.COOLING_SYSTEM: [
                TechnicianSkill.MECHANICAL,
            ],
            ComponentCategory.NETWORK: [
                TechnicianSkill.NETWORKING,
            ],
            ComponentCategory.SAFETY_SYSTEM: [
                TechnicianSkill.SAFETY,
                TechnicianSkill.ROBOTICS,
            ],
            ComponentCategory.SOFTWARE: [
                TechnicianSkill.SOFTWARE,
            ],
        }

        return mapping.get(
            category,
            [TechnicianSkill.GENERAL_MAINTENANCE],
        )

    # --------------------------------------------------------
    # Recommended Actions
    # --------------------------------------------------------

    @staticmethod
    def recommended_actions(
        *,
        maintenance_type: MaintenanceType,
        category: ComponentCategory,
    ) -> list[WorkOrderAction]:

        if maintenance_type == MaintenanceType.CORRECTIVE:
            return [
                WorkOrderAction.INSPECT,
                WorkOrderAction.DIAGNOSE,
                WorkOrderAction.REPAIR,
                WorkOrderAction.TEST,
                WorkOrderAction.VERIFY,
            ]

        if maintenance_type == MaintenanceType.PREDICTIVE:
            return [
                WorkOrderAction.INSPECT,
                WorkOrderAction.DIAGNOSE,
                WorkOrderAction.SERVICE,
                WorkOrderAction.VERIFY,
            ]

        if category in (
            ComponentCategory.CAMERA,
            ComponentCategory.LIDAR,
            ComponentCategory.RADAR,
            ComponentCategory.IMU,
            ComponentCategory.GPS,
            ComponentCategory.ENCODER,
        ):
            return [
                WorkOrderAction.INSPECT,
                WorkOrderAction.CALIBRATE,
                WorkOrderAction.TEST,
                WorkOrderAction.VERIFY,
            ]

        return [
            WorkOrderAction.INSPECT,
            WorkOrderAction.SERVICE,
            WorkOrderAction.TEST,
            WorkOrderAction.VERIFY,
        ]

    # --------------------------------------------------------
    # Recommended Deadline
    # --------------------------------------------------------

    def recommended_deadline(
        self,
        *,
        priority: MaintenancePriority,
        remaining_useful_life_hours: float | None,
    ) -> datetime:

        now = utc_now()

        priority_hours = {
            MaintenancePriority.LOW: 24 * 30,
            MaintenancePriority.NORMAL: 24 * 7,
            MaintenancePriority.HIGH: 48,
            MaintenancePriority.URGENT: 12,
            MaintenancePriority.CRITICAL: 1,
        }

        deadline_hours = priority_hours[priority]

        if remaining_useful_life_hours is not None:
            # Planning margin only; not a guarantee that
            # the component remains safe until this time.
            deadline_hours = min(
                deadline_hours,
                max(0.0, remaining_useful_life_hours * 0.5),
            )

        return now + timedelta(hours=deadline_hours)

    # --------------------------------------------------------
    # Recommendation Generation
    # --------------------------------------------------------

    async def generate_recommendation(
        self,
        *,
        scope: MaintenanceScope,
        context: RobotMaintenanceContext,
        component: ComponentMaintenanceSignal,
    ) -> MaintenanceRecommendation | None:

        risk = self.assess_component_risk(component)

        if (
            risk == FailureRiskLevel.LOW
            and not component.maintenance_due
            and not component.anomaly_detected
            and not component.fault_active
        ):
            return None

        priority = self.priority_from_risk(risk)

        maintenance_type = (
            self.determine_maintenance_type(component)
        )

        trigger = self.determine_trigger(component)

        reasoning: list[str] = []

        if component.fault_active:
            reasoning.append(
                "Component reports an active fault."
            )

        if component.failure_probability is not None:
            reasoning.append(
                "Estimated failure probability: "
                f"{component.failure_probability:.2%}."
            )

        if component.remaining_useful_life_hours is not None:
            reasoning.append(
                "Estimated remaining useful life: "
                f"{component.remaining_useful_life_hours:.1f} hours."
            )

        if component.health_score is not None:
            reasoning.append(
                "Component health score: "
                f"{component.health_score:.2f}."
            )

        if component.anomaly_detected:
            reasoning.append(
                "Component anomaly was detected."
            )

        if component.maintenance_due:
            reasoning.append(
                "Scheduled maintenance is due."
            )

        if component.fault_codes:
            reasoning.append(
                "Reported fault codes: "
                + ", ".join(component.fault_codes[:10])
            )

        safety_review = (
            risk == FailureRiskLevel.CRITICAL
            or context.safety_fault_active
            or context.emergency_stop_active
            or component.category
            == ComponentCategory.SAFETY_SYSTEM
        )

        required_parts: list[RequiredPart] = []

        if self.parts_provider is not None:
            required_parts = (
                await self.parts_provider.recommend_parts(
                    tenant_id=scope.tenant_id,
                    robot_id=context.robot_id,
                    component_id=component.component_id,
                    maintenance_type=maintenance_type,
                )
            )

            if len(required_parts) > MAX_REQUIRED_PARTS:
                raise MaintenanceValidationError(
                    "Parts provider returned too many parts"
                )

        duration_minutes = (
            DEFAULT_WORK_ORDER_DURATION_MINUTES
        )

        cost_estimate = None

        if self.cost_provider is not None:
            cost_estimate = (
                await self.cost_provider.estimate_cost(
                    tenant_id=scope.tenant_id,
                    robot_id=context.robot_id,
                    component_id=component.component_id,
                    maintenance_type=maintenance_type,
                    estimated_duration_minutes=duration_minutes,
                )
            )

        return MaintenanceRecommendation(
            recommendation_id=generate_id(),
            tenant_id=scope.tenant_id,
            robot_id=context.robot_id,
            component_id=component.component_id,
            category=component.category,
            maintenance_type=maintenance_type,
            priority=priority,
            title=(
                f"{maintenance_type.value.replace('_', ' ').title()} "
                f"maintenance: {component.component_id}"
            ),
            description=(
                "Review and perform qualified maintenance "
                f"for component {component.component_id}."
            ),
            risk_level=risk,
            trigger=trigger,
            recommended_actions=self.recommended_actions(
                maintenance_type=maintenance_type,
                category=component.category,
            ),
            required_skills=self.required_skills(
                component.category
            ),
            failure_probability=(
                component.failure_probability
            ),
            remaining_useful_life_hours=(
                component.remaining_useful_life_hours
            ),
            recommended_by=self.recommended_deadline(
                priority=priority,
                remaining_useful_life_hours=(
                    component.remaining_useful_life_hours
                ),
            ),
            estimated_duration_minutes=duration_minutes,
            required_parts=required_parts,
            cost_estimate=cost_estimate,
            requires_human_approval=True,
            requires_safety_review=safety_review,
            reasoning=reasoning,
        )

    # --------------------------------------------------------
    # Maintenance Plan
    # --------------------------------------------------------

    async def create_maintenance_plan(
        self,
        *,
        scope: MaintenanceScope,
        robot_id: str,
    ) -> MaintenancePlan:

        validate_identifier(robot_id)

        await self._authorize(
            scope=scope,
            robot_id=robot_id,
            permission="maintenance:plan",
        )

        context = await self.data_provider.get_robot_context(
            tenant_id=scope.tenant_id,
            robot_id=robot_id,
        )

        if context is None:
            raise MaintenanceNotFoundError(
                "Robot maintenance context not found"
            )

        if (
            context.tenant_id != scope.tenant_id
            or context.robot_id != robot_id
        ):
            raise MaintenancePermissionError(
                "Maintenance context scope mismatch"
            )

        recommendations: list[
            MaintenanceRecommendation
        ] = []

        for component in context.components:
            recommendation = (
                await self.generate_recommendation(
                    scope=scope,
                    context=context,
                    component=component,
                )
            )

            if recommendation is not None:
                recommendations.append(recommendation)

        if len(recommendations) > MAX_RECOMMENDATIONS:
            raise MaintenanceValidationError(
                "Too many maintenance recommendations"
            )

        recommendations.sort(
            key=lambda recommendation: (
                -self.PRIORITY_RANK[
                    recommendation.priority
                ],
                recommendation.recommended_by
                or datetime.max.replace(
                    tzinfo=timezone.utc
                ),
            )
        )

        if recommendations:
            overall_priority = max(
                (
                    recommendation.priority
                    for recommendation in recommendations
                ),
                key=lambda priority: (
                    self.PRIORITY_RANK[priority]
                ),
            )
        else:
            overall_priority = MaintenancePriority.LOW

        requires_safety_review = any(
            recommendation.requires_safety_review
            for recommendation in recommendations
        )

        duration = sum(
            recommendation.estimated_duration_minutes
            for recommendation in recommendations
        )

        estimates = [
            recommendation.cost_estimate
            for recommendation in recommendations
        ]

        known_estimates = [
            estimate
            for estimate in estimates
            if estimate is not None
        ]

        currencies = {
            estimate.currency
            for estimate in known_estimates
        }

        if (
            recommendations
            and len(known_estimates) == len(recommendations)
            and len(currencies) == 1
        ):
            total_cost = round(
                sum(
                    estimate.total_cost
                    for estimate in known_estimates
                ),
                2,
            )
            currency = next(iter(currencies))
        else:
            total_cost = None
            currency = None

        plan = MaintenancePlan(
            plan_id=generate_id(),
            tenant_id=scope.tenant_id,
            robot_id=robot_id,
            recommendations=recommendations,
            overall_priority=overall_priority,
            requires_human_approval=bool(recommendations),
            requires_safety_review=requires_safety_review,
            total_estimated_duration_minutes=duration,
            total_estimated_cost=total_cost,
            currency=currency,
        )

        await self._audit(
            scope=scope,
            action="maintenance_agent.plan_created",
            resource_id=plan.plan_id,
            metadata={
                "robot_id": robot_id,
                "recommendation_count": len(
                    recommendations
                ),
                "overall_priority": overall_priority.value,
                "requires_safety_review": (
                    requires_safety_review
                ),
            },
        )

        return plan

    # --------------------------------------------------------
    # Create Work Order
    # --------------------------------------------------------

    async def create_work_order(
        self,
        *,
        scope: MaintenanceScope,
        recommendation: MaintenanceRecommendation,
    ) -> WorkOrderCreateResult:

        if self.work_order_repository is None:
            raise MaintenanceServiceUnavailableError(
                "Work-order repository is not configured"
            )

        if recommendation.tenant_id != scope.tenant_id:
            raise MaintenancePermissionError(
                "Recommendation tenant mismatch"
            )

        await self._authorize(
            scope=scope,
            robot_id=recommendation.robot_id,
            permission="maintenance:work_order:create",
        )

        work_order = MaintenanceWorkOrder(
            work_order_id=generate_id(),
            tenant_id=scope.tenant_id,
            robot_id=recommendation.robot_id,
            component_id=recommendation.component_id,
            recommendation_id=(
                recommendation.recommendation_id
            ),
            title=recommendation.title,
            description=recommendation.description,
            maintenance_type=recommendation.maintenance_type,
            priority=recommendation.priority,
            status=MaintenanceStatus.PROPOSED,
            required_skills=copy.deepcopy(
                recommendation.required_skills
            ),
            required_parts=copy.deepcopy(
                recommendation.required_parts
            ),
            estimated_duration_minutes=(
                recommendation.estimated_duration_minutes
            ),
            created_by=scope.user_id,
        )

        saved, created = (
            await self.work_order_repository.create_if_absent(
                work_order=work_order,
                open_statuses=self.OPEN_WORK_ORDER_STATUSES,
            )
        )

        if (
            saved.tenant_id != scope.tenant_id
            or saved.robot_id != recommendation.robot_id
        ):
            raise MaintenancePermissionError(
                "Repository returned mismatched work order"
            )

        if not created:
            return WorkOrderCreateResult(
                work_order=saved,
                duplicate_detected=True,
                approval_required=(
                    saved.status
                    == MaintenanceStatus.PENDING_APPROVAL
                ),
                approval_id=saved.approval_id,
            )

        # The work order is only proposed. No approval
        # or robot maintenance execution occurs here.
        await self._audit(
            scope=scope,
            action="maintenance_agent.work_order_created",
            resource_id=saved.work_order_id,
            metadata={
                "robot_id": saved.robot_id,
                "component_id": saved.component_id,
                "priority": saved.priority.value,
            },
        )

        return WorkOrderCreateResult(
            work_order=saved,
            duplicate_detected=False,
            approval_required=True,
            approval_id=None,
        )

    # --------------------------------------------------------
    # Request Human Approval
    # --------------------------------------------------------

    async def request_work_order_approval(
        self,
        *,
        scope: MaintenanceScope,
        work_order_id: str,
        requires_safety_review: bool,
    ) -> MaintenanceWorkOrder:

        if self.work_order_repository is None:
            raise MaintenanceServiceUnavailableError(
                "Work-order repository is not configured"
            )

        if self.approval_provider is None:
            raise MaintenanceServiceUnavailableError(
                "Approval provider is not configured"
            )

        validate_identifier(work_order_id)

        order = await self.work_order_repository.get(
            tenant_id=scope.tenant_id,
            work_order_id=work_order_id,
        )

        if order is None:
            raise MaintenanceNotFoundError(
                "Work order not found"
            )

        await self._authorize(
            scope=scope,
            robot_id=order.robot_id,
            permission="maintenance:approval:request",
        )

        if order.status != MaintenanceStatus.PROPOSED:
            raise MaintenanceConflictError(
                "Only proposed work orders can request approval"
            )

        if order.approval_id is not None:
            raise MaintenanceConflictError(
                "Work order already has an approval request"
            )

        approval_id = (
            await self.approval_provider.request_approval(
                tenant_id=scope.tenant_id,
                requested_by=scope.user_id,
                work_order=order,
                requires_safety_review=requires_safety_review,
            )
        )

        validate_identifier(approval_id)

        previous_version = order.version

        order.approval_id = approval_id
        order.status = MaintenanceStatus.PENDING_APPROVAL
        order.updated_at = utc_now()
        order.version += 1

        saved = await self.work_order_repository.save(
            work_order=order,
            expected_version=previous_version,
        )

        await self._audit(
            scope=scope,
            action="maintenance_agent.approval_requested",
            resource_id=work_order_id,
            metadata={
                "approval_id": approval_id,
                "requires_safety_review": (
                    requires_safety_review
                ),
            },
        )

        return saved

    # --------------------------------------------------------
    # Fleet Maintenance Summary
    # --------------------------------------------------------

    async def assess_fleet_maintenance(
        self,
        *,
        scope: MaintenanceScope,
        robot_ids: Sequence[str],
    ) -> FleetMaintenanceSummary:

        if len(robot_ids) > MAX_ROBOTS_PER_BATCH:
            raise MaintenanceValidationError(
                "Too many robots in batch"
            )

        if len(robot_ids) != len(set(robot_ids)):
            raise MaintenanceValidationError(
                "Duplicate robot IDs"
            )

        for robot_id in robot_ids:
            validate_identifier(robot_id)

        semaphore = asyncio.Semaphore(10)

        async def process_robot(
            robot_id: str,
        ) -> MaintenancePlan:
            async with semaphore:
                return await self.create_maintenance_plan(
                    scope=scope,
                    robot_id=robot_id,
                )

        plans = await asyncio.gather(
            *(
                process_robot(robot_id)
                for robot_id in robot_ids
            )
        )

        recommendations = [
            recommendation
            for plan in plans
            for recommendation in plan.recommendations
        ]

        def count_priority(
            priority: MaintenancePriority,
        ) -> int:
            return sum(
                recommendation.priority == priority
                for recommendation in recommendations
            )

        return FleetMaintenanceSummary(
            tenant_id=scope.tenant_id,
            robot_count=len(plans),
            total_recommendations=len(recommendations),
            critical_count=count_priority(
                MaintenancePriority.CRITICAL
            ),
            urgent_count=count_priority(
                MaintenancePriority.URGENT
            ),
            high_count=count_priority(
                MaintenancePriority.HIGH
            ),
            normal_count=count_priority(
                MaintenancePriority.NORMAL
            ),
            low_count=count_priority(
                MaintenancePriority.LOW
            ),
            robots_requiring_maintenance=[
                plan.robot_id
                for plan in plans
                if plan.recommendations
            ],
        )

    # --------------------------------------------------------
    # Explainable Maintenance Report
    # --------------------------------------------------------

    @staticmethod
    def generate_maintenance_report(
        plan: MaintenancePlan,
    ) -> dict[str, Any]:

        return {
            "plan_id": plan.plan_id,
            "tenant_id": plan.tenant_id,
            "robot_id": plan.robot_id,
            "generated_at": plan.generated_at.isoformat(),
            "summary": {
                "recommendation_count": len(
                    plan.recommendations
                ),
                "overall_priority": (
                    plan.overall_priority.value
                ),
                "requires_human_approval": (
                    plan.requires_human_approval
                ),
                "requires_safety_review": (
                    plan.requires_safety_review
                ),
                "estimated_duration_minutes": (
                    plan.total_estimated_duration_minutes
                ),
                "estimated_cost": (
                    plan.total_estimated_cost
                ),
                "currency": plan.currency,
            },
            "recommendations": [
                {
                    "recommendation_id": (
                        recommendation.recommendation_id
                    ),
                    "component_id": (
                        recommendation.component_id
                    ),
                    "component_category": (
                        recommendation.category.value
                    ),
                    "maintenance_type": (
                        recommendation.maintenance_type.value
                    ),
                    "priority": (
                        recommendation.priority.value
                    ),
                    "risk_level": (
                        recommendation.risk_level.value
                    ),
                    "trigger": (
                        recommendation.trigger.value
                    ),
                    "title": recommendation.title,
                    "description": recommendation.description,
                    "recommended_actions": [
                        action.value
                        for action
                        in recommendation.recommended_actions
                    ],
                    "required_skills": [
                        skill.value
                        for skill
                        in recommendation.required_skills
                    ],
                    "failure_probability": (
                        recommendation.failure_probability
                    ),
                    "remaining_useful_life_hours": (
                        recommendation.remaining_useful_life_hours
                    ),
                    "recommended_by": (
                        recommendation.recommended_by.isoformat()
                        if recommendation.recommended_by
                        else None
                    ),
                    "estimated_duration_minutes": (
                        recommendation.estimated_duration_minutes
                    ),
                    "requires_human_approval": (
                        recommendation.requires_human_approval
                    ),
                    "requires_safety_review": (
                        recommendation.requires_safety_review
                    ),
                    "reasoning": recommendation.reasoning,
                }
                for recommendation
                in plan.recommendations
            ],
            "safety": {
                "physical_actuation_enabled": False,
                "autonomous_repair_enabled": False,
                "emergency_stop_override_enabled": False,
                "human_approval_required": True,
                "maintenance_plan_is_advisory": True,
            },
        }


# ============================================================
# Factory
# ============================================================

def create_maintenance_agent(
    *,
    data_provider: MaintenanceDataProvider,
    authorizer: MaintenanceAuthorizer,
    work_order_repository: (
        MaintenanceWorkOrderRepository | None
    ) = None,
    cost_provider: MaintenanceCostProvider | None = None,
    parts_provider: MaintenancePartsProvider | None = None,
    approval_provider: (
        MaintenanceApprovalProvider | None
    ) = None,
    audit_sink: MaintenanceAuditSink | None = None,
    policy: MaintenancePolicy | None = None,
) -> MaintenanceAgent:

    return MaintenanceAgent(
        data_provider=data_provider,
        authorizer=authorizer,
        work_order_repository=work_order_repository,
        cost_provider=cost_provider,
        parts_provider=parts_provider,
        approval_provider=approval_provider,
        audit_sink=audit_sink,
        policy=policy,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "MaintenanceType",
    "MaintenancePriority",
    "MaintenanceStatus",
    "ComponentCategory",
    "FailureRiskLevel",
    "MaintenanceTrigger",
    "TechnicianSkill",
    "WorkOrderAction",
    "MaintenanceAgentError",
    "MaintenanceValidationError",
    "MaintenanceNotFoundError",
    "MaintenancePermissionError",
    "MaintenanceConflictError",
    "MaintenanceServiceUnavailableError",
    "MaintenanceScope",
    "MaintenancePolicy",
    "ComponentMaintenanceSignal",
    "RobotMaintenanceContext",
    "RequiredPart",
    "MaintenanceCostEstimate",
    "MaintenanceRecommendation",
    "MaintenancePlan",
    "MaintenanceWorkOrder",
    "WorkOrderCreateResult",
    "FleetMaintenanceSummary",
    "MaintenanceDataProvider",
    "MaintenanceAuthorizer",
    "MaintenanceWorkOrderRepository",
    "MaintenanceCostProvider",
    "MaintenancePartsProvider",
    "MaintenanceApprovalProvider",
    "MaintenanceAuditSink",
    "MaintenanceAgent",
    "create_maintenance_agent",
]
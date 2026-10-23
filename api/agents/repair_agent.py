"""Repair Agent module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/repair_agent.py

Physical AI & Autonomous Robotics Intelligence Platform
Robot Repair, Diagnostics & Parts Intelligence Agent

Python 3.11+

Features:
- Robot fault and damage assessment
- AI-assisted repair recommendations
- Component failure analysis
- Repair versus replacement decisions
- Robot part identification
- Uploaded component image analysis integration
- Spare-parts compatibility verification
- Repair labor and parts cost estimation
- Robot downtime cost estimation
- Maintenance and repair prioritization
- Remaining useful life integration
- Repair workflow planning
- Technician and human review
- Repair risk assessment
- Tenant-aware authorization
- Audit logging
- Structured repair reports

Safety:
- Advisory analysis only.
- No automatic physical repair.
- No direct actuator or motor commands.
- No bypassing emergency stops or interlocks.
- Safety-critical repairs require qualified
  technician review and independent validation.
"""

from __future__ import annotations

import asyncio
import copy
import math
import re
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


# ============================================================
# Constants
# ============================================================

MAX_FAULTS = 200
MAX_PARTS = 500
MAX_REPAIR_STEPS = 100
MAX_IMAGES = 30
MAX_BATCH_REQUESTS = 25
MAX_RECOMMENDATIONS = 100

DEFAULT_LABOR_RATE = Decimal("500.00")
DEFAULT_DOWNTIME_COST_PER_HOUR = Decimal("1000.00")

MONEY_QUANTUM = Decimal("0.01")

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)


# ============================================================
# Exceptions
# ============================================================

class RepairAgentError(Exception):
    """Base repair agent exception."""


class RepairValidationError(RepairAgentError):
    """Invalid repair input."""


class RepairPermissionError(RepairAgentError):
    """Repair authorization failure."""


class RepairNotFoundError(RepairAgentError):
    """Robot or repair resource not found."""


class RepairModelError(RepairAgentError):
    """AI repair model failure."""


class RepairCompatibilityError(RepairAgentError):
    """Part compatibility could not be established."""


class RepairSafetyError(RepairAgentError):
    """Unsafe repair recommendation."""


class RepairServiceUnavailableError(RepairAgentError):
    """Required repair service unavailable."""


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
        raise RepairValidationError(
            f"Invalid identifier: {value!r}"
        )
    return value


def validate_finite(value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value):
        raise RepairValidationError(
            "Value must be finite"
        )
    return value


def validate_nonnegative(value: float) -> float:
    validate_finite(value)

    if value < 0:
        raise RepairValidationError(
            "Value must be nonnegative"
        )

    return value


def validate_positive(value: float) -> float:
    validate_finite(value)

    if value <= 0:
        raise RepairValidationError(
            "Value must be positive"
        )

    return value


def validate_probability(value: float) -> float:
    validate_finite(value)

    if not 0.0 <= value <= 1.0:
        raise RepairValidationError(
            "Probability must be between 0 and 1"
        )

    return value


def validate_timestamp(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
    ):
        raise RepairValidationError(
            "Datetime must be timezone-aware"
        )

    return value


def money(value: Decimal | str | int) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise RepairValidationError(
            "Use Decimal, string, or integer for money"
        )

    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RepairValidationError(
            "Invalid monetary amount"
        ) from exc

    if not amount.is_finite() or amount < 0:
        raise RepairValidationError(
            "Money must be finite and nonnegative"
        )

    return amount.quantize(
        MONEY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, value))


# ============================================================
# Enumerations
# ============================================================

class RobotComponentType(str, Enum):
    BATTERY = "battery"
    MOTOR = "motor"
    ACTUATOR = "actuator"
    GEARBOX = "gearbox"
    BEARING = "bearing"
    WHEEL = "wheel"
    TRACK = "track"
    ARM = "arm"
    JOINT = "joint"
    GRIPPER = "gripper"
    END_EFFECTOR = "end_effector"
    CONTROLLER = "controller"
    CPU = "cpu"
    GPU = "gpu"
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    ENCODER = "encoder"
    GPS = "gps"
    NETWORK_MODULE = "network_module"
    POWER_SUPPLY = "power_supply"
    COOLING_SYSTEM = "cooling_system"
    SAFETY_CONTROLLER = "safety_controller"
    EMERGENCY_STOP = "emergency_stop"
    WIRING = "wiring"
    CONNECTOR = "connector"
    CHASSIS = "chassis"
    CUSTOM = "custom"


class FaultType(str, Enum):
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    ELECTRONIC = "electronic"
    SOFTWARE = "software"
    SENSOR = "sensor"
    COMMUNICATION = "communication"
    THERMAL = "thermal"
    POWER = "power"
    STRUCTURAL = "structural"
    HYDRAULIC = "hydraulic"
    PNEUMATIC = "pneumatic"
    CALIBRATION = "calibration"
    WEAR = "wear"
    UNKNOWN = "unknown"


class FaultSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RepairPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class RepairDecision(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    INSPECT = "inspect"
    DEFER = "defer"
    UNDETERMINED = "undetermined"


class RepairStatus(str, Enum):
    DRAFT = "draft"
    ANALYZED = "analyzed"
    REVIEW_REQUIRED = "review_required"
    BLOCKED = "blocked"
    APPROVED_FOR_PLANNING = "approved_for_planning"


class RepairRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class PartCompatibilityStatus(str, Enum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class RepairStepType(str, Enum):
    INSPECTION = "inspection"
    DIAGNOSTICS = "diagnostics"
    SAFETY_ISOLATION = "safety_isolation"
    DISASSEMBLY = "disassembly"
    PART_REPLACEMENT = "part_replacement"
    REPAIR = "repair"
    CALIBRATION = "calibration"
    SOFTWARE_VALIDATION = "software_validation"
    FUNCTIONAL_TEST = "functional_test"
    SAFETY_TEST = "safety_test"
    HUMAN_REVIEW = "human_review"
    DOCUMENTATION = "documentation"


class Currency(str, Enum):
    INR = "INR"
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    AED = "AED"
    CAD = "CAD"
    AUD = "AUD"
    SGD = "SGD"


# ============================================================
# Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class RepairScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Robot Models
# ============================================================

@dataclass(frozen=True, slots=True)
class RepairRobot:
    robot_id: str
    tenant_id: str

    manufacturer: str
    model: str

    serial_number: str | None = None

    operational: bool = False
    emergency_stop_active: bool = False
    safety_fault_active: bool = False

    warranty_active: bool | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.robot_id)
        validate_identifier(self.tenant_id)

        if not self.manufacturer.strip():
            raise RepairValidationError(
                "Robot manufacturer is required"
            )

        if not self.model.strip():
            raise RepairValidationError(
                "Robot model is required"
            )


# ============================================================
# Fault Models
# ============================================================

@dataclass(frozen=True, slots=True)
class RobotFault:
    fault_id: str
    component_id: str

    component_type: RobotComponentType

    fault_type: FaultType
    severity: FaultSeverity

    description: str

    confidence: float = 1.0

    failure_probability: float | None = None
    remaining_useful_life_hours: float | None = None

    detected_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(self.fault_id)
        validate_identifier(self.component_id)

        validate_probability(self.confidence)
        validate_timestamp(self.detected_at)

        if self.failure_probability is not None:
            validate_probability(
                self.failure_probability
            )

        if self.remaining_useful_life_hours is not None:
            validate_nonnegative(
                self.remaining_useful_life_hours
            )


@dataclass(frozen=True, slots=True)
class ComponentImage:
    image_id: str

    component_id: str | None

    image_uri: str

    captured_at: datetime

    description: str = ""

    def __post_init__(self) -> None:
        validate_identifier(self.image_id)

        if self.component_id is not None:
            validate_identifier(self.component_id)

        validate_timestamp(self.captured_at)

        if not self.image_uri.strip():
            raise RepairValidationError(
                "Image URI is required"
            )


@dataclass(frozen=True, slots=True)
class VisualDamageFinding:
    finding_id: str

    component_id: str | None

    damage_type: str
    description: str

    severity: FaultSeverity
    confidence: float

    source_image_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.finding_id)
        validate_identifier(self.source_image_id)

        if self.component_id is not None:
            validate_identifier(self.component_id)

        validate_probability(self.confidence)


# ============================================================
# Spare Parts
# ============================================================

@dataclass(frozen=True, slots=True)
class RobotSparePart:
    part_id: str
    part_number: str

    name: str

    component_type: RobotComponentType

    manufacturer: str

    compatible_robot_models: frozenset[str]

    unit_price: Decimal

    currency: Currency

    available_quantity: int = 0

    estimated_delivery_days: int = 0

    def __post_init__(self) -> None:
        validate_identifier(self.part_id)

        if not self.part_number.strip():
            raise RepairValidationError(
                "Part number is required"
            )

        if not self.name.strip():
            raise RepairValidationError(
                "Part name is required"
            )

        object.__setattr__(
            self,
            "unit_price",
            money(self.unit_price),
        )

        if self.available_quantity < 0:
            raise RepairValidationError(
                "Part stock cannot be negative"
            )

        if self.estimated_delivery_days < 0:
            raise RepairValidationError(
                "Delivery days cannot be negative"
            )


@dataclass(frozen=True, slots=True)
class PartCompatibilityAssessment:
    part_id: str

    status: PartCompatibilityStatus

    confidence: float

    reasons: tuple[str, ...]

    requires_technician_verification: bool = True


# ============================================================
# Repair Cost Models
# ============================================================

@dataclass(frozen=True, slots=True)
class RepairCostConfiguration:
    currency: Currency = Currency.INR

    labor_rate_per_hour: Decimal = (
        DEFAULT_LABOR_RATE
    )

    downtime_cost_per_hour: Decimal = (
        DEFAULT_DOWNTIME_COST_PER_HOUR
    )

    contingency_percentage: Decimal = Decimal("10.00")

    tax_percentage: Decimal = Decimal("0.00")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "labor_rate_per_hour",
            money(self.labor_rate_per_hour),
        )

        object.__setattr__(
            self,
            "downtime_cost_per_hour",
            money(self.downtime_cost_per_hour),
        )

        for name in (
            "contingency_percentage",
            "tax_percentage",
        ):
            value = Decimal(
                str(getattr(self, name))
            )

            if (
                not value.is_finite()
                or not Decimal("0") <= value <= Decimal("100")
            ):
                raise RepairValidationError(
                    f"Invalid {name}"
                )

            object.__setattr__(
                self,
                name,
                value,
            )


@dataclass(frozen=True, slots=True)
class RepairCostEstimate:
    currency: Currency

    parts_cost: Decimal
    labor_cost: Decimal
    downtime_cost: Decimal

    contingency_cost: Decimal
    tax_cost: Decimal

    total_cost: Decimal

    estimated_labor_hours: float
    estimated_downtime_hours: float

    assumptions: tuple[str, ...]


# ============================================================
# Repair Workflow
# ============================================================

@dataclass(frozen=True, slots=True)
class RepairStep:
    step_id: str

    step_type: RepairStepType

    title: str
    description: str

    estimated_duration_minutes: float

    requires_qualified_technician: bool = True

    requires_human_approval: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.step_id)

        validate_nonnegative(
            self.estimated_duration_minutes
        )


@dataclass(frozen=True, slots=True)
class RepairRecommendation:
    recommendation_id: str

    component_id: str

    decision: RepairDecision

    priority: RepairPriority

    title: str
    explanation: str

    estimated_repair_cost: Decimal | None

    estimated_replacement_cost: Decimal | None

    requires_human_review: bool

    confidence: float

    def __post_init__(self) -> None:
        validate_identifier(self.recommendation_id)
        validate_identifier(self.component_id)

        validate_probability(self.confidence)


@dataclass(frozen=True, slots=True)
class RepairSafetyAssessment:
    risk_level: RepairRiskLevel

    safe_for_automatic_execution: bool

    requires_qualified_technician: bool

    requires_human_review: bool

    findings: tuple[str, ...]


# ============================================================
# Repair Request
# ============================================================

@dataclass(slots=True)
class RepairAnalysisRequest:
    tenant_id: str
    robot_id: str

    faults: list[RobotFault] = field(
        default_factory=list
    )

    component_images: list[ComponentImage] = field(
        default_factory=list
    )

    candidate_parts: list[RobotSparePart] = field(
        default_factory=list
    )

    estimated_labor_hours: float = 0.0
    estimated_downtime_hours: float = 0.0

    replacement_cost: Decimal | None = None

    cost_configuration: RepairCostConfiguration = field(
        default_factory=RepairCostConfiguration
    )

    include_visual_analysis: bool = True
    include_part_compatibility: bool = True

    simulation_only: bool = True

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.robot_id)

        if len(self.faults) > MAX_FAULTS:
            raise RepairValidationError(
                "Too many faults"
            )

        if len(self.component_images) > MAX_IMAGES:
            raise RepairValidationError(
                "Too many component images"
            )

        if len(self.candidate_parts) > MAX_PARTS:
            raise RepairValidationError(
                "Too many candidate parts"
            )

        validate_nonnegative(
            self.estimated_labor_hours
        )

        validate_nonnegative(
            self.estimated_downtime_hours
        )

        if self.replacement_cost is not None:
            self.replacement_cost = money(
                self.replacement_cost
            )

        if not self.simulation_only:
            raise RepairSafetyError(
                "RepairAgent is advisory only"
            )


# ============================================================
# Repair Result
# ============================================================

@dataclass(slots=True)
class RepairAnalysisResult:
    analysis_id: str

    tenant_id: str
    robot_id: str

    status: RepairStatus

    faults: list[RobotFault]

    visual_findings: list[VisualDamageFinding]

    part_assessments: list[PartCompatibilityAssessment]

    cost_estimate: RepairCostEstimate

    recommendations: list[RepairRecommendation]

    repair_steps: list[RepairStep]

    safety_assessment: RepairSafetyAssessment

    reasoning: list[str]

    requires_human_review: bool

    generated_at: datetime = field(
        default_factory=utc_now
    )

    simulation_only: bool = True


# ============================================================
# Provider Protocols
# ============================================================

class RepairRobotProvider(Protocol):

    async def get_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> RepairRobot | None:
        ...


class VisualDamageAnalysisProvider(Protocol):

    async def analyze_images(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        images: Sequence[ComponentImage],
    ) -> list[VisualDamageFinding]:
        ...


class PartCompatibilityProvider(Protocol):

    async def assess_compatibility(
        self,
        *,
        tenant_id: str,
        robot: RepairRobot,
        part: RobotSparePart,
    ) -> PartCompatibilityAssessment:
        ...


class RepairKnowledgeProvider(Protocol):

    async def get_repair_guidance(
        self,
        *,
        tenant_id: str,
        robot: RepairRobot,
        fault: RobotFault,
    ) -> list[str]:
        ...


class RepairAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        robot_id: str,
        permission: str,
    ) -> None:
        ...


class RepairRepository(Protocol):

    async def save_analysis(
        self,
        *,
        analysis: RepairAnalysisResult,
    ) -> None:
        ...


class RepairAuditSink(Protocol):

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
# Repair Cost Engine
# ============================================================

class RepairCostEngine:

    @staticmethod
    def calculate_cost(
        *,
        parts: Sequence[RobotSparePart],
        labor_hours: float,
        downtime_hours: float,
        configuration: RepairCostConfiguration,
    ) -> RepairCostEstimate:

        validate_nonnegative(labor_hours)
        validate_nonnegative(downtime_hours)

        for part in parts:
            if part.currency != configuration.currency:
                raise RepairValidationError(
                    "Mixed currencies require an explicit "
                    "exchange-rate conversion provider"
                )

        parts_cost = money(
            sum(
                (part.unit_price for part in parts),
                Decimal("0"),
            )
        )

        labor_cost = money(
            configuration.labor_rate_per_hour
            * Decimal(str(labor_hours))
        )

        downtime_cost = money(
            configuration.downtime_cost_per_hour
            * Decimal(str(downtime_hours))
        )

        subtotal = (
            parts_cost
            + labor_cost
            + downtime_cost
        )

        contingency = money(
            subtotal
            * configuration.contingency_percentage
            / Decimal("100")
        )

        taxable_amount = subtotal + contingency

        tax = money(
            taxable_amount
            * configuration.tax_percentage
            / Decimal("100")
        )

        total = money(
            taxable_amount + tax
        )

        return RepairCostEstimate(
            currency=configuration.currency,
            parts_cost=parts_cost,
            labor_cost=labor_cost,
            downtime_cost=downtime_cost,
            contingency_cost=contingency,
            tax_cost=tax,
            total_cost=total,
            estimated_labor_hours=labor_hours,
            estimated_downtime_hours=downtime_hours,
            assumptions=(
                "Each selected spare part is counted once.",
                "Labor rate is configurable.",
                "Downtime cost is an estimate.",
                "No currency conversion was performed.",
                "Actual quotations may differ.",
            ),
        )


# ============================================================
# Repair Decision Engine
# ============================================================

class RepairDecisionEngine:

    @staticmethod
    def determine_priority(
        fault: RobotFault,
    ) -> RepairPriority:

        if fault.severity == FaultSeverity.CRITICAL:
            return RepairPriority.URGENT

        if fault.severity == FaultSeverity.HIGH:
            return RepairPriority.HIGH

        if fault.severity == FaultSeverity.MODERATE:
            return RepairPriority.NORMAL

        return RepairPriority.LOW

    @staticmethod
    def recommend_decision(
        *,
        fault: RobotFault,
        repair_cost: Decimal | None,
        replacement_cost: Decimal | None,
        compatible_replacement_available: bool,
    ) -> RepairDecision:

        if fault.confidence < 0.60:
            return RepairDecision.INSPECT

        if fault.severity == FaultSeverity.CRITICAL:
            return RepairDecision.INSPECT

        if (
            fault.remaining_useful_life_hours is not None
            and fault.remaining_useful_life_hours < 50
            and compatible_replacement_available
        ):
            return RepairDecision.REPLACE

        if (
            repair_cost is None
            or replacement_cost is None
        ):
            return RepairDecision.UNDETERMINED

        if replacement_cost <= 0:
            return RepairDecision.UNDETERMINED

        repair_ratio = (
            repair_cost / replacement_cost
        )

        if (
            repair_ratio >= Decimal("0.75")
            and compatible_replacement_available
        ):
            return RepairDecision.REPLACE

        if repair_ratio <= Decimal("0.50"):
            return RepairDecision.REPAIR

        return RepairDecision.INSPECT


# ============================================================
# Repair Agent
# ============================================================

class RepairAgent:

    def __init__(
        self,
        *,
        robot_provider: RepairRobotProvider,
        authorizer: RepairAuthorizer,
        visual_provider: (
            VisualDamageAnalysisProvider | None
        ) = None,
        compatibility_provider: (
            PartCompatibilityProvider | None
        ) = None,
        knowledge_provider: (
            RepairKnowledgeProvider | None
        ) = None,
        repository: RepairRepository | None = None,
        audit_sink: RepairAuditSink | None = None,
        cost_engine: RepairCostEngine | None = None,
        decision_engine: RepairDecisionEngine | None = None,
    ) -> None:

        self.robot_provider = robot_provider
        self.authorizer = authorizer

        self.visual_provider = visual_provider
        self.compatibility_provider = (
            compatibility_provider
        )
        self.knowledge_provider = knowledge_provider

        self.repository = repository
        self.audit_sink = audit_sink

        self.cost_engine = (
            cost_engine or RepairCostEngine()
        )

        self.decision_engine = (
            decision_engine or RepairDecisionEngine()
        )

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    async def _authorize(
        self,
        *,
        scope: RepairScope,
        robot_id: str,
    ) -> None:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            robot_id=robot_id,
            permission="repair:analyze",
        )

    # --------------------------------------------------------
    # Visual Damage Analysis
    # --------------------------------------------------------

    async def analyze_visual_damage(
        self,
        *,
        scope: RepairScope,
        request: RepairAnalysisRequest,
    ) -> list[VisualDamageFinding]:

        if not request.component_images:
            return []

        if self.visual_provider is None:
            raise RepairServiceUnavailableError(
                "Visual damage model is unavailable"
            )

        findings = (
            await self.visual_provider.analyze_images(
                tenant_id=scope.tenant_id,
                robot_id=request.robot_id,
                images=request.component_images,
            )
        )

        known_images = {
            image.image_id
            for image in request.component_images
        }

        for finding in findings:
            if finding.source_image_id not in known_images:
                raise RepairValidationError(
                    "Visual finding references unknown image"
                )

        return findings

    # --------------------------------------------------------
    # Spare Part Compatibility
    # --------------------------------------------------------

    async def assess_parts(
        self,
        *,
        scope: RepairScope,
        robot: RepairRobot,
        parts: Sequence[RobotSparePart],
    ) -> list[PartCompatibilityAssessment]:

        assessments: list[
            PartCompatibilityAssessment
        ] = []

        for part in parts:
            if self.compatibility_provider is not None:
                assessment = (
                    await self.compatibility_provider
                    .assess_compatibility(
                        tenant_id=scope.tenant_id,
                        robot=robot,
                        part=part,
                    )
                )

                if assessment.part_id != part.part_id:
                    raise RepairValidationError(
                        "Compatibility result part mismatch"
                    )

                assessments.append(assessment)
                continue

            if robot.model in part.compatible_robot_models:
                assessments.append(
                    PartCompatibilityAssessment(
                        part_id=part.part_id,
                        status=(
                            PartCompatibilityStatus.UNKNOWN
                        ),
                        confidence=0.50,
                        reasons=(
                            "Robot model appears in the "
                            "part compatibility list.",
                            "OEM part-number, revision and "
                            "serial-range verification required.",
                        ),
                    )
                )

            else:
                assessments.append(
                    PartCompatibilityAssessment(
                        part_id=part.part_id,
                        status=(
                            PartCompatibilityStatus.UNKNOWN
                        ),
                        confidence=0.0,
                        reasons=(
                            "Compatibility could not be verified.",
                        ),
                    )
                )

        return assessments

    # --------------------------------------------------------
    # Safety Assessment
    # --------------------------------------------------------

    @staticmethod
    def assess_repair_safety(
        *,
        robot: RepairRobot,
        faults: Sequence[RobotFault],
        visual_findings: Sequence[VisualDamageFinding],
        part_assessments: Sequence[
            PartCompatibilityAssessment
        ],
    ) -> RepairSafetyAssessment:

        findings: list[str] = []

        critical = any(
            fault.severity == FaultSeverity.CRITICAL
            for fault in faults
        )

        critical_visual = any(
            finding.severity == FaultSeverity.CRITICAL
            for finding in visual_findings
        )

        incompatible_parts = any(
            assessment.status
            == PartCompatibilityStatus.INCOMPATIBLE
            for assessment in part_assessments
        )

        unverified_parts = any(
            assessment.status
            == PartCompatibilityStatus.UNKNOWN
            for assessment in part_assessments
        )

        safety_component_fault = any(
            fault.component_type in (
                RobotComponentType.SAFETY_CONTROLLER,
                RobotComponentType.EMERGENCY_STOP,
            )
            for fault in faults
        )

        if robot.emergency_stop_active:
            findings.append(
                "Emergency stop is active. "
                "Do not bypass or reset it automatically."
            )

        if robot.safety_fault_active:
            findings.append(
                "Robot reports an active safety fault."
            )

        if critical or critical_visual:
            findings.append(
                "Critical fault or damage detected."
            )

        if safety_component_fault:
            findings.append(
                "Safety-related component requires "
                "qualified specialist inspection."
            )

        if incompatible_parts:
            findings.append(
                "At least one candidate part is incompatible."
            )

        if unverified_parts:
            findings.append(
                "One or more replacement parts "
                "require compatibility verification."
            )

        if robot.operational:
            findings.append(
                "Robot is currently operational; "
                "approved maintenance isolation "
                "is required before repair."
            )

        if (
            critical
            or critical_visual
            or safety_component_fault
            or robot.safety_fault_active
            or incompatible_parts
        ):
            risk = RepairRiskLevel.CRITICAL

        elif (
            robot.emergency_stop_active
            or unverified_parts
            or any(
                fault.severity == FaultSeverity.HIGH
                for fault in faults
            )
        ):
            risk = RepairRiskLevel.HIGH

        elif faults or visual_findings:
            risk = RepairRiskLevel.MODERATE

        else:
            risk = RepairRiskLevel.UNKNOWN
            findings.append(
                "Insufficient evidence to verify repair safety."
            )

        return RepairSafetyAssessment(
            risk_level=risk,
            safe_for_automatic_execution=False,
            requires_qualified_technician=True,
            requires_human_review=True,
            findings=tuple(findings),
        )

    # --------------------------------------------------------
    # Repair Workflow Generation
    # --------------------------------------------------------

    @staticmethod
    def generate_repair_steps(
        *,
        faults: Sequence[RobotFault],
        safety: RepairSafetyAssessment,
    ) -> list[RepairStep]:

        steps: list[RepairStep] = [
            RepairStep(
                step_id=generate_id(),
                step_type=RepairStepType.HUMAN_REVIEW,
                title="Qualified technician review",
                description=(
                    "Review robot faults, safety state, "
                    "OEM manuals, and proposed repair scope."
                ),
                estimated_duration_minutes=20,
                requires_human_approval=True,
            ),
            RepairStep(
                step_id=generate_id(),
                step_type=RepairStepType.SAFETY_ISOLATION,
                title="Verify safe maintenance conditions",
                description=(
                    "A qualified technician must apply "
                    "the manufacturer's approved energy "
                    "isolation and maintenance procedures."
                ),
                estimated_duration_minutes=20,
                requires_human_approval=True,
            ),
        ]

        for fault in faults:
            steps.append(
                RepairStep(
                    step_id=generate_id(),
                    step_type=RepairStepType.INSPECTION,
                    title=(
                        f"Inspect component {fault.component_id}"
                    ),
                    description=(
                        f"Investigate reported "
                        f"{fault.fault_type.value} fault. "
                        "Use approved diagnostic procedures "
                        "and verify the root cause."
                    ),
                    estimated_duration_minutes=30,
                )
            )

            steps.append(
                RepairStep(
                    step_id=generate_id(),
                    step_type=RepairStepType.DIAGNOSTICS,
                    title=(
                        f"Validate fault {fault.fault_id}"
                    ),
                    description=(
                        "Confirm fault evidence and determine "
                        "whether repair or replacement is "
                        "appropriate. Do not perform "
                        "unapproved physical actions."
                    ),
                    estimated_duration_minutes=30,
                )
            )

        steps.extend(
            [
                RepairStep(
                    step_id=generate_id(),
                    step_type=RepairStepType.FUNCTIONAL_TEST,
                    title="Controlled functional validation",
                    description=(
                        "Perform manufacturer-approved "
                        "post-repair tests under qualified "
                        "technician supervision."
                    ),
                    estimated_duration_minutes=30,
                    requires_human_approval=True,
                ),
                RepairStep(
                    step_id=generate_id(),
                    step_type=RepairStepType.SAFETY_TEST,
                    title="Independent safety verification",
                    description=(
                        "Verify safety functions, interlocks, "
                        "fault handling, and applicable "
                        "acceptance criteria before any "
                        "return-to-service decision."
                    ),
                    estimated_duration_minutes=45,
                    requires_human_approval=True,
                ),
                RepairStep(
                    step_id=generate_id(),
                    step_type=RepairStepType.DOCUMENTATION,
                    title="Record maintenance evidence",
                    description=(
                        "Record findings, parts, labor, "
                        "test results, approvals, and "
                        "technician sign-off."
                    ),
                    estimated_duration_minutes=15,
                ),
            ]
        )

        if len(steps) > MAX_REPAIR_STEPS:
            raise RepairValidationError(
                "Repair workflow exceeds step limit"
            )

        return steps

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    def generate_recommendations(
        self,
        *,
        faults: Sequence[RobotFault],
        cost_estimate: RepairCostEstimate,
        replacement_cost: Decimal | None,
        part_assessments: Sequence[
            PartCompatibilityAssessment
        ],
    ) -> list[RepairRecommendation]:

        recommendations: list[
            RepairRecommendation
        ] = []

        compatible_parts_available = any(
            assessment.status
            == PartCompatibilityStatus.COMPATIBLE
            for assessment in part_assessments
        )

        for fault in faults:
            decision = (
                self.decision_engine.recommend_decision(
                    fault=fault,
                    repair_cost=cost_estimate.total_cost,
                    replacement_cost=replacement_cost,
                    compatible_replacement_available=(
                        compatible_parts_available
                    ),
                )
            )

            priority = (
                self.decision_engine.determine_priority(
                    fault
                )
            )

            recommendations.append(
                RepairRecommendation(
                    recommendation_id=generate_id(),
                    component_id=fault.component_id,
                    decision=decision,
                    priority=priority,
                    title=(
                        f"{decision.value.title()} assessment "
                        f"for {fault.component_id}"
                    ),
                    explanation=(
                        f"Fault type: {fault.fault_type.value}. "
                        f"Severity: {fault.severity.value}. "
                        "Decision is preliminary and requires "
                        "technician verification."
                    ),
                    estimated_repair_cost=(
                        cost_estimate.total_cost
                    ),
                    estimated_replacement_cost=(
                        replacement_cost
                    ),
                    requires_human_review=True,
                    confidence=min(
                        fault.confidence,
                        0.85,
                    ),
                )
            )

        return recommendations[:MAX_RECOMMENDATIONS]

    # --------------------------------------------------------
    # Main Repair Analysis
    # --------------------------------------------------------

    async def analyze_repair(
        self,
        *,
        scope: RepairScope,
        request: RepairAnalysisRequest,
    ) -> RepairAnalysisResult:

        if request.tenant_id != scope.tenant_id:
            raise RepairPermissionError(
                "Tenant mismatch"
            )

        await self._authorize(
            scope=scope,
            robot_id=request.robot_id,
        )

        robot = await self.robot_provider.get_robot(
            tenant_id=scope.tenant_id,
            robot_id=request.robot_id,
        )

        if robot is None:
            raise RepairNotFoundError(
                "Robot not found"
            )

        if (
            robot.tenant_id != scope.tenant_id
            or robot.robot_id != request.robot_id
        ):
            raise RepairPermissionError(
                "Robot scope mismatch"
            )

        reasoning: list[str] = []

        visual_findings: list[
            VisualDamageFinding
        ] = []

        if (
            request.include_visual_analysis
            and request.component_images
        ):
            visual_findings = (
                await self.analyze_visual_damage(
                    scope=scope,
                    request=request,
                )
            )

            reasoning.append(
                f"Visual analysis identified "
                f"{len(visual_findings)} findings."
            )

        part_assessments: list[
            PartCompatibilityAssessment
        ] = []

        if request.include_part_compatibility:
            part_assessments = await self.assess_parts(
                scope=scope,
                robot=robot,
                parts=request.candidate_parts,
            )

        verified_part_ids = {
            assessment.part_id
            for assessment in part_assessments
            if assessment.status
            == PartCompatibilityStatus.COMPATIBLE
        }

        verified_parts = [
            part
            for part in request.candidate_parts
            if part.part_id in verified_part_ids
        ]

        cost_estimate = (
            self.cost_engine.calculate_cost(
                parts=verified_parts,
                labor_hours=request.estimated_labor_hours,
                downtime_hours=(
                    request.estimated_downtime_hours
                ),
                configuration=request.cost_configuration,
            )
        )

        if len(verified_parts) != len(
            request.candidate_parts
        ):
            reasoning.append(
                "Unverified candidate parts were excluded "
                "from the estimated parts subtotal."
            )

        safety = self.assess_repair_safety(
            robot=robot,
            faults=request.faults,
            visual_findings=visual_findings,
            part_assessments=part_assessments,
        )

        recommendations = (
            self.generate_recommendations(
                faults=request.faults,
                cost_estimate=cost_estimate,
                replacement_cost=request.replacement_cost,
                part_assessments=part_assessments,
            )
        )

        repair_steps = self.generate_repair_steps(
            faults=request.faults,
            safety=safety,
        )

        if (
            safety.risk_level
            == RepairRiskLevel.CRITICAL
        ):
            status = RepairStatus.BLOCKED
        else:
            status = RepairStatus.REVIEW_REQUIRED

        reasoning.extend(
            [
                f"Analyzed {len(request.faults)} faults.",
                (
                    "Estimated repair-related cost: "
                    f"{cost_estimate.total_cost} "
                    f"{cost_estimate.currency.value}."
                ),
                (
                    "Repair safety risk: "
                    f"{safety.risk_level.value}."
                ),
                "All physical repair work requires "
                "qualified human supervision.",
            ]
        )

        reasoning.extend(safety.findings)

        result = RepairAnalysisResult(
            analysis_id=generate_id(),
            tenant_id=scope.tenant_id,
            robot_id=request.robot_id,
            status=status,
            faults=copy.deepcopy(request.faults),
            visual_findings=visual_findings,
            part_assessments=part_assessments,
            cost_estimate=cost_estimate,
            recommendations=recommendations,
            repair_steps=repair_steps,
            safety_assessment=safety,
            reasoning=reasoning,
            requires_human_review=True,
            simulation_only=True,
        )

        if self.repository is not None:
            await self.repository.save_analysis(
                analysis=result
            )

        if self.audit_sink is not None:
            await self.audit_sink.record(
                tenant_id=scope.tenant_id,
                actor_id=scope.user_id,
                action="repair_agent.analysis_created",
                resource_id=result.analysis_id,
                metadata={
                    "robot_id": request.robot_id,
                    "fault_count": len(request.faults),
                    "status": result.status.value,
                    "risk_level": (
                        safety.risk_level.value
                    ),
                    "simulation_only": True,
                },
            )

        return result

    # --------------------------------------------------------
    # Batch Repair Analysis
    # --------------------------------------------------------

    async def analyze_batch(
        self,
        *,
        scope: RepairScope,
        requests: Sequence[RepairAnalysisRequest],
    ) -> list[RepairAnalysisResult]:

        if len(requests) > MAX_BATCH_REQUESTS:
            raise RepairValidationError(
                "Batch request limit exceeded"
            )

        semaphore = asyncio.Semaphore(4)

        async def process(
            request: RepairAnalysisRequest,
        ) -> RepairAnalysisResult:

            async with semaphore:
                return await self.analyze_repair(
                    scope=scope,
                    request=request,
                )

        return list(
            await asyncio.gather(
                *(process(request) for request in requests)
            )
        )

    # --------------------------------------------------------
    # Repair Report
    # --------------------------------------------------------

    @staticmethod
    def generate_repair_report(
        result: RepairAnalysisResult,
    ) -> dict[str, Any]:

        estimate = result.cost_estimate

        return {
            "analysis_id": result.analysis_id,
            "tenant_id": result.tenant_id,
            "robot_id": result.robot_id,
            "status": result.status.value,
            "generated_at": (
                result.generated_at.isoformat()
            ),
            "faults": [
                {
                    "fault_id": fault.fault_id,
                    "component_id": fault.component_id,
                    "component_type": (
                        fault.component_type.value
                    ),
                    "fault_type": fault.fault_type.value,
                    "severity": fault.severity.value,
                    "description": fault.description,
                    "confidence": fault.confidence,
                    "failure_probability": (
                        fault.failure_probability
                    ),
                    "remaining_useful_life_hours": (
                        fault.remaining_useful_life_hours
                    ),
                }
                for fault in result.faults
            ],
            "visual_findings": [
                {
                    "finding_id": finding.finding_id,
                    "component_id": finding.component_id,
                    "damage_type": finding.damage_type,
                    "description": finding.description,
                    "severity": finding.severity.value,
                    "confidence": finding.confidence,
                    "source_image_id": (
                        finding.source_image_id
                    ),
                }
                for finding in result.visual_findings
            ],
            "part_compatibility": [
                {
                    "part_id": assessment.part_id,
                    "status": assessment.status.value,
                    "confidence": assessment.confidence,
                    "reasons": list(assessment.reasons),
                    "requires_technician_verification": (
                        assessment
                        .requires_technician_verification
                    ),
                }
                for assessment in result.part_assessments
            ],
            "cost_estimate": {
                "currency": estimate.currency.value,
                "parts_cost": str(
                    estimate.parts_cost
                ),
                "labor_cost": str(
                    estimate.labor_cost
                ),
                "downtime_cost": str(
                    estimate.downtime_cost
                ),
                "contingency_cost": str(
                    estimate.contingency_cost
                ),
                "tax_cost": str(
                    estimate.tax_cost
                ),
                "total_cost": str(
                    estimate.total_cost
                ),
                "estimated_labor_hours": (
                    estimate.estimated_labor_hours
                ),
                "estimated_downtime_hours": (
                    estimate.estimated_downtime_hours
                ),
                "assumptions": list(
                    estimate.assumptions
                ),
            },
            "recommendations": [
                {
                    "recommendation_id": (
                        recommendation.recommendation_id
                    ),
                    "component_id": (
                        recommendation.component_id
                    ),
                    "decision": (
                        recommendation.decision.value
                    ),
                    "priority": (
                        recommendation.priority.value
                    ),
                    "title": recommendation.title,
                    "explanation": (
                        recommendation.explanation
                    ),
                    "confidence": (
                        recommendation.confidence
                    ),
                    "requires_human_review": True,
                }
                for recommendation
                in result.recommendations
            ],
            "repair_steps": [
                {
                    "step_id": step.step_id,
                    "step_type": step.step_type.value,
                    "title": step.title,
                    "description": step.description,
                    "estimated_duration_minutes": (
                        step.estimated_duration_minutes
                    ),
                    "requires_qualified_technician": (
                        step.requires_qualified_technician
                    ),
                    "requires_human_approval": (
                        step.requires_human_approval
                    ),
                }
                for step in result.repair_steps
            ],
            "safety": {
                "risk_level": (
                    result.safety_assessment.risk_level.value
                ),
                "safe_for_automatic_execution": False,
                "requires_qualified_technician": True,
                "requires_human_review": True,
                "findings": list(
                    result.safety_assessment.findings
                ),
            },
            "reasoning": result.reasoning,
            "simulation_only": True,
            "physical_repair_enabled": False,
            "automatic_actuation_enabled": False,
        }


# ============================================================
# Factory
# ============================================================

def create_repair_agent(
    *,
    robot_provider: RepairRobotProvider,
    authorizer: RepairAuthorizer,
    visual_provider: (
        VisualDamageAnalysisProvider | None
    ) = None,
    compatibility_provider: (
        PartCompatibilityProvider | None
    ) = None,
    knowledge_provider: (
        RepairKnowledgeProvider | None
    ) = None,
    repository: RepairRepository | None = None,
    audit_sink: RepairAuditSink | None = None,
    cost_engine: RepairCostEngine | None = None,
    decision_engine: RepairDecisionEngine | None = None,
) -> RepairAgent:

    return RepairAgent(
        robot_provider=robot_provider,
        authorizer=authorizer,
        visual_provider=visual_provider,
        compatibility_provider=compatibility_provider,
        knowledge_provider=knowledge_provider,
        repository=repository,
        audit_sink=audit_sink,
        cost_engine=cost_engine,
        decision_engine=decision_engine,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "RepairAgentError",
    "RepairValidationError",
    "RepairPermissionError",
    "RepairNotFoundError",
    "RepairModelError",
    "RepairCompatibilityError",
    "RepairSafetyError",
    "RepairServiceUnavailableError",
    "RobotComponentType",
    "FaultType",
    "FaultSeverity",
    "RepairPriority",
    "RepairDecision",
    "RepairStatus",
    "RepairRiskLevel",
    "PartCompatibilityStatus",
    "RepairStepType",
    "Currency",
    "RepairScope",
    "RepairRobot",
    "RobotFault",
    "ComponentImage",
    "VisualDamageFinding",
    "RobotSparePart",
    "PartCompatibilityAssessment",
    "RepairCostConfiguration",
    "RepairCostEstimate",
    "RepairStep",
    "RepairRecommendation",
    "RepairSafetyAssessment",
    "RepairAnalysisRequest",
    "RepairAnalysisResult",
    "RepairRobotProvider",
    "VisualDamageAnalysisProvider",
    "PartCompatibilityProvider",
    "RepairKnowledgeProvider",
    "RepairAuthorizer",
    "RepairRepository",
    "RepairAuditSink",
    "RepairCostEngine",
    "RepairDecisionEngine",
    "RepairAgent",
    "create_repair_agent",
]
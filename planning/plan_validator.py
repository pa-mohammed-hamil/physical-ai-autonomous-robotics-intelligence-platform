"""Plan Validator module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/planning/plan_validator.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: plan_validator.py

Production-oriented, simulation-only plan validation.

Features
--------
- Hierarchical plan structure validation
- Goal and task consistency
- Sequential dependency validation
- Robot capability verification
- Robot safety and telemetry freshness checks
- Human approval verification
- Task precondition and postcondition validation
- Simulated world-state transitions
- Resource and reservation validation
- Time, deadline, energy, and cost budgets
- Geofencing and workspace restrictions
- Motion and manipulation safety gates
- Risk assessment
- Constraint violation reporting
- Multi-tenant access control
- Deterministic validation
- Action scheduler integration payload
- Behavior tree integration payload
- Execution monitor integration payload
- Audit logging and JSON reports
- Unit tests and demonstration

Python 3.11+
Dependencies: Standard library only.

IMPORTANT SAFETY LIMITATIONS
----------------------------
This validator evaluates proposed plans. It does not execute
robot commands or certify that a robot is safe to operate.

Human approval records in this module are advisory and cannot
replace authenticated, independently enforced authorization.

A successful validation means that the supplied plan and
snapshots satisfy the configured software checks. It does not
establish physical collision freedom, safe stopping distance,
functional safety compliance, or permission to actuate hardware.
"""

from __future__ import annotations

import copy
import json
import math
import threading
import unittest
import uuid

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class PlanValidatorError(Exception):
    """Base plan validation exception."""


class PlanValidationInputError(PlanValidatorError):
    """Invalid input or malformed plan."""


class PlanAuthorizationError(PlanValidatorError):
    """Unauthorized validation operation."""


class PlanNotFoundError(PlanValidatorError):
    """Plan or validation report was not found."""


class PlanCapacityError(PlanValidatorError):
    """Validation capacity limit exceeded."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def validate_identifier(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise PlanValidationInputError(
            f"{field_name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise PlanValidationInputError(
            f"{field_name} must contain 1-128 characters"
        )

    return value


def validate_number(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise PlanValidationInputError(
            f"{field_name} must be a finite number"
        )

    number = float(value)

    if minimum is not None and number < minimum:
        raise PlanValidationInputError(
            f"{field_name} must be >= {minimum}"
        )

    if maximum is not None and number > maximum:
        raise PlanValidationInputError(
            f"{field_name} must be <= {maximum}"
        )

    return number


def normalize_timestamp(
    value: datetime,
    field_name: str,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise PlanValidationInputError(
            f"{field_name} must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return {
            name: json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list, set, frozenset, deque)):
        return [json_safe(item) for item in value]

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    return repr(value)


# ============================================================
# ENUMERATIONS
# ============================================================

class ValidationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    BLOCKED = "blocked"


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(str, Enum):
    STRUCTURE = "structure"
    DEPENDENCY = "dependency"
    ROBOT = "robot"
    SAFETY = "safety"
    APPROVAL = "approval"
    CAPABILITY = "capability"
    CONDITION = "condition"
    RESOURCE = "resource"
    ENERGY = "energy"
    TIME = "time"
    COST = "cost"
    GEOFENCE = "geofence"
    WORKSPACE = "workspace"
    RISK = "risk"
    WORLD_STATE = "world_state"


class TaskCategory(str, Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PICK = "pick"
    PLACE = "place"
    INSPECTION = "inspection"
    DIAGNOSTICS = "diagnostics"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    CALIBRATION = "calibration"
    CHARGING = "charging"
    PERCEPTION = "perception"
    MONITORING = "monitoring"
    DATA_COLLECTION = "data_collection"
    SAFETY_CHECK = "safety_check"
    SIMULATION = "simulation"
    CUSTOM = "custom"


class PlanRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RobotSafetyState(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    EMERGENCY_STOP = "emergency_stop"
    UNKNOWN = "unknown"


class ConditionOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    EXISTS = "exists"


class PlanExecutionMode(str, Enum):
    SIMULATION = "simulation"
    DRY_RUN = "dry_run"


class ValidationDecision(str, Enum):
    SIMULATION_ELIGIBLE = "simulation_eligible"
    REVIEW_REQUIRED = "review_required"
    REJECT_PLAN = "reject_plan"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PlanValidatorConfig:
    maximum_steps: int = 1000
    maximum_reports: int = 1000
    maximum_issues_per_report: int = 5000

    robot_state_max_age_seconds: float = 30.0

    minimum_battery_percentage: float = 15.0

    maximum_plan_duration_seconds: float = 86400.0

    maximum_plan_cost: float = 1_000_000.0

    maximum_plan_energy_wh: float = 100_000.0

    maximum_step_risk: PlanRiskLevel = (
        PlanRiskLevel.CRITICAL
    )

    require_safe_robot_state: bool = True
    require_fresh_robot_state: bool = True
    require_human_approval_for_high_risk: bool = True

    reject_unknown_capabilities: bool = True
    reject_unknown_resources: bool = True

    validate_preconditions: bool = True
    validate_postconditions: bool = True
    validate_geofencing: bool = True
    validate_workspace: bool = True

    allow_warnings: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_steps",
            "maximum_reports",
            "maximum_issues_per_report",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise PlanValidationInputError(
                    f"{name} must be a positive integer"
                )

        for name in (
            "robot_state_max_age_seconds",
            "maximum_plan_duration_seconds",
            "maximum_plan_cost",
            "maximum_plan_energy_wh",
        ):
            validate_number(
                getattr(self, name),
                name,
                minimum=0.001,
            )

        validate_number(
            self.minimum_battery_percentage,
            "minimum_battery_percentage",
            minimum=0,
            maximum=100,
        )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class PlanValidationAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    roles: frozenset[str] = frozenset()

    can_validate: bool = False
    can_view_reports: bool = False
    can_manage_reports: bool = False

    def __post_init__(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.user_id,
            "user_id",
        )


# ============================================================
# CONDITIONS
# ============================================================

@dataclass(frozen=True)
class PlanCondition:
    key: str
    operator: ConditionOperator
    expected: Any = None

    def __post_init__(self) -> None:
        validate_identifier(
            self.key,
            "condition key",
        )


class PlanConditionEvaluator:

    @staticmethod
    def evaluate(
        condition: PlanCondition,
        state: Mapping[str, Any],
    ) -> bool:
        exists = condition.key in state
        actual = state.get(condition.key)
        expected = condition.expected

        try:
            if condition.operator == ConditionOperator.EXISTS:
                return exists

            if not exists:
                return False

            if condition.operator == ConditionOperator.EQ:
                return actual == expected

            if condition.operator == ConditionOperator.NE:
                return actual != expected

            if condition.operator == ConditionOperator.GT:
                return actual > expected

            if condition.operator == ConditionOperator.GTE:
                return actual >= expected

            if condition.operator == ConditionOperator.LT:
                return actual < expected

            if condition.operator == ConditionOperator.LTE:
                return actual <= expected

            if condition.operator == ConditionOperator.IN:
                return actual in expected

            if condition.operator == ConditionOperator.NOT_IN:
                return actual not in expected

        except (TypeError, ValueError):
            return False

        return False


# ============================================================
# PLAN STEP
# ============================================================

@dataclass(frozen=True)
class ValidatablePlanStep:
    step_id: str
    task_id: str
    task_name: str

    robot_id: str

    category: TaskCategory

    sequence_index: int

    operator_id: str | None = None

    parent_task_id: str | None = None

    dependencies: tuple[str, ...] = ()

    required_capabilities: frozenset[str] = frozenset()

    required_resources: frozenset[str] = frozenset()

    required_zone: str | None = None

    required_workspace: str | None = None

    preconditions: tuple[PlanCondition, ...] = ()

    postconditions: tuple[PlanCondition, ...] = ()

    effects: dict[str, Any] = field(
        default_factory=dict
    )

    estimated_cost: float = 0.0

    estimated_duration_seconds: float = 0.0

    estimated_energy_wh: float = 0.0

    risk_level: PlanRiskLevel = PlanRiskLevel.LOW

    requires_human_approval: bool = False

    approval_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "step_id",
            "task_id",
            "task_name",
            "robot_id",
        ):
            validate_identifier(
                getattr(self, name),
                name,
            )

        if (
            isinstance(self.sequence_index, bool)
            or not isinstance(self.sequence_index, int)
            or self.sequence_index < 0
        ):
            raise PlanValidationInputError(
                "sequence_index must be nonnegative"
            )

        for name in (
            "estimated_cost",
            "estimated_duration_seconds",
            "estimated_energy_wh",
        ):
            validate_number(
                getattr(self, name),
                name,
                minimum=0,
            )

        if len(set(self.dependencies)) != len(
            self.dependencies
        ):
            raise PlanValidationInputError(
                "Duplicate step dependencies"
            )


# ============================================================
# PLAN SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class ValidatablePlan:
    plan_id: str
    tenant_id: str

    name: str

    root_goal_id: str

    steps: tuple[ValidatablePlanStep, ...]

    initial_state: dict[str, Any] = field(
        default_factory=dict
    )

    expected_final_state: dict[str, Any] = field(
        default_factory=dict
    )

    maximum_cost: float | None = None

    maximum_duration_seconds: float | None = None

    maximum_energy_wh: float | None = None

    deadline: datetime | None = None

    mode: PlanExecutionMode = (
        PlanExecutionMode.SIMULATION
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(self.plan_id, "plan_id")
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.name, "name")
        validate_identifier(
            self.root_goal_id,
            "root_goal_id",
        )

        if not self.steps:
            raise PlanValidationInputError(
                "Plan must contain at least one step"
            )

        for name in (
            "maximum_cost",
            "maximum_duration_seconds",
            "maximum_energy_wh",
        ):
            value = getattr(self, name)

            if value is not None:
                validate_number(
                    value,
                    name,
                    minimum=0,
                )

        normalize_timestamp(
            self.created_at,
            "created_at",
        )

        if self.deadline is not None:
            normalize_timestamp(
                self.deadline,
                "deadline",
            )


# ============================================================
# ROBOT SNAPSHOT
# ============================================================

@dataclass(frozen=True)
class ValidationRobotState:
    tenant_id: str
    robot_id: str

    capabilities: frozenset[str]

    battery_percentage: float

    safety_state: RobotSafetyState

    available: bool

    emergency_stop_active: bool = False

    active_faults: tuple[str, ...] = ()

    allowed_zones: frozenset[str] = frozenset()

    allowed_workspaces: frozenset[str] = frozenset()

    available_resources: frozenset[str] = frozenset()

    energy_budget_wh: float | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.robot_id, "robot_id")

        validate_number(
            self.battery_percentage,
            "battery_percentage",
            minimum=0,
            maximum=100,
        )

        if self.energy_budget_wh is not None:
            validate_number(
                self.energy_budget_wh,
                "energy_budget_wh",
                minimum=0,
            )

        normalize_timestamp(
            self.timestamp,
            "timestamp",
        )


# ============================================================
# HUMAN APPROVAL
# ============================================================

@dataclass(frozen=True)
class PlanApproval:
    approval_id: str
    tenant_id: str
    plan_id: str

    step_id: str

    approved_by: str

    approved_at: datetime

    expires_at: datetime

    revoked: bool = False

    def __post_init__(self) -> None:
        for name in (
            "approval_id",
            "tenant_id",
            "plan_id",
            "step_id",
            "approved_by",
        ):
            validate_identifier(
                getattr(self, name),
                name,
            )

        approved_at = normalize_timestamp(
            self.approved_at,
            "approved_at",
        )

        expires_at = normalize_timestamp(
            self.expires_at,
            "expires_at",
        )

        if expires_at <= approved_at:
            raise PlanValidationInputError(
                "Approval expiration must follow approval time"
            )

    def valid_at(self, timestamp: datetime) -> bool:
        timestamp = normalize_timestamp(
            timestamp,
            "timestamp",
        )

        return (
            not self.revoked
            and self.approved_at <= timestamp < self.expires_at
        )


# ============================================================
# VALIDATION REQUEST
# ============================================================

@dataclass(frozen=True)
class PlanValidationRequest:
    request_id: str

    plan: ValidatablePlan

    robot_states: tuple[ValidationRobotState, ...]

    approvals: tuple[PlanApproval, ...] = ()

    validation_time: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.request_id,
            "request_id",
        )

        normalize_timestamp(
            self.validation_time,
            "validation_time",
        )

        robot_ids = [
            robot.robot_id
            for robot in self.robot_states
        ]

        if len(robot_ids) != len(set(robot_ids)):
            raise PlanValidationInputError(
                "Duplicate robot snapshots"
            )


# ============================================================
# VALIDATION ISSUE
# ============================================================

@dataclass(frozen=True)
class PlanValidationIssue:
    issue_id: str

    category: ValidationCategory

    severity: ValidationSeverity

    code: str
    message: str

    step_id: str | None = None

    robot_id: str | None = None

    recommendation: str = ""

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# VALIDATION REPORT
# ============================================================

@dataclass(frozen=True)
class PlanValidationReport:
    report_id: str

    request_id: str
    plan_id: str
    tenant_id: str

    status: ValidationStatus

    decision: ValidationDecision

    valid: bool

    issues: tuple[PlanValidationIssue, ...]

    total_steps: int

    total_cost: float

    total_duration_seconds: float

    total_energy_wh: float

    maximum_risk: PlanRiskLevel

    simulated_final_state: dict[str, Any]

    validated_at: datetime

    simulation_only: bool = True

    hardware_execution_authorized: bool = False


# ============================================================
# RISK EVALUATOR
# ============================================================

class PlanRiskEvaluator:

    RISK_ORDER = {
        PlanRiskLevel.LOW: 0,
        PlanRiskLevel.MEDIUM: 1,
        PlanRiskLevel.HIGH: 2,
        PlanRiskLevel.CRITICAL: 3,
    }

    @classmethod
    def maximum_risk(
        cls,
        steps: Sequence[ValidatablePlanStep],
    ) -> PlanRiskLevel:
        return max(
            (
                step.risk_level
                for step in steps
            ),
            default=PlanRiskLevel.LOW,
            key=lambda risk: cls.RISK_ORDER[risk],
        )


# ============================================================
# MAIN VALIDATOR
# ============================================================

class RoboticsPlanValidator:
    """
    Thread-safe software plan validator.

    All validation decisions concern simulation eligibility.
    They never constitute authorization for real hardware.
    """

    MOTION_CATEGORIES = frozenset({
        TaskCategory.NAVIGATION,
        TaskCategory.MANIPULATION,
        TaskCategory.PICK,
        TaskCategory.PLACE,
        TaskCategory.REPAIR,
    })

    def __init__(
        self,
        config: PlanValidatorConfig | None = None,
    ) -> None:
        self.config = config or PlanValidatorConfig()

        self._reports: dict[
            str, PlanValidationReport
        ] = {}

        self._audit: deque[dict[str, Any]] = deque(
            maxlen=10000
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # ACCESS CONTROL
    # --------------------------------------------------------

    @staticmethod
    def _require(
        access: PlanValidationAccessContext,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise PlanAuthorizationError(
                f"Permission required: {permission}"
            )

    @staticmethod
    def _authorize_plan(
        plan: ValidatablePlan,
        access: PlanValidationAccessContext,
    ) -> None:
        if plan.tenant_id != access.tenant_id:
            raise PlanAuthorizationError(
                "Cross-tenant plan access denied"
            )

        for step in plan.steps:
            if step.robot_id not in access.allowed_robot_ids:
                raise PlanAuthorizationError(
                    f"Robot access denied: {step.robot_id}"
                )

    # --------------------------------------------------------
    # ISSUE CREATION
    # --------------------------------------------------------

    @staticmethod
    def _issue(
        issues: list[PlanValidationIssue],
        *,
        category: ValidationCategory,
        severity: ValidationSeverity,
        code: str,
        message: str,
        step_id: str | None = None,
        robot_id: str | None = None,
        recommendation: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        issues.append(
            PlanValidationIssue(
                issue_id=generate_id(),
                category=category,
                severity=severity,
                code=code,
                message=message,
                step_id=step_id,
                robot_id=robot_id,
                recommendation=recommendation,
                details=json_safe(details or {}),
            )
        )

    # --------------------------------------------------------
    # STRUCTURAL VALIDATION
    # --------------------------------------------------------

    def _validate_structure(
        self,
        plan: ValidatablePlan,
        issues: list[PlanValidationIssue],
    ) -> list[ValidatablePlanStep]:
        steps = sorted(
            plan.steps,
            key=lambda step: step.sequence_index,
        )

        if len(steps) > self.config.maximum_steps:
            self._issue(
                issues,
                category=ValidationCategory.STRUCTURE,
                severity=ValidationSeverity.CRITICAL,
                code="TOO_MANY_STEPS",
                message="Plan exceeds maximum step count",
            )

        step_ids = [
            step.step_id
            for step in steps
        ]

        if len(step_ids) != len(set(step_ids)):
            self._issue(
                issues,
                category=ValidationCategory.STRUCTURE,
                severity=ValidationSeverity.CRITICAL,
                code="DUPLICATE_STEP_IDS",
                message="Plan contains duplicate step IDs",
            )

        indices = [
            step.sequence_index
            for step in steps
        ]

        if indices != list(range(len(steps))):
            self._issue(
                issues,
                category=ValidationCategory.STRUCTURE,
                severity=ValidationSeverity.ERROR,
                code="INVALID_SEQUENCE",
                message=(
                    "Step sequence indices must be "
                    "contiguous starting at zero"
                ),
            )

        known_ids = set(step_ids)

        positions = {
            step.step_id: step.sequence_index
            for step in steps
        }

        for step in steps:
            for dependency in step.dependencies:
                if dependency not in known_ids:
                    self._issue(
                        issues,
                        category=ValidationCategory.DEPENDENCY,
                        severity=ValidationSeverity.ERROR,
                        code="UNKNOWN_DEPENDENCY",
                        message=(
                            f"Unknown dependency: {dependency}"
                        ),
                        step_id=step.step_id,
                    )

                elif positions[dependency] >= step.sequence_index:
                    self._issue(
                        issues,
                        category=ValidationCategory.DEPENDENCY,
                        severity=ValidationSeverity.ERROR,
                        code="DEPENDENCY_ORDER_VIOLATION",
                        message=(
                            "Dependency must precede dependent step"
                        ),
                        step_id=step.step_id,
                    )

        return steps

    # --------------------------------------------------------
    # ROBOT VALIDATION
    # --------------------------------------------------------

    def _validate_robot(
        self,
        step: ValidatablePlanStep,
        robot: ValidationRobotState | None,
        request: PlanValidationRequest,
        issues: list[PlanValidationIssue],
    ) -> None:
        if robot is None:
            self._issue(
                issues,
                category=ValidationCategory.ROBOT,
                severity=ValidationSeverity.CRITICAL,
                code="ROBOT_STATE_MISSING",
                message="Robot state snapshot unavailable",
                step_id=step.step_id,
                robot_id=step.robot_id,
                recommendation=(
                    "Provide a fresh authenticated robot snapshot"
                ),
            )
            return

        if robot.tenant_id != request.plan.tenant_id:
            self._issue(
                issues,
                category=ValidationCategory.ROBOT,
                severity=ValidationSeverity.CRITICAL,
                code="ROBOT_TENANT_MISMATCH",
                message="Robot tenant does not match plan",
                step_id=step.step_id,
                robot_id=step.robot_id,
            )
            return

        if not robot.available:
            self._issue(
                issues,
                category=ValidationCategory.ROBOT,
                severity=ValidationSeverity.ERROR,
                code="ROBOT_UNAVAILABLE",
                message="Robot is unavailable",
                step_id=step.step_id,
                robot_id=step.robot_id,
            )

        if self.config.require_safe_robot_state:
            if robot.safety_state != RobotSafetyState.SAFE:
                self._issue(
                    issues,
                    category=ValidationCategory.SAFETY,
                    severity=ValidationSeverity.CRITICAL,
                    code="ROBOT_NOT_SAFE",
                    message="Robot safety state is not SAFE",
                    step_id=step.step_id,
                    robot_id=step.robot_id,
                )

        if robot.emergency_stop_active:
            self._issue(
                issues,
                category=ValidationCategory.SAFETY,
                severity=ValidationSeverity.CRITICAL,
                code="EMERGENCY_STOP_ACTIVE",
                message="Emergency stop is active",
                step_id=step.step_id,
                robot_id=step.robot_id,
            )

        if robot.active_faults:
            self._issue(
                issues,
                category=ValidationCategory.SAFETY,
                severity=ValidationSeverity.CRITICAL,
                code="ACTIVE_ROBOT_FAULTS",
                message="Robot reports active faults",
                step_id=step.step_id,
                robot_id=step.robot_id,
                details={
                    "faults": robot.active_faults,
                },
            )

        if self.config.require_fresh_robot_state:
            age = (
                normalize_timestamp(
                    request.validation_time,
                    "validation_time",
                )
                - normalize_timestamp(
                    robot.timestamp,
                    "robot timestamp",
                )
            ).total_seconds()

            if (
                age < 0
                or age > self.config.robot_state_max_age_seconds
            ):
                self._issue(
                    issues,
                    category=ValidationCategory.ROBOT,
                    severity=ValidationSeverity.CRITICAL,
                    code="STALE_ROBOT_STATE",
                    message="Robot state is stale or future-dated",
                    step_id=step.step_id,
                    robot_id=step.robot_id,
                    details={
                        "age_seconds": age,
                    },
                )

        if (
            robot.battery_percentage
            < self.config.minimum_battery_percentage
        ):
            self._issue(
                issues,
                category=ValidationCategory.ENERGY,
                severity=ValidationSeverity.CRITICAL,
                code="LOW_BATTERY",
                message="Robot battery below safety threshold",
                step_id=step.step_id,
                robot_id=step.robot_id,
            )

        if not step.required_capabilities.issubset(
            robot.capabilities
        ):
            missing = (
                step.required_capabilities
                - robot.capabilities
            )

            self._issue(
                issues,
                category=ValidationCategory.CAPABILITY,
                severity=ValidationSeverity.ERROR,
                code="MISSING_CAPABILITIES",
                message="Robot lacks required capabilities",
                step_id=step.step_id,
                robot_id=step.robot_id,
                details={
                    "missing": sorted(missing),
                },
            )

        if self.config.validate_geofencing:
            if (
                step.required_zone is not None
                and step.required_zone not in robot.allowed_zones
            ):
                self._issue(
                    issues,
                    category=ValidationCategory.GEOFENCE,
                    severity=ValidationSeverity.CRITICAL,
                    code="ZONE_NOT_AUTHORIZED",
                    message="Robot is not authorized for zone",
                    step_id=step.step_id,
                    robot_id=step.robot_id,
                )

        if self.config.validate_workspace:
            if (
                step.required_workspace is not None
                and step.required_workspace
                not in robot.allowed_workspaces
            ):
                self._issue(
                    issues,
                    category=ValidationCategory.WORKSPACE,
                    severity=ValidationSeverity.CRITICAL,
                    code="WORKSPACE_NOT_AUTHORIZED",
                    message="Workspace is not authorized",
                    step_id=step.step_id,
                    robot_id=step.robot_id,
                )

        missing_resources = (
            step.required_resources
            - robot.available_resources
        )

        if missing_resources:
            self._issue(
                issues,
                category=ValidationCategory.RESOURCE,
                severity=ValidationSeverity.ERROR,
                code="RESOURCES_UNAVAILABLE",
                message="Required resources unavailable",
                step_id=step.step_id,
                robot_id=step.robot_id,
                details={
                    "missing": sorted(missing_resources),
                },
            )

    # --------------------------------------------------------
    # APPROVAL VALIDATION
    # --------------------------------------------------------

    def _validate_approval(
        self,
        step: ValidatablePlanStep,
        request: PlanValidationRequest,
        issues: list[PlanValidationIssue],
    ) -> None:
        high_risk = step.risk_level in (
            PlanRiskLevel.HIGH,
            PlanRiskLevel.CRITICAL,
        )

        approval_required = (
            step.requires_human_approval
            or (
                high_risk
                and self.config.require_human_approval_for_high_risk
            )
        )

        if not approval_required:
            return

        if step.approval_id is None:
            self._issue(
                issues,
                category=ValidationCategory.APPROVAL,
                severity=ValidationSeverity.CRITICAL,
                code="APPROVAL_ID_MISSING",
                message="Required approval ID is missing",
                step_id=step.step_id,
                robot_id=step.robot_id,
            )
            return

        valid_approval = any(
            approval.approval_id == step.approval_id
            and approval.tenant_id == request.plan.tenant_id
            and approval.plan_id == request.plan.plan_id
            and approval.step_id == step.step_id
            and approval.valid_at(request.validation_time)
            for approval in request.approvals
        )

        if not valid_approval:
            self._issue(
                issues,
                category=ValidationCategory.APPROVAL,
                severity=ValidationSeverity.CRITICAL,
                code="APPROVAL_INVALID",
                message=(
                    "Human approval missing, expired, "
                    "revoked, or incorrectly scoped"
                ),
                step_id=step.step_id,
                robot_id=step.robot_id,
                recommendation=(
                    "Obtain a fresh, authenticated, "
                    "step-scoped approval"
                ),
            )

    # --------------------------------------------------------
    # WORLD STATE VALIDATION
    # --------------------------------------------------------

    def _validate_world_state(
        self,
        steps: Sequence[ValidatablePlanStep],
        plan: ValidatablePlan,
        issues: list[PlanValidationIssue],
    ) -> dict[str, Any]:
        state = copy.deepcopy(
            plan.initial_state
        )

        for step in steps:
            if self.config.validate_preconditions:
                for condition in step.preconditions:
                    if not PlanConditionEvaluator.evaluate(
                        condition,
                        state,
                    ):
                        self._issue(
                            issues,
                            category=ValidationCategory.CONDITION,
                            severity=ValidationSeverity.ERROR,
                            code="PRECONDITION_FAILED",
                            message=(
                                f"Precondition failed: "
                                f"{condition.key}"
                            ),
                            step_id=step.step_id,
                            robot_id=step.robot_id,
                        )

            for key, value in step.effects.items():
                state[key] = copy.deepcopy(value)

            if self.config.validate_postconditions:
                for condition in step.postconditions:
                    if not PlanConditionEvaluator.evaluate(
                        condition,
                        state,
                    ):
                        self._issue(
                            issues,
                            category=ValidationCategory.CONDITION,
                            severity=ValidationSeverity.ERROR,
                            code="POSTCONDITION_FAILED",
                            message=(
                                f"Postcondition failed: "
                                f"{condition.key}"
                            ),
                            step_id=step.step_id,
                            robot_id=step.robot_id,
                        )

        for key, expected in plan.expected_final_state.items():
            if key not in state or state[key] != expected:
                self._issue(
                    issues,
                    category=ValidationCategory.WORLD_STATE,
                    severity=ValidationSeverity.ERROR,
                    code="FINAL_STATE_MISMATCH",
                    message=(
                        f"Expected final state not reached: {key}"
                    ),
                    details={
                        "expected": expected,
                        "actual": state.get(key),
                    },
                )

        return state

    # --------------------------------------------------------
    # BUDGET VALIDATION
    # --------------------------------------------------------

    def _validate_budgets(
        self,
        plan: ValidatablePlan,
        robots: Mapping[str, ValidationRobotState],
        request: PlanValidationRequest,
        issues: list[PlanValidationIssue],
    ) -> tuple[float, float, float]:
        total_cost = sum(
            step.estimated_cost
            for step in plan.steps
        )

        total_duration = sum(
            step.estimated_duration_seconds
            for step in plan.steps
        )

        total_energy = sum(
            step.estimated_energy_wh
            for step in plan.steps
        )

        maximum_cost = min(
            self.config.maximum_plan_cost,
            (
                plan.maximum_cost
                if plan.maximum_cost is not None
                else self.config.maximum_plan_cost
            ),
        )

        maximum_duration = min(
            self.config.maximum_plan_duration_seconds,
            (
                plan.maximum_duration_seconds
                if plan.maximum_duration_seconds is not None
                else self.config.maximum_plan_duration_seconds
            ),
        )

        maximum_energy = min(
            self.config.maximum_plan_energy_wh,
            (
                plan.maximum_energy_wh
                if plan.maximum_energy_wh is not None
                else self.config.maximum_plan_energy_wh
            ),
        )

        if total_cost > maximum_cost:
            self._issue(
                issues,
                category=ValidationCategory.COST,
                severity=ValidationSeverity.ERROR,
                code="COST_BUDGET_EXCEEDED",
                message="Plan cost exceeds configured budget",
                details={
                    "estimated": total_cost,
                    "maximum": maximum_cost,
                },
            )

        if total_duration > maximum_duration:
            self._issue(
                issues,
                category=ValidationCategory.TIME,
                severity=ValidationSeverity.ERROR,
                code="DURATION_BUDGET_EXCEEDED",
                message="Plan duration exceeds budget",
                details={
                    "estimated": total_duration,
                    "maximum": maximum_duration,
                },
            )

        if total_energy > maximum_energy:
            self._issue(
                issues,
                category=ValidationCategory.ENERGY,
                severity=ValidationSeverity.ERROR,
                code="ENERGY_BUDGET_EXCEEDED",
                message="Plan energy exceeds budget",
                details={
                    "estimated": total_energy,
                    "maximum": maximum_energy,
                },
            )

        energy_by_robot = defaultdict(float)

        for step in plan.steps:
            energy_by_robot[step.robot_id] += (
                step.estimated_energy_wh
            )

        for robot_id, energy in energy_by_robot.items():
            robot = robots.get(robot_id)

            if (
                robot is not None
                and robot.energy_budget_wh is not None
                and energy > robot.energy_budget_wh
            ):
                self._issue(
                    issues,
                    category=ValidationCategory.ENERGY,
                    severity=ValidationSeverity.ERROR,
                    code="ROBOT_ENERGY_BUDGET_EXCEEDED",
                    message=(
                        f"Robot {robot_id} energy budget exceeded"
                    ),
                    robot_id=robot_id,
                    details={
                        "estimated": energy,
                        "maximum": robot.energy_budget_wh,
                    },
                )

        if plan.deadline is not None:
            deadline = normalize_timestamp(
                plan.deadline,
                "deadline",
            )

            finish_time = (
                normalize_timestamp(
                    request.validation_time,
                    "validation_time",
                )
                + timedelta(seconds=total_duration)
            )

            if finish_time > deadline:
                self._issue(
                    issues,
                    category=ValidationCategory.TIME,
                    severity=ValidationSeverity.ERROR,
                    code="DEADLINE_EXCEEDED",
                    message=(
                        "Estimated completion exceeds deadline"
                    ),
                    details={
                        "estimated_finish": (
                            finish_time.isoformat()
                        ),
                        "deadline": deadline.isoformat(),
                    },
                )

        return (
            total_cost,
            total_duration,
            total_energy,
        )

    # --------------------------------------------------------
    # VALIDATE PLAN
    # --------------------------------------------------------

    def validate(
        self,
        request: PlanValidationRequest,
        *,
        access: PlanValidationAccessContext,
    ) -> PlanValidationReport:
        self._require(
            access,
            "can_validate",
        )

        self._authorize_plan(
            request.plan,
            access,
        )

        plan = request.plan
        issues: list[PlanValidationIssue] = []

        robots = {
            robot.robot_id: robot
            for robot in request.robot_states
        }

        steps = self._validate_structure(
            plan,
            issues,
        )

        for step in steps:
            self._validate_robot(
                step,
                robots.get(step.robot_id),
                request,
                issues,
            )

            self._validate_approval(
                step,
                request,
                issues,
            )

            if (
                PlanRiskEvaluator.RISK_ORDER[step.risk_level]
                > PlanRiskEvaluator.RISK_ORDER[
                    self.config.maximum_step_risk
                ]
            ):
                self._issue(
                    issues,
                    category=ValidationCategory.RISK,
                    severity=ValidationSeverity.CRITICAL,
                    code="RISK_LIMIT_EXCEEDED",
                    message=(
                        "Step exceeds maximum permitted risk"
                    ),
                    step_id=step.step_id,
                    robot_id=step.robot_id,
                )

            if step.category in self.MOTION_CATEGORIES:
                self._issue(
                    issues,
                    category=ValidationCategory.SAFETY,
                    severity=ValidationSeverity.WARNING,
                    code="PHYSICAL_SAFETY_NOT_VERIFIED",
                    message=(
                        "Static plan validation cannot verify "
                        "collision avoidance, dynamic obstacles, "
                        "stopping distance, or physical safety"
                    ),
                    step_id=step.step_id,
                    robot_id=step.robot_id,
                    recommendation=(
                        "Perform independent runtime and "
                        "hardware-enforced safety verification"
                    ),
                )

        final_state = self._validate_world_state(
            steps,
            plan,
            issues,
        )

        (
            total_cost,
            total_duration,
            total_energy,
        ) = self._validate_budgets(
            plan,
            robots,
            request,
            issues,
        )

        if len(issues) > self.config.maximum_issues_per_report:
            issues = issues[
                :self.config.maximum_issues_per_report
            ]

            self._issue(
                issues,
                category=ValidationCategory.STRUCTURE,
                severity=ValidationSeverity.CRITICAL,
                code="ISSUE_LIMIT_EXCEEDED",
                message=(
                    "Validation issue limit exceeded; "
                    "report may be incomplete"
                ),
            )

        blocking = any(
            issue.severity in (
                ValidationSeverity.ERROR,
                ValidationSeverity.CRITICAL,
            )
            for issue in issues
        )

        warnings = any(
            issue.severity == ValidationSeverity.WARNING
            for issue in issues
        )

        if blocking:
            status = ValidationStatus.INVALID
            decision = ValidationDecision.REJECT_PLAN

        elif warnings and not self.config.allow_warnings:
            status = ValidationStatus.BLOCKED
            decision = ValidationDecision.REVIEW_REQUIRED

        else:
            status = ValidationStatus.VALID

            decision = (
                ValidationDecision.REVIEW_REQUIRED
                if warnings
                else ValidationDecision.SIMULATION_ELIGIBLE
            )

        report = PlanValidationReport(
            report_id=generate_id(),
            request_id=request.request_id,
            plan_id=plan.plan_id,
            tenant_id=plan.tenant_id,
            status=status,
            decision=decision,
            valid=(status == ValidationStatus.VALID),
            issues=tuple(issues),
            total_steps=len(steps),
            total_cost=total_cost,
            total_duration_seconds=total_duration,
            total_energy_wh=total_energy,
            maximum_risk=PlanRiskEvaluator.maximum_risk(
                steps
            ),
            simulated_final_state=final_state,
            validated_at=request.validation_time,
        )

        with self._lock:
            if len(self._reports) >= self.config.maximum_reports:
                raise PlanCapacityError(
                    "Validation report capacity exceeded"
                )

            self._reports[report.report_id] = report

            self._audit.append({
                "event_id": generate_id(),
                "actor": access.user_id,
                "tenant_id": access.tenant_id,
                "action": "plan_validated",
                "plan_id": plan.plan_id,
                "report_id": report.report_id,
                "status": status.value,
                "timestamp": utc_now().isoformat(),
            })

        return report

    # --------------------------------------------------------
    # GET REPORT
    # --------------------------------------------------------

    def get_report(
        self,
        report_id: str,
        *,
        access: PlanValidationAccessContext,
    ) -> PlanValidationReport:
        self._require(
            access,
            "can_view_reports",
        )

        with self._lock:
            report = self._reports.get(
                report_id
            )

        if report is None:
            raise PlanNotFoundError(
                "Validation report not found"
            )

        if report.tenant_id != access.tenant_id:
            raise PlanAuthorizationError(
                "Cross-tenant report access denied"
            )

        return report

    # --------------------------------------------------------
    # REPORT SUMMARY
    # --------------------------------------------------------

    @staticmethod
    def summarize(
        report: PlanValidationReport,
    ) -> dict[str, Any]:
        severities = Counter(
            issue.severity.value
            for issue in report.issues
        )

        categories = Counter(
            issue.category.value
            for issue in report.issues
        )

        return {
            "report_id": report.report_id,
            "plan_id": report.plan_id,
            "status": report.status.value,
            "decision": report.decision.value,
            "valid": report.valid,
            "total_steps": report.total_steps,
            "total_issues": len(report.issues),
            "issue_severities": dict(severities),
            "issue_categories": dict(categories),
            "total_cost": report.total_cost,
            "total_duration_seconds": (
                report.total_duration_seconds
            ),
            "total_energy_wh": report.total_energy_wh,
            "maximum_risk": report.maximum_risk.value,
            "simulation_only": True,
            "hardware_execution_authorized": False,
        }

    # --------------------------------------------------------
    # BEHAVIOR TREE INTEGRATION
    # --------------------------------------------------------

    @staticmethod
    def behavior_tree_payload(
        report: PlanValidationReport,
    ) -> dict[str, Any]:
        return {
            "plan_id": report.plan_id,
            "validation_report_id": report.report_id,
            "validation_status": report.status.value,
            "simulation_eligible": (
                report.decision
                == ValidationDecision.SIMULATION_ELIGIBLE
            ),
            "requires_review": (
                report.decision
                == ValidationDecision.REVIEW_REQUIRED
            ),
            "hardware_execution_authorized": False,
        }

    # --------------------------------------------------------
    # ACTION SCHEDULER INTEGRATION
    # --------------------------------------------------------

    @staticmethod
    def action_scheduler_payload(
        report: PlanValidationReport,
    ) -> dict[str, Any]:
        return {
            "plan_id": report.plan_id,
            "validation_report_id": report.report_id,
            "allow_simulation_scheduling": (
                report.decision
                == ValidationDecision.SIMULATION_ELIGIBLE
            ),
            "estimated_duration_seconds": (
                report.total_duration_seconds
            ),
            "estimated_energy_wh": (
                report.total_energy_wh
            ),
            "estimated_cost": report.total_cost,
            "simulation_only": True,
            "hardware_execution_authorized": False,
        }

    # --------------------------------------------------------
    # EXECUTION MONITOR INTEGRATION
    # --------------------------------------------------------

    @staticmethod
    def execution_monitor_payload(
        report: PlanValidationReport,
    ) -> dict[str, Any]:
        return {
            "plan_id": report.plan_id,
            "validation_report_id": report.report_id,
            "expected_steps": report.total_steps,
            "expected_duration_seconds": (
                report.total_duration_seconds
            ),
            "expected_energy_wh": report.total_energy_wh,
            "risk_level": report.maximum_risk.value,
            "validation_status": report.status.value,
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # JSON EXPORT
    # --------------------------------------------------------

    @staticmethod
    def export_json(
        report: PlanValidationReport,
    ) -> str:
        return json.dumps(
            json_safe(report),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestRoboticsPlanValidator(unittest.TestCase):

    def setUp(self) -> None:
        self.validator = RoboticsPlanValidator()

        self.access = PlanValidationAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_validate=True,
            can_view_reports=True,
            can_manage_reports=True,
        )

        self.robot = ValidationRobotState(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            capabilities=frozenset({
                "inspection",
                "diagnostics",
            }),
            battery_percentage=90,
            safety_state=RobotSafetyState.SAFE,
            available=True,
            allowed_zones=frozenset({
                "ZONE-A",
            }),
            allowed_workspaces=frozenset({
                "WORKSPACE-1",
            }),
            available_resources=frozenset({
                "camera",
            }),
            energy_budget_wh=500,
        )

        self.step = ValidatablePlanStep(
            step_id="STEP-001",
            task_id="TASK-001",
            task_name="inspect_robot",
            robot_id="ROBOT-001",
            category=TaskCategory.INSPECTION,
            sequence_index=0,
            required_capabilities=frozenset({
                "inspection",
            }),
            required_resources=frozenset({
                "camera",
            }),
            required_zone="ZONE-A",
            required_workspace="WORKSPACE-1",
            effects={
                "inspection_completed": True,
            },
            postconditions=(
                PlanCondition(
                    key="inspection_completed",
                    operator=ConditionOperator.EQ,
                    expected=True,
                ),
            ),
            estimated_cost=10,
            estimated_duration_seconds=60,
            estimated_energy_wh=20,
        )

        self.plan = ValidatablePlan(
            plan_id="PLAN-001",
            tenant_id="tenant_alpha",
            name="Robot Inspection Plan",
            root_goal_id="GOAL-001",
            steps=(self.step,),
            expected_final_state={
                "inspection_completed": True,
            },
        )

    def request(
        self,
        *,
        robot: ValidationRobotState | None = None,
    ) -> PlanValidationRequest:
        return PlanValidationRequest(
            request_id=generate_id(),
            plan=self.plan,
            robot_states=(
                robot or self.robot,
            ),
        )

    def test_valid_plan(self) -> None:
        report = self.validator.validate(
            self.request(),
            access=self.access,
        )

        self.assertTrue(report.valid)

        self.assertEqual(
            report.status,
            ValidationStatus.VALID,
        )

    def test_world_state_effect(self) -> None:
        report = self.validator.validate(
            self.request(),
            access=self.access,
        )

        self.assertTrue(
            report.simulated_final_state[
                "inspection_completed"
            ]
        )

    def test_unsafe_robot(self) -> None:
        unsafe = ValidationRobotState(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            capabilities=frozenset({
                "inspection",
            }),
            battery_percentage=90,
            safety_state=RobotSafetyState.UNSAFE,
            available=True,
            allowed_zones=frozenset({
                "ZONE-A",
            }),
            allowed_workspaces=frozenset({
                "WORKSPACE-1",
            }),
            available_resources=frozenset({
                "camera",
            }),
        )

        report = self.validator.validate(
            self.request(robot=unsafe),
            access=self.access,
        )

        self.assertFalse(report.valid)

    def test_low_battery(self) -> None:
        robot = ValidationRobotState(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            capabilities=frozenset({
                "inspection",
            }),
            battery_percentage=5,
            safety_state=RobotSafetyState.SAFE,
            available=True,
            allowed_zones=frozenset({
                "ZONE-A",
            }),
            allowed_workspaces=frozenset({
                "WORKSPACE-1",
            }),
            available_resources=frozenset({
                "camera",
            }),
        )

        report = self.validator.validate(
            self.request(robot=robot),
            access=self.access,
        )

        self.assertFalse(report.valid)

    def test_tenant_isolation(self) -> None:
        other_access = PlanValidationAccessContext(
            tenant_id="tenant_beta",
            user_id="other",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_validate=True,
        )

        with self.assertRaises(
            PlanAuthorizationError
        ):
            self.validator.validate(
                self.request(),
                access=other_access,
            )

    def test_budget_violation(self) -> None:
        expensive_plan = ValidatablePlan(
            plan_id="PLAN-EXPENSIVE",
            tenant_id="tenant_alpha",
            name="Expensive Plan",
            root_goal_id="GOAL-001",
            steps=(self.step,),
            maximum_cost=5,
        )

        request = PlanValidationRequest(
            request_id=generate_id(),
            plan=expensive_plan,
            robot_states=(self.robot,),
        )

        report = self.validator.validate(
            request,
            access=self.access,
        )

        self.assertFalse(report.valid)

    def test_json_export(self) -> None:
        report = self.validator.validate(
            self.request(),
            access=self.access,
        )

        exported = self.validator.export_json(
            report
        )

        self.assertIn(
            "PLAN-001",
            exported,
        )

        self.assertIn(
            "hardware_execution_authorized",
            exported,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Plan Validator — Simulation-Only Safety Checks\n"
    )

    validator = RoboticsPlanValidator()

    access = PlanValidationAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
        }),
        can_validate=True,
        can_view_reports=True,
        can_manage_reports=True,
    )

    robot = ValidationRobotState(
        tenant_id="tenant_alpha",
        robot_id="ROBOT-001",
        capabilities=frozenset({
            "inspection",
            "diagnostics",
        }),
        battery_percentage=95,
        safety_state=RobotSafetyState.SAFE,
        available=True,
        allowed_zones=frozenset({
            "ZONE-A",
        }),
        allowed_workspaces=frozenset({
            "WORKSPACE-1",
        }),
        available_resources=frozenset({
            "camera",
            "diagnostic_tool",
        }),
        energy_budget_wh=500,
    )

    step_1 = ValidatablePlanStep(
        step_id="STEP-001",
        task_id="TASK-INSPECT",
        task_name="inspect_robot",
        robot_id="ROBOT-001",
        category=TaskCategory.INSPECTION,
        sequence_index=0,
        required_capabilities=frozenset({
            "inspection",
        }),
        required_resources=frozenset({
            "camera",
        }),
        required_zone="ZONE-A",
        required_workspace="WORKSPACE-1",
        effects={
            "inspection_completed": True,
        },
        estimated_cost=10,
        estimated_duration_seconds=60,
        estimated_energy_wh=20,
    )

    step_2 = ValidatablePlanStep(
        step_id="STEP-002",
        task_id="TASK-DIAGNOSE",
        task_name="diagnose_robot",
        robot_id="ROBOT-001",
        category=TaskCategory.DIAGNOSTICS,
        sequence_index=1,
        dependencies=(
            "STEP-001",
        ),
        required_capabilities=frozenset({
            "diagnostics",
        }),
        required_resources=frozenset({
            "diagnostic_tool",
        }),
        preconditions=(
            PlanCondition(
                key="inspection_completed",
                operator=ConditionOperator.EQ,
                expected=True,
            ),
        ),
        effects={
            "diagnostics_completed": True,
        },
        estimated_cost=15,
        estimated_duration_seconds=90,
        estimated_energy_wh=30,
    )

    plan = ValidatablePlan(
        plan_id="PLAN-ROBOT-HEALTH",
        tenant_id="tenant_alpha",
        name="Robot Health Inspection Plan",
        root_goal_id="GOAL-ROBOT-HEALTH",
        steps=(
            step_1,
            step_2,
        ),
        expected_final_state={
            "inspection_completed": True,
            "diagnostics_completed": True,
        },
        maximum_cost=100,
        maximum_duration_seconds=600,
        maximum_energy_wh=200,
    )

    request = PlanValidationRequest(
        request_id=generate_id(),
        plan=plan,
        robot_states=(robot,),
    )

    report = validator.validate(
        request,
        access=access,
    )

    print("VALIDATION SUMMARY\n")

    print(
        json.dumps(
            validator.summarize(report),
            indent=2,
        )
    )

    print("\nFULL VALIDATION REPORT\n")

    print(
        validator.export_json(report)
    )

    print("\nACTION SCHEDULER INTEGRATION\n")

    print(
        json.dumps(
            validator.action_scheduler_payload(report),
            indent=2,
        )
    )

    print("\nEXECUTION MONITOR INTEGRATION\n")

    print(
        json.dumps(
            validator.execution_monitor_payload(report),
            indent=2,
        )
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestRoboticsPlanValidator
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
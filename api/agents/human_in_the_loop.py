"""Human In The Loop module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/human_in_the_loop.py

Physical AI & Autonomous Robotics Intelligence Platform
Human-in-the-Loop (HITL) Approval and Safety Review Engine

Python 3.11+

Features:
- Human approval and rejection workflows
- Safety-critical robot action reviews
- Multi-level approval policies
- Role-based reviewer eligibility
- Separation of duties
- Four-eyes / two-person approval
- AI confidence-based escalation
- Risk-based approval routing
- Approval expiration
- Review queues
- Reviewer comments and decisions
- Request cancellation
- Optimistic concurrency control
- Immutable decision history
- Tenant isolation
- Audit integration
- Emergency-stop acknowledgment workflows
- Maintenance and repair approvals
- Simulation-to-real deployment reviews
- Model deployment approvals
- Fleet task assignment approvals
- Robot command authorization decisions
- No direct physical robot actuation

IMPORTANT:
An approved HITL request is not permission to bypass robot
safety interlocks, controller authorization, emergency stops,
or current-state validation.

The actual command executor must independently validate:
- Tenant and resource authorization
- Approval binding to the exact action and parameters
- Approval expiration and single-use consumption
- Robot identity, state, and safety conditions
- Operator permissions
- Hardware safety interlocks
- Command idempotency

This module manages review decisions only.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
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

MAX_TITLE_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 10_000
MAX_COMMENT_LENGTH = 5_000
MAX_REVIEWERS = 20
MAX_METADATA_FIELDS = 50
MAX_QUEUE_LIMIT = 200

DEFAULT_APPROVAL_TTL_SECONDS = 3600
MAX_APPROVAL_TTL_SECONDS = 86_400

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


def validate_confidence(value: float) -> float:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(
            "Confidence must be finite and between 0 and 1"
        )
    return value


def validate_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Datetime must be timezone-aware")
    return value


def canonical_action_hash(
    action: Mapping[str, Any],
) -> str:
    """
    Bind approval to the exact canonical action payload.

    Action payloads must be JSON-compatible and contain
    no secrets. The execution gateway must recompute
    this hash before using an approval.
    """
    try:
        payload = json.dumps(
            action,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Action payload must be valid JSON data"
        ) from exc

    return hashlib.sha256(payload).hexdigest()


# ============================================================
# Enumerations
# ============================================================

class ApprovalType(str, Enum):
    ROBOT_COMMAND = "robot_command"
    ROBOT_TASK = "robot_task"
    ROBOT_RECOVERY = "robot_recovery"
    FLEET_ASSIGNMENT = "fleet_assignment"
    NAVIGATION_PLAN = "navigation_plan"
    MANIPULATION_PLAN = "manipulation_plan"
    VLA_ACTION = "vla_action"
    WORLD_MODEL_ACTION = "world_model_action"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    PART_REPLACEMENT = "part_replacement"
    SAFETY_EXCEPTION = "safety_exception"
    EMERGENCY_RESPONSE = "emergency_response"
    EMERGENCY_STOP_RESET = "emergency_stop_reset"
    MODEL_DEPLOYMENT = "model_deployment"
    POLICY_DEPLOYMENT = "policy_deployment"
    SIM_TO_REAL = "sim_to_real"
    TRAINING = "training"
    DATA_ACCESS = "data_access"
    COST_APPROVAL = "cost_approval"
    CUSTOM = "custom"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    ESCALATED = "escalated"


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


class ReviewerRole(str, Enum):
    ROBOT_OPERATOR = "robot_operator"
    ROBOTICS_ENGINEER = "robotics_engineer"
    SAFETY_OFFICER = "safety_officer"
    MAINTENANCE_ENGINEER = "maintenance_engineer"
    ML_ENGINEER = "ml_engineer"
    MLOPS_ENGINEER = "mlops_engineer"
    FLEET_MANAGER = "fleet_manager"
    ORGANIZATION_ADMIN = "organization_admin"
    SYSTEM_ADMIN = "system_admin"


class ApprovalPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ApprovalEventType(str, Enum):
    CREATED = "created"
    REVIEW_STARTED = "review_started"
    APPROVED_BY_REVIEWER = "approved_by_reviewer"
    REJECTED_BY_REVIEWER = "rejected_by_reviewer"
    CHANGES_REQUESTED = "changes_requested"
    ESCALATED = "escalated"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


# ============================================================
# Exceptions
# ============================================================

class HITLError(Exception):
    pass


class ApprovalNotFoundError(HITLError):
    pass


class ApprovalPermissionError(HITLError):
    pass


class ApprovalValidationError(HITLError):
    pass


class ApprovalConflictError(HITLError):
    pass


class ApprovalExpiredError(HITLError):
    pass


class ApprovalBackendError(HITLError):
    pass


# ============================================================
# Security Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class HITLScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Approval Policy
# ============================================================

@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    policy_id: str

    approval_type: ApprovalType
    risk_level: RiskLevel

    required_approvals: int = 1

    allowed_reviewer_roles: tuple[ReviewerRole, ...] = (
        ReviewerRole.SAFETY_OFFICER,
        ReviewerRole.ROBOTICS_ENGINEER,
    )

    require_distinct_reviewers: bool = True
    prohibit_self_approval: bool = True

    require_safety_officer: bool = False

    expiration_seconds: int = (
        DEFAULT_APPROVAL_TTL_SECONDS
    )

    auto_approval_allowed: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.policy_id)

        if not 1 <= self.required_approvals <= MAX_REVIEWERS:
            raise ApprovalValidationError(
                "Invalid required approval count"
            )

        if not self.allowed_reviewer_roles:
            raise ApprovalValidationError(
                "At least one reviewer role is required"
            )

        if not (
            1 <= self.expiration_seconds
            <= MAX_APPROVAL_TTL_SECONDS
        ):
            raise ApprovalValidationError(
                "Invalid approval expiration"
            )

        if self.require_safety_officer and (
            ReviewerRole.SAFETY_OFFICER
            not in self.allowed_reviewer_roles
        ):
            raise ApprovalValidationError(
                "Safety officer must be an allowed reviewer"
            )

        if self.auto_approval_allowed and (
            self.risk_level != RiskLevel.LOW
            or self.approval_type in SAFETY_CRITICAL_TYPES
        ):
            raise ApprovalValidationError(
                "Automatic approval is prohibited "
                "for safety-sensitive actions"
            )


SAFETY_CRITICAL_TYPES = frozenset({
    ApprovalType.ROBOT_COMMAND,
    ApprovalType.ROBOT_RECOVERY,
    ApprovalType.MANIPULATION_PLAN,
    ApprovalType.VLA_ACTION,
    ApprovalType.SAFETY_EXCEPTION,
    ApprovalType.EMERGENCY_RESPONSE,
    ApprovalType.EMERGENCY_STOP_RESET,
    ApprovalType.POLICY_DEPLOYMENT,
    ApprovalType.SIM_TO_REAL,
})


# ============================================================
# Request Models
# ============================================================

@dataclass(slots=True)
class ApprovalCreateRequest:
    title: str
    description: str

    approval_type: ApprovalType
    risk_level: RiskLevel

    resource_type: str
    resource_id: str

    proposed_action: dict[str, Any]

    priority: ApprovalPriority = ApprovalPriority.NORMAL

    agent_id: str | None = None
    robot_id: str | None = None
    task_id: str | None = None

    ai_confidence: float | None = None

    justification: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ApprovalValidationError(
                "Approval title is required"
            )

        if len(self.title) > MAX_TITLE_LENGTH:
            raise ApprovalValidationError(
                "Approval title is too long"
            )

        if len(self.description) > MAX_DESCRIPTION_LENGTH:
            raise ApprovalValidationError(
                "Approval description is too long"
            )

        validate_identifier(self.resource_type)
        validate_identifier(self.resource_id)

        for identifier in (
            self.agent_id,
            self.robot_id,
            self.task_id,
        ):
            if identifier is not None:
                validate_identifier(identifier)

        if not self.proposed_action:
            raise ApprovalValidationError(
                "Proposed action cannot be empty"
            )

        canonical_action_hash(self.proposed_action)

        if self.ai_confidence is not None:
            validate_confidence(self.ai_confidence)

        if len(self.metadata) > MAX_METADATA_FIELDS:
            raise ApprovalValidationError(
                "Too many metadata fields"
            )


@dataclass(slots=True)
class ApprovalReviewRequest:
    decision: ReviewDecision

    comment: str = ""

    expected_version: int | None = None

    def __post_init__(self) -> None:
        if len(self.comment) > MAX_COMMENT_LENGTH:
            raise ApprovalValidationError(
                "Review comment is too long"
            )

        if (
            self.decision != ReviewDecision.APPROVE
            and not self.comment.strip()
        ):
            raise ApprovalValidationError(
                "A comment is required for rejection "
                "or requested changes"
            )

        if (
            self.expected_version is not None
            and self.expected_version < 1
        ):
            raise ApprovalValidationError(
                "Invalid expected version"
            )


@dataclass(frozen=True, slots=True)
class ReviewerIdentity:
    tenant_id: str
    user_id: str

    roles: frozenset[ReviewerRole]

    active: bool = True

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Review and Audit Models
# ============================================================

@dataclass(frozen=True, slots=True)
class ApprovalReview:
    review_id: str
    reviewer_id: str

    decision: ReviewDecision
    comment: str

    reviewer_roles: tuple[ReviewerRole, ...]

    reviewed_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True, slots=True)
class ApprovalEvent:
    event_id: str

    approval_id: str
    event_type: ApprovalEventType

    actor_id: str

    description: str

    created_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(slots=True)
class ApprovalRecord:
    approval_id: str
    tenant_id: str

    requested_by: str

    title: str
    description: str

    approval_type: ApprovalType
    risk_level: RiskLevel

    resource_type: str
    resource_id: str

    proposed_action: dict[str, Any]
    action_hash: str

    policy_id: str
    required_approvals: int

    allowed_reviewer_roles: tuple[ReviewerRole, ...]

    require_distinct_reviewers: bool
    prohibit_self_approval: bool
    require_safety_officer: bool

    priority: ApprovalPriority

    agent_id: str | None = None
    robot_id: str | None = None
    task_id: str | None = None

    ai_confidence: float | None = None

    justification: str = ""

    status: ApprovalStatus = ApprovalStatus.PENDING

    reviews: list[ApprovalReview] = field(
        default_factory=list
    )

    events: list[ApprovalEvent] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    expires_at: datetime = field(
        default_factory=lambda: (
            utc_now()
            + timedelta(
                seconds=DEFAULT_APPROVAL_TTL_SECONDS
            )
        )
    )

    decided_at: datetime | None = None

    version: int = 1

    def __post_init__(self) -> None:
        for identifier in (
            self.approval_id,
            self.tenant_id,
            self.requested_by,
            self.resource_type,
            self.resource_id,
            self.policy_id,
        ):
            validate_identifier(identifier)

        validate_aware_datetime(self.created_at)
        validate_aware_datetime(self.updated_at)
        validate_aware_datetime(self.expires_at)

        if self.expires_at <= self.created_at:
            raise ApprovalValidationError(
                "Expiration must follow creation"
            )

        if self.action_hash != canonical_action_hash(
            self.proposed_action
        ):
            raise ApprovalValidationError(
                "Action hash does not match proposed action"
            )

    @property
    def approval_count(self) -> int:
        return len({
            review.reviewer_id
            for review in self.reviews
            if review.decision == ReviewDecision.APPROVE
        })

    @property
    def remaining_approvals(self) -> int:
        return max(
            0,
            self.required_approvals - self.approval_count,
        )

    def is_expired(
        self,
        now: datetime | None = None,
    ) -> bool:
        return self.expires_at <= (now or utc_now())

    def has_safety_officer_approval(self) -> bool:
        return any(
            review.decision == ReviewDecision.APPROVE
            and ReviewerRole.SAFETY_OFFICER
            in review.reviewer_roles
            for review in self.reviews
        )


@dataclass(frozen=True, slots=True)
class ApprovalDecisionResult:
    approval_id: str

    status: ApprovalStatus

    required_approvals: int
    completed_approvals: int
    remaining_approvals: int

    requires_additional_review: bool

    action_hash: str
    expires_at: datetime

    version: int


@dataclass(frozen=True, slots=True)
class ApprovalStatistics:
    tenant_id: str

    total: int
    pending: int
    in_review: int
    approved: int
    rejected: int
    expired: int
    cancelled: int
    escalated: int

    generated_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# Repository and Integration Protocols
# ============================================================

class ApprovalRepository(Protocol):

    async def create(
        self,
        record: ApprovalRecord,
    ) -> ApprovalRecord:
        ...

    async def get(
        self,
        *,
        tenant_id: str,
        approval_id: str,
    ) -> ApprovalRecord | None:
        ...

    async def list(
        self,
        *,
        tenant_id: str,
    ) -> list[ApprovalRecord]:
        ...

    async def save(
        self,
        *,
        record: ApprovalRecord,
        expected_version: int,
    ) -> ApprovalRecord:
        ...


class HITLAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
        resource_type: str,
        resource_id: str,
    ) -> None:
        ...


class ReviewerIdentityProvider(Protocol):

    async def get_reviewer(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> ReviewerIdentity:
        ...


class HITLAuditSink(Protocol):

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


class ApprovalNotificationSink(Protocol):

    async def notify(
        self,
        *,
        tenant_id: str,
        approval_id: str,
        event_type: ApprovalEventType,
        message: str,
    ) -> None:
        ...


class ApprovalPolicyProvider(Protocol):

    async def resolve_policy(
        self,
        *,
        tenant_id: str,
        approval_type: ApprovalType,
        risk_level: RiskLevel,
    ) -> ApprovalPolicy:
        ...


# ============================================================
# In-Memory Repository
# ============================================================

class InMemoryApprovalRepository:
    """
    Development/testing implementation.

    Uses a lock for atomic create and compare-and-swap updates.
    Not durable or distributed.

    Production should use transactional database writes,
    version constraints, and a transactional audit outbox.
    """

    def __init__(self) -> None:
        self._records: dict[
            tuple[str, str],
            ApprovalRecord,
        ] = {}

        self._lock = asyncio.Lock()

    async def create(
        self,
        record: ApprovalRecord,
    ) -> ApprovalRecord:

        key = (
            record.tenant_id,
            record.approval_id,
        )

        async with self._lock:
            if key in self._records:
                raise ApprovalConflictError(
                    "Approval already exists"
                )

            self._records[key] = copy.deepcopy(record)

            return copy.deepcopy(record)

    async def get(
        self,
        *,
        tenant_id: str,
        approval_id: str,
    ) -> ApprovalRecord | None:

        key = (tenant_id, approval_id)

        async with self._lock:
            record = self._records.get(key)

            return (
                copy.deepcopy(record)
                if record is not None
                else None
            )

    async def list(
        self,
        *,
        tenant_id: str,
    ) -> list[ApprovalRecord]:

        async with self._lock:
            return [
                copy.deepcopy(record)
                for (record_tenant, _), record
                in self._records.items()
                if record_tenant == tenant_id
            ]

    async def save(
        self,
        *,
        record: ApprovalRecord,
        expected_version: int,
    ) -> ApprovalRecord:

        key = (
            record.tenant_id,
            record.approval_id,
        )

        async with self._lock:
            current = self._records.get(key)

            if current is None:
                raise ApprovalNotFoundError(
                    "Approval not found"
                )

            if current.version != expected_version:
                raise ApprovalConflictError(
                    "Approval was modified concurrently"
                )

            if record.version != expected_version + 1:
                raise ApprovalConflictError(
                    "Invalid next version"
                )

            self._records[key] = copy.deepcopy(record)

            return copy.deepcopy(record)


# ============================================================
# Default Approval Policies
# ============================================================

def default_approval_policy(
    *,
    approval_type: ApprovalType,
    risk_level: RiskLevel,
) -> ApprovalPolicy:

    critical = (
        risk_level == RiskLevel.CRITICAL
    )

    high = (
        risk_level == RiskLevel.HIGH
    )

    safety_sensitive = (
        approval_type in SAFETY_CRITICAL_TYPES
    )

    required = (
        2 if critical or safety_sensitive else 1
    )

    require_safety = (
        critical or safety_sensitive
    )

    roles = (
        ReviewerRole.SAFETY_OFFICER,
        ReviewerRole.ROBOTICS_ENGINEER,
        ReviewerRole.MAINTENANCE_ENGINEER,
        ReviewerRole.FLEET_MANAGER,
        ReviewerRole.ORGANIZATION_ADMIN,
    )

    return ApprovalPolicy(
        policy_id=(
            f"default:{approval_type.value}:{risk_level.value}"
        ),
        approval_type=approval_type,
        risk_level=risk_level,
        required_approvals=required,
        allowed_reviewer_roles=roles,
        require_distinct_reviewers=True,
        prohibit_self_approval=True,
        require_safety_officer=require_safety,
        expiration_seconds=(
            900 if critical else
            1800 if high else
            DEFAULT_APPROVAL_TTL_SECONDS
        ),
        auto_approval_allowed=False,
    )


class DefaultApprovalPolicyProvider:

    async def resolve_policy(
        self,
        *,
        tenant_id: str,
        approval_type: ApprovalType,
        risk_level: RiskLevel,
    ) -> ApprovalPolicy:

        validate_identifier(tenant_id)

        return default_approval_policy(
            approval_type=approval_type,
            risk_level=risk_level,
        )


# ============================================================
# HITL Approval Manager
# ============================================================

class HumanInTheLoopManager:

    def __init__(
        self,
        *,
        repository: ApprovalRepository,
        authorizer: HITLAuthorizer,
        identity_provider: ReviewerIdentityProvider,
        policy_provider: ApprovalPolicyProvider | None = None,
        audit_sink: HITLAuditSink | None = None,
        notification_sink: ApprovalNotificationSink | None = None,
    ) -> None:

        self.repository = repository
        self.authorizer = authorizer
        self.identity_provider = identity_provider

        self.policy_provider = (
            policy_provider
            or DefaultApprovalPolicyProvider()
        )

        self.audit_sink = audit_sink
        self.notification_sink = notification_sink

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    async def _authorize(
        self,
        *,
        scope: HITLScope,
        permission: str,
        resource_type: str,
        resource_id: str,
    ) -> None:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            permission=permission,
            resource_type=resource_type,
            resource_id=resource_id,
        )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    async def _audit(
        self,
        *,
        scope: HITLScope,
        action: str,
        approval_id: str,
        metadata: Mapping[str, Any],
    ) -> None:

        if self.audit_sink is None:
            return

        await self.audit_sink.record(
            tenant_id=scope.tenant_id,
            actor_id=scope.user_id,
            action=action,
            resource_id=approval_id,
            metadata=metadata,
        )

    # --------------------------------------------------------
    # Notifications
    # --------------------------------------------------------

    async def _notify(
        self,
        *,
        tenant_id: str,
        approval_id: str,
        event_type: ApprovalEventType,
        message: str,
    ) -> None:

        if self.notification_sink is None:
            return

        try:
            await self.notification_sink.notify(
                tenant_id=tenant_id,
                approval_id=approval_id,
                event_type=event_type,
                message=message,
            )
        except Exception:
            # Notification delivery must not change
            # the result of an already committed decision.
            pass

    # --------------------------------------------------------
    # Event Creation
    # --------------------------------------------------------

    @staticmethod
    def _event(
        *,
        approval_id: str,
        event_type: ApprovalEventType,
        actor_id: str,
        description: str,
    ) -> ApprovalEvent:

        return ApprovalEvent(
            event_id=generate_id(),
            approval_id=approval_id,
            event_type=event_type,
            actor_id=actor_id,
            description=description,
        )

    # --------------------------------------------------------
    # Create Approval
    # --------------------------------------------------------

    async def create_approval(
        self,
        *,
        scope: HITLScope,
        request: ApprovalCreateRequest,
    ) -> ApprovalRecord:

        await self._authorize(
            scope=scope,
            permission="hitl:create",
            resource_type=request.resource_type,
            resource_id=request.resource_id,
        )

        policy = await self.policy_provider.resolve_policy(
            tenant_id=scope.tenant_id,
            approval_type=request.approval_type,
            risk_level=request.risk_level,
        )

        if (
            policy.approval_type != request.approval_type
            or policy.risk_level != request.risk_level
        ):
            raise ApprovalValidationError(
                "Resolved policy does not match request"
            )

        approval_id = generate_id()
        now = utc_now()

        action = copy.deepcopy(
            request.proposed_action
        )

        record = ApprovalRecord(
            approval_id=approval_id,
            tenant_id=scope.tenant_id,
            requested_by=scope.user_id,
            title=request.title,
            description=request.description,
            approval_type=request.approval_type,
            risk_level=request.risk_level,
            resource_type=request.resource_type,
            resource_id=request.resource_id,
            proposed_action=action,
            action_hash=canonical_action_hash(action),
            policy_id=policy.policy_id,
            required_approvals=policy.required_approvals,
            allowed_reviewer_roles=(
                policy.allowed_reviewer_roles
            ),
            require_distinct_reviewers=(
                policy.require_distinct_reviewers
            ),
            prohibit_self_approval=(
                policy.prohibit_self_approval
            ),
            require_safety_officer=(
                policy.require_safety_officer
            ),
            priority=request.priority,
            agent_id=request.agent_id,
            robot_id=request.robot_id,
            task_id=request.task_id,
            ai_confidence=request.ai_confidence,
            justification=request.justification,
            metadata=copy.deepcopy(request.metadata),
            created_at=now,
            updated_at=now,
            expires_at=(
                now
                + timedelta(
                    seconds=policy.expiration_seconds
                )
            ),
        )

        record.events.append(
            self._event(
                approval_id=approval_id,
                event_type=ApprovalEventType.CREATED,
                actor_id=scope.user_id,
                description="Approval request created",
            )
        )

        saved = await self.repository.create(record)

        await self._audit(
            scope=scope,
            action="hitl.approval.created",
            approval_id=approval_id,
            metadata={
                "approval_type": request.approval_type.value,
                "risk_level": request.risk_level.value,
                "action_hash": record.action_hash,
            },
        )

        await self._notify(
            tenant_id=scope.tenant_id,
            approval_id=approval_id,
            event_type=ApprovalEventType.CREATED,
            message="New human review request",
        )

        return saved

    # --------------------------------------------------------
    # Get Approval
    # --------------------------------------------------------

    async def get_approval(
        self,
        *,
        scope: HITLScope,
        approval_id: str,
    ) -> ApprovalRecord:

        validate_identifier(approval_id)

        record = await self.repository.get(
            tenant_id=scope.tenant_id,
            approval_id=approval_id,
        )

        if record is None:
            raise ApprovalNotFoundError(
                "Approval not found"
            )

        await self._authorize(
            scope=scope,
            permission="hitl:read",
            resource_type=record.resource_type,
            resource_id=record.resource_id,
        )

        return record

    # --------------------------------------------------------
    # Review Eligibility
    # --------------------------------------------------------

    @staticmethod
    def _validate_reviewer(
        *,
        record: ApprovalRecord,
        reviewer: ReviewerIdentity,
    ) -> None:

        if not reviewer.active:
            raise ApprovalPermissionError(
                "Reviewer account is inactive"
            )

        if reviewer.tenant_id != record.tenant_id:
            raise ApprovalPermissionError(
                "Reviewer tenant mismatch"
            )

        if (
            record.prohibit_self_approval
            and reviewer.user_id == record.requested_by
        ):
            raise ApprovalPermissionError(
                "Self-approval is prohibited"
            )

        if not (
            reviewer.roles
            & set(record.allowed_reviewer_roles)
        ):
            raise ApprovalPermissionError(
                "Reviewer does not have an eligible role"
            )

        if record.require_distinct_reviewers:
            if any(
                review.reviewer_id == reviewer.user_id
                for review in record.reviews
            ):
                raise ApprovalConflictError(
                    "Reviewer has already submitted a decision"
                )

    # --------------------------------------------------------
    # Review Approval
    # --------------------------------------------------------

    async def review_approval(
        self,
        *,
        scope: HITLScope,
        approval_id: str,
        request: ApprovalReviewRequest,
    ) -> ApprovalDecisionResult:

        record = await self.get_approval(
            scope=scope,
            approval_id=approval_id,
        )

        await self._authorize(
            scope=scope,
            permission="hitl:review",
            resource_type=record.resource_type,
            resource_id=record.resource_id,
        )

        reviewer = await self.identity_provider.get_reviewer(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
        )

        self._validate_reviewer(
            record=record,
            reviewer=reviewer,
        )

        if request.expected_version is not None:
            if record.version != request.expected_version:
                raise ApprovalConflictError(
                    "Approval version mismatch"
                )

        if record.is_expired():
            raise ApprovalExpiredError(
                "Approval request has expired"
            )

        if record.status not in (
            ApprovalStatus.PENDING,
            ApprovalStatus.IN_REVIEW,
            ApprovalStatus.ESCALATED,
        ):
            raise ApprovalConflictError(
                "Approval request is not reviewable"
            )

        original_version = record.version

        review = ApprovalReview(
            review_id=generate_id(),
            reviewer_id=reviewer.user_id,
            decision=request.decision,
            comment=request.comment,
            reviewer_roles=tuple(
                sorted(
                    reviewer.roles,
                    key=lambda role: role.value,
                )
            ),
        )

        record.reviews.append(review)

        if request.decision == ReviewDecision.REJECT:
            record.status = ApprovalStatus.REJECTED
            record.decided_at = utc_now()

            event_type = (
                ApprovalEventType.REJECTED_BY_REVIEWER
            )

        elif (
            request.decision
            == ReviewDecision.REQUEST_CHANGES
        ):
            record.status = ApprovalStatus.ESCALATED

            event_type = (
                ApprovalEventType.CHANGES_REQUESTED
            )

        else:
            enough_approvals = (
                record.approval_count
                >= record.required_approvals
            )

            safety_approved = (
                not record.require_safety_officer
                or record.has_safety_officer_approval()
            )

            if enough_approvals and safety_approved:
                record.status = ApprovalStatus.APPROVED
                record.decided_at = utc_now()

                event_type = ApprovalEventType.APPROVED

            else:
                record.status = ApprovalStatus.IN_REVIEW

                event_type = (
                    ApprovalEventType.APPROVED_BY_REVIEWER
                )

        record.events.append(
            self._event(
                approval_id=approval_id,
                event_type=event_type,
                actor_id=scope.user_id,
                description=(
                    f"Reviewer decision: "
                    f"{request.decision.value}"
                ),
            )
        )

        record.updated_at = utc_now()
        record.version += 1

        saved = await self.repository.save(
            record=record,
            expected_version=original_version,
        )

        await self._audit(
            scope=scope,
            action="hitl.approval.reviewed",
            approval_id=approval_id,
            metadata={
                "decision": request.decision.value,
                "status": saved.status.value,
                "version": saved.version,
            },
        )

        await self._notify(
            tenant_id=scope.tenant_id,
            approval_id=approval_id,
            event_type=event_type,
            message=(
                f"Approval status changed to "
                f"{saved.status.value}"
            ),
        )

        return self._decision_result(saved)

    # --------------------------------------------------------
    # Decision Result
    # --------------------------------------------------------

    @staticmethod
    def _decision_result(
        record: ApprovalRecord,
    ) -> ApprovalDecisionResult:

        return ApprovalDecisionResult(
            approval_id=record.approval_id,
            status=record.status,
            required_approvals=record.required_approvals,
            completed_approvals=record.approval_count,
            remaining_approvals=record.remaining_approvals,
            requires_additional_review=(
                record.status != ApprovalStatus.APPROVED
            ),
            action_hash=record.action_hash,
            expires_at=record.expires_at,
            version=record.version,
        )

    # --------------------------------------------------------
    # Cancel Approval
    # --------------------------------------------------------

    async def cancel_approval(
        self,
        *,
        scope: HITLScope,
        approval_id: str,
        reason: str,
        expected_version: int | None = None,
    ) -> ApprovalRecord:

        if not reason.strip():
            raise ApprovalValidationError(
                "Cancellation reason is required"
            )

        record = await self.get_approval(
            scope=scope,
            approval_id=approval_id,
        )

        await self._authorize(
            scope=scope,
            permission="hitl:cancel",
            resource_type=record.resource_type,
            resource_id=record.resource_id,
        )

        if expected_version is not None:
            if record.version != expected_version:
                raise ApprovalConflictError(
                    "Approval version mismatch"
                )

        if record.status not in (
            ApprovalStatus.PENDING,
            ApprovalStatus.IN_REVIEW,
            ApprovalStatus.ESCALATED,
        ):
            raise ApprovalConflictError(
                "Approval cannot be cancelled"
            )

        previous_version = record.version

        record.status = ApprovalStatus.CANCELLED
        record.decided_at = utc_now()
        record.updated_at = utc_now()
        record.version += 1

        record.events.append(
            self._event(
                approval_id=approval_id,
                event_type=ApprovalEventType.CANCELLED,
                actor_id=scope.user_id,
                description=reason,
            )
        )

        saved = await self.repository.save(
            record=record,
            expected_version=previous_version,
        )

        await self._audit(
            scope=scope,
            action="hitl.approval.cancelled",
            approval_id=approval_id,
            metadata={"reason": reason},
        )

        return saved

    # --------------------------------------------------------
    # Escalate Approval
    # --------------------------------------------------------

    async def escalate_approval(
        self,
        *,
        scope: HITLScope,
        approval_id: str,
        reason: str,
    ) -> ApprovalRecord:

        if not reason.strip():
            raise ApprovalValidationError(
                "Escalation reason is required"
            )

        record = await self.get_approval(
            scope=scope,
            approval_id=approval_id,
        )

        await self._authorize(
            scope=scope,
            permission="hitl:escalate",
            resource_type=record.resource_type,
            resource_id=record.resource_id,
        )

        if record.is_expired():
            raise ApprovalExpiredError(
                "Approval has expired"
            )

        if record.status not in (
            ApprovalStatus.PENDING,
            ApprovalStatus.IN_REVIEW,
            ApprovalStatus.ESCALATED,
        ):
            raise ApprovalConflictError(
                "Approval cannot be escalated"
            )

        previous_version = record.version

        record.status = ApprovalStatus.ESCALATED
        record.updated_at = utc_now()
        record.version += 1

        record.events.append(
            self._event(
                approval_id=approval_id,
                event_type=ApprovalEventType.ESCALATED,
                actor_id=scope.user_id,
                description=reason,
            )
        )

        saved = await self.repository.save(
            record=record,
            expected_version=previous_version,
        )

        await self._audit(
            scope=scope,
            action="hitl.approval.escalated",
            approval_id=approval_id,
            metadata={"reason": reason},
        )

        return saved

    # --------------------------------------------------------
    # Review Queue
    # --------------------------------------------------------

    async def list_review_queue(
        self,
        *,
        scope: HITLScope,
        limit: int = 50,
        risk_level: RiskLevel | None = None,
        approval_type: ApprovalType | None = None,
    ) -> list[ApprovalRecord]:

        if not 1 <= limit <= MAX_QUEUE_LIMIT:
            raise ApprovalValidationError(
                "Invalid review queue limit"
            )

        await self._authorize(
            scope=scope,
            permission="hitl:queue:read",
            resource_type="tenant",
            resource_id=scope.tenant_id,
        )

        reviewer = await self.identity_provider.get_reviewer(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
        )

        if not reviewer.active:
            return []

        records = await self.repository.list(
            tenant_id=scope.tenant_id
        )

        results: list[ApprovalRecord] = []

        for record in records:

            if record.status not in (
                ApprovalStatus.PENDING,
                ApprovalStatus.IN_REVIEW,
                ApprovalStatus.ESCALATED,
            ):
                continue

            if record.is_expired():
                continue

            if (
                risk_level is not None
                and record.risk_level != risk_level
            ):
                continue

            if (
                approval_type is not None
                and record.approval_type != approval_type
            ):
                continue

            try:
                self._validate_reviewer(
                    record=record,
                    reviewer=reviewer,
                )

                await self._authorize(
                    scope=scope,
                    permission="hitl:review",
                    resource_type=record.resource_type,
                    resource_id=record.resource_id,
                )

            except (
                ApprovalPermissionError,
                ApprovalConflictError,
            ):
                continue

            results.append(record)

        priority_order = {
            ApprovalPriority.URGENT: 4,
            ApprovalPriority.HIGH: 3,
            ApprovalPriority.NORMAL: 2,
            ApprovalPriority.LOW: 1,
        }

        risk_order = {
            RiskLevel.CRITICAL: 4,
            RiskLevel.HIGH: 3,
            RiskLevel.MODERATE: 2,
            RiskLevel.LOW: 1,
        }

        results.sort(
            key=lambda record: (
                -priority_order[record.priority],
                -risk_order[record.risk_level],
                record.created_at,
            )
        )

        return results[:limit]

    # --------------------------------------------------------
    # Approval Validation for Downstream Executor
    # --------------------------------------------------------

    async def validate_approved_action(
        self,
        *,
        scope: HITLScope,
        approval_id: str,
        resource_type: str,
        resource_id: str,
        proposed_action: Mapping[str, Any],
    ) -> bool:
        """
        Read-only approval check.

        IMPORTANT:
        This is not an execution authorization token.

        Production execution must atomically consume a
        single-use approval and recheck the robot's live
        safety state in the trusted execution gateway.
        """

        record = await self.get_approval(
            scope=scope,
            approval_id=approval_id,
        )

        await self._authorize(
            scope=scope,
            permission="hitl:validate",
            resource_type=record.resource_type,
            resource_id=record.resource_id,
        )

        if record.status != ApprovalStatus.APPROVED:
            return False

        if record.is_expired():
            return False

        if (
            record.resource_type != resource_type
            or record.resource_id != resource_id
        ):
            return False

        if record.action_hash != canonical_action_hash(
            proposed_action
        ):
            return False

        if (
            record.approval_count
            < record.required_approvals
        ):
            return False

        if (
            record.require_safety_officer
            and not record.has_safety_officer_approval()
        ):
            return False

        return True

    # --------------------------------------------------------
    # Expiration Cleanup
    # --------------------------------------------------------

    async def expire_pending_requests(
        self,
        *,
        scope: HITLScope,
    ) -> int:

        await self._authorize(
            scope=scope,
            permission="hitl:expire",
            resource_type="tenant",
            resource_id=scope.tenant_id,
        )

        records = await self.repository.list(
            tenant_id=scope.tenant_id
        )

        count = 0

        for record in records:

            if record.status not in (
                ApprovalStatus.PENDING,
                ApprovalStatus.IN_REVIEW,
                ApprovalStatus.ESCALATED,
            ):
                continue

            if not record.is_expired():
                continue

            previous_version = record.version

            record.status = ApprovalStatus.EXPIRED
            record.updated_at = utc_now()
            record.decided_at = utc_now()
            record.version += 1

            record.events.append(
                self._event(
                    approval_id=record.approval_id,
                    event_type=ApprovalEventType.EXPIRED,
                    actor_id=scope.user_id,
                    description="Approval validity expired",
                )
            )

            try:
                await self.repository.save(
                    record=record,
                    expected_version=previous_version,
                )

                count += 1

            except ApprovalConflictError:
                continue

        return count

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    async def get_statistics(
        self,
        *,
        scope: HITLScope,
    ) -> ApprovalStatistics:

        await self._authorize(
            scope=scope,
            permission="hitl:statistics:read",
            resource_type="tenant",
            resource_id=scope.tenant_id,
        )

        records = await self.repository.list(
            tenant_id=scope.tenant_id
        )

        def count_status(
            status: ApprovalStatus,
        ) -> int:
            return sum(
                record.status == status
                for record in records
            )

        return ApprovalStatistics(
            tenant_id=scope.tenant_id,
            total=len(records),
            pending=count_status(ApprovalStatus.PENDING),
            in_review=count_status(
                ApprovalStatus.IN_REVIEW
            ),
            approved=count_status(
                ApprovalStatus.APPROVED
            ),
            rejected=count_status(
                ApprovalStatus.REJECTED
            ),
            expired=count_status(
                ApprovalStatus.EXPIRED
            ),
            cancelled=count_status(
                ApprovalStatus.CANCELLED
            ),
            escalated=count_status(
                ApprovalStatus.ESCALATED
            ),
        )


# ============================================================
# Factory
# ============================================================

def create_human_in_the_loop_manager(
    *,
    repository: ApprovalRepository,
    authorizer: HITLAuthorizer,
    identity_provider: ReviewerIdentityProvider,
    policy_provider: ApprovalPolicyProvider | None = None,
    audit_sink: HITLAuditSink | None = None,
    notification_sink: ApprovalNotificationSink | None = None,
) -> HumanInTheLoopManager:

    return HumanInTheLoopManager(
        repository=repository,
        authorizer=authorizer,
        identity_provider=identity_provider,
        policy_provider=policy_provider,
        audit_sink=audit_sink,
        notification_sink=notification_sink,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "ApprovalType",
    "RiskLevel",
    "ApprovalStatus",
    "ReviewDecision",
    "ReviewerRole",
    "ApprovalPriority",
    "ApprovalEventType",
    "ApprovalPolicy",
    "ApprovalCreateRequest",
    "ApprovalReviewRequest",
    "ReviewerIdentity",
    "ApprovalReview",
    "ApprovalEvent",
    "ApprovalRecord",
    "ApprovalDecisionResult",
    "ApprovalStatistics",
    "HITLScope",
    "ApprovalRepository",
    "HITLAuthorizer",
    "ReviewerIdentityProvider",
    "HITLAuditSink",
    "ApprovalNotificationSink",
    "ApprovalPolicyProvider",
    "InMemoryApprovalRepository",
    "DefaultApprovalPolicyProvider",
    "HumanInTheLoopManager",
    "HITLError",
    "ApprovalNotFoundError",
    "ApprovalPermissionError",
    "ApprovalValidationError",
    "ApprovalConflictError",
    "ApprovalExpiredError",
    "ApprovalBackendError",
    "canonical_action_hash",
    "default_approval_policy",
    "create_human_in_the_loop_manager",
]
"""Human Approval module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/security/human_approval.py

Python 3.11+
Standard library only.

Features:
    - Human-in-the-loop approval workflow
    - Single and dual human approval
    - Risk-based approval requirements
    - Agent tool and robot command approvals
    - Model deployment and repair approvals
    - Expiring approval requests
    - Self-approval prevention
    - Role-based reviewer eligibility
    - Tenant-scoped approval records
    - Command/payload hash binding
    - Rejection and revocation
    - Single-use authorization consumption
    - Audit logging and JSON export

SECURITY:
    Simulation-only approvals.

    An approval is NOT an authorization to
    execute physical robot hardware commands.

    Reviewer identities and roles must come
    from a trusted authentication provider.

    Independent command interlocks and
    safety-rated hardware remain mandatory.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class HumanApprovalError(Exception):
    pass


class ApprovalValidationError(HumanApprovalError):
    pass


class ApprovalNotFoundError(HumanApprovalError):
    pass


class ApprovalAccessDenied(HumanApprovalError):
    pass


class ApprovalTransitionError(HumanApprovalError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REVOKED = "revoked"
    CONSUMED = "consumed"


class ApprovalRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalActionType(str, Enum):
    ROBOT_COMMAND = "robot_command"
    AGENT_TOOL = "agent_tool"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_ROLLBACK = "model_rollback"
    REPAIR_ACTION = "repair_action"
    MAINTENANCE_ACTION = "maintenance_action"
    TASK_EXECUTION = "task_execution"
    SIMULATION_ACTION = "simulation_action"
    SECURITY_CHANGE = "security_change"
    FILE_OPERATION = "file_operation"


class ApprovalRole(str, Enum):
    ADMIN = "admin"
    SAFETY_OFFICER = "safety_officer"
    ENGINEER = "engineer"
    OPERATOR = "operator"
    TECHNICIAN = "technician"
    REVIEWER = "reviewer"


class ApprovalDecisionType(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(
    value: str,
    name: str,
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
        raise ApprovalValidationError(
            f"Invalid {name}"
        )

    return value


def validate_timestamp(
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ApprovalValidationError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def payload_sha256(
    payload: Mapping[str, Any],
) -> str:
    """
    Compute a deterministic hash for a JSON payload.

    The same canonical payload must be used
    when the approval is consumed.

    JSON payloads must contain finite values
    and JSON-compatible data types.
    """
    try:
        serialized = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ApprovalValidationError(
            "Payload must be JSON serializable"
        ) from exc

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class HumanApprovalConfig:
    default_ttl_seconds: int = 300
    maximum_ttl_seconds: int = 3600

    maximum_requests: int = 100000
    maximum_audit_events: int = 20000

    require_two_person_high_risk: bool = True
    require_two_person_critical: bool = True

    prevent_self_approval: bool = True

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        for name in (
            "default_ttl_seconds",
            "maximum_ttl_seconds",
            "maximum_requests",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ApprovalValidationError(
                    f"{name} must be a positive integer"
                )

        if (
            self.default_ttl_seconds
            > self.maximum_ttl_seconds
        ):
            raise ApprovalValidationError(
                "Default TTL exceeds maximum TTL"
            )

        if self.allow_hardware_control:
            raise ApprovalValidationError(
                "Physical hardware control is disabled"
            )


# ============================================================
# ROLE POLICY
# ============================================================

DEFAULT_REVIEWER_ROLES: dict[
    ApprovalActionType,
    frozenset[ApprovalRole],
] = {
    ApprovalActionType.ROBOT_COMMAND: frozenset({
        ApprovalRole.ADMIN,
        ApprovalRole.SAFETY_OFFICER,
        ApprovalRole.OPERATOR,
    }),

    ApprovalActionType.AGENT_TOOL: frozenset({
        ApprovalRole.ADMIN,
        ApprovalRole.ENGINEER,
        ApprovalRole.REVIEWER,
    }),

    ApprovalActionType.MODEL_DEPLOYMENT: frozenset({
        ApprovalRole.ADMIN,
        ApprovalRole.ENGINEER,
        ApprovalRole.SAFETY_OFFICER,
    }),

    ApprovalActionType.MODEL_ROLLBACK: frozenset({
        ApprovalRole.ADMIN,
        ApprovalRole.ENGINEER,
    }),

    ApprovalActionType.REPAIR_ACTION: frozenset({
        ApprovalRole.ADMIN,
        ApprovalRole.TECHNICIAN,
        ApprovalRole.SAFETY_OFFICER,
    }),

    ApprovalActionType.MAINTENANCE_ACTION: frozenset({
        ApprovalRole.ADMIN,
        ApprovalRole.TECHNICIAN,
    }),

    ApprovalActionType.TASK_EXECUTION: frozenset({
        ApprovalRole.ADMIN,
        ApprovalRole.OPERATOR,
        ApprovalRole.SAFETY_OFFICER,
    }),

    ApprovalActionType.SIMULATION_ACTION: frozenset({
        ApprovalRole.ADMIN,
        ApprovalRole.ENGINEER,
        ApprovalRole.OPERATOR,
    }),

    ApprovalActionType.SECURITY_CHANGE: frozenset({
        ApprovalRole.ADMIN,
        ApprovalRole.SAFETY_OFFICER,
    }),

    ApprovalActionType.FILE_OPERATION: frozenset({
        ApprovalRole.ADMIN,
        ApprovalRole.ENGINEER,
    }),
}


# ============================================================
# TRUSTED PRINCIPAL
# ============================================================

@dataclass(frozen=True)
class ApprovalPrincipal:
    principal_id: str
    tenant_id: str

    roles: frozenset[ApprovalRole]

    active: bool = True

    def __post_init__(self) -> None:
        validate_id(
            self.principal_id,
            "principal_id",
        )

        validate_id(
            self.tenant_id,
            "tenant_id",
        )

        if not self.roles:
            raise ApprovalValidationError(
                "Principal requires at least one role"
            )

        object.__setattr__(
            self,
            "roles",
            frozenset(
                ApprovalRole(role)
                for role in self.roles
            ),
        )


# ============================================================
# APPROVAL REQUEST
# ============================================================

@dataclass(frozen=True)
class ApprovalRequestCreate:
    tenant_id: str
    requester_id: str

    action_type: ApprovalActionType
    risk_level: ApprovalRiskLevel

    resource_id: str

    payload: Mapping[str, Any]

    reason: str

    ttl_seconds: int | None = None

    simulation_only: bool = True

    def __post_init__(self) -> None:
        validate_id(
            self.tenant_id,
            "tenant_id",
        )

        validate_id(
            self.requester_id,
            "requester_id",
        )

        validate_id(
            self.resource_id,
            "resource_id",
        )

        if (
            not isinstance(self.reason, str)
            or not self.reason.strip()
            or len(self.reason) > 2000
        ):
            raise ApprovalValidationError(
                "A valid reason is required"
            )

        if not self.simulation_only:
            raise ApprovalAccessDenied(
                "Hardware approvals are not supported"
            )

        payload_sha256(self.payload)


# ============================================================
# REVIEW DECISION
# ============================================================

@dataclass(frozen=True)
class ApprovalReview:
    reviewer_id: str

    decision: ApprovalDecisionType

    comment: str

    reviewed_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "decision": self.decision.value,
            "comment": self.comment,
            "reviewed_at": (
                self.reviewed_at.isoformat()
            ),
        }


# ============================================================
# APPROVAL RECORD
# ============================================================

@dataclass
class ApprovalRecord:
    approval_id: str

    tenant_id: str
    requester_id: str

    action_type: ApprovalActionType
    risk_level: ApprovalRiskLevel

    resource_id: str
    payload_hash: str

    reason: str

    required_approvals: int

    status: ApprovalStatus = (
        ApprovalStatus.PENDING
    )

    reviews: list[ApprovalReview] = field(
        default_factory=list
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    expires_at: datetime = field(
        default_factory=lambda: (
            utc_now() + timedelta(minutes=5)
        )
    )

    consumed_at: datetime | None = None
    revoked_at: datetime | None = None

    def approval_count(self) -> int:
        return sum(
            review.decision
            == ApprovalDecisionType.APPROVE
            for review in self.reviews
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "tenant_id": self.tenant_id,
            "requester_id": self.requester_id,
            "action_type": self.action_type.value,
            "risk_level": self.risk_level.value,
            "resource_id": self.resource_id,
            "payload_hash": self.payload_hash,
            "reason": self.reason,
            "required_approvals": (
                self.required_approvals
            ),
            "approval_count": (
                self.approval_count()
            ),
            "status": self.status.value,
            "reviews": [
                review.to_dict()
                for review in self.reviews
            ],
            "created_at": (
                self.created_at.isoformat()
            ),
            "expires_at": (
                self.expires_at.isoformat()
            ),
            "consumed_at": (
                self.consumed_at.isoformat()
                if self.consumed_at
                else None
            ),
            "revoked_at": (
                self.revoked_at.isoformat()
                if self.revoked_at
                else None
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class ApprovalAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    approval_id: str

    action: str
    actor_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": (
                self.timestamp.isoformat()
            ),
            "tenant_id": self.tenant_id,
            "approval_id": self.approval_id,
            "action": self.action,
            "actor_id": self.actor_id,
        }


# ============================================================
# HUMAN APPROVAL SERVICE
# ============================================================

class RoboticsHumanApprovalService:

    def __init__(
        self,
        config: HumanApprovalConfig | None = None,
    ) -> None:
        self.config = (
            config or HumanApprovalConfig()
        )

        self.records: dict[
            str,
            ApprovalRecord
        ] = {}

        self.audit_events: list[
            ApprovalAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        record: ApprovalRecord,
        action: str,
        actor_id: str,
    ) -> None:
        self.audit_events.append(
            ApprovalAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=record.tenant_id,
                approval_id=record.approval_id,
                action=action,
                actor_id=actor_id,
            )
        )

        if (
            len(self.audit_events)
            > self.config.maximum_audit_events
        ):
            self.audit_events.pop(0)

    # --------------------------------------------------------
    # REQUIRED APPROVAL COUNT
    # --------------------------------------------------------

    def _required_approvals(
        self,
        risk: ApprovalRiskLevel,
    ) -> int:
        if (
            risk == ApprovalRiskLevel.CRITICAL
            and self.config.require_two_person_critical
        ):
            return 2

        if (
            risk == ApprovalRiskLevel.HIGH
            and self.config.require_two_person_high_risk
        ):
            return 2

        return 1

    # --------------------------------------------------------
    # CREATE REQUEST
    # --------------------------------------------------------

    def create_request(
        self,
        request: ApprovalRequestCreate,
    ) -> ApprovalRecord:
        ttl = (
            request.ttl_seconds
            if request.ttl_seconds is not None
            else self.config.default_ttl_seconds
        )

        if (
            isinstance(ttl, bool)
            or not isinstance(ttl, int)
            or not (
                1 <= ttl
                <= self.config.maximum_ttl_seconds
            )
        ):
            raise ApprovalValidationError(
                "Invalid approval TTL"
            )

        created_at = utc_now()

        record = ApprovalRecord(
            approval_id=str(uuid.uuid4()),
            tenant_id=request.tenant_id,
            requester_id=request.requester_id,
            action_type=request.action_type,
            risk_level=request.risk_level,
            resource_id=request.resource_id,
            payload_hash=payload_sha256(
                request.payload
            ),
            reason=request.reason,
            required_approvals=(
                self._required_approvals(
                    request.risk_level
                )
            ),
            created_at=created_at,
            expires_at=(
                created_at
                + timedelta(seconds=ttl)
            ),
        )

        with self._lock:
            if (
                len(self.records)
                >= self.config.maximum_requests
            ):
                raise HumanApprovalError(
                    "Approval capacity reached"
                )

            self.records[
                record.approval_id
            ] = record

            self._audit(
                record,
                "approval_requested",
                request.requester_id,
            )

        return record

    # --------------------------------------------------------
    # LOAD RECORD
    # --------------------------------------------------------

    def _get(
        self,
        tenant_id: str,
        approval_id: str,
    ) -> ApprovalRecord:
        record = self.records.get(
            approval_id
        )

        if (
            record is None
            or record.tenant_id != tenant_id
        ):
            raise ApprovalNotFoundError(
                "Approval request not found"
            )

        self._expire_if_needed(
            record
        )

        return record

    def get_request(
        self,
        *,
        tenant_id: str,
        approval_id: str,
    ) -> ApprovalRecord:
        with self._lock:
            return self._get(
                tenant_id,
                approval_id,
            )

    # --------------------------------------------------------
    # EXPIRATION
    # --------------------------------------------------------

    def _expire_if_needed(
        self,
        record: ApprovalRecord,
    ) -> None:
        if (
            record.status in {
                ApprovalStatus.PENDING,
                ApprovalStatus.APPROVED,
            }
            and utc_now() >= record.expires_at
        ):
            record.status = ApprovalStatus.EXPIRED

            self._audit(
                record,
                "approval_expired",
                "system",
            )

    # --------------------------------------------------------
    # REVIEW
    # --------------------------------------------------------

    def review(
        self,
        *,
        tenant_id: str,
        approval_id: str,
        reviewer: ApprovalPrincipal,
        decision: ApprovalDecisionType,
        comment: str,
    ) -> ApprovalRecord:
        if (
            not isinstance(comment, str)
            or not comment.strip()
            or len(comment) > 2000
        ):
            raise ApprovalValidationError(
                "Review comment is required"
            )

        decision = ApprovalDecisionType(
            decision
        )

        with self._lock:
            record = self._get(
                tenant_id,
                approval_id,
            )

            if record.status != ApprovalStatus.PENDING:
                raise ApprovalTransitionError(
                    "Approval is not pending"
                )

            if (
                not reviewer.active
                or reviewer.tenant_id != tenant_id
            ):
                raise ApprovalAccessDenied(
                    "Reviewer is not eligible"
                )

            if (
                self.config.prevent_self_approval
                and reviewer.principal_id
                == record.requester_id
            ):
                raise ApprovalAccessDenied(
                    "Self-approval is prohibited"
                )

            allowed_roles = DEFAULT_REVIEWER_ROLES[
                record.action_type
            ]

            if not (
                reviewer.roles & allowed_roles
            ):
                raise ApprovalAccessDenied(
                    "Reviewer lacks required role"
                )

            if any(
                review.reviewer_id
                == reviewer.principal_id
                for review in record.reviews
            ):
                raise ApprovalTransitionError(
                    "Reviewer has already voted"
                )

            review = ApprovalReview(
                reviewer_id=reviewer.principal_id,
                decision=decision,
                comment=comment,
            )

            record.reviews.append(
                review
            )

            if decision == ApprovalDecisionType.REJECT:
                record.status = ApprovalStatus.REJECTED

                self._audit(
                    record,
                    "approval_rejected",
                    reviewer.principal_id,
                )

            elif (
                record.approval_count()
                >= record.required_approvals
            ):
                record.status = ApprovalStatus.APPROVED

                self._audit(
                    record,
                    "approval_approved",
                    reviewer.principal_id,
                )

            else:
                self._audit(
                    record,
                    "approval_reviewed",
                    reviewer.principal_id,
                )

            return record

    # --------------------------------------------------------
    # REVOKE
    # --------------------------------------------------------

    def revoke(
        self,
        *,
        tenant_id: str,
        approval_id: str,
        actor: ApprovalPrincipal,
    ) -> ApprovalRecord:
        with self._lock:
            record = self._get(
                tenant_id,
                approval_id,
            )

            if (
                not actor.active
                or actor.tenant_id != tenant_id
                or not (
                    actor.roles
                    & {
                        ApprovalRole.ADMIN,
                        ApprovalRole.SAFETY_OFFICER,
                    }
                )
            ):
                raise ApprovalAccessDenied(
                    "Revocation requires administrator "
                    "or safety officer"
                )

            if record.status not in {
                ApprovalStatus.PENDING,
                ApprovalStatus.APPROVED,
            }:
                raise ApprovalTransitionError(
                    "Approval cannot be revoked"
                )

            record.status = ApprovalStatus.REVOKED
            record.revoked_at = utc_now()

            self._audit(
                record,
                "approval_revoked",
                actor.principal_id,
            )

            return record

    # --------------------------------------------------------
    # CONSUME APPROVAL
    # --------------------------------------------------------

    def consume(
        self,
        *,
        tenant_id: str,
        approval_id: str,
        action_type: ApprovalActionType,
        resource_id: str,
        payload: Mapping[str, Any],
    ) -> bool:
        """
        Single-use approval validation.

        IMPORTANT:
            The calling service must separately
            verify its own authenticated identity,
            current command authorization,
            geofence status and safety interlocks.

            This method does not execute a command.
        """
        with self._lock:
            record = self._get(
                tenant_id,
                approval_id,
            )

            if record.status != ApprovalStatus.APPROVED:
                raise ApprovalAccessDenied(
                    "Approval is not active"
                )

            if (
                record.action_type
                != ApprovalActionType(action_type)
            ):
                raise ApprovalAccessDenied(
                    "Action type mismatch"
                )

            if record.resource_id != resource_id:
                raise ApprovalAccessDenied(
                    "Resource mismatch"
                )

            if (
                record.payload_hash
                != payload_sha256(payload)
            ):
                raise ApprovalAccessDenied(
                    "Payload has changed since approval"
                )

            record.status = ApprovalStatus.CONSUMED
            record.consumed_at = utc_now()

            self._audit(
                record,
                "approval_consumed",
                "trusted_service",
            )

            return True

    # --------------------------------------------------------
    # LIST APPROVALS
    # --------------------------------------------------------

    def list_requests(
        self,
        *,
        tenant_id: str,
        status: ApprovalStatus | None = None,
    ) -> list[ApprovalRecord]:
        with self._lock:
            results = []

            for record in self.records.values():
                if record.tenant_id != tenant_id:
                    continue

                self._expire_if_needed(
                    record
                )

                if (
                    status is not None
                    and record.status != status
                ):
                    continue

                results.append(
                    record
                )

            return results

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            records = [
                record
                for record in self.records.values()
                if (
                    tenant_id is None
                    or record.tenant_id == tenant_id
                )
            ]

            for record in records:
                self._expire_if_needed(
                    record
                )

            return {
                "total_requests": len(records),
                "pending": sum(
                    record.status == ApprovalStatus.PENDING
                    for record in records
                ),
                "approved": sum(
                    record.status == ApprovalStatus.APPROVED
                    for record in records
                ),
                "rejected": sum(
                    record.status == ApprovalStatus.REJECTED
                    for record in records
                ),
                "expired": sum(
                    record.status == ApprovalStatus.EXPIRED
                    for record in records
                ),
                "revoked": sum(
                    record.status == ApprovalStatus.REVOKED
                    for record in records
                ),
                "consumed": sum(
                    record.status == ApprovalStatus.CONSUMED
                    for record in records
                ),
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
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            records = self.list_requests(
                tenant_id=tenant_id
            )

            payload = {
                "generated_at": (
                    utc_now().isoformat()
                ),
                "tenant_id": tenant_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "approvals": [
                    record.to_dict()
                    for record in records
                ],
                "audit_events": [
                    event.to_dict()
                    for event in self.audit_events
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
    service = RoboticsHumanApprovalService()

    command_payload = {
        "robot_id": "robot_001",
        "command": "navigate",
        "target": {
            "x": 10.0,
            "y": 20.0,
        },
        "simulation_only": True,
    }

    request = ApprovalRequestCreate(
        tenant_id="tenant_alpha",
        requester_id="navigation_agent_001",
        action_type=ApprovalActionType.ROBOT_COMMAND,
        risk_level=ApprovalRiskLevel.HIGH,
        resource_id="robot_001",
        payload=command_payload,
        reason=(
            "Navigation simulation requires "
            "human safety approval"
        ),
    )

    approval = service.create_request(
        request
    )

    safety_officer = ApprovalPrincipal(
        principal_id="safety_officer_001",
        tenant_id="tenant_alpha",
        roles=frozenset({
            ApprovalRole.SAFETY_OFFICER
        }),
    )

    operator = ApprovalPrincipal(
        principal_id="operator_001",
        tenant_id="tenant_alpha",
        roles=frozenset({
            ApprovalRole.OPERATOR
        }),
    )

    service.review(
        tenant_id="tenant_alpha",
        approval_id=approval.approval_id,
        reviewer=safety_officer,
        decision=ApprovalDecisionType.APPROVE,
        comment="Simulation safety review passed",
    )

    service.review(
        tenant_id="tenant_alpha",
        approval_id=approval.approval_id,
        reviewer=operator,
        decision=ApprovalDecisionType.APPROVE,
        comment="Simulation operation approved",
    )

    service.consume(
        tenant_id="tenant_alpha",
        approval_id=approval.approval_id,
        action_type=ApprovalActionType.ROBOT_COMMAND,
        resource_id="robot_001",
        payload=command_payload,
    )

    print(
        json.dumps(
            approval.to_dict(),
            indent=2,
        )
    )

    service.export_json(
        "data/security/human_approval_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

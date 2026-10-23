"""Command Authorization module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/security/command_authorization.py

Python 3.11+
Standard library only.

Features:
    - Command authorization
    - Role-based access control
    - Robot-scoped permissions
    - Command allowlisting
    - Operating-mode restrictions
    - Human approval workflow
    - Two-person approval for sensitive commands
    - Robot safety-state validation
    - Battery and health checks
    - Command expiration
    - Replay protection
    - Authorization audit trail
    - JSON report export

Safety:
    SIMULATION ONLY.

    No physical robot commands are executed.
    No emergency-stop bypass or safety override.
    Physical safety must be enforced independently
    by certified robot controllers and interlocks.
"""

from __future__ import annotations

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

class CommandAuthorizationError(Exception):
    pass


class CommandValidationError(CommandAuthorizationError):
    pass


class CommandDeniedError(CommandAuthorizationError):
    pass


class CommandNotFoundError(CommandAuthorizationError):
    pass


class CommandStateError(CommandAuthorizationError):
    pass


# ============================================================
# ENUMS
# ============================================================

class CommandRole(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    ENGINEER = "engineer"
    TECHNICIAN = "technician"
    VIEWER = "viewer"
    SERVICE = "service"


class CommandType(str, Enum):
    NAVIGATE = "navigate"
    MOVE = "move"
    STOP = "stop"

    PICK = "pick"
    PLACE = "place"
    MANIPULATE = "manipulate"

    START_TASK = "start_task"
    CANCEL_TASK = "cancel_task"

    START_SIMULATION = "start_simulation"
    STOP_SIMULATION = "stop_simulation"

    START_INSPECTION = "start_inspection"
    START_DIAGNOSTICS = "start_diagnostics"

    UPDATE_CONFIGURATION = "update_configuration"
    DEPLOY_MODEL = "deploy_model"


class RobotOperatingMode(str, Enum):
    IDLE = "idle"
    SIMULATION = "simulation"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    EMERGENCY = "emergency"
    OFFLINE = "offline"


class AuthorizationStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CONSUMED = "consumed"


class AuthorizationReason(str, Enum):
    ALLOWED = "allowed"
    INVALID_ROLE = "invalid_role"
    COMMAND_NOT_ALLOWED = "command_not_allowed"
    ROBOT_SCOPE_DENIED = "robot_scope_denied"
    INVALID_MODE = "invalid_mode"
    ROBOT_UNAVAILABLE = "robot_unavailable"
    SAFETY_INTERLOCK = "safety_interlock"
    LOW_BATTERY = "low_battery"
    LOW_HEALTH = "low_health"
    SENSOR_UNAVAILABLE = "sensor_unavailable"
    NETWORK_UNAVAILABLE = "network_unavailable"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_DENIED = "approval_denied"
    EXPIRED = "expired"
    REPLAY_DETECTED = "replay_detected"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CommandValidationError(
            f"{name} is required"
        )

    return value.strip()


def finite(
    value: float,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise CommandValidationError(
            f"{name} must be finite"
        )

    result = float(value)

    if not minimum <= result <= maximum:
        raise CommandValidationError(
            f"{name} must be between "
            f"{minimum} and {maximum}"
        )

    return result


def parse_time(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except (ValueError, AttributeError) as exc:
        raise CommandValidationError(
            "Invalid timestamp"
        ) from exc

    if result.tzinfo is None:
        raise CommandValidationError(
            "Timezone-aware timestamp required"
        )

    return result.astimezone(timezone.utc)


def json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(value, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise CommandValidationError(
            "Data must be JSON serializable"
        ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CommandAuthorizationConfig:
    minimum_battery_soc_percent: float = 20.0
    minimum_health_score: float = 70.0

    command_ttl_seconds: int = 120
    maximum_pending_commands: int = 10000
    maximum_audit_events: int = 20000

    require_human_approval: bool = True
    require_two_person_approval: bool = True

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        finite(
            self.minimum_battery_soc_percent,
            "minimum_battery_soc_percent",
            0,
            100,
        )

        finite(
            self.minimum_health_score,
            "minimum_health_score",
            0,
            100,
        )

        for name in (
            "command_ttl_seconds",
            "maximum_pending_commands",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise CommandValidationError(
                    f"{name} must be positive"
                )

        if self.allow_hardware_control:
            raise CommandValidationError(
                "Physical hardware control is disabled"
            )


# ============================================================
# COMMAND ROLE POLICY
# ============================================================

ROLE_COMMANDS: dict[
    CommandRole,
    frozenset[CommandType]
] = {
    CommandRole.ADMIN: frozenset(CommandType),

    CommandRole.OPERATOR: frozenset({
        CommandType.NAVIGATE,
        CommandType.MOVE,
        CommandType.STOP,
        CommandType.PICK,
        CommandType.PLACE,
        CommandType.START_TASK,
        CommandType.CANCEL_TASK,
        CommandType.START_SIMULATION,
        CommandType.STOP_SIMULATION,
        CommandType.START_INSPECTION,
        CommandType.START_DIAGNOSTICS,
    }),

    CommandRole.ENGINEER: frozenset({
        CommandType.NAVIGATE,
        CommandType.MOVE,
        CommandType.STOP,
        CommandType.MANIPULATE,
        CommandType.START_SIMULATION,
        CommandType.STOP_SIMULATION,
        CommandType.START_INSPECTION,
        CommandType.START_DIAGNOSTICS,
        CommandType.UPDATE_CONFIGURATION,
        CommandType.DEPLOY_MODEL,
    }),

    CommandRole.TECHNICIAN: frozenset({
        CommandType.STOP,
        CommandType.START_INSPECTION,
        CommandType.START_DIAGNOSTICS,
    }),

    CommandRole.VIEWER: frozenset(),

    CommandRole.SERVICE: frozenset({
        CommandType.START_SIMULATION,
        CommandType.STOP_SIMULATION,
        CommandType.START_INSPECTION,
        CommandType.START_DIAGNOSTICS,
    }),
}


SENSITIVE_COMMANDS = frozenset({
    CommandType.MOVE,
    CommandType.NAVIGATE,
    CommandType.MANIPULATE,
    CommandType.PICK,
    CommandType.PLACE,
    CommandType.UPDATE_CONFIGURATION,
    CommandType.DEPLOY_MODEL,
})


# ============================================================
# PRINCIPAL
# ============================================================

@dataclass(frozen=True)
class CommandPrincipal:
    principal_id: str
    role: CommandRole

    allowed_robot_ids: frozenset[str] = field(
        default_factory=frozenset
    )

    active: bool = True

    def __post_init__(self) -> None:
        require_text(
            self.principal_id,
            "principal_id",
        )

        object.__setattr__(
            self,
            "role",
            CommandRole(self.role),
        )

    def can_access_robot(
        self,
        robot_id: str,
    ) -> bool:
        return (
            self.active
            and (
                robot_id in self.allowed_robot_ids
                or "*" in self.allowed_robot_ids
            )
        )


# ============================================================
# ROBOT SAFETY SNAPSHOT
# ============================================================

@dataclass(frozen=True)
class RobotCommandState:
    robot_id: str

    mode: RobotOperatingMode

    battery_soc_percent: float
    health_score: float

    emergency_stop_active: bool = False
    safety_interlock_active: bool = False

    sensor_available: bool = True
    network_connected: bool = True

    maintenance_required: bool = False
    repair_required: bool = False

    timestamp: str = field(
        default_factory=utc_iso
    )

    def __post_init__(self) -> None:
        require_text(
            self.robot_id,
            "robot_id",
        )

        object.__setattr__(
            self,
            "mode",
            RobotOperatingMode(self.mode),
        )

        finite(
            self.battery_soc_percent,
            "battery_soc_percent",
            0,
            100,
        )

        finite(
            self.health_score,
            "health_score",
            0,
            100,
        )

        parse_time(self.timestamp)


# ============================================================
# COMMAND REQUEST
# ============================================================

@dataclass(frozen=True)
class RobotCommandRequest:
    command_id: str
    robot_id: str
    command_type: CommandType

    requested_by: str

    parameters: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=utc_iso
    )

    expires_at: str = ""

    def __post_init__(self) -> None:
        require_text(
            self.command_id,
            "command_id",
        )

        require_text(
            self.robot_id,
            "robot_id",
        )

        require_text(
            self.requested_by,
            "requested_by",
        )

        object.__setattr__(
            self,
            "command_type",
            CommandType(self.command_type),
        )

        object.__setattr__(
            self,
            "parameters",
            json_copy(self.parameters),
        )

        parse_time(self.created_at)

        if self.expires_at:
            if (
                parse_time(self.expires_at)
                <= parse_time(self.created_at)
            ):
                raise CommandValidationError(
                    "Command expiration must follow creation"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "robot_id": self.robot_id,
            "command_type": self.command_type.value,
            "requested_by": self.requested_by,
            "parameters": self.parameters,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


# ============================================================
# AUTHORIZATION DECISION
# ============================================================

@dataclass
class CommandAuthorizationDecision:
    command_id: str
    status: AuthorizationStatus
    reason: AuthorizationReason

    required_approvals: int = 0

    approvals: set[str] = field(
        default_factory=set
    )

    created_at: str = field(
        default_factory=utc_iso
    )

    updated_at: str = field(
        default_factory=utc_iso
    )

    details: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "status": self.status.value,
            "reason": self.reason.value,
            "required_approvals": self.required_approvals,
            "approvals": sorted(self.approvals),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "details": self.details,
        }


# ============================================================
# COMMAND POLICY ENGINE
# ============================================================

class RoboticsCommandPolicyEngine:

    def __init__(
        self,
        config: CommandAuthorizationConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        principal: CommandPrincipal,
        request: RobotCommandRequest,
        robot: RobotCommandState,
    ) -> AuthorizationReason:
        if not principal.active:
            return AuthorizationReason.INVALID_ROLE

        if not principal.can_access_robot(
            request.robot_id
        ):
            return AuthorizationReason.ROBOT_SCOPE_DENIED

        if (
            request.command_type
            not in ROLE_COMMANDS[principal.role]
        ):
            return AuthorizationReason.COMMAND_NOT_ALLOWED

        if robot.robot_id != request.robot_id:
            return AuthorizationReason.ROBOT_UNAVAILABLE

        if robot.emergency_stop_active:
            return AuthorizationReason.SAFETY_INTERLOCK

        if robot.safety_interlock_active:
            return AuthorizationReason.SAFETY_INTERLOCK

        if not robot.network_connected:
            return AuthorizationReason.NETWORK_UNAVAILABLE

        if not robot.sensor_available:
            return AuthorizationReason.SENSOR_UNAVAILABLE

        if (
            robot.battery_soc_percent
            < self.config.minimum_battery_soc_percent
        ):
            return AuthorizationReason.LOW_BATTERY

        if (
            robot.health_score
            < self.config.minimum_health_score
        ):
            return AuthorizationReason.LOW_HEALTH

        if (
            robot.maintenance_required
            or robot.repair_required
        ):
            return AuthorizationReason.ROBOT_UNAVAILABLE

        if robot.mode not in (
            RobotOperatingMode.IDLE,
            RobotOperatingMode.SIMULATION,
        ):
            return AuthorizationReason.INVALID_MODE

        return AuthorizationReason.ALLOWED


# ============================================================
# CENTRAL AUTHORIZATION SERVICE
# ============================================================

class RoboticsCommandAuthorizationService:

    def __init__(
        self,
        config: CommandAuthorizationConfig | None = None,
    ) -> None:
        self.config = (
            config or CommandAuthorizationConfig()
        )

        self.policy = RoboticsCommandPolicyEngine(
            self.config
        )

        self.principals: dict[
            str,
            CommandPrincipal
        ] = {}

        self.robots: dict[
            str,
            RobotCommandState
        ] = {}

        self.requests: dict[
            str,
            RobotCommandRequest
        ] = {}

        self.decisions: dict[
            str,
            CommandAuthorizationDecision
        ] = {}

        self.audit_events: list[
            dict[str, Any]
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        action: str,
        command_id: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.audit_events.append({
            "event_id": str(uuid.uuid4()),
            "timestamp": utc_iso(),
            "action": action,
            "command_id": command_id,
            "details": json_copy(
                dict(details or {})
            ),
        })

        if (
            len(self.audit_events)
            > self.config.maximum_audit_events
        ):
            self.audit_events.pop(0)

    # --------------------------------------------------------
    # REGISTER PRINCIPAL
    # --------------------------------------------------------

    def register_principal(
        self,
        principal: CommandPrincipal,
    ) -> None:
        with self._lock:
            self.principals[
                principal.principal_id
            ] = principal

    # --------------------------------------------------------
    # UPDATE ROBOT SAFETY STATE
    # --------------------------------------------------------

    def update_robot_state(
        self,
        state: RobotCommandState,
    ) -> None:
        with self._lock:
            self.robots[
                state.robot_id
            ] = state

    # --------------------------------------------------------
    # CREATE COMMAND REQUEST
    # --------------------------------------------------------

    def create_request(
        self,
        *,
        robot_id: str,
        command_type: CommandType,
        requested_by: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> RobotCommandRequest:
        with self._lock:
            if (
                len(self.requests)
                >= self.config.maximum_pending_commands
            ):
                raise CommandStateError(
                    "Command storage capacity reached"
                )

            now = utc_now()

            request = RobotCommandRequest(
                command_id=str(uuid.uuid4()),
                robot_id=require_text(
                    robot_id,
                    "robot_id",
                ),
                command_type=CommandType(
                    command_type
                ),
                requested_by=require_text(
                    requested_by,
                    "requested_by",
                ),
                parameters=json_copy(
                    dict(parameters or {})
                ),
                created_at=now.isoformat(),
                expires_at=(
                    now + timedelta(
                        seconds=self.config.command_ttl_seconds
                    )
                ).isoformat(),
            )

            self.requests[
                request.command_id
            ] = request

            self._audit(
                "command_requested",
                request.command_id,
                {
                    "robot_id": robot_id,
                    "command_type": (
                        request.command_type.value
                    ),
                    "requested_by": requested_by,
                },
            )

            return request

    # --------------------------------------------------------
    # AUTHORIZE COMMAND
    # --------------------------------------------------------

    def authorize(
        self,
        command_id: str,
    ) -> CommandAuthorizationDecision:
        with self._lock:
            request = self.requests.get(
                command_id
            )

            if request is None:
                raise CommandNotFoundError(
                    "Command not found"
                )

            existing = self.decisions.get(
                command_id
            )

            if existing is not None:
                return existing

            principal = self.principals.get(
                request.requested_by
            )

            robot = self.robots.get(
                request.robot_id
            )

            if principal is None:
                reason = AuthorizationReason.INVALID_ROLE

            elif robot is None:
                reason = AuthorizationReason.ROBOT_UNAVAILABLE

            elif utc_now() >= parse_time(
                request.expires_at
            ):
                reason = AuthorizationReason.EXPIRED

            else:
                reason = self.policy.evaluate(
                    principal,
                    request,
                    robot,
                )

            if reason != AuthorizationReason.ALLOWED:
                decision = CommandAuthorizationDecision(
                    command_id=command_id,
                    status=AuthorizationStatus.DENIED,
                    reason=reason,
                )

            else:
                requires_approval = (
                    self.config.require_human_approval
                    and request.command_type
                    in SENSITIVE_COMMANDS
                )

                required = (
                    2
                    if (
                        requires_approval
                        and self.config.require_two_person_approval
                    )
                    else 1
                    if requires_approval
                    else 0
                )

                decision = CommandAuthorizationDecision(
                    command_id=command_id,
                    status=(
                        AuthorizationStatus.PENDING_APPROVAL
                        if required
                        else AuthorizationStatus.APPROVED
                    ),
                    reason=(
                        AuthorizationReason.APPROVAL_REQUIRED
                        if required
                        else AuthorizationReason.ALLOWED
                    ),
                    required_approvals=required,
                )

            self.decisions[command_id] = decision

            self._audit(
                "command_authorization",
                command_id,
                decision.to_dict(),
            )

            return decision

    # --------------------------------------------------------
    # APPROVE COMMAND
    # --------------------------------------------------------

    def approve(
        self,
        command_id: str,
        *,
        approver_id: str,
    ) -> CommandAuthorizationDecision:
        approver_id = require_text(
            approver_id,
            "approver_id",
        )

        with self._lock:
            decision = self.decisions.get(
                command_id
            )

            request = self.requests.get(
                command_id
            )

            if decision is None or request is None:
                raise CommandNotFoundError(
                    "Authorization decision not found"
                )

            if (
                decision.status
                != AuthorizationStatus.PENDING_APPROVAL
            ):
                raise CommandStateError(
                    "Command is not awaiting approval"
                )

            if utc_now() >= parse_time(
                request.expires_at
            ):
                decision.status = AuthorizationStatus.EXPIRED
                decision.reason = AuthorizationReason.EXPIRED
                decision.updated_at = utc_iso()

                raise CommandStateError(
                    "Command expired"
                )

            approver = self.principals.get(
                approver_id
            )

            if (
                approver is None
                or not approver.active
                or approver.role not in (
                    CommandRole.ADMIN,
                    CommandRole.ENGINEER,
                )
            ):
                raise CommandDeniedError(
                    "Approver is not authorized"
                )

            if not approver.can_access_robot(
                request.robot_id
            ):
                raise CommandDeniedError(
                    "Approver lacks robot scope"
                )

            if approver_id == request.requested_by:
                raise CommandDeniedError(
                    "Requester cannot self-approve"
                )

            if approver_id in decision.approvals:
                raise CommandStateError(
                    "Duplicate approval"
                )

            decision.approvals.add(
                approver_id
            )

            if (
                len(decision.approvals)
                >= decision.required_approvals
            ):
                decision.status = (
                    AuthorizationStatus.APPROVED
                )

                decision.reason = (
                    AuthorizationReason.ALLOWED
                )

            decision.updated_at = utc_iso()

            self._audit(
                "command_approved",
                command_id,
                {
                    "approver_id": approver_id,
                    "approval_count": len(
                        decision.approvals
                    ),
                    "status": decision.status.value,
                },
            )

            return decision

    # --------------------------------------------------------
    # DENY COMMAND
    # --------------------------------------------------------

    def deny(
        self,
        command_id: str,
        *,
        approver_id: str,
        reason: str,
    ) -> CommandAuthorizationDecision:
        with self._lock:
            decision = self.decisions.get(
                command_id
            )

            request = self.requests.get(
                command_id
            )

            if decision is None or request is None:
                raise CommandNotFoundError(
                    "Command not found"
                )

            if (
                decision.status
                != AuthorizationStatus.PENDING_APPROVAL
            ):
                raise CommandStateError(
                    "Command cannot be denied in this state"
                )

            approver = self.principals.get(
                approver_id
            )

            if (
                approver is None
                or not approver.active
                or approver.role not in (
                    CommandRole.ADMIN,
                    CommandRole.ENGINEER,
                )
                or not approver.can_access_robot(
                    request.robot_id
                )
            ):
                raise CommandDeniedError(
                    "Approver is not authorized"
                )

            decision.status = (
                AuthorizationStatus.DENIED
            )

            decision.reason = (
                AuthorizationReason.APPROVAL_DENIED
            )

            decision.updated_at = utc_iso()

            decision.details["denial_reason"] = (
                require_text(reason, "reason")
            )

            self._audit(
                "command_denied",
                command_id,
                {
                    "approver_id": approver_id,
                    "reason": reason,
                },
            )

            return decision

    # --------------------------------------------------------
    # CONSUME AUTHORIZATION
    # --------------------------------------------------------

    def consume_authorization(
        self,
        command_id: str,
    ) -> RobotCommandRequest:
        """
        Marks an approved authorization as consumed.

        Does not execute the command.
        """

        with self._lock:
            request = self.requests.get(
                command_id
            )

            decision = self.decisions.get(
                command_id
            )

            if request is None or decision is None:
                raise CommandNotFoundError(
                    "Command not found"
                )

            if (
                decision.status
                == AuthorizationStatus.CONSUMED
            ):
                raise CommandDeniedError(
                    "Replay detected"
                )

            if (
                decision.status
                != AuthorizationStatus.APPROVED
            ):
                raise CommandDeniedError(
                    "Command is not approved"
                )

            if utc_now() >= parse_time(
                request.expires_at
            ):
                decision.status = AuthorizationStatus.EXPIRED
                decision.reason = AuthorizationReason.EXPIRED
                decision.updated_at = utc_iso()

                raise CommandDeniedError(
                    "Command expired"
                )

            principal = self.principals.get(
                request.requested_by
            )

            robot = self.robots.get(
                request.robot_id
            )

            if principal is None or robot is None:
                raise CommandDeniedError(
                    "Principal or robot unavailable"
                )

            current_reason = self.policy.evaluate(
                principal,
                request,
                robot,
            )

            if current_reason != AuthorizationReason.ALLOWED:
                decision.status = AuthorizationStatus.DENIED
                decision.reason = current_reason
                decision.updated_at = utc_iso()

                self._audit(
                    "command_revalidation_denied",
                    command_id,
                    {"reason": current_reason.value},
                )

                raise CommandDeniedError(
                    f"Command denied: {current_reason.value}"
                )

            decision.status = (
                AuthorizationStatus.CONSUMED
            )

            decision.updated_at = utc_iso()

            self._audit(
                "authorization_consumed",
                command_id,
            )

            return request

    # --------------------------------------------------------
    # GET DECISION
    # --------------------------------------------------------

    def get_decision(
        self,
        command_id: str,
    ) -> CommandAuthorizationDecision:
        with self._lock:
            decision = self.decisions.get(
                command_id
            )

            if decision is None:
                raise CommandNotFoundError(
                    "Decision not found"
                )

            return decision

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        with self._lock:
            counts = {
                status.value: 0
                for status in AuthorizationStatus
            }

            for decision in self.decisions.values():
                counts[
                    decision.status.value
                ] += 1

            return {
                "registered_principals": len(
                    self.principals
                ),
                "registered_robots": len(
                    self.robots
                ),
                "command_requests": len(
                    self.requests
                ),
                "authorization_status": counts,
                "audit_events": len(
                    self.audit_events
                ),
                "hardware_control_enabled": False,
            }

    # --------------------------------------------------------
    # EXPORT REPORT
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
                "commands": [
                    request.to_dict()
                    for request in self.requests.values()
                ],
                "decisions": [
                    decision.to_dict()
                    for decision in self.decisions.values()
                ],
                "audit_events": list(
                    self.audit_events
                ),
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
    service = RoboticsCommandAuthorizationService()

    service.register_principal(
        CommandPrincipal(
            principal_id="operator_001",
            role=CommandRole.OPERATOR,
            allowed_robot_ids=frozenset({
                "robot_001"
            }),
        )
    )

    service.register_principal(
        CommandPrincipal(
            principal_id="engineer_001",
            role=CommandRole.ENGINEER,
            allowed_robot_ids=frozenset({
                "robot_001"
            }),
        )
    )

    service.register_principal(
        CommandPrincipal(
            principal_id="engineer_002",
            role=CommandRole.ENGINEER,
            allowed_robot_ids=frozenset({
                "robot_001"
            }),
        )
    )

    service.update_robot_state(
        RobotCommandState(
            robot_id="robot_001",
            mode=RobotOperatingMode.SIMULATION,
            battery_soc_percent=91.0,
            health_score=95.0,
            emergency_stop_active=False,
            safety_interlock_active=False,
            sensor_available=True,
            network_connected=True,
        )
    )

    request = service.create_request(
        robot_id="robot_001",
        command_type=CommandType.NAVIGATE,
        requested_by="operator_001",
        parameters={
            "target_x_m": 5.0,
            "target_y_m": 8.0,
        },
    )

    decision = service.authorize(
        request.command_id
    )

    print(
        "Initial decision:",
        decision.status.value,
    )

    service.approve(
        request.command_id,
        approver_id="engineer_001",
    )

    service.approve(
        request.command_id,
        approver_id="engineer_002",
    )

    authorized_request = (
        service.consume_authorization(
            request.command_id
        )
    )

    print(
        "Authorized simulation command:",
        authorized_request.command_type.value,
    )

    report_path = service.export_json(
        "data/security/command_authorization.json"
    )

    print(
        json.dumps(
            {
                "statistics": service.statistics(),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

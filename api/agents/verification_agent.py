"""Verification Agent module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/tool_executor.py

Physical AI & Autonomous Robotics Intelligence Platform
Secure Multi-Agent Tool Execution Engine

Python 3.11+

Features:
- Tool registration and discovery
- Async and sync tool execution
- Tenant-aware authorization
- Tool input and output validation
- Role and capability-based access control
- Read-only, advisory and simulation tool support
- Human approval integration
- Execution timeouts
- Bounded retries
- Parallel and batch execution
- Tool execution audit logging
- Structured results and error handling
- Tool execution statistics
- Idempotency protection
- Sensitive-data redaction

Safety:
- No direct physical robot actuation
- No emergency-stop resets
- No safety-interlock bypass
- No unrestricted shell execution
- No arbitrary Python code execution
- No dynamic imports from tool requests
- Tools are registered by trusted application code
- Human approval does not authorize physical actuation
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import math
import re
import time
import uuid

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Mapping,
    Protocol,
    Sequence,
)


logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

MAX_REGISTERED_TOOLS = 500
MAX_BATCH_TOOL_CALLS = 50
MAX_CONCURRENT_TOOL_CALLS = 10
MAX_TOOL_RETRIES = 3

DEFAULT_TOOL_TIMEOUT_SECONDS = 60.0
MAX_TOOL_TIMEOUT_SECONDS = 600.0

MAX_TOOL_NAME_LENGTH = 128
MAX_TOOL_DESCRIPTION_LENGTH = 2000
MAX_TOOL_ARGUMENT_KEYS = 100
MAX_TOOL_OUTPUT_KEYS = 200

TOOL_NAME_PATTERN = re.compile(
    r"^[a-z][a-z0-9_.-]{0,127}$"
)

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)

SENSITIVE_FIELD_PATTERN = re.compile(
    r"(password|secret|token|api[_-]?key|"
    r"authorization|credential|private[_-]?key)",
    re.IGNORECASE,
)


# ============================================================
# Exceptions
# ============================================================

class ToolExecutorError(Exception):
    """Base tool executor exception."""


class ToolValidationError(ToolExecutorError):
    """Invalid tool input or output."""


class ToolNotFoundError(ToolExecutorError):
    """Requested tool does not exist."""


class ToolAlreadyRegisteredError(ToolExecutorError):
    """Tool name already registered."""


class ToolPermissionError(ToolExecutorError):
    """Tool execution permission denied."""


class ToolSafetyError(ToolExecutorError):
    """Unsafe tool operation requested."""


class ToolApprovalError(ToolExecutorError):
    """Required human approval is unavailable."""


class ToolExecutionError(ToolExecutorError):
    """Tool execution failed."""


class ToolTimeoutError(ToolExecutorError):
    """Tool execution exceeded timeout."""


class ToolUnavailableError(ToolExecutorError):
    """Tool is disabled or unavailable."""


class ToolConcurrencyError(ToolExecutorError):
    """Tool concurrency limit exceeded."""


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
        raise ToolValidationError(
            f"Invalid identifier: {value!r}"
        )
    return value


def validate_tool_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not TOOL_NAME_PATTERN.fullmatch(value)
    ):
        raise ToolValidationError(
            f"Invalid tool name: {value!r}"
        )
    return value


def validate_positive(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ToolValidationError(
            "Value must be finite and positive"
        )
    return float(value)


def validate_nonnegative(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ToolValidationError(
            "Value must be finite and nonnegative"
        )
    return float(value)


def redact_sensitive_data(
    value: Any,
    *,
    depth: int = 0,
) -> Any:
    """Redact sensitive mapping fields for audit-safe output."""

    if depth > 12:
        return "[MAX_DEPTH]"

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}

        for key, item in value.items():
            string_key = str(key)

            if SENSITIVE_FIELD_PATTERN.search(string_key):
                result[string_key] = "[REDACTED]"
            else:
                result[string_key] = redact_sensitive_data(
                    item,
                    depth=depth + 1,
                )

        return result

    if isinstance(value, (list, tuple)):
        return [
            redact_sensitive_data(
                item,
                depth=depth + 1,
            )
            for item in value
        ]

    if isinstance(value, (str, int, float, bool)):
        return value

    if value is None:
        return None

    return f"[{type(value).__name__}]"


# ============================================================
# Enumerations
# ============================================================

class ToolCategory(str, Enum):
    GENERAL = "general"
    RESEARCH = "research"
    RAG = "rag"
    MEMORY = "memory"
    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    SAFETY = "safety"
    ROBOT_HEALTH = "robot_health"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    COST_ESTIMATION = "cost_estimation"
    SIMULATION = "simulation"
    DIGITAL_TWIN = "digital_twin"
    WORLD_MODEL = "world_model"
    ANALYTICS = "analytics"
    EVALUATION = "evaluation"
    REPORTING = "reporting"
    FILES = "files"
    DATABASE = "database"
    CUSTOM = "custom"


class ToolSafetyClass(str, Enum):
    READ_ONLY = "read_only"
    ADVISORY = "advisory"
    SIMULATION = "simulation"
    CONTROLLED_WRITE = "controlled_write"
    PHYSICAL_ACTUATION = "physical_actuation"
    PROHIBITED = "prohibited"


class ToolExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"
    APPROVAL_REQUIRED = "approval_required"
    CANCELLED = "cancelled"


class ToolAvailability(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    MAINTENANCE = "maintenance"


class ToolApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ToolErrorCode(str, Enum):
    NONE = "none"
    INVALID_INPUT = "invalid_input"
    INVALID_OUTPUT = "invalid_output"
    TOOL_NOT_FOUND = "tool_not_found"
    PERMISSION_DENIED = "permission_denied"
    SAFETY_REJECTED = "safety_rejected"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_REJECTED = "approval_rejected"
    TIMEOUT = "timeout"
    EXECUTION_FAILED = "execution_failed"
    TOOL_UNAVAILABLE = "tool_unavailable"
    INTERNAL_ERROR = "internal_error"


# ============================================================
# Execution Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class ToolExecutionScope:
    tenant_id: str
    user_id: str

    agent_id: str | None = None
    workflow_id: str | None = None
    task_id: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)

        for identifier in (
            self.agent_id,
            self.workflow_id,
            self.task_id,
        ):
            if identifier is not None:
                validate_identifier(identifier)


# ============================================================
# Tool Definition
# ============================================================

@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str

    category: ToolCategory

    safety_class: ToolSafetyClass = (
        ToolSafetyClass.READ_ONLY
    )

    required_permissions: frozenset[str] = frozenset()
    required_capabilities: frozenset[str] = frozenset()

    allowed_agent_ids: frozenset[str] = frozenset()

    requires_human_approval: bool = False

    timeout_seconds: float = (
        DEFAULT_TOOL_TIMEOUT_SECONDS
    )

    max_retries: int = 0

    maximum_concurrent_calls: int = 3

    availability: ToolAvailability = (
        ToolAvailability.ENABLED
    )

    idempotent: bool = True

    input_schema: Mapping[str, Any] = field(
        default_factory=dict
    )

    output_schema: Mapping[str, Any] = field(
        default_factory=dict
    )

    version: str = "1.0.0"

    def __post_init__(self) -> None:
        validate_tool_name(self.name)

        if (
            not self.description.strip()
            or len(self.description)
            > MAX_TOOL_DESCRIPTION_LENGTH
        ):
            raise ToolValidationError(
                "Invalid tool description"
            )

        validate_positive(self.timeout_seconds)

        if (
            self.timeout_seconds
            > MAX_TOOL_TIMEOUT_SECONDS
        ):
            raise ToolValidationError(
                "Tool timeout exceeds limit"
            )

        if not 0 <= self.max_retries <= MAX_TOOL_RETRIES:
            raise ToolValidationError(
                "Invalid tool retry count"
            )

        if not (
            1
            <= self.maximum_concurrent_calls
            <= MAX_CONCURRENT_TOOL_CALLS
        ):
            raise ToolValidationError(
                "Invalid tool concurrency"
            )

        if self.safety_class in (
            ToolSafetyClass.PHYSICAL_ACTUATION,
            ToolSafetyClass.PROHIBITED,
        ):
            raise ToolSafetyError(
                "Physical actuation and prohibited "
                "tools cannot be registered"
            )

        if (
            self.safety_class
            == ToolSafetyClass.CONTROLLED_WRITE
            and not self.requires_human_approval
        ):
            raise ToolSafetyError(
                "Controlled-write tools require "
                "human approval"
            )

        if not self.idempotent and self.max_retries:
            raise ToolValidationError(
                "Non-idempotent tools cannot "
                "automatically retry"
            )

        for agent_id in self.allowed_agent_ids:
            validate_identifier(agent_id)


# ============================================================
# Tool Request
# ============================================================

@dataclass(frozen=True, slots=True)
class ToolExecutionRequest:
    request_id: str

    tool_name: str

    arguments: Mapping[str, Any] = field(
        default_factory=dict
    )

    idempotency_key: str | None = None

    approval_id: str | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(self.request_id)
        validate_tool_name(self.tool_name)

        if len(self.arguments) > MAX_TOOL_ARGUMENT_KEYS:
            raise ToolValidationError(
                "Too many tool arguments"
            )

        if self.idempotency_key is not None:
            validate_identifier(self.idempotency_key)

        if self.approval_id is not None:
            validate_identifier(self.approval_id)


# ============================================================
# Tool Context
# ============================================================

@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    execution_id: str

    tenant_id: str
    user_id: str

    agent_id: str | None

    workflow_id: str | None
    task_id: str | None

    tool_name: str

    arguments: Mapping[str, Any]

    simulation_only: bool = True

    def __post_init__(self) -> None:
        validate_identifier(self.execution_id)
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)

        if not self.simulation_only:
            raise ToolSafetyError(
                "Physical execution is not authorized"
            )


# ============================================================
# Tool Result
# ============================================================

@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    execution_id: str
    request_id: str

    tool_name: str
    tenant_id: str

    status: ToolExecutionStatus

    output: Mapping[str, Any] | None

    error_code: ToolErrorCode
    error_message: str | None

    attempts: int

    duration_seconds: float

    approval_status: ToolApprovalStatus

    started_at: datetime
    completed_at: datetime

    physical_execution_authorized: bool = False

    def __post_init__(self) -> None:
        if self.physical_execution_authorized:
            raise ToolSafetyError(
                "Tool executor cannot authorize "
                "physical robot execution"
            )


# ============================================================
# Statistics
# ============================================================

@dataclass(frozen=True, slots=True)
class ToolExecutionStatistics:
    tool_name: str

    total_calls: int
    successful_calls: int
    failed_calls: int
    rejected_calls: int
    timed_out_calls: int

    average_duration_seconds: float
    success_rate: float


# ============================================================
# Handler Type
# ============================================================

ToolHandler = Callable[
    [ToolExecutionContext],
    Mapping[str, Any]
    | Awaitable[Mapping[str, Any]],
]


# ============================================================
# Provider Protocols
# ============================================================

class ToolAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        ...


class ToolApprovalProvider(Protocol):

    async def get_approval_status(
        self,
        *,
        tenant_id: str,
        approval_id: str,
        tool_name: str,
        request_id: str,
    ) -> ToolApprovalStatus:
        ...


class ToolInputValidator(Protocol):

    async def validate_input(
        self,
        *,
        definition: ToolDefinition,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


class ToolOutputValidator(Protocol):

    async def validate_output(
        self,
        *,
        definition: ToolDefinition,
        output: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


class ToolAuditSink(Protocol):

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


class ToolResultRepository(Protocol):

    async def save_result(
        self,
        *,
        result: ToolExecutionResult,
    ) -> None:
        ...


# ============================================================
# Registered Tool
# ============================================================

@dataclass(slots=True)
class RegisteredTool:
    definition: ToolDefinition
    handler: ToolHandler
    semaphore: asyncio.Semaphore


# ============================================================
# Tool Registry
# ============================================================

class ToolRegistry:
    """
    Trusted in-process tool registry.

    Tool handlers must be registered by application code.
    User requests cannot register arbitrary callables.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        *,
        definition: ToolDefinition,
        handler: ToolHandler,
    ) -> None:

        if definition.name in self._tools:
            raise ToolAlreadyRegisteredError(
                f"Tool already registered: {definition.name}"
            )

        if len(self._tools) >= MAX_REGISTERED_TOOLS:
            raise ToolValidationError(
                "Tool registry capacity exceeded"
            )

        if not callable(handler):
            raise ToolValidationError(
                "Tool handler must be callable"
            )

        self._tools[definition.name] = RegisteredTool(
            definition=definition,
            handler=handler,
            semaphore=asyncio.Semaphore(
                definition.maximum_concurrent_calls
            ),
        )

    def get(
        self,
        tool_name: str,
    ) -> RegisteredTool:

        validate_tool_name(tool_name)

        tool = self._tools.get(tool_name)

        if tool is None:
            raise ToolNotFoundError(
                f"Tool not found: {tool_name}"
            )

        return tool

    def list_tools(
        self,
        *,
        category: ToolCategory | None = None,
        enabled_only: bool = True,
    ) -> list[ToolDefinition]:

        definitions = []

        for registered in self._tools.values():
            definition = registered.definition

            if (
                category is not None
                and definition.category != category
            ):
                continue

            if (
                enabled_only
                and definition.availability
                != ToolAvailability.ENABLED
            ):
                continue

            definitions.append(definition)

        return sorted(
            definitions,
            key=lambda item: item.name,
        )

    def unregister(
        self,
        tool_name: str,
    ) -> None:
        validate_tool_name(tool_name)

        if tool_name not in self._tools:
            raise ToolNotFoundError(
                f"Tool not found: {tool_name}"
            )

        del self._tools[tool_name]


# ============================================================
# Default Input Validator
# ============================================================

class DefaultToolInputValidator:
    """
    Minimal schema validator.

    Supported schema:
    {
        "required": ["robot_id"],
        "properties": {
            "robot_id": {"type": "string"},
            "limit": {"type": "integer"}
        },
        "additionalProperties": False
    }

    For production, replace with a full JSON Schema
    validator if more complex schemas are required.
    """

    TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    async def validate_input(
        self,
        *,
        definition: ToolDefinition,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:

        schema = definition.input_schema

        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field_name in required:
            if field_name not in arguments:
                raise ToolValidationError(
                    f"Missing required argument: {field_name}"
                )

        if schema.get(
            "additionalProperties"
        ) is False:
            unknown = set(arguments) - set(properties)

            if unknown:
                raise ToolValidationError(
                    "Unexpected tool arguments: "
                    + ", ".join(sorted(unknown))
                )

        for name, value in arguments.items():
            field_schema = properties.get(name)

            if field_schema is None:
                continue

            expected_type = field_schema.get("type")

            if expected_type is None:
                continue

            python_type = self.TYPE_MAP.get(
                expected_type
            )

            if python_type is None:
                raise ToolValidationError(
                    f"Unsupported schema type: {expected_type}"
                )

            if expected_type == "integer":
                valid = (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                )

            elif expected_type == "number":
                valid = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                )

            else:
                valid = isinstance(
                    value,
                    python_type,
                )

            if not valid:
                raise ToolValidationError(
                    f"Invalid type for argument: {name}"
                )

        return copy.deepcopy(dict(arguments))


# ============================================================
# Default Output Validator
# ============================================================

class DefaultToolOutputValidator:

    async def validate_output(
        self,
        *,
        definition: ToolDefinition,
        output: Mapping[str, Any],
    ) -> Mapping[str, Any]:

        if not isinstance(output, Mapping):
            raise ToolValidationError(
                "Tool output must be a mapping"
            )

        if len(output) > MAX_TOOL_OUTPUT_KEYS:
            raise ToolValidationError(
                "Tool output contains too many fields"
            )

        prohibited_true_fields = (
            "physical_execution_authorized",
            "emergency_stop_reset",
            "safety_interlock_bypassed",
            "robot_actuated",
        )

        for field_name in prohibited_true_fields:
            if output.get(field_name) is True:
                raise ToolSafetyError(
                    f"Prohibited tool output: {field_name}"
                )

        return copy.deepcopy(dict(output))


# ============================================================
# Tool Executor
# ============================================================

class ToolExecutor:

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        authorizer: ToolAuthorizer,
        approval_provider: ToolApprovalProvider | None = None,
        input_validator: ToolInputValidator | None = None,
        output_validator: ToolOutputValidator | None = None,
        audit_sink: ToolAuditSink | None = None,
        repository: ToolResultRepository | None = None,
        max_concurrency: int = 5,
    ) -> None:

        if not (
            1
            <= max_concurrency
            <= MAX_CONCURRENT_TOOL_CALLS
        ):
            raise ToolValidationError(
                "Invalid executor concurrency"
            )

        self.registry = registry
        self.authorizer = authorizer

        self.approval_provider = approval_provider

        self.input_validator = (
            input_validator
            or DefaultToolInputValidator()
        )

        self.output_validator = (
            output_validator
            or DefaultToolOutputValidator()
        )

        self.audit_sink = audit_sink
        self.repository = repository

        self._global_semaphore = asyncio.Semaphore(
            max_concurrency
        )

        self._statistics: dict[
            str,
            list[ToolExecutionResult],
        ] = defaultdict(list)

        self._idempotency_results: dict[
            tuple[str, str, str],
            ToolExecutionResult,
        ] = {}

        self._idempotency_locks: dict[
            tuple[str, str, str],
            asyncio.Lock,
        ] = {}

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    async def _authorize(
        self,
        *,
        scope: ToolExecutionScope,
        definition: ToolDefinition,
    ) -> None:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            permission="tools:execute",
        )

        for permission in sorted(
            definition.required_permissions
        ):
            await self.authorizer.require_permission(
                tenant_id=scope.tenant_id,
                user_id=scope.user_id,
                permission=permission,
            )

        if definition.allowed_agent_ids:
            if (
                scope.agent_id
                not in definition.allowed_agent_ids
            ):
                raise ToolPermissionError(
                    "Agent is not permitted to use this tool"
                )

    # --------------------------------------------------------
    # Safety Validation
    # --------------------------------------------------------

    @staticmethod
    def _validate_safety(
        definition: ToolDefinition,
    ) -> None:

        if definition.safety_class in (
            ToolSafetyClass.PHYSICAL_ACTUATION,
            ToolSafetyClass.PROHIBITED,
        ):
            raise ToolSafetyError(
                "Tool safety class is prohibited"
            )

        if definition.availability != (
            ToolAvailability.ENABLED
        ):
            raise ToolUnavailableError(
                "Tool is not enabled"
            )

    # --------------------------------------------------------
    # Approval Validation
    # --------------------------------------------------------

    async def _check_approval(
        self,
        *,
        scope: ToolExecutionScope,
        definition: ToolDefinition,
        request: ToolExecutionRequest,
    ) -> ToolApprovalStatus:

        if not definition.requires_human_approval:
            return ToolApprovalStatus.NOT_REQUIRED

        if self.approval_provider is None:
            raise ToolApprovalError(
                "Human approval provider unavailable"
            )

        if request.approval_id is None:
            return ToolApprovalStatus.PENDING

        status = (
            await self.approval_provider.get_approval_status(
                tenant_id=scope.tenant_id,
                approval_id=request.approval_id,
                tool_name=definition.name,
                request_id=request.request_id,
            )
        )

        return status

    # --------------------------------------------------------
    # Handler Invocation
    # --------------------------------------------------------

    @staticmethod
    async def _invoke_handler(
        *,
        handler: ToolHandler,
        context: ToolExecutionContext,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:

        async def invoke() -> Mapping[str, Any]:
            if inspect.iscoroutinefunction(handler):
                return await handler(context)

            result = await asyncio.to_thread(
                handler,
                context,
            )

            if inspect.isawaitable(result):
                return await result

            return result

        return await asyncio.wait_for(
            invoke(),
            timeout=timeout_seconds,
        )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    async def _audit(
        self,
        *,
        scope: ToolExecutionScope,
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
            metadata=redact_sensitive_data(
                metadata
            ),
        )

    # --------------------------------------------------------
    # Store Result
    # --------------------------------------------------------

    async def _store_result(
        self,
        result: ToolExecutionResult,
    ) -> None:

        self._statistics[result.tool_name].append(
            result
        )

        if self.repository is not None:
            await self.repository.save_result(
                result=result
            )

    # --------------------------------------------------------
    # Result Builder
    # --------------------------------------------------------

    @staticmethod
    def _build_result(
        *,
        execution_id: str,
        request: ToolExecutionRequest,
        scope: ToolExecutionScope,
        status: ToolExecutionStatus,
        output: Mapping[str, Any] | None,
        error_code: ToolErrorCode,
        error_message: str | None,
        attempts: int,
        started_at: datetime,
        started_monotonic: float,
        approval_status: ToolApprovalStatus,
    ) -> ToolExecutionResult:

        return ToolExecutionResult(
            execution_id=execution_id,
            request_id=request.request_id,
            tool_name=request.tool_name,
            tenant_id=scope.tenant_id,
            status=status,
            output=output,
            error_code=error_code,
            error_message=error_message,
            attempts=attempts,
            duration_seconds=max(
                0.0,
                time.monotonic() - started_monotonic,
            ),
            approval_status=approval_status,
            started_at=started_at,
            completed_at=utc_now(),
            physical_execution_authorized=False,
        )

    # --------------------------------------------------------
    # Execute Tool
    # --------------------------------------------------------

    async def execute(
        self,
        *,
        scope: ToolExecutionScope,
        request: ToolExecutionRequest,
    ) -> ToolExecutionResult:

        if request.idempotency_key is None:
            return await self._execute_internal(
                scope=scope,
                request=request,
            )

        key = (
            scope.tenant_id,
            request.tool_name,
            request.idempotency_key,
        )

        lock = self._idempotency_locks.setdefault(
            key,
            asyncio.Lock(),
        )

        async with lock:
            cached = self._idempotency_results.get(
                key
            )

            if cached is not None:
                return cached

            result = await self._execute_internal(
                scope=scope,
                request=request,
            )

            if result.status == (
                ToolExecutionStatus.COMPLETED
            ):
                self._idempotency_results[key] = result

            return result

    async def _execute_internal(
        self,
        *,
        scope: ToolExecutionScope,
        request: ToolExecutionRequest,
    ) -> ToolExecutionResult:

        execution_id = generate_id()
        started_at = utc_now()
        started_monotonic = time.monotonic()

        attempts = 0

        approval_status = (
            ToolApprovalStatus.NOT_REQUIRED
        )

        try:
            registered = self.registry.get(
                request.tool_name
            )

            definition = registered.definition

            self._validate_safety(definition)

            await self._authorize(
                scope=scope,
                definition=definition,
            )

            approval_status = (
                await self._check_approval(
                    scope=scope,
                    definition=definition,
                    request=request,
                )
            )

            if approval_status == (
                ToolApprovalStatus.PENDING
            ):
                result = self._build_result(
                    execution_id=execution_id,
                    request=request,
                    scope=scope,
                    status=(
                        ToolExecutionStatus.APPROVAL_REQUIRED
                    ),
                    output=None,
                    error_code=(
                        ToolErrorCode.APPROVAL_REQUIRED
                    ),
                    error_message=(
                        "Human approval is required"
                    ),
                    attempts=0,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    approval_status=approval_status,
                )

                await self._store_result(result)
                return result

            if approval_status in (
                ToolApprovalStatus.REJECTED,
                ToolApprovalStatus.EXPIRED,
            ):
                raise ToolApprovalError(
                    "Human approval rejected or expired"
                )

            validated_arguments = (
                await self.input_validator.validate_input(
                    definition=definition,
                    arguments=request.arguments,
                )
            )

            context = ToolExecutionContext(
                execution_id=execution_id,
                tenant_id=scope.tenant_id,
                user_id=scope.user_id,
                agent_id=scope.agent_id,
                workflow_id=scope.workflow_id,
                task_id=scope.task_id,
                tool_name=request.tool_name,
                arguments=validated_arguments,
                simulation_only=True,
            )

            output: Mapping[str, Any] | None = None
            last_error: Exception | None = None

            async with self._global_semaphore:
                async with registered.semaphore:

                    for attempt in range(
                        definition.max_retries + 1
                    ):
                        attempts += 1

                        try:
                            raw_output = (
                                await self._invoke_handler(
                                    handler=registered.handler,
                                    context=context,
                                    timeout_seconds=(
                                        definition.timeout_seconds
                                    ),
                                )
                            )

                            output = (
                                await self.output_validator.validate_output(
                                    definition=definition,
                                    output=raw_output,
                                )
                            )

                            last_error = None
                            break

                        except asyncio.CancelledError:
                            raise

                        except ToolSafetyError:
                            raise

                        except Exception as exc:
                            last_error = exc

                            logger.warning(
                                "Tool %s failed on attempt %d: %s",
                                request.tool_name,
                                attempts,
                                type(exc).__name__,
                            )

                            if attempt < definition.max_retries:
                                await asyncio.sleep(
                                    min(2 ** attempt, 8)
                                )

            if last_error is not None:
                if isinstance(
                    last_error,
                    asyncio.TimeoutError,
                ):
                    raise ToolTimeoutError(
                        "Tool execution timed out"
                    ) from last_error

                raise ToolExecutionError(
                    f"Tool failed after {attempts} attempt(s): "
                    f"{type(last_error).__name__}"
                ) from last_error

            result = self._build_result(
                execution_id=execution_id,
                request=request,
                scope=scope,
                status=ToolExecutionStatus.COMPLETED,
                output=output,
                error_code=ToolErrorCode.NONE,
                error_message=None,
                attempts=attempts,
                started_at=started_at,
                started_monotonic=started_monotonic,
                approval_status=approval_status,
            )

        except asyncio.CancelledError:
            raise

        except ToolNotFoundError as exc:
            result = self._error_result(
                execution_id=execution_id,
                request=request,
                scope=scope,
                status=ToolExecutionStatus.REJECTED,
                error_code=ToolErrorCode.TOOL_NOT_FOUND,
                error=exc,
                attempts=attempts,
                started_at=started_at,
                started_monotonic=started_monotonic,
                approval_status=approval_status,
            )

        except ToolPermissionError as exc:
            result = self._error_result(
                execution_id=execution_id,
                request=request,
                scope=scope,
                status=ToolExecutionStatus.REJECTED,
                error_code=ToolErrorCode.PERMISSION_DENIED,
                error=exc,
                attempts=attempts,
                started_at=started_at,
                started_monotonic=started_monotonic,
                approval_status=approval_status,
            )

        except ToolSafetyError as exc:
            result = self._error_result(
                execution_id=execution_id,
                request=request,
                scope=scope,
                status=ToolExecutionStatus.REJECTED,
                error_code=ToolErrorCode.SAFETY_REJECTED,
                error=exc,
                attempts=attempts,
                started_at=started_at,
                started_monotonic=started_monotonic,
                approval_status=approval_status,
            )

        except ToolApprovalError as exc:
            result = self._error_result(
                execution_id=execution_id,
                request=request,
                scope=scope,
                status=ToolExecutionStatus.REJECTED,
                error_code=ToolErrorCode.APPROVAL_REJECTED,
                error=exc,
                attempts=attempts,
                started_at=started_at,
                started_monotonic=started_monotonic,
                approval_status=approval_status,
            )

        except ToolValidationError as exc:
            result = self._error_result(
                execution_id=execution_id,
                request=request,
                scope=scope,
                status=ToolExecutionStatus.REJECTED,
                error_code=ToolErrorCode.INVALID_INPUT,
                error=exc,
                attempts=attempts,
                started_at=started_at,
                started_monotonic=started_monotonic,
                approval_status=approval_status,
            )

        except ToolUnavailableError as exc:
            result = self._error_result(
                execution_id=execution_id,
                request=request,
                scope=scope,
                status=ToolExecutionStatus.REJECTED,
                error_code=ToolErrorCode.TOOL_UNAVAILABLE,
                error=exc,
                attempts=attempts,
                started_at=started_at,
                started_monotonic=started_monotonic,
                approval_status=approval_status,
            )

        except ToolTimeoutError as exc:
            result = self._error_result(
                execution_id=execution_id,
                request=request,
                scope=scope,
                status=ToolExecutionStatus.TIMED_OUT,
                error_code=ToolErrorCode.TIMEOUT,
                error=exc,
                attempts=attempts,
                started_at=started_at,
                started_monotonic=started_monotonic,
                approval_status=approval_status,
            )

        except Exception as exc:
            logger.exception(
                "Unexpected tool execution failure"
            )

            result = self._error_result(
                execution_id=execution_id,
                request=request,
                scope=scope,
                status=ToolExecutionStatus.FAILED,
                error_code=ToolErrorCode.EXECUTION_FAILED,
                error=exc,
                attempts=attempts,
                started_at=started_at,
                started_monotonic=started_monotonic,
                approval_status=approval_status,
            )

        await self._store_result(result)

        await self._audit(
            scope=scope,
            action="tool_executor.execution_finished",
            resource_id=result.execution_id,
            metadata={
                "tool_name": result.tool_name,
                "request_id": result.request_id,
                "status": result.status.value,
                "error_code": result.error_code.value,
                "attempts": result.attempts,
                "duration_seconds": (
                    result.duration_seconds
                ),
                "physical_execution_authorized": False,
            },
        )

        return result

    # --------------------------------------------------------
    # Error Result
    # --------------------------------------------------------

    def _error_result(
        self,
        *,
        execution_id: str,
        request: ToolExecutionRequest,
        scope: ToolExecutionScope,
        status: ToolExecutionStatus,
        error_code: ToolErrorCode,
        error: Exception,
        attempts: int,
        started_at: datetime,
        started_monotonic: float,
        approval_status: ToolApprovalStatus,
    ) -> ToolExecutionResult:

        return self._build_result(
            execution_id=execution_id,
            request=request,
            scope=scope,
            status=status,
            output=None,
            error_code=error_code,
            error_message=(
                f"{type(error).__name__}: {error}"
            ),
            attempts=attempts,
            started_at=started_at,
            started_monotonic=started_monotonic,
            approval_status=approval_status,
        )

    # --------------------------------------------------------
    # Batch Execution
    # --------------------------------------------------------

    async def execute_batch(
        self,
        *,
        scope: ToolExecutionScope,
        requests: Sequence[ToolExecutionRequest],
    ) -> list[ToolExecutionResult]:

        if len(requests) > MAX_BATCH_TOOL_CALLS:
            raise ToolValidationError(
                "Batch tool call limit exceeded"
            )

        return list(
            await asyncio.gather(
                *(
                    self.execute(
                        scope=scope,
                        request=request,
                    )
                    for request in requests
                )
            )
        )

    # --------------------------------------------------------
    # Tool Discovery
    # --------------------------------------------------------

    async def discover_tools(
        self,
        *,
        scope: ToolExecutionScope,
        category: ToolCategory | None = None,
    ) -> list[dict[str, Any]]:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            permission="tools:read",
        )

        tools = self.registry.list_tools(
            category=category,
            enabled_only=True,
        )

        visible: list[dict[str, Any]] = []

        for definition in tools:
            if (
                definition.allowed_agent_ids
                and scope.agent_id
                not in definition.allowed_agent_ids
            ):
                continue

            visible.append(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "category": definition.category.value,
                    "safety_class": (
                        definition.safety_class.value
                    ),
                    "version": definition.version,
                    "requires_human_approval": (
                        definition.requires_human_approval
                    ),
                    "timeout_seconds": (
                        definition.timeout_seconds
                    ),
                    "input_schema": dict(
                        definition.input_schema
                    ),
                    "output_schema": dict(
                        definition.output_schema
                    ),
                }
            )

        return visible

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def get_statistics(
        self,
        *,
        tool_name: str,
    ) -> ToolExecutionStatistics:

        validate_tool_name(tool_name)

        results = self._statistics.get(
            tool_name,
            [],
        )

        total = len(results)

        successful = sum(
            result.status
            == ToolExecutionStatus.COMPLETED
            for result in results
        )

        failed = sum(
            result.status
            == ToolExecutionStatus.FAILED
            for result in results
        )

        rejected = sum(
            result.status
            in (
                ToolExecutionStatus.REJECTED,
                ToolExecutionStatus.APPROVAL_REQUIRED,
            )
            for result in results
        )

        timed_out = sum(
            result.status
            == ToolExecutionStatus.TIMED_OUT
            for result in results
        )

        average_duration = (
            sum(
                result.duration_seconds
                for result in results
            ) / total
            if total
            else 0.0
        )

        return ToolExecutionStatistics(
            tool_name=tool_name,
            total_calls=total,
            successful_calls=successful,
            failed_calls=failed,
            rejected_calls=rejected,
            timed_out_calls=timed_out,
            average_duration_seconds=round(
                average_duration,
                4,
            ),
            success_rate=round(
                successful / total if total else 0.0,
                4,
            ),
        )

    # --------------------------------------------------------
    # Execution Report
    # --------------------------------------------------------

    @staticmethod
    def generate_execution_report(
        result: ToolExecutionResult,
    ) -> dict[str, Any]:

        return {
            "execution_id": result.execution_id,
            "request_id": result.request_id,
            "tool_name": result.tool_name,
            "tenant_id": result.tenant_id,
            "status": result.status.value,
            "output": redact_sensitive_data(
                result.output
            ),
            "error": {
                "code": result.error_code.value,
                "message": result.error_message,
            },
            "attempts": result.attempts,
            "duration_seconds": (
                result.duration_seconds
            ),
            "approval_status": (
                result.approval_status.value
            ),
            "started_at": (
                result.started_at.isoformat()
            ),
            "completed_at": (
                result.completed_at.isoformat()
            ),
            "safety": {
                "physical_execution_authorized": False,
                "emergency_stop_reset_allowed": False,
                "safety_interlock_bypass_allowed": False,
                "arbitrary_code_execution_allowed": False,
                "trusted_tool_registration_required": True,
            },
        }


# ============================================================
# Built-in Read-Only Tool Handlers
# ============================================================

async def health_summary_tool(
    context: ToolExecutionContext,
) -> Mapping[str, Any]:
    """
    Example placeholder handler.

    Replace with a trusted health service adapter.
    Do not return fabricated telemetry.
    """

    robot_id = context.arguments.get(
        "robot_id"
    )

    if not isinstance(robot_id, str):
        raise ToolValidationError(
            "robot_id must be a string"
        )

    validate_identifier(robot_id)

    return {
        "robot_id": robot_id,
        "status": "service_not_configured",
        "message": (
            "Connect a trusted robot health service "
            "to retrieve actual telemetry."
        ),
        "physical_execution_authorized": False,
    }


async def simulation_info_tool(
    context: ToolExecutionContext,
) -> Mapping[str, Any]:

    return {
        "simulation_only": True,
        "physical_execution_authorized": False,
        "supported_operations": [
            "scenario_validation",
            "offline_simulation",
            "monte_carlo_evaluation",
            "simulation_report",
        ],
        "message": (
            "Use the configured SimulationAgent "
            "for actual scenario execution."
        ),
    }


async def safety_policy_info_tool(
    context: ToolExecutionContext,
) -> Mapping[str, Any]:

    return {
        "safety_policy": {
            "physical_actuation_allowed": False,
            "emergency_stop_reset_allowed": False,
            "safety_interlock_bypass_allowed": False,
            "human_review_for_critical_tasks": True,
            "independent_safety_validation_required": True,
        },
        "physical_execution_authorized": False,
    }


# ============================================================
# Default Registry Builder
# ============================================================

def build_default_tool_registry() -> ToolRegistry:

    registry = ToolRegistry()

    registry.register(
        definition=ToolDefinition(
            name="robot.health.summary",
            description=(
                "Read-only robot health service adapter."
            ),
            category=ToolCategory.ROBOT_HEALTH,
            safety_class=ToolSafetyClass.READ_ONLY,
            required_permissions=frozenset(
                {"robots:health:read"}
            ),
            input_schema={
                "required": ["robot_id"],
                "properties": {
                    "robot_id": {
                        "type": "string",
                    },
                },
                "additionalProperties": False,
            },
        ),
        handler=health_summary_tool,
    )

    registry.register(
        definition=ToolDefinition(
            name="simulation.capabilities",
            description=(
                "Describe available offline "
                "simulation capabilities."
            ),
            category=ToolCategory.SIMULATION,
            safety_class=ToolSafetyClass.READ_ONLY,
            input_schema={
                "required": [],
                "properties": {},
                "additionalProperties": False,
            },
        ),
        handler=simulation_info_tool,
    )

    registry.register(
        definition=ToolDefinition(
            name="safety.policy.info",
            description=(
                "Return advisory tool execution "
                "safety restrictions."
            ),
            category=ToolCategory.SAFETY,
            safety_class=ToolSafetyClass.READ_ONLY,
            input_schema={
                "required": [],
                "properties": {},
                "additionalProperties": False,
            },
        ),
        handler=safety_policy_info_tool,
    )

    return registry


# ============================================================
# Factory
# ============================================================

def create_tool_executor(
    *,
    authorizer: ToolAuthorizer,
    registry: ToolRegistry | None = None,
    approval_provider: ToolApprovalProvider | None = None,
    input_validator: ToolInputValidator | None = None,
    output_validator: ToolOutputValidator | None = None,
    audit_sink: ToolAuditSink | None = None,
    repository: ToolResultRepository | None = None,
    max_concurrency: int = 5,
) -> ToolExecutor:

    return ToolExecutor(
        registry=(
            registry
            or build_default_tool_registry()
        ),
        authorizer=authorizer,
        approval_provider=approval_provider,
        input_validator=input_validator,
        output_validator=output_validator,
        audit_sink=audit_sink,
        repository=repository,
        max_concurrency=max_concurrency,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "ToolExecutorError",
    "ToolValidationError",
    "ToolNotFoundError",
    "ToolAlreadyRegisteredError",
    "ToolPermissionError",
    "ToolSafetyError",
    "ToolApprovalError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolUnavailableError",
    "ToolConcurrencyError",
    "ToolCategory",
    "ToolSafetyClass",
    "ToolExecutionStatus",
    "ToolAvailability",
    "ToolApprovalStatus",
    "ToolErrorCode",
    "ToolExecutionScope",
    "ToolDefinition",
    "ToolExecutionRequest",
    "ToolExecutionContext",
    "ToolExecutionResult",
    "ToolExecutionStatistics",
    "ToolHandler",
    "ToolAuthorizer",
    "ToolApprovalProvider",
    "ToolInputValidator",
    "ToolOutputValidator",
    "ToolAuditSink",
    "ToolResultRepository",
    "RegisteredTool",
    "ToolRegistry",
    "DefaultToolInputValidator",
    "DefaultToolOutputValidator",
    "ToolExecutor",
    "health_summary_tool",
    "simulation_info_tool",
    "safety_policy_info_tool",
    "build_default_tool_registry",
    "create_tool_executor",
]
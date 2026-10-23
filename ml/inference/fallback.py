"""
ml/inference/fallback.py

Physical AI & Autonomous Robotics Intelligence Platform

Reliable ML inference fallback management.

Features:
- Primary and fallback model routing
- Multiple fallback providers
- CPU/GPU fallback strategies
- Circuit breaker protection
- Timeout and retry handling
- Exponential retry backoff
- Model health monitoring
- Priority-based provider selection
- Tenant-aware authorization
- Failure tracking
- Recovery and circuit reset
- Inference result validation
- Fallback execution reports
- Simulation-only robot action inference
- No direct robot actuation

Python 3.11+
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import random
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY_SECONDS = 0.25

DEFAULT_CIRCUIT_FAILURE_THRESHOLD = 5
DEFAULT_CIRCUIT_RECOVERY_SECONDS = 30.0

DEFAULT_MAX_PROVIDERS = 20
DEFAULT_MAX_CONCURRENT_REQUESTS = 10

MAX_TIMEOUT_SECONDS = 600.0
MAX_RETRIES = 5


# ============================================================
# Exceptions
# ============================================================

class FallbackError(Exception):
    """Base inference fallback exception."""


class FallbackValidationError(FallbackError):
    """Invalid fallback configuration or request."""


class FallbackPermissionError(FallbackError):
    """Inference permission denied."""


class FallbackSafetyError(FallbackError):
    """Inference violates the safety policy."""


class ProviderUnavailableError(FallbackError):
    """Inference provider is unavailable."""


class ProviderTimeoutError(FallbackError):
    """Inference provider exceeded its deadline."""


class ProviderExecutionError(FallbackError):
    """Inference provider execution failed."""


class ProviderOutputError(FallbackError):
    """Inference provider returned an invalid output."""


class CircuitBreakerOpenError(FallbackError):
    """Provider circuit breaker is open."""


class AllProvidersFailedError(FallbackError):
    """No provider successfully completed inference."""


# ============================================================
# Utilities
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def validate_positive(
    value: float,
    name: str,
) -> float:

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise FallbackValidationError(
            f"{name} must be a finite positive number"
        )

    return float(value)


# ============================================================
# Enumerations
# ============================================================

class ProviderDevice(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    REMOTE = "remote"
    AUTO = "auto"


class ProviderRole(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    EMERGENCY = "emergency"


class ProviderStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class FallbackStrategy(str, Enum):
    PRIORITY = "priority"
    LATENCY = "latency"
    HEALTH = "health"
    CPU_FALLBACK = "cpu_fallback"


class InferenceTaskType(str, Enum):
    PERCEPTION = "perception"
    VLM = "vlm"
    VLA = "vla"
    WORLD_MODEL = "world_model"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    ANOMALY = "anomaly"
    FAILURE_PREDICTION = "failure_prediction"
    RUL = "rul"
    PART_INSPECTION = "part_inspection"
    CUSTOM = "custom"


class FallbackResultStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True, slots=True)
class FallbackConfiguration:

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    max_retries: int = DEFAULT_MAX_RETRIES

    retry_delay_seconds: float = (
        DEFAULT_RETRY_DELAY_SECONDS
    )

    retry_backoff_multiplier: float = 2.0

    circuit_failure_threshold: int = (
        DEFAULT_CIRCUIT_FAILURE_THRESHOLD
    )

    circuit_recovery_seconds: float = (
        DEFAULT_CIRCUIT_RECOVERY_SECONDS
    )

    max_providers: int = DEFAULT_MAX_PROVIDERS

    max_concurrent_requests: int = (
        DEFAULT_MAX_CONCURRENT_REQUESTS
    )

    strategy: FallbackStrategy = (
        FallbackStrategy.PRIORITY
    )

    allow_cpu_fallback: bool = True

    enable_circuit_breaker: bool = True

    enable_retries: bool = True

    def __post_init__(self) -> None:

        validate_positive(
            self.timeout_seconds,
            "timeout_seconds",
        )

        if self.timeout_seconds > MAX_TIMEOUT_SECONDS:
            raise FallbackValidationError(
                "Inference timeout exceeds maximum"
            )

        if not 0 <= self.max_retries <= MAX_RETRIES:
            raise FallbackValidationError(
                "Invalid retry count"
            )

        if (
            not math.isfinite(self.retry_delay_seconds)
            or self.retry_delay_seconds < 0
        ):
            raise FallbackValidationError(
                "Invalid retry delay"
            )

        validate_positive(
            self.retry_backoff_multiplier,
            "retry_backoff_multiplier",
        )

        if self.circuit_failure_threshold < 1:
            raise FallbackValidationError(
                "Circuit failure threshold must be positive"
            )

        validate_positive(
            self.circuit_recovery_seconds,
            "circuit_recovery_seconds",
        )

        if not 1 <= self.max_providers <= 100:
            raise FallbackValidationError(
                "Invalid provider limit"
            )

        if self.max_concurrent_requests < 1:
            raise FallbackValidationError(
                "Concurrency must be positive"
            )


# ============================================================
# Data Models
# ============================================================

@dataclass(frozen=True, slots=True)
class FallbackScope:

    tenant_id: str
    user_id: str

    robot_id: str | None = None
    workflow_id: str | None = None
    task_id: str | None = None

    def __post_init__(self) -> None:

        if not self.tenant_id.strip():
            raise FallbackValidationError(
                "Tenant ID is required"
            )

        if not self.user_id.strip():
            raise FallbackValidationError(
                "User ID is required"
            )


@dataclass(slots=True)
class FallbackRequest:

    model_name: str
    inputs: Any
    scope: FallbackScope

    task_type: InferenceTaskType = (
        InferenceTaskType.CUSTOM
    )

    request_id: str = field(
        default_factory=generate_id
    )

    timeout_seconds: float | None = None

    simulation_only: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:

        if not self.model_name.strip():
            raise FallbackValidationError(
                "Model name is required"
            )

        if self.timeout_seconds is not None:
            validate_positive(
                self.timeout_seconds,
                "timeout_seconds",
            )


@dataclass(slots=True)
class ProviderAttempt:

    provider_name: str

    attempt_number: int

    status: FallbackResultStatus

    duration_ms: float

    error: str | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )


@dataclass(slots=True)
class FallbackResult:

    request_id: str
    model_name: str

    status: FallbackResultStatus

    output: Any = None

    provider_name: str | None = None

    provider_device: ProviderDevice | None = None

    fallback_used: bool = False

    attempts: list[ProviderAttempt] = field(
        default_factory=list
    )

    total_latency_ms: float = 0.0

    error: str | None = None

    completed_at: datetime = field(
        default_factory=utc_now
    )

    physical_execution_authorized: bool = False


@dataclass(slots=True)
class ProviderMetrics:

    total_requests: int = 0

    successful_requests: int = 0

    failed_requests: int = 0

    timeout_requests: int = 0

    consecutive_failures: int = 0

    total_latency_ms: float = 0.0

    last_success_at: datetime | None = None

    last_failure_at: datetime | None = None

    @property
    def success_rate(self) -> float:

        if self.total_requests == 0:
            return 0.0

        return (
            self.successful_requests
            / self.total_requests
        )

    @property
    def average_latency_ms(self) -> float:

        if self.successful_requests == 0:
            return 0.0

        return (
            self.total_latency_ms
            / self.successful_requests
        )


@dataclass(slots=True)
class FallbackStatistics:

    total_requests: int = 0

    successful_requests: int = 0

    failed_requests: int = 0

    fallback_requests: int = 0

    total_provider_attempts: int = 0

    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:

        if self.total_requests == 0:
            return 0.0

        return (
            self.successful_requests
            / self.total_requests
        )

    @property
    def fallback_rate(self) -> float:

        if self.total_requests == 0:
            return 0.0

        return (
            self.fallback_requests
            / self.total_requests
        )


# ============================================================
# Protocols
# ============================================================

class InferenceProvider(Protocol):

    async def predict(
        self,
        inputs: Any,
    ) -> Any:
        ...


class FallbackAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        ...


class FallbackAuditSink(Protocol):

    async def record(
        self,
        *,
        tenant_id: str,
        user_id: str,
        action: str,
        metadata: Mapping[str, Any],
    ) -> None:
        ...


class FallbackResultRepository(Protocol):

    async def save_result(
        self,
        result: FallbackResult,
        scope: FallbackScope,
    ) -> None:
        ...


# ============================================================
# Circuit Breaker
# ============================================================

class CircuitBreaker:

    def __init__(
        self,
        *,
        failure_threshold: int,
        recovery_seconds: float,
    ) -> None:

        self.failure_threshold = (
            failure_threshold
        )

        self.recovery_seconds = (
            recovery_seconds
        )

        self.state = CircuitState.CLOSED

        self.consecutive_failures = 0

        self.opened_at: float | None = None

        self._half_open_probe_active = False

        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """
        Reserve permission for one inference attempt.

        In HALF_OPEN state, only one probe is allowed.
        """

        async with self._lock:

            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:

                if self.opened_at is None:
                    return False

                elapsed = (
                    time.monotonic()
                    - self.opened_at
                )

                if elapsed < self.recovery_seconds:
                    return False

                self.state = CircuitState.HALF_OPEN

            if self.state == CircuitState.HALF_OPEN:

                if self._half_open_probe_active:
                    return False

                self._half_open_probe_active = True

                return True

            return False

    async def record_success(self) -> None:

        async with self._lock:

            self.consecutive_failures = 0

            self.state = CircuitState.CLOSED

            self.opened_at = None

            self._half_open_probe_active = False

    async def record_failure(self) -> None:

        async with self._lock:

            self.consecutive_failures += 1

            if (
                self.state == CircuitState.HALF_OPEN
                or self.consecutive_failures
                >= self.failure_threshold
            ):
                self.state = CircuitState.OPEN

                self.opened_at = time.monotonic()

            self._half_open_probe_active = False

    async def release_probe(self) -> None:

        async with self._lock:
            self._half_open_probe_active = False

    async def reset(self) -> None:

        async with self._lock:

            self.state = CircuitState.CLOSED

            self.consecutive_failures = 0

            self.opened_at = None

            self._half_open_probe_active = False

    async def snapshot(self) -> dict[str, Any]:

        async with self._lock:

            return {
                "state": self.state.value,
                "consecutive_failures": (
                    self.consecutive_failures
                ),
                "failure_threshold": (
                    self.failure_threshold
                ),
                "recovery_seconds": (
                    self.recovery_seconds
                ),
                "half_open_probe_active": (
                    self._half_open_probe_active
                ),
            }


# ============================================================
# Provider Registration
# ============================================================

@dataclass(slots=True)
class RegisteredFallbackProvider:

    name: str

    model_name: str

    backend: InferenceProvider

    role: ProviderRole = ProviderRole.SECONDARY

    device: ProviderDevice = ProviderDevice.AUTO

    priority: int = 100

    status: ProviderStatus = (
        ProviderStatus.HEALTHY
    )

    version: str = "1.0.0"

    timeout_seconds: float | None = None

    metrics: ProviderMetrics = field(
        default_factory=ProviderMetrics
    )

    circuit_breaker: CircuitBreaker | None = None

    def __post_init__(self) -> None:

        if not self.name.strip():
            raise FallbackValidationError(
                "Provider name is required"
            )

        if not self.model_name.strip():
            raise FallbackValidationError(
                "Model name is required"
            )

        if self.timeout_seconds is not None:
            validate_positive(
                self.timeout_seconds,
                "timeout_seconds",
            )


class FallbackProviderRegistry:

    def __init__(
        self,
        config: FallbackConfiguration,
    ) -> None:

        self.config = config

        self._providers: dict[
            str,
            RegisteredFallbackProvider,
        ] = {}

    def register(
        self,
        provider: RegisteredFallbackProvider,
    ) -> None:

        if provider.name in self._providers:
            raise FallbackValidationError(
                f"Provider already registered: "
                f"{provider.name}"
            )

        if (
            len(self._providers)
            >= self.config.max_providers
        ):
            raise FallbackValidationError(
                "Provider registry limit reached"
            )

        if provider.circuit_breaker is None:

            provider.circuit_breaker = CircuitBreaker(
                failure_threshold=(
                    self.config.circuit_failure_threshold
                ),
                recovery_seconds=(
                    self.config.circuit_recovery_seconds
                ),
            )

        self._providers[provider.name] = provider

    def get(
        self,
        provider_name: str,
    ) -> RegisteredFallbackProvider:

        provider = self._providers.get(
            provider_name
        )

        if provider is None:
            raise ProviderUnavailableError(
                f"Provider not found: {provider_name}"
            )

        return provider

    def list_for_model(
        self,
        model_name: str,
    ) -> list[RegisteredFallbackProvider]:

        return [
            provider
            for provider in self._providers.values()
            if provider.model_name == model_name
            and provider.status
            not in (
                ProviderStatus.UNAVAILABLE,
                ProviderStatus.DISABLED,
            )
        ]

    def list_all(
        self,
    ) -> list[RegisteredFallbackProvider]:

        return list(self._providers.values())

    def set_status(
        self,
        provider_name: str,
        status: ProviderStatus,
    ) -> None:

        provider = self.get(provider_name)

        provider.status = status


# ============================================================
# Fallback Manager
# ============================================================

class InferenceFallbackManager:

    """
    Manages resilient model inference across providers.

    Fallback inference is advisory only.

    Model outputs cannot authorize robot movement,
    physical manipulation, emergency-stop reset,
    or other physical operations.
    """

    def __init__(
        self,
        *,
        authorizer: FallbackAuthorizer,
        config: FallbackConfiguration | None = None,
        audit_sink: FallbackAuditSink | None = None,
        repository: FallbackResultRepository | None = None,
    ) -> None:

        self.config = (
            config or FallbackConfiguration()
        )

        self.authorizer = authorizer

        self.audit_sink = audit_sink

        self.repository = repository

        self.registry = FallbackProviderRegistry(
            self.config
        )

        self.statistics = FallbackStatistics()

        self._semaphore = asyncio.Semaphore(
            self.config.max_concurrent_requests
        )

    # --------------------------------------------------------
    # Register Provider
    # --------------------------------------------------------

    def register_provider(
        self,
        provider: RegisteredFallbackProvider,
    ) -> None:

        self.registry.register(provider)

        logger.info(
            "Registered inference provider %s for model %s",
            provider.name,
            provider.model_name,
        )

    # --------------------------------------------------------
    # Safety Validation
    # --------------------------------------------------------

    @staticmethod
    def _validate_request_safety(
        request: FallbackRequest,
    ) -> None:

        action_tasks = {
            InferenceTaskType.VLA,
            InferenceTaskType.NAVIGATION,
            InferenceTaskType.MANIPULATION,
        }

        if (
            request.task_type in action_tasks
            and not request.simulation_only
        ):
            raise FallbackSafetyError(
                "Robot action inference must "
                "remain simulation-only"
            )

        if request.metadata.get(
            "physical_execution_authorized"
        ) is True:
            raise FallbackSafetyError(
                "Inference cannot authorize "
                "physical robot execution"
            )

    @staticmethod
    def _validate_output(
        output: Any,
    ) -> None:

        if output is None:
            raise ProviderOutputError(
                "Provider returned no output"
            )

        if isinstance(output, Mapping):

            forbidden_true_fields = {
                "physical_execution_authorized",
                "emergency_stop_reset_authorized",
                "safety_interlock_bypass_authorized",
            }

            for field_name in forbidden_true_fields:

                if output.get(field_name) is True:
                    raise ProviderOutputError(
                        f"Unsafe output field: {field_name}"
                    )

    # --------------------------------------------------------
    # Provider Ordering
    # --------------------------------------------------------

    def _order_providers(
        self,
        model_name: str,
    ) -> list[RegisteredFallbackProvider]:

        providers = self.registry.list_for_model(
            model_name
        )

        if not providers:
            raise ProviderUnavailableError(
                f"No providers available for {model_name}"
            )

        strategy = self.config.strategy

        if strategy == FallbackStrategy.PRIORITY:

            providers.sort(
                key=lambda provider: (
                    provider.role != ProviderRole.PRIMARY,
                    provider.priority,
                    provider.name,
                )
            )

        elif strategy == FallbackStrategy.LATENCY:

            providers.sort(
                key=lambda provider: (
                    provider.metrics.average_latency_ms
                    if provider.metrics.successful_requests
                    else float("inf"),
                    provider.priority,
                )
            )

        elif strategy == FallbackStrategy.HEALTH:

            providers.sort(
                key=lambda provider: (
                    provider.status
                    != ProviderStatus.HEALTHY,
                    -provider.metrics.success_rate,
                    provider.metrics.consecutive_failures,
                    provider.priority,
                )
            )

        elif strategy == FallbackStrategy.CPU_FALLBACK:

            providers.sort(
                key=lambda provider: (
                    provider.device == ProviderDevice.CPU,
                    provider.role != ProviderRole.PRIMARY,
                    provider.priority,
                )
            )

        if not self.config.allow_cpu_fallback:

            providers = [
                provider
                for provider in providers
                if provider.device
                != ProviderDevice.CPU
            ]

        return providers

    # --------------------------------------------------------
    # Execute Provider
    # --------------------------------------------------------

    async def _execute_provider(
        self,
        *,
        provider: RegisteredFallbackProvider,
        request: FallbackRequest,
        deadline: float,
    ) -> tuple[Any | None, list[ProviderAttempt]]:

        attempts: list[ProviderAttempt] = []

        max_attempts = (
            self.config.max_retries + 1
            if self.config.enable_retries
            else 1
        )

        loop = asyncio.get_running_loop()

        for attempt_number in range(
            1,
            max_attempts + 1,
        ):

            remaining = deadline - loop.time()

            if remaining <= 0:
                break

            breaker = provider.circuit_breaker

            acquired = True

            if (
                self.config.enable_circuit_breaker
                and breaker is not None
            ):
                acquired = await breaker.acquire()

            if not acquired:
                raise CircuitBreakerOpenError(
                    f"Circuit open for {provider.name}"
                )

            started = time.perf_counter()

            provider.metrics.total_requests += 1

            self.statistics.total_provider_attempts += 1

            try:

                provider_timeout = (
                    provider.timeout_seconds
                    or self.config.timeout_seconds
                )

                effective_timeout = min(
                    provider_timeout,
                    remaining,
                )

                output = await asyncio.wait_for(
                    provider.backend.predict(
                        request.inputs
                    ),
                    timeout=effective_timeout,
                )

                self._validate_output(output)

                elapsed_ms = (
                    time.perf_counter() - started
                ) * 1000

                provider.metrics.successful_requests += 1

                provider.metrics.consecutive_failures = 0

                provider.metrics.total_latency_ms += (
                    elapsed_ms
                )

                provider.metrics.last_success_at = (
                    utc_now()
                )

                provider.status = ProviderStatus.HEALTHY

                if breaker is not None:
                    await breaker.record_success()

                attempts.append(
                    ProviderAttempt(
                        provider_name=provider.name,
                        attempt_number=attempt_number,
                        status=(
                            FallbackResultStatus.COMPLETED
                        ),
                        duration_ms=elapsed_ms,
                    )
                )

                return output, attempts

            except asyncio.CancelledError:

                if breaker is not None:
                    await breaker.release_probe()

                raise

            except asyncio.TimeoutError:

                error: Exception = ProviderTimeoutError(
                    f"Provider timed out: {provider.name}"
                )

                status = FallbackResultStatus.TIMED_OUT

                provider.metrics.timeout_requests += 1

            except Exception as exc:

                error = exc

                status = FallbackResultStatus.FAILED

            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000

            provider.metrics.failed_requests += 1

            provider.metrics.consecutive_failures += 1

            provider.metrics.last_failure_at = (
                utc_now()
            )

            if (
                provider.metrics.consecutive_failures
                >= self.config.circuit_failure_threshold
            ):
                provider.status = ProviderStatus.DEGRADED

            if breaker is not None:
                await breaker.record_failure()

            attempts.append(
                ProviderAttempt(
                    provider_name=provider.name,
                    attempt_number=attempt_number,
                    status=status,
                    duration_ms=elapsed_ms,
                    error=str(error),
                )
            )

            logger.warning(
                "Inference provider %s failed "
                "on attempt %s: %s",
                provider.name,
                attempt_number,
                type(error).__name__,
            )

            if attempt_number >= max_attempts:
                break

            retry_delay = (
                self.config.retry_delay_seconds
                * (
                    self.config.retry_backoff_multiplier
                    ** (attempt_number - 1)
                )
            )

            retry_delay *= random.uniform(
                0.8,
                1.2,
            )

            remaining = deadline - loop.time()

            if remaining <= retry_delay:
                break

            await asyncio.sleep(retry_delay)

        return None, attempts

    # --------------------------------------------------------
    # Run Inference
    # --------------------------------------------------------

    async def predict(
        self,
        request: FallbackRequest,
    ) -> FallbackResult:

        self._validate_request_safety(request)

        try:

            await self.authorizer.require_permission(
                tenant_id=request.scope.tenant_id,
                user_id=request.scope.user_id,
                permission="ml:inference",
            )

        except PermissionError as exc:

            raise FallbackPermissionError(
                "Inference permission denied"
            ) from exc

        async with self._semaphore:

            started = time.perf_counter()

            loop = asyncio.get_running_loop()

            timeout = (
                request.timeout_seconds
                or self.config.timeout_seconds
            )

            deadline = loop.time() + timeout

            self.statistics.total_requests += 1

            providers = self._order_providers(
                request.model_name
            )

            all_attempts: list[
                ProviderAttempt
            ] = []

            selected_provider: (
                RegisteredFallbackProvider | None
            ) = None

            output: Any = None

            fallback_used = False

            for provider_index, provider in enumerate(
                providers
            ):

                if loop.time() >= deadline:
                    break

                try:

                    provider_output, attempts = (
                        await self._execute_provider(
                            provider=provider,
                            request=request,
                            deadline=deadline,
                        )
                    )

                except CircuitBreakerOpenError:

                    logger.info(
                        "Skipping provider %s: circuit open",
                        provider.name,
                    )

                    continue

                all_attempts.extend(attempts)

                if (
                    attempts
                    and attempts[-1].status
                    == FallbackResultStatus.COMPLETED
                ):

                    output = provider_output

                    selected_provider = provider

                    fallback_used = (
                        provider_index > 0
                    )

                    break

            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000

            self.statistics.total_latency_ms += (
                elapsed_ms
            )

            if selected_provider is not None:

                self.statistics.successful_requests += 1

                if fallback_used:
                    self.statistics.fallback_requests += 1

                result = FallbackResult(
                    request_id=request.request_id,
                    model_name=request.model_name,
                    status=(
                        FallbackResultStatus.COMPLETED
                    ),
                    output=output,
                    provider_name=(
                        selected_provider.name
                    ),
                    provider_device=(
                        selected_provider.device
                    ),
                    fallback_used=fallback_used,
                    attempts=all_attempts,
                    total_latency_ms=elapsed_ms,
                    physical_execution_authorized=False,
                )

            else:

                self.statistics.failed_requests += 1

                deadline_exceeded = (
                    loop.time() >= deadline
                )

                result = FallbackResult(
                    request_id=request.request_id,
                    model_name=request.model_name,
                    status=(
                        FallbackResultStatus.TIMED_OUT
                        if deadline_exceeded
                        else FallbackResultStatus.FAILED
                    ),
                    fallback_used=(
                        len(providers) > 1
                        and len(
                            {
                                attempt.provider_name
                                for attempt in all_attempts
                            }
                        ) > 1
                    ),
                    attempts=all_attempts,
                    total_latency_ms=elapsed_ms,
                    error=(
                        "Inference deadline exceeded"
                        if deadline_exceeded
                        else "All inference providers failed"
                    ),
                )

            await self._persist_result(
                result=result,
                scope=request.scope,
            )

            await self._audit_result(
                request=request,
                result=result,
            )

            return result

    # --------------------------------------------------------
    # Batch Inference
    # --------------------------------------------------------

    async def predict_many(
        self,
        requests: Sequence[FallbackRequest],
    ) -> list[FallbackResult]:

        return list(
            await asyncio.gather(
                *(
                    self.predict(request)
                    for request in requests
                )
            )
        )

    # --------------------------------------------------------
    # Persist Result
    # --------------------------------------------------------

    async def _persist_result(
        self,
        *,
        result: FallbackResult,
        scope: FallbackScope,
    ) -> None:

        if self.repository is None:
            return

        await self.repository.save_result(
            result,
            scope,
        )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    async def _audit_result(
        self,
        *,
        request: FallbackRequest,
        result: FallbackResult,
    ) -> None:

        if self.audit_sink is None:
            return

        await self.audit_sink.record(
            tenant_id=request.scope.tenant_id,
            user_id=request.scope.user_id,
            action="ml.inference.fallback",
            metadata={
                "request_id": request.request_id,
                "model_name": request.model_name,
                "task_type": request.task_type.value,
                "status": result.status.value,
                "provider_name": result.provider_name,
                "fallback_used": result.fallback_used,
                "attempt_count": len(result.attempts),
                "latency_ms": result.total_latency_ms,
                "simulation_only": (
                    request.simulation_only
                ),
                "physical_execution_authorized": False,
            },
        )

    # --------------------------------------------------------
    # Provider Health Report
    # --------------------------------------------------------

    async def get_provider_health(
        self,
    ) -> list[dict[str, Any]]:

        report: list[dict[str, Any]] = []

        for provider in self.registry.list_all():

            circuit = (
                await provider.circuit_breaker.snapshot()
                if provider.circuit_breaker is not None
                else None
            )

            report.append(
                {
                    "provider_name": provider.name,
                    "model_name": provider.model_name,
                    "role": provider.role.value,
                    "device": provider.device.value,
                    "status": provider.status.value,
                    "version": provider.version,
                    "priority": provider.priority,
                    "total_requests": (
                        provider.metrics.total_requests
                    ),
                    "successful_requests": (
                        provider.metrics.successful_requests
                    ),
                    "failed_requests": (
                        provider.metrics.failed_requests
                    ),
                    "timeout_requests": (
                        provider.metrics.timeout_requests
                    ),
                    "success_rate": (
                        provider.metrics.success_rate
                    ),
                    "average_latency_ms": (
                        provider.metrics.average_latency_ms
                    ),
                    "consecutive_failures": (
                        provider.metrics.consecutive_failures
                    ),
                    "circuit_breaker": circuit,
                }
            )

        return report

    # --------------------------------------------------------
    # Reset Provider Circuit
    # --------------------------------------------------------

    async def reset_provider_circuit(
        self,
        provider_name: str,
        *,
        scope: FallbackScope,
    ) -> None:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            permission="ml:providers:manage",
        )

        provider = self.registry.get(
            provider_name
        )

        if provider.circuit_breaker is not None:
            await provider.circuit_breaker.reset()

        provider.metrics.consecutive_failures = 0

        if provider.status == ProviderStatus.DEGRADED:
            provider.status = ProviderStatus.HEALTHY

        if self.audit_sink is not None:

            await self.audit_sink.record(
                tenant_id=scope.tenant_id,
                user_id=scope.user_id,
                action="ml.provider.circuit_reset",
                metadata={
                    "provider_name": provider_name,
                },
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def get_statistics(
        self,
    ) -> dict[str, Any]:

        return {
            "total_requests": (
                self.statistics.total_requests
            ),
            "successful_requests": (
                self.statistics.successful_requests
            ),
            "failed_requests": (
                self.statistics.failed_requests
            ),
            "fallback_requests": (
                self.statistics.fallback_requests
            ),
            "total_provider_attempts": (
                self.statistics.total_provider_attempts
            ),
            "success_rate": (
                self.statistics.success_rate
            ),
            "fallback_rate": (
                self.statistics.fallback_rate
            ),
            "total_latency_ms": (
                self.statistics.total_latency_ms
            ),
            "registered_providers": len(
                self.registry.list_all()
            ),
        }

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    async def generate_report(
        self,
    ) -> dict[str, Any]:

        return {
            "generated_at": utc_now().isoformat(),
            "configuration": {
                "strategy": (
                    self.config.strategy.value
                ),
                "timeout_seconds": (
                    self.config.timeout_seconds
                ),
                "max_retries": (
                    self.config.max_retries
                ),
                "allow_cpu_fallback": (
                    self.config.allow_cpu_fallback
                ),
                "enable_circuit_breaker": (
                    self.config.enable_circuit_breaker
                ),
            },
            "statistics": self.get_statistics(),
            "providers": (
                await self.get_provider_health()
            ),
            "safety": {
                "physical_actuation": False,
                "emergency_stop_reset": False,
                "interlock_bypass": False,
                "simulation_only_action_models": True,
            },
        }


# ============================================================
# Callable Provider Adapter
# ============================================================

class CallableInferenceProvider:

    """
    Wrap a synchronous or asynchronous inference function.

    Example:
        provider = CallableInferenceProvider(
            lambda inputs: model.predict(inputs)
        )
    """

    def __init__(
        self,
        predict_fn: Callable[
            [Any],
            Any | Awaitable[Any],
        ],
    ) -> None:

        self.predict_fn = predict_fn

    async def predict(
        self,
        inputs: Any,
    ) -> Any:

        if inspect.iscoroutinefunction(
            self.predict_fn
        ):
            return await self.predict_fn(inputs)

        result = await asyncio.to_thread(
            self.predict_fn,
            inputs,
        )

        if inspect.isawaitable(result):
            return await result

        return result


# ============================================================
# Example Authorization Provider
# ============================================================

class DenyByDefaultAuthorizer:

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:

        raise FallbackPermissionError(
            "No authorization provider configured"
        )


# ============================================================
# Factory
# ============================================================

def create_fallback_manager(
    *,
    authorizer: FallbackAuthorizer,
    config: FallbackConfiguration | None = None,
    audit_sink: FallbackAuditSink | None = None,
    repository: FallbackResultRepository | None = None,
) -> InferenceFallbackManager:

    return InferenceFallbackManager(
        authorizer=authorizer,
        config=config,
        audit_sink=audit_sink,
        repository=repository,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "FallbackError",
    "FallbackValidationError",
    "FallbackPermissionError",
    "FallbackSafetyError",
    "ProviderUnavailableError",
    "ProviderTimeoutError",
    "ProviderExecutionError",
    "ProviderOutputError",
    "CircuitBreakerOpenError",
    "AllProvidersFailedError",
    "ProviderDevice",
    "ProviderRole",
    "ProviderStatus",
    "CircuitState",
    "FallbackStrategy",
    "InferenceTaskType",
    "FallbackResultStatus",
    "FallbackConfiguration",
    "FallbackScope",
    "FallbackRequest",
    "ProviderAttempt",
    "FallbackResult",
    "ProviderMetrics",
    "FallbackStatistics",
    "InferenceProvider",
    "FallbackAuthorizer",
    "FallbackAuditSink",
    "FallbackResultRepository",
    "CircuitBreaker",
    "RegisteredFallbackProvider",
    "FallbackProviderRegistry",
    "InferenceFallbackManager",
    "CallableInferenceProvider",
    "DenyByDefaultAuthorizer",
    "create_fallback_manager",
]
"""Rate Limit module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Core Rate Limiting

Features:
- Async Redis-backed rate limiting
- In-memory rate limiting for development/testing
- Fixed-window and sliding-window algorithms
- Token-bucket rate limiting
- Per-user, tenant, API key and IP policies
- Endpoint-specific rate limits
- FastAPI dependency integration
- ASGI rate-limit middleware
- RateLimit response headers
- Retry-After support
- Atomic Redis Lua scripts
- Fail-closed protection for critical endpoints
- Bounded identifiers and safe Redis keys

Security:
- Never trust client-provided tenant/user identifiers.
- Obtain authenticated identities from trusted request state.
- Trust forwarded IP headers only through a configured proxy.
- Rate limiting is not a substitute for authentication,
  authorization, robot safety checks or emergency interlocks.
- Emergency-stop requests should not depend on Redis or
  application rate limiting to remain operational.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import math
import time

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Mapping, Protocol

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

DEFAULT_REQUEST_LIMIT = 100
DEFAULT_WINDOW_SECONDS = 60
DEFAULT_BURST_CAPACITY = 20

MAX_IDENTIFIER_LENGTH = 256
MAX_MEMORY_BUCKETS = 50_000

RATE_LIMIT_PREFIX = "physical_ai:rate_limit"

DEFAULT_EXCLUDED_PATHS = frozenset({
    "/health",
    "/health/live",
    "/health/ready",
    "/live",
    "/ready",
    "/metrics",
})

# These operations must not be made dependent on the
# availability of the application rate limiter.
SAFETY_EXEMPT_PATHS = frozenset({
    "/api/v1/safety/emergency-stop",
    "/api/v1/robots/emergency-stop",
})


# ============================================================
# Enums
# ============================================================


class RateLimitAlgorithm(str, Enum):
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


class RateLimitScope(str, Enum):
    GLOBAL = "global"
    IP = "ip"
    USER = "user"
    TENANT = "tenant"
    API_KEY = "api_key"
    ROBOT = "robot"
    FLEET = "fleet"


class RateLimitBackend(str, Enum):
    MEMORY = "memory"
    REDIS = "redis"


class RateLimitCategory(str, Enum):
    DEFAULT = "default"
    AUTHENTICATION = "authentication"
    AI_INFERENCE = "ai_inference"
    RAG = "rag"
    FILE_UPLOAD = "file_upload"
    TELEMETRY = "telemetry"
    ROBOT_CONTROL = "robot_control"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    MODEL_TRAINING = "model_training"
    EVALUATION = "evaluation"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    ADMINISTRATION = "administration"


# ============================================================
# Exceptions
# ============================================================


class RateLimitError(Exception):
    pass


class RateLimitExceededError(RateLimitError):
    def __init__(
        self,
        *,
        retry_after: float,
        limit: int,
        remaining: int = 0,
    ) -> None:
        self.retry_after = max(0.0, retry_after)
        self.limit = limit
        self.remaining = remaining

        super().__init__("Rate limit exceeded")


class RateLimitBackendError(RateLimitError):
    pass


class RateLimitIdentityError(RateLimitError):
    pass


# ============================================================
# Data Models
# ============================================================


@dataclass(frozen=True)
class RateLimitPolicy:
    name: str
    limit: int
    window_seconds: int

    algorithm: RateLimitAlgorithm = (
        RateLimitAlgorithm.SLIDING_WINDOW
    )

    scope: RateLimitScope = RateLimitScope.IP

    burst_capacity: int | None = None

    enabled: bool = True
    fail_closed: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Policy name is required")

        if self.limit <= 0:
            raise ValueError("Rate limit must be positive")

        if self.window_seconds <= 0:
            raise ValueError(
                "window_seconds must be positive"
            )

        if (
            self.burst_capacity is not None
            and self.burst_capacity <= 0
        ):
            raise ValueError(
                "burst_capacity must be positive"
            )


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_after_seconds: float
    retry_after_seconds: float = 0.0
    policy_name: str = "default"
    backend: str = "unknown"

    def to_headers(self) -> dict[str, str]:
        headers = {
            "RateLimit-Limit": str(self.limit),
            "RateLimit-Remaining": str(
                max(0, self.remaining)
            ),
            "RateLimit-Reset": str(
                max(0, math.ceil(self.reset_after_seconds))
            ),
        }

        if not self.allowed:
            headers["Retry-After"] = str(
                max(1, math.ceil(self.retry_after_seconds))
            )

        return headers


@dataclass(frozen=True)
class RateLimitSettings:
    enabled: bool = True

    backend: RateLimitBackend = RateLimitBackend.MEMORY

    redis_url: str = "redis://localhost:6379/0"

    key_prefix: str = RATE_LIMIT_PREFIX

    default_limit: int = DEFAULT_REQUEST_LIMIT
    default_window_seconds: int = DEFAULT_WINDOW_SECONDS

    max_memory_buckets: int = MAX_MEMORY_BUCKETS

    fail_closed: bool = True

    excluded_paths: frozenset[str] = field(
        default_factory=lambda: DEFAULT_EXCLUDED_PATHS
    )

    safety_exempt_paths: frozenset[str] = field(
        default_factory=lambda: SAFETY_EXEMPT_PATHS
    )

    def __post_init__(self) -> None:
        if self.default_limit <= 0:
            raise ValueError(
                "default_limit must be positive"
            )

        if self.default_window_seconds <= 0:
            raise ValueError(
                "default_window_seconds must be positive"
            )

        if self.max_memory_buckets <= 0:
            raise ValueError(
                "max_memory_buckets must be positive"
            )


# ============================================================
# Key Management
# ============================================================


def hash_identifier(value: str) -> str:
    if not value:
        raise RateLimitIdentityError(
            "Rate limit identity cannot be empty"
        )

    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise RateLimitIdentityError(
            "Rate limit identity exceeds maximum length"
        )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def build_rate_limit_key(
    *,
    prefix: str,
    policy: RateLimitPolicy,
    identifier: str,
) -> str:
    policy_hash = hashlib.sha256(
        (
            f"{policy.name}:"
            f"{policy.algorithm.value}:"
            f"{policy.limit}:"
            f"{policy.window_seconds}:"
            f"{policy.burst_capacity}"
        ).encode("utf-8")
    ).hexdigest()[:20]

    return (
        f"{prefix}:"
        f"{policy_hash}:"
        f"{policy.scope.value}:"
        f"{hash_identifier(identifier)}"
    )


# ============================================================
# Backend Protocol
# ============================================================


class RateLimitStore(Protocol):
    async def check(
        self,
        *,
        key: str,
        policy: RateLimitPolicy,
        cost: int = 1,
    ) -> RateLimitDecision:
        ...

    async def close(self) -> None:
        ...


# ============================================================
# In-Memory Store
# ============================================================


@dataclass
class _MemoryBucket:
    timestamps: Deque[float] = field(
        default_factory=deque
    )

    window_start: float = 0.0
    count: int = 0

    tokens: float = 0.0
    updated_at: float = 0.0

    expires_at: float = 0.0


class MemoryRateLimitStore:
    """
    Async-safe in-memory limiter.

    Intended for local development and tests.

    State is process-local and does not coordinate
    multiple application workers.
    """

    def __init__(
        self,
        *,
        max_buckets: int = MAX_MEMORY_BUCKETS,
    ) -> None:
        self.max_buckets = max_buckets

        self._buckets: OrderedDict[
            str, _MemoryBucket
        ] = OrderedDict()

        self._lock = asyncio.Lock()

    def _cleanup(self, now: float) -> None:
        expired = [
            key
            for key, bucket in self._buckets.items()
            if bucket.expires_at <= now
        ]

        for key in expired:
            self._buckets.pop(key, None)

        while len(self._buckets) > self.max_buckets:
            self._buckets.popitem(last=False)

    async def check(
        self,
        *,
        key: str,
        policy: RateLimitPolicy,
        cost: int = 1,
    ) -> RateLimitDecision:
        if cost <= 0:
            raise ValueError("Cost must be positive")

        async with self._lock:
            now = time.monotonic()

            self._cleanup(now)

            bucket = self._buckets.get(key)

            if bucket is None:
                bucket = _MemoryBucket(
                    window_start=now,
                    updated_at=now,
                    tokens=float(
                        policy.burst_capacity
                        or policy.limit
                    ),
                )

                if len(self._buckets) >= self.max_buckets:
                    self._buckets.popitem(last=False)

                self._buckets[key] = bucket
            else:
                self._buckets.move_to_end(key)

            bucket.expires_at = (
                now + policy.window_seconds * 2
            )

            if policy.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                return self._fixed_window(
                    bucket, policy, now, cost
                )

            if policy.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                return self._sliding_window(
                    bucket, policy, now, cost
                )

            return self._token_bucket(
                bucket, policy, now, cost
            )

    def _fixed_window(
        self,
        bucket: _MemoryBucket,
        policy: RateLimitPolicy,
        now: float,
        cost: int,
    ) -> RateLimitDecision:
        elapsed = now - bucket.window_start

        if elapsed >= policy.window_seconds:
            bucket.window_start = now
            bucket.count = 0

        reset_after = max(
            0.0,
            policy.window_seconds
            - (now - bucket.window_start),
        )

        allowed = bucket.count + cost <= policy.limit

        if allowed:
            bucket.count += cost

        return RateLimitDecision(
            allowed=allowed,
            limit=policy.limit,
            remaining=max(
                0, policy.limit - bucket.count
            ),
            reset_after_seconds=reset_after,
            retry_after_seconds=(
                0.0 if allowed else reset_after
            ),
            policy_name=policy.name,
            backend="memory",
        )

    def _sliding_window(
        self,
        bucket: _MemoryBucket,
        policy: RateLimitPolicy,
        now: float,
        cost: int,
    ) -> RateLimitDecision:
        cutoff = now - policy.window_seconds

        while (
            bucket.timestamps
            and bucket.timestamps[0] <= cutoff
        ):
            bucket.timestamps.popleft()

        allowed = (
            len(bucket.timestamps) + cost
            <= policy.limit
        )

        if allowed:
            bucket.timestamps.extend(
                [now] * cost
            )

        remaining = max(
            0,
            policy.limit - len(bucket.timestamps),
        )

        reset_after = (
            max(
                0.0,
                bucket.timestamps[0]
                + policy.window_seconds
                - now,
            )
            if bucket.timestamps
            else 0.0
        )

        retry_after = 0.0

        if not allowed:
            required_expirations = (
                len(bucket.timestamps)
                + cost
                - policy.limit
            )

            if (
                required_expirations > 0
                and required_expirations
                <= len(bucket.timestamps)
            ):
                timestamp = list(
                    bucket.timestamps
                )[required_expirations - 1]

                retry_after = max(
                    0.0,
                    timestamp
                    + policy.window_seconds
                    - now,
                )
            else:
                retry_after = float(
                    policy.window_seconds
                )

        return RateLimitDecision(
            allowed=allowed,
            limit=policy.limit,
            remaining=remaining,
            reset_after_seconds=reset_after,
            retry_after_seconds=retry_after,
            policy_name=policy.name,
            backend="memory",
        )

    def _token_bucket(
        self,
        bucket: _MemoryBucket,
        policy: RateLimitPolicy,
        now: float,
        cost: int,
    ) -> RateLimitDecision:
        capacity = (
            policy.burst_capacity
            or policy.limit
        )

        refill_rate = (
            policy.limit
            / policy.window_seconds
        )

        elapsed = max(
            0.0,
            now - bucket.updated_at,
        )

        bucket.tokens = min(
            float(capacity),
            bucket.tokens + elapsed * refill_rate,
        )

        bucket.updated_at = now

        allowed = bucket.tokens >= cost

        if allowed:
            bucket.tokens -= cost

        remaining = max(
            0,
            math.floor(bucket.tokens),
        )

        retry_after = (
            0.0
            if allowed
            else max(
                0.0,
                (cost - bucket.tokens)
                / refill_rate,
            )
        )

        reset_after = (
            capacity - bucket.tokens
        ) / refill_rate

        return RateLimitDecision(
            allowed=allowed,
            limit=capacity,
            remaining=remaining,
            reset_after_seconds=max(
                0.0, reset_after
            ),
            retry_after_seconds=retry_after,
            policy_name=policy.name,
            backend="memory",
        )

    async def close(self) -> None:
        async with self._lock:
            self._buckets.clear()


# ============================================================
# Redis Lua Scripts
# ============================================================


FIXED_WINDOW_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])

local now_parts = redis.call("TIME")
local now_ms = now_parts[1] * 1000
    + math.floor(now_parts[2] / 1000)

local bucket = math.floor(now_ms / window_ms)
local window_key = key .. ":" .. tostring(bucket)

local count = tonumber(
    redis.call("GET", window_key) or "0"
)

local allowed = 0

if count + cost <= limit then
    count = redis.call("INCRBY", window_key, cost)
    allowed = 1
end

redis.call("PEXPIRE", window_key, window_ms)

local reset_ms = window_ms - (now_ms % window_ms)

return {
    allowed,
    math.max(0, limit - count),
    reset_ms
}
"""


SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local nonce = ARGV[4]

local now_parts = redis.call("TIME")
local now_ms = now_parts[1] * 1000
    + math.floor(now_parts[2] / 1000)

local cutoff = now_ms - window_ms

redis.call(
    "ZREMRANGEBYSCORE",
    key,
    "-inf",
    cutoff
)

local count = redis.call("ZCARD", key)
local allowed = 0
local retry_ms = 0

if count + cost <= limit then
    for i = 1, cost do
        redis.call(
            "ZADD",
            key,
            now_ms,
            nonce .. ":" .. tostring(i)
        )
    end

    count = count + cost
    allowed = 1
else
    local required = count + cost - limit

    if required <= count then
        local entries = redis.call(
            "ZRANGE",
            key,
            required - 1,
            required - 1,
            "WITHSCORES"
        )

        if #entries >= 2 then
            retry_ms = math.max(
                0,
                tonumber(entries[2])
                + window_ms
                - now_ms
            )
        end
    else
        retry_ms = window_ms
    end
end

redis.call("PEXPIRE", key, window_ms * 2)

local oldest = redis.call(
    "ZRANGE",
    key,
    0,
    0,
    "WITHSCORES"
)

local reset_ms = 0

if #oldest >= 2 then
    reset_ms = math.max(
        0,
        tonumber(oldest[2])
        + window_ms
        - now_ms
    )
end

return {
    allowed,
    math.max(0, limit - count),
    reset_ms,
    retry_ms
}
"""


TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local ttl_ms = tonumber(ARGV[4])

local now_parts = redis.call("TIME")
local now_ms = now_parts[1] * 1000
    + math.floor(now_parts[2] / 1000)

local values = redis.call(
    "HMGET",
    key,
    "tokens",
    "updated_at"
)

local tokens = tonumber(values[1]) or capacity
local updated_at = tonumber(values[2]) or now_ms

local elapsed_seconds = math.max(
    0,
    (now_ms - updated_at) / 1000
)

tokens = math.min(
    capacity,
    tokens + elapsed_seconds * refill_rate
)

local allowed = 0

if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
end

redis.call(
    "HSET",
    key,
    "tokens",
    tokens,
    "updated_at",
    now_ms
)

redis.call("PEXPIRE", key, ttl_ms)

local retry_ms = 0

if allowed == 0 then
    retry_ms = math.ceil(
        math.max(0, cost - tokens)
        / refill_rate * 1000
    )
end

local reset_ms = math.ceil(
    (capacity - tokens)
    / refill_rate * 1000
)

return {
    allowed,
    math.max(0, math.floor(tokens)),
    reset_ms,
    retry_ms
}
"""


# ============================================================
# Redis Store
# ============================================================


class RedisRateLimitStore:
    """
    Distributed rate limiter using atomic Redis scripts.

    Requires:
        pip install redis

    Uses Redis server time to avoid application clock skew.
    """

    def __init__(
        self,
        redis_client: Any,
        *,
        owns_client: bool = False,
    ) -> None:
        self.redis = redis_client
        self.owns_client = owns_client

    @classmethod
    async def from_url(
        cls,
        redis_url: str,
    ) -> "RedisRateLimitStore":
        try:
            from redis.asyncio import Redis
        except ImportError as exc:
            raise RateLimitBackendError(
                "Install redis to use Redis rate limiting"
            ) from exc

        client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )

        await client.ping()

        return cls(
            client,
            owns_client=True,
        )

    async def check(
        self,
        *,
        key: str,
        policy: RateLimitPolicy,
        cost: int = 1,
    ) -> RateLimitDecision:
        if cost <= 0:
            raise ValueError(
                "Rate limit cost must be positive"
            )

        try:
            if policy.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                result = await self.redis.eval(
                    FIXED_WINDOW_LUA,
                    1,
                    key,
                    policy.limit,
                    policy.window_seconds * 1000,
                    cost,
                )

            elif policy.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                import uuid

                result = await self.redis.eval(
                    SLIDING_WINDOW_LUA,
                    1,
                    key,
                    policy.limit,
                    policy.window_seconds * 1000,
                    cost,
                    uuid.uuid4().hex,
                )

            else:
                capacity = (
                    policy.burst_capacity
                    or policy.limit
                )

                refill_rate = (
                    policy.limit
                    / policy.window_seconds
                )

                result = await self.redis.eval(
                    TOKEN_BUCKET_LUA,
                    1,
                    key,
                    capacity,
                    refill_rate,
                    cost,
                    max(
                        1000,
                        policy.window_seconds * 2000,
                    ),
                )

            allowed = bool(int(result[0]))
            remaining = int(result[1])
            reset_after = float(result[2]) / 1000

            retry_after = (
                float(result[3]) / 1000
                if len(result) > 3
                else (
                    0.0
                    if allowed
                    else reset_after
                )
            )

            return RateLimitDecision(
                allowed=allowed,
                limit=(
                    policy.burst_capacity
                    if policy.algorithm
                    == RateLimitAlgorithm.TOKEN_BUCKET
                    and policy.burst_capacity
                    else policy.limit
                ),
                remaining=max(0, remaining),
                reset_after_seconds=max(
                    0.0, reset_after
                ),
                retry_after_seconds=max(
                    0.0, retry_after
                ),
                policy_name=policy.name,
                backend="redis",
            )

        except Exception as exc:
            logger.exception(
                "Redis rate limiting failed"
            )

            raise RateLimitBackendError(
                "Distributed rate limiter unavailable"
            ) from exc

    async def close(self) -> None:
        if self.owns_client:
            await self.redis.aclose()


# ============================================================
# Default Policies
# ============================================================


DEFAULT_POLICIES: dict[
    RateLimitCategory, RateLimitPolicy
] = {
    RateLimitCategory.DEFAULT: RateLimitPolicy(
        name="default",
        limit=100,
        window_seconds=60,
        scope=RateLimitScope.IP,
    ),

    RateLimitCategory.AUTHENTICATION: RateLimitPolicy(
        name="authentication",
        limit=10,
        window_seconds=60,
        scope=RateLimitScope.IP,
        fail_closed=True,
    ),

    RateLimitCategory.AI_INFERENCE: RateLimitPolicy(
        name="ai_inference",
        limit=30,
        window_seconds=60,
        scope=RateLimitScope.USER,
    ),

    RateLimitCategory.RAG: RateLimitPolicy(
        name="rag",
        limit=60,
        window_seconds=60,
        scope=RateLimitScope.USER,
    ),

    RateLimitCategory.FILE_UPLOAD: RateLimitPolicy(
        name="file_upload",
        limit=10,
        window_seconds=60,
        scope=RateLimitScope.USER,
    ),

    RateLimitCategory.TELEMETRY: RateLimitPolicy(
        name="telemetry",
        limit=1000,
        window_seconds=60,
        scope=RateLimitScope.TENANT,
    ),

    RateLimitCategory.ROBOT_CONTROL: RateLimitPolicy(
        name="robot_control",
        limit=30,
        window_seconds=60,
        scope=RateLimitScope.USER,
        fail_closed=True,
    ),

    RateLimitCategory.NAVIGATION: RateLimitPolicy(
        name="navigation",
        limit=20,
        window_seconds=60,
        scope=RateLimitScope.USER,
        fail_closed=True,
    ),

    RateLimitCategory.MANIPULATION: RateLimitPolicy(
        name="manipulation",
        limit=20,
        window_seconds=60,
        scope=RateLimitScope.USER,
        fail_closed=True,
    ),

    RateLimitCategory.MODEL_TRAINING: RateLimitPolicy(
        name="model_training",
        limit=5,
        window_seconds=3600,
        scope=RateLimitScope.USER,
    ),

    RateLimitCategory.EVALUATION: RateLimitPolicy(
        name="evaluation",
        limit=30,
        window_seconds=60,
        scope=RateLimitScope.USER,
    ),

    RateLimitCategory.MAINTENANCE: RateLimitPolicy(
        name="maintenance",
        limit=60,
        window_seconds=60,
        scope=RateLimitScope.USER,
    ),

    RateLimitCategory.REPAIR: RateLimitPolicy(
        name="repair",
        limit=30,
        window_seconds=60,
        scope=RateLimitScope.USER,
    ),

    RateLimitCategory.ADMINISTRATION: RateLimitPolicy(
        name="administration",
        limit=20,
        window_seconds=60,
        scope=RateLimitScope.USER,
        fail_closed=True,
    ),
}


# ============================================================
# Rate Limiter Service
# ============================================================


class RateLimiter:
    """
    Central rate-limit service.

    Redis is recommended for multi-worker deployments.
    """

    def __init__(
        self,
        *,
        store: RateLimitStore,
        settings: RateLimitSettings | None = None,
    ) -> None:
        self.store = store
        self.settings = (
            settings or RateLimitSettings()
        )

        self._policies = dict(DEFAULT_POLICIES)

    def register_policy(
        self,
        category: RateLimitCategory,
        policy: RateLimitPolicy,
    ) -> None:
        self._policies[category] = policy

    def get_policy(
        self,
        category: RateLimitCategory,
    ) -> RateLimitPolicy:
        return self._policies[category]

    async def check(
        self,
        *,
        identifier: str,
        policy: RateLimitPolicy,
        cost: int = 1,
    ) -> RateLimitDecision:
        if not self.settings.enabled or not policy.enabled:
            return RateLimitDecision(
                allowed=True,
                limit=policy.limit,
                remaining=policy.limit,
                reset_after_seconds=0,
                policy_name=policy.name,
                backend="disabled",
            )

        key = build_rate_limit_key(
            prefix=self.settings.key_prefix,
            policy=policy,
            identifier=identifier,
        )

        try:
            return await self.store.check(
                key=key,
                policy=policy,
                cost=cost,
            )

        except RateLimitBackendError:
            if (
                policy.fail_closed
                or self.settings.fail_closed
            ):
                raise

            logger.warning(
                "Rate limiting bypassed due to "
                "backend failure"
            )

            return RateLimitDecision(
                allowed=True,
                limit=policy.limit,
                remaining=0,
                reset_after_seconds=0,
                policy_name=policy.name,
                backend="unavailable",
            )

    async def enforce(
        self,
        *,
        identifier: str,
        policy: RateLimitPolicy,
        cost: int = 1,
    ) -> RateLimitDecision:
        decision = await self.check(
            identifier=identifier,
            policy=policy,
            cost=cost,
        )

        if not decision.allowed:
            raise RateLimitExceededError(
                retry_after=decision.retry_after_seconds,
                limit=decision.limit,
                remaining=decision.remaining,
            )

        return decision

    async def close(self) -> None:
        await self.store.close()


# ============================================================
# Trusted Request Identity
# ============================================================


def get_client_ip(request: Request) -> str:
    """
    Uses the ASGI client address.

    Configure trusted proxy handling at the ASGI server
    or ingress; do not parse arbitrary X-Forwarded-For
    headers here.
    """

    client = request.client

    if client is None:
        raise RateLimitIdentityError(
            "Client IP unavailable"
        )

    try:
        return str(
            ipaddress.ip_address(client.host)
        )
    except ValueError as exc:
        raise RateLimitIdentityError(
            "Invalid client IP"
        ) from exc


def resolve_rate_limit_identity(
    request: Request,
    scope: RateLimitScope,
) -> str:
    """
    Resolve an identity from trusted request state.

    Expected authenticated principal:
        request.state.auth_principal

    Principal must be created by verified authentication.
    """

    if scope == RateLimitScope.GLOBAL:
        return "global"

    if scope == RateLimitScope.IP:
        return get_client_ip(request)

    principal = getattr(
        request.state,
        "auth_principal",
        None,
    )

    if principal is None:
        raise RateLimitIdentityError(
            "Authenticated identity required"
        )

    if scope == RateLimitScope.USER:
        value = getattr(
            principal,
            "user_id",
            None,
        )

    elif scope == RateLimitScope.TENANT:
        value = getattr(
            principal,
            "tenant_id",
            None,
        )

    elif scope == RateLimitScope.API_KEY:
        value = getattr(
            principal,
            "api_key_id",
            None,
        )

    elif scope == RateLimitScope.ROBOT:
        value = getattr(
            request.state,
            "authorized_robot_id",
            None,
        )

    elif scope == RateLimitScope.FLEET:
        value = getattr(
            request.state,
            "authorized_fleet_id",
            None,
        )

    else:
        value = None

    if not value:
        raise RateLimitIdentityError(
            f"Missing identity for scope {scope.value}"
        )

    return str(value)


# ============================================================
# FastAPI Dependency
# ============================================================


def require_rate_limit(
    policy: RateLimitPolicy,
) -> Callable[..., Any]:
    """
    Usage:

        @router.post(
            "/predict",
            dependencies=[
                Depends(
                    require_rate_limit(
                        RateLimitPolicy(
                            name="predict",
                            limit=20,
                            window_seconds=60,
                            scope=RateLimitScope.USER,
                        )
                    )
                )
            ],
        )
        async def predict():
            ...
    """

    async def dependency(
        request: Request,
    ) -> RateLimitDecision:
        limiter = get_rate_limiter()

        try:
            identifier = resolve_rate_limit_identity(
                request,
                policy.scope,
            )

            decision = await limiter.enforce(
                identifier=identifier,
                policy=policy,
            )

        except RateLimitIdentityError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Rate limit identity unavailable",
            ) from exc

        except RateLimitExceededError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests",
                headers={
                    "Retry-After": str(
                        max(
                            1,
                            math.ceil(exc.retry_after),
                        )
                    ),
                },
            ) from exc

        except RateLimitBackendError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limiting service unavailable",
            ) from exc

        return decision

    return dependency


# ============================================================
# ASGI Middleware
# ============================================================


class RateLimitMiddleware:
    """
    Applies a global IP-based rate limit.

    Per-user and per-tenant limits should be applied using
    authenticated FastAPI dependencies.

    Do not use this middleware as the only control for
    sensitive robotics operations.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: RateLimiter,
        policy: RateLimitPolicy | None = None,
        excluded_paths: frozenset[str] = (
            DEFAULT_EXCLUDED_PATHS
        ),
        safety_exempt_paths: frozenset[str] = (
            SAFETY_EXEMPT_PATHS
        ),
    ) -> None:
        self.app = app
        self.limiter = limiter

        self.policy = policy or RateLimitPolicy(
            name="global_ip",
            limit=100,
            window_seconds=60,
            scope=RateLimitScope.IP,
        )

        if self.policy.scope != RateLimitScope.IP:
            raise ValueError(
                "Global middleware requires an IP policy"
            )

        self.excluded_paths = excluded_paths
        self.safety_exempt_paths = safety_exempt_paths

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")

        if (
            path in self.excluded_paths
            or path in self.safety_exempt_paths
        ):
            await self.app(scope, receive, send)
            return

        client = scope.get("client")

        if not client:
            await self._send_error(
                scope,
                receive,
                send,
                status_code=400,
                code="CLIENT_IDENTITY_UNAVAILABLE",
                message="Client identity unavailable",
            )
            return

        try:
            identifier = str(
                ipaddress.ip_address(client[0])
            )

            decision = await self.limiter.check(
                identifier=identifier,
                policy=self.policy,
            )

        except (
            RateLimitBackendError,
            RateLimitIdentityError,
        ):
            await self._send_error(
                scope,
                receive,
                send,
                status_code=503,
                code="RATE_LIMIT_UNAVAILABLE",
                message="Rate limiting service unavailable",
            )
            return

        if not decision.allowed:
            response = JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests",
                    },
                },
                headers=decision.to_headers(),
            )

            await response(scope, receive, send)
            return

        async def send_wrapper(
            message: Message,
        ) -> None:
            if message["type"] == "http.response.start":
                headers = list(
                    message.get("headers", [])
                )

                for name, value in (
                    decision.to_headers().items()
                ):
                    headers.append((
                        name.lower().encode("ascii"),
                        value.encode("ascii"),
                    ))

                message = {
                    **message,
                    "headers": headers,
                }

            await send(message)

        await self.app(
            scope,
            receive,
            send_wrapper,
        )

    @staticmethod
    async def _send_error(
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        status_code: int,
        code: str,
        message: str,
    ) -> None:
        response = JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "error": {
                    "code": code,
                    "message": message,
                },
            },
        )

        await response(scope, receive, send)


# ============================================================
# Initialization
# ============================================================


_rate_limiter: RateLimiter | None = None
_rate_limiter_lock = asyncio.Lock()


async def initialize_rate_limiter(
    settings: RateLimitSettings | None = None,
) -> RateLimiter:
    global _rate_limiter

    config = settings or RateLimitSettings()

    async with _rate_limiter_lock:
        if _rate_limiter is not None:
            return _rate_limiter

        if config.backend == RateLimitBackend.REDIS:
            store: RateLimitStore = (
                await RedisRateLimitStore.from_url(
                    config.redis_url
                )
            )
        else:
            store = MemoryRateLimitStore(
                max_buckets=config.max_memory_buckets
            )

        _rate_limiter = RateLimiter(
            store=store,
            settings=config,
        )

        logger.info(
            "Rate limiter initialized",
            extra={
                "backend": config.backend.value,
                "enabled": config.enabled,
            },
        )

        return _rate_limiter


def get_rate_limiter() -> RateLimiter:
    if _rate_limiter is None:
        raise RuntimeError(
            "Rate limiter is not initialized. "
            "Call initialize_rate_limiter() during startup."
        )

    return _rate_limiter


async def close_rate_limiter() -> None:
    global _rate_limiter

    async with _rate_limiter_lock:
        if _rate_limiter is not None:
            await _rate_limiter.close()
            _rate_limiter = None


# ============================================================
# FastAPI Integration
# ============================================================


def setup_rate_limit_middleware(
    app: Any,
    *,
    limiter: RateLimiter,
    policy: RateLimitPolicy | None = None,
) -> None:
    """
    Register middleware before application startup.

    Example:

        limiter = await initialize_rate_limiter(
            RateLimitSettings(
                backend=RateLimitBackend.REDIS
            )
        )

        setup_rate_limit_middleware(
            app,
            limiter=limiter,
        )
    """

    app.add_middleware(
        RateLimitMiddleware,
        limiter=limiter,
        policy=policy,
    )


# ============================================================
# Exports
# ============================================================


__all__ = [
    "RateLimitAlgorithm",
    "RateLimitScope",
    "RateLimitBackend",
    "RateLimitCategory",
    "RateLimitError",
    "RateLimitExceededError",
    "RateLimitBackendError",
    "RateLimitIdentityError",
    "RateLimitPolicy",
    "RateLimitDecision",
    "RateLimitSettings",
    "RateLimitStore",
    "MemoryRateLimitStore",
    "RedisRateLimitStore",
    "RateLimiter",
    "RateLimitMiddleware",
    "DEFAULT_POLICIES",
    "SAFETY_EXEMPT_PATHS",
    "hash_identifier",
    "build_rate_limit_key",
    "get_client_ip",
    "resolve_rate_limit_identity",
    "require_rate_limit",
    "initialize_rate_limiter",
    "get_rate_limiter",
    "close_rate_limiter",
    "setup_rate_limit_middleware",
]
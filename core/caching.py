"""Caching module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Core Caching Module

Features:
- Async Redis caching
- In-memory TTL fallback
- JSON serialization
- Cache namespaces
- Cache key generation
- Cache expiration
- Cache invalidation
- Cache-aside decorator
- Distributed locks
- Cache statistics
- Robot telemetry caching
- Robot health caching
- Sensor state caching
- Fleet dashboard caching
- Model inference caching

Security:
- Tenant-aware cache keys
- No pickle deserialization
- No plaintext credential caching
- Redis TLS support through connection URL
- Safe lock ownership verification
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
import uuid

from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    ParamSpec,
    TypeVar,
)

try:
    from redis.asyncio import Redis
    from redis.exceptions import RedisError
except ImportError:
    Redis = None

    class RedisError(Exception):
        pass


logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# ============================================================
# Configuration
# ============================================================


class CacheBackend(str, Enum):
    MEMORY = "memory"
    REDIS = "redis"


class CacheNamespace(str, Enum):
    GENERAL = "general"
    ROBOT = "robot"
    ROBOT_STATE = "robot_state"
    ROBOT_HEALTH = "robot_health"
    ROBOT_TELEMETRY = "robot_telemetry"
    SENSOR = "sensor"
    SENSOR_HEALTH = "sensor_health"
    FLEET = "fleet"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    WORLD_MODEL = "world_model"
    VLA = "vla"
    RAG = "rag"
    MODEL = "model"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    SIMULATION = "simulation"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    PARTS = "parts"
    DASHBOARD = "dashboard"
    RATE_LIMIT = "rate_limit"


@dataclass(frozen=True)
class CacheSettings:
    backend: CacheBackend = CacheBackend.MEMORY
    redis_url: str = "redis://localhost:6379/0"
    key_prefix: str = "physical_ai"
    default_ttl_seconds: int = 300
    max_memory_entries: int = 10_000
    redis_socket_timeout: float = 5.0
    redis_connect_timeout: float = 5.0
    fallback_to_memory: bool = True

    def __post_init__(self) -> None:
        if self.default_ttl_seconds <= 0:
            raise ValueError("Cache TTL must be positive")

        if self.max_memory_entries <= 0:
            raise ValueError(
                "Maximum memory entries must be positive"
            )

        if not self.key_prefix.strip():
            raise ValueError("Cache key prefix is required")


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    deletes: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "deletes": self.deletes,
            "errors": self.errors,
            "hit_rate": round(self.hit_rate, 4),
        }


# ============================================================
# JSON Serialization
# ============================================================


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, uuid.UUID):
        return str(value)

    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)

    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)

    if isinstance(value, bytes):
        raise TypeError(
            "Binary data must be stored separately"
        )

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def serialize_cache_value(value: Any) -> str:
    return json.dumps(
        value,
        default=json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def deserialize_cache_value(value: str | bytes) -> Any:
    return json.loads(value)


# ============================================================
# Cache Key Utilities
# ============================================================


def normalize_key_component(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value

    if isinstance(value, uuid.UUID):
        value = str(value)

    if value is None:
        return "none"

    if isinstance(value, (str, int, float, bool)):
        text = str(value)
    else:
        text = serialize_cache_value(value)

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest() if len(text) > 128 else text.replace(":", "_")


def build_cache_key(
    namespace: CacheNamespace | str,
    *,
    tenant_id: str,
    identifier: str | None = None,
    prefix: str = "physical_ai",
    **dimensions: Any,
) -> str:
    if not tenant_id or not tenant_id.strip():
        raise ValueError("tenant_id is required")

    namespace_value = (
        namespace.value
        if isinstance(namespace, CacheNamespace)
        else namespace
    )

    if not namespace_value:
        raise ValueError("Cache namespace is required")

    components = [
        normalize_key_component(prefix),
        normalize_key_component(tenant_id),
        normalize_key_component(namespace_value),
    ]

    if identifier is not None:
        components.append(normalize_key_component(identifier))

    for name, value in sorted(dimensions.items()):
        components.append(
            normalize_key_component(name)
        )
        components.append(
            normalize_key_component(value)
        )

    return ":".join(components)


def hash_cache_arguments(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    payload = serialize_cache_value(
        {
            "args": args,
            "kwargs": kwargs,
        }
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


# ============================================================
# Memory Cache
# ============================================================


@dataclass
class MemoryCacheEntry:
    value: str
    expires_at: float


class MemoryCache:
    """
    Async-safe in-process TTL cache.

    Suitable for development, testing, and non-distributed
    deployments. Not shared between workers.
    """

    def __init__(
        self,
        max_entries: int = 10_000,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self.max_entries = max_entries

        self._entries: OrderedDict[
            str, MemoryCacheEntry
        ] = OrderedDict()

        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                return None

            if time.monotonic() >= entry.expires_at:
                del self._entries[key]
                return None

            self._entries.move_to_end(key)

            return entry.value

    async def set(
        self,
        key: str,
        value: str,
        ttl_seconds: int,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("TTL must be positive")

        async with self._lock:
            self._entries[key] = MemoryCacheEntry(
                value=value,
                expires_at=(
                    time.monotonic() + ttl_seconds
                ),
            )

            self._entries.move_to_end(key)

            await self._evict_expired_locked()

            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    async def delete(self, key: str) -> bool:
        async with self._lock:
            return self._entries.pop(key, None) is not None

    async def delete_prefix(self, prefix: str) -> int:
        async with self._lock:
            keys = [
                key
                for key in self._entries
                if key.startswith(prefix)
            ]

            for key in keys:
                del self._entries[key]

            return len(keys)

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    async def size(self) -> int:
        async with self._lock:
            await self._evict_expired_locked()
            return len(self._entries)

    async def _evict_expired_locked(self) -> None:
        now = time.monotonic()

        expired = [
            key
            for key, entry in self._entries.items()
            if entry.expires_at <= now
        ]

        for key in expired:
            del self._entries[key]


# ============================================================
# Redis Cache
# ============================================================


class RedisCache:
    """
    Async Redis cache adapter.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        socket_timeout: float = 5.0,
        connect_timeout: float = 5.0,
    ) -> None:
        if Redis is None:
            raise RuntimeError(
                "Redis package is not installed. "
                "Install redis>=5.0."
            )

        self.client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=connect_timeout,
            health_check_interval=30,
        )

    async def ping(self) -> bool:
        return bool(await self.client.ping())

    async def get(self, key: str) -> str | None:
        return await self.client.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ttl_seconds: int,
    ) -> None:
        await self.client.set(
            key,
            value,
            ex=ttl_seconds,
        )

    async def delete(self, key: str) -> bool:
        return bool(await self.client.delete(key))

    async def delete_prefix(self, prefix: str) -> int:
        deleted = 0
        batch: list[str] = []

        async for key in self.client.scan_iter(
            match=f"{prefix}*",
            count=500,
        ):
            batch.append(key)

            if len(batch) >= 500:
                deleted += await self.client.delete(*batch)
                batch.clear()

        if batch:
            deleted += await self.client.delete(*batch)

        return deleted

    async def exists(self, key: str) -> bool:
        return bool(await self.client.exists(key))

    async def close(self) -> None:
        await self.client.aclose()


# ============================================================
# Cache Manager
# ============================================================


class CacheManager:
    """
    Application cache manager.

    Redis is preferred for distributed production workloads.
    Memory fallback is optional and is not a substitute for
    distributed consistency.
    """

    def __init__(
        self,
        settings: CacheSettings | None = None,
    ) -> None:
        self.settings = settings or CacheSettings()

        self.memory = MemoryCache(
            max_entries=self.settings.max_memory_entries
        )

        self.redis: RedisCache | None = None

        self.stats = CacheStats()

        self._initialized = False
        self._redis_available = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        if self.settings.backend == CacheBackend.REDIS:
            try:
                self.redis = RedisCache(
                    self.settings.redis_url,
                    socket_timeout=(
                        self.settings.redis_socket_timeout
                    ),
                    connect_timeout=(
                        self.settings.redis_connect_timeout
                    ),
                )

                self._redis_available = await self.redis.ping()

                if not self._redis_available:
                    raise ConnectionError(
                        "Redis health check failed"
                    )

                logger.info("Redis cache initialized")

            except (RedisError, OSError, ConnectionError) as exc:
                self.stats.errors += 1
                self._redis_available = False

                logger.warning(
                    "Redis cache initialization failed: %s",
                    exc,
                )

                if not self.settings.fallback_to_memory:
                    raise

        self._initialized = True

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    @property
    def active_backend(self) -> str:
        if self._redis_available and self.redis is not None:
            return CacheBackend.REDIS.value

        return CacheBackend.MEMORY.value

    async def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        await self._ensure_initialized()

        raw: str | None = None

        if self._redis_available and self.redis is not None:
            try:
                raw = await self.redis.get(key)
            except (RedisError, OSError) as exc:
                self.stats.errors += 1

                logger.error(
                    "Redis GET failed: %s",
                    exc,
                )

                if not self.settings.fallback_to_memory:
                    raise

                raw = await self.memory.get(key)
        else:
            raw = await self.memory.get(key)

        if raw is None:
            self.stats.misses += 1
            return default

        try:
            value = deserialize_cache_value(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.stats.errors += 1
            self.stats.misses += 1

            logger.warning(
                "Invalid cached JSON for key: %s",
                key,
            )

            return default

        self.stats.hits += 1

        return value

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        await self._ensure_initialized()

        ttl = (
            ttl_seconds
            if ttl_seconds is not None
            else self.settings.default_ttl_seconds
        )

        if ttl <= 0:
            raise ValueError("Cache TTL must be positive")

        serialized = serialize_cache_value(value)

        if self._redis_available and self.redis is not None:
            try:
                await self.redis.set(
                    key,
                    serialized,
                    ttl,
                )

                self.stats.writes += 1
                return True

            except (RedisError, OSError) as exc:
                self.stats.errors += 1

                logger.error(
                    "Redis SET failed: %s",
                    exc,
                )

                if not self.settings.fallback_to_memory:
                    raise

        await self.memory.set(
            key,
            serialized,
            ttl,
        )

        self.stats.writes += 1
        return True

    async def delete(self, key: str) -> bool:
        await self._ensure_initialized()

        deleted = False

        if self._redis_available and self.redis is not None:
            try:
                deleted = await self.redis.delete(key)
            except (RedisError, OSError) as exc:
                self.stats.errors += 1

                logger.error(
                    "Redis DELETE failed: %s",
                    exc,
                )

                if not self.settings.fallback_to_memory:
                    raise

        memory_deleted = await self.memory.delete(key)

        if deleted or memory_deleted:
            self.stats.deletes += 1

        return deleted or memory_deleted

    async def delete_prefix(
        self,
        prefix: str,
    ) -> int:
        await self._ensure_initialized()

        deleted = 0

        if self._redis_available and self.redis is not None:
            try:
                deleted += await self.redis.delete_prefix(prefix)
            except (RedisError, OSError) as exc:
                self.stats.errors += 1

                logger.error(
                    "Redis prefix deletion failed: %s",
                    exc,
                )

                if not self.settings.fallback_to_memory:
                    raise

        deleted += await self.memory.delete_prefix(prefix)

        self.stats.deletes += deleted

        return deleted

    async def exists(self, key: str) -> bool:
        sentinel = object()
        return await self.get(key, sentinel) is not sentinel

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
        *,
        ttl_seconds: int | None = None,
    ) -> T:
        sentinel = object()

        cached = await self.get(key, sentinel)

        if cached is not sentinel:
            return cached

        value = await factory()

        await self.set(
            key,
            value,
            ttl_seconds=ttl_seconds,
        )

        return value

    async def invalidate_namespace(
        self,
        namespace: CacheNamespace | str,
        *,
        tenant_id: str,
    ) -> int:
        prefix = build_cache_key(
            namespace,
            tenant_id=tenant_id,
            prefix=self.settings.key_prefix,
        )

        return await self.delete_prefix(prefix + ":")

    async def health_check(self) -> dict[str, Any]:
        await self._ensure_initialized()

        redis_healthy = False

        if self.redis is not None and self._redis_available:
            try:
                redis_healthy = await self.redis.ping()
            except (RedisError, OSError):
                redis_healthy = False

        return {
            "backend": self.active_backend,
            "redis_healthy": redis_healthy,
            "memory_entries": await self.memory.size(),
            "statistics": self.stats.to_dict(),
            "checked_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    async def close(self) -> None:
        if self.redis is not None:
            await self.redis.close()
            self.redis = None

        await self.memory.clear()

        self._redis_available = False
        self._initialized = False


# ============================================================
# Cache Decorator
# ============================================================


def cached(
    *,
    namespace: CacheNamespace | str,
    ttl_seconds: int = 300,
    tenant_argument: str = "tenant_id",
    manager: CacheManager | None = None,
) -> Callable[
    [Callable[P, Awaitable[T]]],
    Callable[P, Awaitable[T]],
]:
    """
    Cache results of asynchronous functions.

    The decorated function must receive tenant_id as
    a keyword argument to preserve tenant isolation.

    Example:

        @cached(namespace=CacheNamespace.ROBOT_HEALTH)
        async def get_health(
            *,
            tenant_id: str,
            robot_id: str,
        ):
            ...
    """

    if ttl_seconds <= 0:
        raise ValueError("TTL must be positive")

    def decorator(
        function: Callable[P, Awaitable[T]],
    ) -> Callable[P, Awaitable[T]]:

        @functools.wraps(function)
        async def wrapper(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> T:
            tenant_id = kwargs.get(tenant_argument)

            if not isinstance(tenant_id, str) or not tenant_id:
                raise ValueError(
                    f"Required tenant argument: {tenant_argument}"
                )

            cache = manager or get_cache_manager()

            argument_hash = hash_cache_arguments(
                args,
                dict(kwargs),
            )

            key = build_cache_key(
                namespace,
                tenant_id=tenant_id,
                identifier=argument_hash,
                prefix=cache.settings.key_prefix,
            )

            return await cache.get_or_set(
                key,
                lambda: function(*args, **kwargs),
                ttl_seconds=ttl_seconds,
            )

        return wrapper

    return decorator


# ============================================================
# Distributed Redis Lock
# ============================================================


class CacheLockError(RuntimeError):
    pass


class RedisDistributedLock:
    """
    Redis distributed lock using unique ownership tokens.

    Suitable for cache coordination and duplicate-work
    suppression.

    Do not use this lock as the sole safety mechanism
    for physical robot commands or emergency operations.
    """

    RELEASE_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    end
    return 0
    """

    def __init__(
        self,
        redis_cache: RedisCache,
        key: str,
        *,
        ttl_seconds: int = 30,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Lock TTL must be positive")

        self.redis_cache = redis_cache
        self.key = key
        self.ttl_seconds = ttl_seconds
        self.token = str(uuid.uuid4())
        self.acquired = False

    async def acquire(self) -> bool:
        result = await self.redis_cache.client.set(
            self.key,
            self.token,
            nx=True,
            ex=self.ttl_seconds,
        )

        self.acquired = bool(result)

        return self.acquired

    async def release(self) -> bool:
        if not self.acquired:
            return False

        result = await self.redis_cache.client.eval(
            self.RELEASE_SCRIPT,
            1,
            self.key,
            self.token,
        )

        self.acquired = False

        return bool(result)


@asynccontextmanager
async def distributed_lock(
    manager: CacheManager,
    key: str,
    *,
    ttl_seconds: int = 30,
    wait_timeout_seconds: float = 5.0,
    retry_interval_seconds: float = 0.1,
) -> AsyncIterator[None]:
    """
    Acquire a Redis-backed distributed lock.

    Fails closed when Redis is unavailable.
    """

    await manager.initialize()

    if not manager._redis_available or manager.redis is None:
        raise CacheLockError(
            "Distributed locking requires Redis"
        )

    if wait_timeout_seconds < 0:
        raise ValueError("Wait timeout cannot be negative")

    if retry_interval_seconds <= 0:
        raise ValueError("Retry interval must be positive")

    lock = RedisDistributedLock(
        manager.redis,
        key,
        ttl_seconds=ttl_seconds,
    )

    deadline = time.monotonic() + wait_timeout_seconds

    while True:
        if await lock.acquire():
            break

        if time.monotonic() >= deadline:
            raise CacheLockError(
                f"Unable to acquire distributed lock: {key}"
            )

        await asyncio.sleep(retry_interval_seconds)

    try:
        yield
    finally:
        await lock.release()


# ============================================================
# Robotics Cache Service
# ============================================================


class RoboticsCacheService:
    """
    High-level cache operations for robotics workloads.
    """

    def __init__(
        self,
        manager: CacheManager,
    ) -> None:
        self.manager = manager

    def _key(
        self,
        namespace: CacheNamespace,
        tenant_id: str,
        identifier: str,
    ) -> str:
        return build_cache_key(
            namespace,
            tenant_id=tenant_id,
            identifier=identifier,
            prefix=self.manager.settings.key_prefix,
        )

    async def cache_robot_state(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        state: dict[str, Any],
        ttl_seconds: int = 10,
    ) -> bool:
        key = self._key(
            CacheNamespace.ROBOT_STATE,
            tenant_id,
            robot_id,
        )

        return await self.manager.set(
            key,
            state,
            ttl_seconds=ttl_seconds,
        )

    async def get_robot_state(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any] | None:
        key = self._key(
            CacheNamespace.ROBOT_STATE,
            tenant_id,
            robot_id,
        )

        return await self.manager.get(key)

    async def cache_robot_health(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        health: dict[str, Any],
        ttl_seconds: int = 15,
    ) -> bool:
        key = self._key(
            CacheNamespace.ROBOT_HEALTH,
            tenant_id,
            robot_id,
        )

        return await self.manager.set(
            key,
            health,
            ttl_seconds=ttl_seconds,
        )

    async def get_robot_health(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any] | None:
        key = self._key(
            CacheNamespace.ROBOT_HEALTH,
            tenant_id,
            robot_id,
        )

        return await self.manager.get(key)

    async def cache_sensor_health(
        self,
        *,
        tenant_id: str,
        sensor_id: str,
        health: dict[str, Any],
        ttl_seconds: int = 15,
    ) -> bool:
        key = self._key(
            CacheNamespace.SENSOR_HEALTH,
            tenant_id,
            sensor_id,
        )

        return await self.manager.set(
            key,
            health,
            ttl_seconds=ttl_seconds,
        )

    async def get_sensor_health(
        self,
        *,
        tenant_id: str,
        sensor_id: str,
    ) -> dict[str, Any] | None:
        key = self._key(
            CacheNamespace.SENSOR_HEALTH,
            tenant_id,
            sensor_id,
        )

        return await self.manager.get(key)

    async def cache_robot_telemetry(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        telemetry: dict[str, Any],
        ttl_seconds: int = 5,
    ) -> bool:
        key = self._key(
            CacheNamespace.ROBOT_TELEMETRY,
            tenant_id,
            robot_id,
        )

        return await self.manager.set(
            key,
            telemetry,
            ttl_seconds=ttl_seconds,
        )

    async def get_robot_telemetry(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any] | None:
        key = self._key(
            CacheNamespace.ROBOT_TELEMETRY,
            tenant_id,
            robot_id,
        )

        return await self.manager.get(key)

    async def cache_fleet_dashboard(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        dashboard: dict[str, Any],
        ttl_seconds: int = 30,
    ) -> bool:
        key = self._key(
            CacheNamespace.FLEET,
            tenant_id,
            fleet_id,
        )

        return await self.manager.set(
            key,
            dashboard,
            ttl_seconds=ttl_seconds,
        )

    async def get_fleet_dashboard(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
    ) -> dict[str, Any] | None:
        key = self._key(
            CacheNamespace.FLEET,
            tenant_id,
            fleet_id,
        )

        return await self.manager.get(key)

    async def cache_inference(
        self,
        *,
        tenant_id: str,
        model_id: str,
        input_hash: str,
        output: dict[str, Any],
        ttl_seconds: int = 300,
    ) -> bool:
        key = build_cache_key(
            CacheNamespace.INFERENCE,
            tenant_id=tenant_id,
            identifier=model_id,
            input_hash=input_hash,
            prefix=self.manager.settings.key_prefix,
        )

        return await self.manager.set(
            key,
            output,
            ttl_seconds=ttl_seconds,
        )

    async def get_inference(
        self,
        *,
        tenant_id: str,
        model_id: str,
        input_hash: str,
    ) -> dict[str, Any] | None:
        key = build_cache_key(
            CacheNamespace.INFERENCE,
            tenant_id=tenant_id,
            identifier=model_id,
            input_hash=input_hash,
            prefix=self.manager.settings.key_prefix,
        )

        return await self.manager.get(key)

    async def invalidate_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> int:
        namespaces = (
            CacheNamespace.ROBOT_STATE,
            CacheNamespace.ROBOT_HEALTH,
            CacheNamespace.ROBOT_TELEMETRY,
        )

        deleted = 0

        for namespace in namespaces:
            key = self._key(
                namespace,
                tenant_id,
                robot_id,
            )

            if await self.manager.delete(key):
                deleted += 1

        return deleted

    async def invalidate_sensor(
        self,
        *,
        tenant_id: str,
        sensor_id: str,
    ) -> bool:
        key = self._key(
            CacheNamespace.SENSOR_HEALTH,
            tenant_id,
            sensor_id,
        )

        return await self.manager.delete(key)


# ============================================================
# Application Singleton
# ============================================================


_cache_manager: CacheManager | None = None


def configure_cache(
    settings: CacheSettings,
) -> CacheManager:
    global _cache_manager

    if _cache_manager is not None:
        raise RuntimeError(
            "Cache manager is already configured. "
            "Close it before reconfiguration."
        )

    _cache_manager = CacheManager(settings)

    return _cache_manager


def get_cache_manager() -> CacheManager:
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = CacheManager()

    return _cache_manager


def get_robotics_cache() -> RoboticsCacheService:
    return RoboticsCacheService(
        get_cache_manager()
    )


async def initialize_cache() -> CacheManager:
    manager = get_cache_manager()

    await manager.initialize()

    return manager


async def close_cache() -> None:
    global _cache_manager

    if _cache_manager is not None:
        await _cache_manager.close()
        _cache_manager = None


# ============================================================
# Exports
# ============================================================


__all__ = [
    "CacheBackend",
    "CacheNamespace",
    "CacheSettings",
    "CacheStats",
    "MemoryCache",
    "RedisCache",
    "CacheManager",
    "CacheLockError",
    "RedisDistributedLock",
    "RoboticsCacheService",
    "serialize_cache_value",
    "deserialize_cache_value",
    "normalize_key_component",
    "build_cache_key",
    "hash_cache_arguments",
    "cached",
    "distributed_lock",
    "configure_cache",
    "get_cache_manager",
    "get_robotics_cache",
    "initialize_cache",
    "close_cache",
]
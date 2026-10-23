"""Event Bus module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Core Event Bus

Features:
- Asynchronous publish/subscribe event bus
- In-memory event transport
- Optional Redis Streams transport
- Multi-tenant event isolation
- Event schema validation
- Event handlers and subscriptions
- Retry with exponential backoff
- Dead-letter queue
- Event deduplication
- Correlation and causation IDs
- Event history
- Event metrics
- Graceful startup and shutdown

Safety:
Events are notifications, not authorization to actuate robots.
Physical robot commands require independent authentication,
authorization, safety validation, and hardware interlocks.
"""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import json
import logging
import random
import time
import uuid

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol

try:
    from redis.asyncio import Redis
except ImportError:
    Redis = None


logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    return str(uuid.uuid4())


def serialize_datetime(value: datetime) -> str:
    return value.isoformat()


def validate_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    if len(value) > 255:
        raise ValueError(f"{name} exceeds 255 characters")
    return value.strip()


def validate_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Event payload must be a dictionary")

    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Event payload must contain valid JSON values"
        ) from exc


# ============================================================
# Enumerations
# ============================================================


class EventPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EventStatus(str, Enum):
    CREATED = "created"
    PUBLISHED = "published"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"


class EventCategory(str, Enum):
    SYSTEM = "system"
    ROBOT = "robot"
    ROBOT_HEALTH = "robot_health"
    TELEMETRY = "telemetry"
    SENSOR = "sensor"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    SAFETY = "safety"
    INCIDENT = "incident"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    PARTS = "parts"
    FLEET = "fleet"
    MODEL = "model"
    INFERENCE = "inference"
    WORLD_MODEL = "world_model"
    VLA = "vla"
    SIMULATION = "simulation"
    EVALUATION = "evaluation"
    SECURITY = "security"
    AUDIT = "audit"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


class EventTransport(str, Enum):
    MEMORY = "memory"
    REDIS_STREAMS = "redis_streams"


class EventDeliveryMode(str, Enum):
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"


# ============================================================
# Event Model
# ============================================================


@dataclass(frozen=True, slots=True)
class Event:
    event_type: str
    tenant_id: str
    payload: dict[str, Any]

    event_id: str = field(default_factory=generate_uuid)
    category: EventCategory = EventCategory.SYSTEM
    priority: EventPriority = EventPriority.NORMAL

    source: str = "physical_ai"
    aggregate_id: str | None = None
    robot_id: str | None = None
    fleet_id: str | None = None
    sensor_id: str | None = None

    correlation_id: str | None = None
    causation_id: str | None = None
    trace_id: str | None = None

    schema_version: str = "1.0"
    occurred_at: datetime = field(default_factory=utc_now)

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier(self.event_type, "event_type")
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.event_id, "event_id")
        validate_identifier(self.source, "source")

        validate_payload(self.payload)
        validate_payload(self.metadata)

        if self.occurred_at.tzinfo is None:
            raise ValueError(
                "occurred_at must be timezone-aware"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "tenant_id": self.tenant_id,
            "payload": self.payload,
            "category": self.category.value,
            "priority": self.priority.value,
            "source": self.source,
            "aggregate_id": self.aggregate_id,
            "robot_id": self.robot_id,
            "fleet_id": self.fleet_id,
            "sensor_id": self.sensor_id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "trace_id": self.trace_id,
            "schema_version": self.schema_version,
            "occurred_at": serialize_datetime(self.occurred_at),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        return cls(
            event_id=data.get("event_id", generate_uuid()),
            event_type=data["event_type"],
            tenant_id=data["tenant_id"],
            payload=data["payload"],
            category=EventCategory(
                data.get("category", EventCategory.SYSTEM.value)
            ),
            priority=EventPriority(
                data.get("priority", EventPriority.NORMAL.value)
            ),
            source=data.get("source", "physical_ai"),
            aggregate_id=data.get("aggregate_id"),
            robot_id=data.get("robot_id"),
            fleet_id=data.get("fleet_id"),
            sensor_id=data.get("sensor_id"),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            trace_id=data.get("trace_id"),
            schema_version=data.get("schema_version", "1.0"),
            occurred_at=datetime.fromisoformat(
                data["occurred_at"]
            ) if data.get("occurred_at") else utc_now(),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, value: str) -> "Event":
        return cls.from_dict(json.loads(value))


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class EventBusSettings:
    transport: EventTransport = EventTransport.MEMORY

    redis_url: str = "redis://localhost:6379/0"
    redis_stream_prefix: str = "physical_ai:events"
    redis_max_stream_length: int = 100_000

    max_queue_size: int = 10_000
    worker_count: int = 4

    max_retries: int = 3
    retry_base_delay_seconds: float = 0.25
    retry_max_delay_seconds: float = 10.0

    event_history_size: int = 1000
    dead_letter_queue_size: int = 1000

    deduplication_ttl_seconds: int = 3600

    handler_timeout_seconds: float = 30.0

    delivery_mode: EventDeliveryMode = (
        EventDeliveryMode.AT_LEAST_ONCE
    )

    def __post_init__(self) -> None:
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")

        if self.worker_count <= 0:
            raise ValueError("worker_count must be positive")

        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        if self.handler_timeout_seconds <= 0:
            raise ValueError(
                "handler_timeout_seconds must be positive"
            )

        if self.redis_max_stream_length <= 0:
            raise ValueError(
                "redis_max_stream_length must be positive"
            )


# ============================================================
# Event Handler
# ============================================================


EventHandler = Callable[[Event], Awaitable[None]]


@dataclass
class EventSubscription:
    subscription_id: str
    event_pattern: str
    handler: EventHandler

    tenant_id: str | None = None
    enabled: bool = True

    max_retries: int | None = None
    timeout_seconds: float | None = None

    description: str | None = None

    def matches(self, event: Event) -> bool:
        if not self.enabled:
            return False

        if (
            self.tenant_id is not None
            and self.tenant_id != event.tenant_id
        ):
            return False

        return fnmatch.fnmatchcase(
            event.event_type,
            self.event_pattern,
        )


# ============================================================
# Metrics
# ============================================================


@dataclass
class EventBusMetrics:
    published: int = 0
    processed: int = 0
    failed: int = 0
    retried: int = 0
    dead_lettered: int = 0
    dropped: int = 0
    duplicate: int = 0

    handler_invocations: int = 0
    handler_failures: int = 0

    total_processing_seconds: float = 0.0

    started_at: datetime = field(default_factory=utc_now)

    @property
    def average_processing_ms(self) -> float:
        if self.processed == 0:
            return 0.0

        return (
            self.total_processing_seconds
            / self.processed
            * 1000.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "published": self.published,
            "processed": self.processed,
            "failed": self.failed,
            "retried": self.retried,
            "dead_lettered": self.dead_lettered,
            "dropped": self.dropped,
            "duplicate": self.duplicate,
            "handler_invocations": self.handler_invocations,
            "handler_failures": self.handler_failures,
            "average_processing_ms": round(
                self.average_processing_ms, 3
            ),
            "started_at": self.started_at.isoformat(),
        }


# ============================================================
# Dead Letter Event
# ============================================================


@dataclass
class DeadLetterEvent:
    event: Event
    subscription_id: str
    error_message: str
    attempts: int
    failed_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "subscription_id": self.subscription_id,
            "error_message": self.error_message,
            "attempts": self.attempts,
            "failed_at": self.failed_at.isoformat(),
        }


# ============================================================
# Exceptions
# ============================================================


class EventBusError(RuntimeError):
    pass


class EventBusNotRunningError(EventBusError):
    pass


class EventQueueFullError(EventBusError):
    pass


class EventHandlerError(EventBusError):
    pass


class EventPublishError(EventBusError):
    pass


# ============================================================
# Transport Protocol
# ============================================================


class EventPublisher(Protocol):
    async def publish(self, event: Event) -> None:
        ...

    async def close(self) -> None:
        ...


# ============================================================
# Redis Streams Publisher
# ============================================================


class RedisStreamsPublisher:
    """
    Redis Streams publisher.

    Publishing to Redis is durable only according to the
    Redis server's persistence configuration.

    Consumer groups, pending-entry recovery, and cross-process
    dispatch are separate concerns and are not implemented
    by this publisher.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        stream_prefix: str,
        max_stream_length: int,
    ) -> None:
        if Redis is None:
            raise RuntimeError(
                "Redis support requires redis>=5.0"
            )

        self.redis = Redis.from_url(
            redis_url,
            decode_responses=True,
        )

        self.stream_prefix = stream_prefix
        self.max_stream_length = max_stream_length

    def stream_name(self, event: Event) -> str:
        tenant_hash = uuid.uuid5(
            uuid.NAMESPACE_DNS,
            event.tenant_id,
        ).hex

        return f"{self.stream_prefix}:{tenant_hash}"

    async def publish(self, event: Event) -> None:
        await self.redis.xadd(
            self.stream_name(event),
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "tenant_id": event.tenant_id,
                "data": event.to_json(),
            },
            maxlen=self.max_stream_length,
            approximate=True,
        )

    async def close(self) -> None:
        await self.redis.aclose()


# ============================================================
# In-Memory Event Bus
# ============================================================


class EventBus:
    """
    Asynchronous event bus.

    In-memory delivery is local to one Python process.
    Subscribers are invoked independently.

    A handler failure does not prevent other handlers
    from processing the same event.
    """

    def __init__(
        self,
        settings: EventBusSettings | None = None,
    ) -> None:
        self.settings = settings or EventBusSettings()

        self._queue: asyncio.Queue[Event] = asyncio.Queue(
            maxsize=self.settings.max_queue_size
        )

        self._subscriptions: dict[
            str, EventSubscription
        ] = {}

        self._workers: list[asyncio.Task[None]] = []

        self._running = False
        self._accepting_events = False

        self._publisher: RedisStreamsPublisher | None = None

        self._history: deque[Event] = deque(
            maxlen=self.settings.event_history_size
        )

        self._dead_letters: deque[DeadLetterEvent] = deque(
            maxlen=self.settings.dead_letter_queue_size
        )

        self._seen_events: dict[str, float] = {}

        self._deduplication_lock = asyncio.Lock()

        self.metrics = EventBusMetrics()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def subscription_count(self) -> int:
        return len(self._subscriptions)

    async def start(self) -> None:
        if self._running:
            return

        if self.settings.transport == EventTransport.REDIS_STREAMS:
            self._publisher = RedisStreamsPublisher(
                self.settings.redis_url,
                stream_prefix=(
                    self.settings.redis_stream_prefix
                ),
                max_stream_length=(
                    self.settings.redis_max_stream_length
                ),
            )

            await self._publisher.redis.ping()

        self._running = True
        self._accepting_events = True

        for worker_index in range(
            self.settings.worker_count
        ):
            task = asyncio.create_task(
                self._worker_loop(worker_index),
                name=f"event-bus-worker-{worker_index}",
            )

            self._workers.append(task)

        logger.info(
            "Event bus started with %s workers",
            len(self._workers),
        )

    async def stop(
        self,
        *,
        drain: bool = True,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not self._running:
            return

        self._accepting_events = False

        if drain:
            try:
                await asyncio.wait_for(
                    self._queue.join(),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Event bus shutdown timed out while draining"
                )

        self._running = False

        for task in self._workers:
            task.cancel()

        await asyncio.gather(
            *self._workers,
            return_exceptions=True,
        )

        self._workers.clear()

        if self._publisher is not None:
            await self._publisher.close()
            self._publisher = None

        logger.info("Event bus stopped")

    def subscribe(
        self,
        event_pattern: str,
        handler: EventHandler,
        *,
        tenant_id: str | None = None,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
        description: str | None = None,
    ) -> str:
        if not event_pattern:
            raise ValueError("Event pattern is required")

        if not callable(handler):
            raise TypeError("Handler must be callable")

        if not inspect.iscoroutinefunction(handler):
            raise TypeError(
                "Event handlers must be async functions"
            )

        if tenant_id is not None:
            validate_identifier(tenant_id, "tenant_id")

        subscription_id = generate_uuid()

        self._subscriptions[subscription_id] = (
            EventSubscription(
                subscription_id=subscription_id,
                event_pattern=event_pattern,
                handler=handler,
                tenant_id=tenant_id,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                description=description,
            )
        )

        logger.debug(
            "Registered event subscription %s for %s",
            subscription_id,
            event_pattern,
        )

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        return (
            self._subscriptions.pop(
                subscription_id, None
            )
            is not None
        )

    def enable_subscription(
        self,
        subscription_id: str,
    ) -> None:
        subscription = self._subscriptions.get(
            subscription_id
        )

        if subscription is None:
            raise KeyError("Subscription not found")

        subscription.enabled = True

    def disable_subscription(
        self,
        subscription_id: str,
    ) -> None:
        subscription = self._subscriptions.get(
            subscription_id
        )

        if subscription is None:
            raise KeyError("Subscription not found")

        subscription.enabled = False

    async def publish(
        self,
        event: Event,
        *,
        wait_if_full: bool = True,
    ) -> str:
        if not self._running or not self._accepting_events:
            raise EventBusNotRunningError(
                "Event bus is not accepting events"
            )

        dedup_key = f"{event.tenant_id}:{event.event_id}"

        async with self._deduplication_lock:
            self._prune_seen_events()

            if dedup_key in self._seen_events:
                self.metrics.duplicate += 1
                return event.event_id

            if self._publisher is not None:
                try:
                    await self._publisher.publish(event)
                except Exception as exc:
                    self.metrics.failed += 1

                    logger.exception(
                        "Redis event publish failed"
                    )

                    raise EventPublishError(
                        "Unable to publish event"
                    ) from exc

            try:
                if wait_if_full:
                    await self._queue.put(event)
                else:
                    self._queue.put_nowait(event)

            except asyncio.QueueFull as exc:
                self.metrics.dropped += 1

                raise EventQueueFullError(
                    "Event queue is full"
                ) from exc

            self._seen_events[dedup_key] = (
                time.monotonic()
                + self.settings.deduplication_ttl_seconds
            )

            self._history.append(event)
            self.metrics.published += 1

        return event.event_id

    async def publish_many(
        self,
        events: list[Event],
    ) -> list[str]:
        published_ids: list[str] = []

        for event in events:
            event_id = await self.publish(event)
            published_ids.append(event_id)

        return published_ids

    async def emit(
        self,
        event_type: str,
        *,
        tenant_id: str,
        payload: dict[str, Any],
        category: EventCategory = EventCategory.SYSTEM,
        priority: EventPriority = EventPriority.NORMAL,
        source: str = "physical_ai",
        robot_id: str | None = None,
        fleet_id: str | None = None,
        sensor_id: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        event = Event(
            event_type=event_type,
            tenant_id=tenant_id,
            payload=payload,
            category=category,
            priority=priority,
            source=source,
            robot_id=robot_id,
            fleet_id=fleet_id,
            sensor_id=sensor_id,
            correlation_id=correlation_id,
            causation_id=causation_id,
            metadata=metadata or {},
        )

        return await self.publish(event)

    async def _worker_loop(
        self,
        worker_index: int,
    ) -> None:
        while True:
            try:
                event = await self._queue.get()

                try:
                    await self._dispatch(event)
                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "Event worker %s encountered an error",
                    worker_index,
                )

    async def _dispatch(self, event: Event) -> None:
        started = time.monotonic()

        subscriptions = [
            subscription
            for subscription in self._subscriptions.values()
            if subscription.matches(event)
        ]

        if not subscriptions:
            self.metrics.processed += 1
            self.metrics.total_processing_seconds += (
                time.monotonic() - started
            )
            return

        results = await asyncio.gather(
            *[
                self._invoke_handler(
                    subscription,
                    event,
                )
                for subscription in subscriptions
            ],
            return_exceptions=True,
        )

        failed = any(
            isinstance(result, BaseException)
            for result in results
        )

        if failed:
            self.metrics.failed += 1
        else:
            self.metrics.processed += 1

        self.metrics.total_processing_seconds += (
            time.monotonic() - started
        )

    async def _invoke_handler(
        self,
        subscription: EventSubscription,
        event: Event,
    ) -> None:
        max_retries = (
            subscription.max_retries
            if subscription.max_retries is not None
            else self.settings.max_retries
        )

        timeout = (
            subscription.timeout_seconds
            if subscription.timeout_seconds is not None
            else self.settings.handler_timeout_seconds
        )

        for attempt in range(max_retries + 1):
            try:
                self.metrics.handler_invocations += 1

                await asyncio.wait_for(
                    subscription.handler(event),
                    timeout=timeout,
                )

                return

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                self.metrics.handler_failures += 1

                if attempt >= max_retries:
                    dead_letter = DeadLetterEvent(
                        event=event,
                        subscription_id=(
                            subscription.subscription_id
                        ),
                        error_message=str(exc),
                        attempts=attempt + 1,
                    )

                    self._dead_letters.append(dead_letter)
                    self.metrics.dead_lettered += 1

                    logger.exception(
                        "Event handler failed permanently: "
                        "event=%s subscription=%s",
                        event.event_id,
                        subscription.subscription_id,
                    )

                    raise EventHandlerError(
                        "Event handler exhausted retries"
                    ) from exc

                self.metrics.retried += 1

                delay = min(
                    self.settings.retry_base_delay_seconds
                    * (2**attempt),
                    self.settings.retry_max_delay_seconds,
                )

                delay *= random.uniform(0.8, 1.2)

                logger.warning(
                    "Retrying event %s handler %s "
                    "after %.2fs",
                    event.event_id,
                    subscription.subscription_id,
                    delay,
                )

                await asyncio.sleep(delay)

    def _prune_seen_events(self) -> None:
        now = time.monotonic()

        expired_keys = [
            key
            for key, expires_at in self._seen_events.items()
            if expires_at <= now
        ]

        for key in expired_keys:
            del self._seen_events[key]

    async def wait_until_idle(self) -> None:
        await self._queue.join()

    def get_history(
        self,
        *,
        tenant_id: str,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_identifier(tenant_id, "tenant_id")

        if limit <= 0:
            raise ValueError("limit must be positive")

        events = [
            event
            for event in reversed(self._history)
            if event.tenant_id == tenant_id
            and (
                event_type is None
                or event.event_type == event_type
            )
        ]

        return [
            event.to_dict()
            for event in events[:limit]
        ]

    def get_dead_letters(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_identifier(tenant_id, "tenant_id")

        if limit <= 0:
            raise ValueError("limit must be positive")

        dead_letters = [
            item
            for item in reversed(self._dead_letters)
            if item.event.tenant_id == tenant_id
        ]

        return [
            item.to_dict()
            for item in dead_letters[:limit]
        ]

    def get_metrics(self) -> dict[str, Any]:
        return {
            **self.metrics.to_dict(),
            "running": self._running,
            "transport": self.settings.transport.value,
            "queue_size": self.queue_size,
            "worker_count": len(self._workers),
            "subscription_count": self.subscription_count,
            "history_size": len(self._history),
            "dead_letter_queue_size": len(self._dead_letters),
        }

    async def health_check(self) -> dict[str, Any]:
        redis_healthy: bool | None = None

        if self._publisher is not None:
            try:
                redis_healthy = bool(
                    await self._publisher.redis.ping()
                )
            except Exception:
                redis_healthy = False

        return {
            "status": (
                "healthy"
                if self._running
                else "stopped"
            ),
            "running": self._running,
            "accepting_events": self._accepting_events,
            "transport": self.settings.transport.value,
            "redis_healthy": redis_healthy,
            "queue_size": self.queue_size,
            "worker_count": len(self._workers),
            "checked_at": utc_now().isoformat(),
        }


# ============================================================
# Robotics Event Types
# ============================================================


class RoboticsEventType(str, Enum):
    ROBOT_REGISTERED = "robot.registered"
    ROBOT_ONLINE = "robot.online"
    ROBOT_OFFLINE = "robot.offline"
    ROBOT_STATE_CHANGED = "robot.state.changed"
    ROBOT_HEALTH_UPDATED = "robot.health.updated"
    ROBOT_FAULT_DETECTED = "robot.fault.detected"

    TELEMETRY_RECEIVED = "telemetry.received"
    TELEMETRY_ANOMALY = "telemetry.anomaly"
    TELEMETRY_STALE = "telemetry.stale"

    SENSOR_REGISTERED = "sensor.registered"
    SENSOR_HEALTH_UPDATED = "sensor.health.updated"
    SENSOR_FAILURE_DETECTED = "sensor.failure.detected"
    SENSOR_CALIBRATION_REQUIRED = (
        "sensor.calibration.required"
    )

    NAVIGATION_STARTED = "navigation.started"
    NAVIGATION_COMPLETED = "navigation.completed"
    NAVIGATION_FAILED = "navigation.failed"

    MANIPULATION_STARTED = "manipulation.started"
    MANIPULATION_COMPLETED = "manipulation.completed"
    MANIPULATION_FAILED = "manipulation.failed"

    SAFETY_ALERT = "safety.alert"
    SAFETY_VIOLATION = "safety.violation"
    SAFETY_REVIEW_REQUIRED = "safety.review.required"
    EMERGENCY_STOP_REPORTED = "safety.emergency_stop.reported"

    MAINTENANCE_PREDICTED = "maintenance.predicted"
    MAINTENANCE_REQUIRED = "maintenance.required"
    MAINTENANCE_COMPLETED = "maintenance.completed"

    REPAIR_REQUESTED = "repair.requested"
    REPAIR_ESTIMATE_CREATED = "repair.estimate.created"
    REPAIR_APPROVED = "repair.approved"
    REPAIR_COMPLETED = "repair.completed"

    PART_INSPECTION_COMPLETED = "part.inspection.completed"
    PART_REPLACEMENT_RECOMMENDED = (
        "part.replacement.recommended"
    )

    MODEL_REGISTERED = "model.registered"
    MODEL_EVALUATED = "model.evaluated"
    MODEL_DEPLOYMENT_REQUESTED = (
        "model.deployment.requested"
    )

    SIMULATION_STARTED = "simulation.started"
    SIMULATION_COMPLETED = "simulation.completed"
    SIMULATION_FAILED = "simulation.failed"

    EVALUATION_COMPLETED = "evaluation.completed"
    BENCHMARK_COMPLETED = "benchmark.completed"

    INCIDENT_CREATED = "incident.created"
    INCIDENT_UPDATED = "incident.updated"
    INCIDENT_RESOLVED = "incident.resolved"

    FLEET_UPDATED = "fleet.updated"
    FLEET_HEALTH_UPDATED = "fleet.health.updated"


# ============================================================
# Convenience Publisher
# ============================================================


class RoboticsEventPublisher:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus

    async def robot_health_updated(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        health_score: float,
        health_status: str,
        correlation_id: str | None = None,
    ) -> str:
        return await self.bus.emit(
            RoboticsEventType.ROBOT_HEALTH_UPDATED.value,
            tenant_id=tenant_id,
            robot_id=robot_id,
            category=EventCategory.ROBOT_HEALTH,
            payload={
                "robot_id": robot_id,
                "health_score": health_score,
                "health_status": health_status,
            },
            correlation_id=correlation_id,
        )

    async def telemetry_anomaly(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        metric_name: str,
        anomaly_score: float,
        severity: str,
        sensor_id: str | None = None,
    ) -> str:
        return await self.bus.emit(
            RoboticsEventType.TELEMETRY_ANOMALY.value,
            tenant_id=tenant_id,
            robot_id=robot_id,
            sensor_id=sensor_id,
            category=EventCategory.TELEMETRY,
            priority=EventPriority.HIGH,
            payload={
                "robot_id": robot_id,
                "sensor_id": sensor_id,
                "metric_name": metric_name,
                "anomaly_score": anomaly_score,
                "severity": severity,
            },
        )

    async def safety_alert(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        severity: str,
        description: str,
        incident_id: str | None = None,
    ) -> str:
        return await self.bus.emit(
            RoboticsEventType.SAFETY_ALERT.value,
            tenant_id=tenant_id,
            robot_id=robot_id,
            category=EventCategory.SAFETY,
            priority=EventPriority.CRITICAL,
            payload={
                "robot_id": robot_id,
                "severity": severity,
                "description": description,
                "incident_id": incident_id,
                "requires_human_review": True,
            },
        )

    async def maintenance_required(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        component_id: str,
        failure_probability: float,
        remaining_useful_life_hours: float | None,
    ) -> str:
        return await self.bus.emit(
            RoboticsEventType.MAINTENANCE_REQUIRED.value,
            tenant_id=tenant_id,
            robot_id=robot_id,
            category=EventCategory.MAINTENANCE,
            priority=EventPriority.HIGH,
            payload={
                "robot_id": robot_id,
                "component_id": component_id,
                "failure_probability": failure_probability,
                "remaining_useful_life_hours": (
                    remaining_useful_life_hours
                ),
                "requires_human_review": True,
            },
        )

    async def simulation_completed(
        self,
        *,
        tenant_id: str,
        simulation_run_id: str,
        result: str,
        metrics: dict[str, Any],
    ) -> str:
        return await self.bus.emit(
            RoboticsEventType.SIMULATION_COMPLETED.value,
            tenant_id=tenant_id,
            category=EventCategory.SIMULATION,
            payload={
                "simulation_run_id": simulation_run_id,
                "result": result,
                "metrics": metrics,
            },
        )


# ============================================================
# Application Singleton
# ============================================================


_event_bus: EventBus | None = None


def configure_event_bus(
    settings: EventBusSettings,
) -> EventBus:
    global _event_bus

    if _event_bus is not None:
        raise RuntimeError(
            "Event bus is already configured"
        )

    _event_bus = EventBus(settings)

    return _event_bus


def get_event_bus() -> EventBus:
    global _event_bus

    if _event_bus is None:
        _event_bus = EventBus()

    return _event_bus


def get_robotics_event_publisher() -> RoboticsEventPublisher:
    return RoboticsEventPublisher(get_event_bus())


async def initialize_event_bus() -> EventBus:
    bus = get_event_bus()
    await bus.start()
    return bus


async def close_event_bus() -> None:
    global _event_bus

    if _event_bus is not None:
        await _event_bus.stop()
        _event_bus = None


# ============================================================
# Exports
# ============================================================


__all__ = [
    "Event",
    "EventPriority",
    "EventStatus",
    "EventCategory",
    "EventTransport",
    "EventDeliveryMode",
    "EventBusSettings",
    "EventSubscription",
    "EventBusMetrics",
    "DeadLetterEvent",
    "EventBusError",
    "EventBusNotRunningError",
    "EventQueueFullError",
    "EventHandlerError",
    "EventPublishError",
    "EventPublisher",
    "RedisStreamsPublisher",
    "EventBus",
    "RoboticsEventType",
    "RoboticsEventPublisher",
    "configure_event_bus",
    "get_event_bus",
    "get_robotics_event_publisher",
    "initialize_event_bus",
    "close_event_bus",
    "utc_now",
    "generate_uuid",
]
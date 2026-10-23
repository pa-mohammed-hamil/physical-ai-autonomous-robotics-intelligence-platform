"""Lifespan module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Application Lifespan Manager
============================

Physical AI & Autonomous Robotics Intelligence Platform

Features:
    - FastAPI startup and shutdown
    - PostgreSQL connection lifecycle
    - Redis connection lifecycle
    - Optional MQTT integration
    - Optional Kafka producer integration
    - Robotics telemetry supervision
    - Background task supervision
    - AI service initialization hooks
    - Analytics service registration
    - Startup readiness tracking
    - Graceful shutdown
    - Resource cleanup
    - Dependency injection compatibility

Python: 3.11+

Core dependencies:
    fastapi
    sqlalchemy[asyncio]
    asyncpg
    redis
    pydantic-settings

Optional:
    aiomqtt
    aiokafka

Safety:
    This module does not issue robot commands.
    Hardware safety systems must operate independently.
"""

from __future__ import annotations

import asyncio
import logging
import time

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import FastAPI
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings


logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ServiceStatus:
    name: str
    enabled: bool
    ready: bool = False
    started_at: str | None = None
    stopped_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "ready": self.ready,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "error": self.error,
        }


@dataclass
class RuntimeState:
    started_at: str = field(
        default_factory=utc_now
    )
    ready: bool = False
    shutting_down: bool = False
    services: dict[str, ServiceStatus] = field(
        default_factory=dict
    )
    background_tasks: set[asyncio.Task[Any]] = field(
        default_factory=set
    )

    def register(
        self,
        name: str,
        *,
        enabled: bool,
    ) -> None:
        self.services[name] = ServiceStatus(
            name=name,
            enabled=enabled,
        )

    def mark_ready(self, name: str) -> None:
        service = self.services[name]
        service.ready = True
        service.error = None
        service.started_at = utc_now()

    def mark_failed(
        self,
        name: str,
        error: str,
    ) -> None:
        service = self.services[name]
        service.ready = False
        service.error = error

    def mark_stopped(self, name: str) -> None:
        service = self.services[name]
        service.ready = False
        service.stopped_at = utc_now()

    def snapshot(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "shutting_down": self.shutting_down,
            "started_at": self.started_at,
            "services": {
                name: service.to_dict()
                for name, service in self.services.items()
            },
            "background_task_count": len(
                self.background_tasks
            ),
        }


class LifespanError(RuntimeError):
    """Application lifecycle failure."""


class ResourceManager:
    """
    Owns application-level async resources.

    Resources are exposed through app.state and must
    not be constructed independently per request.
    """

    def __init__(
        self,
        config: Settings,
    ) -> None:
        self.config = config
        self.runtime = RuntimeState()

        self.engine: AsyncEngine | None = None

        self.session_factory: (
            async_sessionmaker[AsyncSession] | None
        ) = None

        self.redis: Redis | None = None

        self.mqtt_client: Any | None = None
        self.mqtt_context: Any | None = None

        self.kafka_producer: Any | None = None

        self.analytics: dict[str, Any] = {}

        self._shutdown_event = asyncio.Event()
        self._background_failure: str | None = None

        self._register_services()

    def _register_services(self) -> None:
        self.runtime.register(
            "database",
            enabled=True,
        )
        self.runtime.register(
            "redis",
            enabled=self.config.REDIS_ENABLED,
        )
        self.runtime.register(
            "mqtt",
            enabled=self.config.MQTT_ENABLED,
        )
        self.runtime.register(
            "kafka",
            enabled=self.config.KAFKA_ENABLED,
        )
        self.runtime.register(
            "analytics",
            enabled=True,
        )
        self.runtime.register(
            "telemetry_supervisor",
            enabled=self.config.ROBOT_PLATFORM_ENABLED,
        )
        self.runtime.register(
            "background_workers",
            enabled=self.config.BACKGROUND_WORKERS_ENABLED,
        )

    async def initialize_database(self) -> None:
        """Create and verify the database connection pool."""
        name = "database"

        try:
            self.engine = create_async_engine(
                self.config.database_dsn,
                echo=self.config.DB_ECHO,
                pool_size=self.config.DB_POOL_SIZE,
                max_overflow=self.config.DB_MAX_OVERFLOW,
                pool_timeout=(
                    self.config.DB_POOL_TIMEOUT_SECONDS
                ),
                pool_recycle=(
                    self.config.DB_POOL_RECYCLE_SECONDS
                ),
                pool_pre_ping=True,
            )

            self.session_factory = async_sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
                autoflush=False,
            )

            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))

            self.runtime.mark_ready(name)
            logger.info("Database connection established")

        except Exception as exc:
            self.runtime.mark_failed(name, str(exc))
            logger.exception("Database startup failed")
            raise LifespanError(
                "Database initialization failed"
            ) from exc

    async def initialize_redis(self) -> None:
        if not self.config.REDIS_ENABLED:
            return

        name = "redis"

        try:
            self.redis = Redis.from_url(
                self.config.redis_dsn,
                decode_responses=True,
                socket_timeout=(
                    self.config.REDIS_SOCKET_TIMEOUT_SECONDS
                ),
            )

            await self.redis.ping()

            self.runtime.mark_ready(name)
            logger.info("Redis connection established")

        except Exception as exc:
            self.runtime.mark_failed(name, str(exc))
            logger.exception("Redis startup failed")
            raise LifespanError(
                "Redis initialization failed"
            ) from exc

    async def initialize_mqtt(self) -> None:
        if not self.config.MQTT_ENABLED:
            return

        name = "mqtt"

        try:
            import aiomqtt

            password = (
                self.config.MQTT_PASSWORD.get_secret_value()
                if self.config.MQTT_PASSWORD is not None
                else None
            )

            tls_context = None

            if self.config.MQTT_USE_TLS:
                import ssl
                tls_context = ssl.create_default_context()

            self.mqtt_context = aiomqtt.Client(
                hostname=self.config.MQTT_HOST,
                port=self.config.MQTT_PORT,
                username=self.config.MQTT_USERNAME,
                password=password,
                tls_context=tls_context,
                keepalive=self.config.MQTT_KEEPALIVE_SECONDS,
            )

            self.mqtt_client = (
                await self.mqtt_context.__aenter__()
            )

            self.runtime.mark_ready(name)
            logger.info("MQTT client connected")

        except Exception as exc:
            self.runtime.mark_failed(name, str(exc))
            logger.exception("MQTT startup failed")
            raise LifespanError(
                "MQTT initialization failed"
            ) from exc

    async def initialize_kafka(self) -> None:
        if not self.config.KAFKA_ENABLED:
            return

        name = "kafka"

        try:
            from aiokafka import AIOKafkaProducer

            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers=(
                    self.config.KAFKA_BOOTSTRAP_SERVERS
                ),
                client_id="physical-ai-platform",
            )

            await self.kafka_producer.start()

            self.runtime.mark_ready(name)
            logger.info("Kafka producer started")

        except Exception as exc:
            self.runtime.mark_failed(name, str(exc))
            logger.exception("Kafka startup failed")
            raise LifespanError(
                "Kafka initialization failed"
            ) from exc

    async def initialize_analytics(self) -> None:
        """
        Register in-memory analytics engines.

        For production, load data from repositories and
        use persistent storage for incident/task history.
        """
        name = "analytics"

        try:
            from app.analytics.safety_analytics import (
                SafetyAnalytics,
            )
            from app.analytics.task_success_analytics import (
                TaskSuccessAnalytics,
            )
            from app.analytics.robot_utilization import (
                RobotUtilizationAnalytics,
                UtilizationThresholds,
            )

            self.analytics = {
                "safety": SafetyAnalytics(
                    incident_rate_warning=(
                        self.config.SAFETY_INCIDENT_RATE_WARNING
                    ),
                    incident_rate_critical=(
                        self.config.SAFETY_INCIDENT_RATE_CRITICAL
                    ),
                    rate_base_hours=(
                        self.config.SAFETY_INCIDENT_RATE_BASE_HOURS
                    ),
                ),
                "task_success": TaskSuccessAnalytics(
                    success_warning_percentage=(
                        self.config.TASK_SUCCESS_WARNING_PERCENTAGE
                    ),
                    retry_warning_percentage=(
                        self.config.TASK_RETRY_WARNING_PERCENTAGE
                    ),
                    sla_warning_percentage=(
                        self.config.TASK_SLA_WARNING_PERCENTAGE
                    ),
                ),
                "utilization": RobotUtilizationAnalytics(
                    thresholds=UtilizationThresholds(
                        underutilized_below=(
                            self.config.UTILIZATION_LOW_THRESHOLD
                        ),
                        high_at_or_above=(
                            self.config.UTILIZATION_HIGH_THRESHOLD
                        ),
                    )
                ),
            }

            self.runtime.mark_ready(name)
            logger.info(
                "Initialized %d analytics engines",
                len(self.analytics),
            )

        except Exception as exc:
            self.runtime.mark_failed(name, str(exc))
            logger.exception("Analytics startup failed")
            raise LifespanError(
                "Analytics initialization failed"
            ) from exc

    def create_background_task(
        self,
        coroutine: Awaitable[Any],
        *,
        name: str,
    ) -> asyncio.Task[Any]:
        """
        Create and supervise an application background task.

        Any unexpected exception marks the corresponding
        service as failed.
        """
        task = asyncio.create_task(
            coroutine,
            name=name,
        )

        self.runtime.background_tasks.add(task)

        def on_done(
            completed: asyncio.Task[Any],
        ) -> None:
            self.runtime.background_tasks.discard(
                completed
            )

            if completed.cancelled():
                return

            try:
                error = completed.exception()
            except asyncio.CancelledError:
                return

            if error is not None:
                self._background_failure = str(error)

                if name in self.runtime.services:
                    self.runtime.mark_failed(
                        name,
                        str(error),
                    )

                self.runtime.ready = False

                logger.error(
                    "Background task %s failed: %s",
                    name,
                    error,
                    exc_info=(
                        type(error),
                        error,
                        error.__traceback__,
                    ),
                )

        task.add_done_callback(on_done)

        return task

    async def _telemetry_supervisor(self) -> None:
        """
        Lightweight supervisor heartbeat.

        This is NOT a telemetry ingestion worker.
        Actual telemetry ingestion must be implemented
        by a separate ROS/MQTT/Kafka consumer service.
        """
        interval = max(
            1.0,
            self.config.ROBOT_HEARTBEAT_INTERVAL_SECONDS,
        )

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=interval,
                )
            except asyncio.TimeoutError:
                logger.debug(
                    "Telemetry supervisor heartbeat"
                )

    async def _background_supervisor(self) -> None:
        """
        Background worker supervision heartbeat.

        This does not execute queued jobs.
        """
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                logger.debug(
                    "Background supervisor active"
                )

    async def start_background_services(self) -> None:
        if self.config.ROBOT_PLATFORM_ENABLED:
            self.create_background_task(
                self._telemetry_supervisor(),
                name="telemetry_supervisor",
            )
            self.runtime.mark_ready(
                "telemetry_supervisor"
            )

        if self.config.BACKGROUND_WORKERS_ENABLED:
            self.create_background_task(
                self._background_supervisor(),
                name="background_workers",
            )
            self.runtime.mark_ready(
                "background_workers"
            )

    async def startup(self) -> None:
        """Initialize all configured application resources."""
        logger.info(
            "Starting %s",
            self.config.APP_NAME,
        )

        self.config.ensure_directories()

        try:
            await self.initialize_database()
            await self.initialize_redis()
            await self.initialize_mqtt()
            await self.initialize_kafka()
            await self.initialize_analytics()
            await self.start_background_services()

            self.runtime.ready = True

            logger.info(
                "Application startup completed"
            )

        except BaseException:
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Stop workers and release resources safely."""
        if self.runtime.shutting_down:
            return

        self.runtime.shutting_down = True
        self.runtime.ready = False

        logger.info("Application shutdown started")

        self._shutdown_event.set()

        tasks = list(
            self.runtime.background_tasks
        )

        for task in tasks:
            task.cancel()

        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *tasks,
                        return_exceptions=True,
                    ),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Background task shutdown timed out"
                )

        for name in (
            "telemetry_supervisor",
            "background_workers",
        ):
            if self.runtime.services[name].enabled:
                self.runtime.mark_stopped(name)

        if self.kafka_producer is not None:
            try:
                await self.kafka_producer.stop()
            except Exception:
                logger.exception(
                    "Kafka shutdown failed"
                )
            finally:
                self.kafka_producer = None
                self.runtime.mark_stopped("kafka")

        if self.mqtt_context is not None:
            try:
                await self.mqtt_context.__aexit__(
                    None,
                    None,
                    None,
                )
            except Exception:
                logger.exception(
                    "MQTT shutdown failed"
                )
            finally:
                self.mqtt_client = None
                self.mqtt_context = None
                self.runtime.mark_stopped("mqtt")

        if self.redis is not None:
            try:
                await self.redis.aclose()
            except Exception:
                logger.exception(
                    "Redis shutdown failed"
                )
            finally:
                self.redis = None
                self.runtime.mark_stopped("redis")

        if self.engine is not None:
            try:
                await self.engine.dispose()
            except Exception:
                logger.exception(
                    "Database shutdown failed"
                )
            finally:
                self.engine = None
                self.session_factory = None
                self.runtime.mark_stopped("database")

        self.analytics.clear()
        self.runtime.mark_stopped("analytics")

        logger.info("Application shutdown completed")

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "application": self.config.APP_NAME,
            "version": self.config.APP_VERSION,
            "environment": (
                self.config.ENVIRONMENT.value
            ),
            **self.runtime.snapshot(),
            "background_failure": (
                self._background_failure
            ),
        }


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    FastAPI application lifespan.

    Startup:
        1. Load validated settings
        2. Initialize database
        3. Initialize optional Redis
        4. Initialize optional MQTT
        5. Initialize optional Kafka
        6. Initialize analytics engines
        7. Start background supervisors

    Shutdown:
        1. Stop background tasks
        2. Stop Kafka
        3. Disconnect MQTT
        4. Close Redis
        5. Dispose database engine
        6. Clear in-memory services
    """
    config = get_settings()

    manager = ResourceManager(config)

    app.state.resource_manager = manager

    try:
        await manager.startup()

        app.state.db_engine = manager.engine
        app.state.db_session_factory = (
            manager.session_factory
        )
        app.state.redis = manager.redis
        app.state.mqtt = manager.mqtt_client
        app.state.kafka = manager.kafka_producer
        app.state.analytics = manager.analytics

        yield

    finally:
        await manager.shutdown()


async def get_application_health(
    app: FastAPI,
) -> dict[str, Any]:
    manager: ResourceManager | None = getattr(
        app.state,
        "resource_manager",
        None,
    )

    if manager is None:
        return {
            "ready": False,
            "error": "Resource manager unavailable",
        }

    return manager.health_snapshot()


__all__ = [
    "ServiceStatus",
    "RuntimeState",
    "LifespanError",
    "ResourceManager",
    "lifespan",
    "get_application_health",
]

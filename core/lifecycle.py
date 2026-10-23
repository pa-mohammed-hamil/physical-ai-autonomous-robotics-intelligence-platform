"""Lifecycle module.

TODO: implement and test; no real robot control is enabled.
"""
"""
Physical AI & Autonomous Robotics Intelligence Platform
Application Lifecycle Management

Provides:
- FastAPI lifespan integration
- Startup and shutdown orchestration
- Cache and event bus initialization
- Optional database initialization
- Background task supervision
- Application readiness and health reporting
- Graceful resource cleanup
- Fail-closed startup behavior

Physical robot actuation is never started automatically.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import FastAPI

from app.core.caching import (
    CacheManager,
    close_cache,
    get_cache_manager,
)
from app.core.event_bus import (
    EventBus,
    close_event_bus,
    get_event_bus,
)

logger = logging.getLogger(__name__)

AsyncCallback = Callable[[], Awaitable[Any]]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LifecycleState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class ComponentState(str, Enum):
    PENDING = "pending"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"


class LifecycleError(RuntimeError):
    pass


class StartupError(LifecycleError):
    pass


class ShutdownError(LifecycleError):
    pass


@dataclass(frozen=True)
class LifecycleSettings:
    startup_timeout_seconds: float = 60.0
    shutdown_timeout_seconds: float = 30.0
    component_timeout_seconds: float = 20.0
    background_task_shutdown_seconds: float = 10.0
    fail_on_critical_component_error: bool = True
    enable_cache: bool = True
    enable_event_bus: bool = True
    enable_database: bool = True

    def __post_init__(self) -> None:
        for name in (
            "startup_timeout_seconds",
            "shutdown_timeout_seconds",
            "component_timeout_seconds",
            "background_task_shutdown_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass
class ComponentHealth:
    name: str
    state: ComponentState = ComponentState.PENDING
    critical: bool = True
    started_at: datetime | None = None
    checked_at: datetime | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "critical": self.critical,
            "started_at": (
                self.started_at.isoformat()
                if self.started_at
                else None
            ),
            "checked_at": (
                self.checked_at.isoformat()
                if self.checked_at
                else None
            ),
            "error": self.error,
        }


@dataclass
class LifecycleComponent:
    name: str
    startup: AsyncCallback
    shutdown: AsyncCallback
    health_check: AsyncCallback | None = None
    critical: bool = True
    startup_order: int = 100


@dataclass
class BackgroundTaskRecord:
    name: str
    task: asyncio.Task[Any]
    critical: bool = False
    started_at: datetime = field(default_factory=utc_now)


class BackgroundTaskSupervisor:
    """
    Supervises application-owned background tasks.

    A task failure is logged and retained for health checks.
    Critical task failure marks the application unhealthy.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTaskRecord] = {}
        self._failures: dict[str, str] = {}
        self._stopping = False

    def start(
        self,
        name: str,
        coroutine: Awaitable[Any],
        *,
        critical: bool = False,
    ) -> asyncio.Task[Any]:
        if self._stopping:
            if inspect.iscoroutine(coroutine):
                coroutine.close()
            raise LifecycleError(
                "Cannot start background tasks during shutdown"
            )

        existing = self._tasks.get(name)

        if existing is not None and not existing.task.done():
            if inspect.iscoroutine(coroutine):
                coroutine.close()
            raise LifecycleError(
                f"Background task already running: {name}"
            )

        task = asyncio.create_task(
            coroutine,
            name=f"physical-ai:{name}",
        )

        record = BackgroundTaskRecord(
            name=name,
            task=task,
            critical=critical,
        )

        self._tasks[name] = record
        self._failures.pop(name, None)

        task.add_done_callback(
            lambda completed, task_name=name: (
                self._on_task_done(task_name, completed)
            )
        )

        return task

    def _on_task_done(
        self,
        name: str,
        task: asyncio.Task[Any],
    ) -> None:
        if task.cancelled():
            return

        try:
            exception = task.exception()
        except asyncio.CancelledError:
            return

        if exception is not None:
            self._failures[name] = type(exception).__name__
            logger.error(
                "Background task failed: %s",
                name,
                exc_info=(
                    type(exception),
                    exception,
                    exception.__traceback__,
                ),
            )

    @property
    def critical_failure(self) -> bool:
        return any(
            record.critical and name in self._failures
            for name, record in self._tasks.items()
        )

    def health(self) -> dict[str, Any]:
        return {
            "running": [
                name
                for name, record in self._tasks.items()
                if not record.task.done()
            ],
            "failed": dict(self._failures),
            "critical_failure": self.critical_failure,
        }

    async def shutdown(
        self,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._stopping = True

        active = [
            record.task
            for record in self._tasks.values()
            if not record.task.done()
        ]

        for task in active:
            task.cancel()

        if active:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *active,
                        return_exceptions=True,
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Background task shutdown timed out"
                )

        self._tasks.clear()


class ApplicationLifecycle:
    """
    Ordered startup and reverse-order shutdown manager.

    Components should be registered before startup.
    """

    def __init__(
        self,
        settings: LifecycleSettings | None = None,
    ) -> None:
        self.settings = settings or LifecycleSettings()
        self.state = LifecycleState.CREATED

        self.components: dict[str, LifecycleComponent] = {}
        self.component_health: dict[str, ComponentHealth] = {}

        self.background_tasks = BackgroundTaskSupervisor()

        self.started_at: datetime | None = None
        self.stopped_at: datetime | None = None

        self._started_components: list[str] = []
        self._lock = asyncio.Lock()

    def register_component(
        self,
        component: LifecycleComponent,
    ) -> None:
        if self.state != LifecycleState.CREATED:
            raise LifecycleError(
                "Components must be registered before startup"
            )

        if component.name in self.components:
            raise LifecycleError(
                f"Duplicate lifecycle component: {component.name}"
            )

        self.components[component.name] = component
        self.component_health[component.name] = ComponentHealth(
            name=component.name,
            critical=component.critical,
        )

    async def startup(self) -> None:
        async with self._lock:
            if self.state in (
                LifecycleState.RUNNING,
                LifecycleState.DEGRADED,
            ):
                return

            if self.state != LifecycleState.CREATED:
                raise StartupError(
                    f"Cannot start from state {self.state.value}"
                )

            self.state = LifecycleState.STARTING

            try:
                await asyncio.wait_for(
                    self._startup_components(),
                    timeout=self.settings.startup_timeout_seconds,
                )

                self.started_at = utc_now()
                self.state = (
                    LifecycleState.DEGRADED
                    if self._has_component_failures()
                    else LifecycleState.RUNNING
                )

                logger.info(
                    "Application lifecycle started: %s",
                    self.state.value,
                )

            except BaseException:
                self.state = LifecycleState.FAILED

                logger.exception(
                    "Application startup failed"
                )

                await self._shutdown_components()
                raise

    async def _startup_components(self) -> None:
        ordered = sorted(
            self.components.values(),
            key=lambda component: (
                component.startup_order,
                component.name,
            ),
        )

        for component in ordered:
            health = self.component_health[component.name]
            health.state = ComponentState.STARTING

            try:
                await asyncio.wait_for(
                    component.startup(),
                    timeout=self.settings.component_timeout_seconds,
                )

                self._started_components.append(component.name)

                health.state = ComponentState.HEALTHY
                health.started_at = utc_now()
                health.checked_at = utc_now()
                health.error = None

                logger.info(
                    "Lifecycle component started: %s",
                    component.name,
                )

            except Exception as exc:
                health.state = ComponentState.FAILED
                health.error = type(exc).__name__
                health.checked_at = utc_now()

                logger.exception(
                    "Lifecycle component failed: %s",
                    component.name,
                )

                if (
                    component.critical
                    and self.settings.fail_on_critical_component_error
                ):
                    raise StartupError(
                        f"Critical component failed: {component.name}"
                    ) from exc

    async def shutdown(self) -> None:
        async with self._lock:
            if self.state in (
                LifecycleState.STOPPED,
                LifecycleState.STOPPING,
            ):
                return

            self.state = LifecycleState.STOPPING

            try:
                await self.background_tasks.shutdown(
                    self.settings.background_task_shutdown_seconds
                )

                await asyncio.wait_for(
                    self._shutdown_components(),
                    timeout=self.settings.shutdown_timeout_seconds,
                )

            except asyncio.TimeoutError:
                logger.error(
                    "Application shutdown exceeded timeout"
                )

            finally:
                self.stopped_at = utc_now()
                self.state = LifecycleState.STOPPED

                logger.info(
                    "Application lifecycle stopped"
                )

    async def _shutdown_components(self) -> None:
        names = list(reversed(self._started_components))
        self._started_components.clear()

        for name in names:
            component = self.components[name]
            health = self.component_health[name]

            try:
                await asyncio.wait_for(
                    component.shutdown(),
                    timeout=self.settings.component_timeout_seconds,
                )

                health.state = ComponentState.STOPPED
                health.checked_at = utc_now()

            except Exception as exc:
                health.state = ComponentState.FAILED
                health.error = type(exc).__name__
                health.checked_at = utc_now()

                logger.exception(
                    "Lifecycle component shutdown failed: %s",
                    name,
                )

    def _has_component_failures(self) -> bool:
        return any(
            health.state in (
                ComponentState.FAILED,
                ComponentState.DEGRADED,
            )
            for health in self.component_health.values()
        )

    async def check_health(self) -> dict[str, Any]:
        for name in list(self._started_components):
            component = self.components[name]
            health = self.component_health[name]

            if component.health_check is None:
                continue

            try:
                result = await asyncio.wait_for(
                    component.health_check(),
                    timeout=self.settings.component_timeout_seconds,
                )

                if isinstance(result, dict):
                    status = str(
                        result.get("status", "")
                    ).lower()

                    healthy = (
                        result.get("healthy") is True
                        or status in ("healthy", "ok", "ready")
                    )

                    if result.get("healthy") is False:
                        healthy = False
                else:
                    healthy = result is True

                health.state = (
                    ComponentState.HEALTHY
                    if healthy
                    else ComponentState.DEGRADED
                )

                health.error = (
                    None if healthy else "Health check failed"
                )

            except Exception as exc:
                health.state = ComponentState.FAILED
                health.error = type(exc).__name__

                logger.exception(
                    "Component health check failed: %s",
                    name,
                )

            health.checked_at = utc_now()

        critical_healthy = all(
            health.state == ComponentState.HEALTHY
            for health in self.component_health.values()
            if health.critical
        )

        ready = (
            self.state in (
                LifecycleState.RUNNING,
                LifecycleState.DEGRADED,
            )
            and critical_healthy
            and not self.background_tasks.critical_failure
        )

        if self.state in (
            LifecycleState.RUNNING,
            LifecycleState.DEGRADED,
        ):
            self.state = (
                LifecycleState.RUNNING
                if ready and not self._has_component_failures()
                else LifecycleState.DEGRADED
            )

        return {
            "status": "healthy" if ready else "unhealthy",
            "ready": ready,
            "lifecycle_state": self.state.value,
            "started_at": (
                self.started_at.isoformat()
                if self.started_at
                else None
            ),
            "components": {
                name: health.to_dict()
                for name, health in self.component_health.items()
            },
            "background_tasks": self.background_tasks.health(),
            "checked_at": utc_now().isoformat(),
        }

    def is_ready(self) -> bool:
        return (
            self.state in (
                LifecycleState.RUNNING,
                LifecycleState.DEGRADED,
            )
            and not self.background_tasks.critical_failure
            and all(
                health.state == ComponentState.HEALTHY
                for health in self.component_health.values()
                if health.critical
            )
        )


# ============================================================
# Core Resource Lifecycle
# ============================================================


async def startup_cache() -> None:
    await get_cache_manager().initialize()


async def shutdown_cache() -> None:
    await close_cache()


async def check_cache_health() -> dict[str, Any]:
    manager: CacheManager = get_cache_manager()
    result = await manager.health_check()

    backend = result.get("backend")

    healthy = (
        backend == "memory"
        or (
            backend == "redis"
            and result.get("redis_healthy") is True
        )
    )

    return {
        "status": "healthy" if healthy else "degraded",
        "healthy": healthy,
        "backend": backend,
    }


async def startup_event_bus() -> None:
    await get_event_bus().start()


async def shutdown_event_bus() -> None:
    await close_event_bus()


async def check_event_bus_health() -> dict[str, Any]:
    bus: EventBus = get_event_bus()
    result = await bus.health_check()

    healthy = (
        result.get("running") is True
        and result.get("accepting_events") is True
        and result.get("redis_healthy") is not False
    )

    return {
        **result,
        "healthy": healthy,
        "status": "healthy" if healthy else "degraded",
    }


async def startup_database() -> None:
    """
    Verify the database connection without creating tables.

    Schema changes should be performed using Alembic migrations.
    """
    from sqlalchemy import text

    from app.db.engine import get_engine

    engine = get_engine()

    if inspect.isawaitable(engine):
        engine = await engine

    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


async def shutdown_database() -> None:
    from app.db.engine import get_engine

    engine = get_engine()

    if inspect.isawaitable(engine):
        engine = await engine

    await engine.dispose()


async def check_database_health() -> dict[str, Any]:
    try:
        await startup_database()
        return {"status": "healthy", "healthy": True}
    except Exception:
        logger.exception("Database health check failed")
        return {"status": "unhealthy", "healthy": False}


# ============================================================
# Factory
# ============================================================


def create_application_lifecycle(
    settings: LifecycleSettings | None = None,
) -> ApplicationLifecycle:
    config = settings or LifecycleSettings()

    lifecycle = ApplicationLifecycle(config)

    if config.enable_database:
        lifecycle.register_component(
            LifecycleComponent(
                name="database",
                startup=startup_database,
                shutdown=shutdown_database,
                health_check=check_database_health,
                critical=True,
                startup_order=10,
            )
        )

    if config.enable_cache:
        lifecycle.register_component(
            LifecycleComponent(
                name="cache",
                startup=startup_cache,
                shutdown=shutdown_cache,
                health_check=check_cache_health,
                critical=False,
                startup_order=20,
            )
        )

    if config.enable_event_bus:
        lifecycle.register_component(
            LifecycleComponent(
                name="event_bus",
                startup=startup_event_bus,
                shutdown=shutdown_event_bus,
                health_check=check_event_bus_health,
                critical=True,
                startup_order=30,
            )
        )

    return lifecycle


# ============================================================
# FastAPI Lifespan
# ============================================================


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    Usage:

        app = FastAPI(lifespan=lifespan)

    A custom lifecycle may be attached before startup:

        app.state.lifecycle = create_application_lifecycle(
            LifecycleSettings(enable_database=False)
        )
    """

    lifecycle: ApplicationLifecycle | None = getattr(
        app.state,
        "lifecycle",
        None,
    )

    if lifecycle is None:
        lifecycle = create_application_lifecycle()
        app.state.lifecycle = lifecycle

    await lifecycle.startup()

    try:
        yield
    finally:
        await lifecycle.shutdown()


# ============================================================
# Health Endpoint Helpers
# ============================================================


async def get_liveness(
    app: FastAPI,
) -> dict[str, Any]:
    lifecycle: ApplicationLifecycle | None = getattr(
        app.state,
        "lifecycle",
        None,
    )

    return {
        "status": "alive",
        "lifecycle_state": (
            lifecycle.state.value
            if lifecycle
            else LifecycleState.CREATED.value
        ),
        "timestamp": utc_now().isoformat(),
    }


async def get_readiness(
    app: FastAPI,
) -> dict[str, Any]:
    lifecycle: ApplicationLifecycle | None = getattr(
        app.state,
        "lifecycle",
        None,
    )

    if lifecycle is None:
        return {
            "status": "not_ready",
            "ready": False,
            "timestamp": utc_now().isoformat(),
        }

    return await lifecycle.check_health()


# ============================================================
# Exports
# ============================================================


__all__ = [
    "LifecycleState",
    "ComponentState",
    "LifecycleError",
    "StartupError",
    "ShutdownError",
    "LifecycleSettings",
    "ComponentHealth",
    "LifecycleComponent",
    "BackgroundTaskRecord",
    "BackgroundTaskSupervisor",
    "ApplicationLifecycle",
    "startup_cache",
    "shutdown_cache",
    "check_cache_health",
    "startup_event_bus",
    "shutdown_event_bus",
    "check_event_bus_health",
    "startup_database",
    "shutdown_database",
    "check_database_health",
    "create_application_lifecycle",
    "lifespan",
    "get_liveness",
    "get_readiness",
    "utc_now",
]
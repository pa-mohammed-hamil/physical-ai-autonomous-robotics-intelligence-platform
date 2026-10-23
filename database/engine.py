"""Engine module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/monitoring/engine.py

Physical AI & Autonomous Robotics Intelligence Platform

Central Monitoring Engine

Features:
- Unified monitoring collector orchestration
- Robot and fleet monitoring
- Periodic monitoring cycles
- Background monitoring thread
- Collector registration and discovery
- Health aggregation
- Threshold and alert evaluation
- Alert deduplication and cooldown
- Metrics history and snapshots
- JSON and Prometheus export
- Monitoring lifecycle management
- Thread-safe operations
- Graceful startup and shutdown

Python 3.11+
Standard library only.

Monitoring is observational. It does not authorize robot
operations, enforce safety limits, or replace independent
hardware safety systems.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import uuid

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable


logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Enum):
        return value.value

    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    raise TypeError(
        f"Unsupported JSON value: {type(value).__name__}"
    )


class EngineState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


class MonitoringHealth(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MonitoringSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class MonitoringEventType(str, Enum):
    ENGINE_STARTED = "engine_started"
    ENGINE_STOPPED = "engine_stopped"
    COLLECTOR_REGISTERED = "collector_registered"
    COLLECTOR_REMOVED = "collector_removed"
    COLLECTOR_FAILED = "collector_failed"
    HEALTH_CHANGED = "health_changed"
    ALERT_TRIGGERED = "alert_triggered"
    ALERT_RESOLVED = "alert_resolved"
    MONITORING_CYCLE_COMPLETED = "monitoring_cycle_completed"


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    def get_fleet_metrics(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class MonitoringEngineConfig:
    collection_interval_seconds: float = 10.0
    alert_cooldown_seconds: float = 60.0
    collector_timeout_seconds: float = 5.0

    max_history: int = 1000
    max_events: int = 5000
    max_alerts: int = 5000

    enable_background_monitoring: bool = True
    enable_alert_deduplication: bool = True
    enable_health_aggregation: bool = True
    enable_prometheus: bool = True

    def __post_init__(self) -> None:
        for name in (
            "collection_interval_seconds",
            "alert_cooldown_seconds",
            "collector_timeout_seconds",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be finite and nonnegative"
                )

        if self.collection_interval_seconds <= 0:
            raise ValueError(
                "collection_interval_seconds must be positive"
            )

        if self.collector_timeout_seconds <= 0:
            raise ValueError(
                "collector_timeout_seconds must be positive"
            )

        for name in (
            "max_history",
            "max_events",
            "max_alerts",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer"
                )


@dataclass(frozen=True)
class MonitoringEvent:
    event_id: str
    event_type: MonitoringEventType
    message: str
    timestamp: datetime = field(default_factory=utc_now)

    collector_name: Optional[str] = None
    robot_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class MonitoringAlert:
    alert_id: str
    collector_name: str
    alert_key: str
    severity: MonitoringSeverity
    message: str

    robot_id: Optional[str] = None
    timestamp: datetime = field(default_factory=utc_now)

    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class CollectorSnapshot:
    collector_name: str
    collected_at: datetime
    health: MonitoringHealth
    metrics: dict[str, Any]

    duration_ms: float
    success: bool

    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class MonitoringSnapshot:
    snapshot_id: str
    timestamp: datetime
    engine_state: EngineState
    overall_health: MonitoringHealth

    total_collectors: int
    successful_collectors: int
    failed_collectors: int

    active_alerts: int
    collection_duration_ms: float

    collectors: dict[str, CollectorSnapshot]
    robot_health: dict[str, MonitoringHealth]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class MonitoringEngineStatistics:
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0

    total_collector_executions: int = 0
    total_collector_failures: int = 0

    total_alerts_triggered: int = 0
    total_alerts_resolved: int = 0

    last_cycle_at: Optional[datetime] = None
    last_cycle_duration_ms: Optional[float] = None

    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


class MonitoringEngine:
    """
    Central orchestration engine for monitoring collectors.

    Collectors must implement get_fleet_metrics().

    Optional collector methods:
        get_robot_metrics(robot_id)
        export_prometheus()
        clear()

    The engine snapshots collector registrations before
    each cycle and does not hold its registry lock while
    invoking collectors.

    Collector calls are synchronous. The configured
    collector timeout is diagnostic only: an overlong
    call is recorded as a failure after it returns.
    """

    HEALTH_PRIORITY = {
        MonitoringHealth.UNKNOWN: 0,
        MonitoringHealth.HEALTHY: 1,
        MonitoringHealth.WARNING: 2,
        MonitoringHealth.CRITICAL: 3,
    }

    def __init__(
        self,
        config: Optional[MonitoringEngineConfig] = None,
    ) -> None:
        self.config = config or MonitoringEngineConfig()

        self._collectors: dict[
            str, MetricsCollectorProtocol
        ] = {}

        self._history: deque[MonitoringSnapshot] = deque(
            maxlen=self.config.max_history
        )

        self._events: deque[MonitoringEvent] = deque(
            maxlen=self.config.max_events
        )

        self._alerts: deque[MonitoringAlert] = deque(
            maxlen=self.config.max_alerts
        )

        self._active_alerts: dict[
            str, MonitoringAlert
        ] = {}

        self._last_alert_times: dict[
            str, datetime
        ] = {}

        self._lock = threading.RLock()
        self._cycle_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._thread: Optional[threading.Thread] = None
        self._state = EngineState.CREATED

        self._statistics = MonitoringEngineStatistics()

        self._event_callbacks: list[
            Callable[[MonitoringEvent], None]
        ] = []

        self._alert_callbacks: list[
            Callable[[MonitoringAlert], None]
        ] = []

    @property
    def state(self) -> EngineState:
        with self._lock:
            return self._state

    def register_collector(
        self,
        name: str,
        collector: MetricsCollectorProtocol,
        replace: bool = False,
    ) -> None:
        if not name or not name.strip():
            raise ValueError(
                "Collector name is required"
            )

        if not callable(
            getattr(collector, "get_fleet_metrics", None)
        ):
            raise TypeError(
                "Collector must implement get_fleet_metrics"
            )

        with self._lock:
            if name in self._collectors and not replace:
                raise ValueError(
                    f"Collector already registered: {name}"
                )

            self._collectors[name] = collector

        self._emit_event(
            MonitoringEventType.COLLECTOR_REGISTERED,
            f"Registered collector: {name}",
            collector_name=name,
        )

    def unregister_collector(self, name: str) -> bool:
        with self._lock:
            removed = self._collectors.pop(
                name, None
            )

        if removed is None:
            return False

        self._emit_event(
            MonitoringEventType.COLLECTOR_REMOVED,
            f"Removed collector: {name}",
            collector_name=name,
        )

        return True

    def get_collector(
        self,
        name: str,
    ) -> Optional[MetricsCollectorProtocol]:
        with self._lock:
            return self._collectors.get(name)

    def list_collectors(self) -> list[str]:
        with self._lock:
            return sorted(self._collectors)

    def add_event_callback(
        self,
        callback: Callable[[MonitoringEvent], None],
    ) -> None:
        if not callable(callback):
            raise TypeError("Callback must be callable")

        with self._lock:
            self._event_callbacks.append(callback)

    def add_alert_callback(
        self,
        callback: Callable[[MonitoringAlert], None],
    ) -> None:
        if not callable(callback):
            raise TypeError("Callback must be callable")

        with self._lock:
            self._alert_callbacks.append(callback)

    def _emit_event(
        self,
        event_type: MonitoringEventType,
        message: str,
        collector_name: Optional[str] = None,
        robot_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MonitoringEvent:
        event = MonitoringEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            message=message,
            collector_name=collector_name,
            robot_id=robot_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._events.append(event)
            callbacks = list(self._event_callbacks)

        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                logger.exception(
                    "Monitoring event callback failed"
                )

        return event

    @staticmethod
    def _normalize_health(
        value: Any,
    ) -> MonitoringHealth:
        if isinstance(value, MonitoringHealth):
            return value

        if isinstance(value, Enum):
            value = value.value

        if not isinstance(value, str):
            return MonitoringHealth.UNKNOWN

        normalized = value.strip().lower()

        aliases = {
            "ok": MonitoringHealth.HEALTHY,
            "good": MonitoringHealth.HEALTHY,
            "normal": MonitoringHealth.HEALTHY,
            "degraded": MonitoringHealth.WARNING,
            "unhealthy": MonitoringHealth.CRITICAL,
            "failed": MonitoringHealth.CRITICAL,
            "error": MonitoringHealth.CRITICAL,
        }

        if normalized in aliases:
            return aliases[normalized]

        try:
            return MonitoringHealth(normalized)
        except ValueError:
            return MonitoringHealth.UNKNOWN

    @classmethod
    def _worst_health(
        cls,
        statuses: list[MonitoringHealth],
    ) -> MonitoringHealth:
        if not statuses:
            return MonitoringHealth.UNKNOWN

        return max(
            statuses,
            key=lambda status: cls.HEALTH_PRIORITY[status],
        )

    @classmethod
    def _extract_health(
        cls,
        metrics: Mapping[str, Any],
    ) -> MonitoringHealth:
        aggregate = metrics.get("aggregate")

        if isinstance(aggregate, Mapping):
            for key in ("health_status", "health", "status"):
                if key in aggregate:
                    return cls._normalize_health(
                        aggregate[key]
                    )

        for key in ("health_status", "health"):
            if key in metrics:
                return cls._normalize_health(
                    metrics[key]
                )

        return MonitoringHealth.UNKNOWN

    @staticmethod
    def _extract_alerts(
        metrics: Mapping[str, Any],
    ) -> list[str]:
        aggregate = metrics.get("aggregate")

        if isinstance(aggregate, Mapping):
            alerts = aggregate.get("alerts", [])
        else:
            alerts = metrics.get("alerts", [])

        if not isinstance(alerts, list):
            return []

        result: list[str] = []

        for alert in alerts:
            if isinstance(alert, str) and alert:
                result.append(alert)

            elif isinstance(alert, Mapping):
                message = (
                    alert.get("message")
                    or alert.get("alert")
                    or alert.get("name")
                )

                if isinstance(message, str) and message:
                    result.append(message)

        return list(dict.fromkeys(result))

    @staticmethod
    def _extract_robot_health(
        metrics: Mapping[str, Any],
    ) -> dict[str, MonitoringHealth]:
        robots = metrics.get("robots")

        if not isinstance(robots, Mapping):
            return {}

        result: dict[str, MonitoringHealth] = {}

        for robot_id, robot_data in robots.items():
            if not isinstance(robot_data, Mapping):
                continue

            aggregate = robot_data.get("aggregate")

            if isinstance(aggregate, Mapping):
                health_value = aggregate.get(
                    "health_status",
                    aggregate.get("health"),
                )
            else:
                health_value = robot_data.get(
                    "health_status",
                    robot_data.get("health"),
                )

            result[str(robot_id)] = (
                MonitoringEngine._normalize_health(
                    health_value
                )
            )

        return result

    @staticmethod
    def _alert_severity(
        alert_name: str,
        collector_health: MonitoringHealth,
    ) -> MonitoringSeverity:
        normalized = alert_name.lower()

        if (
            "critical" in normalized
            or "collision" in normalized
            or "emergency" in normalized
        ):
            return MonitoringSeverity.CRITICAL

        if collector_health == MonitoringHealth.CRITICAL:
            return MonitoringSeverity.HIGH

        return MonitoringSeverity.WARNING

    def _activate_alert(
        self,
        collector_name: str,
        alert_name: str,
        severity: MonitoringSeverity,
        robot_id: Optional[str] = None,
    ) -> Optional[MonitoringAlert]:
        key = (
            f"{collector_name}:"
            f"{robot_id or 'fleet'}:"
            f"{alert_name}"
        )

        now = utc_now()

        with self._lock:
            existing = self._active_alerts.get(key)

            if existing is not None:
                return None

            last_time = self._last_alert_times.get(key)

            if (
                self.config.enable_alert_deduplication
                and last_time is not None
                and (
                    now - last_time
                ).total_seconds()
                < self.config.alert_cooldown_seconds
            ):
                return None

            alert = MonitoringAlert(
                alert_id=str(uuid.uuid4()),
                collector_name=collector_name,
                alert_key=key,
                severity=severity,
                message=alert_name,
                robot_id=robot_id,
                timestamp=now,
            )

            self._active_alerts[key] = alert
            self._last_alert_times[key] = now
            self._alerts.append(alert)

            self._statistics.total_alerts_triggered += 1

            callbacks = list(self._alert_callbacks)

        self._emit_event(
            MonitoringEventType.ALERT_TRIGGERED,
            alert_name,
            collector_name=collector_name,
            robot_id=robot_id,
            metadata={
                "severity": severity.value,
                "alert_id": alert.alert_id,
            },
        )

        for callback in callbacks:
            try:
                callback(alert)
            except Exception:
                logger.exception(
                    "Monitoring alert callback failed"
                )

        return alert

    def _resolve_missing_alerts(
        self,
        collector_name: str,
        observed_keys: set[str],
    ) -> None:
        with self._lock:
            keys_to_resolve = [
                key
                for key in self._active_alerts
                if key.startswith(f"{collector_name}:")
                and key not in observed_keys
            ]

            resolved = [
                self._active_alerts.pop(key)
                for key in keys_to_resolve
            ]

            self._statistics.total_alerts_resolved += len(
                resolved
            )

        for alert in resolved:
            self._emit_event(
                MonitoringEventType.ALERT_RESOLVED,
                f"Resolved: {alert.message}",
                collector_name=collector_name,
                robot_id=alert.robot_id,
                metadata={
                    "alert_id": alert.alert_id,
                },
            )

    def _collect_one(
        self,
        name: str,
        collector: MetricsCollectorProtocol,
    ) -> CollectorSnapshot:
        started = time.perf_counter()
        now = utc_now()

        try:
            raw_metrics = collector.get_fleet_metrics()

            if not isinstance(raw_metrics, Mapping):
                raise TypeError(
                    "Collector result must be a mapping"
                )

            metrics = json_safe(raw_metrics)

            duration_ms = (
                time.perf_counter() - started
            ) * 1000.0

            if (
                duration_ms
                > self.config.collector_timeout_seconds * 1000
            ):
                raise TimeoutError(
                    f"Collector exceeded diagnostic timeout: "
                    f"{duration_ms:.2f} ms"
                )

            health = self._extract_health(metrics)

            return CollectorSnapshot(
                collector_name=name,
                collected_at=now,
                health=health,
                metrics=metrics,
                duration_ms=round(duration_ms, 3),
                success=True,
            )

        except Exception as exc:
            duration_ms = (
                time.perf_counter() - started
            ) * 1000.0

            logger.exception(
                "Monitoring collector failed: %s",
                name,
            )

            self._emit_event(
                MonitoringEventType.COLLECTOR_FAILED,
                f"Collector {name} failed: {exc}",
                collector_name=name,
            )

            return CollectorSnapshot(
                collector_name=name,
                collected_at=now,
                health=MonitoringHealth.CRITICAL,
                metrics={},
                duration_ms=round(duration_ms, 3),
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def run_cycle(self) -> MonitoringSnapshot:
        """
        Execute one synchronous monitoring cycle.

        Concurrent cycle executions are serialized.
        """

        with self._cycle_lock:
            started = time.perf_counter()

            with self._lock:
                collectors = dict(self._collectors)

            snapshots: dict[str, CollectorSnapshot] = {}
            robot_statuses: dict[
                str, list[MonitoringHealth]
            ] = {}

            successful = 0
            failed = 0

            for name, collector in collectors.items():
                snapshot = self._collect_one(
                    name, collector
                )

                snapshots[name] = snapshot

                if snapshot.success:
                    successful += 1

                    metrics = snapshot.metrics

                    alert_names = self._extract_alerts(
                        metrics
                    )

                    observed_keys: set[str] = set()

                    for alert_name in alert_names:
                        key = (
                            f"{name}:fleet:{alert_name}"
                        )

                        observed_keys.add(key)

                        severity = self._alert_severity(
                            alert_name,
                            snapshot.health,
                        )

                        self._activate_alert(
                            collector_name=name,
                            alert_name=alert_name,
                            severity=severity,
                        )

                    self._resolve_missing_alerts(
                        name,
                        observed_keys,
                    )

                    for robot_id, health in (
                        self._extract_robot_health(
                            metrics
                        ).items()
                    ):
                        robot_statuses.setdefault(
                            robot_id, []
                        ).append(health)

                else:
                    failed += 1

                    self._activate_alert(
                        collector_name=name,
                        alert_name="collector_execution_failed",
                        severity=MonitoringSeverity.CRITICAL,
                    )

                with self._lock:
                    self._statistics.total_collector_executions += 1

                    if not snapshot.success:
                        self._statistics.total_collector_failures += 1

            robot_health = {
                robot_id: self._worst_health(statuses)
                for robot_id, statuses in robot_statuses.items()
            }

            if self.config.enable_health_aggregation:
                statuses = [
                    snapshot.health
                    for snapshot in snapshots.values()
                ]

                overall_health = self._worst_health(
                    statuses
                )
            else:
                overall_health = MonitoringHealth.UNKNOWN

            if failed > 0:
                overall_health = MonitoringHealth.CRITICAL

            duration_ms = (
                time.perf_counter() - started
            ) * 1000.0

            now = utc_now()

            with self._lock:
                active_alert_count = len(
                    self._active_alerts
                )

                current_state = self._state

                previous_health = (
                    self._history[-1].overall_health
                    if self._history
                    else None
                )

                result = MonitoringSnapshot(
                    snapshot_id=str(uuid.uuid4()),
                    timestamp=now,
                    engine_state=current_state,
                    overall_health=overall_health,
                    total_collectors=len(collectors),
                    successful_collectors=successful,
                    failed_collectors=failed,
                    active_alerts=active_alert_count,
                    collection_duration_ms=round(
                        duration_ms, 3
                    ),
                    collectors=snapshots,
                    robot_health=robot_health,
                )

                self._history.append(result)

                self._statistics.total_cycles += 1
                self._statistics.last_cycle_at = now
                self._statistics.last_cycle_duration_ms = (
                    round(duration_ms, 3)
                )

                if failed == 0:
                    self._statistics.successful_cycles += 1
                else:
                    self._statistics.failed_cycles += 1

            if (
                previous_health is not None
                and previous_health != overall_health
            ):
                self._emit_event(
                    MonitoringEventType.HEALTH_CHANGED,
                    (
                        f"Fleet monitoring health changed: "
                        f"{previous_health.value} -> "
                        f"{overall_health.value}"
                    ),
                )

            self._emit_event(
                MonitoringEventType.MONITORING_CYCLE_COMPLETED,
                "Monitoring cycle completed",
                metadata={
                    "duration_ms": round(
                        duration_ms, 3
                    ),
                    "successful_collectors": successful,
                    "failed_collectors": failed,
                },
            )

            return result

    def _background_loop(self) -> None:
        interval = (
            self.config.collection_interval_seconds
        )

        while not self._stop_event.is_set():
            cycle_start = time.monotonic()

            try:
                self.run_cycle()
            except Exception:
                logger.exception(
                    "Unexpected monitoring cycle failure"
                )

            elapsed = time.monotonic() - cycle_start

            remaining = max(
                0.0,
                interval - elapsed,
            )

            if self._stop_event.wait(remaining):
                break

    def start(self) -> None:
        with self._lock:
            if self._state in (
                EngineState.STARTING,
                EngineState.RUNNING,
            ):
                return

            if self._state == EngineState.STOPPING:
                raise RuntimeError(
                    "Cannot start while stopping"
                )

            if (
                self._thread is not None
                and self._thread.is_alive()
            ):
                raise RuntimeError(
                    "Previous monitoring thread is still active"
                )

            self._state = EngineState.STARTING
            self._stop_event.clear()

            self._statistics.started_at = utc_now()
            self._statistics.stopped_at = None

            if self.config.enable_background_monitoring:
                self._thread = threading.Thread(
                    target=self._background_loop,
                    name="robot-monitoring-engine",
                    daemon=True,
                )

                self._state = EngineState.RUNNING
                self._thread.start()

            else:
                self._state = EngineState.RUNNING

        self._emit_event(
            MonitoringEventType.ENGINE_STARTED,
            "Monitoring engine started",
        )

    def stop(
        self,
        timeout: Optional[float] = 10.0,
    ) -> None:
        with self._lock:
            if self._state in (
                EngineState.CREATED,
                EngineState.STOPPED,
            ):
                return

            self._state = EngineState.STOPPING
            self._stop_event.set()
            thread = self._thread

        if (
            thread is not None
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=timeout)

            if thread.is_alive():
                raise TimeoutError(
                    "Monitoring thread did not stop in time"
                )

        with self._lock:
            self._thread = None
            self._state = EngineState.STOPPED
            self._statistics.stopped_at = utc_now()

        self._emit_event(
            MonitoringEventType.ENGINE_STOPPED,
            "Monitoring engine stopped",
        )

    def get_latest_snapshot(
        self,
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            if not self._history:
                return None

            return self._history[-1].to_dict()

    def get_history(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError(
                "limit must be positive"
            )

        with self._lock:
            snapshots = list(self._history)[-limit:]

        return [
            snapshot.to_dict()
            for snapshot in snapshots
        ]

    def get_events(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError(
                "limit must be positive"
            )

        with self._lock:
            events = list(self._events)[-limit:]

        return [
            event.to_dict()
            for event in events
        ]

    def get_alerts(
        self,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError(
                "limit must be positive"
            )

        with self._lock:
            if active_only:
                alerts = list(
                    self._active_alerts.values()
                )
            else:
                alerts = list(self._alerts)

        return [
            alert.to_dict()
            for alert in alerts[-limit:]
        ]

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            statistics = self._statistics.to_dict()

            statistics.update({
                "state": self._state.value,
                "registered_collectors": len(
                    self._collectors
                ),
                "stored_snapshots": len(
                    self._history
                ),
                "stored_events": len(
                    self._events
                ),
                "stored_alerts": len(
                    self._alerts
                ),
                "active_alerts": len(
                    self._active_alerts
                ),
            })

        return statistics

    def get_robot_metrics(
        self,
        robot_id: str,
    ) -> dict[str, Any]:
        if not robot_id:
            raise ValueError(
                "robot_id is required"
            )

        with self._lock:
            collectors = dict(self._collectors)

        results: dict[str, Any] = {}
        errors: dict[str, str] = {}

        for name, collector in collectors.items():
            method = getattr(
                collector,
                "get_robot_metrics",
                None,
            )

            if not callable(method):
                continue

            try:
                results[name] = json_safe(
                    method(robot_id)
                )

            except Exception as exc:
                errors[name] = (
                    f"{type(exc).__name__}: {exc}"
                )

                logger.exception(
                    "Robot metrics failed: %s",
                    name,
                )

        return {
            "robot_id": robot_id,
            "collectors": results,
            "errors": errors,
            "evaluated_at": utc_now().isoformat(),
        }

    def get_fleet_metrics(self) -> dict[str, Any]:
        latest = self.get_latest_snapshot()

        if latest is None:
            return {
                "engine_state": self.state.value,
                "overall_health": MonitoringHealth.UNKNOWN.value,
                "collectors": {},
                "robot_health": {},
                "active_alerts": len(
                    self.get_alerts()
                ),
                "evaluated_at": utc_now().isoformat(),
            }

        return {
            "engine_state": self.state.value,
            "overall_health": latest["overall_health"],
            "collectors": latest["collectors"],
            "robot_health": latest["robot_health"],
            "active_alerts": latest["active_alerts"],
            "evaluated_at": latest["timestamp"],
        }

    @staticmethod
    def _prometheus_name(
        value: str,
    ) -> str:
        name = re.sub(
            r"[^a-zA-Z0-9_:]",
            "_",
            value,
        )

        if not re.match(
            r"^[a-zA-Z_:]",
            name,
        ):
            name = "_" + name

        return name

    def export_prometheus(self) -> str:
        if not self.config.enable_prometheus:
            return ""

        with self._lock:
            collectors = dict(self._collectors)

        lines: list[str] = []

        for name, collector in collectors.items():
            exporter = getattr(
                collector,
                "export_prometheus",
                None,
            )

            if not callable(exporter):
                continue

            try:
                output = exporter()

                if output:
                    lines.append(
                        f"# Collector: {name}"
                    )
                    lines.append(
                        output.rstrip()
                    )

            except Exception:
                logger.exception(
                    "Prometheus export failed: %s",
                    name,
                )

        statistics = self.get_statistics()

        engine_metrics = {
            "robot_monitoring_cycles_total":
                statistics["total_cycles"],
            "robot_monitoring_cycle_failures_total":
                statistics["failed_cycles"],
            "robot_monitoring_collector_failures_total":
                statistics["total_collector_failures"],
            "robot_monitoring_active_alerts":
                statistics["active_alerts"],
            "robot_monitoring_registered_collectors":
                statistics["registered_collectors"],
        }

        for metric_name, value in engine_metrics.items():
            metric_type = (
                "counter"
                if metric_name.endswith("_total")
                else "gauge"
            )

            lines.append(
                f"# TYPE {metric_name} {metric_type}"
            )
            lines.append(
                f"{metric_name} {value}"
            )

        return "\n".join(lines) + "\n"

    def export_json(
        self,
        indent: int = 2,
    ) -> str:
        return json.dumps(
            {
                "fleet": self.get_fleet_metrics(),
                "statistics": self.get_statistics(),
                "active_alerts": self.get_alerts(),
            },
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            self._events.clear()
            self._alerts.clear()
            self._active_alerts.clear()
            self._last_alert_times.clear()

    def __enter__(self) -> MonitoringEngine:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.stop()


monitoring_engine = MonitoringEngine()


def register_monitoring_collector(
    name: str,
    collector: MetricsCollectorProtocol,
    replace: bool = False,
) -> None:
    monitoring_engine.register_collector(
        name=name,
        collector=collector,
        replace=replace,
    )


def run_monitoring_cycle() -> dict[str, Any]:
    return monitoring_engine.run_cycle().to_dict()


def get_monitoring_health() -> str:
    latest = monitoring_engine.get_latest_snapshot()

    if latest is None:
        return MonitoringHealth.UNKNOWN.value

    return latest["overall_health"]


def get_monitoring_dashboard() -> dict[str, Any]:
    return {
        "fleet": monitoring_engine.get_fleet_metrics(),
        "statistics": monitoring_engine.get_statistics(),
        "alerts": monitoring_engine.get_alerts(),
        "events": monitoring_engine.get_events(
            limit=20
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    class DemoCollector:
        def __init__(self) -> None:
            self._count = 0

        def get_fleet_metrics(
            self,
        ) -> dict[str, Any]:
            self._count += 1

            health = (
                "warning"
                if self._count >= 3
                else "healthy"
            )

            return {
                "total_robots": 2,
                "aggregate": {
                    "health_status": health,
                    "health_score": (
                        75.0
                        if health == "warning"
                        else 98.0
                    ),
                    "alerts": (
                        ["high_cpu_utilization"]
                        if health == "warning"
                        else []
                    ),
                },
                "robots": {
                    "ROBOT-001": {
                        "aggregate": {
                            "health_status": "healthy",
                        }
                    },
                    "ROBOT-002": {
                        "aggregate": {
                            "health_status": health,
                        }
                    },
                },
            }

        def export_prometheus(self) -> str:
            return (
                "# TYPE demo_robot_count gauge\n"
                "demo_robot_count 2\n"
            )

    engine = MonitoringEngine(
        MonitoringEngineConfig(
            collection_interval_seconds=1.0,
            enable_background_monitoring=False,
        )
    )

    engine.register_collector(
        "demo",
        DemoCollector(),
    )

    engine.start()

    try:
        for _ in range(4):
            snapshot = engine.run_cycle()

            print(
                json.dumps(
                    snapshot.to_dict(),
                    indent=2,
                )
            )

        print(engine.export_prometheus())

        print(engine.export_json())

    finally:
        engine.stop()
"""Agent Metrics module.

TODO: implement and test; no real robot control is enabled.
"""

"""
backend/app/monitoring/agent_metrics.py

Multi-Agent Performance Monitoring Engine

Features:
- Agent execution and success tracking
- Latency monitoring (average, P95, P99)
- Token usage and estimated LLM costs
- Tool call success/failure metrics
- Safety violations and human escalations
- Agent task completion and retry metrics
- Per-agent and fleet-wide reports
- Thread-safe in-memory metric storage
- JSON-compatible reporting

Dependencies: Python standard library

Note:
In-memory storage is intended for development and
simulation. Use Prometheus or persistent telemetry
storage for production monitoring.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class AgentStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class AgentType(str, Enum):
    SUPERVISOR = "supervisor"
    PERCEPTION = "perception"
    PLANNING = "planning"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    HEALTH = "health"
    DIAGNOSTIC = "diagnostic"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    PARTS = "parts"
    COST = "cost"
    SAFETY = "safety"
    SIMULATION = "simulation"
    VERIFICATION = "verification"
    OTHER = "other"


@dataclass(frozen=True)
class AgentExecution:
    agent_id: str
    agent_type: str
    task_id: str
    status: AgentStatus
    latency_ms: float
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    tool_calls: int = 0
    tool_failures: int = 0
    retries: int = 0
    safety_violations: int = 0
    human_escalations: int = 0

    def __post_init__(self) -> None:
        if not self.agent_id or not self.task_id:
            raise ValueError(
                "Agent ID and task ID are required"
            )

        if self.timestamp.tzinfo is None:
            raise ValueError(
                "Timestamp must be timezone-aware"
            )

        numeric_values = (
            self.latency_ms,
            self.estimated_cost_usd,
        )

        if any(
            not math.isfinite(value) or value < 0
            for value in numeric_values
        ):
            raise ValueError(
                "Latency and cost must be finite "
                "nonnegative values"
            )

        counters = (
            self.input_tokens,
            self.output_tokens,
            self.tool_calls,
            self.tool_failures,
            self.retries,
            self.safety_violations,
            self.human_escalations,
        )

        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in counters
        ):
            raise ValueError(
                "Counters must be nonnegative integers"
            )

        if self.tool_failures > self.tool_calls:
            raise ValueError(
                "Tool failures cannot exceed tool calls"
            )


@dataclass
class AgentMetricsReport:
    agent_id: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    timed_out_executions: int
    cancelled_executions: int
    success_rate_percent: float
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    total_tool_calls: int
    failed_tool_calls: int
    tool_success_rate_percent: Optional[float]
    total_retries: int
    safety_violations: int
    human_escalations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentMetricsCollector:
    """
    Thread-safe in-memory agent metrics collector.

    max_records limits retained history, not total
    executions since process startup.
    """

    def __init__(
        self,
        max_records: int = 10000,
    ) -> None:
        if max_records < 1:
            raise ValueError(
                "max_records must be positive"
            )

        self._records: deque[AgentExecution] = deque(
            maxlen=max_records
        )
        self._lock = threading.RLock()

    def record(
        self,
        execution: AgentExecution,
    ) -> None:
        with self._lock:
            self._records.append(execution)

    def record_execution(
        self,
        agent_id: str,
        agent_type: str,
        task_id: str,
        status: AgentStatus,
        latency_ms: float,
        **kwargs: Any,
    ) -> None:
        self.record(
            AgentExecution(
                agent_id=agent_id,
                agent_type=agent_type,
                task_id=task_id,
                status=status,
                latency_ms=latency_ms,
                **kwargs,
            )
        )

    @staticmethod
    def _percentile(
        values: list[float],
        percentile: float,
    ) -> float:
        if not values:
            return 0.0

        ordered = sorted(values)

        position = (
            (len(ordered) - 1)
            * percentile / 100.0
        )

        lower = math.floor(position)
        upper = math.ceil(position)

        if lower == upper:
            return ordered[lower]

        weight = position - lower

        return (
            ordered[lower] * (1.0 - weight)
            + ordered[upper] * weight
        )

    @classmethod
    def _aggregate(
        cls,
        agent_id: str,
        records: list[AgentExecution],
    ) -> AgentMetricsReport:

        total = len(records)

        successes = sum(
            r.status == AgentStatus.SUCCESS
            for r in records
        )

        failures = sum(
            r.status == AgentStatus.FAILURE
            for r in records
        )

        timeouts = sum(
            r.status == AgentStatus.TIMEOUT
            for r in records
        )

        cancellations = sum(
            r.status == AgentStatus.CANCELLED
            for r in records
        )

        latencies = [
            r.latency_ms for r in records
        ]

        input_tokens = sum(
            r.input_tokens for r in records
        )

        output_tokens = sum(
            r.output_tokens for r in records
        )

        tool_calls = sum(
            r.tool_calls for r in records
        )

        tool_failures = sum(
            r.tool_failures for r in records
        )

        tool_success_rate = (
            100.0 * (tool_calls - tool_failures)
            / tool_calls
            if tool_calls else None
        )

        return AgentMetricsReport(
            agent_id=agent_id,
            total_executions=total,
            successful_executions=successes,
            failed_executions=failures,
            timed_out_executions=timeouts,
            cancelled_executions=cancellations,
            success_rate_percent=round(
                100.0 * successes / total, 2
            ) if total else 0.0,
            average_latency_ms=round(
                statistics.fmean(latencies), 2
            ) if latencies else 0.0,
            p95_latency_ms=round(
                cls._percentile(latencies, 95), 2
            ),
            p99_latency_ms=round(
                cls._percentile(latencies, 99), 2
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=(
                input_tokens + output_tokens
            ),
            estimated_cost_usd=round(
                sum(
                    r.estimated_cost_usd
                    for r in records
                ), 6
            ),
            total_tool_calls=tool_calls,
            failed_tool_calls=tool_failures,
            tool_success_rate_percent=(
                round(tool_success_rate, 2)
                if tool_success_rate is not None
                else None
            ),
            total_retries=sum(
                r.retries for r in records
            ),
            safety_violations=sum(
                r.safety_violations
                for r in records
            ),
            human_escalations=sum(
                r.human_escalations
                for r in records
            ),
        )

    def get_agent_metrics(
        self,
        agent_id: str,
    ) -> AgentMetricsReport:

        with self._lock:
            records = [
                record
                for record in self._records
                if record.agent_id == agent_id
            ]

        return self._aggregate(
            agent_id,
            records,
        )

    def get_all_agent_metrics(
        self,
    ) -> dict[str, dict[str, Any]]:

        with self._lock:
            records = list(self._records)

        grouped: dict[
            str, list[AgentExecution]
        ] = defaultdict(list)

        for record in records:
            grouped[record.agent_id].append(record)

        return {
            agent_id: self._aggregate(
                agent_id, executions
            ).to_dict()
            for agent_id, executions in grouped.items()
        }

    def get_system_metrics(
        self,
    ) -> dict[str, Any]:

        with self._lock:
            records = list(self._records)

        report = self._aggregate(
            "all_agents",
            records,
        ).to_dict()

        report["active_agent_count"] = len(
            {r.agent_id for r in records}
        )

        report["unique_task_count"] = len(
            {r.task_id for r in records}
        )

        report["retained_records"] = len(records)

        return report

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


class AgentExecutionTimer:
    """
    Context manager for recording agent execution time.

    Automatically records SUCCESS on normal exit,
    FAILURE when an exception is raised.

    Does not suppress exceptions.
    """

    def __init__(
        self,
        collector: AgentMetricsCollector,
        agent_id: str,
        agent_type: str,
        task_id: str,
    ):
        self.collector = collector
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.task_id = task_id
        self.started_at = 0.0

    def __enter__(self) -> "AgentExecutionTimer":
        self.started_at = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> bool:

        latency_ms = (
            time.perf_counter() - self.started_at
        ) * 1000.0

        status = (
            AgentStatus.SUCCESS
            if exc_type is None
            else AgentStatus.FAILURE
        )

        self.collector.record_execution(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            task_id=self.task_id,
            status=status,
            latency_ms=latency_ms,
        )

        return False


# Shared process-local collector.
agent_metrics = AgentMetricsCollector()


def record_agent_metric(
    agent_id: str,
    agent_type: str,
    task_id: str,
    success: bool,
    latency_ms: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    tool_calls: int = 0,
    tool_failures: int = 0,
    retries: int = 0,
    safety_violations: int = 0,
    human_escalations: int = 0,
) -> None:
    """Convenience function for agent integrations."""

    agent_metrics.record_execution(
        agent_id=agent_id,
        agent_type=agent_type,
        task_id=task_id,
        status=(
            AgentStatus.SUCCESS
            if success else AgentStatus.FAILURE
        ),
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost_usd,
        tool_calls=tool_calls,
        tool_failures=tool_failures,
        retries=retries,
        safety_violations=safety_violations,
        human_escalations=human_escalations,
    )


if __name__ == "__main__":
    import json

    collector = AgentMetricsCollector()

    collector.record_execution(
        agent_id="navigation-agent-01",
        agent_type=AgentType.NAVIGATION.value,
        task_id="mission-001",
        status=AgentStatus.SUCCESS,
        latency_ms=125.5,
        tool_calls=3,
    )

    collector.record_execution(
        agent_id="navigation-agent-01",
        agent_type=AgentType.NAVIGATION.value,
        task_id="mission-002",
        status=AgentStatus.FAILURE,
        latency_ms=310.2,
        tool_calls=2,
        tool_failures=1,
        retries=1,
    )

    collector.record_execution(
        agent_id="safety-agent-01",
        agent_type=AgentType.SAFETY.value,
        task_id="mission-003",
        status=AgentStatus.SUCCESS,
        latency_ms=18.4,
    )

    print(
        json.dumps(
            collector.get_system_metrics(),
            indent=2,
        )
    )

"""Downtime Cost module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/downtime_cost.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robotics Downtime Cost Estimation Engine

Python 3.11+
Standard library only.

Features:
    - Multi-tenant robot downtime tracking
    - Planned and unplanned downtime
    - Production loss estimation
    - Labor idle-time costs
    - Emergency repair and recovery costs
    - SLA penalty estimation
    - Energy and restart costs
    - Downtime incident lifecycle
    - Actual and projected downtime
    - Cost breakdown by category
    - Fleet downtime analytics
    - MTTR and MTBF calculations
    - Human-reviewed financial estimates
    - Thread-safe incident management
    - JSON report export

Important:
    - Planning and simulation use only.
    - No physical robot control.
    - Monetary inputs are user-supplied assumptions.
    - No automatic market-price retrieval.
    - Avoid double counting overlapping cost categories.
    - MTBF is calculated from supplied operating hours.
"""

from __future__ import annotations

import json
import re
import threading
import uuid

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class DowntimeCostError(Exception):
    pass


class DowntimeValidationError(DowntimeCostError):
    pass


class DowntimeIncidentNotFound(DowntimeCostError):
    pass


class DowntimeConflict(DowntimeCostError):
    pass


class DowntimePermissionDenied(DowntimeCostError):
    pass


# ============================================================
# ENUMS
# ============================================================

class DowntimeType(str, Enum):
    PLANNED = "planned"
    UNPLANNED = "unplanned"
    EMERGENCY = "emergency"


class DowntimeCause(str, Enum):
    MECHANICAL_FAILURE = "mechanical_failure"
    ELECTRICAL_FAILURE = "electrical_failure"
    SENSOR_FAILURE = "sensor_failure"
    SOFTWARE_FAILURE = "software_failure"
    NETWORK_FAILURE = "network_failure"
    BATTERY_FAILURE = "battery_failure"
    COLLISION = "collision"
    MAINTENANCE = "maintenance"
    CALIBRATION = "calibration"
    SAFETY_EVENT = "safety_event"
    POWER_OUTAGE = "power_outage"
    OTHER = "other"


class IncidentStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"


class CostCategory(str, Enum):
    PRODUCTION_LOSS = "production_loss"
    IDLE_LABOR = "idle_labor"
    REPAIR = "repair"
    EMERGENCY = "emergency"
    SLA_PENALTY = "sla_penalty"
    RESTART = "restart"
    ENERGY = "energy"
    OTHER = "other"


class AuditEventType(str, Enum):
    INCIDENT_OPENED = "incident_opened"
    INCIDENT_RESOLVED = "incident_resolved"
    ESTIMATE_CALCULATED = "estimate_calculated"


# ============================================================
# UTILITIES
# ============================================================

MONEY_QUANTUM = Decimal("0.01")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}",
            value,
        )
    ):
        raise DowntimeValidationError(
            f"Invalid {name}"
        )

    return value


def validate_currency(value: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Z]{3}", value)
    ):
        raise DowntimeValidationError(
            "Currency must be a three-letter ISO-style code"
        )

    return value


def validate_text(
    value: str,
    name: str,
    maximum_length: int = 1024,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum_length
    ):
        raise DowntimeValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def decimal_value(
    value: Any,
    name: str,
    *,
    minimum: Decimal = Decimal("0"),
) -> Decimal:
    if isinstance(value, bool):
        raise DowntimeValidationError(
            f"{name} must be numeric"
        )

    try:
        result = Decimal(str(value))
    except (
        InvalidOperation,
        ValueError,
        TypeError,
    ) as exc:
        raise DowntimeValidationError(
            f"Invalid {name}"
        ) from exc

    if not result.is_finite():
        raise DowntimeValidationError(
            f"{name} must be finite"
        )

    if result < minimum:
        raise DowntimeValidationError(
            f"{name} must be at least {minimum}"
        )

    return result


def money(value: Decimal) -> Decimal:
    return value.quantize(
        MONEY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def money_string(value: Decimal) -> str:
    return str(money(value))


def validate_datetime(
    value: datetime,
    name: str,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise DowntimeValidationError(
            f"{name} must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def hours_between(
    start: datetime,
    end: datetime,
) -> Decimal:
    start = validate_datetime(start, "start")
    end = validate_datetime(end, "end")

    if end < start:
        raise DowntimeValidationError(
            "End time cannot precede start time"
        )

    duration = end - start

    seconds = (
        Decimal(duration.days) * Decimal("86400")
        + Decimal(duration.seconds)
        + Decimal(duration.microseconds)
        / Decimal("1000000")
    )

    return seconds / Decimal("3600")


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DowntimeCostConfig:
    maximum_incidents: int = 100000
    maximum_audit_events: int = 20000

    maximum_projected_hours: Decimal = (
        Decimal("8760")
    )

    default_currency: str = "INR"

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_incidents",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise DowntimeValidationError(
                    f"{name} must be a positive integer"
                )

        object.__setattr__(
            self,
            "maximum_projected_hours",
            decimal_value(
                self.maximum_projected_hours,
                "maximum_projected_hours",
                minimum=Decimal("0.01"),
            ),
        )

        validate_currency(
            self.default_currency
        )

        if self.simulation_only is not True:
            raise DowntimeValidationError(
                "Only simulation/planning mode is supported"
            )


# ============================================================
# COST RATES
# ============================================================

@dataclass(frozen=True)
class DowntimeCostRates:
    currency: str = "INR"

    production_loss_per_hour: Decimal = Decimal("0")
    idle_labor_per_hour: Decimal = Decimal("0")
    energy_cost_per_hour: Decimal = Decimal("0")

    repair_fixed_cost: Decimal = Decimal("0")
    emergency_fixed_cost: Decimal = Decimal("0")
    restart_fixed_cost: Decimal = Decimal("0")
    other_fixed_cost: Decimal = Decimal("0")

    sla_penalty_per_hour: Decimal = Decimal("0")
    sla_grace_hours: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        validate_currency(self.currency)

        for name in (
            "production_loss_per_hour",
            "idle_labor_per_hour",
            "energy_cost_per_hour",
            "repair_fixed_cost",
            "emergency_fixed_cost",
            "restart_fixed_cost",
            "other_fixed_cost",
            "sla_penalty_per_hour",
            "sla_grace_hours",
        ):
            object.__setattr__(
                self,
                name,
                decimal_value(
                    getattr(self, name),
                    name,
                ),
            )

    def to_dict(self) -> dict[str, str]:
        result = {
            "currency": self.currency,
        }

        for name in (
            "production_loss_per_hour",
            "idle_labor_per_hour",
            "energy_cost_per_hour",
            "repair_fixed_cost",
            "emergency_fixed_cost",
            "restart_fixed_cost",
            "other_fixed_cost",
            "sla_penalty_per_hour",
            "sla_grace_hours",
        ):
            result[name] = str(
                getattr(self, name)
            )

        return result


# ============================================================
# INCIDENT REQUEST
# ============================================================

@dataclass(frozen=True)
class DowntimeIncidentRequest:
    tenant_id: str
    robot_id: str

    downtime_type: DowntimeType
    cause: DowntimeCause

    description: str
    started_at: datetime

    cost_rates: DowntimeCostRates

    projected_downtime_hours: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        if not isinstance(
            self.downtime_type,
            DowntimeType,
        ):
            raise DowntimeValidationError(
                "Invalid downtime type"
            )

        if not isinstance(
            self.cause,
            DowntimeCause,
        ):
            raise DowntimeValidationError(
                "Invalid downtime cause"
            )

        validate_text(
            self.description,
            "description",
        )

        object.__setattr__(
            self,
            "started_at",
            validate_datetime(
                self.started_at,
                "started_at",
            ),
        )

        if not isinstance(
            self.cost_rates,
            DowntimeCostRates,
        ):
            raise DowntimeValidationError(
                "Invalid cost rates"
            )

        object.__setattr__(
            self,
            "projected_downtime_hours",
            decimal_value(
                self.projected_downtime_hours,
                "projected_downtime_hours",
            ),
        )


# ============================================================
# COST BREAKDOWN
# ============================================================

@dataclass(frozen=True)
class DowntimeCostBreakdown:
    currency: str
    duration_hours: Decimal

    production_loss: Decimal
    idle_labor_cost: Decimal
    energy_cost: Decimal

    repair_cost: Decimal
    emergency_cost: Decimal
    restart_cost: Decimal
    other_cost: Decimal

    sla_penalty: Decimal

    total_cost: Decimal

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "duration_hours": str(
                self.duration_hours
            ),
            "production_loss": money_string(
                self.production_loss
            ),
            "idle_labor_cost": money_string(
                self.idle_labor_cost
            ),
            "energy_cost": money_string(
                self.energy_cost
            ),
            "repair_cost": money_string(
                self.repair_cost
            ),
            "emergency_cost": money_string(
                self.emergency_cost
            ),
            "restart_cost": money_string(
                self.restart_cost
            ),
            "other_cost": money_string(
                self.other_cost
            ),
            "sla_penalty": money_string(
                self.sla_penalty
            ),
            "total_cost": money_string(
                self.total_cost
            ),
        }


# ============================================================
# INCIDENT RECORD
# ============================================================

@dataclass
class DowntimeIncidentRecord:
    incident_id: str
    request: DowntimeIncidentRequest

    status: IncidentStatus = IncidentStatus.OPEN

    resolved_at: datetime | None = None

    resolution_notes: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "tenant_id": self.request.tenant_id,
            "robot_id": self.request.robot_id,
            "downtime_type": (
                self.request.downtime_type.value
            ),
            "cause": self.request.cause.value,
            "description": self.request.description,
            "started_at": (
                self.request.started_at.isoformat()
            ),
            "resolved_at": (
                self.resolved_at.isoformat()
                if self.resolved_at
                else None
            ),
            "status": self.status.value,
            "projected_downtime_hours": str(
                self.request.projected_downtime_hours
            ),
            "cost_rates": (
                self.request.cost_rates.to_dict()
            ),
            "resolution_notes": self.resolution_notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class DowntimeAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    incident_id: str

    event_type: AuditEventType
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "incident_id": self.incident_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# DOWNTIME COST ENGINE
# ============================================================

class RoboticsDowntimeCostEngine:
    """
    Thread-safe downtime tracking and cost estimation.

    Cost formula:

        Production loss =
            downtime_hours * production_loss_per_hour

        Idle labor =
            downtime_hours * idle_labor_per_hour

        Energy =
            downtime_hours * energy_cost_per_hour

        SLA penalty =
            max(0, downtime_hours - grace_hours)
            * sla_penalty_per_hour

        Total =
            production loss
            + idle labor
            + energy
            + repair
            + emergency
            + restart
            + other
            + SLA penalty

    Fixed costs are included once per incident.

    Cost estimates use the rates captured when
    the incident was created.
    """

    def __init__(
        self,
        config: DowntimeCostConfig | None = None,
    ) -> None:
        self.config = (
            config or DowntimeCostConfig()
        )

        self._incidents: dict[
            str,
            DowntimeIncidentRecord,
        ] = {}

        self._audit_events: deque[
            DowntimeAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        record: DowntimeIncidentRecord,
        event_type: AuditEventType,
        details: str,
    ) -> None:
        self._audit_events.append(
            DowntimeAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=record.request.tenant_id,
                robot_id=record.request.robot_id,
                incident_id=record.incident_id,
                event_type=event_type,
                details=details,
            )
        )

    # --------------------------------------------------------
    # GET INTERNAL
    # --------------------------------------------------------

    def _get(
        self,
        incident_id: str,
        tenant_id: str,
    ) -> DowntimeIncidentRecord:
        validate_id(incident_id, "incident_id")
        validate_id(tenant_id, "tenant_id")

        record = self._incidents.get(
            incident_id
        )

        if (
            record is None
            or record.request.tenant_id != tenant_id
        ):
            raise DowntimeIncidentNotFound(
                "Downtime incident not found"
            )

        return record

    # --------------------------------------------------------
    # OPEN INCIDENT
    # --------------------------------------------------------

    def open_incident(
        self,
        request: DowntimeIncidentRequest,
    ) -> dict[str, Any]:
        if not isinstance(
            request,
            DowntimeIncidentRequest,
        ):
            raise DowntimeValidationError(
                "Invalid incident request"
            )

        if (
            request.projected_downtime_hours
            > self.config.maximum_projected_hours
        ):
            raise DowntimeValidationError(
                "Projected downtime exceeds limit"
            )

        if request.started_at > utc_now():
            raise DowntimeValidationError(
                "Incident start cannot be in the future"
            )

        with self._lock:
            if (
                len(self._incidents)
                >= self.config.maximum_incidents
            ):
                raise DowntimeCostError(
                    "Incident capacity reached"
                )

            record = DowntimeIncidentRecord(
                incident_id=str(uuid.uuid4()),
                request=request,
            )

            self._incidents[
                record.incident_id
            ] = record

            self._audit(
                record,
                AuditEventType.INCIDENT_OPENED,
                "Downtime incident opened",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # RESOLVE INCIDENT
    # --------------------------------------------------------

    def resolve_incident(
        self,
        *,
        tenant_id: str,
        incident_id: str,
        resolved_at: datetime,
        resolution_notes: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise DowntimePermissionDenied(
                "Resolution authorization required"
            )

        resolved_at = validate_datetime(
            resolved_at,
            "resolved_at",
        )

        notes = validate_text(
            resolution_notes,
            "resolution_notes",
        )

        with self._lock:
            record = self._get(
                incident_id,
                tenant_id,
            )

            if record.status == IncidentStatus.RESOLVED:
                raise DowntimeConflict(
                    "Incident already resolved"
                )

            if resolved_at < record.request.started_at:
                raise DowntimeValidationError(
                    "Resolution precedes incident start"
                )

            if resolved_at > utc_now():
                raise DowntimeValidationError(
                    "Resolution cannot be in the future"
                )

            record.status = IncidentStatus.RESOLVED
            record.resolved_at = resolved_at
            record.resolution_notes = notes
            record.updated_at = utc_now()

            self._audit(
                record,
                AuditEventType.INCIDENT_RESOLVED,
                "Downtime incident resolved",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # CALCULATE COST
    # --------------------------------------------------------

    @staticmethod
    def calculate_cost(
        duration_hours: Decimal,
        rates: DowntimeCostRates,
    ) -> DowntimeCostBreakdown:
        hours = decimal_value(
            duration_hours,
            "duration_hours",
        )

        if not isinstance(
            rates,
            DowntimeCostRates,
        ):
            raise DowntimeValidationError(
                "Invalid cost rates"
            )

        production_loss = (
            hours * rates.production_loss_per_hour
        )

        idle_labor = (
            hours * rates.idle_labor_per_hour
        )

        energy = (
            hours * rates.energy_cost_per_hour
        )

        sla_billable_hours = max(
            Decimal("0"),
            hours - rates.sla_grace_hours,
        )

        sla_penalty = (
            sla_billable_hours
            * rates.sla_penalty_per_hour
        )

        repair = rates.repair_fixed_cost
        emergency = rates.emergency_fixed_cost
        restart = rates.restart_fixed_cost
        other = rates.other_fixed_cost

        total = sum(
            (
                production_loss,
                idle_labor,
                energy,
                repair,
                emergency,
                restart,
                other,
                sla_penalty,
            ),
            Decimal("0"),
        )

        return DowntimeCostBreakdown(
            currency=rates.currency,
            duration_hours=hours,
            production_loss=production_loss,
            idle_labor_cost=idle_labor,
            energy_cost=energy,
            repair_cost=repair,
            emergency_cost=emergency,
            restart_cost=restart,
            other_cost=other,
            sla_penalty=sla_penalty,
            total_cost=total,
        )

    # --------------------------------------------------------
    # ESTIMATE INCIDENT
    # --------------------------------------------------------

    def estimate_incident(
        self,
        *,
        tenant_id: str,
        incident_id: str,
        projected_hours: Decimal | None = None,
        as_of: datetime | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                incident_id,
                tenant_id,
            )

            if record.status == IncidentStatus.RESOLVED:
                duration = hours_between(
                    record.request.started_at,
                    record.resolved_at,
                )

                duration_source = "actual"

            elif projected_hours is not None:
                duration = decimal_value(
                    projected_hours,
                    "projected_hours",
                )

                duration_source = "projection"

            elif (
                record.request.projected_downtime_hours
                > 0
            ):
                duration = (
                    record.request.projected_downtime_hours
                )

                duration_source = "stored_projection"

            else:
                now = (
                    validate_datetime(as_of, "as_of")
                    if as_of is not None
                    else utc_now()
                )

                duration = hours_between(
                    record.request.started_at,
                    now,
                )

                duration_source = "elapsed_so_far"

            if (
                duration_source != "actual"
                and duration
                > self.config.maximum_projected_hours
            ):
                raise DowntimeValidationError(
                    "Projected duration exceeds limit"
                )

            breakdown = self.calculate_cost(
                duration,
                record.request.cost_rates,
            )

            result = {
                "incident_id": record.incident_id,
                "tenant_id": record.request.tenant_id,
                "robot_id": record.request.robot_id,
                "downtime_type": (
                    record.request.downtime_type.value
                ),
                "cause": record.request.cause.value,
                "status": record.status.value,
                "duration_source": duration_source,
                "cost_breakdown": (
                    breakdown.to_dict()
                ),
                "estimated_at": utc_now().isoformat(),
            }

            self._audit(
                record,
                AuditEventType.ESTIMATE_CALCULATED,
                (
                    "Downtime cost calculated "
                    f"using {duration_source}"
                ),
            )

            return result

    # --------------------------------------------------------
    # GET INCIDENT
    # --------------------------------------------------------

    def get_incident(
        self,
        *,
        tenant_id: str,
        incident_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get(
                incident_id,
                tenant_id,
            ).to_dict()

    # --------------------------------------------------------
    # LIST INCIDENTS
    # --------------------------------------------------------

    def list_incidents(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        status: IncidentStatus | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        if (
            status is not None
            and not isinstance(status, IncidentStatus)
        ):
            raise DowntimeValidationError(
                "Invalid incident status"
            )

        with self._lock:
            records = [
                record
                for record in self._incidents.values()
                if (
                    record.request.tenant_id == tenant_id
                    and (
                        robot_id is None
                        or record.request.robot_id == robot_id
                    )
                    and (
                        status is None
                        or record.status == status
                    )
                )
            ]

            records.sort(
                key=lambda item: item.request.started_at
            )

            return [
                record.to_dict()
                for record in records
            ]

    # --------------------------------------------------------
    # FLEET COST SUMMARY
    # --------------------------------------------------------

    def fleet_cost_summary(
        self,
        *,
        tenant_id: str,
        as_of: datetime | None = None,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")

        now = (
            validate_datetime(as_of, "as_of")
            if as_of is not None
            else utc_now()
        )

        with self._lock:
            records = [
                record
                for record in self._incidents.values()
                if record.request.tenant_id == tenant_id
            ]

            totals: dict[
                str,
                dict[str, Decimal],
            ] = defaultdict(
                lambda: defaultdict(
                    lambda: Decimal("0")
                )
            )

            robot_totals: dict[
                str,
                dict[str, dict[str, Decimal]],
            ] = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: Decimal("0")
                    )
                )
            )

            for record in records:
                if record.status == IncidentStatus.RESOLVED:
                    duration = hours_between(
                        record.request.started_at,
                        record.resolved_at,
                    )
                elif (
                    record.request.projected_downtime_hours
                    > 0
                ):
                    duration = (
                        record.request.projected_downtime_hours
                    )
                else:
                    duration = hours_between(
                        record.request.started_at,
                        now,
                    )

                breakdown = self.calculate_cost(
                    duration,
                    record.request.cost_rates,
                )

                currency = (
                    record.request.cost_rates.currency
                )

                robot_id = record.request.robot_id

                totals[currency]["total_cost"] += (
                    breakdown.total_cost
                )

                totals[currency]["downtime_hours"] += (
                    duration
                )

                robot_totals[robot_id][currency][
                    "total_cost"
                ] += breakdown.total_cost

                robot_totals[robot_id][currency][
                    "downtime_hours"
                ] += duration

            return {
                "tenant_id": tenant_id,
                "incident_count": len(records),
                "open_incidents": sum(
                    record.status == IncidentStatus.OPEN
                    for record in records
                ),
                "resolved_incidents": sum(
                    record.status == IncidentStatus.RESOLVED
                    for record in records
                ),
                "totals_by_currency": {
                    currency: {
                        "total_cost": money_string(
                            values["total_cost"]
                        ),
                        "downtime_hours": str(
                            values["downtime_hours"]
                        ),
                    }
                    for currency, values in totals.items()
                },
                "robots": {
                    robot_id: {
                        currency: {
                            "total_cost": money_string(
                                values["total_cost"]
                            ),
                            "downtime_hours": str(
                                values["downtime_hours"]
                            ),
                        }
                        for currency, values
                        in currencies.items()
                    }
                    for robot_id, currencies
                    in robot_totals.items()
                },
                "note": (
                    "Open incidents use projections or "
                    "elapsed duration; these are not "
                    "finalized actual costs."
                ),
            }

    # --------------------------------------------------------
    # RELIABILITY METRICS
    # --------------------------------------------------------

    def reliability_metrics(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        operating_hours: Decimal,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(robot_id, "robot_id")

        operating_hours = decimal_value(
            operating_hours,
            "operating_hours",
        )

        with self._lock:
            records = [
                record
                for record in self._incidents.values()
                if (
                    record.request.tenant_id == tenant_id
                    and record.request.robot_id == robot_id
                    and record.status
                    == IncidentStatus.RESOLVED
                    and record.request.downtime_type
                    != DowntimeType.PLANNED
                )
            ]

            failure_count = len(records)

            total_repair_hours = sum(
                (
                    hours_between(
                        record.request.started_at,
                        record.resolved_at,
                    )
                    for record in records
                ),
                Decimal("0"),
            )

            mttr = (
                total_repair_hours / failure_count
                if failure_count
                else None
            )

            mtbf = (
                operating_hours / failure_count
                if failure_count
                else None
            )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "failure_incident_count": failure_count,
                "operating_hours": str(operating_hours),
                "total_repair_hours": str(
                    total_repair_hours
                ),
                "mttr_hours": (
                    str(mttr)
                    if mttr is not None
                    else None
                ),
                "mtbf_hours": (
                    str(mtbf)
                    if mtbf is not None
                    else None
                ),
                "note": (
                    "Simplified reliability metrics. "
                    "Each resolved unplanned/emergency "
                    "incident is counted as one failure. "
                    "Operating hours must refer to the "
                    "same observation period."
                ),
            }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
    ) -> Path:
        validate_id(tenant_id, "tenant_id")

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "fleet_summary": (
                    self.fleet_cost_summary(
                        tenant_id=tenant_id
                    )
                ),
                "incidents": self.list_incidents(
                    tenant_id=tenant_id
                ),
                "audit_events": [
                    event.to_dict()
                    for event in self._audit_events
                    if event.tenant_id == tenant_id
                ],
                "simulation_only": True,
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
    engine = RoboticsDowntimeCostEngine()

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"

    rates = DowntimeCostRates(
        currency="INR",
        production_loss_per_hour=Decimal("5000"),
        idle_labor_per_hour=Decimal("1200"),
        energy_cost_per_hour=Decimal("100"),
        repair_fixed_cost=Decimal("8000"),
        emergency_fixed_cost=Decimal("2000"),
        restart_fixed_cost=Decimal("1500"),
        sla_penalty_per_hour=Decimal("500"),
        sla_grace_hours=Decimal("2"),
    )

    started_at = datetime(
        2026, 10, 13, 8, 0,
        tzinfo=timezone.utc,
    )

    incident = engine.open_incident(
        DowntimeIncidentRequest(
            tenant_id=tenant_id,
            robot_id=robot_id,
            downtime_type=DowntimeType.UNPLANNED,
            cause=DowntimeCause.MECHANICAL_FAILURE,
            description=(
                "Simulated robot motor failure"
            ),
            started_at=started_at,
            cost_rates=rates,
            projected_downtime_hours=Decimal("5"),
        )
    )

    print(
        "Opened incident:",
        json.dumps(
            incident,
            indent=2,
        ),
    )

    estimate = engine.estimate_incident(
        tenant_id=tenant_id,
        incident_id=incident["incident_id"],
    )

    print(
        "\nDowntime cost estimate:",
        json.dumps(
            estimate,
            indent=2,
        ),
    )

    resolved = engine.resolve_incident(
        tenant_id=tenant_id,
        incident_id=incident["incident_id"],
        resolved_at=datetime(
            2026, 10, 13, 11, 30,
            tzinfo=timezone.utc,
        ),
        resolution_notes=(
            "Simulated motor repair completed"
        ),
        authorized=True,
    )

    print(
        "\nResolved incident:",
        json.dumps(
            resolved,
            indent=2,
        ),
    )

    actual = engine.estimate_incident(
        tenant_id=tenant_id,
        incident_id=incident["incident_id"],
    )

    print(
        "\nActual-duration cost calculation:",
        json.dumps(
            actual,
            indent=2,
        ),
    )

    print(
        "\nReliability metrics:",
        json.dumps(
            engine.reliability_metrics(
                tenant_id=tenant_id,
                robot_id=robot_id,
                operating_hours=Decimal("1000"),
            ),
            indent=2,
        ),
    )

    report = engine.export_json(
        "data/robotics/downtime_cost_report.json",
        tenant_id=tenant_id,
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()
"""Maintenance Service module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/services/maintenance_service.py

Robotics Maintenance Intelligence Service

Python 3.11+
Standard library only.

Features:
    - Preventive maintenance
    - Predictive maintenance
    - Robot health monitoring
    - Remaining useful life (RUL)
    - Battery, motor and vibration assessment
    - Maintenance scheduling
    - Work order management
    - Repair and replacement recommendations
    - Spare parts inventory
    - Repair cost estimation
    - Maintenance history
    - Fleet maintenance analytics
    - Simulation telemetry integration
    - JSON and CSV reports

Safety:
    Decision-support only.
    No physical robot or actuator control.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class MaintenanceServiceError(Exception):
    pass


class MaintenanceValidationError(MaintenanceServiceError):
    pass


class MaintenanceNotFoundError(MaintenanceServiceError):
    pass


class MaintenanceStateError(MaintenanceServiceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class MaintenanceType(str, Enum):
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"


class MaintenancePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WorkOrderStatus(str, Enum):
    OPEN = "open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ComponentType(str, Enum):
    BATTERY = "battery"
    MOTOR = "motor"
    SENSOR = "sensor"
    JOINT = "joint"
    WHEEL = "wheel"
    CONTROLLER = "controller"
    CAMERA = "camera"
    GEARBOX = "gearbox"
    OTHER = "other"


class MaintenanceAction(str, Enum):
    INSPECT = "inspect"
    SERVICE = "service"
    REPAIR = "repair"
    REPLACE = "replace"
    CALIBRATE = "calibrate"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def validate_number(
    value: float,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise MaintenanceValidationError(
            f"{name} must be finite"
        )

    result = float(value)

    if minimum is not None and result < minimum:
        raise MaintenanceValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and result > maximum:
        raise MaintenanceValidationError(
            f"{name} must be <= {maximum}"
        )

    return result


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    return max(minimum, min(maximum, value))


def require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MaintenanceValidationError(
            f"{name} is required"
        )

    return value.strip()


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class MaintenanceServiceConfig:
    currency: str = "INR"

    preventive_interval_hours: float = 250.0

    warning_health_score: float = 70.0
    critical_health_score: float = 40.0

    warning_rul_hours: float = 100.0
    critical_rul_hours: float = 24.0

    motor_warning_temperature_c: float = 70.0
    motor_critical_temperature_c: float = 90.0

    vibration_warning_mm_s: float = 6.0
    vibration_critical_mm_s: float = 12.0

    battery_warning_soc_percent: float = 20.0
    battery_critical_soc_percent: float = 10.0

    labor_rate_per_hour: float = 350.0
    downtime_loss_per_hour: float = 1500.0

    maximum_history: int = 10000

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        require_text(self.currency, "currency")

        for name in (
            "preventive_interval_hours",
            "warning_rul_hours",
            "critical_rul_hours",
            "vibration_warning_mm_s",
            "vibration_critical_mm_s",
            "labor_rate_per_hour",
            "downtime_loss_per_hour",
        ):
            validate_number(
                getattr(self, name),
                name,
                minimum=0.0,
            )

        if self.preventive_interval_hours <= 0:
            raise MaintenanceValidationError(
                "Preventive interval must be positive"
            )

        if not (
            0 <= self.critical_health_score
            < self.warning_health_score
            <= 100
        ):
            raise MaintenanceValidationError(
                "Invalid health thresholds"
            )

        if not (
            self.critical_rul_hours
            < self.warning_rul_hours
        ):
            raise MaintenanceValidationError(
                "Invalid RUL thresholds"
            )

        if not (
            self.motor_warning_temperature_c
            < self.motor_critical_temperature_c
        ):
            raise MaintenanceValidationError(
                "Invalid temperature thresholds"
            )

        if not (
            self.vibration_warning_mm_s
            < self.vibration_critical_mm_s
        ):
            raise MaintenanceValidationError(
                "Invalid vibration thresholds"
            )

        if not (
            0 <= self.battery_critical_soc_percent
            < self.battery_warning_soc_percent
            <= 100
        ):
            raise MaintenanceValidationError(
                "Invalid battery thresholds"
            )

        if self.maximum_history < 1:
            raise MaintenanceValidationError(
                "maximum_history must be positive"
            )

        if self.allow_hardware_control:
            raise MaintenanceValidationError(
                "Hardware control is prohibited"
            )


# ============================================================
# ROBOT HEALTH TELEMETRY
# ============================================================

@dataclass(frozen=True)
class RobotMaintenanceTelemetry:
    robot_id: str
    operating_hours: float

    battery_soc_percent: float
    motor_temperature_c: float
    vibration_mm_s: float

    health_score: float

    rul_hours: float | None = None

    fault_count: int = 0

    sensor_available: bool = True

    def __post_init__(self) -> None:
        require_text(self.robot_id, "robot_id")

        for name in (
            "operating_hours",
            "vibration_mm_s",
        ):
            validate_number(
                getattr(self, name),
                name,
                minimum=0.0,
            )

        for name in (
            "battery_soc_percent",
            "health_score",
        ):
            validate_number(
                getattr(self, name),
                name,
                minimum=0.0,
                maximum=100.0,
            )

        validate_number(
            self.motor_temperature_c,
            "motor_temperature_c",
        )

        if self.rul_hours is not None:
            validate_number(
                self.rul_hours,
                "rul_hours",
                minimum=0.0,
            )

        if self.fault_count < 0:
            raise MaintenanceValidationError(
                "fault_count cannot be negative"
            )


# ============================================================
# MAINTENANCE RECOMMENDATION
# ============================================================

@dataclass(frozen=True)
class MaintenanceRecommendation:
    recommendation_id: str

    robot_id: str

    maintenance_type: MaintenanceType
    priority: MaintenancePriority
    action: MaintenanceAction

    component: ComponentType

    reason: str

    estimated_rul_hours: float | None

    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "robot_id": self.robot_id,
            "maintenance_type": (
                self.maintenance_type.value
            ),
            "priority": self.priority.value,
            "action": self.action.value,
            "component": self.component.value,
            "reason": self.reason,
            "estimated_rul_hours": (
                self.estimated_rul_hours
            ),
            "created_at": self.created_at,
        }


# ============================================================
# PREDICTIVE MAINTENANCE ENGINE
# ============================================================

class PredictiveMaintenanceEngine:

    def __init__(
        self,
        config: MaintenanceServiceConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        telemetry: RobotMaintenanceTelemetry,
    ) -> list[MaintenanceRecommendation]:
        recommendations = []

        def add(
            *,
            maintenance_type: MaintenanceType,
            priority: MaintenancePriority,
            action: MaintenanceAction,
            component: ComponentType,
            reason: str,
        ) -> None:
            recommendations.append(
                MaintenanceRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    robot_id=telemetry.robot_id,
                    maintenance_type=maintenance_type,
                    priority=priority,
                    action=action,
                    component=component,
                    reason=reason,
                    estimated_rul_hours=telemetry.rul_hours,
                    created_at=utc_iso(),
                )
            )

        if (
            telemetry.health_score
            <= self.config.critical_health_score
        ):
            add(
                maintenance_type=MaintenanceType.PREDICTIVE,
                priority=MaintenancePriority.CRITICAL,
                action=MaintenanceAction.INSPECT,
                component=ComponentType.OTHER,
                reason="Critical robot health score",
            )

        elif (
            telemetry.health_score
            <= self.config.warning_health_score
        ):
            add(
                maintenance_type=MaintenanceType.PREDICTIVE,
                priority=MaintenancePriority.HIGH,
                action=MaintenanceAction.INSPECT,
                component=ComponentType.OTHER,
                reason="Degraded robot health",
            )

        if telemetry.rul_hours is not None:
            if (
                telemetry.rul_hours
                <= self.config.critical_rul_hours
            ):
                add(
                    maintenance_type=MaintenanceType.PREDICTIVE,
                    priority=MaintenancePriority.CRITICAL,
                    action=MaintenanceAction.INSPECT,
                    component=ComponentType.OTHER,
                    reason="Critically low estimated RUL",
                )

            elif (
                telemetry.rul_hours
                <= self.config.warning_rul_hours
            ):
                add(
                    maintenance_type=MaintenanceType.PREDICTIVE,
                    priority=MaintenancePriority.HIGH,
                    action=MaintenanceAction.SERVICE,
                    component=ComponentType.OTHER,
                    reason="Low estimated RUL",
                )

        if (
            telemetry.motor_temperature_c
            >= self.config.motor_critical_temperature_c
        ):
            add(
                maintenance_type=MaintenanceType.CORRECTIVE,
                priority=MaintenancePriority.CRITICAL,
                action=MaintenanceAction.INSPECT,
                component=ComponentType.MOTOR,
                reason="Critical motor temperature",
            )

        elif (
            telemetry.motor_temperature_c
            >= self.config.motor_warning_temperature_c
        ):
            add(
                maintenance_type=MaintenanceType.PREDICTIVE,
                priority=MaintenancePriority.HIGH,
                action=MaintenanceAction.SERVICE,
                component=ComponentType.MOTOR,
                reason="Elevated motor temperature",
            )

        if (
            telemetry.vibration_mm_s
            >= self.config.vibration_critical_mm_s
        ):
            add(
                maintenance_type=MaintenanceType.CORRECTIVE,
                priority=MaintenancePriority.CRITICAL,
                action=MaintenanceAction.INSPECT,
                component=ComponentType.GEARBOX,
                reason="Critical vibration detected",
            )

        elif (
            telemetry.vibration_mm_s
            >= self.config.vibration_warning_mm_s
        ):
            add(
                maintenance_type=MaintenanceType.PREDICTIVE,
                priority=MaintenancePriority.HIGH,
                action=MaintenanceAction.INSPECT,
                component=ComponentType.GEARBOX,
                reason="Abnormal vibration detected",
            )

        if not telemetry.sensor_available:
            add(
                maintenance_type=MaintenanceType.CORRECTIVE,
                priority=MaintenancePriority.HIGH,
                action=MaintenanceAction.CALIBRATE,
                component=ComponentType.SENSOR,
                reason="Sensor unavailable",
            )

        if telemetry.fault_count > 0:
            add(
                maintenance_type=MaintenanceType.CORRECTIVE,
                priority=MaintenancePriority.HIGH,
                action=MaintenanceAction.INSPECT,
                component=ComponentType.OTHER,
                reason=(
                    f"{telemetry.fault_count} active "
                    "or recorded faults"
                ),
            )

        return recommendations


# ============================================================
# PREVENTIVE MAINTENANCE ENGINE
# ============================================================

class PreventiveMaintenanceEngine:

    def __init__(
        self,
        config: MaintenanceServiceConfig,
    ) -> None:
        self.config = config

    def is_due(
        self,
        operating_hours: float,
        last_service_hours: float,
    ) -> bool:
        validate_number(
            operating_hours,
            "operating_hours",
            minimum=0.0,
        )

        validate_number(
            last_service_hours,
            "last_service_hours",
            minimum=0.0,
        )

        if last_service_hours > operating_hours:
            raise MaintenanceValidationError(
                "Last service hours exceed operating hours"
            )

        return (
            operating_hours - last_service_hours
            >= self.config.preventive_interval_hours
        )

    def hours_until_due(
        self,
        operating_hours: float,
        last_service_hours: float,
    ) -> float:
        elapsed = (
            operating_hours - last_service_hours
        )

        if elapsed < 0:
            raise MaintenanceValidationError(
                "Invalid service-hour history"
            )

        return max(
            0.0,
            self.config.preventive_interval_hours
            - elapsed,
        )


# ============================================================
# SPARE PART
# ============================================================

@dataclass
class SparePart:
    part_id: str
    name: str
    component: ComponentType

    unit_price: float

    quantity_available: int

    minimum_stock: int = 1

    supplier: str | None = None

    def __post_init__(self) -> None:
        require_text(self.part_id, "part_id")
        require_text(self.name, "name")

        validate_number(
            self.unit_price,
            "unit_price",
            minimum=0.0,
        )

        if (
            self.quantity_available < 0
            or self.minimum_stock < 0
        ):
            raise MaintenanceValidationError(
                "Stock quantities cannot be negative"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "name": self.name,
            "component": self.component.value,
            "unit_price": self.unit_price,
            "quantity_available": (
                self.quantity_available
            ),
            "minimum_stock": self.minimum_stock,
            "supplier": self.supplier,
        }


# ============================================================
# SPARE PART INVENTORY
# ============================================================

class SparePartInventory:

    def __init__(self) -> None:
        self.parts: dict[str, SparePart] = {}

        self._lock = threading.RLock()

    def add_part(
        self,
        part: SparePart,
    ) -> None:
        with self._lock:
            if part.part_id in self.parts:
                raise MaintenanceValidationError(
                    "Duplicate part ID"
                )

            self.parts[part.part_id] = part

    def get_part(
        self,
        part_id: str,
    ) -> SparePart:
        part = self.parts.get(part_id)

        if part is None:
            raise MaintenanceNotFoundError(
                f"Part not found: {part_id}"
            )

        return part

    def consume(
        self,
        part_id: str,
        quantity: int,
    ) -> float:
        if quantity <= 0:
            raise MaintenanceValidationError(
                "Quantity must be positive"
            )

        with self._lock:
            part = self.get_part(part_id)

            if part.quantity_available < quantity:
                raise MaintenanceStateError(
                    "Insufficient spare part stock"
                )

            part.quantity_available -= quantity

            return (
                part.unit_price * quantity
            )

    def restock(
        self,
        part_id: str,
        quantity: int,
    ) -> None:
        if quantity <= 0:
            raise MaintenanceValidationError(
                "Quantity must be positive"
            )

        with self._lock:
            part = self.get_part(part_id)
            part.quantity_available += quantity

    def low_stock_parts(
        self,
    ) -> list[SparePart]:
        return [
            part
            for part in self.parts.values()
            if (
                part.quantity_available
                <= part.minimum_stock
            )
        ]


# ============================================================
# MAINTENANCE WORK ORDER
# ============================================================

@dataclass
class MaintenanceWorkOrder:
    work_order_id: str

    robot_id: str

    maintenance_type: MaintenanceType
    priority: MaintenancePriority
    status: WorkOrderStatus

    title: str
    description: str

    component: ComponentType
    action: MaintenanceAction

    estimated_labor_hours: float
    estimated_downtime_hours: float

    estimated_parts_cost: float

    assigned_technician: str | None = None

    actual_labor_hours: float = 0.0
    actual_downtime_hours: float = 0.0
    actual_parts_cost: float = 0.0

    created_at: str = field(
        default_factory=utc_iso
    )

    started_at: str | None = None
    completed_at: str | None = None

    notes: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_order_id": self.work_order_id,
            "robot_id": self.robot_id,
            "maintenance_type": (
                self.maintenance_type.value
            ),
            "priority": self.priority.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "component": self.component.value,
            "action": self.action.value,
            "estimated_labor_hours": (
                self.estimated_labor_hours
            ),
            "estimated_downtime_hours": (
                self.estimated_downtime_hours
            ),
            "estimated_parts_cost": (
                self.estimated_parts_cost
            ),
            "assigned_technician": (
                self.assigned_technician
            ),
            "actual_labor_hours": (
                self.actual_labor_hours
            ),
            "actual_downtime_hours": (
                self.actual_downtime_hours
            ),
            "actual_parts_cost": (
                self.actual_parts_cost
            ),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "notes": list(self.notes),
        }


# ============================================================
# COST ESTIMATION
# ============================================================

class MaintenanceCostEstimator:

    def __init__(
        self,
        config: MaintenanceServiceConfig,
    ) -> None:
        self.config = config

    def estimate(
        self,
        *,
        labor_hours: float,
        downtime_hours: float,
        parts_cost: float,
    ) -> dict[str, float | str]:
        for name, value in (
            ("labor_hours", labor_hours),
            ("downtime_hours", downtime_hours),
            ("parts_cost", parts_cost),
        ):
            validate_number(
                value,
                name,
                minimum=0.0,
            )

        labor_cost = (
            labor_hours
            * self.config.labor_rate_per_hour
        )

        downtime_cost = (
            downtime_hours
            * self.config.downtime_loss_per_hour
        )

        total = (
            labor_cost
            + downtime_cost
            + parts_cost
        )

        return {
            "currency": self.config.currency,
            "labor_cost": round(labor_cost, 2),
            "downtime_cost": round(downtime_cost, 2),
            "parts_cost": round(parts_cost, 2),
            "total_cost": round(total, 2),
        }


# ============================================================
# UNIFIED MAINTENANCE SERVICE
# ============================================================

class RoboticsMaintenanceService:

    def __init__(
        self,
        config: MaintenanceServiceConfig | None = None,
    ) -> None:
        self.config = (
            config or MaintenanceServiceConfig()
        )

        self.predictive_engine = (
            PredictiveMaintenanceEngine(
                self.config
            )
        )

        self.preventive_engine = (
            PreventiveMaintenanceEngine(
                self.config
            )
        )

        self.cost_estimator = (
            MaintenanceCostEstimator(
                self.config
            )
        )

        self.inventory = SparePartInventory()

        self.work_orders: dict[
            str,
            MaintenanceWorkOrder
        ] = {}

        self.last_service_hours: dict[
            str,
            float
        ] = {}

        self.telemetry: dict[
            str,
            RobotMaintenanceTelemetry
        ] = {}

        self.history: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.config.maximum_history
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # EVENT HISTORY
    # --------------------------------------------------------

    def _record_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        self.history.append({
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": utc_iso(),
            "payload": dict(payload),
        })

    # --------------------------------------------------------
    # INGEST TELEMETRY
    # --------------------------------------------------------

    def ingest_telemetry(
        self,
        telemetry: RobotMaintenanceTelemetry,
    ) -> list[MaintenanceRecommendation]:
        with self._lock:
            self.telemetry[
                telemetry.robot_id
            ] = telemetry

            recommendations = (
                self.predictive_engine.evaluate(
                    telemetry
                )
            )

            last_service = (
                self.last_service_hours.get(
                    telemetry.robot_id,
                    0.0,
                )
            )

            if self.preventive_engine.is_due(
                telemetry.operating_hours,
                last_service,
            ):
                recommendations.append(
                    MaintenanceRecommendation(
                        recommendation_id=str(uuid.uuid4()),
                        robot_id=telemetry.robot_id,
                        maintenance_type=(
                            MaintenanceType.PREVENTIVE
                        ),
                        priority=(
                            MaintenancePriority.MEDIUM
                        ),
                        action=MaintenanceAction.SERVICE,
                        component=ComponentType.OTHER,
                        reason=(
                            "Scheduled preventive "
                            "maintenance interval reached"
                        ),
                        estimated_rul_hours=(
                            telemetry.rul_hours
                        ),
                        created_at=utc_iso(),
                    )
                )

            self._record_event(
                "maintenance_telemetry_received",
                {
                    "robot_id": telemetry.robot_id,
                    "recommendation_count": (
                        len(recommendations)
                    ),
                },
            )

            return recommendations

    # --------------------------------------------------------
    # CREATE WORK ORDER
    # --------------------------------------------------------

    def create_work_order(
        self,
        *,
        robot_id: str,
        title: str,
        description: str,
        maintenance_type: MaintenanceType,
        priority: MaintenancePriority,
        component: ComponentType,
        action: MaintenanceAction,
        estimated_labor_hours: float = 1.0,
        estimated_downtime_hours: float = 1.0,
        estimated_parts_cost: float = 0.0,
    ) -> MaintenanceWorkOrder:
        require_text(robot_id, "robot_id")
        require_text(title, "title")

        for name, value in (
            ("estimated_labor_hours", estimated_labor_hours),
            ("estimated_downtime_hours", estimated_downtime_hours),
            ("estimated_parts_cost", estimated_parts_cost),
        ):
            validate_number(
                value,
                name,
                minimum=0.0,
            )

        order = MaintenanceWorkOrder(
            work_order_id=str(uuid.uuid4()),
            robot_id=robot_id,
            maintenance_type=MaintenanceType(
                maintenance_type
            ),
            priority=MaintenancePriority(
                priority
            ),
            status=WorkOrderStatus.OPEN,
            title=title,
            description=description,
            component=ComponentType(component),
            action=MaintenanceAction(action),
            estimated_labor_hours=estimated_labor_hours,
            estimated_downtime_hours=estimated_downtime_hours,
            estimated_parts_cost=estimated_parts_cost,
        )

        with self._lock:
            self.work_orders[
                order.work_order_id
            ] = order

            self._record_event(
                "work_order_created",
                {
                    "work_order_id": order.work_order_id,
                    "robot_id": robot_id,
                },
            )

        return order

    # --------------------------------------------------------
    # GET WORK ORDER
    # --------------------------------------------------------

    def get_work_order(
        self,
        work_order_id: str,
    ) -> MaintenanceWorkOrder:
        order = self.work_orders.get(
            work_order_id
        )

        if order is None:
            raise MaintenanceNotFoundError(
                f"Work order not found: {work_order_id}"
            )

        return order

    # --------------------------------------------------------
    # ASSIGN TECHNICIAN
    # --------------------------------------------------------

    def assign_technician(
        self,
        work_order_id: str,
        technician: str,
    ) -> MaintenanceWorkOrder:
        require_text(
            technician,
            "technician",
        )

        with self._lock:
            order = self.get_work_order(
                work_order_id
            )

            if order.status != WorkOrderStatus.OPEN:
                raise MaintenanceStateError(
                    "Only open work orders can be assigned"
                )

            order.assigned_technician = technician
            order.status = WorkOrderStatus.ASSIGNED

            self._record_event(
                "technician_assigned",
                {
                    "work_order_id": work_order_id,
                    "technician": technician,
                },
            )

            return order

    # --------------------------------------------------------
    # START WORK ORDER
    # --------------------------------------------------------

    def start_work_order(
        self,
        work_order_id: str,
    ) -> MaintenanceWorkOrder:
        with self._lock:
            order = self.get_work_order(
                work_order_id
            )

            if order.status != WorkOrderStatus.ASSIGNED:
                raise MaintenanceStateError(
                    "Work order must be assigned first"
                )

            order.status = (
                WorkOrderStatus.IN_PROGRESS
            )

            order.started_at = utc_iso()

            self._record_event(
                "work_order_started",
                {
                    "work_order_id": work_order_id,
                },
            )

            return order

    # --------------------------------------------------------
    # COMPLETE WORK ORDER
    # --------------------------------------------------------

    def complete_work_order(
        self,
        work_order_id: str,
        *,
        actual_labor_hours: float,
        actual_downtime_hours: float,
        actual_parts_cost: float,
        notes: str | None = None,
    ) -> MaintenanceWorkOrder:
        for name, value in (
            ("actual_labor_hours", actual_labor_hours),
            ("actual_downtime_hours", actual_downtime_hours),
            ("actual_parts_cost", actual_parts_cost),
        ):
            validate_number(
                value,
                name,
                minimum=0.0,
            )

        with self._lock:
            order = self.get_work_order(
                work_order_id
            )

            if order.status != WorkOrderStatus.IN_PROGRESS:
                raise MaintenanceStateError(
                    "Work order is not in progress"
                )

            order.actual_labor_hours = actual_labor_hours
            order.actual_downtime_hours = actual_downtime_hours
            order.actual_parts_cost = actual_parts_cost

            order.status = WorkOrderStatus.COMPLETED
            order.completed_at = utc_iso()

            if notes:
                order.notes.append(notes)

            if order.maintenance_type == MaintenanceType.PREVENTIVE:
                telemetry = self.telemetry.get(
                    order.robot_id
                )

                if telemetry is not None:
                    self.last_service_hours[
                        order.robot_id
                    ] = telemetry.operating_hours

            self._record_event(
                "work_order_completed",
                {
                    "work_order_id": work_order_id,
                    "robot_id": order.robot_id,
                },
            )

            return order

    # --------------------------------------------------------
    # CANCEL WORK ORDER
    # --------------------------------------------------------

    def cancel_work_order(
        self,
        work_order_id: str,
    ) -> MaintenanceWorkOrder:
        with self._lock:
            order = self.get_work_order(
                work_order_id
            )

            if order.status in (
                WorkOrderStatus.COMPLETED,
                WorkOrderStatus.CANCELLED,
            ):
                raise MaintenanceStateError(
                    "Work order is already terminal"
                )

            order.status = WorkOrderStatus.CANCELLED

            self._record_event(
                "work_order_cancelled",
                {
                    "work_order_id": work_order_id,
                },
            )

            return order

    # --------------------------------------------------------
    # COST ESTIMATE
    # --------------------------------------------------------

    def estimate_work_order_cost(
        self,
        work_order_id: str,
    ) -> dict[str, float | str]:
        order = self.get_work_order(
            work_order_id
        )

        return self.cost_estimator.estimate(
            labor_hours=order.estimated_labor_hours,
            downtime_hours=order.estimated_downtime_hours,
            parts_cost=order.estimated_parts_cost,
        )

    # --------------------------------------------------------
    # ACTUAL COST
    # --------------------------------------------------------

    def actual_work_order_cost(
        self,
        work_order_id: str,
    ) -> dict[str, float | str]:
        order = self.get_work_order(
            work_order_id
        )

        if order.status != WorkOrderStatus.COMPLETED:
            raise MaintenanceStateError(
                "Actual cost requires completed work order"
            )

        return self.cost_estimator.estimate(
            labor_hours=order.actual_labor_hours,
            downtime_hours=order.actual_downtime_hours,
            parts_cost=order.actual_parts_cost,
        )

    # --------------------------------------------------------
    # LIST WORK ORDERS
    # --------------------------------------------------------

    def list_work_orders(
        self,
        *,
        robot_id: str | None = None,
        status: WorkOrderStatus | None = None,
        priority: MaintenancePriority | None = None,
    ) -> list[MaintenanceWorkOrder]:
        orders = list(
            self.work_orders.values()
        )

        if robot_id is not None:
            orders = [
                order
                for order in orders
                if order.robot_id == robot_id
            ]

        if status is not None:
            orders = [
                order
                for order in orders
                if order.status == status
            ]

        if priority is not None:
            orders = [
                order
                for order in orders
                if order.priority == priority
            ]

        return sorted(
            orders,
            key=lambda order: order.created_at,
            reverse=True,
        )

    # --------------------------------------------------------
    # MAINTENANCE DASHBOARD
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        orders = list(
            self.work_orders.values()
        )

        completed = [
            order
            for order in orders
            if order.status == WorkOrderStatus.COMPLETED
        ]

        total_actual_cost = sum(
            float(
                self.actual_work_order_cost(
                    order.work_order_id
                )["total_cost"]
            )
            for order in completed
        )

        return {
            "total_work_orders": len(orders),
            "completed_work_orders": len(completed),
            "open_work_orders": sum(
                order.status == WorkOrderStatus.OPEN
                for order in orders
            ),
            "in_progress_work_orders": sum(
                order.status == WorkOrderStatus.IN_PROGRESS
                for order in orders
            ),
            "priority_distribution": dict(
                Counter(
                    order.priority.value
                    for order in orders
                )
            ),
            "status_distribution": dict(
                Counter(
                    order.status.value
                    for order in orders
                )
            ),
            "maintenance_type_distribution": dict(
                Counter(
                    order.maintenance_type.value
                    for order in orders
                )
            ),
            "total_actual_cost": round(
                total_actual_cost,
                2,
            ),
            "currency": self.config.currency,
            "tracked_robots": len(
                self.telemetry
            ),
            "inventory_parts": len(
                self.inventory.parts
            ),
            "low_stock_parts": len(
                self.inventory.low_stock_parts()
            ),
        }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "generated_at": utc_iso(),
            "statistics": self.statistics(),
            "work_orders": [
                order.to_dict()
                for order in self.work_orders.values()
            ],
            "inventory": [
                part.to_dict()
                for part in self.inventory.parts.values()
            ],
            "events": list(self.history),
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

    # --------------------------------------------------------
    # EXPORT CSV
    # --------------------------------------------------------

    def export_csv(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "work_order_id",
            "robot_id",
            "maintenance_type",
            "priority",
            "status",
            "title",
            "component",
            "action",
            "estimated_labor_hours",
            "estimated_downtime_hours",
            "estimated_parts_cost",
            "actual_labor_hours",
            "actual_downtime_hours",
            "actual_parts_cost",
            "assigned_technician",
            "created_at",
            "completed_at",
        ]

        with output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for order in self.work_orders.values():
                data = order.to_dict()

                writer.writerow({
                    key: data.get(key)
                    for key in fieldnames
                })

        return output


# ============================================================
# PHYSICS SIMULATION ADAPTER
# ============================================================

class PhysicsMaintenanceAdapter:

    @staticmethod
    def extract(
        physics: Any,
        *,
        health_scores: Mapping[str, float] | None = None,
        rul_hours: Mapping[str, float] | None = None,
        vibration_values: Mapping[str, float] | None = None,
        fault_counts: Mapping[str, int] | None = None,
    ) -> list[RobotMaintenanceTelemetry]:
        health_scores = health_scores or {}
        rul_hours = rul_hours or {}
        vibration_values = vibration_values or {}
        fault_counts = fault_counts or {}

        operating_hours = (
            physics.simulation_time_seconds
            / 3600.0
        )

        return [
            RobotMaintenanceTelemetry(
                robot_id=robot_id,
                operating_hours=operating_hours,
                battery_soc_percent=(
                    robot.battery_soc_percent()
                ),
                motor_temperature_c=(
                    robot.motor_temperature_c
                ),
                vibration_mm_s=(
                    vibration_values.get(
                        robot_id,
                        0.0,
                    )
                ),
                health_score=(
                    health_scores.get(
                        robot_id,
                        100.0,
                    )
                ),
                rul_hours=(
                    rul_hours.get(robot_id)
                ),
                fault_count=(
                    fault_counts.get(
                        robot_id,
                        0,
                    )
                ),
            )
            for robot_id, robot in physics.robots.items()
        ]


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsMaintenanceService()

    service.inventory.add_part(
        SparePart(
            part_id="motor_001",
            name="Robot Drive Motor",
            component=ComponentType.MOTOR,
            unit_price=8500.0,
            quantity_available=5,
            minimum_stock=2,
            supplier="Example Supplier",
        )
    )

    telemetry = RobotMaintenanceTelemetry(
        robot_id="robot_001",
        operating_hours=320.0,
        battery_soc_percent=65.0,
        motor_temperature_c=82.0,
        vibration_mm_s=8.0,
        health_score=58.0,
        rul_hours=72.0,
        fault_count=1,
    )

    recommendations = service.ingest_telemetry(
        telemetry
    )

    order = service.create_work_order(
        robot_id="robot_001",
        title="Inspect robot drive motor",
        description=(
            "High motor temperature and "
            "abnormal vibration detected"
        ),
        maintenance_type=MaintenanceType.PREDICTIVE,
        priority=MaintenancePriority.HIGH,
        component=ComponentType.MOTOR,
        action=MaintenanceAction.INSPECT,
        estimated_labor_hours=3.0,
        estimated_downtime_hours=4.0,
        estimated_parts_cost=1500.0,
    )

    service.assign_technician(
        order.work_order_id,
        "Technician 001",
    )

    service.start_work_order(
        order.work_order_id
    )

    service.complete_work_order(
        order.work_order_id,
        actual_labor_hours=2.5,
        actual_downtime_hours=3.0,
        actual_parts_cost=1200.0,
        notes="Motor inspected and serviced",
    )

    service.export_json(
        "data/maintenance/maintenance_report.json"
    )

    service.export_csv(
        "data/maintenance/work_orders.csv"
    )

    print(
        json.dumps(
            {
                "recommendations": [
                    item.to_dict()
                    for item in recommendations
                ],
                "statistics": service.statistics(),
                "actual_cost": (
                    service.actual_work_order_cost(
                        order.work_order_id
                    )
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

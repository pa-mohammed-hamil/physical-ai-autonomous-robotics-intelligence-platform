"""Parts Catalog module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/parts_catalog.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robot Parts Catalog & Spare Parts Intelligence

Python 3.11+
Standard library only.

Features:
    - Robot parts catalog
    - Part registration and updates
    - Manufacturer and model information
    - Part number and SKU management
    - Serial number tracking
    - Robot model compatibility
    - Replacement and alternative parts
    - Spare-part inventory
    - Stock availability
    - Low-stock and out-of-stock alerts
    - Purchase and replacement cost estimates
    - Supplier information
    - Warranty tracking
    - Maintenance and repair references
    - Part identification integration
    - Inventory reservations
    - Stock movement history
    - Multi-tenant isolation
    - Thread-safe operations
    - JSON import and export
    - SQLite persistence
    - Audit logging

Notes:
    - Monetary values use Decimal.
    - SQLite provides persistent storage.
    - Stock movements are transactional.
    - Compatibility information must be verified
      against manufacturer documentation.
    - This module does not operate physical robots.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


# ============================================================
# EXCEPTIONS
# ============================================================

class PartsCatalogError(Exception):
    pass


class PartValidationError(PartsCatalogError):
    pass


class PartNotFound(PartsCatalogError):
    pass


class PartAlreadyExists(PartsCatalogError):
    pass


class InsufficientStock(PartsCatalogError):
    pass


class ReservationNotFound(PartsCatalogError):
    pass


class CatalogConflict(PartsCatalogError):
    pass


# ============================================================
# ENUMS
# ============================================================

class PartCategory(str, Enum):
    MOTOR = "motor"
    GEARBOX = "gearbox"
    BEARING = "bearing"
    JOINT = "joint"
    ACTUATOR = "actuator"
    SERVO = "servo"
    SENSOR = "sensor"
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    BATTERY = "battery"
    CONTROLLER = "controller"
    CIRCUIT_BOARD = "circuit_board"
    POWER_SUPPLY = "power_supply"
    WHEEL = "wheel"
    GRIPPER = "gripper"
    ROBOT_ARM = "robot_arm"
    CONNECTOR = "connector"
    CABLE = "cable"
    FASTENER = "fastener"
    FRAME = "frame"
    PUMP = "pump"
    VALVE = "valve"
    COOLING = "cooling"
    OTHER = "other"


class PartCondition(str, Enum):
    NEW = "new"
    REFURBISHED = "refurbished"
    USED = "used"
    DAMAGED = "damaged"
    UNKNOWN = "unknown"


class PartStatus(str, Enum):
    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    OBSOLETE = "obsolete"
    UNDER_REVIEW = "under_review"


class StockStatus(str, Enum):
    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"


class MovementType(str, Enum):
    RECEIPT = "receipt"
    ADJUSTMENT_IN = "adjustment_in"
    ADJUSTMENT_OUT = "adjustment_out"
    ISSUE = "issue"
    RETURN = "return"
    RESERVATION = "reservation"
    RESERVATION_RELEASE = "reservation_release"
    RESERVATION_CONSUMPTION = "reservation_consumption"


class ReservationStatus(str, Enum):
    ACTIVE = "active"
    RELEASED = "released"
    CONSUMED = "consumed"


# ============================================================
# UTILITIES
# ============================================================

MONEY_UNIT = Decimal("0.01")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp() -> str:
    return utc_now().isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def validate_id(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}",
            value,
        ) is None
    ):
        raise PartValidationError(
            f"Invalid {name}"
        )

    return value


def validate_text(
    value: str,
    name: str,
    max_length: int = 512,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise PartValidationError(
            f"{name} must be text"
        )

    value = value.strip()

    if (
        len(value) > max_length
        or (not allow_empty and not value)
    ):
        raise PartValidationError(
            f"Invalid {name}"
        )

    return value


def validate_quantity(
    value: int,
    name: str = "quantity",
    *,
    allow_zero: bool = True,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or (not allow_zero and value == 0)
        or value > 1_000_000_000
    ):
        raise PartValidationError(
            f"Invalid {name}"
        )

    return value


def decimal_value(
    value: Any,
    name: str,
) -> Decimal:
    if isinstance(value, bool):
        raise PartValidationError(
            f"Invalid {name}"
        )

    try:
        result = Decimal(str(value))
    except (
        InvalidOperation,
        TypeError,
        ValueError,
    ) as exc:
        raise PartValidationError(
            f"Invalid {name}"
        ) from exc

    if (
        not result.is_finite()
        or result < 0
        or result.adjusted() > 15
    ):
        raise PartValidationError(
            f"Invalid {name}"
        )

    return result


def money(value: Decimal) -> Decimal:
    return value.quantize(
        MONEY_UNIT,
        rounding=ROUND_HALF_UP,
    )


def money_str(value: Decimal) -> str:
    return str(money(value))


def validate_currency(value: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"[A-Z]{3}",
            value,
        ) is None
    ):
        raise PartValidationError(
            "Currency must be a three-letter ISO code"
        )

    return value


def normalize_search(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]",
        "",
        value.casefold(),
    )


def validate_iso_datetime(
    value: str,
    name: str,
) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise PartValidationError(
            f"Invalid {name}"
        ) from exc

    if parsed.tzinfo is None:
        raise PartValidationError(
            f"{name} requires a timezone"
        )

    return value


# ============================================================
# SUPPLIER
# ============================================================

@dataclass(frozen=True)
class PartSupplier:
    supplier_id: str
    supplier_name: str

    website: str = ""
    contact_email: str = ""

    lead_time_days: int = 0

    def __post_init__(self) -> None:
        validate_id(
            self.supplier_id,
            "supplier_id",
        )

        validate_text(
            self.supplier_name,
            "supplier_name",
        )

        validate_text(
            self.website,
            "website",
            1024,
            allow_empty=True,
        )

        validate_text(
            self.contact_email,
            "contact_email",
            320,
            allow_empty=True,
        )

        validate_quantity(
            self.lead_time_days,
            "lead_time_days",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "website": self.website,
            "contact_email": self.contact_email,
            "lead_time_days": self.lead_time_days,
        }


# ============================================================
# PART MODEL
# ============================================================

@dataclass(frozen=True)
class CatalogPart:
    tenant_id: str
    part_id: str

    part_name: str
    category: PartCategory

    manufacturer: str = ""
    model_number: str = ""
    part_number: str = ""
    sku: str = ""

    description: str = ""

    compatible_robot_models: tuple[str, ...] = ()
    replacement_part_ids: tuple[str, ...] = ()

    unit_price: Decimal = Decimal("0")
    currency: str = "INR"

    stock_quantity: int = 0
    reorder_level: int = 2

    condition: PartCondition = PartCondition.NEW
    status: PartStatus = PartStatus.ACTIVE

    supplier_id: str = ""

    warranty_months: int = 0

    safety_critical: bool = False

    maintenance_reference: str = ""
    repair_reference: str = ""

    created_at: str = field(
        default_factory=timestamp
    )

    updated_at: str = field(
        default_factory=timestamp
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.part_id, "part_id")

        validate_text(
            self.part_name,
            "part_name",
            256,
        )

        if not isinstance(
            self.category,
            PartCategory,
        ):
            raise PartValidationError(
                "Invalid part category"
            )

        for name in (
            "manufacturer",
            "model_number",
            "part_number",
            "sku",
            "description",
        ):
            validate_text(
                getattr(self, name),
                name,
                1024,
                allow_empty=True,
            )

        validate_currency(self.currency)

        object.__setattr__(
            self,
            "unit_price",
            money(
                decimal_value(
                    self.unit_price,
                    "unit_price",
                )
            ),
        )

        validate_quantity(
            self.stock_quantity,
            "stock_quantity",
        )

        validate_quantity(
            self.reorder_level,
            "reorder_level",
        )

        validate_quantity(
            self.warranty_months,
            "warranty_months",
        )

        if not isinstance(
            self.condition,
            PartCondition,
        ):
            raise PartValidationError(
                "Invalid part condition"
            )

        if not isinstance(
            self.status,
            PartStatus,
        ):
            raise PartValidationError(
                "Invalid part status"
            )

        if not isinstance(
            self.safety_critical,
            bool,
        ):
            raise PartValidationError(
                "safety_critical must be boolean"
            )

        if self.supplier_id:
            validate_id(
                self.supplier_id,
                "supplier_id",
            )

        for name in (
            "maintenance_reference",
            "repair_reference",
        ):
            value = getattr(self, name)

            if value:
                validate_id(value, name)

        for model in self.compatible_robot_models:
            validate_text(
                model,
                "compatible_robot_model",
                256,
            )

        for replacement_id in self.replacement_part_ids:
            validate_id(
                replacement_id,
                "replacement_part_id",
            )

        validate_iso_datetime(
            self.created_at,
            "created_at",
        )

        validate_iso_datetime(
            self.updated_at,
            "updated_at",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "part_id": self.part_id,
            "part_name": self.part_name,
            "category": self.category.value,
            "manufacturer": self.manufacturer,
            "model_number": self.model_number,
            "part_number": self.part_number,
            "sku": self.sku,
            "description": self.description,
            "compatible_robot_models": list(
                self.compatible_robot_models
            ),
            "replacement_part_ids": list(
                self.replacement_part_ids
            ),
            "unit_price": money_str(
                self.unit_price
            ),
            "currency": self.currency,
            "stock_quantity": self.stock_quantity,
            "reorder_level": self.reorder_level,
            "condition": self.condition.value,
            "status": self.status.value,
            "supplier_id": self.supplier_id,
            "warranty_months": self.warranty_months,
            "safety_critical": self.safety_critical,
            "maintenance_reference": (
                self.maintenance_reference
            ),
            "repair_reference": self.repair_reference,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "CatalogPart":
        payload = dict(data)

        payload["category"] = PartCategory(
            payload["category"]
        )

        payload["condition"] = PartCondition(
            payload.get("condition", "new")
        )

        payload["status"] = PartStatus(
            payload.get("status", "active")
        )

        payload["compatible_robot_models"] = tuple(
            payload.get(
                "compatible_robot_models",
                (),
            )
        )

        payload["replacement_part_ids"] = tuple(
            payload.get(
                "replacement_part_ids",
                (),
            )
        )

        payload["unit_price"] = Decimal(
            str(payload.get("unit_price", "0"))
        )

        return cls(**payload)


# ============================================================
# PART SERIAL NUMBER
# ============================================================

@dataclass(frozen=True)
class PartSerialRecord:
    tenant_id: str
    serial_number: str
    part_id: str

    robot_id: str = ""
    installed_at: str = ""

    notes: str = ""

    def __post_init__(self) -> None:
        validate_id(
            self.tenant_id,
            "tenant_id",
        )

        validate_text(
            self.serial_number,
            "serial_number",
            256,
        )

        validate_id(
            self.part_id,
            "part_id",
        )

        if self.robot_id:
            validate_id(
                self.robot_id,
                "robot_id",
            )

        if self.installed_at:
            validate_iso_datetime(
                self.installed_at,
                "installed_at",
            )

        validate_text(
            self.notes,
            "notes",
            1024,
            allow_empty=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "serial_number": self.serial_number,
            "part_id": self.part_id,
            "robot_id": self.robot_id,
            "installed_at": self.installed_at,
            "notes": self.notes,
        }


# ============================================================
# CATALOG CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PartsCatalogConfig:
    database_path: str = (
        "data/robotics/parts_catalog.sqlite3"
    )

    maximum_search_results: int = 500

    simulation_only: bool = True

    def __post_init__(self) -> None:
        validate_text(
            self.database_path,
            "database_path",
            2048,
        )

        validate_quantity(
            self.maximum_search_results,
            "maximum_search_results",
            allow_zero=False,
        )

        if self.simulation_only is not True:
            raise PartValidationError(
                "Only advisory/simulation mode is supported"
            )


# ============================================================
# PARTS CATALOG ENGINE
# ============================================================

class RoboticsPartsCatalog:

    def __init__(
        self,
        config: PartsCatalogConfig | None = None,
    ) -> None:
        self.config = (
            config or PartsCatalogConfig()
        )

        self._lock = threading.RLock()

        database_path = Path(
            self.config.database_path
        )

        if str(database_path) != ":memory:":
            database_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._connection = sqlite3.connect(
            str(database_path),
            check_same_thread=False,
            isolation_level=None,
            timeout=30,
        )

        self._connection.row_factory = (
            sqlite3.Row
        )

        self._connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        self._connection.execute(
            "PRAGMA busy_timeout = 30000"
        )

        self._initialize_database()

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    def _initialize_database(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS parts (
                    tenant_id TEXT NOT NULL,
                    part_id TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, part_id)
                );

                CREATE TABLE IF NOT EXISTS suppliers (
                    tenant_id TEXT NOT NULL,
                    supplier_id TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, supplier_id)
                );

                CREATE TABLE IF NOT EXISTS serial_numbers (
                    tenant_id TEXT NOT NULL,
                    serial_number TEXT NOT NULL,
                    part_id TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (
                        tenant_id,
                        serial_number
                    ),
                    FOREIGN KEY (
                        tenant_id,
                        part_id
                    )
                    REFERENCES parts (
                        tenant_id,
                        part_id
                    )
                );

                CREATE TABLE IF NOT EXISTS reservations (
                    reservation_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    part_id TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    work_order_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (
                        tenant_id,
                        part_id
                    )
                    REFERENCES parts (
                        tenant_id,
                        part_id
                    )
                );

                CREATE INDEX IF NOT EXISTS
                idx_reservations_part
                ON reservations (
                    tenant_id,
                    part_id,
                    status
                );

                CREATE TABLE IF NOT EXISTS
                stock_movements (
                    movement_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    part_id TEXT NOT NULL,
                    movement_type TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    previous_quantity INTEGER NOT NULL,
                    new_quantity INTEGER NOT NULL,
                    reference TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS
                idx_movements_part
                ON stock_movements (
                    tenant_id,
                    part_id,
                    created_at
                );

                CREATE TABLE IF NOT EXISTS
                audit_events (
                    event_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS
                idx_audit_tenant
                ON audit_events (
                    tenant_id,
                    created_at
                );
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "RoboticsPartsCatalog":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    def _begin(self) -> None:
        self._connection.execute(
            "BEGIN IMMEDIATE"
        )

    def _commit(self) -> None:
        self._connection.execute(
            "COMMIT"
        )

    def _rollback(self) -> None:
        self._connection.execute(
            "ROLLBACK"
        )

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        tenant_id: str,
        entity_type: str,
        entity_id: str,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO audit_events (
                event_id,
                tenant_id,
                entity_type,
                entity_id,
                action,
                details_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id(),
                tenant_id,
                entity_type,
                entity_id,
                action,
                json.dumps(details or {}),
                timestamp(),
            ),
        )

    # --------------------------------------------------------
    # INTERNAL PART LOOKUP
    # --------------------------------------------------------

    def _get_part(
        self,
        tenant_id: str,
        part_id: str,
    ) -> CatalogPart:
        row = self._connection.execute(
            """
            SELECT data_json
            FROM parts
            WHERE tenant_id = ?
              AND part_id = ?
            """,
            (
                tenant_id,
                part_id,
            ),
        ).fetchone()

        if row is None:
            raise PartNotFound(
                f"Part not found: {part_id}"
            )

        return CatalogPart.from_dict(
            json.loads(row["data_json"])
        )

    def _save_part(
        self,
        part: CatalogPart,
    ) -> None:
        cursor = self._connection.execute(
            """
            UPDATE parts
            SET data_json = ?
            WHERE tenant_id = ?
              AND part_id = ?
            """,
            (
                json.dumps(part.to_dict()),
                part.tenant_id,
                part.part_id,
            ),
        )

        if cursor.rowcount != 1:
            raise PartNotFound(
                "Part not found during update"
            )

    # --------------------------------------------------------
    # REGISTER PART
    # --------------------------------------------------------

    def register_part(
        self,
        part: CatalogPart,
    ) -> dict[str, Any]:
        if not isinstance(
            part,
            CatalogPart,
        ):
            raise PartValidationError(
                "Invalid catalog part"
            )

        with self._lock:
            try:
                self._begin()

                self._connection.execute(
                    """
                    INSERT INTO parts (
                        tenant_id,
                        part_id,
                        data_json
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        part.tenant_id,
                        part.part_id,
                        json.dumps(
                            part.to_dict()
                        ),
                    ),
                )

                self._audit(
                    part.tenant_id,
                    "part",
                    part.part_id,
                    "registered",
                    {
                        "part_name": part.part_name,
                        "initial_stock": (
                            part.stock_quantity
                        ),
                    },
                )

                if part.stock_quantity:
                    self._record_movement(
                        part.tenant_id,
                        part.part_id,
                        MovementType.RECEIPT,
                        part.stock_quantity,
                        0,
                        part.stock_quantity,
                        "initial_inventory",
                    )

                self._commit()

            except sqlite3.IntegrityError as exc:
                self._rollback()

                raise PartAlreadyExists(
                    "Part already registered"
                ) from exc

            except Exception:
                self._rollback()
                raise

        return part.to_dict()

    # --------------------------------------------------------
    # GET PART
    # --------------------------------------------------------

    def get_part(
        self,
        *,
        tenant_id: str,
        part_id: str,
    ) -> dict[str, Any]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            part_id,
            "part_id",
        )

        with self._lock:
            return self._get_part(
                tenant_id,
                part_id,
            ).to_dict()

    # --------------------------------------------------------
    # UPDATE PART
    # --------------------------------------------------------

    def update_part(
        self,
        *,
        tenant_id: str,
        part_id: str,
        changes: dict[str, Any],
    ) -> dict[str, Any]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            part_id,
            "part_id",
        )

        allowed_fields = {
            "part_name",
            "category",
            "manufacturer",
            "model_number",
            "part_number",
            "sku",
            "description",
            "compatible_robot_models",
            "replacement_part_ids",
            "unit_price",
            "currency",
            "reorder_level",
            "condition",
            "status",
            "supplier_id",
            "warranty_months",
            "safety_critical",
            "maintenance_reference",
            "repair_reference",
        }

        if (
            not isinstance(changes, dict)
            or not changes
            or not set(changes).issubset(
                allowed_fields
            )
        ):
            raise PartValidationError(
                "Invalid update fields"
            )

        normalized = dict(changes)

        enum_fields = {
            "category": PartCategory,
            "condition": PartCondition,
            "status": PartStatus,
        }

        for name, enum_type in enum_fields.items():
            if (
                name in normalized
                and not isinstance(
                    normalized[name],
                    enum_type,
                )
            ):
                normalized[name] = enum_type(
                    normalized[name]
                )

        for name in (
            "compatible_robot_models",
            "replacement_part_ids",
        ):
            if name in normalized:
                normalized[name] = tuple(
                    normalized[name]
                )

        with self._lock:
            try:
                self._begin()

                part = self._get_part(
                    tenant_id,
                    part_id,
                )

                updated = replace(
                    part,
                    **normalized,
                    updated_at=timestamp(),
                )

                self._save_part(updated)

                self._audit(
                    tenant_id,
                    "part",
                    part_id,
                    "updated",
                    {
                        "fields": list(changes),
                    },
                )

                self._commit()

                return updated.to_dict()

            except Exception:
                self._rollback()
                raise

    # --------------------------------------------------------
    # LIST PARTS
    # --------------------------------------------------------

    def list_parts(
        self,
        *,
        tenant_id: str,
        category: PartCategory | None = None,
        status: PartStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_quantity(
            limit,
            "limit",
            allow_zero=False,
        )

        validate_quantity(
            offset,
            "offset",
        )

        if (
            limit
            > self.config.maximum_search_results
        ):
            raise PartValidationError(
                "Result limit exceeded"
            )

        if (
            category is not None
            and not isinstance(category, PartCategory)
        ):
            raise PartValidationError(
                "Invalid category"
            )

        if (
            status is not None
            and not isinstance(status, PartStatus)
        ):
            raise PartValidationError(
                "Invalid status"
            )

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT data_json
                FROM parts
                WHERE tenant_id = ?
                ORDER BY part_id
                """,
                (tenant_id,),
            ).fetchall()

        parts = [
            CatalogPart.from_dict(
                json.loads(row["data_json"])
            )
            for row in rows
        ]

        if category is not None:
            parts = [
                part
                for part in parts
                if part.category == category
            ]

        if status is not None:
            parts = [
                part
                for part in parts
                if part.status == status
            ]

        return [
            part.to_dict()
            for part in parts[
                offset:offset + limit
            ]
        ]

    # --------------------------------------------------------
    # SEARCH PARTS
    # --------------------------------------------------------

    def search_parts(
        self,
        *,
        tenant_id: str,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        query = validate_text(
            query,
            "query",
            256,
        )

        validate_quantity(
            limit,
            "limit",
            allow_zero=False,
        )

        if (
            limit
            > self.config.maximum_search_results
        ):
            raise PartValidationError(
                "Result limit exceeded"
            )

        needle = normalize_search(query)

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT data_json
                FROM parts
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchall()

        matches = []

        for row in rows:
            part = CatalogPart.from_dict(
                json.loads(row["data_json"])
            )

            fields = (
                part.part_id,
                part.part_name,
                part.part_number,
                part.sku,
                part.manufacturer,
                part.model_number,
                part.description,
                part.category.value,
            )

            if any(
                needle in normalize_search(value)
                for value in fields
            ):
                matches.append(part.to_dict())

        return matches[:limit]

    # --------------------------------------------------------
    # REGISTER SUPPLIER
    # --------------------------------------------------------

    def register_supplier(
        self,
        *,
        tenant_id: str,
        supplier: PartSupplier,
    ) -> dict[str, Any]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        if not isinstance(
            supplier,
            PartSupplier,
        ):
            raise PartValidationError(
                "Invalid supplier"
            )

        with self._lock:
            try:
                self._begin()

                self._connection.execute(
                    """
                    INSERT INTO suppliers (
                        tenant_id,
                        supplier_id,
                        data_json
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        tenant_id,
                        supplier.supplier_id,
                        json.dumps(
                            supplier.to_dict()
                        ),
                    ),
                )

                self._audit(
                    tenant_id,
                    "supplier",
                    supplier.supplier_id,
                    "registered",
                )

                self._commit()

            except sqlite3.IntegrityError as exc:
                self._rollback()

                raise PartAlreadyExists(
                    "Supplier already registered"
                ) from exc

            except Exception:
                self._rollback()
                raise

        return supplier.to_dict()

    # --------------------------------------------------------
    # GET SUPPLIER
    # --------------------------------------------------------

    def get_supplier(
        self,
        *,
        tenant_id: str,
        supplier_id: str,
    ) -> dict[str, Any]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            supplier_id,
            "supplier_id",
        )

        with self._lock:
            row = self._connection.execute(
                """
                SELECT data_json
                FROM suppliers
                WHERE tenant_id = ?
                  AND supplier_id = ?
                """,
                (
                    tenant_id,
                    supplier_id,
                ),
            ).fetchone()

        if row is None:
            raise PartNotFound(
                "Supplier not found"
            )

        return json.loads(
            row["data_json"]
        )

    # --------------------------------------------------------
    # SERIAL NUMBER REGISTRATION
    # --------------------------------------------------------

    def register_serial_number(
        self,
        record: PartSerialRecord,
    ) -> dict[str, Any]:
        if not isinstance(
            record,
            PartSerialRecord,
        ):
            raise PartValidationError(
                "Invalid serial record"
            )

        with self._lock:
            try:
                self._begin()

                self._get_part(
                    record.tenant_id,
                    record.part_id,
                )

                self._connection.execute(
                    """
                    INSERT INTO serial_numbers (
                        tenant_id,
                        serial_number,
                        part_id,
                        data_json
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        record.tenant_id,
                        record.serial_number,
                        record.part_id,
                        json.dumps(
                            record.to_dict()
                        ),
                    ),
                )

                self._audit(
                    record.tenant_id,
                    "serial_number",
                    record.serial_number,
                    "registered",
                    {
                        "part_id": record.part_id,
                    },
                )

                self._commit()

            except sqlite3.IntegrityError as exc:
                self._rollback()

                raise PartAlreadyExists(
                    "Serial number already registered"
                ) from exc

            except Exception:
                self._rollback()
                raise

        return record.to_dict()

    # --------------------------------------------------------
    # LOOKUP SERIAL NUMBER
    # --------------------------------------------------------

    def lookup_serial_number(
        self,
        *,
        tenant_id: str,
        serial_number: str,
    ) -> dict[str, Any]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_text(
            serial_number,
            "serial_number",
            256,
        )

        with self._lock:
            row = self._connection.execute(
                """
                SELECT data_json
                FROM serial_numbers
                WHERE tenant_id = ?
                  AND serial_number = ?
                """,
                (
                    tenant_id,
                    serial_number,
                ),
            ).fetchone()

        if row is None:
            raise PartNotFound(
                "Serial number not found"
            )

        return json.loads(
            row["data_json"]
        )

    # --------------------------------------------------------
    # AVAILABLE STOCK
    # --------------------------------------------------------

    def _reserved_quantity(
        self,
        tenant_id: str,
        part_id: str,
    ) -> int:
        row = self._connection.execute(
            """
            SELECT COALESCE(
                SUM(quantity),
                0
            ) AS reserved
            FROM reservations
            WHERE tenant_id = ?
              AND part_id = ?
              AND status = ?
            """,
            (
                tenant_id,
                part_id,
                ReservationStatus.ACTIVE.value,
            ),
        ).fetchone()

        return int(row["reserved"])

    def stock_availability(
        self,
        *,
        tenant_id: str,
        part_id: str,
    ) -> dict[str, Any]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            part_id,
            "part_id",
        )

        with self._lock:
            part = self._get_part(
                tenant_id,
                part_id,
            )

            reserved = self._reserved_quantity(
                tenant_id,
                part_id,
            )

            available = max(
                0,
                part.stock_quantity - reserved,
            )

            if available == 0:
                status = StockStatus.OUT_OF_STOCK

            elif available <= part.reorder_level:
                status = StockStatus.LOW_STOCK

            else:
                status = StockStatus.IN_STOCK

            return {
                "tenant_id": tenant_id,
                "part_id": part_id,
                "part_name": part.part_name,
                "stock_quantity": (
                    part.stock_quantity
                ),
                "reserved_quantity": reserved,
                "available_quantity": available,
                "reorder_level": part.reorder_level,
                "stock_status": status.value,
            }

    # --------------------------------------------------------
    # STOCK MOVEMENT
    # --------------------------------------------------------

    def _record_movement(
        self,
        tenant_id: str,
        part_id: str,
        movement_type: MovementType,
        quantity: int,
        previous_quantity: int,
        new_quantity: int,
        reference: str,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO stock_movements (
                movement_id,
                tenant_id,
                part_id,
                movement_type,
                quantity,
                previous_quantity,
                new_quantity,
                reference,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id(),
                tenant_id,
                part_id,
                movement_type.value,
                quantity,
                previous_quantity,
                new_quantity,
                reference,
                timestamp(),
            ),
        )

    def adjust_stock(
        self,
        *,
        tenant_id: str,
        part_id: str,
        quantity_change: int,
        reference: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise CatalogConflict(
                "Stock adjustment authorization required"
            )

        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            part_id,
            "part_id",
        )

        reference = validate_text(
            reference,
            "reference",
            512,
        )

        if (
            isinstance(quantity_change, bool)
            or not isinstance(quantity_change, int)
            or quantity_change == 0
            or abs(quantity_change) > 1_000_000_000
        ):
            raise PartValidationError(
                "Invalid quantity change"
            )

        with self._lock:
            try:
                self._begin()

                part = self._get_part(
                    tenant_id,
                    part_id,
                )

                previous = part.stock_quantity

                new_quantity = (
                    previous + quantity_change
                )

                validate_quantity(
                    new_quantity,
                    "new_quantity",
                )

                reserved = self._reserved_quantity(
                    tenant_id,
                    part_id,
                )

                if new_quantity < reserved:
                    raise InsufficientStock(
                        "Adjustment would reduce stock "
                        "below active reservations"
                    )

                updated = replace(
                    part,
                    stock_quantity=new_quantity,
                    updated_at=timestamp(),
                )

                self._save_part(updated)

                movement_type = (
                    MovementType.ADJUSTMENT_IN
                    if quantity_change > 0
                    else MovementType.ADJUSTMENT_OUT
                )

                self._record_movement(
                    tenant_id,
                    part_id,
                    movement_type,
                    abs(quantity_change),
                    previous,
                    new_quantity,
                    reference,
                )

                self._audit(
                    tenant_id,
                    "part",
                    part_id,
                    "stock_adjusted",
                    {
                        "change": quantity_change,
                        "reference": reference,
                    },
                )

                self._commit()

            except Exception:
                self._rollback()
                raise

        return self.stock_availability(
            tenant_id=tenant_id,
            part_id=part_id,
        )

    # --------------------------------------------------------
    # RESERVE PARTS
    # --------------------------------------------------------

    def reserve_parts(
        self,
        *,
        tenant_id: str,
        part_id: str,
        quantity: int,
        work_order_id: str,
    ) -> dict[str, Any]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            part_id,
            "part_id",
        )

        validate_id(
            work_order_id,
            "work_order_id",
        )

        validate_quantity(
            quantity,
            allow_zero=False,
        )

        with self._lock:
            try:
                self._begin()

                part = self._get_part(
                    tenant_id,
                    part_id,
                )

                reserved = self._reserved_quantity(
                    tenant_id,
                    part_id,
                )

                available = (
                    part.stock_quantity - reserved
                )

                if quantity > available:
                    raise InsufficientStock(
                        "Not enough available stock"
                    )

                reservation_id = new_id()
                now = timestamp()

                self._connection.execute(
                    """
                    INSERT INTO reservations (
                        reservation_id,
                        tenant_id,
                        part_id,
                        quantity,
                        status,
                        work_order_id,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        reservation_id,
                        tenant_id,
                        part_id,
                        quantity,
                        ReservationStatus.ACTIVE.value,
                        work_order_id,
                        now,
                        now,
                    ),
                )

                self._record_movement(
                    tenant_id,
                    part_id,
                    MovementType.RESERVATION,
                    quantity,
                    part.stock_quantity,
                    part.stock_quantity,
                    work_order_id,
                )

                self._audit(
                    tenant_id,
                    "reservation",
                    reservation_id,
                    "created",
                    {
                        "part_id": part_id,
                        "quantity": quantity,
                    },
                )

                self._commit()

            except Exception:
                self._rollback()
                raise

        return {
            "reservation_id": reservation_id,
            "tenant_id": tenant_id,
            "part_id": part_id,
            "quantity": quantity,
            "work_order_id": work_order_id,
            "status": ReservationStatus.ACTIVE.value,
        }

    # --------------------------------------------------------
    # RELEASE OR CONSUME RESERVATION
    # --------------------------------------------------------

    def finalize_reservation(
        self,
        *,
        tenant_id: str,
        reservation_id: str,
        consume: bool,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise CatalogConflict(
                "Reservation authorization required"
            )

        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            reservation_id,
            "reservation_id",
        )

        if not isinstance(consume, bool):
            raise PartValidationError(
                "consume must be boolean"
            )

        with self._lock:
            try:
                self._begin()

                row = self._connection.execute(
                    """
                    SELECT *
                    FROM reservations
                    WHERE reservation_id = ?
                      AND tenant_id = ?
                    """,
                    (
                        reservation_id,
                        tenant_id,
                    ),
                ).fetchone()

                if row is None:
                    raise ReservationNotFound(
                        "Reservation not found"
                    )

                if (
                    row["status"]
                    != ReservationStatus.ACTIVE.value
                ):
                    raise CatalogConflict(
                        "Reservation already finalized"
                    )

                part_id = row["part_id"]
                quantity = int(row["quantity"])

                part = self._get_part(
                    tenant_id,
                    part_id,
                )

                previous = part.stock_quantity

                if consume:
                    if quantity > previous:
                        raise InsufficientStock(
                            "Insufficient physical stock"
                        )

                    new_quantity = (
                        previous - quantity
                    )

                    updated = replace(
                        part,
                        stock_quantity=new_quantity,
                        updated_at=timestamp(),
                    )

                    self._save_part(updated)

                    new_status = (
                        ReservationStatus.CONSUMED
                    )

                    movement_type = (
                        MovementType.RESERVATION_CONSUMPTION
                    )

                else:
                    new_quantity = previous

                    new_status = (
                        ReservationStatus.RELEASED
                    )

                    movement_type = (
                        MovementType.RESERVATION_RELEASE
                    )

                self._connection.execute(
                    """
                    UPDATE reservations
                    SET status = ?,
                        updated_at = ?
                    WHERE reservation_id = ?
                      AND tenant_id = ?
                    """,
                    (
                        new_status.value,
                        timestamp(),
                        reservation_id,
                        tenant_id,
                    ),
                )

                self._record_movement(
                    tenant_id,
                    part_id,
                    movement_type,
                    quantity,
                    previous,
                    new_quantity,
                    reservation_id,
                )

                self._audit(
                    tenant_id,
                    "reservation",
                    reservation_id,
                    new_status.value,
                )

                self._commit()

            except Exception:
                self._rollback()
                raise

        return {
            "reservation_id": reservation_id,
            "tenant_id": tenant_id,
            "part_id": part_id,
            "quantity": quantity,
            "status": new_status.value,
            "remaining_stock": new_quantity,
        }

    # --------------------------------------------------------
    # COMPATIBILITY
    # --------------------------------------------------------

    def check_compatibility(
        self,
        *,
        tenant_id: str,
        part_id: str,
        robot_model: str,
    ) -> dict[str, Any]:
        robot_model = validate_text(
            robot_model,
            "robot_model",
            256,
        )

        with self._lock:
            part = self._get_part(
                tenant_id,
                part_id,
            )

        if not part.compatible_robot_models:
            compatible = None

        else:
            target = normalize_search(
                robot_model
            )

            compatible = any(
                normalize_search(model) == target
                for model in part.compatible_robot_models
            )

        return {
            "tenant_id": tenant_id,
            "part_id": part_id,
            "robot_model": robot_model,
            "compatible": compatible,
            "requires_manufacturer_verification": True,
            "source": "catalog_metadata",
        }

    # --------------------------------------------------------
    # REPLACEMENT OPTIONS
    # --------------------------------------------------------

    def replacement_options(
        self,
        *,
        tenant_id: str,
        part_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            part = self._get_part(
                tenant_id,
                part_id,
            )

            alternatives = []

            for replacement_id in (
                part.replacement_part_ids
            ):
                try:
                    replacement = self._get_part(
                        tenant_id,
                        replacement_id,
                    )

                except PartNotFound:
                    continue

                alternatives.append(
                    replacement.to_dict()
                )

        return {
            "part_id": part_id,
            "part_name": part.part_name,
            "alternatives": alternatives,
            "manufacturer_verification_required": True,
        }

    # --------------------------------------------------------
    # REPLACEMENT COST
    # --------------------------------------------------------

    def estimate_replacement_cost(
        self,
        *,
        tenant_id: str,
        part_id: str,
        quantity: int = 1,
        labor_cost: Any = 0,
        shipping_cost: Any = 0,
        additional_cost: Any = 0,
    ) -> dict[str, Any]:
        validate_quantity(
            quantity,
            allow_zero=False,
        )

        labor = decimal_value(
            labor_cost,
            "labor_cost",
        )

        shipping = decimal_value(
            shipping_cost,
            "shipping_cost",
        )

        additional = decimal_value(
            additional_cost,
            "additional_cost",
        )

        with self._lock:
            part = self._get_part(
                tenant_id,
                part_id,
            )

        parts_cost = (
            part.unit_price * quantity
        )

        total = (
            parts_cost
            + labor
            + shipping
            + additional
        )

        return {
            "tenant_id": tenant_id,
            "part_id": part_id,
            "quantity": quantity,
            "currency": part.currency,
            "unit_price": money_str(
                part.unit_price
            ),
            "parts_cost": money_str(
                parts_cost
            ),
            "labor_cost": money_str(labor),
            "shipping_cost": money_str(shipping),
            "additional_cost": money_str(additional),
            "estimated_total": money_str(total),
            "estimate_only": True,
        }

    # --------------------------------------------------------
    # INVENTORY ALERTS
    # --------------------------------------------------------

    def inventory_alerts(
        self,
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT data_json
                FROM parts
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchall()

            alerts = []

            for row in rows:
                part = CatalogPart.from_dict(
                    json.loads(row["data_json"])
                )

                reserved = self._reserved_quantity(
                    tenant_id,
                    part.part_id,
                )

                available = max(
                    0,
                    part.stock_quantity - reserved,
                )

                if available == 0:
                    stock_status = (
                        StockStatus.OUT_OF_STOCK
                    )

                elif available <= part.reorder_level:
                    stock_status = (
                        StockStatus.LOW_STOCK
                    )

                else:
                    continue

                alerts.append(
                    {
                        "part_id": part.part_id,
                        "part_name": part.part_name,
                        "category": part.category.value,
                        "stock_status": (
                            stock_status.value
                        ),
                        "available_quantity": available,
                        "reorder_level": (
                            part.reorder_level
                        ),
                        "suggested_reorder_quantity": max(
                            0,
                            part.reorder_level * 2
                            - available,
                        ),
                        "safety_critical": (
                            part.safety_critical
                        ),
                    }
                )

        alerts.sort(
            key=lambda alert: (
                not alert["safety_critical"],
                alert["stock_status"]
                != StockStatus.OUT_OF_STOCK.value,
                alert["part_id"],
            )
        )

        return alerts

    # --------------------------------------------------------
    # INVENTORY VALUATION
    # --------------------------------------------------------

    def inventory_valuation(
        self,
        *,
        tenant_id: str,
    ) -> dict[str, Any]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT data_json
                FROM parts
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchall()

        totals: dict[str, Decimal] = {}
        total_units = 0

        for row in rows:
            part = CatalogPart.from_dict(
                json.loads(row["data_json"])
            )

            total_units += part.stock_quantity

            totals.setdefault(
                part.currency,
                Decimal("0"),
            )

            totals[part.currency] += (
                part.unit_price
                * part.stock_quantity
            )

        return {
            "tenant_id": tenant_id,
            "part_count": len(rows),
            "total_units": total_units,
            "valuation_by_currency": {
                currency: money_str(value)
                for currency, value in totals.items()
            },
            "valuation_method": (
                "catalog_unit_price_times_quantity"
            ),
        }

    # --------------------------------------------------------
    # STOCK MOVEMENT HISTORY
    # --------------------------------------------------------

    def stock_movement_history(
        self,
        *,
        tenant_id: str,
        part_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            part_id,
            "part_id",
        )

        validate_quantity(
            limit,
            "limit",
            allow_zero=False,
        )

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM stock_movements
                WHERE tenant_id = ?
                  AND part_id = ?
                ORDER BY created_at DESC,
                         movement_id DESC
                LIMIT ?
                """,
                (
                    tenant_id,
                    part_id,
                    limit,
                ),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_quantity(
            limit,
            "limit",
            allow_zero=False,
        )

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM audit_events
                WHERE tenant_id = ?
                ORDER BY created_at DESC,
                         event_id DESC
                LIMIT ?
                """,
                (
                    tenant_id,
                    limit,
                ),
            ).fetchall()

        results = []

        for row in rows:
            item = dict(row)

            item["details"] = json.loads(
                item.pop("details_json")
            )

            results.append(item)

        return results

    # --------------------------------------------------------
    # IDENTIFICATION INTEGRATION
    # --------------------------------------------------------

    def lookup_identified_part(
        self,
        *,
        tenant_id: str,
        identification_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Integrates with part_identification.py.

        Prefer human-verified part IDs.
        Unverified candidate matches are advisory.
        """
        validate_id(
            tenant_id,
            "tenant_id",
        )

        if not isinstance(
            identification_result,
            dict,
        ):
            raise PartValidationError(
                "Invalid identification result"
            )

        if (
            identification_result.get("tenant_id")
            != tenant_id
        ):
            raise PartValidationError(
                "Tenant mismatch"
            )

        verified = (
            identification_result.get(
                "verified_part_id"
            )
        )

        review_status = (
            identification_result.get(
                "review_status"
            )
        )

        if (
            review_status == "approved"
            and verified
        ):
            part_id = verified
            verified_match = True

        else:
            part_id = (
                identification_result.get(
                    "best_part_id"
                )
            )

            verified_match = False

        if not part_id:
            return {
                "identified": False,
                "verified": False,
                "part": None,
            }

        try:
            part = self.get_part(
                tenant_id=tenant_id,
                part_id=part_id,
            )

        except PartNotFound:
            return {
                "identified": False,
                "verified": verified_match,
                "part_id": part_id,
                "part": None,
                "reason": "Part not present in catalog",
            }

        return {
            "identified": True,
            "verified": verified_match,
            "part_id": part_id,
            "part": part,
            "stock": self.stock_availability(
                tenant_id=tenant_id,
                part_id=part_id,
            ),
            "human_review_required": (
                not verified_match
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
        validate_id(
            tenant_id,
            "tenant_id",
        )

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            parts = self._connection.execute(
                """
                SELECT data_json
                FROM parts
                WHERE tenant_id = ?
                ORDER BY part_id
                """,
                (tenant_id,),
            ).fetchall()

            suppliers = self._connection.execute(
                """
                SELECT data_json
                FROM suppliers
                WHERE tenant_id = ?
                ORDER BY supplier_id
                """,
                (tenant_id,),
            ).fetchall()

            serials = self._connection.execute(
                """
                SELECT data_json
                FROM serial_numbers
                WHERE tenant_id = ?
                ORDER BY serial_number
                """,
                (tenant_id,),
            ).fetchall()

            reservations = self._connection.execute(
                """
                SELECT *
                FROM reservations
                WHERE tenant_id = ?
                ORDER BY created_at
                """,
                (tenant_id,),
            ).fetchall()

            movements = self._connection.execute(
                """
                SELECT *
                FROM stock_movements
                WHERE tenant_id = ?
                ORDER BY created_at
                """,
                (tenant_id,),
            ).fetchall()

            payload = {
                "generated_at": timestamp(),
                "tenant_id": tenant_id,
                "parts": [
                    json.loads(row["data_json"])
                    for row in parts
                ],
                "suppliers": [
                    json.loads(row["data_json"])
                    for row in suppliers
                ],
                "serial_numbers": [
                    json.loads(row["data_json"])
                    for row in serials
                ],
                "reservations": [
                    dict(row)
                    for row in reservations
                ],
                "stock_movements": [
                    dict(row)
                    for row in movements
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

    # --------------------------------------------------------
    # IMPORT PARTS JSON
    # --------------------------------------------------------

    def import_parts_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        path = Path(path)

        if not path.is_file():
            raise PartValidationError(
                "Import file not found"
            )

        if path.stat().st_size > 20_000_000:
            raise PartValidationError(
                "Import file too large"
            )

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(payload, dict):
            raise PartValidationError(
                "Invalid import payload"
            )

        if payload.get("tenant_id") != tenant_id:
            raise PartValidationError(
                "Import tenant mismatch"
            )

        raw_parts = payload.get(
            "parts",
            [],
        )

        if not isinstance(raw_parts, list):
            raise PartValidationError(
                "parts must be a list"
            )

        parts = [
            CatalogPart.from_dict(item)
            for item in raw_parts
        ]

        for part in parts:
            if part.tenant_id != tenant_id:
                raise PartValidationError(
                    "Cross-tenant import is prohibited"
                )

        imported = 0
        skipped = 0

        with self._lock:
            try:
                self._begin()

                for part in parts:
                    existing = self._connection.execute(
                        """
                        SELECT 1
                        FROM parts
                        WHERE tenant_id = ?
                          AND part_id = ?
                        """,
                        (
                            tenant_id,
                            part.part_id,
                        ),
                    ).fetchone()

                    if existing is not None:
                        if skip_existing:
                            skipped += 1
                            continue

                        raise PartAlreadyExists(
                            f"Part exists: {part.part_id}"
                        )

                    self._connection.execute(
                        """
                        INSERT INTO parts (
                            tenant_id,
                            part_id,
                            data_json
                        )
                        VALUES (?, ?, ?)
                        """,
                        (
                            tenant_id,
                            part.part_id,
                            json.dumps(
                                part.to_dict()
                            ),
                        ),
                    )

                    if part.stock_quantity:
                        self._record_movement(
                            tenant_id,
                            part.part_id,
                            MovementType.RECEIPT,
                            part.stock_quantity,
                            0,
                            part.stock_quantity,
                            "json_import",
                        )

                    self._audit(
                        tenant_id,
                        "part",
                        part.part_id,
                        "imported",
                    )

                    imported += 1

                self._commit()

            except Exception:
                self._rollback()
                raise

        return {
            "imported": imported,
            "skipped": skipped,
        }


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    config = PartsCatalogConfig(
        database_path=":memory:"
    )

    with RoboticsPartsCatalog(config) as catalog:

        supplier = PartSupplier(
            supplier_id="supplier_001",
            supplier_name="Demo Robotics Components",
            website="https://example.com",
            lead_time_days=7,
        )

        catalog.register_supplier(
            tenant_id="tenant_alpha",
            supplier=supplier,
        )

        motor = CatalogPart(
            tenant_id="tenant_alpha",
            part_id="motor_001",
            part_name="Industrial Robot Joint Motor",
            category=PartCategory.MOTOR,
            manufacturer="Demo Robotics",
            model_number="MTR-100",
            part_number="DR-MTR-100",
            sku="SKU-MOTOR-001",
            description=(
                "Industrial robot arm joint motor"
            ),
            compatible_robot_models=(
                "ROBOT-X100",
                "ROBOT-X200",
            ),
            replacement_part_ids=(
                "motor_002",
            ),
            unit_price=Decimal("8500.00"),
            currency="INR",
            stock_quantity=10,
            reorder_level=3,
            supplier_id="supplier_001",
            warranty_months=24,
            safety_critical=True,
        )

        replacement = CatalogPart(
            tenant_id="tenant_alpha",
            part_id="motor_002",
            part_name="Robot Joint Motor V2",
            category=PartCategory.MOTOR,
            manufacturer="Demo Robotics",
            model_number="MTR-200",
            part_number="DR-MTR-200",
            unit_price=Decimal("9500.00"),
            currency="INR",
            stock_quantity=5,
            reorder_level=2,
            safety_critical=True,
        )

        catalog.register_part(motor)
        catalog.register_part(replacement)

        catalog.register_serial_number(
            PartSerialRecord(
                tenant_id="tenant_alpha",
                serial_number="MTR100-SN-001",
                part_id="motor_001",
                robot_id="robot_001",
                installed_at=timestamp(),
            )
        )

        reservation = catalog.reserve_parts(
            tenant_id="tenant_alpha",
            part_id="motor_001",
            quantity=2,
            work_order_id="repair_001",
        )

        print(
            "\nStock after reservation:"
        )

        print(
            json.dumps(
                catalog.stock_availability(
                    tenant_id="tenant_alpha",
                    part_id="motor_001",
                ),
                indent=2,
            )
        )

        catalog.finalize_reservation(
            tenant_id="tenant_alpha",
            reservation_id=(
                reservation["reservation_id"]
            ),
            consume=True,
            authorized=True,
        )

        print(
            "\nReplacement cost:"
        )

        print(
            json.dumps(
                catalog.estimate_replacement_cost(
                    tenant_id="tenant_alpha",
                    part_id="motor_001",
                    quantity=1,
                    labor_cost="1500",
                    shipping_cost="350",
                ),
                indent=2,
            )
        )

        print(
            "\nCompatibility:"
        )

        print(
            json.dumps(
                catalog.check_compatibility(
                    tenant_id="tenant_alpha",
                    part_id="motor_001",
                    robot_model="ROBOT-X100",
                ),
                indent=2,
            )
        )

        print(
            "\nInventory valuation:"
        )

        print(
            json.dumps(
                catalog.inventory_valuation(
                    tenant_id="tenant_alpha"
                ),
                indent=2,
            )
        )

        report = catalog.export_json(
            "data/robotics/parts_catalog_report.json",
            tenant_id="tenant_alpha",
        )

        print("\nReport:", report)


if __name__ == "__main__":
    main()
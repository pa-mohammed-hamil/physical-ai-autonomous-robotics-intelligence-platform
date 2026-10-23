"""Part module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/db/models/part.py

Physical AI & Autonomous Robotics Intelligence Platform

SQLAlchemy 2.x ORM models for:
- Robot spare parts and components
- Parts catalog and inventory
- Robot-part compatibility
- Part health and failure tracking
- Repair and replacement cost estimates
- Part inspection and image analysis
- Supplier and warranty information
- Inventory transactions
- Part lifecycle event history
- Multi-tenant data isolation

Database: PostgreSQL recommended.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    validates,
)

from app.db.base import Base


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    return str(uuid.uuid4())


def serialize_datetime(
    value: Optional[datetime],
) -> Optional[str]:
    return value.isoformat() if value is not None else None


def validate_money(
    value: Optional[Decimal | float | int | str],
    field_name: str,
) -> Optional[Decimal]:
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")

    try:
        amount = Decimal(str(value))
    except Exception as exc:
        raise ValueError(
            f"{field_name} must be a valid monetary amount"
        ) from exc

    if not amount.is_finite():
        raise ValueError(f"{field_name} must be finite")

    if amount < 0:
        raise ValueError(f"{field_name} cannot be negative")

    return amount


def validate_score(
    value: Optional[float],
    field_name: str,
    maximum: float = 100.0,
) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")

    if not 0 <= number <= maximum:
        raise ValueError(
            f"{field_name} must be between 0 and {maximum}"
        )

    return number


# ============================================================
# Enumerations
# ============================================================


class PartCategory(str, Enum):
    MOTOR = "motor"
    ACTUATOR = "actuator"
    SERVO = "servo"
    GEARBOX = "gearbox"
    BEARING = "bearing"
    JOINT = "joint"
    GRIPPER = "gripper"
    END_EFFECTOR = "end_effector"
    SENSOR = "sensor"
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    ENCODER = "encoder"
    BATTERY = "battery"
    CHARGER = "charger"
    POWER_SUPPLY = "power_supply"
    CONTROLLER = "controller"
    COMPUTE_MODULE = "compute_module"
    GPU_MODULE = "gpu_module"
    MEMORY = "memory"
    STORAGE = "storage"
    COMMUNICATION = "communication"
    NETWORK_MODULE = "network_module"
    CIRCUIT_BOARD = "circuit_board"
    CABLE = "cable"
    CONNECTOR = "connector"
    WHEEL = "wheel"
    TRACK = "track"
    ARM = "arm"
    LEG = "leg"
    FRAME = "frame"
    CHASSIS = "chassis"
    COOLING_SYSTEM = "cooling_system"
    HYDRAULIC = "hydraulic"
    PNEUMATIC = "pneumatic"
    SAFETY_COMPONENT = "safety_component"
    TOOL = "tool"
    CONSUMABLE = "consumable"
    OTHER = "other"


class PartType(str, Enum):
    ORIGINAL = "original"
    OEM = "oem"
    AFTERMARKET = "aftermarket"
    REFURBISHED = "refurbished"
    CUSTOM = "custom"
    PROTOTYPE = "prototype"
    CONSUMABLE = "consumable"
    OTHER = "other"


class PartCondition(str, Enum):
    NEW = "new"
    LIKE_NEW = "like_new"
    GOOD = "good"
    FAIR = "fair"
    WORN = "worn"
    DAMAGED = "damaged"
    FAILED = "failed"
    REFURBISHED = "refurbished"
    UNKNOWN = "unknown"


class PartStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    AVAILABLE = "available"
    RESERVED = "reserved"
    INSTALLED = "installed"
    UNDER_INSPECTION = "under_inspection"
    UNDER_REPAIR = "under_repair"
    DEFECTIVE = "defective"
    DISCONTINUED = "discontinued"
    OBSOLETE = "obsolete"
    RECALLED = "recalled"
    ARCHIVED = "archived"


class PartTransactionType(str, Enum):
    STOCK_IN = "stock_in"
    STOCK_OUT = "stock_out"
    RESERVATION = "reservation"
    RESERVATION_RELEASE = "reservation_release"
    INSTALLATION = "installation"
    REMOVAL = "removal"
    RETURN = "return"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    ADJUSTMENT = "adjustment"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    SCRAP = "scrap"


class PartInspectionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    HUMAN_REVIEW = "human_review"
    VERIFIED = "verified"
    REJECTED = "rejected"


class PartEventType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    STATUS_CHANGED = "status_changed"
    CONDITION_CHANGED = "condition_changed"
    STOCK_ADDED = "stock_added"
    STOCK_REMOVED = "stock_removed"
    STOCK_RESERVED = "stock_reserved"
    STOCK_RELEASED = "stock_released"
    STOCK_TRANSFERRED = "stock_transferred"
    LOW_STOCK = "low_stock"
    INSTALLED = "installed"
    REMOVED = "removed"
    INSPECTION_STARTED = "inspection_started"
    INSPECTION_COMPLETED = "inspection_completed"
    DAMAGE_DETECTED = "damage_detected"
    FAILURE_DETECTED = "failure_detected"
    REPAIR_RECOMMENDED = "repair_recommended"
    REPLACEMENT_RECOMMENDED = "replacement_recommended"
    REPAIRED = "repaired"
    REPLACED = "replaced"
    WARRANTY_UPDATED = "warranty_updated"
    PRICE_UPDATED = "price_updated"
    RECALLED = "recalled"
    DISCONTINUED = "discontinued"
    NOTE_ADDED = "note_added"
    DELETED = "deleted"
    RESTORED = "restored"


# ============================================================
# Part Catalog
# ============================================================


class Part(Base):
    """
    Master catalog of robot parts.

    Stock quantities belong to PartInventory.
    Individual physical installations can be tracked
    by the robot component or maintenance subsystem.
    """

    __tablename__ = "parts"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_part_tenant_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "part_number",
            name="uq_part_tenant_number",
        ),
        Index(
            "ix_part_tenant_category",
            "tenant_id",
            "category",
        ),
        Index(
            "ix_part_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_part_tenant_manufacturer",
            "tenant_id",
            "manufacturer",
        ),
        Index(
            "ix_part_tenant_supplier",
            "tenant_id",
            "supplier_id",
        ),
        CheckConstraint(
            "warranty_months IS NULL OR warranty_months >= 0",
            name="ck_part_warranty_months",
        ),
        CheckConstraint(
            "unit_cost IS NULL OR unit_cost >= 0",
            name="ck_part_unit_cost",
        ),
        CheckConstraint(
            "selling_price IS NULL OR selling_price >= 0",
            name="ck_part_selling_price",
        ),
        CheckConstraint(
            "replacement_cost IS NULL OR replacement_cost >= 0",
            name="ck_part_replacement_cost",
        ),
        CheckConstraint(
            "repair_cost IS NULL OR repair_cost >= 0",
            name="ck_part_repair_cost",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    organization_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    part_number: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    sku: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=PartCategory.OTHER.value,
    )

    manufacturer: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_number: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    part_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=PartType.OEM.value,
    )

    condition: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=PartCondition.NEW.value,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=PartStatus.ACTIVE.value,
    )

    unit: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="piece",
    )

    currency: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
        default="INR",
    )

    unit_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )

    selling_price: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )

    replacement_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )

    repair_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )

    supplier_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    supplier_part_number: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    warranty_months: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    image_uri: Mapped[Optional[str]] = mapped_column(
        String(2048),
        nullable=True,
    )

    datasheet_uri: Mapped[Optional[str]] = mapped_column(
        String(2048),
        nullable=True,
    )

    specifications_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    maintenance_guidelines_json: Mapped[
        dict[str, Any]
    ] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    inventories: Mapped[list["PartInventory"]] = relationship(
        "PartInventory",
        back_populates="part",
        cascade="save-update, merge",
        lazy="selectin",
    )

    transactions: Mapped[list["PartTransaction"]] = relationship(
        "PartTransaction",
        back_populates="part",
        cascade="save-update, merge",
        lazy="select",
    )

    compatibilities: Mapped[list["PartCompatibility"]] = relationship(
        "PartCompatibility",
        back_populates="part",
        cascade="save-update, merge",
        lazy="selectin",
    )

    inspections: Mapped[list["PartInspection"]] = relationship(
        "PartInspection",
        back_populates="part",
        cascade="save-update, merge",
        lazy="select",
    )

    events: Mapped[list["PartEvent"]] = relationship(
        "PartEvent",
        back_populates="part",
        cascade="save-update, merge",
        lazy="selectin",
    )

    @validates(
        "unit_cost",
        "selling_price",
        "replacement_cost",
        "repair_cost",
    )
    def validate_cost(
        self,
        key: str,
        value: Optional[Decimal],
    ) -> Optional[Decimal]:
        return validate_money(value, key)

    @validates("warranty_months")
    def validate_warranty(
        self,
        key: str,
        value: Optional[int],
    ) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("warranty_months must be an integer")

        if value < 0:
            raise ValueError("warranty_months cannot be negative")

        return value

    @property
    def is_available(self) -> bool:
        return (
            not self.is_deleted
            and self.status in {
                PartStatus.ACTIVE.value,
                PartStatus.AVAILABLE.value,
            }
        )

    @property
    def total_stock(self) -> int:
        return sum(
            inventory.quantity_on_hand
            for inventory in self.inventories
        )

    @property
    def available_stock(self) -> int:
        return sum(
            inventory.available_quantity
            for inventory in self.inventories
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "part_number": self.part_number,
            "sku": self.sku,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "manufacturer": self.manufacturer,
            "model_number": self.model_number,
            "part_type": self.part_type,
            "condition": self.condition,
            "status": self.status,
            "unit": self.unit,
            "currency": self.currency,
            "unit_cost": (
                str(self.unit_cost)
                if self.unit_cost is not None
                else None
            ),
            "selling_price": (
                str(self.selling_price)
                if self.selling_price is not None
                else None
            ),
            "replacement_cost": (
                str(self.replacement_cost)
                if self.replacement_cost is not None
                else None
            ),
            "repair_cost": (
                str(self.repair_cost)
                if self.repair_cost is not None
                else None
            ),
            "supplier_id": self.supplier_id,
            "warranty_months": self.warranty_months,
            "image_uri": self.image_uri,
            "datasheet_uri": self.datasheet_uri,
            "specifications": self.specifications_json,
            "metadata": self.metadata_json,
            "is_deleted": self.is_deleted,
            "created_at": serialize_datetime(self.created_at),
            "updated_at": serialize_datetime(self.updated_at),
        }

    def __repr__(self) -> str:
        return (
            f"<Part(id={self.id!r}, "
            f"part_number={self.part_number!r}, "
            f"name={self.name!r})>"
        )


# ============================================================
# Part Inventory
# ============================================================


class PartInventory(Base):
    __tablename__ = "part_inventory"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "part_id"],
            ["parts.tenant_id", "parts.id"],
            name="fk_part_inventory_tenant_part",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_part_inventory_tenant_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "part_id",
            "warehouse_id",
            name="uq_part_inventory_warehouse",
        ),
        CheckConstraint(
            "quantity_on_hand >= 0",
            name="ck_part_inventory_quantity",
        ),
        CheckConstraint(
            "quantity_reserved >= 0",
            name="ck_part_inventory_reserved",
        ),
        CheckConstraint(
            "quantity_reserved <= quantity_on_hand",
            name="ck_part_inventory_reservation_limit",
        ),
        CheckConstraint(
            "reorder_point >= 0",
            name="ck_part_inventory_reorder_point",
        ),
        CheckConstraint(
            "reorder_quantity >= 0",
            name="ck_part_inventory_reorder_quantity",
        ),
        Index(
            "ix_part_inventory_tenant_part",
            "tenant_id",
            "part_id",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    part_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    warehouse_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    quantity_on_hand: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    quantity_reserved: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    reorder_point: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    reorder_quantity: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    location_code: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    part: Mapped["Part"] = relationship(
        "Part",
        back_populates="inventories",
    )

    transactions: Mapped[list["PartTransaction"]] = relationship(
        "PartTransaction",
        back_populates="inventory",
        lazy="select",
    )

    @property
    def available_quantity(self) -> int:
        return self.quantity_on_hand - self.quantity_reserved

    @property
    def needs_reorder(self) -> bool:
        return self.available_quantity <= self.reorder_point

    def reserve(self, quantity: int) -> None:
        if isinstance(quantity, bool) or quantity <= 0:
            raise ValueError("Quantity must be positive")

        if quantity > self.available_quantity:
            raise ValueError("Insufficient available stock")

        self.quantity_reserved += quantity
        self.updated_at = utc_now()

    def release(self, quantity: int) -> None:
        if isinstance(quantity, bool) or quantity <= 0:
            raise ValueError("Quantity must be positive")

        if quantity > self.quantity_reserved:
            raise ValueError("Cannot release more than reserved")

        self.quantity_reserved -= quantity
        self.updated_at = utc_now()

    def add_stock(self, quantity: int) -> None:
        if isinstance(quantity, bool) or quantity <= 0:
            raise ValueError("Quantity must be positive")

        self.quantity_on_hand += quantity
        self.updated_at = utc_now()

    def remove_stock(self, quantity: int) -> None:
        if isinstance(quantity, bool) or quantity <= 0:
            raise ValueError("Quantity must be positive")

        if quantity > self.available_quantity:
            raise ValueError("Insufficient available stock")

        self.quantity_on_hand -= quantity
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "part_id": self.part_id,
            "warehouse_id": self.warehouse_id,
            "quantity_on_hand": self.quantity_on_hand,
            "quantity_reserved": self.quantity_reserved,
            "available_quantity": self.available_quantity,
            "reorder_point": self.reorder_point,
            "reorder_quantity": self.reorder_quantity,
            "needs_reorder": self.needs_reorder,
            "location_code": self.location_code,
            "metadata": self.metadata_json,
            "updated_at": serialize_datetime(self.updated_at),
        }


# ============================================================
# Part Inventory Transaction
# ============================================================


class PartTransaction(Base):
    __tablename__ = "part_transactions"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "part_id"],
            ["parts.tenant_id", "parts.id"],
            name="fk_part_transaction_tenant_part",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "inventory_id"],
            ["part_inventory.tenant_id", "part_inventory.id"],
            name="fk_part_transaction_tenant_inventory",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_part_transaction_tenant_part",
            "tenant_id",
            "part_id",
        ),
        Index(
            "ix_part_transaction_tenant_created",
            "tenant_id",
            "created_at",
        ),
        CheckConstraint(
            "quantity > 0",
            name="ck_part_transaction_quantity",
        ),
        CheckConstraint(
            "unit_cost IS NULL OR unit_cost >= 0",
            name="ck_part_transaction_unit_cost",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    part_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    inventory_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
    )

    transaction_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    quantity: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    unit_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(14, 2),
        nullable=True,
    )

    reference_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    reference_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    actor_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    part: Mapped["Part"] = relationship(
        "Part",
        back_populates="transactions",
    )

    inventory: Mapped[Optional["PartInventory"]] = relationship(
        "PartInventory",
        back_populates="transactions",
    )

    @validates("unit_cost")
    def validate_unit_cost(
        self,
        key: str,
        value: Optional[Decimal],
    ) -> Optional[Decimal]:
        return validate_money(value, key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "part_id": self.part_id,
            "inventory_id": self.inventory_id,
            "transaction_type": self.transaction_type,
            "quantity": self.quantity,
            "unit_cost": (
                str(self.unit_cost)
                if self.unit_cost is not None
                else None
            ),
            "reference_type": self.reference_type,
            "reference_id": self.reference_id,
            "actor_id": self.actor_id,
            "notes": self.notes,
            "metadata": self.metadata_json,
            "created_at": serialize_datetime(self.created_at),
        }


# ============================================================
# Robot Part Compatibility
# ============================================================


class PartCompatibility(Base):
    __tablename__ = "part_compatibilities"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "part_id"],
            ["parts.tenant_id", "parts.id"],
            name="fk_part_compatibility_tenant_part",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_part_compatibility_tenant_part",
            "tenant_id",
            "part_id",
        ),
        Index(
            "ix_part_compatibility_robot_model",
            "tenant_id",
            "robot_model",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    part_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    robot_model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    robot_manufacturer: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    component_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    minimum_firmware_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    maximum_firmware_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    verified_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    part: Mapped["Part"] = relationship(
        "Part",
        back_populates="compatibilities",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "part_id": self.part_id,
            "robot_model": self.robot_model,
            "robot_manufacturer": self.robot_manufacturer,
            "component_type": self.component_type,
            "minimum_firmware_version": (
                self.minimum_firmware_version
            ),
            "maximum_firmware_version": (
                self.maximum_firmware_version
            ),
            "is_verified": self.is_verified,
            "verified_by": self.verified_by,
            "metadata": self.metadata_json,
            "created_at": serialize_datetime(self.created_at),
        }


# ============================================================
# Part Inspection and AI Damage Analysis
# ============================================================


class PartInspection(Base):
    """
    Inspection results from uploaded part images,
    sensor measurements or manual inspection.

    AI-generated diagnoses and prices are estimates,
    not verified repair or replacement decisions.
    """

    __tablename__ = "part_inspections"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "part_id"],
            ["parts.tenant_id", "parts.id"],
            name="fk_part_inspection_tenant_part",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_part_inspection_tenant_part",
            "tenant_id",
            "part_id",
        ),
        Index(
            "ix_part_inspection_tenant_status",
            "tenant_id",
            "status",
        ),
        CheckConstraint(
            "health_score IS NULL OR "
            "(health_score >= 0 AND health_score <= 100)",
            name="ck_part_inspection_health",
        ),
        CheckConstraint(
            "confidence_score IS NULL OR "
            "(confidence_score >= 0 AND confidence_score <= 1)",
            name="ck_part_inspection_confidence",
        ),
        CheckConstraint(
            "estimated_repair_cost IS NULL OR "
            "estimated_repair_cost >= 0",
            name="ck_part_inspection_repair_cost",
        ),
        CheckConstraint(
            "estimated_replacement_cost IS NULL OR "
            "estimated_replacement_cost >= 0",
            name="ck_part_inspection_replacement_cost",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    part_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    robot_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    inspection_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="visual",
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=PartInspectionStatus.PENDING.value,
    )

    image_uri: Mapped[Optional[str]] = mapped_column(
        String(2048),
        nullable=True,
    )

    model_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    detected_issue: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    damage_description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    root_cause: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    recommended_action: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    health_score: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 2),
        nullable=True,
    )

    confidence_score: Mapped[Optional[float]] = mapped_column(
        Numeric(5, 4),
        nullable=True,
    )

    estimated_repair_cost: Mapped[Optional[Decimal]] = (
        mapped_column(
            Numeric(14, 2),
            nullable=True,
        )
    )

    estimated_replacement_cost: Mapped[Optional[Decimal]] = (
        mapped_column(
            Numeric(14, 2),
            nullable=True,
        )
    )

    currency: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
        default="INR",
    )

    requires_human_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    reviewed_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    ai_analysis_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    damage_detection_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    inspected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    part: Mapped["Part"] = relationship(
        "Part",
        back_populates="inspections",
    )

    @validates("health_score")
    def validate_health(
        self,
        key: str,
        value: Optional[float],
    ) -> Optional[float]:
        return validate_score(value, key, 100.0)

    @validates("confidence_score")
    def validate_confidence(
        self,
        key: str,
        value: Optional[float],
    ) -> Optional[float]:
        return validate_score(value, key, 1.0)

    @validates(
        "estimated_repair_cost",
        "estimated_replacement_cost",
    )
    def validate_inspection_cost(
        self,
        key: str,
        value: Optional[Decimal],
    ) -> Optional[Decimal]:
        return validate_money(value, key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "part_id": self.part_id,
            "robot_id": self.robot_id,
            "inspection_type": self.inspection_type,
            "status": self.status,
            "image_uri": self.image_uri,
            "model_id": self.model_id,
            "detected_issue": self.detected_issue,
            "damage_description": self.damage_description,
            "root_cause": self.root_cause,
            "recommended_action": self.recommended_action,
            "health_score": (
                float(self.health_score)
                if self.health_score is not None
                else None
            ),
            "confidence_score": (
                float(self.confidence_score)
                if self.confidence_score is not None
                else None
            ),
            "estimated_repair_cost": (
                str(self.estimated_repair_cost)
                if self.estimated_repair_cost is not None
                else None
            ),
            "estimated_replacement_cost": (
                str(self.estimated_replacement_cost)
                if self.estimated_replacement_cost is not None
                else None
            ),
            "currency": self.currency,
            "requires_human_review": self.requires_human_review,
            "reviewed_by": self.reviewed_by,
            "ai_analysis": self.ai_analysis_json,
            "damage_detection": self.damage_detection_json,
            "inspected_at": serialize_datetime(self.inspected_at),
            "created_at": serialize_datetime(self.created_at),
        }


# ============================================================
# Part Event History
# ============================================================


class PartEvent(Base):
    __tablename__ = "part_events"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "part_id"],
            ["parts.tenant_id", "parts.id"],
            name="fk_part_event_tenant_part",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_part_event_tenant_part",
            "tenant_id",
            "part_id",
        ),
        Index(
            "ix_part_event_tenant_created",
            "tenant_id",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    part_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    actor_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    part: Mapped["Part"] = relationship(
        "Part",
        back_populates="events",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "part_id": self.part_id,
            "event_type": self.event_type,
            "message": self.message,
            "actor_id": self.actor_id,
            "metadata": self.metadata_json,
            "created_at": serialize_datetime(self.created_at),
        }


__all__ = [
    "Part",
    "PartInventory",
    "PartTransaction",
    "PartCompatibility",
    "PartInspection",
    "PartEvent",
    "PartCategory",
    "PartType",
    "PartCondition",
    "PartStatus",
    "PartTransactionType",
    "PartInspectionStatus",
    "PartEventType",
    "utc_now",
    "generate_uuid",
    "serialize_datetime",
    "validate_money",
    "validate_score",
]
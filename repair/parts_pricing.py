"""Parts Pricing module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/parts_pricing.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robot Spare Parts Pricing & Cost Intelligence Engine

Python 3.11+
Dependencies: Standard library only

Features:
    - Spare-part price management
    - Manufacturer and supplier pricing
    - Multi-supplier price comparison
    - Currency-aware pricing
    - Quantity-based pricing tiers
    - Bulk purchase discounts
    - Tax and shipping calculations
    - Labor and installation costs
    - Emergency procurement costs
    - Price history tracking
    - Price change detection
    - Price trend analysis
    - Price volatility calculation
    - Budget comparison
    - Repair and replacement estimates
    - Supplier lead-time comparison
    - Quote expiration validation
    - Stock availability checks
    - Lowest-cost supplier recommendations
    - Multi-tenant isolation
    - Thread-safe operations
    - Audit logging
    - JSON report export

Integrations:
    - parts_catalog.py
    - part_identification.py
    - cost_estimation.py
    - labor_cost.py
    - repair_service.py
    - maintenance_service.py

Important:
    Prices must be supplied from verified catalogs,
    supplier quotations, or authorized integrations.

    This engine does not retrieve live market prices.
    Currency conversion requires an explicitly supplied
    exchange rate.

    Estimates are advisory. No automatic purchasing
    or physical robot actuation is performed.
"""

from __future__ import annotations

import json
import math
import statistics
import threading
import uuid

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_HALF_UP,
)
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class PartsPricingError(Exception):
    pass


class PricingValidationError(PartsPricingError):
    pass


class PartPriceNotFound(PartsPricingError):
    pass


class SupplierPriceNotFound(PartsPricingError):
    pass


class PricingConflict(PartsPricingError):
    pass


class CurrencyMismatch(PartsPricingError):
    pass


class QuoteExpired(PartsPricingError):
    pass


class PricingPermissionDenied(PartsPricingError):
    pass


# ============================================================
# ENUMS
# ============================================================

class PriceType(str, Enum):
    CATALOG = "catalog"
    MANUFACTURER = "manufacturer"
    SUPPLIER_QUOTE = "supplier_quote"
    CONTRACT = "contract"
    ESTIMATED = "estimated"


class PartCondition(str, Enum):
    NEW = "new"
    REFURBISHED = "refurbished"
    USED = "used"
    UNKNOWN = "unknown"


class SupplierStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNDER_REVIEW = "under_review"


class PriceTrend(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"


class QuoteStatus(str, Enum):
    VALID = "valid"
    EXPIRED = "expired"
    UNAVAILABLE = "unavailable"


class ProcurementPriority(str, Enum):
    STANDARD = "standard"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class RecommendationType(str, Enum):
    LOWEST_COST = "lowest_cost"
    FASTEST_DELIVERY = "fastest_delivery"
    BEST_VALUE = "best_value"
    NO_AVAILABLE_OFFER = "no_available_offer"


# ============================================================
# UTILITIES
# ============================================================

MONEY_QUANTUM = Decimal("0.01")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


def validate_id(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or not all(
            character.isalnum()
            or character in "._-"
            for character in value
        )
    ):
        raise PricingValidationError(
            f"Invalid {name}"
        )

    return value


def validate_text(
    value: str,
    name: str,
    maximum_length: int = 512,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise PricingValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if (
        len(value) > maximum_length
        or (not allow_empty and not value)
    ):
        raise PricingValidationError(
            f"Invalid {name}"
        )

    return value


def validate_currency(
    currency: str,
) -> str:
    if (
        not isinstance(currency, str)
        or len(currency) != 3
        or not currency.isascii()
        or not currency.isalpha()
        or currency != currency.upper()
    ):
        raise PricingValidationError(
            "Currency must be a three-letter uppercase code"
        )

    return currency


def decimal_number(
    value: Any,
    name: str,
    *,
    allow_negative: bool = False,
) -> Decimal:
    if isinstance(value, bool):
        raise PricingValidationError(
            f"Invalid {name}"
        )

    try:
        result = Decimal(str(value))
    except (
        InvalidOperation,
        TypeError,
        ValueError,
    ) as exc:
        raise PricingValidationError(
            f"Invalid {name}"
        ) from exc

    if not result.is_finite():
        raise PricingValidationError(
            f"{name} must be finite"
        )

    if (
        not allow_negative
        and result < 0
    ):
        raise PricingValidationError(
            f"{name} cannot be negative"
        )

    if result.copy_abs() > Decimal("100000000000000"):
        raise PricingValidationError(
            f"{name} exceeds the supported range"
        )

    return result


def monetary(
    value: Decimal,
) -> Decimal:
    return value.quantize(
        MONEY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def monetary_string(
    value: Decimal,
) -> str:
    return str(monetary(value))


def validate_quantity(
    value: int,
    name: str = "quantity",
    *,
    allow_zero: bool = False,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or (not allow_zero and value == 0)
        or value > 1_000_000_000
    ):
        raise PricingValidationError(
            f"Invalid {name}"
        )

    return value


def validate_datetime(
    value: datetime,
    name: str,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise PricingValidationError(
            f"{name} must be timezone-aware"
        )

    return value.astimezone(
        timezone.utc
    )


def percentage(
    value: Any,
    name: str,
) -> Decimal:
    result = decimal_number(
        value,
        name,
    )

    if result > 100:
        raise PricingValidationError(
            f"{name} cannot exceed 100%"
        )

    return result


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PartsPricingConfig:
    maximum_parts: int = 100000
    maximum_suppliers: int = 10000
    maximum_offers: int = 500000
    maximum_history_events: int = 100000
    maximum_audit_events: int = 20000

    default_currency: str = "INR"

    stable_trend_threshold_percent: Decimal = (
        Decimal("2")
    )

    emergency_surcharge_percent: Decimal = (
        Decimal("10")
    )

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_parts",
            "maximum_suppliers",
            "maximum_offers",
            "maximum_history_events",
            "maximum_audit_events",
        ):
            validate_quantity(
                getattr(self, name),
                name,
            )

        validate_currency(
            self.default_currency
        )

        percentage(
            self.stable_trend_threshold_percent,
            "stable_trend_threshold_percent",
        )

        percentage(
            self.emergency_surcharge_percent,
            "emergency_surcharge_percent",
        )

        if self.simulation_only is not True:
            raise PricingValidationError(
                "Only advisory/simulation mode is supported"
            )


# ============================================================
# PART DETAILS
# ============================================================

@dataclass(frozen=True)
class PricingPart:
    tenant_id: str
    part_id: str
    part_name: str

    manufacturer: str = ""
    model_number: str = ""
    part_number: str = ""
    category: str = "other"

    safety_critical: bool = False

    def __post_init__(self) -> None:
        validate_id(
            self.tenant_id,
            "tenant_id",
        )

        validate_id(
            self.part_id,
            "part_id",
        )

        validate_text(
            self.part_name,
            "part_name",
            256,
        )

        for name in (
            "manufacturer",
            "model_number",
            "part_number",
        ):
            validate_text(
                getattr(self, name),
                name,
                256,
                allow_empty=True,
            )

        validate_text(
            self.category,
            "category",
            128,
        )

        if not isinstance(
            self.safety_critical,
            bool,
        ):
            raise PricingValidationError(
                "safety_critical must be boolean"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "part_id": self.part_id,
            "part_name": self.part_name,
            "manufacturer": self.manufacturer,
            "model_number": self.model_number,
            "part_number": self.part_number,
            "category": self.category,
            "safety_critical": self.safety_critical,
        }


# ============================================================
# SUPPLIER
# ============================================================

@dataclass(frozen=True)
class PricingSupplier:
    tenant_id: str
    supplier_id: str
    supplier_name: str

    country: str = ""
    website: str = ""

    status: SupplierStatus = SupplierStatus.ACTIVE

    reliability_score: Decimal = Decimal("0.80")

    def __post_init__(self) -> None:
        validate_id(
            self.tenant_id,
            "tenant_id",
        )

        validate_id(
            self.supplier_id,
            "supplier_id",
        )

        validate_text(
            self.supplier_name,
            "supplier_name",
            256,
        )

        validate_text(
            self.country,
            "country",
            128,
            allow_empty=True,
        )

        validate_text(
            self.website,
            "website",
            1024,
            allow_empty=True,
        )

        if not isinstance(
            self.status,
            SupplierStatus,
        ):
            raise PricingValidationError(
                "Invalid supplier status"
            )

        score = decimal_number(
            self.reliability_score,
            "reliability_score",
        )

        if score > 1:
            raise PricingValidationError(
                "Reliability score must be between 0 and 1"
            )

        object.__setattr__(
            self,
            "reliability_score",
            score,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "country": self.country,
            "website": self.website,
            "status": self.status.value,
            "reliability_score": str(
                self.reliability_score
            ),
        }


# ============================================================
# QUANTITY DISCOUNT
# ============================================================

@dataclass(frozen=True)
class QuantityPriceTier:
    minimum_quantity: int
    discount_percent: Decimal

    def __post_init__(self) -> None:
        validate_quantity(
            self.minimum_quantity,
            "minimum_quantity",
        )

        object.__setattr__(
            self,
            "discount_percent",
            percentage(
                self.discount_percent,
                "discount_percent",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_quantity": (
                self.minimum_quantity
            ),
            "discount_percent": str(
                self.discount_percent
            ),
        }


# ============================================================
# SUPPLIER OFFER
# ============================================================

@dataclass(frozen=True)
class PartPriceOffer:
    tenant_id: str
    offer_id: str

    part_id: str
    supplier_id: str

    unit_price: Decimal
    currency: str

    price_type: PriceType = (
        PriceType.SUPPLIER_QUOTE
    )

    condition: PartCondition = PartCondition.NEW

    available_quantity: int = 0
    minimum_order_quantity: int = 1

    lead_time_days: int = 0

    shipping_cost: Decimal = Decimal("0")
    handling_cost: Decimal = Decimal("0")

    tax_percent: Decimal = Decimal("0")

    quantity_tiers: tuple[
        QuantityPriceTier, ...
    ] = ()

    valid_until: datetime | None = None

    reference: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "offer_id",
            "part_id",
            "supplier_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        validate_currency(
            self.currency
        )

        for name in (
            "unit_price",
            "shipping_cost",
            "handling_cost",
        ):
            object.__setattr__(
                self,
                name,
                monetary(
                    decimal_number(
                        getattr(self, name),
                        name,
                    )
                ),
            )

        object.__setattr__(
            self,
            "tax_percent",
            percentage(
                self.tax_percent,
                "tax_percent",
            ),
        )

        for name in (
            "available_quantity",
            "lead_time_days",
        ):
            validate_quantity(
                getattr(self, name),
                name,
                allow_zero=True,
            )

        validate_quantity(
            self.minimum_order_quantity,
            "minimum_order_quantity",
        )

        if not isinstance(
            self.price_type,
            PriceType,
        ):
            raise PricingValidationError(
                "Invalid price type"
            )

        if not isinstance(
            self.condition,
            PartCondition,
        ):
            raise PricingValidationError(
                "Invalid part condition"
            )

        seen_quantities = set()

        for tier in self.quantity_tiers:
            if not isinstance(
                tier,
                QuantityPriceTier,
            ):
                raise PricingValidationError(
                    "Invalid quantity tier"
                )

            if tier.minimum_quantity in seen_quantities:
                raise PricingValidationError(
                    "Duplicate quantity tier"
                )

            seen_quantities.add(
                tier.minimum_quantity
            )

        validate_text(
            self.reference,
            "reference",
            1024,
            allow_empty=True,
        )

        validate_datetime(
            self.created_at,
            "created_at",
        )

        validate_datetime(
            self.updated_at,
            "updated_at",
        )

        if self.valid_until is not None:
            validate_datetime(
                self.valid_until,
                "valid_until",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "offer_id": self.offer_id,
            "part_id": self.part_id,
            "supplier_id": self.supplier_id,
            "unit_price": monetary_string(
                self.unit_price
            ),
            "currency": self.currency,
            "price_type": self.price_type.value,
            "condition": self.condition.value,
            "available_quantity": (
                self.available_quantity
            ),
            "minimum_order_quantity": (
                self.minimum_order_quantity
            ),
            "lead_time_days": (
                self.lead_time_days
            ),
            "shipping_cost": monetary_string(
                self.shipping_cost
            ),
            "handling_cost": monetary_string(
                self.handling_cost
            ),
            "tax_percent": str(
                self.tax_percent
            ),
            "quantity_tiers": [
                tier.to_dict()
                for tier in self.quantity_tiers
            ],
            "valid_until": (
                self.valid_until.isoformat()
                if self.valid_until
                else None
            ),
            "reference": self.reference,
            "created_at": (
                self.created_at.isoformat()
            ),
            "updated_at": (
                self.updated_at.isoformat()
            ),
        }


# ============================================================
# PRICE HISTORY
# ============================================================

@dataclass(frozen=True)
class PriceHistoryEvent:
    event_id: str

    tenant_id: str
    offer_id: str
    part_id: str
    supplier_id: str

    previous_price: Decimal | None
    new_price: Decimal

    currency: str

    recorded_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "offer_id": self.offer_id,
            "part_id": self.part_id,
            "supplier_id": self.supplier_id,
            "previous_price": (
                monetary_string(
                    self.previous_price
                )
                if self.previous_price is not None
                else None
            ),
            "new_price": monetary_string(
                self.new_price
            ),
            "currency": self.currency,
            "recorded_at": (
                self.recorded_at.isoformat()
            ),
        }


# ============================================================
# PRICE ESTIMATE REQUEST
# ============================================================

@dataclass(frozen=True)
class PartPricingRequest:
    tenant_id: str
    part_id: str

    quantity: int = 1

    currency: str = "INR"

    priority: ProcurementPriority = (
        ProcurementPriority.STANDARD
    )

    installation_labor_cost: Decimal = (
        Decimal("0")
    )

    additional_cost: Decimal = (
        Decimal("0")
    )

    budget: Decimal | None = None

    maximum_lead_time_days: int | None = None

    allow_refurbished: bool = False
    allow_used: bool = False

    def __post_init__(self) -> None:
        validate_id(
            self.tenant_id,
            "tenant_id",
        )

        validate_id(
            self.part_id,
            "part_id",
        )

        validate_quantity(
            self.quantity
        )

        validate_currency(
            self.currency
        )

        if not isinstance(
            self.priority,
            ProcurementPriority,
        ):
            raise PricingValidationError(
                "Invalid procurement priority"
            )

        for name in (
            "installation_labor_cost",
            "additional_cost",
        ):
            object.__setattr__(
                self,
                name,
                monetary(
                    decimal_number(
                        getattr(self, name),
                        name,
                    )
                ),
            )

        if self.budget is not None:
            object.__setattr__(
                self,
                "budget",
                monetary(
                    decimal_number(
                        self.budget,
                        "budget",
                    )
                ),
            )

        if self.maximum_lead_time_days is not None:
            validate_quantity(
                self.maximum_lead_time_days,
                "maximum_lead_time_days",
                allow_zero=True,
            )

        if not isinstance(
            self.allow_refurbished,
            bool,
        ):
            raise PricingValidationError(
                "allow_refurbished must be boolean"
            )

        if not isinstance(
            self.allow_used,
            bool,
        ):
            raise PricingValidationError(
                "allow_used must be boolean"
            )


# ============================================================
# PRICE BREAKDOWN
# ============================================================

@dataclass(frozen=True)
class PartPriceBreakdown:
    offer_id: str
    supplier_id: str
    supplier_name: str

    part_id: str
    quantity: int

    currency: str

    base_unit_price: Decimal
    discount_percent: Decimal
    discounted_unit_price: Decimal

    parts_subtotal: Decimal
    shipping_cost: Decimal
    handling_cost: Decimal

    emergency_surcharge: Decimal
    tax_amount: Decimal

    installation_labor_cost: Decimal
    additional_cost: Decimal

    total_cost: Decimal

    lead_time_days: int
    supplier_reliability_score: Decimal

    within_budget: bool | None

    def to_dict(self) -> dict[str, Any]:
        monetary_fields = (
            "base_unit_price",
            "discounted_unit_price",
            "parts_subtotal",
            "shipping_cost",
            "handling_cost",
            "emergency_surcharge",
            "tax_amount",
            "installation_labor_cost",
            "additional_cost",
            "total_cost",
        )

        result = {
            "offer_id": self.offer_id,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "part_id": self.part_id,
            "quantity": self.quantity,
            "currency": self.currency,
            "discount_percent": str(
                self.discount_percent
            ),
            "lead_time_days": self.lead_time_days,
            "supplier_reliability_score": str(
                self.supplier_reliability_score
            ),
            "within_budget": self.within_budget,
        }

        for name in monetary_fields:
            result[name] = monetary_string(
                getattr(self, name)
            )

        return result


# ============================================================
# PRICING ENGINE
# ============================================================

class RoboticsPartsPricingEngine:

    def __init__(
        self,
        config: PartsPricingConfig | None = None,
    ) -> None:
        self.config = (
            config or PartsPricingConfig()
        )

        self._parts: dict[
            tuple[str, str],
            PricingPart,
        ] = {}

        self._suppliers: dict[
            tuple[str, str],
            PricingSupplier,
        ] = {}

        self._offers: dict[
            tuple[str, str],
            PartPriceOffer,
        ] = {}

        self._history: deque[
            PriceHistoryEvent
        ] = deque(
            maxlen=self.config.maximum_history_events
        )

        self._audit: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit_event(
        self,
        *,
        tenant_id: str,
        action: str,
        entity_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._audit.append(
            {
                "event_id": new_id(),
                "timestamp": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "action": action,
                "entity_id": entity_id,
                "details": details or {},
            }
        )

    # --------------------------------------------------------
    # REGISTER PART
    # --------------------------------------------------------

    def register_part(
        self,
        part: PricingPart,
    ) -> dict[str, Any]:
        if not isinstance(
            part,
            PricingPart,
        ):
            raise PricingValidationError(
                "Invalid part"
            )

        key = (
            part.tenant_id,
            part.part_id,
        )

        with self._lock:
            if key in self._parts:
                raise PricingConflict(
                    "Part already registered"
                )

            if (
                len(self._parts)
                >= self.config.maximum_parts
            ):
                raise PartsPricingError(
                    "Part registry capacity reached"
                )

            self._parts[key] = part

            self._audit_event(
                tenant_id=part.tenant_id,
                action="part_registered",
                entity_id=part.part_id,
            )

        return part.to_dict()

    # --------------------------------------------------------
    # REGISTER SUPPLIER
    # --------------------------------------------------------

    def register_supplier(
        self,
        supplier: PricingSupplier,
    ) -> dict[str, Any]:
        if not isinstance(
            supplier,
            PricingSupplier,
        ):
            raise PricingValidationError(
                "Invalid supplier"
            )

        key = (
            supplier.tenant_id,
            supplier.supplier_id,
        )

        with self._lock:
            if key in self._suppliers:
                raise PricingConflict(
                    "Supplier already registered"
                )

            if (
                len(self._suppliers)
                >= self.config.maximum_suppliers
            ):
                raise PartsPricingError(
                    "Supplier registry capacity reached"
                )

            self._suppliers[key] = supplier

            self._audit_event(
                tenant_id=supplier.tenant_id,
                action="supplier_registered",
                entity_id=supplier.supplier_id,
            )

        return supplier.to_dict()

    # --------------------------------------------------------
    # REGISTER OFFER
    # --------------------------------------------------------

    def register_offer(
        self,
        offer: PartPriceOffer,
    ) -> dict[str, Any]:
        if not isinstance(
            offer,
            PartPriceOffer,
        ):
            raise PricingValidationError(
                "Invalid price offer"
            )

        key = (
            offer.tenant_id,
            offer.offer_id,
        )

        with self._lock:
            if key in self._offers:
                raise PricingConflict(
                    "Offer already registered"
                )

            if (
                offer.tenant_id,
                offer.part_id,
            ) not in self._parts:
                raise PartPriceNotFound(
                    "Part not registered"
                )

            if (
                offer.tenant_id,
                offer.supplier_id,
            ) not in self._suppliers:
                raise SupplierPriceNotFound(
                    "Supplier not registered"
                )

            if (
                len(self._offers)
                >= self.config.maximum_offers
            ):
                raise PartsPricingError(
                    "Offer registry capacity reached"
                )

            self._offers[key] = offer

            self._history.append(
                PriceHistoryEvent(
                    event_id=new_id(),
                    tenant_id=offer.tenant_id,
                    offer_id=offer.offer_id,
                    part_id=offer.part_id,
                    supplier_id=offer.supplier_id,
                    previous_price=None,
                    new_price=offer.unit_price,
                    currency=offer.currency,
                )
            )

            self._audit_event(
                tenant_id=offer.tenant_id,
                action="offer_registered",
                entity_id=offer.offer_id,
                details={
                    "part_id": offer.part_id,
                    "supplier_id": offer.supplier_id,
                },
            )

        return offer.to_dict()

    # --------------------------------------------------------
    # UPDATE OFFER PRICE
    # --------------------------------------------------------

    def update_offer_price(
        self,
        *,
        tenant_id: str,
        offer_id: str,
        new_price: Any,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PricingPermissionDenied(
                "Pricing authorization required"
            )

        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            offer_id,
            "offer_id",
        )

        new_price = monetary(
            decimal_number(
                new_price,
                "new_price",
            )
        )

        with self._lock:
            key = (
                tenant_id,
                offer_id,
            )

            offer = self._offers.get(
                key
            )

            if offer is None:
                raise SupplierPriceNotFound(
                    "Offer not found"
                )

            previous_price = (
                offer.unit_price
            )

            if new_price == previous_price:
                return offer.to_dict()

            updated = replace(
                offer,
                unit_price=new_price,
                updated_at=utc_now(),
            )

            self._offers[key] = updated

            self._history.append(
                PriceHistoryEvent(
                    event_id=new_id(),
                    tenant_id=tenant_id,
                    offer_id=offer_id,
                    part_id=offer.part_id,
                    supplier_id=offer.supplier_id,
                    previous_price=previous_price,
                    new_price=new_price,
                    currency=offer.currency,
                )
            )

            self._audit_event(
                tenant_id=tenant_id,
                action="offer_price_updated",
                entity_id=offer_id,
                details={
                    "previous_price": (
                        monetary_string(
                            previous_price
                        )
                    ),
                    "new_price": (
                        monetary_string(
                            new_price
                        )
                    ),
                },
            )

            return updated.to_dict()

    # --------------------------------------------------------
    # UPDATE OFFER AVAILABILITY
    # --------------------------------------------------------

    def update_offer_availability(
        self,
        *,
        tenant_id: str,
        offer_id: str,
        available_quantity: int,
        lead_time_days: int | None = None,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise PricingPermissionDenied(
                "Pricing authorization required"
            )

        validate_quantity(
            available_quantity,
            "available_quantity",
            allow_zero=True,
        )

        if lead_time_days is not None:
            validate_quantity(
                lead_time_days,
                "lead_time_days",
                allow_zero=True,
            )

        with self._lock:
            key = (
                tenant_id,
                offer_id,
            )

            offer = self._offers.get(key)

            if offer is None:
                raise SupplierPriceNotFound(
                    "Offer not found"
                )

            changes = {
                "available_quantity": (
                    available_quantity
                ),
                "updated_at": utc_now(),
            }

            if lead_time_days is not None:
                changes["lead_time_days"] = (
                    lead_time_days
                )

            updated = replace(
                offer,
                **changes,
            )

            self._offers[key] = updated

            self._audit_event(
                tenant_id=tenant_id,
                action="offer_availability_updated",
                entity_id=offer_id,
                details={
                    "available_quantity": (
                        available_quantity
                    ),
                },
            )

            return updated.to_dict()

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
            part = self._parts.get(
                (tenant_id, part_id)
            )

            if part is None:
                raise PartPriceNotFound(
                    "Part not found"
                )

            return part.to_dict()

    # --------------------------------------------------------
    # GET OFFER
    # --------------------------------------------------------

    def get_offer(
        self,
        *,
        tenant_id: str,
        offer_id: str,
    ) -> dict[str, Any]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            offer_id,
            "offer_id",
        )

        with self._lock:
            offer = self._offers.get(
                (tenant_id, offer_id)
            )

            if offer is None:
                raise SupplierPriceNotFound(
                    "Offer not found"
                )

            return offer.to_dict()

    # --------------------------------------------------------
    # LIST OFFERS
    # --------------------------------------------------------

    def list_offers(
        self,
        *,
        tenant_id: str,
        part_id: str | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        if part_id is not None:
            validate_id(
                part_id,
                "part_id",
            )

        with self._lock:
            return [
                offer.to_dict()
                for (tenant, _), offer
                in self._offers.items()
                if (
                    tenant == tenant_id
                    and (
                        part_id is None
                        or offer.part_id == part_id
                    )
                )
            ]

    # --------------------------------------------------------
    # QUOTE VALIDITY
    # --------------------------------------------------------

    @staticmethod
    def quote_status(
        offer: PartPriceOffer,
        now: datetime | None = None,
    ) -> QuoteStatus:
        now = validate_datetime(
            now or utc_now(),
            "now",
        )

        if (
            offer.valid_until is not None
            and validate_datetime(
                offer.valid_until,
                "valid_until",
            ) < now
        ):
            return QuoteStatus.EXPIRED

        if offer.available_quantity == 0:
            return QuoteStatus.UNAVAILABLE

        return QuoteStatus.VALID

    # --------------------------------------------------------
    # QUANTITY DISCOUNT
    # --------------------------------------------------------

    @staticmethod
    def applicable_discount(
        offer: PartPriceOffer,
        quantity: int,
    ) -> Decimal:
        validate_quantity(
            quantity
        )

        discount = Decimal("0")

        for tier in offer.quantity_tiers:
            if (
                quantity >= tier.minimum_quantity
            ):
                discount = (
                    tier.discount_percent
                )

        # Select the tier with the greatest
        # qualifying minimum quantity.
        eligible = [
            tier
            for tier in offer.quantity_tiers
            if quantity >= tier.minimum_quantity
        ]

        if eligible:
            selected = max(
                eligible,
                key=lambda tier: (
                    tier.minimum_quantity
                ),
            )

            discount = (
                selected.discount_percent
            )

        return discount

    # --------------------------------------------------------
    # CALCULATE OFFER PRICE
    # --------------------------------------------------------

    def calculate_offer_price(
        self,
        *,
        offer: PartPriceOffer,
        supplier: PricingSupplier,
        request: PartPricingRequest,
    ) -> PartPriceBreakdown:
        if (
            offer.tenant_id != request.tenant_id
            or offer.part_id != request.part_id
        ):
            raise PricingValidationError(
                "Offer does not match request"
            )

        if (
            supplier.tenant_id != request.tenant_id
            or supplier.supplier_id
            != offer.supplier_id
        ):
            raise PricingValidationError(
                "Supplier does not match offer"
            )

        if offer.currency != request.currency:
            raise CurrencyMismatch(
                "Explicit currency conversion required"
            )

        if (
            supplier.status
            != SupplierStatus.ACTIVE
        ):
            raise PricingConflict(
                "Supplier is not active"
            )

        if (
            self.quote_status(offer)
            == QuoteStatus.EXPIRED
        ):
            raise QuoteExpired(
                "Supplier quote has expired"
            )

        if (
            request.quantity
            < offer.minimum_order_quantity
        ):
            raise PricingValidationError(
                "Quantity below minimum order"
            )

        if (
            request.quantity
            > offer.available_quantity
        ):
            raise PricingConflict(
                "Insufficient supplier stock"
            )

        if (
            request.maximum_lead_time_days
            is not None
            and offer.lead_time_days
            > request.maximum_lead_time_days
        ):
            raise PricingConflict(
                "Supplier lead time exceeds limit"
            )

        if (
            offer.condition == PartCondition.REFURBISHED
            and not request.allow_refurbished
        ):
            raise PricingConflict(
                "Refurbished parts not allowed"
            )

        if (
            offer.condition == PartCondition.USED
            and not request.allow_used
        ):
            raise PricingConflict(
                "Used parts not allowed"
            )

        if offer.condition == PartCondition.UNKNOWN:
            raise PricingConflict(
                "Unknown part condition requires review"
            )

        discount_percent = (
            self.applicable_discount(
                offer,
                request.quantity,
            )
        )

        discount_multiplier = (
            Decimal("1")
            - discount_percent
            / Decimal("100")
        )

        discounted_unit_price = monetary(
            offer.unit_price
            * discount_multiplier
        )

        parts_subtotal = monetary(
            discounted_unit_price
            * request.quantity
        )

        shipping = offer.shipping_cost
        handling = offer.handling_cost

        emergency_surcharge = Decimal("0")

        if (
            request.priority
            == ProcurementPriority.EMERGENCY
        ):
            emergency_surcharge = monetary(
                parts_subtotal
                * decimal_number(
                    self.config.emergency_surcharge_percent,
                    "emergency_surcharge_percent",
                )
                / Decimal("100")
            )

        taxable_amount = (
            parts_subtotal
            + shipping
            + handling
            + emergency_surcharge
        )

        tax_amount = monetary(
            taxable_amount
            * offer.tax_percent
            / Decimal("100")
        )

        total = monetary(
            taxable_amount
            + tax_amount
            + request.installation_labor_cost
            + request.additional_cost
        )

        within_budget = (
            total <= request.budget
            if request.budget is not None
            else None
        )

        return PartPriceBreakdown(
            offer_id=offer.offer_id,
            supplier_id=offer.supplier_id,
            supplier_name=(
                supplier.supplier_name
            ),
            part_id=offer.part_id,
            quantity=request.quantity,
            currency=request.currency,
            base_unit_price=offer.unit_price,
            discount_percent=discount_percent,
            discounted_unit_price=(
                discounted_unit_price
            ),
            parts_subtotal=parts_subtotal,
            shipping_cost=shipping,
            handling_cost=handling,
            emergency_surcharge=(
                emergency_surcharge
            ),
            tax_amount=tax_amount,
            installation_labor_cost=(
                request.installation_labor_cost
            ),
            additional_cost=(
                request.additional_cost
            ),
            total_cost=total,
            lead_time_days=offer.lead_time_days,
            supplier_reliability_score=(
                supplier.reliability_score
            ),
            within_budget=within_budget,
        )

    # --------------------------------------------------------
    # COMPARE SUPPLIER PRICES
    # --------------------------------------------------------

    def compare_suppliers(
        self,
        request: PartPricingRequest,
    ) -> dict[str, Any]:
        if not isinstance(
            request,
            PartPricingRequest,
        ):
            raise PricingValidationError(
                "Invalid pricing request"
            )

        with self._lock:
            if (
                request.tenant_id,
                request.part_id,
            ) not in self._parts:
                raise PartPriceNotFound(
                    "Part not registered"
                )

            offers = [
                offer
                for (tenant, _), offer
                in self._offers.items()
                if (
                    tenant == request.tenant_id
                    and offer.part_id
                    == request.part_id
                )
            ]

            suppliers = dict(
                self._suppliers
            )

        comparisons = []
        rejected_offers = []

        for offer in offers:
            supplier = suppliers.get(
                (
                    request.tenant_id,
                    offer.supplier_id,
                )
            )

            if supplier is None:
                rejected_offers.append(
                    {
                        "offer_id": offer.offer_id,
                        "reason": "supplier_not_found",
                    }
                )
                continue

            try:
                breakdown = (
                    self.calculate_offer_price(
                        offer=offer,
                        supplier=supplier,
                        request=request,
                    )
                )

                comparisons.append(
                    breakdown
                )

            except PartsPricingError as exc:
                rejected_offers.append(
                    {
                        "offer_id": offer.offer_id,
                        "reason": str(exc),
                    }
                )

        comparisons.sort(
            key=lambda item: (
                item.total_cost,
                item.lead_time_days,
                -item.supplier_reliability_score,
            )
        )

        lowest_cost = (
            min(
                comparisons,
                key=lambda item: (
                    item.total_cost,
                    item.lead_time_days,
                ),
            )
            if comparisons
            else None
        )

        fastest_delivery = (
            min(
                comparisons,
                key=lambda item: (
                    item.lead_time_days,
                    item.total_cost,
                ),
            )
            if comparisons
            else None
        )

        best_value = (
            min(
                comparisons,
                key=lambda item: (
                    item.total_cost
                    / max(
                        item.supplier_reliability_score,
                        Decimal("0.01"),
                    ),
                    item.lead_time_days,
                ),
            )
            if comparisons
            else None
        )

        return {
            "tenant_id": request.tenant_id,
            "part_id": request.part_id,
            "quantity": request.quantity,
            "currency": request.currency,
            "priority": request.priority.value,
            "valid_offer_count": len(
                comparisons
            ),
            "offers": [
                item.to_dict()
                for item in comparisons
            ],
            "rejected_offers": rejected_offers,
            "recommendations": {
                "lowest_cost": (
                    lowest_cost.offer_id
                    if lowest_cost
                    else None
                ),
                "fastest_delivery": (
                    fastest_delivery.offer_id
                    if fastest_delivery
                    else None
                ),
                "best_value": (
                    best_value.offer_id
                    if best_value
                    else None
                ),
            },
            "supplier_reliability_is_unverified": True,
            "automatic_purchase": False,
            "human_approval_required": True,
        }

    # --------------------------------------------------------
    # GET BEST PRICE
    # --------------------------------------------------------

    def get_best_price(
        self,
        request: PartPricingRequest,
        *,
        recommendation: RecommendationType = (
            RecommendationType.LOWEST_COST
        ),
    ) -> dict[str, Any]:
        comparison = self.compare_suppliers(
            request
        )

        offers = comparison["offers"]

        if not offers:
            return {
                "part_id": request.part_id,
                "available": False,
                "selected_offer": None,
                "recommendation": (
                    RecommendationType.NO_AVAILABLE_OFFER.value
                ),
            }

        if (
            recommendation
            == RecommendationType.FASTEST_DELIVERY
        ):
            selected = min(
                offers,
                key=lambda item: (
                    item["lead_time_days"],
                    Decimal(item["total_cost"]),
                ),
            )

        elif (
            recommendation
            == RecommendationType.BEST_VALUE
        ):
            selected = min(
                offers,
                key=lambda item: (
                    Decimal(item["total_cost"])
                    / max(
                        Decimal(
                            item[
                                "supplier_reliability_score"
                            ]
                        ),
                        Decimal("0.01"),
                    )
                ),
            )

        else:
            selected = min(
                offers,
                key=lambda item: (
                    Decimal(item["total_cost"])
                ),
            )

        return {
            "part_id": request.part_id,
            "available": True,
            "selected_offer": selected,
            "recommendation": (
                recommendation.value
            ),
            "human_approval_required": True,
        }

    # --------------------------------------------------------
    # PRICE HISTORY
    # --------------------------------------------------------

    def price_history(
        self,
        *,
        tenant_id: str,
        part_id: str,
        supplier_id: str | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            part_id,
            "part_id",
        )

        if supplier_id is not None:
            validate_id(
                supplier_id,
                "supplier_id",
            )

        with self._lock:
            events = [
                event
                for event in self._history
                if (
                    event.tenant_id == tenant_id
                    and event.part_id == part_id
                    and (
                        supplier_id is None
                        or event.supplier_id
                        == supplier_id
                    )
                )
            ]

        events.sort(
            key=lambda event: (
                event.recorded_at
            )
        )

        return [
            event.to_dict()
            for event in events
        ]

    # --------------------------------------------------------
    # PRICE TREND
    # --------------------------------------------------------

    def analyze_price_trend(
        self,
        *,
        tenant_id: str,
        part_id: str,
        supplier_id: str,
    ) -> dict[str, Any]:
        history = self.price_history(
            tenant_id=tenant_id,
            part_id=part_id,
            supplier_id=supplier_id,
        )

        if len(history) < 2:
            return {
                "part_id": part_id,
                "supplier_id": supplier_id,
                "trend": (
                    PriceTrend.INSUFFICIENT_DATA.value
                ),
                "observations": len(history),
            }

        # Analyze the latest offer's own price
        # history, rather than mixing offers.
        latest_offer_id = (
            history[-1]["offer_id"]
        )

        observations = [
            item
            for item in history
            if item["offer_id"] == latest_offer_id
        ]

        if len(observations) < 2:
            return {
                "part_id": part_id,
                "supplier_id": supplier_id,
                "offer_id": latest_offer_id,
                "trend": (
                    PriceTrend.INSUFFICIENT_DATA.value
                ),
                "observations": len(observations),
            }

        prices = [
            Decimal(item["new_price"])
            for item in observations
        ]

        first_price = prices[0]
        last_price = prices[-1]

        absolute_change = (
            last_price - first_price
        )

        if first_price == 0:
            percentage_change = None

            trend = (
                PriceTrend.STABLE
                if last_price == 0
                else PriceTrend.INCREASING
            )

        else:
            percentage_change = (
                absolute_change
                / first_price
                * Decimal("100")
            )

            threshold = decimal_number(
                self.config.stable_trend_threshold_percent,
                "stable_trend_threshold_percent",
            )

            if (
                percentage_change > threshold
            ):
                trend = PriceTrend.INCREASING

            elif (
                percentage_change < -threshold
            ):
                trend = PriceTrend.DECREASING

            else:
                trend = PriceTrend.STABLE

        volatility = (
            statistics.pstdev(
                float(price)
                for price in prices
            )
            if len(prices) >= 2
            else 0.0
        )

        return {
            "tenant_id": tenant_id,
            "part_id": part_id,
            "supplier_id": supplier_id,
            "offer_id": latest_offer_id,
            "currency": observations[-1]["currency"],
            "trend": trend.value,
            "observations": len(prices),
            "first_price": monetary_string(
                first_price
            ),
            "latest_price": monetary_string(
                last_price
            ),
            "absolute_change": monetary_string(
                absolute_change
            ),
            "percentage_change": (
                str(
                    percentage_change.quantize(
                        Decimal("0.01")
                    )
                )
                if percentage_change is not None
                else None
            ),
            "price_standard_deviation": round(
                volatility,
                4,
            ),
            "forecast": None,
            "analysis_type": "historical_descriptive",
        }

    # --------------------------------------------------------
    # DETECT PRICE CHANGES
    # --------------------------------------------------------

    def detect_price_changes(
        self,
        *,
        tenant_id: str,
        minimum_change_percent: Any = 5,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        threshold = percentage(
            minimum_change_percent,
            "minimum_change_percent",
        )

        with self._lock:
            events = [
                event
                for event in self._history
                if (
                    event.tenant_id == tenant_id
                    and event.previous_price is not None
                )
            ]

        alerts = []

        for event in events:
            previous = event.previous_price

            if previous is None:
                continue

            if previous == 0:
                continue

            change_percent = (
                (
                    event.new_price - previous
                )
                / previous
                * Decimal("100")
            )

            if abs(change_percent) < threshold:
                continue

            alerts.append(
                {
                    "event_id": event.event_id,
                    "part_id": event.part_id,
                    "supplier_id": event.supplier_id,
                    "offer_id": event.offer_id,
                    "previous_price": (
                        monetary_string(previous)
                    ),
                    "new_price": (
                        monetary_string(
                            event.new_price
                        )
                    ),
                    "currency": event.currency,
                    "change_percent": str(
                        change_percent.quantize(
                            Decimal("0.01")
                        )
                    ),
                    "direction": (
                        "increase"
                        if change_percent > 0
                        else "decrease"
                    ),
                    "recorded_at": (
                        event.recorded_at.isoformat()
                    ),
                }
            )

        return alerts

    # --------------------------------------------------------
    # CURRENCY CONVERSION
    # --------------------------------------------------------

    @staticmethod
    def convert_currency(
        *,
        amount: Any,
        source_currency: str,
        target_currency: str,
        exchange_rate: Any,
    ) -> dict[str, Any]:
        amount = decimal_number(
            amount,
            "amount",
        )

        source_currency = validate_currency(
            source_currency
        )

        target_currency = validate_currency(
            target_currency
        )

        rate = decimal_number(
            exchange_rate,
            "exchange_rate",
        )

        if rate <= 0:
            raise PricingValidationError(
                "Exchange rate must be positive"
            )

        converted = monetary(
            amount * rate
        )

        return {
            "source_currency": source_currency,
            "target_currency": target_currency,
            "original_amount": monetary_string(
                amount
            ),
            "exchange_rate": str(rate),
            "converted_amount": monetary_string(
                converted
            ),
            "exchange_rate_source": "user_supplied",
            "live_rate_verified": False,
        }

    # --------------------------------------------------------
    # MULTI-PART COST ESTIMATE
    # --------------------------------------------------------

    def estimate_parts_bundle(
        self,
        *,
        tenant_id: str,
        items: Sequence[tuple[str, int]],
        currency: str = "INR",
        priority: ProcurementPriority = (
            ProcurementPriority.STANDARD
        ),
        installation_labor_cost: Any = 0,
    ) -> dict[str, Any]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_currency(
            currency
        )

        if not items:
            raise PricingValidationError(
                "At least one part is required"
            )

        labor = monetary(
            decimal_number(
                installation_labor_cost,
                "installation_labor_cost",
            )
        )

        results = []
        subtotal = Decimal("0")
        unavailable = []

        for part_id, quantity in items:
            request = PartPricingRequest(
                tenant_id=tenant_id,
                part_id=part_id,
                quantity=quantity,
                currency=currency,
                priority=priority,
            )

            result = self.get_best_price(
                request
            )

            if not result["available"]:
                unavailable.append(
                    part_id
                )
                continue

            selected = result[
                "selected_offer"
            ]

            subtotal += Decimal(
                selected["total_cost"]
            )

            results.append(
                {
                    "part_id": part_id,
                    "quantity": quantity,
                    "selected_offer": selected,
                }
            )

        total = monetary(
            subtotal + labor
        )

        return {
            "tenant_id": tenant_id,
            "currency": currency,
            "items": results,
            "unavailable_part_ids": unavailable,
            "parts_total": monetary_string(
                subtotal
            ),
            "installation_labor_cost": (
                monetary_string(labor)
            ),
            "estimated_total": (
                monetary_string(total)
            ),
            "complete_estimate": (
                len(unavailable) == 0
            ),
            "human_approval_required": True,
        }

    # --------------------------------------------------------
    # PARTS CATALOG INTEGRATION
    # --------------------------------------------------------

    def register_from_catalog_part(
        self,
        catalog_part: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(
            catalog_part,
            dict,
        ):
            raise PricingValidationError(
                "Invalid catalog part"
            )

        category = catalog_part.get(
            "category",
            "other",
        )

        if isinstance(category, Enum):
            category = category.value

        part = PricingPart(
            tenant_id=catalog_part["tenant_id"],
            part_id=catalog_part["part_id"],
            part_name=catalog_part["part_name"],
            manufacturer=catalog_part.get(
                "manufacturer",
                "",
            ),
            model_number=catalog_part.get(
                "model_number",
                "",
            ),
            part_number=catalog_part.get(
                "part_number",
                "",
            ),
            category=category,
            safety_critical=catalog_part.get(
                "safety_critical",
                False,
            ),
        )

        return self.register_part(
            part
        )

    # --------------------------------------------------------
    # IDENTIFICATION INTEGRATION
    # --------------------------------------------------------

    def estimate_identified_part(
        self,
        *,
        tenant_id: str,
        identification_result: dict[str, Any],
        quantity: int = 1,
        currency: str = "INR",
    ) -> dict[str, Any]:
        if not isinstance(
            identification_result,
            dict,
        ):
            raise PricingValidationError(
                "Invalid identification result"
            )

        if (
            identification_result.get("tenant_id")
            != tenant_id
        ):
            raise PricingValidationError(
                "Tenant mismatch"
            )

        verified = (
            identification_result.get(
                "review_status"
            ) == "approved"
            and bool(
                identification_result.get(
                    "verified_part_id"
                )
            )
        )

        part_id = (
            identification_result.get(
                "verified_part_id"
            )
            if verified
            else identification_result.get(
                "best_part_id"
            )
        )

        if not part_id:
            return {
                "identified": False,
                "pricing_available": False,
                "human_review_required": True,
            }

        request = PartPricingRequest(
            tenant_id=tenant_id,
            part_id=part_id,
            quantity=quantity,
            currency=currency,
        )

        estimate = self.get_best_price(
            request
        )

        return {
            "identified": True,
            "part_id": part_id,
            "identification_verified": verified,
            "pricing": estimate,
            "human_review_required": True,
        }

    # --------------------------------------------------------
    # PRICING SUMMARY
    # --------------------------------------------------------

    def pricing_summary(
        self,
        *,
        tenant_id: str,
    ) -> dict[str, Any]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        with self._lock:
            parts = [
                part
                for (tenant, _), part
                in self._parts.items()
                if tenant == tenant_id
            ]

            suppliers = [
                supplier
                for (tenant, _), supplier
                in self._suppliers.items()
                if tenant == tenant_id
            ]

            offers = [
                offer
                for (tenant, _), offer
                in self._offers.items()
                if tenant == tenant_id
            ]

        valid_offers = sum(
            self.quote_status(offer)
            == QuoteStatus.VALID
            for offer in offers
        )

        expired_offers = sum(
            self.quote_status(offer)
            == QuoteStatus.EXPIRED
            for offer in offers
        )

        currency_counts = defaultdict(int)

        for offer in offers:
            currency_counts[
                offer.currency
            ] += 1

        return {
            "tenant_id": tenant_id,
            "registered_parts": len(parts),
            "registered_suppliers": len(suppliers),
            "total_offers": len(offers),
            "valid_offers": valid_offers,
            "expired_offers": expired_offers,
            "offers_by_currency": dict(
                currency_counts
            ),
            "simulation_only": True,
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
            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "parts": [
                    part.to_dict()
                    for (tenant, _), part
                    in self._parts.items()
                    if tenant == tenant_id
                ],
                "suppliers": [
                    supplier.to_dict()
                    for (tenant, _), supplier
                    in self._suppliers.items()
                    if tenant == tenant_id
                ],
                "offers": [
                    offer.to_dict()
                    for (tenant, _), offer
                    in self._offers.items()
                    if tenant == tenant_id
                ],
                "price_history": [
                    event.to_dict()
                    for event in self._history
                    if event.tenant_id == tenant_id
                ],
                "audit_events": [
                    event
                    for event in self._audit
                    if event["tenant_id"] == tenant_id
                ],
                "summary": self.pricing_summary(
                    tenant_id=tenant_id
                ),
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
    engine = RoboticsPartsPricingEngine()

    tenant_id = "tenant_alpha"

    engine.register_part(
        PricingPart(
            tenant_id=tenant_id,
            part_id="motor_001",
            part_name="Industrial Robot Joint Motor",
            manufacturer="Demo Robotics",
            model_number="MTR-100",
            part_number="DR-MTR-100",
            category="motor",
            safety_critical=True,
        )
    )

    engine.register_supplier(
        PricingSupplier(
            tenant_id=tenant_id,
            supplier_id="supplier_001",
            supplier_name="Robotics Parts India",
            country="India",
            reliability_score=Decimal("0.92"),
        )
    )

    engine.register_supplier(
        PricingSupplier(
            tenant_id=tenant_id,
            supplier_id="supplier_002",
            supplier_name="Industrial Automation Supply",
            country="India",
            reliability_score=Decimal("0.85"),
        )
    )

    engine.register_offer(
        PartPriceOffer(
            tenant_id=tenant_id,
            offer_id="offer_001",
            part_id="motor_001",
            supplier_id="supplier_001",
            unit_price=Decimal("8500"),
            currency="INR",
            available_quantity=25,
            minimum_order_quantity=1,
            lead_time_days=5,
            shipping_cost=Decimal("250"),
            handling_cost=Decimal("100"),
            tax_percent=Decimal("18"),
            quantity_tiers=(
                QuantityPriceTier(
                    minimum_quantity=5,
                    discount_percent=Decimal("5"),
                ),
                QuantityPriceTier(
                    minimum_quantity=10,
                    discount_percent=Decimal("10"),
                ),
            ),
            reference="DEMO-QUOTE-001",
        )
    )

    engine.register_offer(
        PartPriceOffer(
            tenant_id=tenant_id,
            offer_id="offer_002",
            part_id="motor_001",
            supplier_id="supplier_002",
            unit_price=Decimal("8100"),
            currency="INR",
            available_quantity=15,
            minimum_order_quantity=1,
            lead_time_days=9,
            shipping_cost=Decimal("450"),
            handling_cost=Decimal("150"),
            tax_percent=Decimal("18"),
            reference="DEMO-QUOTE-002",
        )
    )

    request = PartPricingRequest(
        tenant_id=tenant_id,
        part_id="motor_001",
        quantity=5,
        currency="INR",
        priority=ProcurementPriority.STANDARD,
        installation_labor_cost=Decimal("1500"),
        budget=Decimal("60000"),
    )

    comparison = engine.compare_suppliers(
        request
    )

    print("\nSUPPLIER COMPARISON")

    print(
        json.dumps(
            comparison,
            indent=2,
        )
    )

    print("\nBEST PRICE")

    print(
        json.dumps(
            engine.get_best_price(
                request
            ),
            indent=2,
        )
    )

    engine.update_offer_price(
        tenant_id=tenant_id,
        offer_id="offer_001",
        new_price=Decimal("8900"),
        authorized=True,
    )

    print("\nPRICE TREND")

    print(
        json.dumps(
            engine.analyze_price_trend(
                tenant_id=tenant_id,
                part_id="motor_001",
                supplier_id="supplier_001",
            ),
            indent=2,
        )
    )

    print("\nPRICE CHANGE ALERTS")

    print(
        json.dumps(
            engine.detect_price_changes(
                tenant_id=tenant_id,
                minimum_change_percent=3,
            ),
            indent=2,
        )
    )

    print("\nPRICING SUMMARY")

    print(
        json.dumps(
            engine.pricing_summary(
                tenant_id=tenant_id
            ),
            indent=2,
        )
    )

    report = engine.export_json(
        "data/robotics/parts_pricing_report.json",
        tenant_id=tenant_id,
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()
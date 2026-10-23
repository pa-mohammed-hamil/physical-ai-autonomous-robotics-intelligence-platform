
"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/robotics/control/message_converter.py

Python 3.11+
Standard library only.

Features:
- Agent-to-robot message conversion
- Robot-to-agent telemetry conversion
- Navigation and manipulation message schemas
- VLA action message conversion
- Simulation action request conversion
- Strict JSON serialization
- Message validation
- Timestamp normalization
- Tenant and robot isolation
- Message versioning
- Payload size limits
- Audit logging
- Batch conversion
- JSON report export

SAFETY:
    Data conversion only.

    No physical robot control, command execution,
    safety authorization or ROS 2 publication
    occurs in this module.

    Converted messages must pass independent
    safety and command authorization checks
    before simulation dispatch.
"""

from __future__ import annotations

import json
import math
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class MessageConverterError(Exception):
    pass


class MessageValidationError(MessageConverterError):
    pass


class UnsupportedMessageType(MessageConverterError):
    pass


# ============================================================
# ENUMS
# ============================================================

class MessageType(str, Enum):
    NAVIGATION_COMMAND = "navigation_command"
    MANIPULATION_COMMAND = "manipulation_command"
    VLA_ACTION = "vla_action"
    AGENT_ACTION = "agent_action"
    ROBOT_TELEMETRY = "robot_telemetry"
    SENSOR_DATA = "sensor_data"
    ROBOT_STATUS = "robot_status"
    SAFETY_EVENT = "safety_event"
    ACTION_RESULT = "action_result"
    SIMULATION_EVENT = "simulation_event"


class MessageSource(str, Enum):
    AGENT = "agent"
    VLA_MODEL = "vla_model"
    PLANNER = "planner"
    ROBOT = "robot"
    SENSOR = "sensor"
    SIMULATOR = "simulator"
    SAFETY_SERVICE = "safety_service"


class MessagePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            char.isalnum() or char in "_-"
            for char in value
        )
    ):
        raise MessageValidationError(
            f"Invalid {name}"
        )

    return value


def finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise MessageValidationError(
            f"{name} must be a finite number"
        )

    return float(value)


def validate_timestamp(
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise MessageValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


def normalize_json(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise MessageValidationError(
            "Payload must be a mapping"
        )

    try:
        encoded = json.dumps(
            dict(payload),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        decoded = json.loads(encoded)

    except (TypeError, ValueError) as exc:
        raise MessageValidationError(
            "Payload is not valid JSON data"
        ) from exc

    if not isinstance(decoded, dict):
        raise MessageValidationError(
            "Payload must be a JSON object"
        )

    return decoded


def require_vector3(
    value: Any,
    name: str,
) -> dict[str, float]:
    if not isinstance(value, dict):
        raise MessageValidationError(
            f"{name} must be an object"
        )

    if set(value) != {"x", "y", "z"}:
        raise MessageValidationError(
            f"{name} requires x, y and z"
        )

    return {
        axis: finite(value[axis], f"{name}.{axis}")
        for axis in ("x", "y", "z")
    }


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class MessageConverterConfig:
    schema_version: str = "1.0"

    maximum_payload_bytes: int = 65536
    maximum_batch_size: int = 1000
    maximum_audit_events: int = 20000

    simulation_only: bool = True

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise MessageValidationError(
                "Schema version is required"
            )

        for name in (
            "maximum_payload_bytes",
            "maximum_batch_size",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise MessageValidationError(
                    f"{name} must be positive"
                )

        if not self.simulation_only:
            raise MessageValidationError(
                "Hardware message dispatch unsupported"
            )


# ============================================================
# STANDARD MESSAGE
# ============================================================

@dataclass(frozen=True)
class RobotMessage:
    tenant_id: str
    robot_id: str

    message_type: MessageType
    source: MessageSource

    payload: Mapping[str, Any]

    priority: MessagePriority = (
        MessagePriority.NORMAL
    )

    message_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    timestamp: datetime = field(
        default_factory=utc_now
    )

    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.message_id, "message_id")

        if not isinstance(self.message_type, MessageType):
            raise MessageValidationError(
                "Invalid message type"
            )

        if not isinstance(self.source, MessageSource):
            raise MessageValidationError(
                "Invalid message source"
            )

        if not isinstance(self.priority, MessagePriority):
            raise MessageValidationError(
                "Invalid message priority"
            )

        validate_timestamp(self.timestamp)

        object.__setattr__(
            self,
            "payload",
            normalize_json(self.payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "message_type": self.message_type.value,
            "source": self.source.value,
            "priority": self.priority.value,
            "schema_version": self.schema_version,
            "timestamp": self.timestamp.isoformat(),
            "payload": dict(self.payload),
        }


# ============================================================
# MESSAGE VALIDATOR
# ============================================================

class RobotMessageValidator:

    @staticmethod
    def validate_payload(
        message: RobotMessage,
    ) -> None:
        payload = message.payload

        if message.message_type == (
            MessageType.NAVIGATION_COMMAND
        ):
            require_vector3(
                payload.get("target"),
                "target",
            )

            speed = finite(
                payload.get("speed_mps"),
                "speed_mps",
            )

            if speed < 0:
                raise MessageValidationError(
                    "Speed cannot be negative"
                )

        elif message.message_type == (
            MessageType.MANIPULATION_COMMAND
        ):
            operation = payload.get("operation")

            if operation not in {
                "pick",
                "place",
                "move_arm",
                "gripper",
            }:
                raise MessageValidationError(
                    "Unsupported manipulation operation"
                )

        elif message.message_type == (
            MessageType.VLA_ACTION
        ):
            action = payload.get("action")

            if (
                not isinstance(action, str)
                or not action
            ):
                raise MessageValidationError(
                    "VLA action is required"
                )

        elif message.message_type == (
            MessageType.AGENT_ACTION
        ):
            action_type = payload.get(
                "action_type"
            )

            if (
                not isinstance(action_type, str)
                or not action_type
            ):
                raise MessageValidationError(
                    "Agent action type required"
                )

        elif message.message_type == (
            MessageType.ROBOT_TELEMETRY
        ):
            require_vector3(
                payload.get("position"),
                "position",
            )

            speed = finite(
                payload.get("speed_mps"),
                "speed_mps",
            )

            if speed < 0:
                raise MessageValidationError(
                    "Telemetry speed cannot be negative"
                )

        elif message.message_type == (
            MessageType.SENSOR_DATA
        ):
            if not isinstance(
                payload.get("sensor_type"),
                str,
            ):
                raise MessageValidationError(
                    "Sensor type required"
                )

        elif message.message_type == (
            MessageType.ROBOT_STATUS
        ):
            if not isinstance(
                payload.get("status"),
                str,
            ):
                raise MessageValidationError(
                    "Robot status required"
                )

        elif message.message_type == (
            MessageType.SAFETY_EVENT
        ):
            if not isinstance(
                payload.get("event_type"),
                str,
            ):
                raise MessageValidationError(
                    "Safety event type required"
                )

        elif message.message_type == (
            MessageType.ACTION_RESULT
        ):
            if not isinstance(
                payload.get("action_id"),
                str,
            ):
                raise MessageValidationError(
                    "Action ID required"
                )

        elif message.message_type == (
            MessageType.SIMULATION_EVENT
        ):
            if not isinstance(
                payload.get("event"),
                str,
            ):
                raise MessageValidationError(
                    "Simulation event required"
                )


# ============================================================
# AUDIT RECORD
# ============================================================

@dataclass(frozen=True)
class ConversionAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    message_id: str

    operation: str
    message_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "message_id": self.message_id,
            "operation": self.operation,
            "message_type": self.message_type,
        }


# ============================================================
# MESSAGE CONVERTER SERVICE
# ============================================================

class RoboticsMessageConverter:

    def __init__(
        self,
        config: MessageConverterConfig | None = None,
    ) -> None:
        self.config = (
            config or MessageConverterConfig()
        )

        self.validator = RobotMessageValidator()

        self.audit_events: list[
            ConversionAuditEvent
        ] = []

        self._lock = threading.RLock()

    def _audit(
        self,
        message: RobotMessage,
        operation: str,
    ) -> None:
        with self._lock:
            self.audit_events.append(
                ConversionAuditEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=utc_now(),
                    tenant_id=message.tenant_id,
                    robot_id=message.robot_id,
                    message_id=message.message_id,
                    operation=operation,
                    message_type=(
                        message.message_type.value
                    ),
                )
            )

            if (
                len(self.audit_events)
                > self.config.maximum_audit_events
            ):
                self.audit_events.pop(0)

    def _validate(
        self,
        message: RobotMessage,
    ) -> None:
        if (
            message.schema_version
            != self.config.schema_version
        ):
            raise MessageValidationError(
                "Unsupported schema version"
            )

        self.validator.validate_payload(
            message
        )

        encoded = json.dumps(
            message.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")

        if (
            len(encoded)
            > self.config.maximum_payload_bytes
        ):
            raise MessageValidationError(
                "Message exceeds maximum size"
            )

    # --------------------------------------------------------
    # ENCODE MESSAGE
    # --------------------------------------------------------

    def encode(
        self,
        message: RobotMessage,
    ) -> str:
        self._validate(message)

        encoded = json.dumps(
            message.to_dict(),
            separators=(",", ":"),
            allow_nan=False,
        )

        self._audit(
            message,
            "encode",
        )

        return encoded

    # --------------------------------------------------------
    # DECODE MESSAGE
    # --------------------------------------------------------

    def decode(
        self,
        raw_message: str | bytes,
    ) -> RobotMessage:
        if not isinstance(
            raw_message,
            (str, bytes),
        ):
            raise MessageValidationError(
                "Message must be text or bytes"
            )

        encoded = (
            raw_message.encode("utf-8")
            if isinstance(raw_message, str)
            else raw_message
        )

        if (
            len(encoded)
            > self.config.maximum_payload_bytes
        ):
            raise MessageValidationError(
                "Message exceeds maximum size"
            )

        try:
            data = json.loads(encoded)
        except (ValueError, UnicodeError) as exc:
            raise MessageValidationError(
                "Invalid JSON message"
            ) from exc

        if not isinstance(data, dict):
            raise MessageValidationError(
                "Message must be a JSON object"
            )

        required = {
            "message_id",
            "tenant_id",
            "robot_id",
            "message_type",
            "source",
            "priority",
            "schema_version",
            "timestamp",
            "payload",
        }

        if set(data) != required:
            raise MessageValidationError(
                "Message fields do not match schema"
            )

        try:
            timestamp = datetime.fromisoformat(
                data["timestamp"]
            )

            message = RobotMessage(
                message_id=data["message_id"],
                tenant_id=data["tenant_id"],
                robot_id=data["robot_id"],
                message_type=MessageType(
                    data["message_type"]
                ),
                source=MessageSource(
                    data["source"]
                ),
                priority=MessagePriority(
                    data["priority"]
                ),
                schema_version=data["schema_version"],
                timestamp=timestamp,
                payload=data["payload"],
            )

        except (
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            raise MessageValidationError(
                "Invalid message schema"
            ) from exc

        self._validate(message)

        self._audit(
            message,
            "decode",
        )

        return message

    # --------------------------------------------------------
    # NAVIGATION CONVERSION
    # --------------------------------------------------------

    def navigation_message(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        target_x: float,
        target_y: float,
        target_z: float,
        speed_mps: float,
    ) -> RobotMessage:
        message = RobotMessage(
            tenant_id=tenant_id,
            robot_id=robot_id,
            message_type=(
                MessageType.NAVIGATION_COMMAND
            ),
            source=MessageSource.PLANNER,
            payload={
                "target": {
                    "x": target_x,
                    "y": target_y,
                    "z": target_z,
                },
                "speed_mps": speed_mps,
            },
        )

        self._validate(message)
        return message

    # --------------------------------------------------------
    # MANIPULATION CONVERSION
    # --------------------------------------------------------

    def manipulation_message(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        operation: str,
        parameters: Mapping[str, Any],
    ) -> RobotMessage:
        message = RobotMessage(
            tenant_id=tenant_id,
            robot_id=robot_id,
            message_type=(
                MessageType.MANIPULATION_COMMAND
            ),
            source=MessageSource.AGENT,
            payload={
                "operation": operation,
                "parameters": dict(parameters),
            },
        )

        self._validate(message)
        return message

    # --------------------------------------------------------
    # VLA CONVERSION
    # --------------------------------------------------------

    def vla_action_message(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        action: str,
        confidence: float,
        parameters: Mapping[str, Any],
    ) -> RobotMessage:
        confidence = finite(
            confidence,
            "confidence",
        )

        if not 0 <= confidence <= 1:
            raise MessageValidationError(
                "Confidence must be between 0 and 1"
            )

        message = RobotMessage(
            tenant_id=tenant_id,
            robot_id=robot_id,
            message_type=MessageType.VLA_ACTION,
            source=MessageSource.VLA_MODEL,
            payload={
                "action": action,
                "confidence": confidence,
                "parameters": dict(parameters),
            },
        )

        self._validate(message)
        return message

    # --------------------------------------------------------
    # TELEMETRY CONVERSION
    # --------------------------------------------------------

    def telemetry_message(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        position: Mapping[str, float],
        speed_mps: float,
        battery_percent: float,
        temperature_c: float,
    ) -> RobotMessage:
        battery = finite(
            battery_percent,
            "battery_percent",
        )

        if not 0 <= battery <= 100:
            raise MessageValidationError(
                "Battery percentage out of range"
            )

        message = RobotMessage(
            tenant_id=tenant_id,
            robot_id=robot_id,
            message_type=MessageType.ROBOT_TELEMETRY,
            source=MessageSource.ROBOT,
            payload={
                "position": dict(position),
                "speed_mps": speed_mps,
                "battery_percent": battery,
                "temperature_c": temperature_c,
            },
        )

        self._validate(message)
        return message

    # --------------------------------------------------------
    # ACTION RESULT CONVERSION
    # --------------------------------------------------------

    def action_result_message(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        action_id: str,
        status: str,
        result: Mapping[str, Any],
    ) -> RobotMessage:
        validate_id(
            action_id,
            "action_id",
        )

        message = RobotMessage(
            tenant_id=tenant_id,
            robot_id=robot_id,
            message_type=MessageType.ACTION_RESULT,
            source=MessageSource.SIMULATOR,
            payload={
                "action_id": action_id,
                "status": status,
                "result": dict(result),
            },
        )

        self._validate(message)
        return message

    # --------------------------------------------------------
    # BATCH ENCODING
    # --------------------------------------------------------

    def encode_batch(
        self,
        messages: list[RobotMessage],
    ) -> list[str]:
        if (
            len(messages)
            > self.config.maximum_batch_size
        ):
            raise MessageValidationError(
                "Batch exceeds maximum size"
            )

        # Validate entire batch before encoding.
        for message in messages:
            self._validate(message)

        return [
            self.encode(message)
            for message in messages
        ]

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            events = [
                event
                for event in self.audit_events
                if (
                    tenant_id is None
                    or event.tenant_id == tenant_id
                )
            ]

            return {
                "total_conversions": len(events),
                "encoded": sum(
                    event.operation == "encode"
                    for event in events
                ),
                "decoded": sum(
                    event.operation == "decode"
                    for event in events
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
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": (
                    utc_now().isoformat()
                ),
                "tenant_id": tenant_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "audit_events": [
                    event.to_dict()
                    for event in self.audit_events
                    if event.tenant_id == tenant_id
                ],
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
    converter = RoboticsMessageConverter()

    navigation = converter.navigation_message(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        target_x=4.0,
        target_y=3.0,
        target_z=0.0,
        speed_mps=0.5,
    )

    encoded = converter.encode(
        navigation
    )

    print("Encoded navigation message:")
    print(encoded)

    decoded = converter.decode(
        encoded
    )

    print("\nDecoded navigation message:")
    print(
        json.dumps(
            decoded.to_dict(),
            indent=2,
        )
    )

    telemetry = converter.telemetry_message(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        position={
            "x": 1.0,
            "y": 2.0,
            "z": 0.0,
        },
        speed_mps=0.4,
        battery_percent=82.0,
        temperature_c=39.0,
    )

    print("\nTelemetry:")
    print(
        converter.encode(
            telemetry
        )
    )

    converter.export_json(
        "data/robotics/message_converter_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

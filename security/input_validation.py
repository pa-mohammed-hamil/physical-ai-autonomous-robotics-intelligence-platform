"""Input Validation module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/security/input_validation.py

Python 3.11+
Standard library only.

Features:
    - Strict API payload validation
    - Robot and resource identifier validation
    - Numeric range and finite-value checks
    - Telemetry validation
    - Navigation goal validation
    - Simulation configuration validation
    - Task request validation
    - AI tool argument allowlisting
    - JSON depth and size limits
    - Uploaded filename validation
    - Safe upload path resolution
    - String normalization
    - Rejection of unknown fields
    - Structured validation errors

Security:
    Validation is deny-by-default.
    No arbitrary Python execution.
    No physical robot command execution.
    No automatic removal of suspicious input.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class InputValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        field_name: str | None = None,
        code: str = "invalid_input",
    ) -> None:
        super().__init__(message)
        self.field_name = field_name
        self.code = code

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "field": self.field_name,
            "message": str(self),
        }


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class InputValidationConfig:
    maximum_json_bytes: int = 1024 * 1024
    maximum_json_depth: int = 16
    maximum_collection_items: int = 10000

    maximum_identifier_length: int = 128
    maximum_text_length: int = 4096
    maximum_filename_length: int = 255

    maximum_robot_speed_mps: float = 2.0
    maximum_angular_speed_radps: float = 3.0

    maximum_world_coordinate_m: float = 10000.0
    maximum_simulation_duration_seconds: float = 86400.0

    allowed_upload_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset({
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".pdf",
            ".txt",
            ".csv",
            ".json",
            ".jsonl",
            ".docx",
            ".xlsx",
            ".zip",
            ".onnx",
        })
    )

    def __post_init__(self) -> None:
        for name in (
            "maximum_json_bytes",
            "maximum_json_depth",
            "maximum_collection_items",
            "maximum_identifier_length",
            "maximum_text_length",
            "maximum_filename_length",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise InputValidationError(
                    f"{name} must be a positive integer",
                    field_name=name,
                )

        for name in (
            "maximum_robot_speed_mps",
            "maximum_angular_speed_radps",
            "maximum_world_coordinate_m",
            "maximum_simulation_duration_seconds",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise InputValidationError(
                    f"{name} must be finite and positive",
                    field_name=name,
                )


# ============================================================
# VALIDATION PRIMITIVES
# ============================================================

class ValidationPrimitives:

    IDENTIFIER_PATTERN = re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
    )

    @staticmethod
    def text(
        value: Any,
        *,
        field_name: str,
        maximum_length: int = 4096,
        allow_empty: bool = False,
    ) -> str:
        if not isinstance(value, str):
            raise InputValidationError(
                "Expected a string",
                field_name=field_name,
                code="invalid_type",
            )

        normalized = unicodedata.normalize(
            "NFC",
            value,
        ).strip()

        if not normalized and not allow_empty:
            raise InputValidationError(
                "Value cannot be empty",
                field_name=field_name,
                code="empty_value",
            )

        if len(normalized) > maximum_length:
            raise InputValidationError(
                "Text exceeds maximum length",
                field_name=field_name,
                code="text_too_long",
            )

        if any(
            unicodedata.category(char) == "Cc"
            for char in normalized
        ):
            raise InputValidationError(
                "Control characters are not allowed",
                field_name=field_name,
                code="invalid_character",
            )

        return normalized

    @classmethod
    def identifier(
        cls,
        value: Any,
        *,
        field_name: str,
        maximum_length: int = 128,
    ) -> str:
        result = cls.text(
            value,
            field_name=field_name,
            maximum_length=maximum_length,
        )

        if not cls.IDENTIFIER_PATTERN.fullmatch(
            result
        ):
            raise InputValidationError(
                "Identifier may contain only letters, "
                "numbers, underscores and hyphens",
                field_name=field_name,
                code="invalid_identifier",
            )

        return result

    @staticmethod
    def number(
        value: Any,
        *,
        field_name: str,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise InputValidationError(
                "Expected a finite number",
                field_name=field_name,
                code="invalid_number",
            )

        result = float(value)

        if minimum is not None and result < minimum:
            raise InputValidationError(
                f"Must be at least {minimum}",
                field_name=field_name,
                code="below_minimum",
            )

        if maximum is not None and result > maximum:
            raise InputValidationError(
                f"Must be at most {maximum}",
                field_name=field_name,
                code="above_maximum",
            )

        return result

    @staticmethod
    def integer(
        value: Any,
        *,
        field_name: str,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise InputValidationError(
                "Expected an integer",
                field_name=field_name,
                code="invalid_integer",
            )

        if minimum is not None and value < minimum:
            raise InputValidationError(
                "Integer below minimum",
                field_name=field_name,
            )

        if maximum is not None and value > maximum:
            raise InputValidationError(
                "Integer above maximum",
                field_name=field_name,
            )

        return value

    @staticmethod
    def boolean(
        value: Any,
        *,
        field_name: str,
    ) -> bool:
        if not isinstance(value, bool):
            raise InputValidationError(
                "Expected a boolean",
                field_name=field_name,
                code="invalid_boolean",
            )

        return value

    @staticmethod
    def choice(
        value: Any,
        *,
        field_name: str,
        allowed: set[str] | frozenset[str],
    ) -> str:
        if (
            not isinstance(value, str)
            or value not in allowed
        ):
            raise InputValidationError(
                f"Allowed values: {sorted(allowed)}",
                field_name=field_name,
                code="invalid_choice",
            )

        return value


# ============================================================
# JSON VALIDATOR
# ============================================================

class JSONInputValidator:

    def __init__(
        self,
        config: InputValidationConfig,
    ) -> None:
        self.config = config

    def _check_structure(
        self,
        value: Any,
        *,
        depth: int = 0,
    ) -> None:
        if depth > self.config.maximum_json_depth:
            raise InputValidationError(
                "JSON nesting depth exceeded",
                code="json_depth_exceeded",
            )

        if isinstance(value, dict):
            if (
                len(value)
                > self.config.maximum_collection_items
            ):
                raise InputValidationError(
                    "JSON object has too many fields",
                    code="json_collection_too_large",
                )

            for key, item in value.items():
                if not isinstance(key, str):
                    raise InputValidationError(
                        "JSON object keys must be strings",
                        code="invalid_json_key",
                    )

                self._check_structure(
                    item,
                    depth=depth + 1,
                )

        elif isinstance(value, list):
            if (
                len(value)
                > self.config.maximum_collection_items
            ):
                raise InputValidationError(
                    "JSON array is too large",
                    code="json_collection_too_large",
                )

            for item in value:
                self._check_structure(
                    item,
                    depth=depth + 1,
                )

        elif isinstance(value, float):
            if not math.isfinite(value):
                raise InputValidationError(
                    "Non-finite JSON number",
                    code="invalid_json_number",
                )

        elif (
            value is not None
            and not isinstance(
                value,
                (str, int, bool),
            )
        ):
            raise InputValidationError(
                "Unsupported JSON value type",
                code="invalid_json_type",
            )

    def validate(
        self,
        value: Any,
    ) -> Any:
        self._check_structure(value)

        try:
            encoded = json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                "Invalid JSON-compatible value",
                code="invalid_json",
            ) from exc

        if (
            len(encoded)
            > self.config.maximum_json_bytes
        ):
            raise InputValidationError(
                "JSON payload exceeds size limit",
                code="json_too_large",
            )

        return json.loads(encoded)

    def parse_bytes(
        self,
        raw: bytes,
    ) -> Any:
        if not isinstance(raw, bytes):
            raise InputValidationError(
                "Expected JSON bytes",
                code="invalid_type",
            )

        if (
            len(raw)
            > self.config.maximum_json_bytes
        ):
            raise InputValidationError(
                "JSON request exceeds size limit",
                code="json_too_large",
            )

        def reject_constant(value: str) -> None:
            raise InputValidationError(
                f"Non-standard JSON constant: {value}",
                code="invalid_json_number",
            )

        def reject_duplicates(
            pairs: list[tuple[str, Any]],
        ) -> dict[str, Any]:
            result: dict[str, Any] = {}

            for key, value in pairs:
                if key in result:
                    raise InputValidationError(
                        f"Duplicate JSON key: {key}",
                        code="duplicate_json_key",
                    )

                result[key] = value

            return result

        try:
            parsed = json.loads(
                raw.decode("utf-8"),
                parse_constant=reject_constant,
                object_pairs_hook=reject_duplicates,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise InputValidationError(
                "Malformed JSON request",
                code="invalid_json",
            ) from exc

        return self.validate(parsed)


# ============================================================
# SCHEMA VALIDATOR
# ============================================================

class ObjectSchemaValidator:

    @staticmethod
    def validate_fields(
        payload: Mapping[str, Any],
        *,
        required: set[str],
        optional: set[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise InputValidationError(
                "Expected a JSON object",
                code="invalid_object",
            )

        optional = optional or set()

        keys = set(payload)

        missing = required - keys
        unknown = keys - required - optional

        if missing:
            raise InputValidationError(
                f"Missing fields: {sorted(missing)}",
                code="missing_fields",
            )

        if unknown:
            raise InputValidationError(
                f"Unknown fields: {sorted(unknown)}",
                code="unknown_fields",
            )

        return dict(payload)


# ============================================================
# ROBOT INPUT VALIDATOR
# ============================================================

class RobotInputValidator:

    ROBOT_TYPES = {
        "mobile",
        "manipulator",
        "humanoid",
        "industrial",
        "warehouse",
        "service",
        "medical",
        "agricultural",
        "inspection",
        "drone",
        "other",
    }

    def __init__(
        self,
        config: InputValidationConfig,
    ) -> None:
        self.config = config

    def validate_registration(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        data = ObjectSchemaValidator.validate_fields(
            payload,
            required={
                "robot_id",
                "name",
                "robot_type",
            },
            optional={
                "manufacturer",
                "model",
                "serial_number",
                "capabilities",
            },
        )

        result = {
            "robot_id": ValidationPrimitives.identifier(
                data["robot_id"],
                field_name="robot_id",
                maximum_length=(
                    self.config.maximum_identifier_length
                ),
            ),
            "name": ValidationPrimitives.text(
                data["name"],
                field_name="name",
                maximum_length=256,
            ),
            "robot_type": ValidationPrimitives.choice(
                data["robot_type"],
                field_name="robot_type",
                allowed=self.ROBOT_TYPES,
            ),
        }

        for name in (
            "manufacturer",
            "model",
            "serial_number",
        ):
            if name in data:
                result[name] = ValidationPrimitives.text(
                    data[name],
                    field_name=name,
                    maximum_length=256,
                )

        if "capabilities" in data:
            capabilities = data["capabilities"]

            if (
                not isinstance(capabilities, list)
                or len(capabilities) > 100
            ):
                raise InputValidationError(
                    "capabilities must be a list of "
                    "at most 100 entries",
                    field_name="capabilities",
                )

            validated = [
                ValidationPrimitives.identifier(
                    capability,
                    field_name="capabilities",
                    maximum_length=64,
                )
                for capability in capabilities
            ]

            if len(set(validated)) != len(validated):
                raise InputValidationError(
                    "Duplicate capabilities",
                    field_name="capabilities",
                )

            result["capabilities"] = validated

        return result


# ============================================================
# TELEMETRY INPUT VALIDATOR
# ============================================================

class TelemetryInputValidator:

    def __init__(
        self,
        config: InputValidationConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        data = ObjectSchemaValidator.validate_fields(
            payload,
            required={
                "robot_id",
                "battery_soc_percent",
                "motor_temperature_c",
                "vibration_mm_s",
            },
            optional={
                "health_score",
                "x_m",
                "y_m",
                "heading_rad",
                "linear_velocity_mps",
                "angular_velocity_radps",
                "sensor_available",
                "network_connected",
            },
        )

        result = {
            "robot_id": ValidationPrimitives.identifier(
                data["robot_id"],
                field_name="robot_id",
            ),
            "battery_soc_percent": (
                ValidationPrimitives.number(
                    data["battery_soc_percent"],
                    field_name="battery_soc_percent",
                    minimum=0,
                    maximum=100,
                )
            ),
            "motor_temperature_c": (
                ValidationPrimitives.number(
                    data["motor_temperature_c"],
                    field_name="motor_temperature_c",
                    minimum=-50,
                    maximum=250,
                )
            ),
            "vibration_mm_s": (
                ValidationPrimitives.number(
                    data["vibration_mm_s"],
                    field_name="vibration_mm_s",
                    minimum=0,
                    maximum=1000,
                )
            ),
        }

        if "health_score" in data:
            result["health_score"] = (
                ValidationPrimitives.number(
                    data["health_score"],
                    field_name="health_score",
                    minimum=0,
                    maximum=100,
                )
            )

        coordinate_limit = (
            self.config.maximum_world_coordinate_m
        )

        for name in ("x_m", "y_m"):
            if name in data:
                result[name] = ValidationPrimitives.number(
                    data[name],
                    field_name=name,
                    minimum=-coordinate_limit,
                    maximum=coordinate_limit,
                )

        if "heading_rad" in data:
            result["heading_rad"] = (
                ValidationPrimitives.number(
                    data["heading_rad"],
                    field_name="heading_rad",
                    minimum=-math.pi,
                    maximum=math.pi,
                )
            )

        if "linear_velocity_mps" in data:
            speed = (
                self.config.maximum_robot_speed_mps
            )

            result["linear_velocity_mps"] = (
                ValidationPrimitives.number(
                    data["linear_velocity_mps"],
                    field_name="linear_velocity_mps",
                    minimum=-speed,
                    maximum=speed,
                )
            )

        if "angular_velocity_radps" in data:
            speed = (
                self.config.maximum_angular_speed_radps
            )

            result["angular_velocity_radps"] = (
                ValidationPrimitives.number(
                    data["angular_velocity_radps"],
                    field_name="angular_velocity_radps",
                    minimum=-speed,
                    maximum=speed,
                )
            )

        for name in (
            "sensor_available",
            "network_connected",
        ):
            if name in data:
                result[name] = ValidationPrimitives.boolean(
                    data[name],
                    field_name=name,
                )

        return result


# ============================================================
# NAVIGATION INPUT VALIDATOR
# ============================================================

class NavigationInputValidator:

    def __init__(
        self,
        config: InputValidationConfig,
    ) -> None:
        self.config = config

    def validate_goal(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        data = ObjectSchemaValidator.validate_fields(
            payload,
            required={
                "robot_id",
                "goal_x",
                "goal_y",
            },
            optional={
                "maximum_speed_mps",
            },
        )

        limit = (
            self.config.maximum_world_coordinate_m
        )

        result = {
            "robot_id": ValidationPrimitives.identifier(
                data["robot_id"],
                field_name="robot_id",
            ),
            "goal_x": ValidationPrimitives.number(
                data["goal_x"],
                field_name="goal_x",
                minimum=-limit,
                maximum=limit,
            ),
            "goal_y": ValidationPrimitives.number(
                data["goal_y"],
                field_name="goal_y",
                minimum=-limit,
                maximum=limit,
            ),
        }

        if "maximum_speed_mps" in data:
            result["maximum_speed_mps"] = (
                ValidationPrimitives.number(
                    data["maximum_speed_mps"],
                    field_name="maximum_speed_mps",
                    minimum=0,
                    maximum=(
                        self.config.maximum_robot_speed_mps
                    ),
                )
            )

        return result


# ============================================================
# SIMULATION INPUT VALIDATOR
# ============================================================

class SimulationInputValidator:

    def __init__(
        self,
        config: InputValidationConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        data = ObjectSchemaValidator.validate_fields(
            payload,
            required={
                "name",
                "duration_seconds",
                "timestep_seconds",
            },
            optional={
                "world_width_m",
                "world_height_m",
                "seed",
            },
        )

        result = {
            "name": ValidationPrimitives.text(
                data["name"],
                field_name="name",
                maximum_length=256,
            ),
            "duration_seconds": (
                ValidationPrimitives.number(
                    data["duration_seconds"],
                    field_name="duration_seconds",
                    minimum=0.001,
                    maximum=(
                        self.config
                        .maximum_simulation_duration_seconds
                    ),
                )
            ),
            "timestep_seconds": (
                ValidationPrimitives.number(
                    data["timestep_seconds"],
                    field_name="timestep_seconds",
                    minimum=0.001,
                    maximum=1.0,
                )
            ),
        }

        for name in (
            "world_width_m",
            "world_height_m",
        ):
            if name in data:
                result[name] = ValidationPrimitives.number(
                    data[name],
                    field_name=name,
                    minimum=0.1,
                    maximum=(
                        self.config.maximum_world_coordinate_m
                    ),
                )

        if "seed" in data:
            result["seed"] = ValidationPrimitives.integer(
                data["seed"],
                field_name="seed",
                minimum=0,
                maximum=2**32 - 1,
            )

        if (
            result["timestep_seconds"]
            > result["duration_seconds"]
        ):
            raise InputValidationError(
                "Simulation timestep exceeds duration",
                field_name="timestep_seconds",
            )

        return result


# ============================================================
# TASK INPUT VALIDATOR
# ============================================================

class TaskInputValidator:

    TASK_TYPES = {
        "navigation",
        "manipulation",
        "pick_and_place",
        "inspection",
        "transport",
        "maintenance",
        "repair",
        "charging",
        "perception",
        "data_collection",
        "simulation",
        "other",
    }

    PRIORITIES = {
        "low",
        "normal",
        "high",
        "critical",
    }

    def __init__(
        self,
        config: InputValidationConfig,
    ) -> None:
        self.config = config
        self.json_validator = JSONInputValidator(
            config
        )

    def validate(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        data = ObjectSchemaValidator.validate_fields(
            payload,
            required={
                "name",
                "task_type",
            },
            optional={
                "priority",
                "robot_id",
                "payload",
                "timeout_seconds",
            },
        )

        result = {
            "name": ValidationPrimitives.text(
                data["name"],
                field_name="name",
                maximum_length=256,
            ),
            "task_type": ValidationPrimitives.choice(
                data["task_type"],
                field_name="task_type",
                allowed=self.TASK_TYPES,
            ),
            "priority": ValidationPrimitives.choice(
                data.get("priority", "normal"),
                field_name="priority",
                allowed=self.PRIORITIES,
            ),
        }

        if "robot_id" in data:
            result["robot_id"] = (
                ValidationPrimitives.identifier(
                    data["robot_id"],
                    field_name="robot_id",
                )
            )

        if "timeout_seconds" in data:
            result["timeout_seconds"] = (
                ValidationPrimitives.number(
                    data["timeout_seconds"],
                    field_name="timeout_seconds",
                    minimum=0.001,
                    maximum=86400,
                )
            )

        if "payload" in data:
            if not isinstance(data["payload"], dict):
                raise InputValidationError(
                    "Task payload must be an object",
                    field_name="payload",
                )

            result["payload"] = (
                self.json_validator.validate(
                    data["payload"]
                )
            )

        return result


# ============================================================
# FILE INPUT VALIDATOR
# ============================================================

class FileInputValidator:

    def __init__(
        self,
        config: InputValidationConfig,
    ) -> None:
        self.config = config

    def validate_filename(
        self,
        filename: str,
    ) -> str:
        name = ValidationPrimitives.text(
            filename,
            field_name="filename",
            maximum_length=(
                self.config.maximum_filename_length
            ),
        )

        if name in {".", ".."}:
            raise InputValidationError(
                "Invalid filename",
                field_name="filename",
            )

        if (
            "/" in name
            or "\\" in name
            or ":" in name
            or name.startswith(".")
        ):
            raise InputValidationError(
                "Filename contains forbidden path syntax",
                field_name="filename",
            )

        if name.endswith((" ", ".")):
            raise InputValidationError(
                "Filename has unsafe trailing characters",
                field_name="filename",
            )

        if any(
            char in '<>"|?*'
            for char in name
        ):
            raise InputValidationError(
                "Filename contains forbidden characters",
                field_name="filename",
            )

        stem = Path(name).stem.upper()

        reserved = {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }

        if stem in reserved:
            raise InputValidationError(
                "Reserved filename",
                field_name="filename",
            )

        extension = Path(name).suffix.lower()

        if (
            extension
            not in self.config.allowed_upload_extensions
        ):
            raise InputValidationError(
                "File extension not allowed",
                field_name="filename",
                code="invalid_extension",
            )

        return name

    def resolve_upload_path(
        self,
        upload_directory: str | Path,
        filename: str,
    ) -> Path:
        validated = self.validate_filename(
            filename
        )

        root = Path(
            upload_directory
        ).resolve(strict=True)

        if not root.is_dir():
            raise InputValidationError(
                "Upload root is not a directory",
                field_name="upload_directory",
            )

        destination = (
            root / validated
        ).resolve(strict=False)

        if not destination.is_relative_to(root):
            raise InputValidationError(
                "Upload path escapes permitted directory",
                field_name="filename",
                code="path_traversal",
            )

        if destination.exists() or destination.is_symlink():
            raise InputValidationError(
                "Upload destination already exists",
                field_name="filename",
                code="file_exists",
            )

        return destination


# ============================================================
# AI AGENT TOOL ARGUMENT VALIDATOR
# ============================================================

class AgentToolInputValidator:
    """
    Validate allowlisted agent tool arguments.

    Validation does not authorize a tool call.
    Call command_authorization.py separately.
    """

    def __init__(
        self,
        config: InputValidationConfig,
    ) -> None:
        self.config = config

        self.navigation = NavigationInputValidator(
            config
        )

        self.simulation = SimulationInputValidator(
            config
        )

        self._validators: dict[
            str,
            Callable[
                [Mapping[str, Any]],
                dict[str, Any],
            ]
        ] = {
            "navigation.plan": (
                self.navigation.validate_goal
            ),
            "simulation.create": (
                self.simulation.validate
            ),
            "robot.get_status": (
                self._validate_robot_status
            ),
        }

    @staticmethod
    def _validate_robot_status(
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        data = ObjectSchemaValidator.validate_fields(
            payload,
            required={"robot_id"},
        )

        return {
            "robot_id": ValidationPrimitives.identifier(
                data["robot_id"],
                field_name="robot_id",
            )
        }

    def validate(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        validator = self._validators.get(
            tool_name
        )

        if validator is None:
            raise InputValidationError(
                "Tool is not allowlisted",
                field_name="tool_name",
                code="tool_not_allowed",
            )

        return validator(arguments)


# ============================================================
# CENTRAL VALIDATION SERVICE
# ============================================================

class RoboticsInputValidationService:

    def __init__(
        self,
        config: InputValidationConfig | None = None,
    ) -> None:
        self.config = (
            config or InputValidationConfig()
        )

        self.json = JSONInputValidator(
            self.config
        )

        self.robots = RobotInputValidator(
            self.config
        )

        self.telemetry = TelemetryInputValidator(
            self.config
        )

        self.navigation = NavigationInputValidator(
            self.config
        )

        self.simulation = SimulationInputValidator(
            self.config
        )

        self.tasks = TaskInputValidator(
            self.config
        )

        self.files = FileInputValidator(
            self.config
        )

        self.agent_tools = AgentToolInputValidator(
            self.config
        )

    def validate_robot_registration(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.robots.validate_registration(
            payload
        )

    def validate_telemetry(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.telemetry.validate(
            payload
        )

    def validate_navigation_goal(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.navigation.validate_goal(
            payload
        )

    def validate_simulation(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.simulation.validate(
            payload
        )

    def validate_task(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.tasks.validate(
            payload
        )

    def validate_tool_arguments(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.agent_tools.validate(
            tool_name,
            arguments,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsInputValidationService()

    robot = service.validate_robot_registration({
        "robot_id": "robot_001",
        "name": "Warehouse Robot",
        "robot_type": "mobile",
        "capabilities": [
            "navigation",
            "inspection",
        ],
    })

    telemetry = service.validate_telemetry({
        "robot_id": "robot_001",
        "battery_soc_percent": 87.5,
        "motor_temperature_c": 45.2,
        "vibration_mm_s": 2.1,
        "health_score": 94.0,
        "x_m": 3.0,
        "y_m": 4.0,
        "sensor_available": True,
        "network_connected": True,
    })

    navigation = service.validate_navigation_goal({
        "robot_id": "robot_001",
        "goal_x": 10.0,
        "goal_y": 6.0,
        "maximum_speed_mps": 0.8,
    })

    simulation = service.validate_simulation({
        "name": "Warehouse Navigation Test",
        "duration_seconds": 120.0,
        "timestep_seconds": 0.05,
        "world_width_m": 30.0,
        "world_height_m": 30.0,
        "seed": 42,
    })

    task = service.validate_task({
        "name": "Inspect warehouse shelf",
        "task_type": "inspection",
        "priority": "high",
        "robot_id": "robot_001",
        "timeout_seconds": 600,
        "payload": {
            "shelf_id": "shelf_001",
        },
    })

    agent_arguments = service.validate_tool_arguments(
        "navigation.plan",
        {
            "robot_id": "robot_001",
            "goal_x": 10.0,
            "goal_y": 6.0,
        },
    )

    results = {
        "robot": robot,
        "telemetry": telemetry,
        "navigation": navigation,
        "simulation": simulation,
        "task": task,
        "agent_arguments": agent_arguments,
    }

    print(
        json.dumps(
            results,
            indent=2,
            allow_nan=False,
        )
    )

    # Demonstrate rejection of invalid telemetry.
    try:
        service.validate_telemetry({
            "robot_id": "robot_001",
            "battery_soc_percent": 150,
            "motor_temperature_c": 45,
            "vibration_mm_s": 2,
        })
    except InputValidationError as exc:
        print(
            "Rejected invalid input:",
            exc.to_dict(),
        )


if __name__ == "__main__":
    main()

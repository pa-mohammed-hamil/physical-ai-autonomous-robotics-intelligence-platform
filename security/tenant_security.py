"""Tool Permissions module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/security/secrets.py

Python 3.11+
Standard library only.

Features:
    - Environment variable secrets
    - Docker/Kubernetes mounted secret files
    - Secret source priority
    - Secret validation
    - Secret redaction
    - Secret caching with expiration
    - Secret refresh and rotation support
    - Secret access audit metadata
    - No plaintext secret values in audit logs
    - Thread-safe operations
    - Fail-closed missing-secret handling

Production notes:
    - Use a managed secrets service or KMS.
    - Do not commit .env files or secrets.
    - Do not expose secrets in API responses.
    - Avoid printing secret values.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import stat
import threading
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class SecretManagementError(Exception):
    pass


class SecretNotFoundError(SecretManagementError):
    pass


class SecretValidationError(SecretManagementError):
    pass


class SecretAccessError(SecretManagementError):
    pass


# ============================================================
# ENUMS
# ============================================================

class SecretSource(str, Enum):
    ENVIRONMENT = "environment"
    FILE = "file"


class SecretCategory(str, Enum):
    LLM = "llm"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    MODEL_INTEGRITY = "model_integrity"
    MQTT = "mqtt"
    ROS2 = "ros2"
    CLOUD = "cloud"
    INTERNAL = "internal"


class SecretAuditAction(str, Enum):
    ACCESSED = "accessed"
    REFRESHED = "refreshed"
    MISSING = "missing"
    INVALID = "invalid"
    CACHE_CLEARED = "cache_cleared"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_secret_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not re.fullmatch(
            r"[A-Z][A-Z0-9_]{0,127}",
            name,
        )
    ):
        raise SecretValidationError(
            "Secret name must use uppercase "
            "letters, digits and underscores"
        )

    return name


def redact_secret(value: str) -> str:
    """
    Never reveal any part of a secret.
    """
    return "[REDACTED]"


def constant_time_equal(
    first: str | bytes,
    second: str | bytes,
) -> bool:
    if type(first) is not type(second):
        return False

    return hmac.compare_digest(
        first,
        second,
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SecretManagerConfig:
    secrets_directory: str | None = "/run/secrets"

    allow_environment: bool = True
    allow_secret_files: bool = True

    prefer_secret_files: bool = True

    cache_ttl_seconds: float = 60.0

    maximum_secret_bytes: int = 65536

    maximum_audit_events: int = 10000

    reject_empty_secrets: bool = True

    enforce_private_file_permissions: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.cache_ttl_seconds, bool)
            or not isinstance(
                self.cache_ttl_seconds,
                (int, float),
            )
            or not 0 <= self.cache_ttl_seconds <= 3600
        ):
            raise SecretValidationError(
                "Invalid cache TTL"
            )

        for name in (
            "maximum_secret_bytes",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise SecretValidationError(
                    f"{name} must be positive"
                )

        if not (
            self.allow_environment
            or self.allow_secret_files
        ):
            raise SecretValidationError(
                "At least one secret source is required"
            )


# ============================================================
# SECRET DEFINITION
# ============================================================

@dataclass(frozen=True)
class SecretDefinition:
    name: str
    category: SecretCategory

    required: bool = True

    minimum_length: int = 1
    maximum_length: int = 65536

    def __post_init__(self) -> None:
        validate_secret_name(
            self.name
        )

        object.__setattr__(
            self,
            "category",
            SecretCategory(self.category),
        )

        if (
            self.minimum_length < 0
            or self.maximum_length < self.minimum_length
        ):
            raise SecretValidationError(
                "Invalid secret length requirements"
            )


# ============================================================
# SECRET VALUE
# ============================================================

@dataclass(frozen=True, repr=False)
class SecretValue:
    name: str
    value: str
    source: SecretSource
    loaded_at: str

    def __repr__(self) -> str:
        return (
            f"SecretValue(name={self.name!r}, "
            f"value='[REDACTED]', "
            f"source={self.source.value!r})"
        )

    def __str__(self) -> str:
        return "[REDACTED]"

    def reveal(self) -> str:
        """
        Explicit access to the underlying secret.

        Only trusted application code should call this.
        """
        return self.value


# ============================================================
# SECRET AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class SecretAuditEvent:
    event_id: str
    timestamp: str
    secret_name: str
    action: SecretAuditAction

    source: SecretSource | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "secret_name": self.secret_name,
            "action": self.action.value,
            "source": (
                self.source.value
                if self.source is not None
                else None
            ),
        }


# ============================================================
# CACHE ENTRY
# ============================================================

@dataclass(repr=False)
class SecretCacheEntry:
    secret: SecretValue
    expires_at_monotonic: float

    def is_valid(self) -> bool:
        return (
            time.monotonic()
            < self.expires_at_monotonic
        )


# ============================================================
# ENVIRONMENT SECRET PROVIDER
# ============================================================

class EnvironmentSecretProvider:

    def get(
        self,
        name: str,
    ) -> str | None:
        validate_secret_name(name)
        return os.environ.get(name)


# ============================================================
# FILE SECRET PROVIDER
# ============================================================

class FileSecretProvider:

    def __init__(
        self,
        config: SecretManagerConfig,
    ) -> None:
        self.config = config

    def get(
        self,
        name: str,
    ) -> str | None:
        validate_secret_name(name)

        if not self.config.secrets_directory:
            return None

        root = Path(
            self.config.secrets_directory
        )

        if not root.exists():
            return None

        if not root.is_dir():
            raise SecretAccessError(
                "Secrets path is not a directory"
            )

        path = root / name

        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return None

        if not stat.S_ISREG(metadata.st_mode):
            raise SecretAccessError(
                "Secret must be a regular file"
            )

        if (
            self.config.enforce_private_file_permissions
            and metadata.st_mode
            & (stat.S_IRWXG | stat.S_IRWXO)
        ):
            raise SecretAccessError(
                "Secret file permissions are too broad"
            )

        if (
            metadata.st_size
            > self.config.maximum_secret_bytes
        ):
            raise SecretValidationError(
                "Secret file exceeds size limit"
            )

        try:
            # O_NOFOLLOW prevents final-component
            # symlink traversal on supported systems.
            flags = os.O_RDONLY

            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW

            fd = os.open(
                path,
                flags,
            )

            with os.fdopen(
                fd,
                "rb",
            ) as handle:
                actual = os.fstat(
                    handle.fileno()
                )

                if not stat.S_ISREG(
                    actual.st_mode
                ):
                    raise SecretAccessError(
                        "Secret is not a regular file"
                    )

                if (
                    self.config.enforce_private_file_permissions
                    and actual.st_mode
                    & (stat.S_IRWXG | stat.S_IRWXO)
                ):
                    raise SecretAccessError(
                        "Secret file permissions are too broad"
                    )

                data = handle.read(
                    self.config.maximum_secret_bytes + 1
                )

            if (
                len(data)
                > self.config.maximum_secret_bytes
            ):
                raise SecretValidationError(
                    "Secret exceeds size limit"
                )

            # Mounted secrets frequently contain
            # a final newline.
            return data.decode("utf-8").rstrip(
                "\r\n"
            )

        except UnicodeDecodeError as exc:
            raise SecretValidationError(
                "Secret file is not UTF-8"
            ) from exc

        except OSError as exc:
            raise SecretAccessError(
                "Unable to read secret file"
            ) from exc


# ============================================================
# CENTRAL SECRET MANAGER
# ============================================================

class RoboticsSecretManager:

    def __init__(
        self,
        config: SecretManagerConfig | None = None,
    ) -> None:
        self.config = (
            config or SecretManagerConfig()
        )

        self.environment_provider = (
            EnvironmentSecretProvider()
        )

        self.file_provider = (
            FileSecretProvider(self.config)
        )

        self.definitions: dict[
            str,
            SecretDefinition
        ] = {}

        self._cache: dict[
            str,
            SecretCacheEntry
        ] = {}

        self.audit_events: list[
            SecretAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        name: str,
        action: SecretAuditAction,
        source: SecretSource | None = None,
    ) -> None:
        self.audit_events.append(
            SecretAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                secret_name=name,
                action=action,
                source=source,
            )
        )

        if (
            len(self.audit_events)
            > self.config.maximum_audit_events
        ):
            self.audit_events.pop(0)

    # --------------------------------------------------------
    # REGISTER SECRET
    # --------------------------------------------------------

    def register(
        self,
        definition: SecretDefinition,
    ) -> None:
        with self._lock:
            self.definitions[
                definition.name
            ] = definition

            self._cache.pop(
                definition.name,
                None,
            )

    # --------------------------------------------------------
    # LOAD FROM PROVIDERS
    # --------------------------------------------------------

    def _load(
        self,
        name: str,
    ) -> tuple[str, SecretSource] | None:
        providers = []

        if self.config.prefer_secret_files:
            if self.config.allow_secret_files:
                providers.append((
                    SecretSource.FILE,
                    self.file_provider,
                ))

            if self.config.allow_environment:
                providers.append((
                    SecretSource.ENVIRONMENT,
                    self.environment_provider,
                ))

        else:
            if self.config.allow_environment:
                providers.append((
                    SecretSource.ENVIRONMENT,
                    self.environment_provider,
                ))

            if self.config.allow_secret_files:
                providers.append((
                    SecretSource.FILE,
                    self.file_provider,
                ))

        for source, provider in providers:
            value = provider.get(name)

            if value is not None:
                return value, source

        return None

    # --------------------------------------------------------
    # VALIDATE VALUE
    # --------------------------------------------------------

    def _validate(
        self,
        definition: SecretDefinition,
        value: str,
    ) -> None:
        if (
            self.config.reject_empty_secrets
            and not value
        ):
            raise SecretValidationError(
                "Secret value cannot be empty"
            )

        if (
            len(value)
            < definition.minimum_length
            or len(value)
            > definition.maximum_length
        ):
            raise SecretValidationError(
                "Secret does not satisfy length policy"
            )

        if (
            len(value.encode("utf-8"))
            > self.config.maximum_secret_bytes
        ):
            raise SecretValidationError(
                "Secret exceeds maximum byte size"
            )

    # --------------------------------------------------------
    # GET SECRET
    # --------------------------------------------------------

    def get(
        self,
        name: str,
        *,
        refresh: bool = False,
    ) -> SecretValue | None:
        validate_secret_name(name)

        with self._lock:
            definition = self.definitions.get(
                name
            )

            if definition is None:
                raise SecretValidationError(
                    "Secret is not registered"
                )

            cached = self._cache.get(
                name
            )

            if (
                not refresh
                and cached is not None
                and cached.is_valid()
            ):
                self._audit(
                    name,
                    SecretAuditAction.ACCESSED,
                    cached.secret.source,
                )

                return cached.secret

            loaded = self._load(name)

            if loaded is None:
                self._cache.pop(name, None)

                self._audit(
                    name,
                    SecretAuditAction.MISSING,
                )

                if definition.required:
                    raise SecretNotFoundError(
                        f"Required secret unavailable: {name}"
                    )

                return None

            value, source = loaded

            try:
                self._validate(
                    definition,
                    value,
                )
            except SecretValidationError:
                self._cache.pop(name, None)

                self._audit(
                    name,
                    SecretAuditAction.INVALID,
                    source,
                )

                raise

            secret = SecretValue(
                name=name,
                value=value,
                source=source,
                loaded_at=utc_now(),
            )

            if self.config.cache_ttl_seconds > 0:
                self._cache[name] = SecretCacheEntry(
                    secret=secret,
                    expires_at_monotonic=(
                        time.monotonic()
                        + self.config.cache_ttl_seconds
                    ),
                )
            else:
                self._cache.pop(name, None)

            self._audit(
                name,
                (
                    SecretAuditAction.REFRESHED
                    if refresh
                    else SecretAuditAction.ACCESSED
                ),
                source,
            )

            return secret

    # --------------------------------------------------------
    # REQUIRE SECRET
    # --------------------------------------------------------

    def require(
        self,
        name: str,
        *,
        refresh: bool = False,
    ) -> SecretValue:
        secret = self.get(
            name,
            refresh=refresh,
        )

        if secret is None:
            raise SecretNotFoundError(
                f"Secret unavailable: {name}"
            )

        return secret

    # --------------------------------------------------------
    # GET SECRET BYTES
    # --------------------------------------------------------

    def require_bytes(
        self,
        name: str,
        *,
        refresh: bool = False,
    ) -> bytes:
        return self.require(
            name,
            refresh=refresh,
        ).reveal().encode("utf-8")

    # --------------------------------------------------------
    # CLEAR CACHE
    # --------------------------------------------------------

    def clear_cache(
        self,
        name: str | None = None,
    ) -> None:
        with self._lock:
            if name is None:
                names = list(
                    self._cache
                )

                self._cache.clear()

                for secret_name in names:
                    self._audit(
                        secret_name,
                        SecretAuditAction.CACHE_CLEARED,
                    )

            else:
                validate_secret_name(name)

                self._cache.pop(
                    name,
                    None,
                )

                self._audit(
                    name,
                    SecretAuditAction.CACHE_CLEARED,
                )

    # --------------------------------------------------------
    # REFRESH ALL
    # --------------------------------------------------------

    def refresh_all(
        self,
    ) -> dict[str, bool]:
        results = {}

        with self._lock:
            names = list(
                self.definitions
            )

        for name in names:
            try:
                self.get(
                    name,
                    refresh=True,
                )

                results[name] = True

            except SecretManagementError:
                results[name] = False

        return results

    # --------------------------------------------------------
    # HEALTH
    # --------------------------------------------------------

    def health(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            return {
                "registered_secrets": len(
                    self.definitions
                ),
                "cached_secrets": sum(
                    entry.is_valid()
                    for entry in self._cache.values()
                ),
                "audit_events": len(
                    self.audit_events
                ),
                "environment_enabled": (
                    self.config.allow_environment
                ),
                "secret_files_enabled": (
                    self.config.allow_secret_files
                ),
            }

    # --------------------------------------------------------
    # EXPORT AUDIT
    # --------------------------------------------------------

    def export_audit_json(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now(),
                "health": self.health(),
                "events": [
                    event.to_dict()
                    for event in self.audit_events
                ],
            }

        output.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        return output


# ============================================================
# DEFAULT SECRET DEFINITIONS
# ============================================================

DEFAULT_SECRET_DEFINITIONS = (
    SecretDefinition(
        name="OPENAI_API_KEY",
        category=SecretCategory.LLM,
        required=False,
        minimum_length=10,
    ),
    SecretDefinition(
        name="GEMINI_API_KEY",
        category=SecretCategory.LLM,
        required=False,
        minimum_length=10,
    ),
    SecretDefinition(
        name="DATABASE_PASSWORD",
        category=SecretCategory.DATABASE,
        required=False,
        minimum_length=12,
    ),
    SecretDefinition(
        name="JWT_SIGNING_SECRET",
        category=SecretCategory.AUTHENTICATION,
        required=False,
        minimum_length=32,
    ),
    SecretDefinition(
        name="MODEL_INTEGRITY_HMAC_KEY",
        category=SecretCategory.MODEL_INTEGRITY,
        required=False,
        minimum_length=32,
    ),
    SecretDefinition(
        name="MQTT_PASSWORD",
        category=SecretCategory.MQTT,
        required=False,
        minimum_length=12,
    ),
    SecretDefinition(
        name="ROS2_SECURITY_KEY",
        category=SecretCategory.ROS2,
        required=False,
        minimum_length=32,
    ),
    SecretDefinition(
        name="AWS_SECRET_ACCESS_KEY",
        category=SecretCategory.CLOUD,
        required=False,
        minimum_length=20,
    ),
)


# ============================================================
# FACTORY
# ============================================================

def create_default_secret_manager(
    config: SecretManagerConfig | None = None,
) -> RoboticsSecretManager:
    manager = RoboticsSecretManager(
        config
    )

    for definition in DEFAULT_SECRET_DEFINITIONS:
        manager.register(
            definition
        )

    return manager


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    # Demo-only value. Never hardcode real credentials.
    demo_name = "ROBOTICS_DEMO_SECRET"

    manager = RoboticsSecretManager(
        SecretManagerConfig(
            allow_secret_files=False,
            cache_ttl_seconds=10,
        )
    )

    manager.register(
        SecretDefinition(
            name=demo_name,
            category=SecretCategory.INTERNAL,
            required=False,
            minimum_length=8,
        )
    )

    # This demonstration does not create or expose
    # a production secret.
    secret = manager.get(
        demo_name
    )

    print(
        "Demo secret available:",
        secret is not None,
    )

    print(
        "Secret manager health:",
        json.dumps(
            manager.health(),
            indent=2,
        ),
    )

    report = manager.export_audit_json(
        "data/security/secrets_audit.json"
    )

    print(
        "Audit report:",
        report,
    )


if __name__ == "__main__":
    main()

"""Audit Logging module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/security/audit_logging.py

Python 3.11+
Standard library only.

Features:
    - Structured security audit logging
    - Robot and fleet activity tracking
    - Authentication and authorization events
    - Task and simulation audit trails
    - Maintenance and repair events
    - Model lifecycle auditing
    - Append-only JSONL records
    - SHA-256 hash chaining
    - Optional HMAC authentication
    - Audit integrity verification
    - Sensitive-field redaction
    - Time-range filtering
    - JSON and CSV reporting
    - Thread-safe event recording

Limitations:
    Local files are not immutable.
    Hash chains alone cannot prevent a privileged
    attacker from rewriting the entire log.
    Production deployments need protected storage,
    centralized collection and external checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import re
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

class AuditLoggingError(Exception):
    pass


class AuditValidationError(AuditLoggingError):
    pass


class AuditIntegrityError(AuditLoggingError):
    pass


class AuditStorageError(AuditLoggingError):
    pass


# ============================================================
# ENUMS
# ============================================================

class AuditSeverity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditCategory(str, Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    API = "api"
    ROBOT = "robot"
    FLEET = "fleet"
    TASK = "task"
    TELEMETRY = "telemetry"
    NAVIGATION = "navigation"
    PERCEPTION = "perception"
    MANIPULATION = "manipulation"
    SIMULATION = "simulation"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    MODEL = "model"
    EVALUATION = "evaluation"
    SECURITY = "security"
    SYSTEM = "system"


class AuditOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    PENDING = "pending"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except (ValueError, AttributeError) as exc:
        raise AuditValidationError(
            "Invalid timestamp"
        ) from exc

    if result.tzinfo is None:
        raise AuditValidationError(
            "Timestamp must include timezone"
        )

    return result.astimezone(timezone.utc)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(data: str) -> str:
    return hashlib.sha256(
        data.encode("utf-8")
    ).hexdigest()


def require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditValidationError(
            f"{name} is required"
        )

    return value.strip()


# ============================================================
# REDACTION
# ============================================================

class AuditRedactor:
    """
    Redacts sensitive fields recursively.

    Redaction is based on key names and is not a
    substitute for data classification or DLP.
    """

    SENSITIVE_KEYS = {
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "authorization",
        "cookie",
        "set_cookie",
        "private_key",
        "client_secret",
        "credential",
        "credentials",
        "session_token",
    }

    @classmethod
    def _normalized_key(cls, key: str) -> str:
        return re.sub(
            r"[^a-z0-9]+",
            "_",
            key.lower(),
        ).strip("_")

    @classmethod
    def redact(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, Mapping):
            result = {}

            for key, item in value.items():
                normalized = cls._normalized_key(
                    str(key)
                )

                if (
                    normalized in cls.SENSITIVE_KEYS
                    or normalized.endswith("_password")
                    or normalized.endswith("_secret")
                    or normalized.endswith("_token")
                    or normalized.endswith("_api_key")
                ):
                    result[str(key)] = "[REDACTED]"
                else:
                    result[str(key)] = cls.redact(item)

            return result

        if isinstance(value, (list, tuple)):
            return [
                cls.redact(item)
                for item in value
            ]

        if isinstance(
            value,
            (str, int, float, bool),
        ) or value is None:
            return value

        raise AuditValidationError(
            "Unsupported audit detail type"
        )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class AuditLoggingConfig:
    log_directory: str = "data/security/audit"

    log_filename: str = "audit_events.jsonl"

    maximum_detail_bytes: int = 16384

    fsync_each_event: bool = True

    hmac_key: bytes | None = None

    def __post_init__(self) -> None:
        require_text(
            self.log_directory,
            "log_directory",
        )

        require_text(
            self.log_filename,
            "log_filename",
        )

        if (
            Path(self.log_filename).name
            != self.log_filename
        ):
            raise AuditValidationError(
                "log_filename must be a filename"
            )

        if (
            isinstance(self.maximum_detail_bytes, bool)
            or not isinstance(
                self.maximum_detail_bytes,
                int,
            )
            or self.maximum_detail_bytes < 1
        ):
            raise AuditValidationError(
                "maximum_detail_bytes must be positive"
            )

        if (
            self.hmac_key is not None
            and (
                not isinstance(self.hmac_key, bytes)
                or len(self.hmac_key) < 32
            )
        ):
            raise AuditValidationError(
                "hmac_key must contain at least 32 bytes"
            )


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    sequence: int

    timestamp: str

    category: AuditCategory
    action: str

    severity: AuditSeverity
    outcome: AuditOutcome

    actor_id: str | None = None
    actor_role: str | None = None

    resource_type: str | None = None
    resource_id: str | None = None

    request_id: str | None = None
    correlation_id: str | None = None

    source_ip: str | None = None

    details: dict[str, Any] = field(
        default_factory=dict
    )

    previous_hash: str = ""
    event_hash: str = ""

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "category": self.category.value,
            "action": self.action,
            "severity": self.severity.value,
            "outcome": self.outcome.value,
            "actor_id": self.actor_id,
            "actor_role": self.actor_role,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "source_ip": self.source_ip,
            "details": self.details,
            "previous_hash": self.previous_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.unsigned_dict()
        result["event_hash"] = self.event_hash
        return result


# ============================================================
# INTEGRITY ENGINE
# ============================================================

class AuditIntegrityEngine:

    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        hmac_key: bytes | None = None,
    ) -> None:
        self.hmac_key = hmac_key

    def calculate_hash(
        self,
        event_data: Mapping[str, Any],
    ) -> str:
        payload = canonical_json(
            dict(event_data)
        ).encode("utf-8")

        if self.hmac_key is not None:
            return hmac.new(
                self.hmac_key,
                payload,
                hashlib.sha256,
            ).hexdigest()

        return hashlib.sha256(
            payload
        ).hexdigest()

    def verify(
        self,
        records: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        previous_hash = self.GENESIS_HASH

        for expected_sequence, record in enumerate(
            records,
            start=1,
        ):
            try:
                sequence = record["sequence"]
                stored_previous = record["previous_hash"]
                stored_hash = record["event_hash"]

                if (
                    isinstance(sequence, bool)
                    or sequence != expected_sequence
                ):
                    raise AuditIntegrityError(
                        f"Sequence mismatch at record "
                        f"{expected_sequence}"
                    )

                if stored_previous != previous_hash:
                    raise AuditIntegrityError(
                        f"Chain mismatch at record "
                        f"{expected_sequence}"
                    )

                unsigned = dict(record)
                unsigned.pop("event_hash")

                calculated = self.calculate_hash(
                    unsigned
                )

                if not hmac.compare_digest(
                    calculated,
                    stored_hash,
                ):
                    raise AuditIntegrityError(
                        f"Hash mismatch at record "
                        f"{expected_sequence}"
                    )

                previous_hash = stored_hash

            except (KeyError, TypeError) as exc:
                raise AuditIntegrityError(
                    f"Malformed record "
                    f"{expected_sequence}"
                ) from exc

        return {
            "valid": True,
            "verified_records": len(records),
            "last_hash": previous_hash,
        }


# ============================================================
# JSONL STORAGE
# ============================================================

class AuditJSONLStorage:

    def __init__(
        self,
        config: AuditLoggingConfig,
    ) -> None:
        self.config = config

        self.directory = Path(
            config.log_directory
        )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.path = (
            self.directory
            / config.log_filename
        )

        self._lock = threading.RLock()

    def read_all(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.path.exists():
                return []

            records = []

            try:
                with self.path.open(
                    "r",
                    encoding="utf-8",
                ) as handle:
                    for line_number, line in enumerate(
                        handle,
                        start=1,
                    ):
                        if not line.strip():
                            raise AuditStorageError(
                                f"Empty record at line "
                                f"{line_number}"
                            )

                        record = json.loads(line)

                        if not isinstance(record, dict):
                            raise AuditStorageError(
                                "Audit record must be an object"
                            )

                        records.append(record)

            except (OSError, json.JSONDecodeError) as exc:
                raise AuditStorageError(
                    "Failed to read audit records"
                ) from exc

            return records

    def append(
        self,
        record: Mapping[str, Any],
    ) -> None:
        encoded = (
            canonical_json(dict(record)) + "\n"
        ).encode("utf-8")

        with self._lock:
            try:
                with self.path.open("ab") as handle:
                    handle.write(encoded)
                    handle.flush()

                    if self.config.fsync_each_event:
                        os.fsync(handle.fileno())

            except OSError as exc:
                raise AuditStorageError(
                    "Failed to append audit record"
                ) from exc


# ============================================================
# CENTRAL AUDIT LOGGING SERVICE
# ============================================================

class RoboticsAuditLoggingService:

    def __init__(
        self,
        config: AuditLoggingConfig | None = None,
    ) -> None:
        self.config = (
            config or AuditLoggingConfig()
        )

        self.storage = AuditJSONLStorage(
            self.config
        )

        self.integrity = AuditIntegrityEngine(
            self.config.hmac_key
        )

        self.redactor = AuditRedactor()

        self._lock = threading.RLock()

        # Validate the existing chain before allowing
        # new events to be appended.
        records = self.storage.read_all()
        self.integrity.verify(records)

        self._sequence = len(records)

        self._previous_hash = (
            records[-1]["event_hash"]
            if records
            else AuditIntegrityEngine.GENESIS_HASH
        )

    # --------------------------------------------------------
    # LOG EVENT
    # --------------------------------------------------------

    def log(
        self,
        *,
        category: AuditCategory,
        action: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        actor_id: str | None = None,
        actor_role: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
        source_ip: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> AuditEvent:
        action = require_text(
            action,
            "action",
        )

        sanitized_details = self.redactor.redact(
            dict(details or {})
        )

        detail_size = len(
            canonical_json(
                sanitized_details
            ).encode("utf-8")
        )

        if (
            detail_size
            > self.config.maximum_detail_bytes
        ):
            raise AuditValidationError(
                "Audit details exceed size limit"
            )

        with self._lock:
            sequence = self._sequence + 1

            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                sequence=sequence,
                timestamp=utc_now(),
                category=AuditCategory(category),
                action=action,
                severity=AuditSeverity(severity),
                outcome=AuditOutcome(outcome),
                actor_id=actor_id,
                actor_role=actor_role,
                resource_type=resource_type,
                resource_id=resource_id,
                request_id=request_id,
                correlation_id=correlation_id,
                source_ip=source_ip,
                details=sanitized_details,
                previous_hash=self._previous_hash,
            )

            event_hash = self.integrity.calculate_hash(
                event.unsigned_dict()
            )

            signed_event = AuditEvent(
                **{
                    **event.__dict__,
                    "event_hash": event_hash,
                }
            )

            self.storage.append(
                signed_event.to_dict()
            )

            self._sequence = sequence
            self._previous_hash = event_hash

            return signed_event

    # --------------------------------------------------------
    # QUERY EVENTS
    # --------------------------------------------------------

    def query(
        self,
        *,
        category: AuditCategory | None = None,
        severity: AuditSeverity | None = None,
        outcome: AuditOutcome | None = None,
        actor_id: str | None = None,
        resource_id: str | None = None,
        correlation_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        records = self.storage.read_all()

        start_time = (
            parse_timestamp(start)
            if start is not None
            else None
        )

        end_time = (
            parse_timestamp(end)
            if end is not None
            else None
        )

        if (
            start_time is not None
            and end_time is not None
            and start_time > end_time
        ):
            raise AuditValidationError(
                "start must not exceed end"
            )

        filtered = []

        for record in records:
            if (
                category is not None
                and record["category"]
                != AuditCategory(category).value
            ):
                continue

            if (
                severity is not None
                and record["severity"]
                != AuditSeverity(severity).value
            ):
                continue

            if (
                outcome is not None
                and record["outcome"]
                != AuditOutcome(outcome).value
            ):
                continue

            if (
                actor_id is not None
                and record["actor_id"] != actor_id
            ):
                continue

            if (
                resource_id is not None
                and record["resource_id"] != resource_id
            ):
                continue

            if (
                correlation_id is not None
                and record["correlation_id"]
                != correlation_id
            ):
                continue

            timestamp = parse_timestamp(
                record["timestamp"]
            )

            if (
                start_time is not None
                and timestamp < start_time
            ):
                continue

            if (
                end_time is not None
                and timestamp > end_time
            ):
                continue

            filtered.append(record)

        if limit is not None:
            if (
                isinstance(limit, bool)
                or not isinstance(limit, int)
                or limit < 1
            ):
                raise AuditValidationError(
                    "limit must be positive"
                )

            filtered = filtered[-limit:]

        return filtered

    # --------------------------------------------------------
    # VERIFY INTEGRITY
    # --------------------------------------------------------

    def verify_integrity(
        self,
    ) -> dict[str, Any]:
        records = self.storage.read_all()
        return self.integrity.verify(records)

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        records = self.storage.read_all()

        category_counts: dict[str, int] = {}
        outcome_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}

        for record in records:
            for counter, key in (
                (category_counts, "category"),
                (outcome_counts, "outcome"),
                (severity_counts, "severity"),
            ):
                value = record[key]
                counter[value] = counter.get(
                    value,
                    0,
                ) + 1

        return {
            "total_events": len(records),
            "category_distribution": category_counts,
            "outcome_distribution": outcome_counts,
            "severity_distribution": severity_counts,
            "latest_event_at": (
                records[-1]["timestamp"]
                if records
                else None
            ),
            "integrity": self.integrity.verify(
                records
            ),
        }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        **filters: Any,
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        records = self.query(**filters)

        output.write_text(
            json.dumps(
                records,
                indent=2,
                ensure_ascii=False,
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
        **filters: Any,
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        records = self.query(**filters)

        fieldnames = [
            "sequence",
            "event_id",
            "timestamp",
            "category",
            "action",
            "severity",
            "outcome",
            "actor_id",
            "actor_role",
            "resource_type",
            "resource_id",
            "request_id",
            "correlation_id",
            "source_ip",
            "details",
            "previous_hash",
            "event_hash",
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

            for record in records:
                row = dict(record)

                row["details"] = canonical_json(
                    row["details"]
                )

                writer.writerow(row)

        return output


# ============================================================
# API SECURITY ADAPTER
# ============================================================

class APISecurityAuditAdapter:

    def __init__(
        self,
        audit: RoboticsAuditLoggingService,
    ) -> None:
        self.audit = audit

    def authentication(
        self,
        *,
        principal_id: str | None,
        successful: bool,
        source_ip: str | None = None,
        reason: str | None = None,
    ) -> AuditEvent:
        return self.audit.log(
            category=AuditCategory.AUTHENTICATION,
            action=(
                "authentication_success"
                if successful
                else "authentication_failure"
            ),
            severity=(
                AuditSeverity.INFO
                if successful
                else AuditSeverity.WARNING
            ),
            outcome=(
                AuditOutcome.SUCCESS
                if successful
                else AuditOutcome.FAILURE
            ),
            actor_id=principal_id,
            source_ip=source_ip,
            details={
                "reason": reason,
            },
        )

    def authorization(
        self,
        *,
        principal_id: str,
        permission: str,
        resource: str,
        allowed: bool,
    ) -> AuditEvent:
        return self.audit.log(
            category=AuditCategory.AUTHORIZATION,
            action="permission_check",
            severity=(
                AuditSeverity.INFO
                if allowed
                else AuditSeverity.WARNING
            ),
            outcome=(
                AuditOutcome.SUCCESS
                if allowed
                else AuditOutcome.DENIED
            ),
            actor_id=principal_id,
            resource_type="api_resource",
            resource_id=resource,
            details={
                "permission": permission,
                "allowed": allowed,
            },
        )


# ============================================================
# ROBOT OPERATIONS AUDIT ADAPTER
# ============================================================

class RobotOperationsAuditAdapter:

    def __init__(
        self,
        audit: RoboticsAuditLoggingService,
    ) -> None:
        self.audit = audit

    def robot_event(
        self,
        *,
        robot_id: str,
        action: str,
        actor_id: str,
        successful: bool = True,
        details: Mapping[str, Any] | None = None,
    ) -> AuditEvent:
        return self.audit.log(
            category=AuditCategory.ROBOT,
            action=action,
            severity=AuditSeverity.INFO,
            outcome=(
                AuditOutcome.SUCCESS
                if successful
                else AuditOutcome.FAILURE
            ),
            actor_id=actor_id,
            resource_type="robot",
            resource_id=robot_id,
            details=details,
        )

    def task_event(
        self,
        *,
        task_id: str,
        robot_id: str,
        action: str,
        actor_id: str,
        successful: bool = True,
    ) -> AuditEvent:
        return self.audit.log(
            category=AuditCategory.TASK,
            action=action,
            outcome=(
                AuditOutcome.SUCCESS
                if successful
                else AuditOutcome.FAILURE
            ),
            actor_id=actor_id,
            resource_type="task",
            resource_id=task_id,
            details={
                "robot_id": robot_id,
            },
        )


# ============================================================
# MODEL AUDIT ADAPTER
# ============================================================

class ModelAuditAdapter:

    def __init__(
        self,
        audit: RoboticsAuditLoggingService,
    ) -> None:
        self.audit = audit

    def model_event(
        self,
        *,
        model_id: str,
        action: str,
        actor_id: str,
        version: str | None = None,
        successful: bool = True,
    ) -> AuditEvent:
        return self.audit.log(
            category=AuditCategory.MODEL,
            action=action,
            actor_id=actor_id,
            resource_type="model",
            resource_id=model_id,
            outcome=(
                AuditOutcome.SUCCESS
                if successful
                else AuditOutcome.FAILURE
            ),
            details={
                "version": version,
            },
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    # For persistent authenticated audit logs,
    # configure a securely managed HMAC key.
    audit = RoboticsAuditLoggingService(
        AuditLoggingConfig(
            log_directory="data/security/audit",
        )
    )

    api_adapter = APISecurityAuditAdapter(
        audit
    )

    robot_adapter = RobotOperationsAuditAdapter(
        audit
    )

    model_adapter = ModelAuditAdapter(
        audit
    )

    api_adapter.authentication(
        principal_id="engineer_001",
        successful=True,
        source_ip="127.0.0.1",
    )

    api_adapter.authorization(
        principal_id="engineer_001",
        permission="simulation:write",
        resource="simulation/demo",
        allowed=True,
    )

    robot_adapter.robot_event(
        robot_id="robot_001",
        action="robot_registered",
        actor_id="engineer_001",
        details={
            "status": "idle",
            "api_key": "sensitive-example",
        },
    )

    robot_adapter.task_event(
        task_id="task_001",
        robot_id="robot_001",
        action="task_completed",
        actor_id="scheduler",
    )

    model_adapter.model_event(
        model_id="vision_model_001",
        action="model_deployed",
        actor_id="engineer_001",
        version="1.0.0",
    )

    integrity = audit.verify_integrity()

    json_path = audit.export_json(
        "data/security/reports/audit_report.json"
    )

    csv_path = audit.export_csv(
        "data/security/reports/audit_report.csv"
    )

    print(
        json.dumps(
            {
                "statistics": audit.statistics(),
                "integrity": integrity,
                "files": [
                    str(json_path),
                    str(csv_path),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

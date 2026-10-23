"""Upload Security module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/security/upload_security.py

Python 3.11+
Standard library only.

Security features:
    - File extension allowlist
    - Filename validation
    - Upload size limits
    - MIME/content signature checks
    - ZIP traversal and archive-bomb checks
    - SHA-256 integrity hashing
    - Secure temporary staging
    - Tenant-scoped upload storage
    - Duplicate file detection
    - Upload metadata tracking
    - Atomic publication
    - Upload audit logging
    - JSON report export

This module does not deserialize AI models,
execute uploaded code, or control physical robots.

IMPORTANT:
    CLEAN means the structural checks passed.
    It does not mean a file is malware-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading
import uuid
import zipfile

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO


# ============================================================
# EXCEPTIONS
# ============================================================

class UploadSecurityError(Exception):
    pass


class UploadValidationError(UploadSecurityError):
    pass


class UploadRejectedError(UploadSecurityError):
    pass


class UploadStorageError(UploadSecurityError):
    pass


# ============================================================
# ENUMS
# ============================================================

class UploadStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


class UploadCategory(str, Enum):
    IMAGE = "image"
    DOCUMENT = "document"
    DATASET = "dataset"
    MODEL = "model"
    ARCHIVE = "archive"


class UploadThreat(str, Enum):
    INVALID_FILENAME = "invalid_filename"
    INVALID_EXTENSION = "invalid_extension"
    OVERSIZED = "oversized"
    EMPTY = "empty"
    SIGNATURE_MISMATCH = "signature_mismatch"
    ARCHIVE_TRAVERSAL = "archive_traversal"
    ARCHIVE_BOMB = "archive_bomb"
    NESTED_ARCHIVE = "nested_archive"
    INVALID_ARCHIVE = "invalid_archive"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def validate_identifier(
    value: str,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}",
            value,
        )
    ):
        raise UploadValidationError(
            f"Invalid {field_name}"
        )

    return value


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(
            1024 * 1024
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class UploadSecurityConfig:
    storage_directory: str = (
        "data/uploads/secure"
    )

    maximum_file_bytes: int = (
        100 * 1024 * 1024
    )

    maximum_archive_entries: int = 1000

    maximum_archive_uncompressed_bytes: int = (
        500 * 1024 * 1024
    )

    maximum_archive_entry_bytes: int = (
        100 * 1024 * 1024
    )

    maximum_archive_ratio: float = 100.0

    maximum_audit_events: int = 20000

    allowed_extensions: frozenset[str] = field(
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
            ".zip",
            ".onnx",
            ".safetensors",
        })
    )

    def __post_init__(self) -> None:
        for name in (
            "maximum_file_bytes",
            "maximum_archive_entries",
            "maximum_archive_uncompressed_bytes",
            "maximum_archive_entry_bytes",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise UploadValidationError(
                    f"{name} must be positive"
                )

        if self.maximum_archive_ratio <= 1:
            raise UploadValidationError(
                "Archive ratio must exceed 1"
            )


# ============================================================
# FILE TYPES
# ============================================================

EXTENSION_CATEGORIES = {
    ".png": UploadCategory.IMAGE,
    ".jpg": UploadCategory.IMAGE,
    ".jpeg": UploadCategory.IMAGE,
    ".webp": UploadCategory.IMAGE,
    ".pdf": UploadCategory.DOCUMENT,
    ".txt": UploadCategory.DOCUMENT,
    ".csv": UploadCategory.DATASET,
    ".json": UploadCategory.DATASET,
    ".jsonl": UploadCategory.DATASET,
    ".zip": UploadCategory.ARCHIVE,
    ".onnx": UploadCategory.MODEL,
    ".safetensors": UploadCategory.MODEL,
}


# ============================================================
# FILENAME VALIDATOR
# ============================================================

class UploadFilenameValidator:

    @staticmethod
    def validate(
        filename: str,
        allowed_extensions: frozenset[str],
    ) -> tuple[str, str]:
        if (
            not isinstance(filename, str)
            or not filename
            or len(filename) > 255
        ):
            raise UploadRejectedError(
                "Invalid upload filename"
            )

        if filename != filename.strip():
            raise UploadRejectedError(
                "Filename has invalid whitespace"
            )

        if (
            "/" in filename
            or "\\" in filename
            or "\x00" in filename
            or filename in {".", ".."}
        ):
            raise UploadRejectedError(
                "Filename contains path characters"
            )

        if filename.startswith("."):
            raise UploadRejectedError(
                "Hidden filenames are not allowed"
            )

        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_. -]{0,254}",
            filename,
        ):
            raise UploadRejectedError(
                "Filename contains unsupported characters"
            )

        extension = Path(
            filename
        ).suffix.lower()

        if extension not in allowed_extensions:
            raise UploadRejectedError(
                "File extension is not permitted"
            )

        return filename, extension


# ============================================================
# FILE SIGNATURE VALIDATOR
# ============================================================

class UploadSignatureValidator:

    SIGNATURES = {
        ".png": (b"\x89PNG\r\n\x1a\n",),
        ".jpg": (b"\xff\xd8\xff",),
        ".jpeg": (b"\xff\xd8\xff",),
        ".pdf": (b"%PDF-",),
        ".zip": (
            b"PK\x03\x04",
            b"PK\x05\x06",
        ),
    }

    @classmethod
    def validate(
        cls,
        path: Path,
        extension: str,
    ) -> None:
        with path.open("rb") as handle:
            header = handle.read(32)

        signatures = cls.SIGNATURES.get(
            extension
        )

        if signatures is not None:
            if not any(
                header.startswith(signature)
                for signature in signatures
            ):
                raise UploadRejectedError(
                    "File signature mismatch"
                )

        if extension == ".webp":
            if not (
                header.startswith(b"RIFF")
                and header[8:12] == b"WEBP"
            ):
                raise UploadRejectedError(
                    "Invalid WebP signature"
                )

        if extension == ".safetensors":
            # Safetensors begins with an 8-byte
            # little-endian header length followed
            # by a JSON metadata header.
            if len(header) < 9:
                raise UploadRejectedError(
                    "Invalid safetensors header"
                )

            header_length = int.from_bytes(
                header[:8],
                "little",
            )

            file_size = path.stat().st_size

            if (
                header_length < 2
                or header_length > file_size - 8
                or header_length > 10 * 1024 * 1024
            ):
                raise UploadRejectedError(
                    "Invalid safetensors metadata length"
                )


# ============================================================
# ZIP ARCHIVE SECURITY
# ============================================================

class UploadArchiveValidator:

    def __init__(
        self,
        config: UploadSecurityConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        path: Path,
    ) -> None:
        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()

                if (
                    len(entries)
                    > self.config.maximum_archive_entries
                ):
                    raise UploadRejectedError(
                        "Archive contains too many entries"
                    )

                total_uncompressed = 0

                for entry in entries:
                    name = entry.filename.replace(
                        "\\",
                        "/",
                    )

                    components = name.split("/")

                    if (
                        name.startswith("/")
                        or ".." in components
                        or re.match(
                            r"^[A-Za-z]:",
                            name,
                        )
                    ):
                        raise UploadRejectedError(
                            "Archive path traversal detected"
                        )

                    unix_mode = (
                        entry.external_attr >> 16
                    )

                    if (
                        stat.S_IFMT(unix_mode)
                        == stat.S_IFLNK
                    ):
                        raise UploadRejectedError(
                            "Archive symlink rejected"
                        )

                    if entry.file_size > (
                        self.config.maximum_archive_entry_bytes
                    ):
                        raise UploadRejectedError(
                            "Archive entry too large"
                        )

                    total_uncompressed += (
                        entry.file_size
                    )

                    if total_uncompressed > (
                        self.config.maximum_archive_uncompressed_bytes
                    ):
                        raise UploadRejectedError(
                            "Archive uncompressed size too large"
                        )

                    ratio = (
                        entry.file_size
                        / max(entry.compress_size, 1)
                    )

                    if ratio > (
                        self.config.maximum_archive_ratio
                    ):
                        raise UploadRejectedError(
                            "Suspicious archive compression ratio"
                        )

                    if name.lower().endswith(
                        (
                            ".zip",
                            ".tar",
                            ".gz",
                            ".7z",
                            ".rar",
                        )
                    ):
                        raise UploadRejectedError(
                            "Nested archives are not allowed"
                        )

        except (
            zipfile.BadZipFile,
            zipfile.LargeZipFile,
        ) as exc:
            raise UploadRejectedError(
                "Invalid ZIP archive"
            ) from exc


# ============================================================
# UPLOAD RESULT
# ============================================================

@dataclass(frozen=True)
class SecureUploadResult:
    upload_id: str
    tenant_id: str
    original_filename: str

    category: UploadCategory
    extension: str

    size_bytes: int
    sha256: str

    storage_path: str
    status: UploadStatus

    uploaded_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "tenant_id": self.tenant_id,
            "original_filename": self.original_filename,
            "category": self.category.value,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "storage_path": self.storage_path,
            "status": self.status.value,
            "uploaded_at": self.uploaded_at,
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class UploadAuditEvent:
    event_id: str
    timestamp: str

    tenant_id: str
    filename: str

    action: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "tenant_id": self.tenant_id,
            "filename": self.filename,
            "action": self.action,
            "details": self.details,
        }


# ============================================================
# SECURE UPLOAD SERVICE
# ============================================================

class RoboticsUploadSecurityService:

    def __init__(
        self,
        config: UploadSecurityConfig | None = None,
    ) -> None:
        self.config = (
            config or UploadSecurityConfig()
        )

        self.storage_root = Path(
            self.config.storage_directory
        ).resolve()

        self.storage_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.archive_validator = (
            UploadArchiveValidator(
                self.config
            )
        )

        self.uploads: dict[
            str,
            SecureUploadResult
        ] = {}

        self.audit_events: list[
            UploadAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        tenant_id: str,
        filename: str,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.audit_events.append(
            UploadAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=tenant_id,
                filename=filename,
                action=action,
                details=dict(details or {}),
            )
        )

        if (
            len(self.audit_events)
            > self.config.maximum_audit_events
        ):
            self.audit_events.pop(0)

    # --------------------------------------------------------
    # UPLOAD STREAM
    # --------------------------------------------------------

    def upload_stream(
        self,
        *,
        tenant_id: str,
        filename: str,
        stream: BinaryIO,
    ) -> SecureUploadResult:
        tenant_id = validate_identifier(
            tenant_id,
            "tenant_id",
        )

        filename, extension = (
            UploadFilenameValidator.validate(
                filename,
                self.config.allowed_extensions,
            )
        )

        if not hasattr(stream, "read"):
            raise UploadValidationError(
                "Stream must provide read()"
            )

        tenant_directory = (
            self.storage_root / tenant_id
        )

        tenant_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        fd, temporary_name = tempfile.mkstemp(
            prefix=".upload_",
            suffix=".tmp",
            dir=tenant_directory,
        )

        temporary_path = Path(
            temporary_name
        )

        upload_id = str(uuid.uuid4())

        size = 0
        digest = hashlib.sha256()

        try:
            with os.fdopen(
                fd,
                "wb",
            ) as output:
                while True:
                    chunk = stream.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    if not isinstance(
                        chunk,
                        bytes,
                    ):
                        raise UploadValidationError(
                            "Upload stream must return bytes"
                        )

                    size += len(chunk)

                    if size > (
                        self.config.maximum_file_bytes
                    ):
                        raise UploadRejectedError(
                            "Upload exceeds maximum size"
                        )

                    digest.update(
                        chunk
                    )

                    output.write(
                        chunk
                    )

            if size == 0:
                raise UploadRejectedError(
                    "Empty uploads are not allowed"
                )

            UploadSignatureValidator.validate(
                temporary_path,
                extension,
            )

            if extension == ".zip":
                self.archive_validator.validate(
                    temporary_path
                )

            sha256 = digest.hexdigest()

            with self._lock:
                for previous in self.uploads.values():
                    if (
                        previous.tenant_id == tenant_id
                        and previous.sha256 == sha256
                        and previous.extension == extension
                    ):
                        self._audit(
                            tenant_id,
                            filename,
                            "duplicate_detected",
                            {
                                "existing_upload_id": (
                                    previous.upload_id
                                ),
                            },
                        )

                        return SecureUploadResult(
                            upload_id=previous.upload_id,
                            tenant_id=tenant_id,
                            original_filename=filename,
                            category=previous.category,
                            extension=extension,
                            size_bytes=size,
                            sha256=sha256,
                            storage_path=previous.storage_path,
                            status=UploadStatus.DUPLICATE,
                            uploaded_at=utc_now(),
                        )

                final_path = (
                    tenant_directory
                    / f"{upload_id}{extension}"
                )

                os.replace(
                    temporary_path,
                    final_path,
                )

                result = SecureUploadResult(
                    upload_id=upload_id,
                    tenant_id=tenant_id,
                    original_filename=filename,
                    category=EXTENSION_CATEGORIES[
                        extension
                    ],
                    extension=extension,
                    size_bytes=size,
                    sha256=sha256,
                    storage_path=str(final_path),
                    status=UploadStatus.ACCEPTED,
                    uploaded_at=utc_now(),
                )

                self.uploads[
                    upload_id
                ] = result

                self._audit(
                    tenant_id,
                    filename,
                    "upload_accepted",
                    {
                        "upload_id": upload_id,
                        "sha256": sha256,
                        "size_bytes": size,
                    },
                )

                return result

        except Exception as exc:
            with self._lock:
                self._audit(
                    tenant_id,
                    filename,
                    "upload_rejected",
                    {
                        "reason": type(exc).__name__,
                    },
                )

            raise

        finally:
            temporary_path.unlink(
                missing_ok=True
            )

    # --------------------------------------------------------
    # UPLOAD BYTES
    # --------------------------------------------------------

    def upload_bytes(
        self,
        *,
        tenant_id: str,
        filename: str,
        data: bytes,
    ) -> SecureUploadResult:
        from io import BytesIO

        if not isinstance(data, bytes):
            raise UploadValidationError(
                "Data must be bytes"
            )

        return self.upload_stream(
            tenant_id=tenant_id,
            filename=filename,
            stream=BytesIO(data),
        )

    # --------------------------------------------------------
    # GET UPLOAD
    # --------------------------------------------------------

    def get_upload(
        self,
        *,
        tenant_id: str,
        upload_id: str,
    ) -> SecureUploadResult:
        with self._lock:
            result = self.uploads.get(
                upload_id
            )

            if (
                result is None
                or result.tenant_id != tenant_id
            ):
                raise UploadSecurityError(
                    "Upload not found"
                )

            return result

    # --------------------------------------------------------
    # VERIFY UPLOAD INTEGRITY
    # --------------------------------------------------------

    def verify_integrity(
        self,
        *,
        tenant_id: str,
        upload_id: str,
    ) -> bool:
        result = self.get_upload(
            tenant_id=tenant_id,
            upload_id=upload_id,
        )

        path = Path(
            result.storage_path
        )

        if not path.is_file():
            return False

        if path.is_symlink():
            return False

        if path.stat().st_size != result.size_bytes:
            return False

        return (
            sha256_file(path)
            == result.sha256
        )

    # --------------------------------------------------------
    # LIST UPLOADS
    # --------------------------------------------------------

    def list_uploads(
        self,
        *,
        tenant_id: str,
    ) -> list[SecureUploadResult]:
        with self._lock:
            return [
                result
                for result in self.uploads.values()
                if result.tenant_id == tenant_id
            ]

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            return {
                "registered_uploads": len(
                    self.uploads
                ),
                "total_stored_bytes": sum(
                    result.size_bytes
                    for result in self.uploads.values()
                ),
                "audit_events": len(
                    self.audit_events
                ),
            }

    # --------------------------------------------------------
    # EXPORT AUDIT REPORT
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

        with self._lock:
            payload = {
                "generated_at": utc_now(),
                "statistics": self.statistics(),
                "uploads": [
                    upload.to_dict()
                    for upload in self.uploads.values()
                ],
                "audit_events": [
                    event.to_dict()
                    for event in self.audit_events
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
# FILE SERVICE ADAPTER
# ============================================================

class UploadSecurityAdapter:

    def __init__(
        self,
        security: RoboticsUploadSecurityService,
    ) -> None:
        self.security = security

    def validate_and_store(
        self,
        *,
        tenant_id: str,
        filename: str,
        data: bytes,
    ) -> SecureUploadResult:
        return self.security.upload_bytes(
            tenant_id=tenant_id,
            filename=filename,
            data=data,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsUploadSecurityService()

    result = service.upload_bytes(
        tenant_id="tenant_alpha",
        filename="robot_telemetry.json",
        data=json.dumps({
            "robot_id": "robot_001",
            "battery_soc": 85.0,
            "temperature_c": 35.5,
            "status": "healthy",
        }).encode("utf-8"),
    )

    integrity_ok = service.verify_integrity(
        tenant_id="tenant_alpha",
        upload_id=result.upload_id,
    )

    report = service.export_json(
        "data/security/upload_security_report.json"
    )

    print(
        json.dumps(
            {
                "upload": result.to_dict(),
                "integrity_verified": integrity_ok,
                "statistics": service.statistics(),
                "report": str(report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

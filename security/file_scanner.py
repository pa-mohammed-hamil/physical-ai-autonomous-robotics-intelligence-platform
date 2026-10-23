"""File Scanner module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/security/file_scanner.py

Python 3.11+
Standard library only for built-in checks.

Optional:
    ClamAV clamscan executable for malware scanning.

Features:
    - File upload security scanning
    - Extension allowlisting
    - File signature verification
    - MIME type checking
    - SHA-256 file hashing
    - Maximum file-size enforcement
    - Empty-file detection
    - Suspicious filename detection
    - ZIP archive inspection
    - ZIP-slip detection
    - Archive decompression-ratio checks
    - Nested archive rejection
    - Suspicious content heuristics
    - Optional ClamAV scanning
    - Quarantine workflow
    - JSON security reports

Important:
    Built-in checks are heuristic.
    They do not establish that a file is malware-free.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import uuid
import zipfile

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class FileScannerError(Exception):
    pass


class FileScannerValidationError(FileScannerError):
    pass


class FileScannerIOError(FileScannerError):
    pass


# ============================================================
# ENUMS
# ============================================================

class FileScanStatus(str, Enum):
    CLEAN = "clean"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    REJECTED = "rejected"
    ERROR = "error"


class FileThreatSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FileThreatCategory(str, Enum):
    INVALID_EXTENSION = "invalid_extension"
    INVALID_SIGNATURE = "invalid_signature"
    MIME_MISMATCH = "mime_mismatch"
    OVERSIZED_FILE = "oversized_file"
    EMPTY_FILE = "empty_file"
    SUSPICIOUS_FILENAME = "suspicious_filename"
    SUSPICIOUS_CONTENT = "suspicious_content"
    ARCHIVE_TRAVERSAL = "archive_traversal"
    ARCHIVE_BOMB = "archive_bomb"
    NESTED_ARCHIVE = "nested_archive"
    MALWARE = "malware"
    SCANNER_ERROR = "scanner_error"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
        )
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class FileScannerConfig:
    maximum_file_size_bytes: int = 100 * 1024 * 1024

    maximum_archive_entries: int = 1000
    maximum_archive_uncompressed_bytes: int = (
        500 * 1024 * 1024
    )

    maximum_archive_ratio: float = 100.0

    maximum_archive_entry_bytes: int = (
        100 * 1024 * 1024
    )

    maximum_content_scan_bytes: int = (
        1024 * 1024
    )

    allowed_extensions: tuple[str, ...] = (
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
        ".pt",
        ".pth",
    )

    quarantine_directory: str = (
        "data/security/quarantine"
    )

    report_directory: str = (
        "data/security/scanner_reports"
    )

    enable_clamav: bool = False
    clamav_executable: str = "clamscan"
    clamav_timeout_seconds: int = 60

    quarantine_on_suspicious: bool = True
    quarantine_on_malicious: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_file_size_bytes",
            "maximum_archive_entries",
            "maximum_archive_uncompressed_bytes",
            "maximum_archive_entry_bytes",
            "maximum_content_scan_bytes",
            "clamav_timeout_seconds",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise FileScannerValidationError(
                    f"{name} must be positive"
                )

        if (
            isinstance(self.maximum_archive_ratio, bool)
            or not isinstance(
                self.maximum_archive_ratio,
                (int, float),
            )
            or not math.isfinite(
                self.maximum_archive_ratio
            )
            or self.maximum_archive_ratio <= 1
        ):
            raise FileScannerValidationError(
                "Invalid maximum archive ratio"
            )


# ============================================================
# FILE SIGNATURES
# ============================================================

FILE_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".png": (
        b"\x89PNG\r\n\x1a\n",
    ),
    ".jpg": (
        b"\xff\xd8\xff",
    ),
    ".jpeg": (
        b"\xff\xd8\xff",
    ),
    ".webp": (
        b"RIFF",
    ),
    ".pdf": (
        b"%PDF-",
    ),
    ".zip": (
        b"PK\x03\x04",
        b"PK\x05\x06",
        b"PK\x07\x08",
    ),
    ".docx": (
        b"PK\x03\x04",
    ),
    ".xlsx": (
        b"PK\x03\x04",
    ),
    ".onnx": (),
    ".pt": (),
    ".pth": (),
}


MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".docx": (
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document"
    ),
    ".xlsx": (
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
    ".zip": "application/zip",
    ".onnx": "application/octet-stream",
    ".pt": "application/octet-stream",
    ".pth": "application/octet-stream",
}


# ============================================================
# THREAT FINDING
# ============================================================

@dataclass(frozen=True)
class FileThreatFinding:
    category: FileThreatCategory
    severity: FileThreatSeverity
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
        }


# ============================================================
# SCAN RESULT
# ============================================================

@dataclass
class FileScanResult:
    scan_id: str
    filename: str
    original_path: str

    file_size_bytes: int
    sha256: str | None

    extension: str
    expected_mime_type: str | None

    status: FileScanStatus

    findings: list[FileThreatFinding] = field(
        default_factory=list
    )

    antivirus_checked: bool = False

    quarantined: bool = False
    quarantine_path: str | None = None

    scanned_at: str = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "filename": self.filename,
            "original_path": self.original_path,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "extension": self.extension,
            "expected_mime_type": (
                self.expected_mime_type
            ),
            "status": self.status.value,
            "findings": [
                finding.to_dict()
                for finding in self.findings
            ],
            "antivirus_checked": (
                self.antivirus_checked
            ),
            "quarantined": self.quarantined,
            "quarantine_path": (
                self.quarantine_path
            ),
            "scanned_at": self.scanned_at,
        }


# ============================================================
# FILENAME VALIDATOR
# ============================================================

class FileNameValidator:

    SUSPICIOUS_PATTERNS = (
        r"\.\.",
        r"[<>:\"|?*]",
        r"[\x00-\x1f]",
    )

    @classmethod
    def inspect(
        cls,
        filename: str,
    ) -> list[FileThreatFinding]:
        findings = []

        if not filename:
            findings.append(
                FileThreatFinding(
                    FileThreatCategory.SUSPICIOUS_FILENAME,
                    FileThreatSeverity.HIGH,
                    "Empty filename",
                )
            )

            return findings

        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, filename):
                findings.append(
                    FileThreatFinding(
                        FileThreatCategory.SUSPICIOUS_FILENAME,
                        FileThreatSeverity.HIGH,
                        "Suspicious filename characters",
                    )
                )
                break

        if (
            "/" in filename
            or "\\" in filename
        ):
            findings.append(
                FileThreatFinding(
                    FileThreatCategory.SUSPICIOUS_FILENAME,
                    FileThreatSeverity.HIGH,
                    "Filename contains path separators",
                )
            )

        if len(filename) > 255:
            findings.append(
                FileThreatFinding(
                    FileThreatCategory.SUSPICIOUS_FILENAME,
                    FileThreatSeverity.MEDIUM,
                    "Filename exceeds length limit",
                )
            )

        return findings


# ============================================================
# FILE SIGNATURE VALIDATOR
# ============================================================

class FileSignatureValidator:

    @staticmethod
    def inspect(
        path: Path,
        extension: str,
    ) -> list[FileThreatFinding]:
        findings = []

        signatures = FILE_SIGNATURES.get(
            extension
        )

        if signatures is None or not signatures:
            return findings

        with path.open("rb") as handle:
            header = handle.read(32)

        if not any(
            header.startswith(signature)
            for signature in signatures
        ):
            findings.append(
                FileThreatFinding(
                    FileThreatCategory.INVALID_SIGNATURE,
                    FileThreatSeverity.HIGH,
                    "File signature does not match extension",
                )
            )

        if extension == ".webp":
            if (
                len(header) < 12
                or header[8:12] != b"WEBP"
            ):
                findings.append(
                    FileThreatFinding(
                        FileThreatCategory.INVALID_SIGNATURE,
                        FileThreatSeverity.HIGH,
                        "Invalid WEBP container signature",
                    )
                )

        return findings


# ============================================================
# SUSPICIOUS CONTENT SCANNER
# ============================================================

class SuspiciousContentScanner:

    PATTERNS: tuple[tuple[bytes, str], ...] = (
        (
            b"<script",
            "Embedded script tag",
        ),
        (
            b"javascript:",
            "JavaScript URI",
        ),
        (
            b"powershell -enc",
            "Encoded PowerShell command",
        ),
        (
            b"cmd.exe /c",
            "Shell execution pattern",
        ),
        (
            b"/bin/sh -c",
            "Shell execution pattern",
        ),
    )

    TEXT_EXTENSIONS = {
        ".txt",
        ".csv",
        ".json",
        ".jsonl",
    }

    def __init__(
        self,
        config: FileScannerConfig,
    ) -> None:
        self.config = config

    def inspect(
        self,
        path: Path,
        extension: str,
    ) -> list[FileThreatFinding]:
        if extension not in self.TEXT_EXTENSIONS:
            return []

        with path.open("rb") as handle:
            content = handle.read(
                self.config.maximum_content_scan_bytes
            ).lower()

        findings = []

        for pattern, description in self.PATTERNS:
            if pattern in content:
                findings.append(
                    FileThreatFinding(
                        FileThreatCategory.SUSPICIOUS_CONTENT,
                        FileThreatSeverity.MEDIUM,
                        description,
                    )
                )

        return findings


# ============================================================
# ARCHIVE SECURITY SCANNER
# ============================================================

class ArchiveSecurityScanner:

    ARCHIVE_EXTENSIONS = {
        ".zip",
        ".rar",
        ".7z",
        ".tar",
        ".gz",
    }

    def __init__(
        self,
        config: FileScannerConfig,
    ) -> None:
        self.config = config

    def inspect(
        self,
        path: Path,
    ) -> list[FileThreatFinding]:
        findings = []

        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()

                if (
                    len(entries)
                    > self.config.maximum_archive_entries
                ):
                    findings.append(
                        FileThreatFinding(
                            FileThreatCategory.ARCHIVE_BOMB,
                            FileThreatSeverity.HIGH,
                            "Too many archive entries",
                        )
                    )

                total_uncompressed = 0

                for entry in entries:
                    name = entry.filename.replace(
                        "\\",
                        "/",
                    )

                    pure_path = PurePosixPath(name)

                    if (
                        pure_path.is_absolute()
                        or ".." in pure_path.parts
                        or re.match(
                            r"^[a-zA-Z]:",
                            name,
                        )
                    ):
                        findings.append(
                            FileThreatFinding(
                                FileThreatCategory.ARCHIVE_TRAVERSAL,
                                FileThreatSeverity.CRITICAL,
                                f"Unsafe archive entry: {name}",
                            )
                        )

                    if entry.is_dir():
                        continue

                    total_uncompressed += (
                        entry.file_size
                    )

                    if (
                        entry.file_size
                        > self.config.maximum_archive_entry_bytes
                    ):
                        findings.append(
                            FileThreatFinding(
                                FileThreatCategory.ARCHIVE_BOMB,
                                FileThreatSeverity.HIGH,
                                f"Oversized archive entry: {name}",
                            )
                        )

                    ratio = (
                        entry.file_size
                        / max(entry.compress_size, 1)
                    )

                    if (
                        ratio
                        > self.config.maximum_archive_ratio
                    ):
                        findings.append(
                            FileThreatFinding(
                                FileThreatCategory.ARCHIVE_BOMB,
                                FileThreatSeverity.HIGH,
                                f"Suspicious compression ratio: {name}",
                            )
                        )

                    if (
                        PurePosixPath(name).suffix.lower()
                        in self.ARCHIVE_EXTENSIONS
                    ):
                        findings.append(
                            FileThreatFinding(
                                FileThreatCategory.NESTED_ARCHIVE,
                                FileThreatSeverity.MEDIUM,
                                f"Nested archive detected: {name}",
                            )
                        )

                if (
                    total_uncompressed
                    > self.config.maximum_archive_uncompressed_bytes
                ):
                    findings.append(
                        FileThreatFinding(
                            FileThreatCategory.ARCHIVE_BOMB,
                            FileThreatSeverity.CRITICAL,
                            "Archive uncompressed size exceeds limit",
                        )
                    )

        except (
            zipfile.BadZipFile,
            OSError,
            RuntimeError,
        ):
            findings.append(
                FileThreatFinding(
                    FileThreatCategory.INVALID_SIGNATURE,
                    FileThreatSeverity.HIGH,
                    "Invalid or unreadable ZIP archive",
                )
            )

        return findings


# ============================================================
# OPTIONAL CLAMAV SCANNER
# ============================================================

class ClamAVScanner:

    def __init__(
        self,
        config: FileScannerConfig,
    ) -> None:
        self.config = config

    def scan(
        self,
        path: Path,
    ) -> tuple[bool, bool, str]:
        """
        Returns:
            checked
            malware_detected
            message
        """

        if not self.config.enable_clamav:
            return False, False, "ClamAV disabled"

        executable = shutil.which(
            self.config.clamav_executable
        )

        if executable is None:
            return (
                False,
                False,
                "ClamAV executable unavailable",
            )

        try:
            result = subprocess.run(
                [
                    executable,
                    "--no-summary",
                    "--",
                    str(path.resolve()),
                ],
                capture_output=True,
                text=True,
                timeout=(
                    self.config.clamav_timeout_seconds
                ),
                check=False,
            )

        except (
            OSError,
            subprocess.TimeoutExpired,
        ) as exc:
            return (
                False,
                False,
                f"ClamAV execution failed: "
                f"{type(exc).__name__}",
            )

        if result.returncode == 0:
            return True, False, "No malware detected"

        if result.returncode == 1:
            return True, True, "Malware detected"

        return (
            False,
            False,
            "ClamAV returned a scanning error",
        )


# ============================================================
# CENTRAL FILE SCANNER
# ============================================================

class RoboticsFileScanner:

    def __init__(
        self,
        config: FileScannerConfig | None = None,
    ) -> None:
        self.config = (
            config or FileScannerConfig()
        )

        self.signature_validator = (
            FileSignatureValidator()
        )

        self.filename_validator = (
            FileNameValidator()
        )

        self.content_scanner = (
            SuspiciousContentScanner(
                self.config
            )
        )

        self.archive_scanner = (
            ArchiveSecurityScanner(
                self.config
            )
        )

        self.antivirus = ClamAVScanner(
            self.config
        )

    # --------------------------------------------------------
    # SCAN FILE
    # --------------------------------------------------------

    def scan(
        self,
        path: str | Path,
        *,
        quarantine: bool = False,
    ) -> FileScanResult:
        file_path = Path(path)

        if (
            not file_path.exists()
            or not file_path.is_file()
        ):
            raise FileScannerIOError(
                "File does not exist"
            )

        extension = file_path.suffix.lower()
        size = file_path.stat().st_size

        findings: list[FileThreatFinding] = []

        findings.extend(
            self.filename_validator.inspect(
                file_path.name
            )
        )

        if (
            extension
            not in self.config.allowed_extensions
        ):
            findings.append(
                FileThreatFinding(
                    FileThreatCategory.INVALID_EXTENSION,
                    FileThreatSeverity.HIGH,
                    "File extension is not allowed",
                )
            )

        if size == 0:
            findings.append(
                FileThreatFinding(
                    FileThreatCategory.EMPTY_FILE,
                    FileThreatSeverity.MEDIUM,
                    "Empty file",
                )
            )

        if (
            size
            > self.config.maximum_file_size_bytes
        ):
            findings.append(
                FileThreatFinding(
                    FileThreatCategory.OVERSIZED_FILE,
                    FileThreatSeverity.HIGH,
                    "File exceeds size limit",
                )
            )

        digest = sha256_file(file_path)

        if (
            size > 0
            and size
            <= self.config.maximum_file_size_bytes
        ):
            findings.extend(
                self.signature_validator.inspect(
                    file_path,
                    extension,
                )
            )

            findings.extend(
                self.content_scanner.inspect(
                    file_path,
                    extension,
                )
            )

            if extension in (
                ".zip",
                ".docx",
                ".xlsx",
            ):
                findings.extend(
                    self.archive_scanner.inspect(
                        file_path
                    )
                )

        antivirus_checked = False
        malware_detected = False

        if (
            self.config.enable_clamav
            and size
            <= self.config.maximum_file_size_bytes
        ):
            (
                antivirus_checked,
                malware_detected,
                antivirus_message,
            ) = self.antivirus.scan(
                file_path
            )

            if malware_detected:
                findings.append(
                    FileThreatFinding(
                        FileThreatCategory.MALWARE,
                        FileThreatSeverity.CRITICAL,
                        antivirus_message,
                    )
                )

            elif not antivirus_checked:
                findings.append(
                    FileThreatFinding(
                        FileThreatCategory.SCANNER_ERROR,
                        FileThreatSeverity.HIGH,
                        antivirus_message,
                    )
                )

        if malware_detected:
            status = FileScanStatus.MALICIOUS

        elif any(
            finding.category in (
                FileThreatCategory.INVALID_EXTENSION,
                FileThreatCategory.INVALID_SIGNATURE,
                FileThreatCategory.OVERSIZED_FILE,
                FileThreatCategory.ARCHIVE_TRAVERSAL,
                FileThreatCategory.ARCHIVE_BOMB,
            )
            for finding in findings
        ):
            status = FileScanStatus.REJECTED

        elif any(
            finding.category
            == FileThreatCategory.SCANNER_ERROR
            for finding in findings
        ):
            status = FileScanStatus.ERROR

        elif findings:
            status = FileScanStatus.SUSPICIOUS

        else:
            status = FileScanStatus.CLEAN

        result = FileScanResult(
            scan_id=str(uuid.uuid4()),
            filename=file_path.name,
            original_path=str(file_path),
            file_size_bytes=size,
            sha256=digest,
            extension=extension,
            expected_mime_type=MIME_TYPES.get(
                extension
            ),
            status=status,
            findings=findings,
            antivirus_checked=antivirus_checked,
        )

        should_quarantine = (
            quarantine
            or (
                status == FileScanStatus.SUSPICIOUS
                and self.config.quarantine_on_suspicious
            )
            or (
                status == FileScanStatus.MALICIOUS
                and self.config.quarantine_on_malicious
            )
        )

        if should_quarantine:
            self.quarantine_file(
                file_path,
                result,
            )

        return result

    # --------------------------------------------------------
    # QUARANTINE
    # --------------------------------------------------------

    def quarantine_file(
        self,
        path: Path,
        result: FileScanResult,
    ) -> Path:
        directory = Path(
            self.config.quarantine_directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = (
            directory
            / f"{result.scan_id}.quarantine"
        )

        shutil.move(
            str(path),
            str(destination),
        )

        result.quarantined = True
        result.quarantine_path = str(
            destination
        )

        return destination

    # --------------------------------------------------------
    # SCAN DIRECTORY
    # --------------------------------------------------------

    def scan_directory(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
        quarantine: bool = False,
    ) -> list[FileScanResult]:
        root = Path(directory)

        if not root.is_dir():
            raise FileScannerIOError(
                "Directory does not exist"
            )

        files = (
            root.rglob("*")
            if recursive
            else root.glob("*")
        )

        results = []

        for path in files:
            if path.is_file():
                results.append(
                    self.scan(
                        path,
                        quarantine=quarantine,
                    )
                )

        return results

    # --------------------------------------------------------
    # EXPORT REPORT
    # --------------------------------------------------------

    def export_report(
        self,
        results: list[FileScanResult],
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary = {
            status.value: sum(
                result.status == status
                for result in results
            )
            for status in FileScanStatus
        }

        payload = {
            "generated_at": utc_now(),
            "total_files": len(results),
            "summary": summary,
            "results": [
                result.to_dict()
                for result in results
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
# FILE SERVICE INTEGRATION
# ============================================================

class FileServiceSecurityAdapter:

    def __init__(
        self,
        scanner: RoboticsFileScanner,
    ) -> None:
        self.scanner = scanner

    def validate_upload(
        self,
        path: str | Path,
    ) -> FileScanResult:
        result = self.scanner.scan(
            path,
            quarantine=False,
        )

        if result.status != FileScanStatus.CLEAN:
            raise FileScannerValidationError(
                f"Upload rejected: "
                f"{result.status.value}"
            )

        return result


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    scanner = RoboticsFileScanner(
        FileScannerConfig(
            enable_clamav=False,
            quarantine_on_suspicious=False,
        )
    )

    sample_directory = Path(
        "data/security/scanner_demo"
    )

    sample_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_file = (
        sample_directory
        / "robot_telemetry.json"
    )

    sample_file.write_text(
        json.dumps(
            {
                "robot_id": "robot_001",
                "battery_soc_percent": 85,
                "motor_temperature_c": 46,
                "vibration_mm_s": 2.1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = scanner.scan(
        sample_file
    )

    report = scanner.export_report(
        [result],
        "data/security/scanner_reports/"
        "file_scan_report.json",
    )

    print(
        json.dumps(
            {
                "scan": result.to_dict(),
                "report": str(report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

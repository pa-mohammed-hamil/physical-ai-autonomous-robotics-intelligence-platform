#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/health_check.py

Production-oriented, read-only health diagnostics for:

- Backend FastAPI service
- Machine learning inference service
- Model registry
- PostgreSQL
- Redis
- RAG/vector index
- ROS 2 environment
- Docker daemon
- CPU, RAM and disk
- NVIDIA GPU
- Robot telemetry freshness
- Safety configuration
- Emergency-stop and safety-interlock observations
- Local storage and model artifacts

Features:
- Async HTTP health checks
- Per-service timeouts
- Retry with exponential backoff
- Parallel service checks
- Required/optional dependency classification
- System resource monitoring
- Local file integrity checks
- Robot safety fail-closed diagnostics
- JSON reports
- CLI output
- Exit codes suitable for CI/CD
- No robot actuation or E-stop reset

Dependencies:
    pip install httpx psutil

Optional:
    pip install redis asyncpg pynvml

Examples:
    python scripts/health_check.py

    python scripts/health_check.py --full

    python scripts/health_check.py --json

    python scripts/health_check.py --strict

    python scripts/health_check.py --config config/health_check.json

    python scripts/health_check.py --output reports/health/report.json

    python scripts/health_check.py --services backend ml redis

Exit codes:
    0 = Healthy
    1 = Degraded or unknown
    2 = Unhealthy
    3 = Configuration or execution error

Security:
- Never executes physical robot commands.
- Never resets an emergency stop.
- Never bypasses safety interlocks.
- Does not expose credentials in reports.
- Remote health endpoints should be trusted and authenticated.
- A healthy result does not certify a robot as safe.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import urlsplit


LOGGER = logging.getLogger("robotics.health_check")

SCHEMA_VERSION = "1.0.0"


# ============================================================
# Enumerations
# ============================================================

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentCategory(str, Enum):
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    ML = "machine_learning"
    STORAGE = "storage"
    SYSTEM = "system"
    ROBOTICS = "robotics"
    SAFETY = "safety"
    INFRASTRUCTURE = "infrastructure"


# ============================================================
# Data Models
# ============================================================

@dataclass
class HealthCheckResult:
    name: str
    category: str
    status: str

    required: bool
    message: str

    latency_ms: float = 0.0

    details: dict[str, Any] = field(
        default_factory=dict
    )

    checked_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )


@dataclass
class HealthCheckConfig:
    backend_url: str = "http://127.0.0.1:8000/health"
    ml_url: str = "http://127.0.0.1:8001/health"

    backend_required: bool = True
    ml_required: bool = False

    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_required: bool = False

    postgres_dsn: str = ""
    postgres_required: bool = False

    project_root: str = "."
    model_registry_path: str = "storage/model_registry"
    rag_index_path: str = "storage/rag_index"
    telemetry_path: str = "data/demo/robot_telemetry.csv"

    check_ros: bool = False
    check_docker: bool = False
    check_gpu: bool = False
    check_robot_safety: bool = True

    http_timeout_seconds: float = 5.0
    retries: int = 2
    retry_backoff_seconds: float = 0.4

    max_cpu_percent: float = 95.0
    max_memory_percent: float = 95.0
    max_disk_percent: float = 95.0

    max_telemetry_age_seconds: float = 5.0

    minimum_free_disk_gb: float = 1.0

    allowed_http_hosts: list[str] = field(
        default_factory=lambda: [
            "127.0.0.1",
            "localhost",
            "::1",
        ]
    )

    simulation_only: bool = True

    def validate(self) -> None:
        if self.http_timeout_seconds <= 0:
            raise ValueError(
                "HTTP timeout must be positive"
            )

        if self.retries < 0 or self.retries > 10:
            raise ValueError(
                "Retries must be between 0 and 10"
            )

        if self.retry_backoff_seconds < 0:
            raise ValueError(
                "Retry backoff cannot be negative"
            )

        if self.max_telemetry_age_seconds <= 0:
            raise ValueError(
                "Telemetry age threshold must be positive"
            )

        if not self.simulation_only:
            raise ValueError(
                "Physical robot health execution is unsupported"
            )

        for name in (
            "max_cpu_percent",
            "max_memory_percent",
            "max_disk_percent",
        ):
            value = getattr(self, name)

            if not 0 <= value <= 100:
                raise ValueError(
                    f"{name} must be between 0 and 100"
                )

        for url in (
            self.backend_url,
            self.ml_url,
        ):
            parsed = urlsplit(url)

            if parsed.scheme not in {
                "http",
                "https",
            }:
                raise ValueError(
                    "Health endpoint must use HTTP or HTTPS"
                )

            if (
                parsed.username
                or parsed.password
                or parsed.hostname not in self.allowed_http_hosts
            ):
                raise ValueError(
                    "Health endpoint host is not allowlisted"
                )


@dataclass
class HealthReport:
    report_id: str
    schema_version: str

    status: str

    started_at: str
    completed_at: str
    duration_ms: float

    total_checks: int

    healthy_count: int
    degraded_count: int
    unhealthy_count: int
    unknown_count: int

    checks: list[dict[str, Any]]

    simulation_only: bool = True
    physical_execution_authorized: bool = False
    certified_safe: bool = False


# ============================================================
# Helpers
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def parse_boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        value = value.strip().lower()

        if value in ("true", "yes", "1", "on"):
            return True

        if value in ("false", "no", "0", "off"):
            return False

    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def make_result(
    name: str,
    category: ComponentCategory,
    status: HealthStatus,
    message: str,
    *,
    required: bool = False,
    latency_ms: float = 0.0,
    details: dict[str, Any] | None = None,
) -> HealthCheckResult:
    return HealthCheckResult(
        name=name,
        category=category.value,
        status=status.value,
        required=required,
        message=message,
        latency_ms=round(
            latency_ms,
            3,
        ),
        details=details or {},
    )


def project_path(
    config: HealthCheckConfig,
    relative: str,
) -> Path:
    root = Path(
        config.project_root
    ).expanduser().resolve()

    candidate = Path(
        relative
    ).expanduser()

    if candidate.is_absolute():
        return candidate.resolve()

    return (
        root / candidate
    ).resolve()


# ============================================================
# HTTP Health Checker
# ============================================================

class HTTPHealthChecker:
    def __init__(
        self,
        config: HealthCheckConfig,
    ) -> None:
        self.config = config

    async def check(
        self,
        *,
        name: str,
        url: str,
        category: ComponentCategory,
        required: bool,
    ) -> HealthCheckResult:
        try:
            import httpx
        except ImportError:
            return make_result(
                name,
                category,
                HealthStatus.UNKNOWN,
                "httpx is not installed",
                required=required,
            )

        parsed = urlsplit(url)

        if parsed.hostname not in (
            self.config.allowed_http_hosts
        ):
            return make_result(
                name,
                category,
                HealthStatus.UNHEALTHY,
                "Health endpoint host is not allowlisted",
                required=required,
            )

        started = time.perf_counter()

        last_error = "Unknown HTTP error"

        # Do not follow redirects to untrusted destinations.
        async with httpx.AsyncClient(
            timeout=self.config.http_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            for attempt in range(
                self.config.retries + 1
            ):
                try:
                    response = await client.get(
                        url
                    )

                    latency_ms = (
                        time.perf_counter() - started
                    ) * 1000

                    if response.status_code != 200:
                        return make_result(
                            name,
                            category,
                            HealthStatus.UNHEALTHY,
                            f"HTTP {response.status_code}",
                            required=required,
                            latency_ms=latency_ms,
                            details={
                                "http_status": response.status_code,
                            },
                        )

                    content_type = response.headers.get(
                        "content-type",
                        "",
                    ).lower()

                    if "json" not in content_type:
                        return make_result(
                            name,
                            category,
                            HealthStatus.DEGRADED,
                            "Endpoint returned HTTP 200 without JSON health status",
                            required=required,
                            latency_ms=latency_ms,
                        )

                    payload = response.json()

                    if not isinstance(payload, dict):
                        raise ValueError(
                            "Health payload must be a JSON object"
                        )

                    remote_status = str(
                        payload.get(
                            "status",
                            "unknown",
                        )
                    ).lower()

                    if remote_status in {
                        "healthy",
                        "ok",
                        "ready",
                        "up",
                    }:
                        status = HealthStatus.HEALTHY

                    elif remote_status in {
                        "degraded",
                        "warning",
                    }:
                        status = HealthStatus.DEGRADED

                    elif remote_status in {
                        "unhealthy",
                        "error",
                        "down",
                        "failed",
                    }:
                        status = HealthStatus.UNHEALTHY

                    else:
                        status = HealthStatus.UNKNOWN

                    return make_result(
                        name,
                        category,
                        status,
                        f"Service status: {remote_status}",
                        required=required,
                        latency_ms=latency_ms,
                        details={
                            "http_status": response.status_code,
                            "service_status": remote_status,
                        },
                    )

                except (
                    httpx.RequestError,
                    ValueError,
                ) as exc:
                    last_error = type(exc).__name__

                    if attempt < self.config.retries:
                        await asyncio.sleep(
                            self.config.retry_backoff_seconds
                            * (2 ** attempt)
                        )

        return make_result(
            name,
            category,
            HealthStatus.UNHEALTHY,
            f"Service unavailable: {last_error}",
            required=required,
            latency_ms=(
                time.perf_counter() - started
            ) * 1000,
        )


# ============================================================
# System Resource Checker
# ============================================================

class SystemHealthChecker:
    def __init__(
        self,
        config: HealthCheckConfig,
    ) -> None:
        self.config = config

    async def check(self) -> HealthCheckResult:
        try:
            import psutil
        except ImportError:
            return make_result(
                "system_resources",
                ComponentCategory.SYSTEM,
                HealthStatus.UNKNOWN,
                "psutil is not installed",
                required=True,
            )

        started = time.perf_counter()

        cpu_percent = await asyncio.to_thread(
            psutil.cpu_percent,
            interval=0.1,
        )

        memory = psutil.virtual_memory()

        root = Path(
            self.config.project_root
        ).resolve()

        disk = psutil.disk_usage(
            str(root)
        )

        free_gb = (
            disk.free / (1024 ** 3)
        )

        warnings = []

        if cpu_percent > self.config.max_cpu_percent:
            warnings.append(
                "CPU usage exceeds threshold"
            )

        if memory.percent > self.config.max_memory_percent:
            warnings.append(
                "Memory usage exceeds threshold"
            )

        if disk.percent > self.config.max_disk_percent:
            warnings.append(
                "Disk usage exceeds threshold"
            )

        if free_gb < self.config.minimum_free_disk_gb:
            warnings.append(
                "Available disk space is below threshold"
            )

        status = (
            HealthStatus.DEGRADED
            if warnings
            else HealthStatus.HEALTHY
        )

        return make_result(
            "system_resources",
            ComponentCategory.SYSTEM,
            status,
            "; ".join(warnings)
            if warnings
            else "System resource usage is within thresholds",
            required=True,
            latency_ms=(
                time.perf_counter() - started
            ) * 1000,
            details={
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "disk_free_gb": round(
                    free_gb,
                    3,
                ),
                "cpu_count": psutil.cpu_count(),
            },
        )


# ============================================================
# Redis Health Checker
# ============================================================

class RedisHealthChecker:
    def __init__(
        self,
        config: HealthCheckConfig,
    ) -> None:
        self.config = config

    async def check(self) -> HealthCheckResult:
        try:
            import redis.asyncio as redis
        except ImportError:
            return make_result(
                "redis",
                ComponentCategory.CACHE,
                HealthStatus.UNKNOWN,
                "redis package is not installed",
                required=self.config.redis_required,
            )

        started = time.perf_counter()

        client = None

        try:
            client = redis.from_url(
                self.config.redis_url,
                socket_connect_timeout=3,
                socket_timeout=3,
                decode_responses=True,
            )

            response = await asyncio.wait_for(
                client.ping(),
                timeout=4,
            )

            status = (
                HealthStatus.HEALTHY
                if response
                else HealthStatus.UNHEALTHY
            )

            return make_result(
                "redis",
                ComponentCategory.CACHE,
                status,
                "Redis PING successful"
                if response
                else "Redis PING failed",
                required=self.config.redis_required,
                latency_ms=(
                    time.perf_counter() - started
                ) * 1000,
            )

        except Exception as exc:
            return make_result(
                "redis",
                ComponentCategory.CACHE,
                HealthStatus.UNHEALTHY,
                f"Redis unavailable: {type(exc).__name__}",
                required=self.config.redis_required,
            )

        finally:
            if client is not None:
                await client.aclose()


# ============================================================
# PostgreSQL Health Checker
# ============================================================

class PostgreSQLHealthChecker:
    def __init__(
        self,
        config: HealthCheckConfig,
    ) -> None:
        self.config = config

    async def check(self) -> HealthCheckResult:
        if not self.config.postgres_dsn:
            return make_result(
                "postgresql",
                ComponentCategory.DATABASE,
                HealthStatus.UNKNOWN,
                "PostgreSQL DSN is not configured",
                required=self.config.postgres_required,
            )

        try:
            import asyncpg
        except ImportError:
            return make_result(
                "postgresql",
                ComponentCategory.DATABASE,
                HealthStatus.UNKNOWN,
                "asyncpg is not installed",
                required=self.config.postgres_required,
            )

        started = time.perf_counter()

        connection = None

        try:
            connection = await asyncio.wait_for(
                asyncpg.connect(
                    dsn=self.config.postgres_dsn,
                    timeout=4,
                ),
                timeout=5,
            )

            value = await connection.fetchval(
                "SELECT 1"
            )

            status = (
                HealthStatus.HEALTHY
                if value == 1
                else HealthStatus.UNHEALTHY
            )

            return make_result(
                "postgresql",
                ComponentCategory.DATABASE,
                status,
                "PostgreSQL query successful"
                if value == 1
                else "PostgreSQL query returned unexpected result",
                required=self.config.postgres_required,
                latency_ms=(
                    time.perf_counter() - started
                ) * 1000,
            )

        except Exception as exc:
            return make_result(
                "postgresql",
                ComponentCategory.DATABASE,
                HealthStatus.UNHEALTHY,
                f"PostgreSQL unavailable: {type(exc).__name__}",
                required=self.config.postgres_required,
            )

        finally:
            if connection is not None:
                await connection.close()


# ============================================================
# Model Registry Health Checker
# ============================================================

class ModelRegistryHealthChecker:
    def __init__(
        self,
        config: HealthCheckConfig,
    ) -> None:
        self.config = config

    async def check(self) -> HealthCheckResult:
        registry_path = project_path(
            self.config,
            self.config.model_registry_path,
        )

        if not registry_path.is_dir():
            return make_result(
                "model_registry",
                ComponentCategory.ML,
                HealthStatus.UNKNOWN,
                "Model registry directory not initialized",
                details={
                    "path": str(registry_path),
                },
            )

        model_files = []

        for extension in (
            "*.pt",
            "*.pth",
            "*.onnx",
            "*.safetensors",
        ):
            model_files.extend(
                registry_path.rglob(
                    extension
                )
            )

        return make_result(
            "model_registry",
            ComponentCategory.ML,
            HealthStatus.HEALTHY,
            "Model registry directory accessible",
            details={
                "model_artifact_count": len(model_files),
                "path": str(registry_path),
            },
        )


# ============================================================
# RAG Index Health Checker
# ============================================================

class RAGIndexHealthChecker:
    def __init__(
        self,
        config: HealthCheckConfig,
    ) -> None:
        self.config = config

    async def check(self) -> HealthCheckResult:
        index_path = project_path(
            self.config,
            self.config.rag_index_path,
        )

        if not index_path.is_dir():
            return make_result(
                "rag_index",
                ComponentCategory.STORAGE,
                HealthStatus.UNKNOWN,
                "RAG index directory not initialized",
            )

        manifests = list(
            index_path.rglob(
                "current.json"
            )
        )

        if not manifests:
            return make_result(
                "rag_index",
                ComponentCategory.STORAGE,
                HealthStatus.UNKNOWN,
                "No active RAG index manifests found",
            )

        valid_count = 0
        invalid_count = 0

        for manifest_path in manifests:
            try:
                payload = json.loads(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )

                generation_id = payload.get(
                    "generation_id"
                )

                if (
                    not isinstance(generation_id, str)
                    or not generation_id.isalnum()
                ):
                    invalid_count += 1
                    continue

                generation = (
                    manifest_path.parent
                    / "generations"
                    / generation_id
                )

                if (
                    generation / "manifest.json"
                ).is_file():
                    valid_count += 1
                else:
                    invalid_count += 1

            except (
                OSError,
                ValueError,
            ):
                invalid_count += 1

        status = (
            HealthStatus.DEGRADED
            if invalid_count
            else HealthStatus.HEALTHY
        )

        return make_result(
            "rag_index",
            ComponentCategory.STORAGE,
            status,
            "RAG index manifest inspection completed",
            details={
                "valid_manifests": valid_count,
                "invalid_manifests": invalid_count,
                "note": (
                    "Manifest presence only; full index "
                    "checksum and retrieval verification "
                    "require build_rag_index.py verify."
                ),
            },
        )


# ============================================================
# ROS 2 Health Checker
# ============================================================

class ROSHealthChecker:
    async def check(self) -> HealthCheckResult:
        ros_distro = os.environ.get(
            "ROS_DISTRO"
        )

        ros_command = shutil.which(
            "ros2"
        )

        if not ros_distro or not ros_command:
            return make_result(
                "ros2",
                ComponentCategory.ROBOTICS,
                HealthStatus.UNKNOWN,
                "ROS 2 environment not available",
                details={
                    "ros_distro": ros_distro,
                    "ros_cli_found": bool(ros_command),
                },
            )

        return make_result(
            "ros2",
            ComponentCategory.ROBOTICS,
            HealthStatus.HEALTHY,
            "ROS 2 environment variables and CLI available",
            details={
                "ros_distro": ros_distro,
                "note": (
                    "ROS graph, nodes and hardware "
                    "communication were not tested."
                ),
            },
        )


# ============================================================
# Docker Health Checker
# ============================================================

class DockerHealthChecker:
    async def check(self) -> HealthCheckResult:
        if not shutil.which("docker"):
            return make_result(
                "docker",
                ComponentCategory.INFRASTRUCTURE,
                HealthStatus.UNKNOWN,
                "Docker CLI is not installed",
            )

        started = time.perf_counter()

        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "info",
                "--format",
                "{{.ServerVersion}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=5,
                )

            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()

                return make_result(
                    "docker",
                    ComponentCategory.INFRASTRUCTURE,
                    HealthStatus.UNHEALTHY,
                    "Docker daemon check timed out",
                )

            if process.returncode != 0:
                return make_result(
                    "docker",
                    ComponentCategory.INFRASTRUCTURE,
                    HealthStatus.UNHEALTHY,
                    "Docker daemon unavailable",
                )

            return make_result(
                "docker",
                ComponentCategory.INFRASTRUCTURE,
                HealthStatus.HEALTHY,
                "Docker daemon available",
                latency_ms=(
                    time.perf_counter() - started
                ) * 1000,
                details={
                    "server_version": (
                        stdout.decode(
                            errors="replace"
                        ).strip()
                    ),
                },
            )

        except OSError as exc:
            return make_result(
                "docker",
                ComponentCategory.INFRASTRUCTURE,
                HealthStatus.UNKNOWN,
                f"Docker inspection failed: {type(exc).__name__}",
            )


# ============================================================
# GPU Health Checker
# ============================================================

class GPUHealthChecker:
    async def check(self) -> HealthCheckResult:
        try:
            import torch
        except ImportError:
            return make_result(
                "gpu",
                ComponentCategory.ML,
                HealthStatus.UNKNOWN,
                "PyTorch is not installed",
            )

        if not torch.cuda.is_available():
            return make_result(
                "gpu",
                ComponentCategory.ML,
                HealthStatus.UNKNOWN,
                "CUDA GPU is unavailable",
            )

        devices = []

        for index in range(
            torch.cuda.device_count()
        ):
            properties = (
                torch.cuda.get_device_properties(
                    index
                )
            )

            devices.append(
                {
                    "device_index": index,
                    "name": properties.name,
                    "total_memory_gb": round(
                        properties.total_memory
                        / (1024 ** 3),
                        3,
                    ),
                }
            )

        return make_result(
            "gpu",
            ComponentCategory.ML,
            HealthStatus.HEALTHY,
            "CUDA GPU devices detected",
            details={
                "device_count": len(devices),
                "devices": devices,
                "note": (
                    "GPU computation and thermal "
                    "stress tests were not performed."
                ),
            },
        )


# ============================================================
# Telemetry Health Checker
# ============================================================

class TelemetryHealthChecker:
    def __init__(
        self,
        config: HealthCheckConfig,
    ) -> None:
        self.config = config

    async def check(self) -> HealthCheckResult:
        path = project_path(
            self.config,
            self.config.telemetry_path,
        )

        if not path.is_file():
            return make_result(
                "robot_telemetry",
                ComponentCategory.ROBOTICS,
                HealthStatus.UNKNOWN,
                "Telemetry file not found",
                details={
                    "path": str(path),
                },
            )

        age_seconds = max(
            0.0,
            time.time() - path.stat().st_mtime,
        )

        if age_seconds > self.config.max_telemetry_age_seconds:
            status = HealthStatus.DEGRADED
            message = "Telemetry file is stale"
        else:
            status = HealthStatus.HEALTHY
            message = "Telemetry file was recently updated"

        return make_result(
            "robot_telemetry",
            ComponentCategory.ROBOTICS,
            status,
            message,
            details={
                "file_age_seconds": round(
                    age_seconds,
                    3,
                ),
                "maximum_age_seconds": (
                    self.config.max_telemetry_age_seconds
                ),
                "size_bytes": path.stat().st_size,
                "note": (
                    "File modification time is not proof "
                    "of live sensor freshness."
                ),
            },
        )


# ============================================================
# Safety Configuration Checker
# ============================================================

class RobotSafetyHealthChecker:
    REQUIRED_SAFE_SETTINGS = {
        "PHYSICAL_ROBOT_CONTROL_ENABLED": False,
        "ACTUATOR_COMMANDS_ENABLED": False,
        "REMOTE_MOTION_COMMANDS_ENABLED": False,
        "AUTONOMOUS_PHYSICAL_EXECUTION_ENABLED": False,
        "EMERGENCY_STOP_RESET_ENABLED": False,
        "SAFETY_INTERLOCKS_REQUIRED": True,
    }

    def __init__(
        self,
        config: HealthCheckConfig,
    ) -> None:
        self.config = config

    @staticmethod
    def read_env_file(
        path: Path,
    ) -> dict[str, str]:
        values = {}

        if not path.is_file():
            return values

        for line in path.read_text(
            encoding="utf-8",
        ).splitlines():
            line = line.strip()

            if (
                not line
                or line.startswith("#")
                or "=" not in line
            ):
                continue

            key, value = line.split(
                "=",
                1,
            )

            values[key.strip()] = (
                value.strip().strip('"').strip("'")
            )

        return values

    async def check(self) -> HealthCheckResult:
        env_path = project_path(
            self.config,
            ".env",
        )

        settings = self.read_env_file(
            env_path
        )

        missing = []
        unsafe = []

        for key, expected in (
            self.REQUIRED_SAFE_SETTINGS.items()
        ):
            if key not in settings:
                missing.append(key)
                continue

            observed = parse_boolean(
                settings[key]
            )

            if observed is not expected:
                unsafe.append(key)

        if unsafe:
            status = HealthStatus.UNHEALTHY
            message = (
                "Unsafe or invalid robot execution settings detected"
            )

        elif missing:
            status = HealthStatus.UNKNOWN
            message = (
                "Required simulation safety settings are missing"
            )

        else:
            status = HealthStatus.HEALTHY
            message = (
                "Simulation-only safety configuration matches defaults"
            )

        return make_result(
            "robot_safety_configuration",
            ComponentCategory.SAFETY,
            status,
            message,
            required=True,
            details={
                "missing_settings": missing,
                "unsafe_settings": unsafe,
                "simulation_only": True,
                "physical_execution_authorized": False,
                "note": (
                    "Configuration inspection is not "
                    "hardware interlock verification."
                ),
            },
        )


# ============================================================
# Health Check Orchestrator
# ============================================================

class HealthCheckOrchestrator:
    def __init__(
        self,
        config: HealthCheckConfig,
    ) -> None:
        config.validate()

        self.config = config

        self.http_checker = HTTPHealthChecker(
            config
        )

        self.available_checks: dict[
            str,
            Callable[
                [],
                Awaitable[HealthCheckResult],
            ],
        ] = {
            "backend": lambda: self.http_checker.check(
                name="backend",
                url=config.backend_url,
                category=ComponentCategory.API,
                required=config.backend_required,
            ),
            "ml": lambda: self.http_checker.check(
                name="ml",
                url=config.ml_url,
                category=ComponentCategory.ML,
                required=config.ml_required,
            ),
            "redis": RedisHealthChecker(
                config
            ).check,
            "postgresql": PostgreSQLHealthChecker(
                config
            ).check,
            "system": SystemHealthChecker(
                config
            ).check,
            "model_registry": ModelRegistryHealthChecker(
                config
            ).check,
            "rag_index": RAGIndexHealthChecker(
                config
            ).check,
            "telemetry": TelemetryHealthChecker(
                config
            ).check,
            "safety": RobotSafetyHealthChecker(
                config
            ).check,
            "ros": ROSHealthChecker().check,
            "docker": DockerHealthChecker().check,
            "gpu": GPUHealthChecker().check,
        }

    def default_services(
        self,
        full: bool,
    ) -> list[str]:
        services = [
            "backend",
            "ml",
            "redis",
            "postgresql",
            "system",
            "model_registry",
            "rag_index",
            "telemetry",
        ]

        if self.config.check_robot_safety:
            services.append("safety")

        if full or self.config.check_ros:
            services.append("ros")

        if full or self.config.check_docker:
            services.append("docker")

        if full or self.config.check_gpu:
            services.append("gpu")

        return services

    async def _safe_check(
        self,
        name: str,
    ) -> HealthCheckResult:
        try:
            return await asyncio.wait_for(
                self.available_checks[name](),
                timeout=max(
                    10.0,
                    self.config.http_timeout_seconds
                    * (self.config.retries + 1)
                    + 5.0,
                ),
            )

        except asyncio.TimeoutError:
            return make_result(
                name,
                ComponentCategory.INFRASTRUCTURE,
                HealthStatus.UNKNOWN,
                "Health check timed out",
            )

        except Exception as exc:
            LOGGER.exception(
                "Health check failed: %s",
                name,
            )

            return make_result(
                name,
                ComponentCategory.INFRASTRUCTURE,
                HealthStatus.UNKNOWN,
                f"Check error: {type(exc).__name__}",
            )

    async def run(
        self,
        *,
        services: list[str] | None = None,
        full: bool = False,
    ) -> HealthReport:
        started_at = utc_now()
        started = time.perf_counter()

        selected = (
            services
            if services is not None
            else self.default_services(full)
        )

        unknown = [
            name
            for name in selected
            if name not in self.available_checks
        ]

        if unknown:
            raise ValueError(
                f"Unknown health checks: {unknown}"
            )

        if not selected:
            raise ValueError(
                "At least one health check must be selected"
            )

        results = await asyncio.gather(
            *[
                self._safe_check(name)
                for name in selected
            ]
        )

        counts = {
            status.value: sum(
                result.status == status.value
                for result in results
            )
            for status in HealthStatus
        }

        required_unhealthy = any(
            result.required
            and result.status == HealthStatus.UNHEALTHY.value
            for result in results
        )

        required_unknown = any(
            result.required
            and result.status == HealthStatus.UNKNOWN.value
            for result in results
        )

        any_nonhealthy = any(
            result.status != HealthStatus.HEALTHY.value
            for result in results
        )

        if required_unhealthy:
            overall_status = HealthStatus.UNHEALTHY

        elif required_unknown or any_nonhealthy:
            overall_status = HealthStatus.DEGRADED

        else:
            overall_status = HealthStatus.HEALTHY

        return HealthReport(
            report_id=uuid.uuid4().hex,
            schema_version=SCHEMA_VERSION,
            status=overall_status.value,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=round(
                (
                    time.perf_counter() - started
                ) * 1000,
                3,
            ),
            total_checks=len(results),
            healthy_count=counts["healthy"],
            degraded_count=counts["degraded"],
            unhealthy_count=counts["unhealthy"],
            unknown_count=counts["unknown"],
            checks=[
                asdict(result)
                for result in results
            ],
        )


# ============================================================
# Report Export
# ============================================================

def save_report(
    report: HealthReport,
    output_path: str | Path,
) -> Path:
    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                asdict(report),
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

            stream.flush()
            os.fsync(stream.fileno())

        os.replace(
            temporary,
            path,
        )

    finally:
        if temporary.exists():
            temporary.unlink()

    return path


# ============================================================
# Console Output
# ============================================================

def print_report(
    report: HealthReport,
) -> None:
    print()
    print("=" * 72)
    print("PHYSICAL AI & AUTONOMOUS ROBOTICS HEALTH CHECK")
    print("=" * 72)

    print(
        f"Overall status: {report.status.upper()}"
    )

    print(
        f"Checks: {report.total_checks}"
    )

    print(
        f"Healthy: {report.healthy_count} | "
        f"Degraded: {report.degraded_count} | "
        f"Unhealthy: {report.unhealthy_count} | "
        f"Unknown: {report.unknown_count}"
    )

    print("-" * 72)

    for check in report.checks:
        name = check["name"]
        status = check["status"].upper()
        message = check["message"]

        print(
            f"{name:<28} {status:<12} {message}"
        )

    print("-" * 72)

    print(
        f"Duration: {report.duration_ms:.2f} ms"
    )

    print(
        "Physical execution authorized: NO"
    )

    print(
        "Safety certification: NOT PROVIDED"
    )

    print("=" * 72)


# ============================================================
# Configuration Loading
# ============================================================

def load_config(
    path: str | None,
) -> HealthCheckConfig:
    if path is None:
        config = HealthCheckConfig()
        config.validate()
        return config

    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}"
        )

    payload = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Configuration must be a JSON object"
        )

    config = HealthCheckConfig(
        **payload
    )

    config.validate()

    return config


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Physical AI platform health checks"
        )
    )

    parser.add_argument(
        "--config",
        default=None,
    )

    parser.add_argument(
        "--full",
        action="store_true",
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
    )

    parser.add_argument(
        "--services",
        nargs="+",
        default=None,
        choices=[
            "backend",
            "ml",
            "redis",
            "postgresql",
            "system",
            "model_registry",
            "rag_index",
            "telemetry",
            "safety",
            "ros",
            "docker",
            "gpu",
        ],
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--backend-url",
        default=None,
    )

    parser.add_argument(
        "--ml-url",
        default=None,
    )

    parser.add_argument(
        "--project-root",
        default=None,
    )

    return parser


async def async_main() -> int:
    args = build_parser().parse_args()

    config = load_config(
        args.config
    )

    if args.backend_url:
        config.backend_url = args.backend_url

    if args.ml_url:
        config.ml_url = args.ml_url

    if args.project_root:
        config.project_root = args.project_root

    config.validate()

    orchestrator = HealthCheckOrchestrator(
        config
    )

    report = await orchestrator.run(
        services=args.services,
        full=args.full,
    )

    if args.output:
        save_report(
            report,
            args.output,
        )

    if args.json:
        print(
            json.dumps(
                asdict(report),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        )

    else:
        print_report(
            report
        )

    if report.status == HealthStatus.UNHEALTHY.value:
        return 2

    if report.status != HealthStatus.HEALTHY.value:
        return 1 if args.strict else 0

    return 0


def main() -> int:
    try:
        return asyncio.run(
            async_main()
        )

    except (
        ValueError,
        OSError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        LOGGER.error(
            "Health check failed: %s",
            exc,
        )

        return 3


if __name__ == "__main__":
    sys.exit(
        main()
    )
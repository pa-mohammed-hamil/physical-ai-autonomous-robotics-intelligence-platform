"""Network Security module.

TODO: implement and test; no real robot control is enabled.
"""
"""Prompt Injection module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/security/network_security.py

Python 3.11+
Standard library only.

Features:
    - IPv4 and IPv6 address validation
    - CIDR allowlist and denylist
    - Network endpoint authorization
    - TLS transport policy
    - Trusted proxy configuration
    - Per-client rate limiting
    - Request body size limits
    - Network protocol allowlisting
    - Suspicious request heuristics
    - Security audit events
    - JSON reporting
    - Thread-safe policy evaluation

Security notes:
    This is an application-layer policy engine.

    It does not configure operating-system
    firewalls, network interfaces, ROS 2 DDS
    security, MQTT broker ACLs or TLS servers.

    Physical robot safety must be enforced
    independently of network availability.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
import threading
import time
import uuid

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class NetworkSecurityError(Exception):
    pass


class NetworkValidationError(NetworkSecurityError):
    pass


class NetworkAccessDenied(NetworkSecurityError):
    pass


class NetworkRateLimitExceeded(NetworkSecurityError):
    pass


# ============================================================
# ENUMS
# ============================================================

class NetworkProtocol(str, Enum):
    HTTP = "http"
    HTTPS = "https"
    WS = "ws"
    WSS = "wss"
    MQTT = "mqtt"
    MQTTS = "mqtts"
    ROS2 = "ros2"
    GRPC = "grpc"
    GRPCS = "grpcs"


class NetworkAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class NetworkSecuritySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class NetworkSecurityEventType(str, Enum):
    ACCESS_ALLOWED = "access_allowed"
    ACCESS_DENIED = "access_denied"
    RATE_LIMITED = "rate_limited"
    INVALID_REQUEST = "invalid_request"
    TLS_REQUIRED = "tls_required"
    SUSPICIOUS_REQUEST = "suspicious_request"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NetworkValidationError(
            f"{name} is required"
        )

    return value.strip()


def parse_ip(
    value: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    try:
        return ipaddress.ip_address(
            require_text(value, "ip_address")
        )
    except ValueError as exc:
        raise NetworkValidationError(
            "Invalid IP address"
        ) from exc


def parse_network(
    value: str,
) -> ipaddress.IPv4Network | ipaddress.IPv6Network:
    try:
        return ipaddress.ip_network(
            require_text(value, "network"),
            strict=False,
        )
    except ValueError as exc:
        raise NetworkValidationError(
            "Invalid CIDR network"
        ) from exc


def json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise NetworkValidationError(
            "Invalid JSON-compatible data"
        ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class NetworkSecurityConfig:
    require_tls: bool = True

    maximum_request_body_bytes: int = (
        10 * 1024 * 1024
    )

    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60

    maximum_audit_events: int = 20000

    allowed_protocols: frozenset[
        NetworkProtocol
    ] = field(
        default_factory=lambda: frozenset({
            NetworkProtocol.HTTPS,
            NetworkProtocol.WSS,
            NetworkProtocol.MQTTS,
            NetworkProtocol.GRPCS,
        })
    )

    trusted_proxy_networks: tuple[str, ...] = ()

    allow_private_networks: bool = True

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        for name in (
            "maximum_request_body_bytes",
            "rate_limit_requests",
            "rate_limit_window_seconds",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise NetworkValidationError(
                    f"{name} must be positive"
                )

        if self.allow_hardware_control:
            raise NetworkValidationError(
                "Hardware control cannot be enabled "
                "through network security configuration"
            )

        for network in self.trusted_proxy_networks:
            parse_network(network)


# ============================================================
# NETWORK REQUEST
# ============================================================

@dataclass(frozen=True)
class NetworkRequest:
    client_ip: str
    protocol: NetworkProtocol

    endpoint: str
    method: str = "GET"

    body_size_bytes: int = 0

    headers: dict[str, str] = field(
        default_factory=dict
    )

    client_id: str | None = None

    request_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    timestamp: str = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        parse_ip(self.client_ip)

        object.__setattr__(
            self,
            "protocol",
            NetworkProtocol(self.protocol),
        )

        require_text(
            self.endpoint,
            "endpoint",
        )

        require_text(
            self.method,
            "method",
        )

        if (
            isinstance(self.body_size_bytes, bool)
            or not isinstance(
                self.body_size_bytes,
                int,
            )
            or self.body_size_bytes < 0
        ):
            raise NetworkValidationError(
                "Invalid request body size"
            )

        if not isinstance(self.headers, dict):
            raise NetworkValidationError(
                "Headers must be a dictionary"
            )

        if not all(
            isinstance(key, str)
            and isinstance(value, str)
            for key, value in self.headers.items()
        ):
            raise NetworkValidationError(
                "Headers must contain strings"
            )


# ============================================================
# SECURITY DECISION
# ============================================================

@dataclass(frozen=True)
class NetworkSecurityDecision:
    request_id: str

    action: NetworkAction
    reason: str

    evaluated_at: str = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "action": self.action.value,
            "reason": self.reason,
            "evaluated_at": self.evaluated_at,
        }


# ============================================================
# IP ACCESS POLICY
# ============================================================

class NetworkIPPolicy:

    def __init__(
        self,
        *,
        allowed_networks: tuple[str, ...] = (),
        denied_networks: tuple[str, ...] = (),
        default_allow: bool = False,
    ) -> None:
        self.allowed_networks = tuple(
            parse_network(value)
            for value in allowed_networks
        )

        self.denied_networks = tuple(
            parse_network(value)
            for value in denied_networks
        )

        self.default_allow = default_allow

    def is_allowed(
        self,
        client_ip: str,
    ) -> bool:
        address = parse_ip(client_ip)

        if any(
            address in network
            for network in self.denied_networks
        ):
            return False

        if self.allowed_networks:
            return any(
                address in network
                for network in self.allowed_networks
            )

        return self.default_allow


# ============================================================
# ENDPOINT POLICY
# ============================================================

@dataclass(frozen=True)
class NetworkEndpointPolicy:
    endpoint_prefix: str

    allowed_methods: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "GET",
        })
    )

    allowed_protocols: frozenset[
        NetworkProtocol
    ] = field(
        default_factory=lambda: frozenset({
            NetworkProtocol.HTTPS,
        })
    )

    allowed_networks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.endpoint_prefix.startswith("/"):
            raise NetworkValidationError(
                "Endpoint prefix must begin with '/'"
            )

        for network in self.allowed_networks:
            parse_network(network)

    def matches(
        self,
        endpoint: str,
    ) -> bool:
        prefix = self.endpoint_prefix.rstrip("/")

        if not prefix:
            return endpoint.startswith("/")

        return (
            endpoint == prefix
            or endpoint.startswith(prefix + "/")
        )

    def permits(
        self,
        request: NetworkRequest,
    ) -> bool:
        if request.method.upper() not in self.allowed_methods:
            return False

        if request.protocol not in self.allowed_protocols:
            return False

        if self.allowed_networks:
            address = parse_ip(
                request.client_ip
            )

            if not any(
                address in parse_network(network)
                for network in self.allowed_networks
            ):
                return False

        return True


# ============================================================
# RATE LIMITER
# ============================================================

class NetworkRateLimiter:

    def __init__(
        self,
        requests: int,
        window_seconds: int,
    ) -> None:
        self.requests = requests
        self.window_seconds = window_seconds

        self._timestamps: dict[
            str,
            deque[float]
        ] = defaultdict(deque)

        self._lock = threading.RLock()

    def check(
        self,
        identifier: str,
    ) -> bool:
        identifier = require_text(
            identifier,
            "identifier",
        )

        now = time.monotonic()

        with self._lock:
            history = self._timestamps[
                identifier
            ]

            cutoff = (
                now - self.window_seconds
            )

            while (
                history
                and history[0] <= cutoff
            ):
                history.popleft()

            if len(history) >= self.requests:
                return False

            history.append(now)
            return True


# ============================================================
# SUSPICIOUS REQUEST ANALYZER
# ============================================================

class SuspiciousNetworkRequestAnalyzer:
    """
    Basic request anomaly indicators.

    These heuristics are not a web application
    firewall and may produce false positives.
    """

    SUSPICIOUS_PATTERNS = (
        r"\.\./",
        r"%2e%2e",
        r"<script",
        r"%3cscript",
        r"\x00",
    )

    def inspect(
        self,
        request: NetworkRequest,
    ) -> list[str]:
        findings = []

        endpoint = request.endpoint.lower()

        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(
                pattern,
                endpoint,
                flags=re.IGNORECASE,
            ):
                findings.append(
                    "Suspicious endpoint pattern"
                )
                break

        if len(request.endpoint) > 2048:
            findings.append(
                "Endpoint length exceeds limit"
            )

        if len(request.headers) > 100:
            findings.append(
                "Too many request headers"
            )

        for key, value in request.headers.items():
            if len(key) > 256 or len(value) > 8192:
                findings.append(
                    "Oversized HTTP header"
                )
                break

        return findings


# ============================================================
# SECURITY AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class NetworkSecurityAuditEvent:
    event_id: str
    timestamp: str

    event_type: NetworkSecurityEventType
    severity: NetworkSecuritySeverity

    request_id: str
    client_ip: str
    endpoint: str

    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "request_id": self.request_id,
            "client_ip": self.client_ip,
            "endpoint": self.endpoint,
            "details": self.details,
        }


# ============================================================
# CENTRAL NETWORK SECURITY SERVICE
# ============================================================

class RoboticsNetworkSecurityService:

    def __init__(
        self,
        config: NetworkSecurityConfig | None = None,
        ip_policy: NetworkIPPolicy | None = None,
    ) -> None:
        self.config = (
            config or NetworkSecurityConfig()
        )

        self.ip_policy = (
            ip_policy
            or NetworkIPPolicy(
                allowed_networks=(
                    "127.0.0.0/8",
                    "::1/128",
                ),
                default_allow=False,
            )
        )

        self.rate_limiter = NetworkRateLimiter(
            requests=self.config.rate_limit_requests,
            window_seconds=(
                self.config.rate_limit_window_seconds
            ),
        )

        self.analyzer = (
            SuspiciousNetworkRequestAnalyzer()
        )

        self.endpoint_policies: list[
            NetworkEndpointPolicy
        ] = []

        self.audit_events: list[
            NetworkSecurityAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # REGISTER ENDPOINT POLICY
    # --------------------------------------------------------

    def register_endpoint_policy(
        self,
        policy: NetworkEndpointPolicy,
    ) -> None:
        with self._lock:
            self.endpoint_policies.append(
                policy
            )

            self.endpoint_policies.sort(
                key=lambda item: len(
                    item.endpoint_prefix
                ),
                reverse=True,
            )

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        request: NetworkRequest,
        event_type: NetworkSecurityEventType,
        severity: NetworkSecuritySeverity,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.audit_events.append(
                NetworkSecurityAuditEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=utc_now(),
                    event_type=event_type,
                    severity=severity,
                    request_id=request.request_id,
                    client_ip=request.client_ip,
                    endpoint=request.endpoint,
                    details=json_copy(
                        dict(details or {})
                    ),
                )
            )

            if (
                len(self.audit_events)
                > self.config.maximum_audit_events
            ):
                self.audit_events.pop(0)

    # --------------------------------------------------------
    # DENY HELPER
    # --------------------------------------------------------

    def _deny(
        self,
        request: NetworkRequest,
        reason: str,
        event_type: NetworkSecurityEventType = (
            NetworkSecurityEventType.ACCESS_DENIED
        ),
    ) -> NetworkSecurityDecision:
        self._audit(
            request,
            event_type,
            NetworkSecuritySeverity.WARNING,
            {"reason": reason},
        )

        return NetworkSecurityDecision(
            request_id=request.request_id,
            action=NetworkAction.DENY,
            reason=reason,
        )

    # --------------------------------------------------------
    # EVALUATE NETWORK REQUEST
    # --------------------------------------------------------

    def evaluate(
        self,
        request: NetworkRequest,
    ) -> NetworkSecurityDecision:
        if (
            request.body_size_bytes
            > self.config.maximum_request_body_bytes
        ):
            return self._deny(
                request,
                "Request body exceeds size limit",
                NetworkSecurityEventType.INVALID_REQUEST,
            )

        if (
            request.protocol
            not in self.config.allowed_protocols
        ):
            return self._deny(
                request,
                "Protocol is not permitted",
            )

        insecure_protocols = {
            NetworkProtocol.HTTP,
            NetworkProtocol.WS,
            NetworkProtocol.MQTT,
            NetworkProtocol.GRPC,
        }

        if (
            self.config.require_tls
            and request.protocol in insecure_protocols
        ):
            return self._deny(
                request,
                "TLS transport required",
                NetworkSecurityEventType.TLS_REQUIRED,
            )

        if not self.ip_policy.is_allowed(
            request.client_ip
        ):
            return self._deny(
                request,
                "Client IP is not allowed",
            )

        findings = self.analyzer.inspect(
            request
        )

        if findings:
            self._audit(
                request,
                NetworkSecurityEventType.SUSPICIOUS_REQUEST,
                NetworkSecuritySeverity.HIGH,
                {"findings": findings},
            )

            return self._deny(
                request,
                "Suspicious request rejected",
            )

        with self._lock:
            matching_policy = next(
                (
                    policy
                    for policy in self.endpoint_policies
                    if policy.matches(request.endpoint)
                ),
                None,
            )

        if matching_policy is None:
            return self._deny(
                request,
                "No endpoint policy configured",
            )

        if not matching_policy.permits(
            request
        ):
            return self._deny(
                request,
                "Endpoint policy denied request",
            )

        if not self.rate_limiter.check(
            request.client_ip
        ):
            return self._deny(
                request,
                "Rate limit exceeded",
                NetworkSecurityEventType.RATE_LIMITED,
            )

        self._audit(
            request,
            NetworkSecurityEventType.ACCESS_ALLOWED,
            NetworkSecuritySeverity.INFO,
            {
                "protocol": request.protocol.value,
                "method": request.method.upper(),
            },
        )

        return NetworkSecurityDecision(
            request_id=request.request_id,
            action=NetworkAction.ALLOW,
            reason="Network policy checks passed",
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        with self._lock:
            counts = {
                event_type.value: 0
                for event_type in NetworkSecurityEventType
            }

            for event in self.audit_events:
                counts[
                    event.event_type.value
                ] += 1

            return {
                "registered_endpoint_policies": len(
                    self.endpoint_policies
                ),
                "audit_events": len(
                    self.audit_events
                ),
                "event_distribution": counts,
                "tls_required": self.config.require_tls,
                "hardware_control_enabled": False,
            }

    # --------------------------------------------------------
    # EXPORT REPORT
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
# FASTAPI INTEGRATION ADAPTER
# ============================================================

def create_fastapi_network_middleware(
    security: RoboticsNetworkSecurityService,
):
    """
    Optional FastAPI / Starlette middleware factory.

    Install FastAPI separately.

    Configure the reverse proxy to enforce TLS
    and request-size limits before this middleware.
    """

    try:
        from fastapi import Request
        from starlette.responses import JSONResponse
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI and Starlette are required"
        ) from exc

    async def network_security_middleware(
        request: Request,
        call_next,
    ):
        client_ip = (
            request.client.host
            if request.client is not None
            else "0.0.0.0"
        )

        protocol = (
            NetworkProtocol.HTTPS
            if request.url.scheme == "https"
            else NetworkProtocol.HTTP
        )

        content_length = request.headers.get(
            "content-length"
        )

        try:
            body_size = (
                int(content_length)
                if content_length is not None
                else 0
            )

            if body_size < 0:
                raise ValueError

        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "Invalid Content-Length"
                },
            )

        network_request = NetworkRequest(
            client_ip=client_ip,
            protocol=protocol,
            endpoint=request.url.path,
            method=request.method,
            body_size_bytes=body_size,
            headers=dict(request.headers),
        )

        decision = security.evaluate(
            network_request
        )

        if decision.action == NetworkAction.DENY:
            return JSONResponse(
                status_code=403,
                content={
                    "detail": decision.reason,
                    "request_id": decision.request_id,
                },
            )

        response = await call_next(
            request
        )

        response.headers[
            "X-Content-Type-Options"
        ] = "nosniff"

        response.headers[
            "X-Frame-Options"
        ] = "DENY"

        response.headers[
            "Referrer-Policy"
        ] = "no-referrer"

        return response

    return network_security_middleware


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    security = RoboticsNetworkSecurityService(
        config=NetworkSecurityConfig(
            require_tls=True,
            rate_limit_requests=10,
            rate_limit_window_seconds=60,
        ),
        ip_policy=NetworkIPPolicy(
            allowed_networks=(
                "127.0.0.0/8",
                "192.168.10.0/24",
                "::1/128",
            ),
            denied_networks=(
                "192.168.10.200/32",
            ),
            default_allow=False,
        ),
    )

    security.register_endpoint_policy(
        NetworkEndpointPolicy(
            endpoint_prefix="/api/v1/robots",
            allowed_methods=frozenset({
                "GET",
                "POST",
            }),
            allowed_protocols=frozenset({
                NetworkProtocol.HTTPS,
            }),
            allowed_networks=(
                "127.0.0.0/8",
                "192.168.10.0/24",
            ),
        )
    )

    allowed_request = NetworkRequest(
        client_ip="192.168.10.25",
        protocol=NetworkProtocol.HTTPS,
        endpoint="/api/v1/robots",
        method="GET",
    )

    denied_request = NetworkRequest(
        client_ip="192.168.10.200",
        protocol=NetworkProtocol.HTTPS,
        endpoint="/api/v1/robots",
        method="GET",
    )

    allowed = security.evaluate(
        allowed_request
    )

    denied = security.evaluate(
        denied_request
    )

    report = security.export_json(
        "data/security/network_security_report.json"
    )

    print(
        json.dumps(
            {
                "allowed_request": allowed.to_dict(),
                "denied_request": denied.to_dict(),
                "statistics": security.statistics(),
                "report": str(report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

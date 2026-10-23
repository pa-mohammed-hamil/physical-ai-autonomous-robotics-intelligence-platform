"""Risk Assessment module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/security/risk_assessment.py

Python 3.11+
Standard library only.

Features:
    - Unified multi-domain robotics risk assessment
    - Collision and human-proximity risk
    - Geofence and hazard evaluation
    - Emergency and robot-health assessment
    - Cybersecurity and authorization checks
    - Safety interlock integration
    - Risk scoring and severity classification
    - Mandatory safety gates
    - Missing/stale evidence detection
    - Human review recommendations
    - Tenant-scoped reporting
    - Audit logging and JSON export

IMPORTANT:
    Simulation and supervisory decision support only.

    ALLOW_SIMULATION never authorizes real
    physical robot movement.

    Independent safety-rated controllers,
    protective stops and emergency stops
    are required for physical hardware.
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
from typing import Any, Iterable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class RiskAssessmentError(Exception):
    pass


class RiskValidationError(RiskAssessmentError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RiskDomain(str, Enum):
    COLLISION = "collision"
    HUMAN_PROXIMITY = "human_proximity"
    GEOFENCE = "geofence"
    HAZARD = "hazard"
    EMERGENCY = "emergency"
    ROBOT_HEALTH = "robot_health"
    CYBERSECURITY = "cybersecurity"
    COMMAND_INTERLOCK = "command_interlock"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RiskDecision(str, Enum):
    ALLOW_SIMULATION = "allow_simulation"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    DENY = "deny"


class EvidenceStatus(str, Enum):
    VALID = "valid"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class RiskRecommendation(str, Enum):
    CONTINUE_SIMULATION = "continue_simulation"
    REQUIRE_HUMAN_REVIEW = "require_human_review"
    REQUEST_SAFE_STOP = "request_safe_stop"
    INVESTIGATE_HAZARD = "investigate_hazard"
    INSPECT_ROBOT = "inspect_robot"
    REVIEW_GEOFENCE = "review_geofence"
    CHECK_PERCEPTION = "check_perception"
    CHECK_AUTHORIZATION = "check_authorization"
    CHECK_SECURITY = "check_security"
    REFRESH_TELEMETRY = "refresh_telemetry"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def validate_identifier(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            character.isalnum()
            or character in "_-"
            for character in value
        )
    ):
        raise RiskValidationError(
            f"Invalid {name}"
        )

    return value


def validate_score(
    value: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not (0 <= value <= 100)
    ):
        raise RiskValidationError(
            "Risk score must be between 0 and 100"
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
        raise RiskValidationError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_WEIGHTS: dict[RiskDomain, float] = {
    RiskDomain.COLLISION: 0.20,
    RiskDomain.HUMAN_PROXIMITY: 0.20,
    RiskDomain.GEOFENCE: 0.10,
    RiskDomain.HAZARD: 0.15,
    RiskDomain.EMERGENCY: 0.10,
    RiskDomain.ROBOT_HEALTH: 0.10,
    RiskDomain.CYBERSECURITY: 0.05,
    RiskDomain.COMMAND_INTERLOCK: 0.10,
}


@dataclass(frozen=True)
class RiskAssessmentConfig:
    domain_weights: Mapping[
        RiskDomain, float
    ] = field(
        default_factory=lambda: dict(
            DEFAULT_WEIGHTS
        )
    )

    moderate_threshold: float = 30.0
    high_threshold: float = 60.0
    critical_threshold: float = 85.0

    maximum_evidence_age_seconds: float = 3.0

    maximum_audit_events: int = 20000

    require_all_domains: bool = True
    deny_on_failed_evidence: bool = True
    deny_on_critical_domain: bool = True
    deny_on_mandatory_gate: bool = True

    simulation_only: bool = True

    def __post_init__(self) -> None:
        if not self.simulation_only:
            raise RiskValidationError(
                "Hardware authorization is not supported"
            )

        if set(self.domain_weights) != set(RiskDomain):
            raise RiskValidationError(
                "Weights must cover every risk domain"
            )

        weights = {
            RiskDomain(domain): float(weight)
            for domain, weight
            in self.domain_weights.items()
        }

        if any(
            not math.isfinite(weight)
            or weight < 0
            for weight in weights.values()
        ):
            raise RiskValidationError(
                "Weights must be finite and nonnegative"
            )

        if not math.isclose(
            sum(weights.values()),
            1.0,
            abs_tol=1e-8,
        ):
            raise RiskValidationError(
                "Risk weights must sum to 1"
            )

        object.__setattr__(
            self,
            "domain_weights",
            weights,
        )

        thresholds = (
            self.moderate_threshold,
            self.high_threshold,
            self.critical_threshold,
        )

        for threshold in thresholds:
            validate_score(threshold)

        if not (
            thresholds[0]
            < thresholds[1]
            < thresholds[2]
        ):
            raise RiskValidationError(
                "Risk thresholds must be ordered"
            )

        if (
            not math.isfinite(
                self.maximum_evidence_age_seconds
            )
            or self.maximum_evidence_age_seconds <= 0
        ):
            raise RiskValidationError(
                "Evidence age limit must be positive"
            )

        if (
            isinstance(self.maximum_audit_events, bool)
            or not isinstance(self.maximum_audit_events, int)
            or self.maximum_audit_events <= 0
        ):
            raise RiskValidationError(
                "Maximum audit events must be positive"
            )


# ============================================================
# RISK EVIDENCE
# ============================================================

@dataclass(frozen=True)
class RiskEvidence:
    domain: RiskDomain
    risk_score: float

    status: EvidenceStatus

    source: str
    description: str

    mandatory_gate_passed: bool = True

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "domain",
            RiskDomain(self.domain),
        )

        object.__setattr__(
            self,
            "status",
            EvidenceStatus(self.status),
        )

        validate_score(self.risk_score)
        validate_timestamp(self.timestamp)

        if (
            not isinstance(self.source, str)
            or not self.source.strip()
        ):
            raise RiskValidationError(
                "Evidence source is required"
            )

        if (
            not isinstance(self.description, str)
            or not self.description.strip()
        ):
            raise RiskValidationError(
                "Evidence description is required"
            )

        if not isinstance(
            self.mandatory_gate_passed,
            bool,
        ):
            raise RiskValidationError(
                "mandatory_gate_passed must be boolean"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "risk_score": self.risk_score,
            "status": self.status.value,
            "source": self.source,
            "description": self.description,
            "mandatory_gate_passed": (
                self.mandatory_gate_passed
            ),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================
# ASSESSMENT REQUEST
# ============================================================

@dataclass(frozen=True)
class RiskAssessmentRequest:
    tenant_id: str
    robot_id: str

    evidences: tuple[
        RiskEvidence,
        ...
    ]

    operation: str = "navigation_simulation"

    request_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.robot_id,
            "robot_id",
        )

        if (
            not isinstance(self.operation, str)
            or not self.operation.strip()
            or len(self.operation) > 128
        ):
            raise RiskValidationError(
                "Invalid operation"
            )

        if not self.evidences:
            raise RiskValidationError(
                "Risk evidence is required"
            )


# ============================================================
# DOMAIN RESULT
# ============================================================

@dataclass(frozen=True)
class DomainRiskResult:
    domain: RiskDomain

    score: float
    weighted_score: float

    status: EvidenceStatus

    mandatory_gate_passed: bool
    fresh: bool

    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "score": self.score,
            "weighted_score": self.weighted_score,
            "status": self.status.value,
            "mandatory_gate_passed": (
                self.mandatory_gate_passed
            ),
            "fresh": self.fresh,
            "source": self.source,
        }


# ============================================================
# ASSESSMENT REPORT
# ============================================================

@dataclass(frozen=True)
class RiskAssessmentReport:
    report_id: str
    request_id: str

    tenant_id: str
    robot_id: str

    operation: str

    overall_score: float
    risk_level: RiskLevel

    decision: RiskDecision

    domain_results: tuple[
        DomainRiskResult,
        ...
    ]

    recommendations: tuple[
        RiskRecommendation,
        ...
    ]

    reasons: tuple[str, ...]

    evaluated_at: str = field(
        default_factory=utc_iso
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "operation": self.operation,
            "overall_score": self.overall_score,
            "risk_level": self.risk_level.value,
            "decision": self.decision.value,
            "domain_results": [
                item.to_dict()
                for item in self.domain_results
            ],
            "recommendations": [
                item.value
                for item in self.recommendations
            ],
            "reasons": list(self.reasons),
            "evaluated_at": self.evaluated_at,
        }


# ============================================================
# RISK ENGINE
# ============================================================

class RoboticsRiskAssessmentEngine:

    def __init__(
        self,
        config: RiskAssessmentConfig,
    ) -> None:
        self.config = config

    def classify(
        self,
        score: float,
    ) -> RiskLevel:
        if score >= self.config.critical_threshold:
            return RiskLevel.CRITICAL

        if score >= self.config.high_threshold:
            return RiskLevel.HIGH

        if score >= self.config.moderate_threshold:
            return RiskLevel.MODERATE

        return RiskLevel.LOW

    def assess(
        self,
        request: RiskAssessmentRequest,
    ) -> RiskAssessmentReport:
        now = utc_now()

        evidence_map: dict[
            RiskDomain,
            RiskEvidence
        ] = {}

        for evidence in request.evidences:
            if evidence.domain in evidence_map:
                raise RiskValidationError(
                    f"Duplicate evidence: {evidence.domain.value}"
                )

            evidence_map[
                evidence.domain
            ] = evidence

        missing_domains = (
            set(RiskDomain)
            - set(evidence_map)
        )

        domain_results: list[
            DomainRiskResult
        ] = []

        reasons: list[str] = []
        recommendations: set[
            RiskRecommendation
        ] = set()

        mandatory_failure = False
        evidence_failure = False
        critical_domain = False

        weighted_total = 0.0

        for domain in RiskDomain:
            evidence = evidence_map.get(
                domain
            )

            if evidence is None:
                if self.config.require_all_domains:
                    evidence_failure = True

                    reasons.append(
                        f"Missing evidence: {domain.value}"
                    )

                    recommendations.add(
                        RiskRecommendation.REFRESH_TELEMETRY
                    )

                domain_results.append(
                    DomainRiskResult(
                        domain=domain,
                        score=100.0,
                        weighted_score=(
                            100.0
                            * self.config.domain_weights[domain]
                        ),
                        status=EvidenceStatus.UNKNOWN,
                        mandatory_gate_passed=False,
                        fresh=False,
                        source="missing",
                    )
                )

                weighted_total += (
                    100.0
                    * self.config.domain_weights[domain]
                )

                continue

            age = (
                now
                - validate_timestamp(
                    evidence.timestamp
                )
            ).total_seconds()

            fresh = (
                -1.0 <= age
                <= self.config.maximum_evidence_age_seconds
            )

            if (
                not fresh
                or evidence.status in {
                    EvidenceStatus.FAILED,
                    EvidenceStatus.UNKNOWN,
                }
            ):
                evidence_failure = True

                reasons.append(
                    f"Unreliable evidence: {domain.value}"
                )

            if not evidence.mandatory_gate_passed:
                mandatory_failure = True

                reasons.append(
                    f"Mandatory gate failed: {domain.value}"
                )

            if (
                evidence.risk_score
                >= self.config.critical_threshold
            ):
                critical_domain = True

                reasons.append(
                    f"Critical risk domain: {domain.value}"
                )

            effective_score = (
                evidence.risk_score
                if (
                    fresh
                    and evidence.status
                    not in {
                        EvidenceStatus.FAILED,
                        EvidenceStatus.UNKNOWN,
                    }
                )
                else 100.0
            )

            weighted = (
                effective_score
                * self.config.domain_weights[domain]
            )

            weighted_total += weighted

            domain_results.append(
                DomainRiskResult(
                    domain=domain,
                    score=effective_score,
                    weighted_score=weighted,
                    status=evidence.status,
                    mandatory_gate_passed=(
                        evidence.mandatory_gate_passed
                    ),
                    fresh=fresh,
                    source=evidence.source,
                )
            )

        overall_score = round(
            min(100.0, weighted_total),
            2,
        )

        level = self.classify(
            overall_score
        )

        should_deny = (
            (
                evidence_failure
                and self.config.deny_on_failed_evidence
            )
            or (
                mandatory_failure
                and self.config.deny_on_mandatory_gate
            )
            or (
                critical_domain
                and self.config.deny_on_critical_domain
            )
            or level == RiskLevel.CRITICAL
        )

        if should_deny:
            decision = RiskDecision.DENY

            recommendations.add(
                RiskRecommendation.REQUEST_SAFE_STOP
            )

            recommendations.add(
                RiskRecommendation.REQUIRE_HUMAN_REVIEW
            )

        elif level in {
            RiskLevel.HIGH,
            RiskLevel.MODERATE,
        }:
            decision = (
                RiskDecision.HUMAN_REVIEW_REQUIRED
            )

            recommendations.add(
                RiskRecommendation.REQUIRE_HUMAN_REVIEW
            )

        else:
            decision = (
                RiskDecision.ALLOW_SIMULATION
            )

            recommendations.add(
                RiskRecommendation.CONTINUE_SIMULATION
            )

        domain_recommendations = {
            RiskDomain.COLLISION:
                RiskRecommendation.REQUEST_SAFE_STOP,

            RiskDomain.HUMAN_PROXIMITY:
                RiskRecommendation.CHECK_PERCEPTION,

            RiskDomain.GEOFENCE:
                RiskRecommendation.REVIEW_GEOFENCE,

            RiskDomain.HAZARD:
                RiskRecommendation.INVESTIGATE_HAZARD,

            RiskDomain.EMERGENCY:
                RiskRecommendation.REQUIRE_HUMAN_REVIEW,

            RiskDomain.ROBOT_HEALTH:
                RiskRecommendation.INSPECT_ROBOT,

            RiskDomain.CYBERSECURITY:
                RiskRecommendation.CHECK_SECURITY,

            RiskDomain.COMMAND_INTERLOCK:
                RiskRecommendation.CHECK_AUTHORIZATION,
        }

        for result in domain_results:
            if (
                result.score
                >= self.config.high_threshold
                or not result.mandatory_gate_passed
            ):
                recommendations.add(
                    domain_recommendations[
                        result.domain
                    ]
                )

        if missing_domains:
            reasons.append(
                "One or more risk domains are missing"
            )

        if not reasons:
            reasons.append(
                "Risk assessment completed"
            )

        return RiskAssessmentReport(
            report_id=str(uuid.uuid4()),
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            robot_id=request.robot_id,
            operation=request.operation,
            overall_score=overall_score,
            risk_level=level,
            decision=decision,
            domain_results=tuple(domain_results),
            recommendations=tuple(
                sorted(
                    recommendations,
                    key=lambda item: item.value,
                )
            ),
            reasons=tuple(reasons),
        )


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class RiskAuditEvent:
    event_id: str
    timestamp: str

    tenant_id: str
    robot_id: str

    report_id: str

    decision: RiskDecision
    risk_level: RiskLevel
    risk_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "report_id": self.report_id,
            "decision": self.decision.value,
            "risk_level": self.risk_level.value,
            "risk_score": self.risk_score,
        }


# ============================================================
# CENTRAL RISK ASSESSMENT SERVICE
# ============================================================

class RoboticsRiskAssessmentService:

    def __init__(
        self,
        config: RiskAssessmentConfig | None = None,
    ) -> None:
        self.config = (
            config or RiskAssessmentConfig()
        )

        self.engine = RoboticsRiskAssessmentEngine(
            self.config
        )

        self.reports: dict[
            str,
            RiskAssessmentReport
        ] = {}

        self.audit_events: list[
            RiskAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # ASSESS
    # --------------------------------------------------------

    def assess(
        self,
        request: RiskAssessmentRequest,
    ) -> RiskAssessmentReport:
        report = self.engine.assess(
            request
        )

        with self._lock:
            self.reports[
                report.report_id
            ] = report

            self.audit_events.append(
                RiskAuditEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=utc_iso(),
                    tenant_id=report.tenant_id,
                    robot_id=report.robot_id,
                    report_id=report.report_id,
                    decision=report.decision,
                    risk_level=report.risk_level,
                    risk_score=report.overall_score,
                )
            )

            if (
                len(self.audit_events)
                > self.config.maximum_audit_events
            ):
                self.audit_events.pop(0)

        return report

    # --------------------------------------------------------
    # GET REPORT
    # --------------------------------------------------------

    def get_report(
        self,
        *,
        tenant_id: str,
        report_id: str,
    ) -> RiskAssessmentReport:
        with self._lock:
            report = self.reports.get(
                report_id
            )

            if (
                report is None
                or report.tenant_id != tenant_id
            ):
                raise RiskAssessmentError(
                    "Risk report not found"
                )

            return report

    # --------------------------------------------------------
    # LIST REPORTS
    # --------------------------------------------------------

    def list_reports(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
    ) -> list[RiskAssessmentReport]:
        with self._lock:
            return [
                report
                for report in self.reports.values()
                if (
                    report.tenant_id == tenant_id
                    and (
                        robot_id is None
                        or report.robot_id == robot_id
                    )
                )
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
            reports = [
                report
                for report in self.reports.values()
                if (
                    tenant_id is None
                    or report.tenant_id == tenant_id
                )
            ]

            return {
                "total_assessments": len(reports),
                "denied": sum(
                    report.decision == RiskDecision.DENY
                    for report in reports
                ),
                "human_review_required": sum(
                    report.decision
                    == RiskDecision.HUMAN_REVIEW_REQUIRED
                    for report in reports
                ),
                "critical": sum(
                    report.risk_level == RiskLevel.CRITICAL
                    for report in reports
                ),
                "average_risk_score": (
                    round(
                        sum(
                            report.overall_score
                            for report in reports
                        ) / len(reports),
                        2,
                    )
                    if reports
                    else 0.0
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
                "generated_at": utc_iso(),
                "tenant_id": tenant_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "reports": [
                    report.to_dict()
                    for report in self.reports.values()
                    if report.tenant_id == tenant_id
                ],
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
    service = RoboticsRiskAssessmentService()

    scores = {
        RiskDomain.COLLISION: 15.0,
        RiskDomain.HUMAN_PROXIMITY: 20.0,
        RiskDomain.GEOFENCE: 0.0,
        RiskDomain.HAZARD: 10.0,
        RiskDomain.EMERGENCY: 0.0,
        RiskDomain.ROBOT_HEALTH: 12.0,
        RiskDomain.CYBERSECURITY: 5.0,
        RiskDomain.COMMAND_INTERLOCK: 0.0,
    }

    evidences = tuple(
        RiskEvidence(
            domain=domain,
            risk_score=score,
            status=EvidenceStatus.VALID,
            source=f"{domain.value}_service",
            description=(
                f"Simulation evidence for {domain.value}"
            ),
            mandatory_gate_passed=True,
        )
        for domain, score in scores.items()
    )

    request = RiskAssessmentRequest(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        operation="warehouse_navigation_simulation",
        evidences=evidences,
    )

    report = service.assess(
        request
    )

    print(
        json.dumps(
            report.to_dict(),
            indent=2,
        )
    )

    service.export_json(
        "data/security/risk_assessment_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

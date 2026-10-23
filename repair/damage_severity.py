"""Damage Severity module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/damage_severity.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robotics Damage Severity Assessment Engine

Python 3.11+
Standard library only.

Features:
    - Structured damage severity assessment
    - Integration with damage_segmentation.py outputs
    - Crack, corrosion, wear, deformation and other damage types
    - Component-specific damage risk
    - Safety-critical component prioritization
    - Damage area and intensity scoring
    - Confidence and uncertainty assessment
    - Multi-region damage aggregation
    - Repair/replacement recommendations
    - Human inspection requirements
    - Multi-tenant robot isolation
    - Thread-safe assessment history
    - JSON report export
    - Runnable demonstration

IMPORTANT:
    This is a heuristic decision-support module.
    Segmentation candidates are not confirmed damage.
    Severity scores are not calibrated failure probabilities.
    No physical robot commands are issued.
"""

from __future__ import annotations

import json
import math
import re
import threading
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class DamageSeverityError(Exception):
    pass


class DamageSeverityValidationError(DamageSeverityError):
    pass


class AssessmentNotFound(DamageSeverityError):
    pass


class AssessmentConflict(DamageSeverityError):
    pass


# ============================================================
# ENUMS
# ============================================================

class DamageType(str, Enum):
    CRACK = "crack"
    CORROSION = "corrosion"
    SCRATCH = "scratch"
    DENT = "dent"
    DEFORMATION = "deformation"
    FRACTURE = "fracture"
    WEAR = "wear"
    OVERHEATING = "overheating"
    ELECTRICAL = "electrical"
    MISALIGNMENT = "misalignment"
    LEAK = "leak"
    MISSING_PART = "missing_part"
    UNKNOWN = "unknown"


class DamageSeverityLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentCriticality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SAFETY_CRITICAL = "safety_critical"


class DamageRecommendation(str, Enum):
    MONITOR = "monitor"
    INSPECT = "inspect"
    PLAN_REPAIR = "plan_repair"
    PRIORITY_REPAIR = "priority_repair"
    CONSIDER_REPLACEMENT = "consider_replacement"
    STOP_AND_REVIEW = "stop_and_review"


class AssessmentStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}",
            value,
        )
    ):
        raise DamageSeverityValidationError(
            f"Invalid {name}"
        )

    return value


def finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise DamageSeverityValidationError(
            f"{name} must be a finite number"
        )

    return float(value)


def unit_interval(value: Any, name: str) -> float:
    result = finite(value, name)

    if not 0.0 <= result <= 1.0:
        raise DamageSeverityValidationError(
            f"{name} must be between 0 and 1"
        )

    return result


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, value))


def validate_text(
    value: str,
    name: str,
    maximum_length: int = 1024,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) > maximum_length
    ):
        raise DamageSeverityValidationError(
            f"Invalid {name}"
        )

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DamageSeverityConfig:
    maximum_assessments: int = 100000
    maximum_regions: int = 1000
    maximum_audit_events: int = 20000

    minimum_confidence: float = 0.80

    low_threshold: float = 0.20
    moderate_threshold: float = 0.45
    high_threshold: float = 0.70

    area_weight: float = 0.30
    intensity_weight: float = 0.45
    damage_type_weight: float = 0.25

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_assessments",
            "maximum_regions",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise DamageSeverityValidationError(
                    f"{name} must be positive"
                )

        unit_interval(
            self.minimum_confidence,
            "minimum_confidence",
        )

        if not (
            0 < self.low_threshold
            < self.moderate_threshold
            < self.high_threshold
            < 1
        ):
            raise DamageSeverityValidationError(
                "Invalid severity thresholds"
            )

        weights = (
            self.area_weight,
            self.intensity_weight,
            self.damage_type_weight,
        )

        if any(
            not 0 <= finite(w, "weight") <= 1
            for w in weights
        ):
            raise DamageSeverityValidationError(
                "Weights must be between 0 and 1"
            )

        if not math.isclose(
            sum(weights),
            1.0,
            abs_tol=1e-9,
        ):
            raise DamageSeverityValidationError(
                "Severity weights must sum to 1"
            )

        if not self.simulation_only:
            raise DamageSeverityValidationError(
                "Only simulation/decision support is supported"
            )


# ============================================================
# DAMAGE TYPE WEIGHTS
# ============================================================

DAMAGE_TYPE_RISK: dict[DamageType, float] = {
    DamageType.CRACK: 0.85,
    DamageType.CORROSION: 0.55,
    DamageType.SCRATCH: 0.15,
    DamageType.DENT: 0.40,
    DamageType.DEFORMATION: 0.80,
    DamageType.FRACTURE: 1.00,
    DamageType.WEAR: 0.55,
    DamageType.OVERHEATING: 0.90,
    DamageType.ELECTRICAL: 0.95,
    DamageType.MISALIGNMENT: 0.70,
    DamageType.LEAK: 0.80,
    DamageType.MISSING_PART: 0.90,
    DamageType.UNKNOWN: 0.50,
}

CRITICALITY_MULTIPLIER: dict[
    ComponentCriticality, float
] = {
    ComponentCriticality.LOW: 0.80,
    ComponentCriticality.MEDIUM: 1.00,
    ComponentCriticality.HIGH: 1.15,
    ComponentCriticality.SAFETY_CRITICAL: 1.30,
}


# ============================================================
# DAMAGE REGION
# ============================================================

@dataclass(frozen=True)
class DamageRegionObservation:
    region_id: str
    damage_type: DamageType

    area_fraction: float
    intensity: float
    confidence: float

    description: str = ""

    def __post_init__(self) -> None:
        validate_id(
            self.region_id,
            "region_id",
        )

        if not isinstance(
            self.damage_type,
            DamageType,
        ):
            raise DamageSeverityValidationError(
                "Invalid damage type"
            )

        for name in (
            "area_fraction",
            "intensity",
            "confidence",
        ):
            unit_interval(
                getattr(self, name),
                name,
            )

        validate_text(
            self.description,
            "description",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "damage_type": self.damage_type.value,
            "area_fraction": self.area_fraction,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "description": self.description,
        }


# ============================================================
# COMPONENT
# ============================================================

@dataclass(frozen=True)
class DamageComponent:
    tenant_id: str
    robot_id: str
    component_id: str

    component_name: str
    criticality: ComponentCriticality

    repairable: bool = True

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "component_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if (
            not isinstance(self.component_name, str)
            or not self.component_name.strip()
            or len(self.component_name) > 256
        ):
            raise DamageSeverityValidationError(
                "Invalid component name"
            )

        if not isinstance(
            self.criticality,
            ComponentCriticality,
        ):
            raise DamageSeverityValidationError(
                "Invalid component criticality"
            )

        if not isinstance(self.repairable, bool):
            raise DamageSeverityValidationError(
                "repairable must be boolean"
            )


# ============================================================
# ASSESSMENT REQUEST
# ============================================================

@dataclass(frozen=True)
class DamageSeverityRequest:
    component: DamageComponent

    regions: tuple[
        DamageRegionObservation, ...
    ]

    inspection_reference: str = ""

    def __post_init__(self) -> None:
        if not isinstance(
            self.component,
            DamageComponent,
        ):
            raise DamageSeverityValidationError(
                "Invalid component"
            )

        if not isinstance(self.regions, tuple):
            raise DamageSeverityValidationError(
                "Regions must be a tuple"
            )

        seen = set()

        for region in self.regions:
            if not isinstance(
                region,
                DamageRegionObservation,
            ):
                raise DamageSeverityValidationError(
                    "Invalid damage region"
                )

            if region.region_id in seen:
                raise DamageSeverityValidationError(
                    "Duplicate region ID"
                )

            seen.add(region.region_id)

        validate_text(
            self.inspection_reference,
            "inspection_reference",
        )


# ============================================================
# REGION RESULT
# ============================================================

@dataclass(frozen=True)
class DamageRegionSeverity:
    region_id: str
    damage_type: DamageType

    severity_score: float
    severity: DamageSeverityLevel

    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "damage_type": self.damage_type.value,
            "severity_score": round(
                self.severity_score,
                4,
            ),
            "severity": self.severity.value,
            "confidence": self.confidence,
        }


# ============================================================
# ASSESSMENT RECORD
# ============================================================

@dataclass
class DamageSeverityAssessment:
    assessment_id: str

    tenant_id: str
    robot_id: str
    component_id: str
    component_name: str

    criticality: ComponentCriticality

    severity: DamageSeverityLevel
    severity_score: float

    confidence: float
    uncertainty: float

    recommendation: DamageRecommendation

    human_review_required: bool

    region_results: list[
        DamageRegionSeverity
    ]

    inspection_reference: str = ""

    status: AssessmentStatus = (
        AssessmentStatus.PENDING_REVIEW
    )

    review_notes: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    reviewed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "component_id": self.component_id,
            "component_name": self.component_name,
            "criticality": self.criticality.value,
            "severity": self.severity.value,
            "severity_score": round(
                self.severity_score, 4
            ),
            "confidence": round(
                self.confidence, 4
            ),
            "uncertainty": round(
                self.uncertainty, 4
            ),
            "recommendation": (
                self.recommendation.value
            ),
            "human_review_required": (
                self.human_review_required
            ),
            "region_count": len(
                self.region_results
            ),
            "regions": [
                item.to_dict()
                for item in self.region_results
            ],
            "inspection_reference": (
                self.inspection_reference
            ),
            "status": self.status.value,
            "review_notes": self.review_notes,
            "created_at": (
                self.created_at.isoformat()
            ),
            "reviewed_at": (
                self.reviewed_at.isoformat()
                if self.reviewed_at
                else None
            ),
            "simulation_only": True,
        }


# ============================================================
# DAMAGE SEVERITY ENGINE
# ============================================================

class RoboticsDamageSeverityEngine:

    def __init__(
        self,
        config: DamageSeverityConfig | None = None,
    ) -> None:
        self.config = (
            config or DamageSeverityConfig()
        )

        self._assessments: dict[
            str,
            DamageSeverityAssessment,
        ] = {}

        self._audit_events: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # SEVERITY CLASSIFICATION
    # --------------------------------------------------------

    def classify_severity(
        self,
        score: float,
    ) -> DamageSeverityLevel:
        score = unit_interval(
            score,
            "score",
        )

        if score == 0:
            return DamageSeverityLevel.NONE

        if score < self.config.low_threshold:
            return DamageSeverityLevel.LOW

        if score < self.config.moderate_threshold:
            return DamageSeverityLevel.MODERATE

        if score < self.config.high_threshold:
            return DamageSeverityLevel.HIGH

        return DamageSeverityLevel.CRITICAL

    # --------------------------------------------------------
    # REGION SCORING
    # --------------------------------------------------------

    def score_region(
        self,
        region: DamageRegionObservation,
        criticality: ComponentCriticality,
    ) -> DamageRegionSeverity:
        type_risk = DAMAGE_TYPE_RISK[
            region.damage_type
        ]

        base_score = (
            self.config.area_weight
            * region.area_fraction
            + self.config.intensity_weight
            * region.intensity
            + self.config.damage_type_weight
            * type_risk
        )

        score = clamp(
            base_score
            * CRITICALITY_MULTIPLIER[
                criticality
            ]
        )

        severity = self.classify_severity(
            score
        )

        return DamageRegionSeverity(
            region_id=region.region_id,
            damage_type=region.damage_type,
            severity_score=score,
            severity=severity,
            confidence=region.confidence,
        )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    def _recommend(
        self,
        *,
        severity: DamageSeverityLevel,
        confidence: float,
        component: DamageComponent,
    ) -> DamageRecommendation:

        if (
            component.criticality
            == ComponentCriticality.SAFETY_CRITICAL
            and severity in (
                DamageSeverityLevel.HIGH,
                DamageSeverityLevel.CRITICAL,
            )
        ):
            return DamageRecommendation.STOP_AND_REVIEW

        if confidence < self.config.minimum_confidence:
            return DamageRecommendation.INSPECT

        if severity == DamageSeverityLevel.NONE:
            return DamageRecommendation.MONITOR

        if severity == DamageSeverityLevel.UNKNOWN:
            return DamageRecommendation.INSPECT

        if severity == DamageSeverityLevel.LOW:
            return DamageRecommendation.MONITOR

        if severity == DamageSeverityLevel.MODERATE:
            return DamageRecommendation.PLAN_REPAIR

        if not component.repairable:
            return DamageRecommendation.CONSIDER_REPLACEMENT

        if severity == DamageSeverityLevel.HIGH:
            return DamageRecommendation.PRIORITY_REPAIR

        return DamageRecommendation.STOP_AND_REVIEW

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    def analyze(
        self,
        request: DamageSeverityRequest,
    ) -> dict[str, Any]:
        if not isinstance(
            request,
            DamageSeverityRequest,
        ):
            raise DamageSeverityValidationError(
                "Invalid assessment request"
            )

        if (
            len(request.regions)
            > self.config.maximum_regions
        ):
            raise DamageSeverityValidationError(
                "Too many damage regions"
            )

        component = request.component

        region_results = [
            self.score_region(
                region,
                component.criticality,
            )
            for region in request.regions
        ]

        if region_results:
            # Conservative worst-region aggregation.
            severity_score = max(
                item.severity_score
                for item in region_results
            )

            confidence = min(
                item.confidence
                for item in region_results
            )

            severity = self.classify_severity(
                severity_score
            )
        else:
            severity_score = 0.0
            confidence = 0.0
            severity = DamageSeverityLevel.UNKNOWN

        uncertainty = 1.0 - confidence

        recommendation = self._recommend(
            severity=severity,
            confidence=confidence,
            component=component,
        )

        human_review_required = (
            severity in (
                DamageSeverityLevel.HIGH,
                DamageSeverityLevel.CRITICAL,
                DamageSeverityLevel.UNKNOWN,
            )
            or confidence
            < self.config.minimum_confidence
            or component.criticality
            == ComponentCriticality.SAFETY_CRITICAL
        )

        assessment = DamageSeverityAssessment(
            assessment_id=str(uuid.uuid4()),
            tenant_id=component.tenant_id,
            robot_id=component.robot_id,
            component_id=component.component_id,
            component_name=component.component_name,
            criticality=component.criticality,
            severity=severity,
            severity_score=severity_score,
            confidence=confidence,
            uncertainty=uncertainty,
            recommendation=recommendation,
            human_review_required=(
                human_review_required
            ),
            region_results=region_results,
            inspection_reference=(
                request.inspection_reference
            ),
        )

        with self._lock:
            if (
                len(self._assessments)
                >= self.config.maximum_assessments
            ):
                raise DamageSeverityError(
                    "Assessment capacity reached"
                )

            self._assessments[
                assessment.assessment_id
            ] = assessment

            self._audit_events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "timestamp": utc_now().isoformat(),
                    "tenant_id": component.tenant_id,
                    "robot_id": component.robot_id,
                    "assessment_id": (
                        assessment.assessment_id
                    ),
                    "event": "assessment_created",
                }
            )

            return assessment.to_dict()

    # --------------------------------------------------------
    # GET ASSESSMENT
    # --------------------------------------------------------

    def get_assessment(
        self,
        *,
        tenant_id: str,
        assessment_id: str,
    ) -> dict[str, Any]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            assessment_id,
            "assessment_id",
        )

        with self._lock:
            assessment = self._assessments.get(
                assessment_id
            )

            if (
                assessment is None
                or assessment.tenant_id != tenant_id
            ):
                raise AssessmentNotFound(
                    "Assessment not found"
                )

            return assessment.to_dict()

    # --------------------------------------------------------
    # REVIEW
    # --------------------------------------------------------

    def review_assessment(
        self,
        *,
        tenant_id: str,
        assessment_id: str,
        approved: bool,
        notes: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise AssessmentConflict(
                "Authorization required"
            )

        if not isinstance(approved, bool):
            raise DamageSeverityValidationError(
                "approved must be boolean"
            )

        validate_text(
            notes,
            "notes",
            2048,
        )

        with self._lock:
            assessment = self._assessments.get(
                assessment_id
            )

            if (
                assessment is None
                or assessment.tenant_id != tenant_id
            ):
                raise AssessmentNotFound(
                    "Assessment not found"
                )

            if (
                assessment.status
                != AssessmentStatus.PENDING_REVIEW
            ):
                raise AssessmentConflict(
                    "Assessment already reviewed"
                )

            assessment.status = (
                AssessmentStatus.REVIEWED
                if approved
                else AssessmentStatus.REJECTED
            )

            assessment.review_notes = notes
            assessment.reviewed_at = utc_now()

            self._audit_events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "timestamp": utc_now().isoformat(),
                    "tenant_id": tenant_id,
                    "robot_id": assessment.robot_id,
                    "assessment_id": assessment_id,
                    "event": (
                        "assessment_reviewed"
                        if approved
                        else "assessment_rejected"
                    ),
                }
            )

            return assessment.to_dict()

    # --------------------------------------------------------
    # LIST ASSESSMENTS
    # --------------------------------------------------------

    def list_assessments(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        if robot_id is not None:
            validate_id(
                robot_id,
                "robot_id",
            )

        with self._lock:
            return [
                assessment.to_dict()
                for assessment in self._assessments.values()
                if (
                    assessment.tenant_id == tenant_id
                    and (
                        robot_id is None
                        or assessment.robot_id == robot_id
                    )
                )
            ]

    # --------------------------------------------------------
    # ROBOT SUMMARY
    # --------------------------------------------------------

    def robot_damage_summary(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        assessments = self.list_assessments(
            tenant_id=tenant_id,
            robot_id=robot_id,
        )

        counts = Counter(
            item["severity"]
            for item in assessments
        )

        priority_order = (
            DamageSeverityLevel.CRITICAL,
            DamageSeverityLevel.HIGH,
            DamageSeverityLevel.MODERATE,
            DamageSeverityLevel.LOW,
            DamageSeverityLevel.NONE,
        )

        overall = DamageSeverityLevel.UNKNOWN

        for level in priority_order:
            if counts[level.value] > 0:
                overall = level
                break

        return {
            "tenant_id": tenant_id,
            "robot_id": robot_id,
            "total_assessments": len(assessments),
            "overall_severity": overall.value,
            "severity_counts": {
                level.value: counts[level.value]
                for level in DamageSeverityLevel
            },
            "pending_reviews": sum(
                item["status"]
                == AssessmentStatus.PENDING_REVIEW.value
                for item in assessments
            ),
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # EXPORT
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
                "assessments": self.list_assessments(
                    tenant_id=tenant_id
                ),
                "audit_events": [
                    event
                    for event in self._audit_events
                    if event["tenant_id"] == tenant_id
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
# SEGMENTATION INTEGRATION
# ============================================================

def observation_from_segmentation_region(
    region: Mapping[str, Any],
    *,
    intensity: float,
    confidence: float,
    damage_type: DamageType = DamageType.UNKNOWN,
) -> DamageRegionObservation:
    """
    Convert a region dictionary produced by
    damage_segmentation.py into a severity input.

    Segmentation does not provide calibrated
    intensity or confidence, so callers must
    supply those values from another source.

    The area percentage is relative to the
    full image, not necessarily the component.
    """

    if not isinstance(region, Mapping):
        raise DamageSeverityValidationError(
            "Invalid segmentation region"
        )

    region_id = validate_id(
        region["region_id"],
        "region_id",
    )

    area_percentage = finite(
        region["area_percentage"],
        "area_percentage",
    )

    if not 0 <= area_percentage <= 100:
        raise DamageSeverityValidationError(
            "Invalid area percentage"
        )

    return DamageRegionObservation(
        region_id=region_id,
        damage_type=damage_type,
        area_fraction=area_percentage / 100.0,
        intensity=unit_interval(
            intensity,
            "intensity",
        ),
        confidence=unit_interval(
            confidence,
            "confidence",
        ),
        description=(
            "Candidate region imported from "
            "damage segmentation"
        ),
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    engine = RoboticsDamageSeverityEngine()

    component = DamageComponent(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        component_id="arm_joint_001",
        component_name="Robot Arm Joint",
        criticality=(
            ComponentCriticality.SAFETY_CRITICAL
        ),
        repairable=True,
    )

    request = DamageSeverityRequest(
        component=component,
        inspection_reference="inspection_001",
        regions=(
            DamageRegionObservation(
                region_id="region_001",
                damage_type=DamageType.CRACK,
                area_fraction=0.35,
                intensity=0.90,
                confidence=0.94,
                description=(
                    "Simulated crack observation"
                ),
            ),
            DamageRegionObservation(
                region_id="region_002",
                damage_type=DamageType.CORROSION,
                area_fraction=0.15,
                intensity=0.60,
                confidence=0.91,
                description=(
                    "Simulated corrosion observation"
                ),
            ),
        ),
    )

    result = engine.analyze(
        request
    )

    print(
        "Damage severity assessment:"
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    summary = engine.robot_damage_summary(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
    )

    print(
        "\nRobot damage summary:"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    report = engine.export_json(
        "data/robotics/damage_severity_report.json",
        tenant_id="tenant_alpha",
    )

    print(
        "\nReport:",
        report,
    )


if __name__ == "__main__":
    main()
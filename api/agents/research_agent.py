"""Research Agent module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/research_agent.py

Physical AI & Autonomous Robotics Intelligence Platform
Autonomous Research & Evidence Intelligence Agent

Python 3.11+

Capabilities:
- Technical and scientific research
- Robotics documentation retrieval
- Advanced RAG and hybrid search integration
- Academic paper and patent research
- Evidence extraction and verification
- Citation and source provenance tracking
- Research question decomposition
- Multi-step research planning
- Cross-source comparison
- Contradiction detection
- Research gap identification
- Evidence-grounded report generation
- Research quality evaluation
- Tenant-aware authorization and audit logging

Safety:
- Research is read-only and advisory.
- Retrieved content is untrusted.
- External documents cannot issue agent instructions.
- No physical robot commands or safety bypasses.
- Safety-critical recommendations require expert review.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import math
import re
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


MAX_RESEARCH_QUESTIONS = 30
MAX_SOURCES = 150
MAX_EVIDENCE_ITEMS = 500
MAX_CLAIMS = 200
MAX_QUERY_LENGTH = 2000
MAX_SOURCE_CONTENT_CHARS = 100_000
MAX_BATCH_REQUESTS = 10
DEFAULT_MAX_SOURCES = 20
DEFAULT_SEARCH_CONCURRENCY = 4

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class ResearchAgentError(Exception):
    """Base research exception."""


class ResearchValidationError(ResearchAgentError):
    """Invalid research input or provider output."""


class ResearchPermissionError(ResearchAgentError):
    """Research authorization failure."""


class ResearchNotFoundError(ResearchAgentError):
    """Requested research resource does not exist."""


class ResearchProviderError(ResearchAgentError):
    """External research provider failed."""


class ResearchEvidenceError(ResearchAgentError):
    """Evidence could not be verified."""


class ResearchServiceUnavailableError(ResearchAgentError):
    """Required research service is unavailable."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def validate_identifier(value: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        raise ResearchValidationError(f"Invalid identifier: {value!r}")
    return value


def validate_probability(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ResearchValidationError("Value must be between 0 and 1")
    return float(value)


def validate_timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ResearchValidationError("Timezone-aware timestamp required")
    return value


def validate_text(value: str, name: str, maximum: int = MAX_QUERY_LENGTH) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchValidationError(f"{name} is required")
    if len(value) > maximum:
        raise ResearchValidationError(f"{name} exceeds length limit")
    return value.strip()


def content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ResearchDomain(str, Enum):
    ROBOTICS = "robotics"
    PHYSICAL_AI = "physical_ai"
    AUTONOMOUS_SYSTEMS = "autonomous_systems"
    COMPUTER_VISION = "computer_vision"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    GENERATIVE_AI = "generative_ai"
    WORLD_MODELS = "world_models"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    SLAM = "slam"
    SENSOR_FUSION = "sensor_fusion"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    ROBOT_SAFETY = "robot_safety"
    HARDWARE = "hardware"
    SCIENTIFIC = "scientific"
    GENERAL = "general"


class ResearchMode(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    LITERATURE_REVIEW = "literature_review"
    TECHNICAL_INVESTIGATION = "technical_investigation"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    ROOT_CAUSE_RESEARCH = "root_cause_research"


class SourceType(str, Enum):
    ACADEMIC_PAPER = "academic_paper"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    MANUFACTURER_MANUAL = "manufacturer_manual"
    STANDARD = "standard"
    PATENT = "patent"
    RESEARCH_DATASET = "research_dataset"
    OFFICIAL_WEBSITE = "official_website"
    REPOSITORY = "repository"
    BOOK = "book"
    ARTICLE = "article"
    INTERNAL_DOCUMENT = "internal_document"
    OTHER = "other"


class EvidenceStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    UNVERIFIED = "unverified"


class ClaimVerificationStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ResearchStatus(str, Enum):
    CREATED = "created"
    RESEARCHING = "researching"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"


class ResearchRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ResearchScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


@dataclass(frozen=True, slots=True)
class ResearchQuestion:
    question_id: str
    question: str
    domain: ResearchDomain
    priority: int = 1
    parent_question_id: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.question_id)
        validate_text(self.question, "Question")
        if not 1 <= self.priority <= 5:
            raise ResearchValidationError("Question priority must be 1-5")
        if self.parent_question_id is not None:
            validate_identifier(self.parent_question_id)


@dataclass(frozen=True, slots=True)
class ResearchSource:
    source_id: str
    title: str
    source_type: SourceType
    uri: str
    publisher: str | None = None
    authors: tuple[str, ...] = ()
    published_at: datetime | None = None
    retrieved_at: datetime = field(default_factory=utc_now)
    doi: str | None = None
    content: str = ""
    relevance_score: float = 0.0
    credibility_score: float = 0.0
    tenant_id: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.source_id)
        validate_text(self.title, "Source title", 1000)
        validate_text(self.uri, "Source URI", 4000)
        validate_probability(self.relevance_score)
        validate_probability(self.credibility_score)
        validate_timestamp(self.retrieved_at)

        if self.published_at is not None:
            validate_timestamp(self.published_at)

        if self.tenant_id is not None:
            validate_identifier(self.tenant_id)

        if len(self.content) > MAX_SOURCE_CONTENT_CHARS:
            raise ResearchValidationError("Source content exceeds limit")


@dataclass(frozen=True, slots=True)
class ResearchEvidence:
    evidence_id: str
    source_id: str
    question_id: str
    excerpt: str
    interpretation: str
    confidence: float
    strength: EvidenceStrength
    location: str | None = None
    supports_claim: bool = True

    def __post_init__(self) -> None:
        validate_identifier(self.evidence_id)
        validate_identifier(self.source_id)
        validate_identifier(self.question_id)
        validate_text(self.excerpt, "Evidence excerpt", 5000)
        validate_probability(self.confidence)


@dataclass(frozen=True, slots=True)
class ResearchClaim:
    claim_id: str
    statement: str
    evidence_ids: tuple[str, ...]
    verification_status: ClaimVerificationStatus
    confidence: float
    explanation: str

    def __post_init__(self) -> None:
        validate_identifier(self.claim_id)
        validate_text(self.statement, "Claim", 5000)
        validate_probability(self.confidence)
        for evidence_id in self.evidence_ids:
            validate_identifier(evidence_id)


@dataclass(frozen=True, slots=True)
class ResearchContradiction:
    contradiction_id: str
    description: str
    evidence_ids: tuple[str, ...]
    requires_review: bool = True


@dataclass(frozen=True, slots=True)
class ResearchGap:
    gap_id: str
    question_id: str
    description: str
    suggested_follow_up: str
    priority: int = 1


@dataclass(frozen=True, slots=True)
class ResearchQualityMetrics:
    source_count: int
    evidence_count: int
    verified_claim_count: int
    contradicted_claim_count: int
    unsupported_claim_count: int
    question_coverage: float
    average_source_credibility: float
    evidence_grounding_score: float
    overall_quality_score: float


@dataclass(slots=True)
class ResearchRequest:
    tenant_id: str
    topic: str
    domain: ResearchDomain = ResearchDomain.GENERAL
    mode: ResearchMode = ResearchMode.STANDARD
    questions: list[ResearchQuestion] = field(default_factory=list)
    max_sources: int = DEFAULT_MAX_SOURCES
    include_academic_sources: bool = True
    include_technical_documentation: bool = True
    include_internal_documents: bool = False
    require_citations: bool = True
    require_cross_source_verification: bool = True
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_text(self.topic, "Research topic")
        if not 1 <= self.max_sources <= MAX_SOURCES:
            raise ResearchValidationError("Invalid source limit")
        if len(self.questions) > MAX_RESEARCH_QUESTIONS:
            raise ResearchValidationError("Too many research questions")
        identifiers = [question.question_id for question in self.questions]
        if len(identifiers) != len(set(identifiers)):
            raise ResearchValidationError("Duplicate question identifiers")


@dataclass(slots=True)
class ResearchResult:
    research_id: str
    tenant_id: str
    topic: str
    domain: ResearchDomain
    mode: ResearchMode
    status: ResearchStatus
    questions: list[ResearchQuestion]
    sources: list[ResearchSource]
    evidence: list[ResearchEvidence]
    claims: list[ResearchClaim]
    contradictions: list[ResearchContradiction]
    gaps: list[ResearchGap]
    findings: list[str]
    recommendations: list[str]
    limitations: list[str]
    quality_metrics: ResearchQualityMetrics
    requires_human_review: bool
    created_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None


class ResearchQuestionPlanner(Protocol):
    async def generate_questions(
        self,
        *,
        tenant_id: str,
        topic: str,
        domain: ResearchDomain,
        mode: ResearchMode,
        context: Mapping[str, Any],
    ) -> list[ResearchQuestion]:
        ...


class ResearchSearchProvider(Protocol):
    async def search(
        self,
        *,
        tenant_id: str,
        question: ResearchQuestion,
        max_results: int,
        allowed_source_types: frozenset[SourceType],
    ) -> list[ResearchSource]:
        ...


class ResearchEvidenceProvider(Protocol):
    async def extract_evidence(
        self,
        *,
        tenant_id: str,
        question: ResearchQuestion,
        sources: Sequence[ResearchSource],
    ) -> list[ResearchEvidence]:
        ...


class ResearchClaimProvider(Protocol):
    async def synthesize_claims(
        self,
        *,
        tenant_id: str,
        questions: Sequence[ResearchQuestion],
        evidence: Sequence[ResearchEvidence],
    ) -> list[ResearchClaim]:
        ...


class ResearchContradictionProvider(Protocol):
    async def detect_contradictions(
        self,
        *,
        tenant_id: str,
        claims: Sequence[ResearchClaim],
        evidence: Sequence[ResearchEvidence],
    ) -> list[ResearchContradiction]:
        ...


class ResearchAuthorizer(Protocol):
    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        ...


class ResearchRepository(Protocol):
    async def save_research(
        self,
        *,
        result: ResearchResult,
    ) -> None:
        ...

    async def get_research(
        self,
        *,
        tenant_id: str,
        research_id: str,
    ) -> ResearchResult | None:
        ...


class ResearchAuditSink(Protocol):
    async def record(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        action: str,
        resource_id: str,
        metadata: Mapping[str, Any],
    ) -> None:
        ...


class ResearchAgent:
    """Evidence-grounded research orchestration without physical actuation."""

    def __init__(
        self,
        *,
        search_provider: ResearchSearchProvider,
        evidence_provider: ResearchEvidenceProvider,
        claim_provider: ResearchClaimProvider,
        authorizer: ResearchAuthorizer,
        question_planner: ResearchQuestionPlanner | None = None,
        contradiction_provider: ResearchContradictionProvider | None = None,
        repository: ResearchRepository | None = None,
        audit_sink: ResearchAuditSink | None = None,
        max_concurrency: int = DEFAULT_SEARCH_CONCURRENCY,
    ) -> None:
        if not 1 <= max_concurrency <= 10:
            raise ResearchValidationError("Invalid search concurrency")

        self.search_provider = search_provider
        self.evidence_provider = evidence_provider
        self.claim_provider = claim_provider
        self.question_planner = question_planner
        self.contradiction_provider = contradiction_provider
        self.authorizer = authorizer
        self.repository = repository
        self.audit_sink = audit_sink
        self.max_concurrency = max_concurrency

    async def _authorize(
        self,
        *,
        scope: ResearchScope,
        permission: str,
    ) -> None:
        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            permission=permission,
        )

    async def _audit(
        self,
        *,
        scope: ResearchScope,
        action: str,
        resource_id: str,
        metadata: Mapping[str, Any],
    ) -> None:
        if self.audit_sink is not None:
            await self.audit_sink.record(
                tenant_id=scope.tenant_id,
                actor_id=scope.user_id,
                action=action,
                resource_id=resource_id,
                metadata=metadata,
            )

    async def plan_questions(
        self,
        *,
        scope: ResearchScope,
        request: ResearchRequest,
    ) -> list[ResearchQuestion]:
        if request.questions:
            return copy.deepcopy(request.questions)

        if self.question_planner is None:
            return [
                ResearchQuestion(
                    question_id=generate_id(),
                    question=request.topic,
                    domain=request.domain,
                    priority=1,
                )
            ]

        questions = await self.question_planner.generate_questions(
            tenant_id=scope.tenant_id,
            topic=request.topic,
            domain=request.domain,
            mode=request.mode,
            context=copy.deepcopy(request.context),
        )

        if not questions or len(questions) > MAX_RESEARCH_QUESTIONS:
            raise ResearchValidationError(
                "Question planner returned an invalid question count"
            )

        question_ids = [question.question_id for question in questions]
        if len(question_ids) != len(set(question_ids)):
            raise ResearchValidationError("Duplicate planned questions")

        return questions

    @staticmethod
    def allowed_source_types(
        request: ResearchRequest,
    ) -> frozenset[SourceType]:
        allowed = set(SourceType)

        if not request.include_academic_sources:
            allowed.discard(SourceType.ACADEMIC_PAPER)

        if not request.include_technical_documentation:
            allowed.discard(SourceType.TECHNICAL_DOCUMENTATION)
            allowed.discard(SourceType.MANUFACTURER_MANUAL)

        if not request.include_internal_documents:
            allowed.discard(SourceType.INTERNAL_DOCUMENT)

        return frozenset(allowed)

    async def collect_sources(
        self,
        *,
        scope: ResearchScope,
        request: ResearchRequest,
        questions: Sequence[ResearchQuestion],
    ) -> tuple[list[ResearchSource], list[str]]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        allowed = self.allowed_source_types(request)
        limitations: list[str] = []

        async def search_question(
            question: ResearchQuestion,
        ) -> list[ResearchSource]:
            async with semaphore:
                return await self.search_provider.search(
                    tenant_id=scope.tenant_id,
                    question=question,
                    max_results=request.max_sources,
                    allowed_source_types=allowed,
                )

        search_results = await asyncio.gather(
            *(search_question(question) for question in questions),
            return_exceptions=True,
        )

        sources_by_id: dict[str, ResearchSource] = {}
        sources_by_uri: set[str] = set()

        for question, search_result in zip(questions, search_results):
            if isinstance(search_result, BaseException):
                if isinstance(search_result, asyncio.CancelledError):
                    raise search_result
                limitations.append(
                    f"Search failed for question {question.question_id}: "
                    f"{type(search_result).__name__}"
                )
                continue

            for source in search_result:
                if source.source_type not in allowed:
                    continue

                if (
                    source.tenant_id is not None
                    and source.tenant_id != scope.tenant_id
                ):
                    raise ResearchPermissionError(
                        "Search returned cross-tenant source"
                    )

                if source.source_id in sources_by_id:
                    if sources_by_id[source.source_id].uri != source.uri:
                        raise ResearchValidationError(
                            "Conflicting source identifier"
                        )
                    continue

                if source.uri in sources_by_uri:
                    continue

                sources_by_id[source.source_id] = source
                sources_by_uri.add(source.uri)

        sources = sorted(
            sources_by_id.values(),
            key=lambda source: (
                -source.relevance_score,
                -source.credibility_score,
                source.source_id,
            ),
        )[: request.max_sources]

        if not sources:
            limitations.append("No usable research sources were retrieved.")

        return sources, limitations

    async def extract_evidence(
        self,
        *,
        scope: ResearchScope,
        questions: Sequence[ResearchQuestion],
        sources: Sequence[ResearchSource],
    ) -> tuple[list[ResearchEvidence], list[str]]:
        limitations: list[str] = []
        source_map = {source.source_id: source for source in sources}
        question_ids = {question.question_id for question in questions}
        evidence_map: dict[str, ResearchEvidence] = {}

        for question in questions:
            try:
                extracted = await self.evidence_provider.extract_evidence(
                    tenant_id=scope.tenant_id,
                    question=question,
                    sources=sources,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                limitations.append(
                    f"Evidence extraction failed for "
                    f"{question.question_id}: {type(exc).__name__}"
                )
                continue

            for item in extracted:
                if item.source_id not in source_map:
                    raise ResearchEvidenceError(
                        "Evidence references an unknown source"
                    )

                if item.question_id not in question_ids:
                    raise ResearchEvidenceError(
                        "Evidence references an unknown question"
                    )

                if item.question_id != question.question_id:
                    raise ResearchEvidenceError(
                        "Evidence returned for the wrong question"
                    )

                source_content = source_map[item.source_id].content

                if source_content and item.excerpt not in source_content:
                    limitations.append(
                        f"Excerpt for evidence {item.evidence_id} "
                        "could not be matched to retrieved source text."
                    )
                    continue

                if not source_content:
                    limitations.append(
                        f"Source text unavailable for independent excerpt "
                        f"verification: {item.source_id}."
                    )

                existing = evidence_map.get(item.evidence_id)
                if existing is not None and existing != item:
                    raise ResearchEvidenceError(
                        "Conflicting evidence identifier"
                    )

                evidence_map[item.evidence_id] = item

                if len(evidence_map) >= MAX_EVIDENCE_ITEMS:
                    limitations.append("Evidence limit reached.")
                    return list(evidence_map.values()), limitations

        return list(evidence_map.values()), limitations

    @staticmethod
    def validate_claims(
        *,
        claims: Sequence[ResearchClaim],
        evidence: Sequence[ResearchEvidence],
        require_cross_source_verification: bool,
    ) -> tuple[list[ResearchClaim], list[str]]:
        evidence_map = {item.evidence_id: item for item in evidence}
        validated: list[ResearchClaim] = []
        limitations: list[str] = []

        if len(claims) > MAX_CLAIMS:
            raise ResearchValidationError("Claim limit exceeded")

        seen_ids: set[str] = set()

        for claim in claims:
            if claim.claim_id in seen_ids:
                raise ResearchValidationError("Duplicate claim identifier")
            seen_ids.add(claim.claim_id)

            unknown = set(claim.evidence_ids) - set(evidence_map)

            if unknown:
                raise ResearchEvidenceError(
                    f"Claim {claim.claim_id} references unknown evidence"
                )

            supporting_sources = {
                evidence_map[evidence_id].source_id
                for evidence_id in claim.evidence_ids
                if evidence_map[evidence_id].supports_claim
            }

            verification_status = claim.verification_status
            confidence = claim.confidence

            if not claim.evidence_ids:
                verification_status = (
                    ClaimVerificationStatus.INSUFFICIENT_EVIDENCE
                )
                confidence = 0.0

            elif (
                require_cross_source_verification
                and len(supporting_sources) < 2
                and verification_status
                == ClaimVerificationStatus.SUPPORTED
            ):
                verification_status = (
                    ClaimVerificationStatus.PARTIALLY_SUPPORTED
                )
                confidence = min(confidence, 0.60)
                limitations.append(
                    f"Claim {claim.claim_id} lacks independent "
                    "cross-source support."
                )

            validated.append(
                ResearchClaim(
                    claim_id=claim.claim_id,
                    statement=claim.statement,
                    evidence_ids=claim.evidence_ids,
                    verification_status=verification_status,
                    confidence=confidence,
                    explanation=claim.explanation,
                )
            )

        return validated, limitations

    @staticmethod
    def identify_gaps(
        *,
        questions: Sequence[ResearchQuestion],
        evidence: Sequence[ResearchEvidence],
    ) -> list[ResearchGap]:
        evidence_by_question: dict[str, list[ResearchEvidence]] = {}

        for item in evidence:
            evidence_by_question.setdefault(
                item.question_id, []
            ).append(item)

        gaps: list[ResearchGap] = []

        for question in questions:
            items = evidence_by_question.get(question.question_id, [])
            strong_items = [
                item
                for item in items
                if item.strength in (
                    EvidenceStrength.STRONG,
                    EvidenceStrength.MODERATE,
                )
            ]

            if not strong_items:
                gaps.append(
                    ResearchGap(
                        gap_id=generate_id(),
                        question_id=question.question_id,
                        description=(
                            "Insufficient strong or moderate evidence "
                            "for this research question."
                        ),
                        suggested_follow_up=(
                            "Retrieve primary research, official "
                            "documentation, or validated experimental data."
                        ),
                        priority=question.priority,
                    )
                )

        return gaps

    @staticmethod
    def calculate_quality_metrics(
        *,
        questions: Sequence[ResearchQuestion],
        sources: Sequence[ResearchSource],
        evidence: Sequence[ResearchEvidence],
        claims: Sequence[ResearchClaim],
    ) -> ResearchQualityMetrics:
        covered_questions = {
            item.question_id
            for item in evidence
            if item.strength != EvidenceStrength.UNVERIFIED
        }

        coverage = (
            len(covered_questions) / len(questions)
            if questions
            else 0.0
        )

        credibility = (
            sum(source.credibility_score for source in sources)
            / len(sources)
            if sources
            else 0.0
        )

        verified = sum(
            claim.verification_status
            == ClaimVerificationStatus.SUPPORTED
            for claim in claims
        )

        contradicted = sum(
            claim.verification_status
            == ClaimVerificationStatus.CONTRADICTED
            for claim in claims
        )

        unsupported = sum(
            claim.verification_status
            == ClaimVerificationStatus.INSUFFICIENT_EVIDENCE
            for claim in claims
        )

        grounding = (
            sum(
                claim.confidence
                for claim in claims
                if claim.verification_status
                == ClaimVerificationStatus.SUPPORTED
            ) / len(claims)
            if claims
            else 0.0
        )

        overall = (
            0.35 * coverage
            + 0.25 * credibility
            + 0.40 * grounding
        )

        return ResearchQualityMetrics(
            source_count=len(sources),
            evidence_count=len(evidence),
            verified_claim_count=verified,
            contradicted_claim_count=contradicted,
            unsupported_claim_count=unsupported,
            question_coverage=round(coverage, 4),
            average_source_credibility=round(credibility, 4),
            evidence_grounding_score=round(grounding, 4),
            overall_quality_score=round(overall, 4),
        )

    async def conduct_research(
        self,
        *,
        scope: ResearchScope,
        request: ResearchRequest,
    ) -> ResearchResult:
        if request.tenant_id != scope.tenant_id:
            raise ResearchPermissionError("Research tenant mismatch")

        await self._authorize(
            scope=scope,
            permission="research:execute",
        )

        created_at = utc_now()
        limitations: list[str] = []

        questions = await self.plan_questions(
            scope=scope,
            request=request,
        )

        sources, source_limitations = await self.collect_sources(
            scope=scope,
            request=request,
            questions=questions,
        )
        limitations.extend(source_limitations)

        evidence, evidence_limitations = await self.extract_evidence(
            scope=scope,
            questions=questions,
            sources=sources,
        )
        limitations.extend(evidence_limitations)

        if evidence:
            raw_claims = await self.claim_provider.synthesize_claims(
                tenant_id=scope.tenant_id,
                questions=questions,
                evidence=evidence,
            )
        else:
            raw_claims = []

        claims, claim_limitations = self.validate_claims(
            claims=raw_claims,
            evidence=evidence,
            require_cross_source_verification=(
                request.require_cross_source_verification
            ),
        )
        limitations.extend(claim_limitations)

        contradictions: list[ResearchContradiction] = []

        if self.contradiction_provider is not None and claims:
            contradictions = (
                await self.contradiction_provider.detect_contradictions(
                    tenant_id=scope.tenant_id,
                    claims=claims,
                    evidence=evidence,
                )
            )

            known_evidence = {item.evidence_id for item in evidence}
            for contradiction in contradictions:
                if not set(contradiction.evidence_ids).issubset(
                    known_evidence
                ):
                    raise ResearchEvidenceError(
                        "Contradiction references unknown evidence"
                    )

        gaps = self.identify_gaps(
            questions=questions,
            evidence=evidence,
        )

        metrics = self.calculate_quality_metrics(
            questions=questions,
            sources=sources,
            evidence=evidence,
            claims=claims,
        )

        findings = [
            claim.statement
            for claim in claims
            if claim.verification_status
            == ClaimVerificationStatus.SUPPORTED
        ]

        recommendations = [
            gap.suggested_follow_up
            for gap in gaps
        ]

        if contradictions:
            recommendations.append(
                "Resolve contradictory findings using independent "
                "primary sources and expert review."
            )

        requires_review = bool(
            contradictions
            or gaps
            or metrics.overall_quality_score < 0.70
            or request.domain == ResearchDomain.ROBOT_SAFETY
        )

        if not sources:
            status = ResearchStatus.FAILED
        elif requires_review:
            status = ResearchStatus.REVIEW_REQUIRED
        elif limitations:
            status = ResearchStatus.PARTIAL
        else:
            status = ResearchStatus.COMPLETED

        result = ResearchResult(
            research_id=generate_id(),
            tenant_id=scope.tenant_id,
            topic=request.topic,
            domain=request.domain,
            mode=request.mode,
            status=status,
            questions=list(questions),
            sources=list(sources),
            evidence=list(evidence),
            claims=list(claims),
            contradictions=contradictions,
            gaps=gaps,
            findings=findings,
            recommendations=recommendations,
            limitations=limitations,
            quality_metrics=metrics,
            requires_human_review=requires_review,
            created_at=created_at,
            completed_at=utc_now(),
        )

        if self.repository is not None:
            await self.repository.save_research(result=result)

        await self._audit(
            scope=scope,
            action="research_agent.research_completed",
            resource_id=result.research_id,
            metadata={
                "status": status.value,
                "domain": request.domain.value,
                "source_count": len(sources),
                "evidence_count": len(evidence),
                "claim_count": len(claims),
                "quality_score": metrics.overall_quality_score,
            },
        )

        return result

    async def get_research(
        self,
        *,
        scope: ResearchScope,
        research_id: str,
    ) -> ResearchResult:
        validate_identifier(research_id)

        await self._authorize(
            scope=scope,
            permission="research:read",
        )

        if self.repository is None:
            raise ResearchServiceUnavailableError(
                "Research repository is unavailable"
            )

        result = await self.repository.get_research(
            tenant_id=scope.tenant_id,
            research_id=research_id,
        )

        if result is None:
            raise ResearchNotFoundError("Research result not found")

        if result.tenant_id != scope.tenant_id:
            raise ResearchPermissionError("Research tenant mismatch")

        return result

    async def conduct_batch_research(
        self,
        *,
        scope: ResearchScope,
        requests: Sequence[ResearchRequest],
    ) -> list[ResearchResult]:
        if len(requests) > MAX_BATCH_REQUESTS:
            raise ResearchValidationError("Batch limit exceeded")

        semaphore = asyncio.Semaphore(3)

        async def run(request: ResearchRequest) -> ResearchResult:
            async with semaphore:
                return await self.conduct_research(
                    scope=scope,
                    request=request,
                )

        return list(
            await asyncio.gather(
                *(run(request) for request in requests)
            )
        )

    @staticmethod
    def generate_research_report(
        result: ResearchResult,
    ) -> dict[str, Any]:
        source_map = {
            source.source_id: source
            for source in result.sources
        }
        evidence_map = {
            item.evidence_id: item
            for item in result.evidence
        }

        return {
            "research_id": result.research_id,
            "tenant_id": result.tenant_id,
            "topic": result.topic,
            "domain": result.domain.value,
            "mode": result.mode.value,
            "status": result.status.value,
            "created_at": result.created_at.isoformat(),
            "completed_at": (
                result.completed_at.isoformat()
                if result.completed_at
                else None
            ),
            "questions": [
                {
                    "question_id": question.question_id,
                    "question": question.question,
                    "domain": question.domain.value,
                    "priority": question.priority,
                }
                for question in result.questions
            ],
            "sources": [
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "source_type": source.source_type.value,
                    "uri": source.uri,
                    "publisher": source.publisher,
                    "authors": list(source.authors),
                    "doi": source.doi,
                    "published_at": (
                        source.published_at.isoformat()
                        if source.published_at
                        else None
                    ),
                    "credibility_score": source.credibility_score,
                    "relevance_score": source.relevance_score,
                }
                for source in result.sources
            ],
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "statement": claim.statement,
                    "verification_status": (
                        claim.verification_status.value
                    ),
                    "confidence": claim.confidence,
                    "explanation": claim.explanation,
                    "citations": [
                        {
                            "evidence_id": evidence_id,
                            "source_id": evidence_map[evidence_id].source_id,
                            "source_title": source_map[
                                evidence_map[evidence_id].source_id
                            ].title,
                            "source_uri": source_map[
                                evidence_map[evidence_id].source_id
                            ].uri,
                            "excerpt": evidence_map[evidence_id].excerpt,
                            "location": evidence_map[evidence_id].location,
                        }
                        for evidence_id in claim.evidence_ids
                        if (
                            evidence_id in evidence_map
                            and evidence_map[evidence_id].source_id
                            in source_map
                        )
                    ],
                }
                for claim in result.claims
            ],
            "contradictions": [
                {
                    "contradiction_id": item.contradiction_id,
                    "description": item.description,
                    "evidence_ids": list(item.evidence_ids),
                    "requires_review": item.requires_review,
                }
                for item in result.contradictions
            ],
            "research_gaps": [
                {
                    "gap_id": gap.gap_id,
                    "question_id": gap.question_id,
                    "description": gap.description,
                    "suggested_follow_up": gap.suggested_follow_up,
                    "priority": gap.priority,
                }
                for gap in result.gaps
            ],
            "quality_metrics": {
                "source_count": result.quality_metrics.source_count,
                "evidence_count": result.quality_metrics.evidence_count,
                "verified_claim_count": (
                    result.quality_metrics.verified_claim_count
                ),
                "contradicted_claim_count": (
                    result.quality_metrics.contradicted_claim_count
                ),
                "unsupported_claim_count": (
                    result.quality_metrics.unsupported_claim_count
                ),
                "question_coverage": (
                    result.quality_metrics.question_coverage
                ),
                "average_source_credibility": (
                    result.quality_metrics.average_source_credibility
                ),
                "evidence_grounding_score": (
                    result.quality_metrics.evidence_grounding_score
                ),
                "overall_quality_score": (
                    result.quality_metrics.overall_quality_score
                ),
            },
            "findings": result.findings,
            "recommendations": result.recommendations,
            "limitations": result.limitations,
            "requires_human_review": result.requires_human_review,
            "physical_actuation_enabled": False,
            "research_is_advisory": True,
        }


def create_research_agent(
    *,
    search_provider: ResearchSearchProvider,
    evidence_provider: ResearchEvidenceProvider,
    claim_provider: ResearchClaimProvider,
    authorizer: ResearchAuthorizer,
    question_planner: ResearchQuestionPlanner | None = None,
    contradiction_provider: ResearchContradictionProvider | None = None,
    repository: ResearchRepository | None = None,
    audit_sink: ResearchAuditSink | None = None,
    max_concurrency: int = DEFAULT_SEARCH_CONCURRENCY,
) -> ResearchAgent:
    return ResearchAgent(
        search_provider=search_provider,
        evidence_provider=evidence_provider,
        claim_provider=claim_provider,
        authorizer=authorizer,
        question_planner=question_planner,
        contradiction_provider=contradiction_provider,
        repository=repository,
        audit_sink=audit_sink,
        max_concurrency=max_concurrency,
    )


__all__ = [
    "ResearchAgentError",
    "ResearchValidationError",
    "ResearchPermissionError",
    "ResearchNotFoundError",
    "ResearchProviderError",
    "ResearchEvidenceError",
    "ResearchServiceUnavailableError",
    "ResearchDomain",
    "ResearchMode",
    "SourceType",
    "EvidenceStrength",
    "ClaimVerificationStatus",
    "ResearchStatus",
    "ResearchRiskLevel",
    "ResearchScope",
    "ResearchQuestion",
    "ResearchSource",
    "ResearchEvidence",
    "ResearchClaim",
    "ResearchContradiction",
    "ResearchGap",
    "ResearchQualityMetrics",
    "ResearchRequest",
    "ResearchResult",
    "ResearchQuestionPlanner",
    "ResearchSearchProvider",
    "ResearchEvidenceProvider",
    "ResearchClaimProvider",
    "ResearchContradictionProvider",
    "ResearchAuthorizer",
    "ResearchRepository",
    "ResearchAuditSink",
    "ResearchAgent",
    "create_research_agent",
]
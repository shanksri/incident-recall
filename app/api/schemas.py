from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    database: str


class GitHubIngestRequest(BaseModel):
    owner: str = Field(min_length=1, max_length=100)
    repo: str = Field(min_length=1, max_length=100)
    state: str = Field(default="all", pattern="^(open|closed|all)$")
    limit: int = Field(default=50, ge=1, le=500)
    include_comments: bool = True


class GitHubIngestResponse(BaseModel):
    source: str
    fetched: int
    inserted: int
    updated: int
    skipped: int


class JiraIngestRequest(BaseModel):
    base_url: str = Field(min_length=1, max_length=500)
    project_key: str = Field(min_length=1, max_length=50)
    limit: int = Field(default=50, ge=1, le=500)
    force_backfill: bool = False


class JiraIngestResponse(BaseModel):
    source: str
    fetched: int
    inserted: int
    updated: int
    skipped: int


class IncidentResponse(BaseModel):
    id: uuid.UUID
    source_type: str
    source_external_id: str
    source_url: str | None
    owner: str | None
    repo: str | None
    source: str | None
    state: str | None
    title: str
    description: str
    severity: str
    status: str
    incident_type: str
    environment: dict[str, Any]
    affected_components: list[str]
    tags: list[str]
    canonical_text: str
    created_at_source: datetime | None
    updated_at_source: datetime | None

    model_config = {"from_attributes": True}


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    limit: int = Field(default=10, ge=1, le=50)
    source_type: str | None = Field(default=None, max_length=100)
    tags: list[str] | None = Field(default=None, max_length=50)
    owner: str | None = Field(default=None, max_length=100)
    repo: str | None = Field(default=None, max_length=100)
    source: str | None = Field(default=None, max_length=100)
    state: str | None = Field(default=None, max_length=100)


class SearchResult(BaseModel):
    incident: IncidentResponse
    similarity_score: float
    distance: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    top1_score: float | None = None
    confidence_level: str = "LOW"


class SearchDebugRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    owner: str | None = Field(default=None, max_length=100)
    repo: str | None = Field(default=None, max_length=100)
    source: str | None = Field(default=None, max_length=100)
    state: str | None = Field(default=None, max_length=100)


class SearchDebugResult(BaseModel):
    title: str
    repo: str | None
    similarity_score: float


class SearchDebugResponse(BaseModel):
    query: str
    filters: dict[str, str | None]
    results: list[SearchDebugResult]
    top1_score: float | None = None
    confidence_level: str = "LOW"


class InvestigationRequest(BaseModel):
    """Phase 23A: the single investigation request shape — previously
    duplicated (with narrower fields/no n_hypotheses control) across
    ``/investigate`` and ``/investigate-advanced``'s now-removed request
    models. Those two single-shot agents were earlier, less capable
    implementations of the same business capability this now serves;
    see docs/architecture/19_multi_agent_investigation.md.
    """

    problem: str = Field(min_length=3, max_length=5000)
    n_hypotheses: int = Field(default=3, ge=1, le=10)


class OrchestratedHypothesis(BaseModel):
    id: str
    root_cause: str
    rationale: str
    validation_keywords: list[str]
    raw_confidence: float


class OrchestratedCritique(BaseModel):
    verdict: str
    confidence: float
    explanation: str
    findings: list[str]
    unresolved_questions: list[str]
    missing_evidence: list[str]
    recommended_actions: list[str]


class InvestigationResponse(BaseModel):
    """The single investigation response shape, wiring Phase 19A-19D's
    ``MultiAgentInvestigationOrchestrator`` (planner, evidence-driven
    hypothesis generation, critic, iterative loop) — see
    docs/architecture/19_multi_agent_investigation.md. Reflects the
    orchestrator's final iteration only; use the evaluation API
    (docs/architecture/22_evaluation_api.md) to inspect full iteration
    history for a given problem.
    """

    problem: str
    selected_root_cause: str | None
    confidence: float
    confidence_level: str
    is_uncertain: bool
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    remaining_uncertainty: list[str]
    rejected_hypotheses: list[OrchestratedHypothesis]
    critique: OrchestratedCritique
    total_iterations: int
    stopping_reason: str
    stop_explanation: str


class GitHubIssueRef(BaseModel):
    url: HttpUrl


# ── Phase 25: alert webhook (Prometheus/PagerDuty -> investigation) ────────


class AlertWebhookRequest(BaseModel):
    """Wraps the monitoring provider's own raw payload rather than
    reshaping it — Prometheus and PagerDuty both send their native JSON
    unmodified, and forcing that into a bespoke schema would mean
    maintaining a mapping for every provider at this boundary AND inside
    ``app.services.alert_normalization``. ``repo_owner``/``repo_name`` are
    supplied by the caller (this platform has no service-to-repository
    mapping of its own) and are optional — omit them to skip deployment
    history and get an investigation with no deployment context.
    """

    provider_payload: dict[str, Any] = Field(
        description="The raw webhook body exactly as sent by Prometheus Alertmanager or PagerDuty."
    )
    repo_owner: str | None = Field(default=None, max_length=200)
    repo_name: str | None = Field(default=None, max_length=200)
    n_hypotheses: int = Field(default=3, ge=1, le=10)


class NormalizedAlertSummary(BaseModel):
    source_type: str
    alert_name: str
    severity: str
    service: str | None
    summary: str
    problem_statement: str


class DeploymentEventSummary(BaseModel):
    sha: str
    short_sha: str
    author: str | None
    message: str
    committed_at: datetime
    url: str


class ProposedActionSummary(BaseModel):
    action_type: str
    description: str
    target: str
    confidence: float | None


class PolicyDecisionSummary(BaseModel):
    category: str
    requires_approval: bool
    auto_executable: bool
    reason: str


class AlertWebhookResponse(BaseModel):
    """The alert is normalized, an investigation is run against its problem
    statement, and — only when a repository was supplied and the
    investigation is not uncertain — a deployment-history lookup and a
    heuristic action suggestion are attached, each passed through the
    execution policy engine. ``suggested_action``/``policy_decision`` are
    null whenever the investigation abstained: there is nothing to act on.
    """

    alert: NormalizedAlertSummary
    investigation: InvestigationResponse
    deployment_context: list[DeploymentEventSummary]
    suggested_action: ProposedActionSummary | None
    policy_decision: PolicyDecisionSummary | None

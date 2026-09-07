"""Phase 25: alert-webhook ingestion — turns a monitoring provider's alert
into a triggered investigation, with an optional deployment-history lookup
and a policy-gated action suggestion layered on top.

# Where this sits relative to /agent/investigate

This route does not introduce a new investigation capability.
``MultiAgentInvestigationOrchestrator`` is called exactly as
``/agent/investigate`` calls it — same class, same method, same response
shape underneath. What this route adds is entirely upstream (turning a
provider payload into a problem statement) and entirely downstream
(deployment context and a policy-gated suggestion). A human typing the same
problem into ``/agent/investigate`` gets an identical investigation.

# The suggested-action heuristic, and why it is labelled as one

After the investigation returns, this route makes exactly one deterministic
judgment call to decide what action to *suggest* (never to execute):

  - investigation abstained (``is_uncertain``)      -> suggest nothing
  - a deployment landed in the failure window
    AND confidence is HIGH                          -> suggest a rollback
  - otherwise                                        -> suggest gathering
                                                         more diagnostics

This is authored by inspection, exactly like ``RuleBasedPlanner``'s keyword
priority order and ``_CATALOG`` in ``action_policy.py`` — it is a plausible
default, not validated against real incident/deployment correlation data.
Whatever it suggests is then run through ``ActionPolicyEngine``, which is
the part of this pipeline that is actually load-bearing: the suggestion can
be as wrong as it likes, because a state-changing suggestion NEVER executes
without human approval regardless of what this heuristic or the
investigation's confidence say. See ``app.services.action_policy`` for that
guarantee.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import require_api_key
from app.api.dependencies import DbSession
from app.api.rate_limit import RATE_LIMIT_RESPONSES, alerts_rate_limit
from app.api.schemas import (
    AlertWebhookRequest,
    AlertWebhookResponse,
    DeploymentEventSummary,
    InvestigationResponse,
    NormalizedAlertSummary,
    OrchestratedCritique,
    OrchestratedHypothesis,
    PolicyDecisionSummary,
    ProposedActionSummary,
)
from app.core.config import settings
from app.services.action_policy import ActionPolicyEngine, ProposedAction
from app.services.alert_normalization import (
    UnrecognizedAlertPayloadError,
    build_problem_statement,
    normalize_alert_payload,
)
from app.services.deployment_history import DeploymentHistoryClient, DeploymentHistoryError
from app.services.investigation_orchestrator import MultiAgentInvestigationOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/alerts",
    tags=["alerts"],
    dependencies=[Depends(require_api_key), Depends(alerts_rate_limit)],
    responses=RATE_LIMIT_RESPONSES,
)

_policy_engine = ActionPolicyEngine()


def _investigation_response(session) -> InvestigationResponse:
    """Build the same response shape /agent/investigate returns, so a
    client parses an alert-triggered investigation identically to a
    manually-submitted one.
    """
    investigation = session.final_report.investigation
    critique = session.final_report.critique
    return InvestigationResponse(
        problem=investigation.problem,
        selected_root_cause=(
            investigation.selected_hypothesis.root_cause
            if investigation.selected_hypothesis
            else None
        ),
        confidence=investigation.confidence,
        confidence_level=investigation.confidence_level,
        is_uncertain=investigation.is_uncertain,
        supporting_evidence=list(investigation.supporting_evidence),
        contradicting_evidence=list(investigation.contradicting_evidence),
        remaining_uncertainty=list(investigation.remaining_uncertainty),
        rejected_hypotheses=[
            OrchestratedHypothesis(
                id=hypothesis.id,
                root_cause=hypothesis.root_cause,
                rationale=hypothesis.rationale,
                validation_keywords=list(hypothesis.validation_keywords),
                raw_confidence=hypothesis.raw_confidence,
            )
            for hypothesis in investigation.rejected_hypotheses
        ],
        critique=OrchestratedCritique(
            verdict=critique.verdict.value,
            confidence=critique.confidence,
            explanation=critique.explanation,
            findings=list(critique.findings),
            unresolved_questions=list(critique.unresolved_questions),
            missing_evidence=list(critique.missing_evidence),
            recommended_actions=list(critique.recommended_actions),
        ),
        total_iterations=session.total_iterations,
        stopping_reason=session.stopping_reason.value,
        stop_explanation=session.stop_explanation,
    )


@router.post("/webhook", response_model=AlertWebhookResponse)
def alert_webhook(request: AlertWebhookRequest, db: DbSession) -> AlertWebhookResponse:
    try:
        alert = normalize_alert_payload(request.provider_payload)
    except UnrecognizedAlertPayloadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    problem = build_problem_statement(alert)

    try:
        session = MultiAgentInvestigationOrchestrator(db).investigate(
            problem, n_hypotheses=request.n_hypotheses
        )
    except ValueError as exc:
        logger.exception("alert_webhook.investigation_unavailable")
        raise HTTPException(
            status_code=503, detail="Investigation is temporarily unavailable."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("alert_webhook.investigation_failed")
        raise HTTPException(status_code=500, detail="Investigation failed.") from exc

    investigation_response = _investigation_response(session)
    investigation = session.final_report.investigation

    deployment_context: list[DeploymentEventSummary] = []
    if request.repo_owner and request.repo_name and alert.started_at is not None:
        try:
            with DeploymentHistoryClient(token=settings.github_token) as history:
                events = history.find_near(request.repo_owner, request.repo_name, alert.started_at)
            deployment_context = [
                DeploymentEventSummary(
                    sha=e.sha, short_sha=e.short_sha, author=e.author,
                    message=e.message, committed_at=e.committed_at, url=e.url,
                )
                for e in events
            ]
        except DeploymentHistoryError:
            logger.warning("alert_webhook.deployment_history_unavailable", exc_info=True)

    suggested_action: ProposedActionSummary | None = None
    policy_decision: PolicyDecisionSummary | None = None

    if not investigation.is_uncertain:
        target = f"{request.repo_owner}/{request.repo_name}" if request.repo_owner else (
            alert.service or "unknown"
        )
        if deployment_context and investigation.confidence_level == "HIGH":
            recent = deployment_context[0]
            action = ProposedAction(
                action_type="rollback_deployment",
                description=(
                    f"A deployment ({recent.short_sha}: {recent.message}) landed near the "
                    f"failure window and confidence in the diagnosis is HIGH. Suggest "
                    f"rolling back to before this commit."
                ),
                target=target,
                confidence=investigation.confidence,
            )
        else:
            action = ProposedAction(
                action_type="gather_diagnostics",
                description=(
                    "No recent deployment was found near the failure window, or confidence "
                    "in the diagnosis is not HIGH enough to propose remediation."
                ),
                target=target,
                confidence=investigation.confidence,
            )
        decision = _policy_engine.evaluate(action)
        suggested_action = ProposedActionSummary(
            action_type=action.action_type, description=action.description,
            target=action.target, confidence=action.confidence,
        )
        policy_decision = PolicyDecisionSummary(
            category=decision.category.value,
            requires_approval=decision.requires_approval,
            auto_executable=decision.auto_executable,
            reason=decision.reason,
        )

    return AlertWebhookResponse(
        alert=NormalizedAlertSummary(
            source_type=alert.source_type, alert_name=alert.alert_name,
            severity=alert.severity, service=alert.service, summary=alert.summary,
            problem_statement=problem,
        ),
        investigation=investigation_response,
        deployment_context=deployment_context,
        suggested_action=suggested_action,
        policy_decision=policy_decision,
    )

"""MCP server: exposes this platform's investigation, retrieval, and
governance capabilities as tools any MCP client (Claude Desktop, Claude
Code, a custom agent) can call directly, over stdio.

# Why this exists

Every capability this server exposes already exists as a plain Python
service or a FastAPI route: ``MultiAgentInvestigationOrchestrator``,
``IncidentSearchService``, ``ActionPolicyEngine``, ``DeploymentHistoryClient``.
This module adds no new logic — it is a thin MCP-shaped adapter over
services that are independently unit-tested. If a tool call here produces a
wrong answer, the bug is in the underlying service, not in this file, and
the fix belongs in that service's own tests, not in a test of this adapter.

# Why four tools, and why these four

  investigate            -- the core capability: propose and validate a
                             root cause for a described problem.
  search_incidents       -- the retrieval primitive investigate() itself
                             uses internally, exposed directly so a client
                             can look at raw evidence without paying for a
                             full multi-iteration investigation.
  check_action_policy    -- lets a client ask, BEFORE proposing to do
                             anything, whether an action would need human
                             approval. See app.services.action_policy for
                             the rule this enforces: state-changing actions
                             always require approval, regardless of
                             confidence.
  deployment_history     -- "what deployed near this failure window",
                             exposed directly so a client building its own
                             RCA can pull deployment context without going
                             through investigate().

None of these tools execute a remediation. There is no ``rollback`` or
``restart_service`` tool here, on purpose — that boundary is exactly what
``check_action_policy`` exists to enforce, and giving an MCP client a tool
that skips the policy check would defeat the entire point of building it.

# Process and session model

Each tool call opens its own short-lived DB session via
``app.db.session.SessionLocal`` and closes it before returning — an MCP
server process is expected to sit idle between calls for long stretches, so
holding one connection open for the process lifetime would just be an idle
connection leaking out of the pool. This mirrors ``get_db()``'s per-request
lifecycle in the FastAPI app; the two entry points (HTTP request, MCP tool
call) share the same session-per-call discipline even though neither reuses
the other's code path.

# Running it

    python -m app.mcp.server

Requires the same environment variables as the FastAPI app (``OPENAI_API_KEY``
for ``investigate``, ``DATABASE_URL`` for anything that touches the corpus).
``check_action_policy`` needs neither and works with no configuration at all.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from mcp.server.fastmcp import FastMCP

from app.core.config import settings
from app.db.session import SessionLocal
from app.services.action_policy import ActionPolicyEngine, ProposedAction
from app.services.deployment_history import DeploymentHistoryClient, DeploymentHistoryError
from app.services.investigation_orchestrator import MultiAgentInvestigationOrchestrator
from app.services.search import IncidentSearchService

logger = logging.getLogger(__name__)

mcp = FastMCP("incident-intelligence")

_policy_engine = ActionPolicyEngine()


@mcp.tool()
def investigate(problem: str, n_hypotheses: int = 3) -> dict:
    """Run the multi-agent investigation pipeline against a described
    problem and return the accepted root cause, its confidence, and the
    critic's review of it.

    Internally: retrieval -> planner -> hypothesis generation -> evidence
    evaluation -> critic -> orchestrator loop (bounded iterations). Costs
    3-5 LLM calls. Returns is_uncertain=True with selected_root_cause=None
    when no hypothesis clears the confidence floor — that is a valid,
    common result, not an error.
    """
    db = SessionLocal()
    try:
        session = MultiAgentInvestigationOrchestrator(db).investigate(
            problem, n_hypotheses=n_hypotheses
        )
    finally:
        db.close()

    investigation = session.final_report.investigation
    critique = session.final_report.critique
    return {
        "problem": investigation.problem,
        "selected_root_cause": (
            investigation.selected_hypothesis.root_cause
            if investigation.selected_hypothesis
            else None
        ),
        "confidence": investigation.confidence,
        "confidence_level": investigation.confidence_level,
        "is_uncertain": investigation.is_uncertain,
        "supporting_evidence": list(investigation.supporting_evidence),
        "contradicting_evidence": list(investigation.contradicting_evidence),
        "critique_verdict": critique.verdict.value,
        "critique_explanation": critique.explanation,
        "total_iterations": session.total_iterations,
        "stopping_reason": session.stopping_reason.value,
    }


@mcp.tool()
def search_incidents(query: str, limit: int = 5) -> list[dict]:
    """Search the incident corpus directly (hybrid dense retrieval, no
    hypothesis generation or critic review) and return the top matches with
    their similarity scores. Use this to inspect raw evidence, or when you
    only need "has this happened before" rather than a full root-cause
    investigation.
    """
    db = SessionLocal()
    try:
        results = IncidentSearchService(db).search(
            query, limit=limit, call_site="mcp_server.search_incidents"
        )
    finally:
        db.close()

    return [
        {
            "source_type": r.incident.source_type,
            "source_external_id": r.incident.source_external_id,
            "title": r.incident.title,
            "similarity_score": round(r.similarity_score, 4),
            "status": r.incident.status,
            "url": r.incident.source_url,
        }
        for r in results
    ]


@mcp.tool()
def check_action_policy(
    action_type: str,
    description: str,
    target: str,
    confidence: float | None = None,
) -> dict:
    """Ask whether a proposed action would need human approval before it
    could run. This NEVER executes the action -- it only classifies it.

    A read-only action (query_logs, fetch_traces, search_incidents,
    investigate, ...) is always auto-executable. A state-changing action
    (rollback_deployment, restart_service, apply_patch, ...) ALWAYS
    requires approval, regardless of how high ``confidence`` is -- a
    confident diagnosis is not evidence that an action is safe or
    reversible, and this tool will not let a high confidence value change
    its answer. An action type this tool has never seen is treated as
    state-changing (fails closed), not assumed safe.
    """
    decision = _policy_engine.evaluate(
        ProposedAction(
            action_type=action_type, description=description,
            target=target, confidence=confidence,
        )
    )
    return {
        "category": decision.category.value,
        "requires_approval": decision.requires_approval,
        "auto_executable": decision.auto_executable,
        "reason": decision.reason,
    }


@mcp.tool()
def deployment_history(
    owner: str,
    repo: str,
    around_iso: str,
    lookback_hours: float = 24.0,
    lookahead_minutes: float = 15.0,
) -> list[dict]:
    """Find commits to ``owner/repo`` in a window around a failure
    timestamp ("what deployed near this failure window"). ``around_iso``
    is an ISO-8601 timestamp, e.g. "2026-06-15T12:00:00Z". This is a
    correlation-in-time lookup, not a causal claim -- returned commits are
    candidates for a human (or investigate()) to weigh, not a verdict that
    any of them caused the incident.

    Returns an empty list, not an error, on a transient failure (timeout,
    network error) -- the same "fail open on a read" posture as everywhere
    else this platform does diagnostics. Raises only when the repository
    itself cannot be found.
    """
    try:
        around = datetime.fromisoformat(around_iso.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError as exc:
        raise ValueError(
            f"around_iso={around_iso!r} is not a valid ISO-8601 timestamp"
        ) from exc

    with DeploymentHistoryClient(token=settings.github_token) as client:
        try:
            events = client.find_near(
                owner, repo, around,
                lookback=timedelta(hours=lookback_hours),
                lookahead=timedelta(minutes=lookahead_minutes),
            )
        except DeploymentHistoryError as exc:
            raise ValueError(str(exc)) from exc

    return [
        {
            "sha": e.sha, "short_sha": e.short_sha, "author": e.author,
            "message": e.message, "committed_at": e.committed_at.isoformat(), "url": e.url,
        }
        for e in events
    ]


if __name__ == "__main__":
    mcp.run()

"""Tests for the MCP tool adapters in app/mcp/server.py.

These tools add no logic of their own -- they wrap already-tested services
(MultiAgentInvestigationOrchestrator, IncidentSearchService,
ActionPolicyEngine, DeploymentHistoryClient). So these tests exercise the
ADAPTER: does the tool open and close a DB session per call, does it shape
the service's return value into the documented dict correctly, does it
propagate the right kind of failure. They deliberately do not re-test
service-level behavior that already has its own test file.

@mcp.tool()-decorated functions remain directly callable as plain Python
functions (verified: FastMCP's decorator returns the original function), so
no MCP client or protocol session is needed here.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import app.mcp.server as server


# ── investigate() ────────────────────────────────────────────────────────────


def _fake_session(*, uncertain: bool = False):
    from app.services.critic_agent import (
        CritiqueResult,
        CritiqueVerdict,
        CritiquedInvestigationReport,
    )
    from app.services.hypothesis_investigation import (
        InvestigationHypothesis,
        InvestigationReport,
    )
    from app.services.investigation_orchestrator import InvestigationSession, StoppingReason

    accepted = None if uncertain else InvestigationHypothesis(
        id="h1", root_cause="expired cert", rationale="matches timing",
        validation_keywords=("cert",), raw_confidence=0.9,
    )
    investigation = InvestigationReport(
        problem="tls handshake failing", selected_hypothesis=accepted,
        confidence=0.0 if uncertain else 0.8,
        confidence_level="LOW" if uncertain else "HIGH",
        supporting_evidence=() if uncertain else ("Incident #7",),
        contradicting_evidence=(), remaining_uncertainty=(),
        is_uncertain=uncertain, rejected_hypotheses=(),
    )
    critique = CritiqueResult(
        verdict=CritiqueVerdict.INCONCLUSIVE if uncertain else CritiqueVerdict.APPROVED,
        confidence=0.0 if uncertain else 0.8,
        findings=(), unresolved_questions=(), missing_evidence=(), recommended_actions=(),
        explanation="ok",
    )
    return InvestigationSession(
        final_report=CritiquedInvestigationReport(investigation=investigation, critique=critique),
        iterations=(), stopping_reason=StoppingReason.MAX_ITERATIONS if uncertain
        else StoppingReason.CRITIC_APPROVED,
        total_iterations=1, stop_explanation="stopped",
    )


def test_investigate_shapes_the_orchestrator_result(monkeypatch) -> None:
    fake_orch = MagicMock()
    fake_orch.investigate.return_value = _fake_session()
    monkeypatch.setattr(server, "MultiAgentInvestigationOrchestrator", lambda db: fake_orch)
    fake_db = MagicMock()
    monkeypatch.setattr(server, "SessionLocal", lambda: fake_db)

    result = server.investigate("tls handshake failing")

    assert result["selected_root_cause"] == "expired cert"
    assert result["confidence"] == 0.8
    assert result["confidence_level"] == "HIGH"
    assert result["is_uncertain"] is False
    assert result["critique_verdict"] == "approved"
    assert result["stopping_reason"] == "critic_approved"
    fake_orch.investigate.assert_called_once_with("tls handshake failing", n_hypotheses=3)


def test_investigate_passes_through_n_hypotheses(monkeypatch) -> None:
    fake_orch = MagicMock()
    fake_orch.investigate.return_value = _fake_session()
    monkeypatch.setattr(server, "MultiAgentInvestigationOrchestrator", lambda db: fake_orch)
    monkeypatch.setattr(server, "SessionLocal", lambda: MagicMock())

    server.investigate("x", n_hypotheses=5)
    fake_orch.investigate.assert_called_once_with("x", n_hypotheses=5)


def test_investigate_uncertain_result_has_no_root_cause(monkeypatch) -> None:
    fake_orch = MagicMock()
    fake_orch.investigate.return_value = _fake_session(uncertain=True)
    monkeypatch.setattr(server, "MultiAgentInvestigationOrchestrator", lambda db: fake_orch)
    monkeypatch.setattr(server, "SessionLocal", lambda: MagicMock())

    result = server.investigate("x")
    assert result["selected_root_cause"] is None
    assert result["is_uncertain"] is True


def test_investigate_closes_the_db_session_even_on_failure(monkeypatch) -> None:
    fake_orch = MagicMock()
    fake_orch.investigate.side_effect = RuntimeError("boom")
    monkeypatch.setattr(server, "MultiAgentInvestigationOrchestrator", lambda db: fake_orch)
    fake_db = MagicMock()
    monkeypatch.setattr(server, "SessionLocal", lambda: fake_db)

    with pytest.raises(RuntimeError, match="boom"):
        server.investigate("x")
    fake_db.close.assert_called_once()


def test_investigate_closes_the_db_session_on_success(monkeypatch) -> None:
    fake_orch = MagicMock()
    fake_orch.investigate.return_value = _fake_session()
    monkeypatch.setattr(server, "MultiAgentInvestigationOrchestrator", lambda db: fake_orch)
    fake_db = MagicMock()
    monkeypatch.setattr(server, "SessionLocal", lambda: fake_db)

    server.investigate("x")
    fake_db.close.assert_called_once()


# ── search_incidents() ───────────────────────────────────────────────────────


def _fake_result(title: str, distance: float, *, source_external_id: str = "o/r#1"):
    incident = SimpleNamespace(
        source_type="github", source_external_id=source_external_id, title=title,
        status="open", source_url=f"https://github.com/{source_external_id}",
    )
    return SimpleNamespace(incident=incident, distance=distance,
                            similarity_score=max(0.0, 1.0 - distance))


def test_search_incidents_shapes_results(monkeypatch) -> None:
    fake_search_service = MagicMock()
    fake_search_service.search.return_value = [_fake_result("Pod OOMKilled", 0.2)]
    monkeypatch.setattr(server, "IncidentSearchService", lambda db: fake_search_service)
    monkeypatch.setattr(server, "SessionLocal", lambda: MagicMock())

    results = server.search_incidents("pod crash")
    assert len(results) == 1
    assert results[0]["title"] == "Pod OOMKilled"
    assert results[0]["similarity_score"] == 0.8
    assert results[0]["source_external_id"] == "o/r#1"


def test_search_incidents_passes_through_limit(monkeypatch) -> None:
    fake_search_service = MagicMock()
    fake_search_service.search.return_value = []
    monkeypatch.setattr(server, "IncidentSearchService", lambda db: fake_search_service)
    monkeypatch.setattr(server, "SessionLocal", lambda: MagicMock())

    server.search_incidents("q", limit=15)
    _, kwargs = fake_search_service.search.call_args
    assert kwargs["limit"] == 15


def test_search_incidents_empty_result_is_an_empty_list_not_an_error(monkeypatch) -> None:
    fake_search_service = MagicMock()
    fake_search_service.search.return_value = []
    monkeypatch.setattr(server, "IncidentSearchService", lambda db: fake_search_service)
    monkeypatch.setattr(server, "SessionLocal", lambda: MagicMock())

    assert server.search_incidents("nothing matches this") == []


def test_search_incidents_closes_the_db_session(monkeypatch) -> None:
    fake_search_service = MagicMock()
    fake_search_service.search.return_value = []
    monkeypatch.setattr(server, "IncidentSearchService", lambda db: fake_search_service)
    fake_db = MagicMock()
    monkeypatch.setattr(server, "SessionLocal", lambda: fake_db)

    server.search_incidents("q")
    fake_db.close.assert_called_once()


# ── check_action_policy() ────────────────────────────────────────────────────


def test_check_action_policy_read_only_is_auto_executable() -> None:
    result = server.check_action_policy("query_logs", "read the logs", "checkout-api")
    assert result["category"] == "read_only"
    assert result["requires_approval"] is False
    assert result["auto_executable"] is True


def test_check_action_policy_state_changing_requires_approval_regardless_of_confidence() -> None:
    result = server.check_action_policy(
        "rollback_deployment", "roll it back", "checkout-api", confidence=0.99
    )
    assert result["category"] == "state_changing"
    assert result["requires_approval"] is True
    assert result["auto_executable"] is False
    assert "does not change this" in result["reason"]


def test_check_action_policy_unknown_action_fails_closed() -> None:
    result = server.check_action_policy("do_something_novel", "d", "t")
    assert result["category"] == "state_changing"
    assert result["requires_approval"] is True


def test_check_action_policy_never_executes_anything() -> None:
    """There is no execution path in this tool at all -- it can only
    classify. Asserted by checking the return shape carries no side-effect
    markers (no 'executed', no 'status: done', etc.)."""
    result = server.check_action_policy("rollback_deployment", "d", "t")
    assert set(result.keys()) == {"category", "requires_approval", "auto_executable", "reason"}


# ── deployment_history() ─────────────────────────────────────────────────────


def _fake_event(sha="a" * 40):
    from app.services.deployment_history import DeploymentEvent
    return DeploymentEvent(
        sha=sha, short_sha=sha[:7], author="octocat", message="ship it",
        committed_at=datetime(2026, 6, 15, 11, 0, tzinfo=timezone.utc),
        url=f"https://github.com/o/r/commit/{sha}",
    )


class _FakeHistoryClient:
    def __init__(self, events=None, *, raises=None):
        self._events = events or []
        self._raises = raises
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    def find_near(self, owner, repo, around, **kwargs):
        self.calls.append({"owner": owner, "repo": repo, "around": around, **kwargs})
        if self._raises is not None:
            raise self._raises
        return self._events


def test_deployment_history_shapes_results(monkeypatch) -> None:
    fake = _FakeHistoryClient(events=[_fake_event()])
    monkeypatch.setattr(server, "DeploymentHistoryClient", lambda **kw: fake)

    results = server.deployment_history("o", "r", "2026-06-15T12:00:00Z")
    assert len(results) == 1
    assert results[0]["short_sha"] == "aaaaaaa"
    assert results[0]["committed_at"] == "2026-06-15T11:00:00+00:00"


def test_deployment_history_parses_the_timestamp_and_builds_the_window(monkeypatch) -> None:
    fake = _FakeHistoryClient(events=[])
    monkeypatch.setattr(server, "DeploymentHistoryClient", lambda **kw: fake)

    server.deployment_history(
        "o", "r", "2026-06-15T12:00:00Z", lookback_hours=2, lookahead_minutes=30
    )
    call = fake.calls[0]
    assert call["around"] == datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    assert call["lookback"] == timedelta(hours=2)
    assert call["lookahead"] == timedelta(minutes=30)


def test_deployment_history_invalid_timestamp_raises_value_error(monkeypatch) -> None:
    monkeypatch.setattr(server, "DeploymentHistoryClient", lambda **kw: _FakeHistoryClient())
    with pytest.raises(ValueError, match="not a valid ISO-8601"):
        server.deployment_history("o", "r", "not-a-date")


def test_deployment_history_error_is_surfaced_as_value_error(monkeypatch) -> None:
    from app.services.deployment_history import DeploymentHistoryError

    fake = _FakeHistoryClient(raises=DeploymentHistoryError("repo not found"))
    monkeypatch.setattr(server, "DeploymentHistoryClient", lambda **kw: fake)

    with pytest.raises(ValueError, match="repo not found"):
        server.deployment_history("o", "nonexistent", "2026-06-15T12:00:00Z")


def test_deployment_history_empty_result_is_an_empty_list(monkeypatch) -> None:
    fake = _FakeHistoryClient(events=[])
    monkeypatch.setattr(server, "DeploymentHistoryClient", lambda **kw: fake)
    assert server.deployment_history("o", "r", "2026-06-15T12:00:00Z") == []

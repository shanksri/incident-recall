"""API tests for POST /alerts/webhook.

No database, no OpenAI, no retrieval, no real GitHub call —
MultiAgentInvestigationOrchestrator and DeploymentHistoryClient are
monkeypatched; only routing, normalization dispatch, and the
suggested-action/policy wiring are exercised.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.auth import require_api_key
from app.db.session import get_db
from app.main import app


def _client() -> TestClient:
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[require_api_key] = lambda: None
    return TestClient(app)


def _teardown() -> None:
    app.dependency_overrides.clear()


PROMETHEUS_PAYLOAD = {
    "alerts": [
        {
            "labels": {"alertname": "HighErrorRate", "severity": "critical", "service": "checkout-api"},
            "annotations": {"summary": "Error rate above 5%", "description": "5xx spike"},
            "startsAt": "2026-06-15T12:00:00Z",
        }
    ]
}


def _fake_session(*, uncertain: bool = False, confidence: float = 0.85, confidence_level: str = "HIGH"):
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
        id="h1", root_cause="bad deploy of payment-service",
        rationale="matches timing", validation_keywords=("deploy",), raw_confidence=confidence,
    )
    investigation = InvestigationReport(
        problem="HighErrorRate firing", selected_hypothesis=accepted,
        confidence=0.0 if uncertain else confidence,
        confidence_level="LOW" if uncertain else confidence_level,
        supporting_evidence=() if uncertain else ("Incident #42",),
        contradicting_evidence=(), remaining_uncertainty=(),
        is_uncertain=uncertain, rejected_hypotheses=(),
    )
    critique = CritiqueResult(
        verdict=CritiqueVerdict.INCONCLUSIVE if uncertain else CritiqueVerdict.APPROVED,
        confidence=0.0 if uncertain else confidence,
        findings=(), unresolved_questions=(), missing_evidence=(), recommended_actions=(),
        explanation="ok",
    )
    return InvestigationSession(
        final_report=CritiquedInvestigationReport(investigation=investigation, critique=critique),
        iterations=(),
        stopping_reason=StoppingReason.MAX_ITERATIONS if uncertain else StoppingReason.CRITIC_APPROVED,
        total_iterations=1, stop_explanation="stopped",
    )


def _patch_orchestrator(monkeypatch, session):
    import app.api.routes.alerts as alerts_mod

    fake = MagicMock()
    fake.investigate.return_value = session
    monkeypatch.setattr(alerts_mod, "MultiAgentInvestigationOrchestrator", lambda db: fake)
    return fake


# ── Basic dispatch and normalization ────────────────────────────────────────


def test_prometheus_alert_triggers_an_investigation(monkeypatch) -> None:
    fake = _patch_orchestrator(monkeypatch, _fake_session())
    client = _client()
    try:
        resp = client.post("/alerts/webhook", json={"provider_payload": PROMETHEUS_PAYLOAD})
        assert resp.status_code == 200
        body = resp.json()
        assert body["alert"]["source_type"] == "prometheus"
        assert body["alert"]["alert_name"] == "HighErrorRate"
        assert "HighErrorRate" in body["alert"]["problem_statement"]
        assert body["investigation"]["selected_root_cause"] == "bad deploy of payment-service"
        fake.investigate.assert_called_once()
        called_problem = fake.investigate.call_args.args[0]
        assert "HighErrorRate" in called_problem
    finally:
        _teardown()


def test_unrecognized_payload_returns_422(monkeypatch) -> None:
    client = _client()
    try:
        resp = client.post("/alerts/webhook", json={"provider_payload": {"nonsense": True}})
        assert resp.status_code == 422
    finally:
        _teardown()


def test_pagerduty_alert_is_also_recognized(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch, _fake_session())
    payload = {
        "event": {
            "event_type": "incident.triggered",
            "data": {
                "title": "Payment service down", "urgency": "high",
                "service": {"summary": "payment-service"},
                "created_at": "2026-06-15T11:55:00Z",
            },
        }
    }
    client = _client()
    try:
        resp = client.post("/alerts/webhook", json={"provider_payload": payload})
        assert resp.status_code == 200
        assert resp.json()["alert"]["source_type"] == "pagerduty"
    finally:
        _teardown()


# ── Deployment context and suggested action ─────────────────────────────────


def test_no_repo_supplied_means_no_deployment_context_or_suggestion(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch, _fake_session())
    client = _client()
    try:
        resp = client.post("/alerts/webhook", json={"provider_payload": PROMETHEUS_PAYLOAD})
        body = resp.json()
        assert body["deployment_context"] == []
        # No repo was supplied, but the investigation was NOT uncertain, so a
        # suggestion should still be produced (gather_diagnostics, since there's
        # no deployment context to justify a rollback).
        assert body["suggested_action"]["action_type"] == "gather_diagnostics"
        assert body["policy_decision"]["requires_approval"] is False
    finally:
        _teardown()


def test_uncertain_investigation_suggests_nothing(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch, _fake_session(uncertain=True))
    client = _client()
    try:
        resp = client.post("/alerts/webhook", json={"provider_payload": PROMETHEUS_PAYLOAD})
        body = resp.json()
        assert body["suggested_action"] is None
        assert body["policy_decision"] is None
    finally:
        _teardown()


def test_recent_deployment_plus_high_confidence_suggests_rollback(monkeypatch) -> None:
    import app.api.routes.alerts as alerts_mod
    from app.services.deployment_history import DeploymentEvent

    _patch_orchestrator(monkeypatch, _fake_session(confidence=0.9, confidence_level="HIGH"))

    fake_history = MagicMock()
    fake_history.find_near.return_value = [
        DeploymentEvent(
            sha="a" * 40, short_sha="aaaaaaa", author="octocat", message="ship it",
            committed_at=datetime(2026, 6, 15, 11, 0, tzinfo=timezone.utc),
            url="https://github.com/o/r/commit/aaa",
        )
    ]
    fake_history.__enter__ = lambda self: fake_history
    fake_history.__exit__ = lambda self, *a: None
    monkeypatch.setattr(alerts_mod, "DeploymentHistoryClient", lambda **kw: fake_history)

    client = _client()
    try:
        resp = client.post(
            "/alerts/webhook",
            json={
                "provider_payload": PROMETHEUS_PAYLOAD,
                "repo_owner": "octo", "repo_name": "checkout",
            },
        )
        body = resp.json()
        assert len(body["deployment_context"]) == 1
        assert body["deployment_context"][0]["short_sha"] == "aaaaaaa"
        assert body["suggested_action"]["action_type"] == "rollback_deployment"
        # The property that matters: state-changing still requires approval,
        # even though confidence is HIGH and a deployment was found.
        assert body["policy_decision"]["category"] == "state_changing"
        assert body["policy_decision"]["requires_approval"] is True
        assert body["policy_decision"]["auto_executable"] is False
    finally:
        _teardown()


def test_deployment_found_but_confidence_not_high_suggests_diagnostics_not_rollback(
    monkeypatch,
) -> None:
    import app.api.routes.alerts as alerts_mod
    from app.services.deployment_history import DeploymentEvent

    _patch_orchestrator(monkeypatch, _fake_session(confidence=0.5, confidence_level="MEDIUM"))

    fake_history = MagicMock()
    fake_history.find_near.return_value = [
        DeploymentEvent(
            sha="b" * 40, short_sha="bbbbbbb", author=None, message="tweak",
            committed_at=datetime(2026, 6, 15, 11, 30, tzinfo=timezone.utc), url="",
        )
    ]
    fake_history.__enter__ = lambda self: fake_history
    fake_history.__exit__ = lambda self, *a: None
    monkeypatch.setattr(alerts_mod, "DeploymentHistoryClient", lambda **kw: fake_history)

    client = _client()
    try:
        resp = client.post(
            "/alerts/webhook",
            json={
                "provider_payload": PROMETHEUS_PAYLOAD,
                "repo_owner": "octo", "repo_name": "checkout",
            },
        )
        body = resp.json()
        assert body["suggested_action"]["action_type"] == "gather_diagnostics"
        assert body["policy_decision"]["category"] == "read_only"
        assert body["policy_decision"]["requires_approval"] is False
    finally:
        _teardown()


def test_deployment_history_error_degrades_to_empty_context_not_a_500(monkeypatch) -> None:
    import app.api.routes.alerts as alerts_mod
    from app.services.deployment_history import DeploymentHistoryError

    _patch_orchestrator(monkeypatch, _fake_session())

    class RaisingHistory:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def find_near(self, *a, **kw):
            raise DeploymentHistoryError("repo not found")

    monkeypatch.setattr(alerts_mod, "DeploymentHistoryClient", lambda **kw: RaisingHistory())
    client = _client()
    try:
        resp = client.post(
            "/alerts/webhook",
            json={
                "provider_payload": PROMETHEUS_PAYLOAD,
                "repo_owner": "octo", "repo_name": "nonexistent",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["deployment_context"] == []
    finally:
        _teardown()


# ── Investigation-level failures ────────────────────────────────────────────


def test_investigation_construction_failure_returns_503(monkeypatch) -> None:
    import app.api.routes.alerts as alerts_mod

    def _raise(db):
        raise ValueError("OPENAI_API_KEY is required")

    monkeypatch.setattr(alerts_mod, "MultiAgentInvestigationOrchestrator", _raise)
    client = _client()
    try:
        resp = client.post("/alerts/webhook", json={"provider_payload": PROMETHEUS_PAYLOAD})
        assert resp.status_code == 503
    finally:
        _teardown()


def test_investigation_runtime_failure_returns_500_not_a_traceback(monkeypatch) -> None:
    import app.api.routes.alerts as alerts_mod

    fake = MagicMock()
    fake.investigate.side_effect = RuntimeError("boom")
    monkeypatch.setattr(alerts_mod, "MultiAgentInvestigationOrchestrator", lambda db: fake)
    client = _client()
    try:
        resp = client.post("/alerts/webhook", json={"provider_payload": PROMETHEUS_PAYLOAD})
        assert resp.status_code == 500
        assert "boom" not in resp.text
    finally:
        _teardown()


# ── Auth and rate limiting are wired (not bypassed by accident) ─────────────


def test_webhook_requires_authentication() -> None:
    app.dependency_overrides[get_db] = lambda: MagicMock()
    client = TestClient(app)
    try:
        resp = client.post("/alerts/webhook", json={"provider_payload": PROMETHEUS_PAYLOAD})
        assert resp.status_code == 401
    finally:
        _teardown()

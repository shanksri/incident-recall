"""Tests for the execution policy engine.

The one property that matters more than any other: confidence must never be
able to buy a state-changing action out of requiring approval. That is
asserted directly, not just implied by the other tests passing.
"""
from __future__ import annotations

import pytest

from app.services.action_policy import (
    ActionCategory,
    ActionPolicyEngine,
    PolicyDecision,
    ProposedAction,
)

engine = ActionPolicyEngine()


def _action(action_type: str, *, confidence: float | None = None) -> ProposedAction:
    return ProposedAction(
        action_type=action_type, description="d", target="t", confidence=confidence,
    )


# ── Read-only actions never require approval ────────────────────────────────


@pytest.mark.parametrize(
    "action_type",
    [
        "query_logs", "fetch_traces", "fetch_metrics", "search_incidents",
        "investigate", "fetch_deployment_history", "gather_diagnostics",
        "read_configuration", "generate_rca_draft",
    ],
)
def test_every_read_only_action_is_auto_executable(action_type: str) -> None:
    decision = engine.evaluate(_action(action_type))
    assert decision.category == ActionCategory.READ_ONLY
    assert decision.requires_approval is False
    assert decision.auto_executable is True


# ── State-changing actions always require approval, regardless of confidence ─


@pytest.mark.parametrize(
    "action_type",
    [
        "rollback_deployment", "restart_service", "scale_service",
        "apply_config_change", "apply_patch", "revert_commit",
        "delete_resource", "restart_pod",
    ],
)
def test_every_state_changing_action_requires_approval(action_type: str) -> None:
    decision = engine.evaluate(_action(action_type))
    assert decision.category == ActionCategory.STATE_CHANGING
    assert decision.requires_approval is True
    assert decision.auto_executable is False


@pytest.mark.parametrize("confidence", [0.0, 0.4, 0.8, 0.99, 1.0])
def test_confidence_never_overrides_the_state_changing_gate(confidence: float) -> None:
    """The property the whole module exists for."""
    decision = engine.evaluate(_action("rollback_deployment", confidence=confidence))
    assert decision.requires_approval is True
    assert decision.auto_executable is False


def test_a_high_confidence_misdiagnosis_is_still_gated() -> None:
    """The exact failure mode named in the module docstring."""
    decision = engine.evaluate(_action("rollback_deployment", confidence=0.99))
    assert decision.requires_approval is True


def test_reason_mentions_confidence_did_not_decide_the_outcome() -> None:
    decision = engine.evaluate(_action("restart_service", confidence=0.9))
    assert "0.90" in decision.reason
    assert "does not change this" in decision.reason


def test_reason_omits_confidence_value_when_none_supplied() -> None:
    decision = engine.evaluate(_action("restart_service"))
    assert "does not change this" not in decision.reason
    assert "regardless of confidence" in decision.reason


# ── Unknown actions fail closed ─────────────────────────────────────────────


def test_unknown_action_type_fails_closed_to_state_changing() -> None:
    decision = engine.evaluate(_action("launch_the_missiles"))
    assert decision.category == ActionCategory.STATE_CHANGING
    assert decision.requires_approval is True
    assert decision.auto_executable is False
    assert "not in the policy catalog" in decision.reason


def test_unknown_action_with_high_confidence_still_fails_closed() -> None:
    decision = engine.evaluate(_action("some_new_untested_action", confidence=1.0))
    assert decision.requires_approval is True


def test_typo_of_a_known_action_does_not_fuzzy_match() -> None:
    """A near-miss must not silently inherit the real action's category."""
    decision = engine.evaluate(_action("rollback_deploymnt"))
    assert decision.category == ActionCategory.STATE_CHANGING
    assert "not in the policy catalog" in decision.reason


# ── Matching is case/whitespace tolerant but exact otherwise ────────────────


@pytest.mark.parametrize(
    "raw", ["QUERY_LOGS", " query_logs ", "Query_Logs", "query_logs\n"],
)
def test_action_type_matching_is_case_and_whitespace_insensitive(raw: str) -> None:
    decision = engine.evaluate(_action(raw))
    assert decision.category == ActionCategory.READ_ONLY


# ── Decision record carries the input action through unmodified ────────────


def test_decision_carries_the_original_action() -> None:
    action = _action("rollback_deployment", confidence=0.7)
    decision = engine.evaluate(action)
    assert decision.action is action


def test_decision_and_action_are_immutable() -> None:
    action = _action("query_logs")
    decision = engine.evaluate(action)
    with pytest.raises(Exception):
        action.action_type = "restart_service"  # type: ignore[misc]
    with pytest.raises(Exception):
        decision.requires_approval = True  # type: ignore[misc]


def test_decision_is_the_documented_type() -> None:
    assert isinstance(engine.evaluate(_action("query_logs")), PolicyDecision)


# ── Engine holds no state across calls ──────────────────────────────────────


def test_engine_is_stateless_across_calls() -> None:
    first = engine.evaluate(_action("query_logs"))
    engine.evaluate(_action("rollback_deployment"))
    second = engine.evaluate(_action("query_logs"))
    assert first.category == second.category == ActionCategory.READ_ONLY

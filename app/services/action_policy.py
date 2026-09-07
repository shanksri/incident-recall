"""Execution policy engine — the guardrail between a diagnosis and an action.

# Why this exists

Everything upstream of this module (planner, hypothesis generator, evidence
evaluator, critic) produces a DIAGNOSIS: a root cause and a confidence score.
None of it takes an action, and none of it should be trusted to decide when
an action is safe — confidence is a statement about how sure the model is
that it identified the right problem, not a statement about how safe or
reversible a proposed fix is. Those are orthogonal axes. A model can be
completely confident and completely wrong, and the failure mode this module
exists to prevent is exactly that: a high-confidence misdiagnosis triggering
an automatic rollback, restart, or config change with no human in the loop.

# The rule, deliberately not tunable by confidence

``ActionPolicyEngine`` classifies a proposed action into exactly one of two
categories and applies exactly one rule per category:

  READ_ONLY       -> may proceed automatically. It cannot make anything
                      worse; the cost of being wrong is a wasted API call.
  STATE_CHANGING  -> ALWAYS requires human approval, regardless of
                      confidence. Not "requires approval below some
                      threshold" — there is no threshold. A rollback
                      proposed at confidence 0.99 gets exactly the same
                      gate as one proposed at 0.40.

This is the same design posture as every other rule-based component in this
codebase (``RuleBasedPlanner``, ``HeuristicCriticAgent``): deterministic,
explainable in one sentence, and cheap enough to have zero excuse not to run
on every single action. It is deliberately NOT machine-learned and
deliberately NOT influenced by the investigation's confidence score — the
one thing this gate must never do is let a sufficiently confident diagnosis
buy its way out of a human review.

# Fail-closed classification

An action type this engine has never seen is classified STATE_CHANGING, not
READ_ONLY. The failure mode of under-classifying an action is "an
unreviewed rollback runs automatically"; the failure mode of
over-classifying is "a harmless diagnostic query waits for a human to click
Approve." Those two failures are not symmetric, so the unknown case resolves
toward the safe one. See ``app.api.auth``'s fail-closed API-key check and
``RetrievalGateDecision``'s fail-OPEN posture for the same reasoning applied
in the opposite direction — there the failure mode of blocking a query is
mild, so an unavailable scorer fails open. The direction always follows the
blast radius of being wrong, never a blanket "always fail safe" rule.

# What this module deliberately does not do

- It does not execute anything. ``PolicyDecision`` is a recommendation
  record; wiring ``allowed=True`` to an actual rollback/restart call is a
  separate, out-of-scope integration this project does not implement.
- It does not learn or adapt the catalog from outcomes. The catalog below
  is authored by inspection, exactly like the planner's keyword lists —
  illustrative defaults, not validated against production incident data.
- It does not consider *how* state-changing an action is. A config flag
  flip and a full database rollback are both STATE_CHANGING and both
  require the same one thing: a human. Grading blast radius within that
  category is a real next step this module does not attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionCategory(str, Enum):
    READ_ONLY = "read_only"
    STATE_CHANGING = "state_changing"


# Authored by inspection, not tuned against production incident data — same
# status as RuleBasedPlanner's keyword lists (see planner_agent.py). Keys are
# lowercased, underscore-separated action-type identifiers a caller supplies;
# matching is exact, not fuzzy, so a typo fails closed to STATE_CHANGING
# rather than silently matching the wrong entry.
_CATALOG: dict[str, ActionCategory] = {
    # Diagnostics: cannot change system state, so nothing here needs review.
    "query_logs": ActionCategory.READ_ONLY,
    "fetch_traces": ActionCategory.READ_ONLY,
    "fetch_metrics": ActionCategory.READ_ONLY,
    "search_incidents": ActionCategory.READ_ONLY,
    "investigate": ActionCategory.READ_ONLY,
    "fetch_deployment_history": ActionCategory.READ_ONLY,
    "gather_diagnostics": ActionCategory.READ_ONLY,
    "read_configuration": ActionCategory.READ_ONLY,
    "generate_rca_draft": ActionCategory.READ_ONLY,
    # Remediation: changes what is actually running or configured.
    "rollback_deployment": ActionCategory.STATE_CHANGING,
    "restart_service": ActionCategory.STATE_CHANGING,
    "scale_service": ActionCategory.STATE_CHANGING,
    "apply_config_change": ActionCategory.STATE_CHANGING,
    "apply_patch": ActionCategory.STATE_CHANGING,
    "revert_commit": ActionCategory.STATE_CHANGING,
    "delete_resource": ActionCategory.STATE_CHANGING,
    "restart_pod": ActionCategory.STATE_CHANGING,
}


@dataclass(frozen=True)
class ProposedAction:
    """One candidate action a caller is asking the policy engine to review.

    ``confidence`` is carried through only so ``PolicyDecision.reason`` can
    state plainly that it was NOT what decided the category — it is never
    read by the classification logic itself.
    """

    action_type: str
    description: str
    target: str
    confidence: float | None = None


@dataclass(frozen=True)
class PolicyDecision:
    """The engine's verdict on one ``ProposedAction``. Immutable, same
    reasoning-record posture as ``CritiqueResult`` and ``RetrievalGateDecision``
    — a caller should never have to re-derive *why* a decision came out the
    way it did.
    """

    action: ProposedAction
    category: ActionCategory
    requires_approval: bool
    auto_executable: bool
    reason: str


class ActionPolicyEngine:
    """Deterministic, catalog-based policy gate. Makes zero LLM calls, holds
    no state between calls, and is safe to construct once and reuse for
    every proposed action in a process.
    """

    def evaluate(self, action: ProposedAction) -> PolicyDecision:
        category = _CATALOG.get(action.action_type.strip().lower())

        if category is None:
            return PolicyDecision(
                action=action,
                category=ActionCategory.STATE_CHANGING,
                requires_approval=True,
                auto_executable=False,
                reason=(
                    f"action type {action.action_type!r} is not in the policy catalog; "
                    "unrecognized actions fail closed to state-changing rather than "
                    "being assumed safe"
                ),
            )

        if category is ActionCategory.READ_ONLY:
            return PolicyDecision(
                action=action,
                category=category,
                requires_approval=False,
                auto_executable=True,
                reason=f"{action.action_type!r} is read-only and cannot change system state",
            )

        confidence_note = (
            f" (confidence was {action.confidence:.2f}, which does not change this)"
            if action.confidence is not None
            else ""
        )
        return PolicyDecision(
            action=action,
            category=category,
            requires_approval=True,
            auto_executable=False,
            reason=(
                f"{action.action_type!r} is state-changing and always requires human "
                f"approval regardless of confidence{confidence_note}"
            ),
        )

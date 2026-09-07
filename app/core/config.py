from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Phase 24A: the development-only DB default and the .env.example placeholder
# API key. Named here (not just inlined in the field defaults) so the
# production-config validator below can check against the exact same values
# without duplicating the literals.
_DEV_DATABASE_URL_DEFAULT = "postgresql+psycopg://postgres:postgres@localhost:5432/incidents"
_PLACEHOLDER_API_KEY = "changeme-generate-a-real-secret"
_LOCALHOST_MARKERS = ("localhost", "127.0.0.1")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Phase 24A: distinguishes development from production so a handful of
    # settings below (docs/OpenAPI exposure, the production-config validator)
    # can be gated by it. Does not itself change any retrieval/generation/
    # routing/agent behavior — nothing else in the codebase reads this field.
    environment: Literal["development", "production"] = "development"

    database_url: str = Field(default=_DEV_DATABASE_URL_DEFAULT)
    github_token: str | None = None
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = Field(default=384, gt=0)
    relevance_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    evidence_relevance_enabled: bool = Field(
        default=False,
        description=(
            "Score evidence with the cross-encoder relevance model against the "
            "ORIGINAL PROBLEM instead of thresholding cosine similarity to the "
            "hypothesis's own keywords. Off by default so the change is A/B "
            "measurable against the existing probe baseline before it becomes "
            "the default (same rollout posture as SEARCH_ROUTING_ENABLED)."
        ),
    )
    retrieval_gate_enabled: bool = Field(
        default=False,
        description=(
            "Reject an investigation up front when no retrieved incident is "
            "genuinely relevant to the query, scored by the cross-encoder "
            "rather than by top-1 cosine. Measured on 56 gold queries this "
            "cuts the hard-negative false-accept rate from 0.900 to 0.150 "
            "while keeping recall at 0.885. Off by default pending a "
            "held-out validation split."
        ),
    )
    retrieval_gate_threshold: float = Field(
        default=2.0,
        description=(
            "Cross-encoder logit at or above which retrieval is considered to "
            "have found something relevant. 2.0 is the knee of the measured "
            "recall/FPR curve, read off that curve rather than fitted on a "
            "held-out split. Raw logits, NOT the 0.40 cosine scale."
        ),
    )
    relevance_threshold: float = Field(
        default=0.0,
        description=(
            "Cross-encoder logit at or above which evidence counts as "
            "supporting. These are raw logits (~-11..+11), NOT probabilities "
            "and NOT comparable to LOW_CONFIDENCE_THRESHOLD's 0.40 cosine "
            "scale. 0.0 is the model's own natural boundary, not a value "
            "calibrated against any gold dataset."
        ),
    )
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = Field(
        default=None,
        description=(
            "Anthropic API credits, billed separately from a Claude Pro "
            "subscription -- Pro does not grant API access. Used only by the "
            "evaluation judge, so that answers written by OPENAI_MODEL are "
            "not also graded by it. Unset means the Anthropic judge cannot "
            "be constructed; nothing else in the app requires it."
        ),
    )
    anthropic_model: str = "claude-haiku-4-5"
    gemini_api_key: str | None = Field(
        default=None,
        description=(
            "Google Gemini key for the evaluation judge, an alternative to "
            "ANTHROPIC_API_KEY. Either satisfies the requirement that answers "
            "written by OPENAI_MODEL are not also graded by the same model "
            "family. Optional; nothing outside evaluation reads it."
        ),
    )
    gemini_model: str = "gemini-3.1-flash-lite"
    llm_critic_enabled: bool = Field(
        default=False,
        description=(
            "Use the LLM-backed CriticAgent (GEMINI_MODEL) instead of the "
            "arithmetic HeuristicCriticAgent. This makes the pipeline "
            "genuinely two-model: OPENAI_MODEL proposes root causes, a "
            "different family challenges them, so the critic is not grading "
            "its own work. Costs one extra LLM call per iteration and falls "
            "back to the heuristic critic on any failure, which the free "
            "tier's 20-requests/day/model quota makes routine. Off by "
            "default."
        ),
    )
    log_level: str = "INFO"
    api_key: str | None = Field(
        default=None,
        description=(
            "Shared secret for the platform's Bearer API-key authentication "
            "(Phase 23B) — required on every /ingestion, /search, /agent, "
            "/incidents, and /evaluation request as 'Authorization: Bearer "
            "<API_KEY>'. Unset means no key can ever match, so every "
            "protected request is rejected (fail-closed), not left open."
        ),
    )
    search_routing_enabled: bool = Field(
        default=False,
        description=(
            "Opt-in switch for adaptive retrieval routing (Phase 18A-18C) in "
            "production /search and the investigation orchestrator. False "
            "preserves dense-only behavior exactly "
            "(see RoutedSearchConfig.routing_enabled)."
        ),
    )

    # ── Phase 23C: endpoint-aware rate limiting ─────────────────────────────
    #
    # All limits are "requests per 60-second fixed window" per caller
    # identity (see app/api/rate_limit.py). Values below the Search/Agent/
    # Evaluation-*/Interactive-Evaluation line are the Phase 23C spec's own
    # suggested defaults. `rate_limit_incidents_per_minute` and
    # `rate_limit_ingestion_per_minute` are NOT in the spec's suggested
    # list — added because leaving those two routers completely unlimited
    # would leave real abuse vectors unaddressed (ingestion triggers
    # external HTTP calls; see the Phase 23 production-readiness review's
    # SSRF/cost-exhaustion findings), which would contradict this phase's
    # own stated objective. `rate_limit_evaluation_runs_per_minute` covers
    # the read-only GET /evaluation/runs*, /evaluation/stats views, also
    # not named in the spec, for the same reason.
    rate_limit_enabled: bool = Field(
        default=True,
        description="Global kill switch — False disables all rate limiting (health is always unlimited regardless).",
    )
    rate_limit_search_per_minute: int = Field(default=100, gt=0)
    rate_limit_agent_per_minute: int = Field(default=20, gt=0)
    rate_limit_evaluation_query_per_minute: int = Field(default=20, gt=0)
    rate_limit_evaluation_retrieval_per_minute: int = Field(default=5, gt=0)
    rate_limit_evaluation_reasoning_per_minute: int = Field(default=5, gt=0)
    rate_limit_evaluation_full_per_minute: int = Field(default=2, gt=0)
    rate_limit_interactive_evaluation_per_minute: int = Field(default=20, gt=0)
    rate_limit_incidents_per_minute: int = Field(default=100, gt=0)
    rate_limit_ingestion_per_minute: int = Field(default=10, gt=0)
    rate_limit_evaluation_runs_per_minute: int = Field(default=60, gt=0)
    rate_limit_alerts_per_minute: int = Field(
        default=20, gt=0,
        description=(
            "POST /alerts/webhook triggers a full investigation (3-5 LLM calls), "
            "same cost class as /agent/investigate, so it shares that default."
        ),
    )

    # ── Phase 24A: production-only configuration validation ────────────────
    #
    # Development/test behavior is completely unaffected: this validator is
    # a no-op unless ENVIRONMENT=production is explicitly set. It fails at
    # Settings construction time (process startup), not at first request, so
    # a misconfigured production deploy never silently serves traffic with a
    # weak/default DB or an unset API key.
    @model_validator(mode="after")
    def _validate_production_config(self) -> Settings:
        if self.environment != "production":
            return self

        if not self.api_key or self.api_key == _PLACEHOLDER_API_KEY:
            raise ValueError(
                "ENVIRONMENT=production requires API_KEY to be set to a real "
                "secret (it is currently unset or still the .env.example "
                "placeholder). Generate one with e.g. `openssl rand -hex 32`."
            )

        if any(marker in self.database_url for marker in _LOCALHOST_MARKERS):
            raise ValueError(
                "ENVIRONMENT=production requires DATABASE_URL to point at a "
                "real database, not localhost/127.0.0.1 (the development "
                "default)."
            )

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

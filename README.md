# Enterprise Incident Intelligence Platform

An agentic RAG system for software incident investigation, built together with the evaluation
platform needed to find out whether it actually works — and then measured hard enough to establish
where it doesn't.

**What it does:** ingests real incidents from GitHub Issues and Jira, retrieves similar past ones
via hybrid semantic + lexical search, and runs a four-agent (planner → hypothesis generation →
critic → orchestrator) loop that surfaces the most relevant prior incidents, the evidence linking
them, and what resolved them — abstaining rather than answering when retrieval evidence is weak.

**What makes it worth reading:** most agentic-RAG projects stop at a working demo. Here the
evaluation platform is 36 of the ~89 Python modules — roughly 40% of the codebase — covering
versioned gold datasets, retrieval metrics, LLM-as-judge with judge *validation*, and RAGAS-style
grounding metrics. The interesting results are the negative ones:

- The system **fabricated** a confident root cause for a query with nothing matching it in the
  corpus. Faithfulness scored it 0.0 after the fact; the system itself never noticed. Fixed with a
  pre-LLM confidence gate.
- Hybrid retrieval's rank-fusion **silently dropped a correct result** that dense search had ranked
  8th, because the fusion rewards documents both retrievers weakly agree on. Fixed with a dense
  floor; hybrid now strictly dominates dense on the benchmark.
- The confidence signal used for abstention **does not reliably separate** a genuine match from a
  topically-adjacent wrong one. Measured against 20 purpose-built hard negatives, then re-measured
  against the full investigation path, which narrowed the gap to one specific band. A candidate fix
  (a targeted LLM relevance check) measures well but **is not shipped**, because one promising run
  isn't a validated signal — see [Evaluation findings](#evaluation-findings).

25 phases of incremental, tested development (1,606 tests), plus two production-hardening passes.
See [Project status](#project-status) for what's done, what's open, and what's deliberately not
claimed.

## What it does, end to end

```
GitHub Issues ──┐
Jira            ├─▶ Ingestion ─▶ Normalize ─▶ Deduplicate ─▶ Embed ─▶ pgvector
                │      (source-agnostic canonical incident shape)
                ▼
        Semantic + Hybrid (Dense/BM25/RRF) Retrieval, adaptively routed per query
                │
                ▼
   Multi-Agent Investigation:  Planner → Hypothesis Generation → Critic → Orchestrator
                │                (iterative: critic can send it back for more evidence)
                ▼
     Evidence-backed report — relevant prior incidents, supporting/contradicting
     evidence, rejected hypotheses — or an explicit "insufficient evidence"
     abstention when no hypothesis clears the confidence floor

                                    ▲
                                    │  measured by
                                    │
   Evaluation Platform: Retrieval metrics (Recall/MRR/NDCG) · Reasoning accuracy ·
   LLM-as-judge · BERTScore + RAGAS grounding (Faithfulness/Relevancy/Precision/Recall) ·
   evaluator-stability tracking · cost/skip diagnostics · health dashboards · trends
```

## Measurement first

The organizing principle: **retrieval and reasoning quality are measured, not asserted.** That
wasn't a philosophical choice — it was forced. An early retrieval baseline was invalidated when the
corpus grew from ~400 to ~8,000 incidents and the numbers stopped being comparable, and the first
gold set rotted entirely because it keyed on database UUIDs that re-ingestion regenerates. So the
roadmap deliberately put the measurement platform *before* the next retrieval algorithm.

- **A real evaluation platform** — versioned gold datasets with corpus fingerprinting (a run records
  what data it ran against), Recall@K/MRR/NDCG, an LLM-as-judge framework that also **validates the
  judge** (inter-judge agreement, calibration, a trustworthiness verdict — an unvalidated judge is
  just another opinion), BERTScore + RAGAS-style grounding metrics (Faithfulness, Answer Relevancy,
  Context Precision/Recall/Entity Recall) with cost tiers and repeated-run variance tracking, and a
  diagnostics layer surfacing outliers, cost, and skip reasons.
  See [`15`](docs/architecture/15_evaluation_framework.md),
  [`20`](docs/architecture/20_reasoning_evaluation_and_judges.md),
  [`21`](docs/architecture/21_evaluation_platform_productionization.md).
- **Changes ship only when measured, and stay off when unproven** — adaptive Dense/BM25/Hybrid
  routing is fully built, benchmarked, and wired into production, but ships **disabled by default**
  because its thresholds were never tuned against real traffic. Reranking is opt-in for the same
  reason: measurement showed it near-neutral and occasionally harmful. Hypothesis generation was cut
  from 5 candidates to 2 (escalating to 4 only on weak confidence) because the data showed ranks 3
  and 5 added zero recall at real cost.
  See [`18`](docs/architecture/18_adaptive_routing_and_hybrid_confidence.md).
- **A staged investigation loop**, not a single prompt — `PlannerAgent` sets strategy, hypothesis
  generation proposes and evidence-checks root causes, `CriticAgent` reviews and can force another
  iteration, an orchestrator coordinates with bounded stopping conditions. Only hypothesis
  generation calls an LLM; the planner, evidence evaluator, critic and orchestrator are rule-based,
  so they're deterministic, cheap, and testable without an API key. An investigation costs **3-5 LLM
  calls**: query expansion and reranking once each during retrieval, plus one per iteration
  (`DEFAULT_MAX_ITERATIONS = 3`). Calling this "multi-agent" is a claim about component structure —
  separable roles behind swappable interfaces — not about multiple models reasoning independently.
  Opt-in via `LLM_CRITIC_ENABLED`, the critic becomes a second reasoning model on a *different
  vendor* (`GEMINI_MODEL`), so the model proposing root causes is not the one grading them — the
  proposer/challenger split that makes "multi-agent" mean something here.
  See [Evaluation findings](#evaluation-findings).
  See [`19`](docs/architecture/19_multi_agent_investigation.md).
- **Two production-hardening passes** — input validation, graceful degradation (typed errors, never
  raw tracebacks or leaked secrets), injection/path-traversal testing, load testing, environment-aware
  configuration with fail-fast production validation, and a failure-handling audit.
  See [Phase 23](#production-hardening-phase-23) and [Phase 24](#production-hardening-phase-24).

**Scope honesty:** the corpus is public open-source bug trackers, not production incident data from
a running system — structurally similar (title, description, discussion thread, resolution) and rich
in the hard cases that matter here (vocabulary mismatch, near-duplicates, topical neighbors), but not
the same thing. The ingestion layer is source-agnostic by design, so adding postmortems or an
incident-management export is a new collector and normalizer with nothing downstream changing.

## Tech stack

FastAPI · SQLAlchemy 2 + Alembic · PostgreSQL + pgvector + `pg_trgm` · SentenceTransformers ·
OpenAI API · Gemini API · MCP · Docker / docker-compose · pytest (1,606 tests) · Python 3.12

## Documentation map

This README is the entry point. The full engineering reference — written for someone joining the
project cold — lives in [`docs/README.md`](docs/README.md) and 23 numbered architecture docs
(`docs/architecture/01`–`23`), covering everything from ingestion and normalization through
retrieval, the multi-agent investigation framework, the evaluation platform's REST API, and Phase 23
production hardening. Start there for depth; this file is the map. (Phase 24's configuration and
error-handling hardening, below, postdates that numbered doc series and is documented only here and
inline in the code it touches.)

## Project structure

```
app/
  api/routes/       FastAPI routers: health, incidents, ingestion, search, agent, alerts, evaluation(+interactive)
  api/schemas.py     Pydantic request/response models (validated: length/format/UUID bounds)
  api/validation.py  Shared identifier/UUID validators (Phase 23)
  core/              Settings (pydantic-settings) and logging configuration
  db/                SQLAlchemy models + session management
  ingestion/         Source collectors (GitHub, Jira) and normalization
  services/          Embedding, LLM, relevance scoring, retrieval (dense/BM25/hybrid/routed), the 4 investigation
                     agents, alert normalization, deployment history, execution policy engine
  evaluation/        Gold datasets, harnesses, metrics, judges, experiment tracking, diagnostics, judge backends — 39 modules
  mcp/               MCP server (Phase 25) exposing investigate/search/policy/deployment-history as tools
alembic/             Database migrations (creates the `vector` and `pg_trgm` extensions)
docs/                Full architecture documentation (23 numbered docs + index)
scripts/             Benchmark/evaluation CLI scripts, load_test.py, profile_performance.py
tests/               1,606 tests: tests/unit, tests/api, tests/eval
Dockerfile, docker-compose.yml   Multi-stage build, non-root user, healthcheck (Phase 23)
```

## Quick start (Docker)

```bash
docker compose up --build
```

This starts PostgreSQL (with pgvector) and the API, running migrations automatically on
container start. API docs: `http://localhost:8000/docs`. Liveness: `GET /health`. Readiness
(checks DB connectivity): `GET /health/ready`.

Set `OPENAI_API_KEY` (required for the investigation agents and LLM-backed retrieval features),
`API_KEY` (required — see [Authentication](#authentication) below), and optionally `GITHUB_TOKEN`
(avoids GitHub API rate limits) as environment variables before starting, or in a `.env` file at
the repo root — see `.env.example`.

## Manual setup (no Docker)

For a from-scratch local environment (Python venv, PostgreSQL via `docker run`, Alembic, running
the API and tests directly) see [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md) — a step-by-step
guide with verification commands for every stage.

## Configuration

All settings are read from environment variables (or a `.env` file) via `app/core/config.py`:

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` additionally requires `API_KEY` to be a real secret and `DATABASE_URL` to not point at localhost — `Settings()` raises at startup otherwise (Phase 24A). `production` also disables `/docs`, `/redoc`, `/openapi.json` |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/incidents` | Required in production (must not be the localhost default there) |
| `GITHUB_TOKEN` | — | Optional; avoids GitHub API rate limits |
| `OPENAI_API_KEY` | — | Required for `/agent/*` and any LLM-backed evaluation |
| `OPENAI_MODEL` | `gpt-4o-mini` | |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Must match the DB's vector dimension |
| `EMBEDDING_DIMENSIONS` | `384` | Matches the current migration's `VECTOR(384)` column |
| `LOG_LEVEL` | `INFO` | |
| `SEARCH_ROUTING_ENABLED` | `false` | Opt-in for adaptive Dense/BM25/Hybrid routing; `false` preserves dense-only behavior |
| `RELEVANCE_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Local cross-encoder; runs beside the embedding model, no API key |
| `RETRIEVAL_GATE_ENABLED` | `false` | Opt-in. Abstains before any LLM call when nothing retrieved is relevant to the query — see [Evaluation findings](#evaluation-findings). Fails open if the scorer errors |
| `RETRIEVAL_GATE_THRESHOLD` | `2.0` | Cross-encoder **logit** (~-11..+11), not the 0.40 cosine scale. Knee of the measured recall/FPR curve |
| `EVIDENCE_RELEVANCE_ENABLED` | `false` | Opt-in. Splits supporting/contradicting evidence by relevance to the problem rather than cosine to the hypothesis's own keywords. Ships off: as built it measures evidence drift rather than fixing it |
| `RELEVANCE_THRESHOLD` | `0.0` | Logit boundary for that split. Not calibrated against any dataset, unlike `RETRIEVAL_GATE_THRESHOLD` |
| `ANTHROPIC_API_KEY` | — | Optional. Only the evaluation judge reads it, so answers written by `OPENAI_MODEL` aren't graded by the same model family. Billed separately from a Claude Pro subscription |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | |
| `GEMINI_API_KEY` | — | Optional alternative to `ANTHROPIC_API_KEY` for the same purpose. Free tier is 20 requests/day **per model**, which is below one full 56-query gold-set pass |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | |
| `LLM_CRITIC_ENABLED` | `false` | Opt-in. Replaces the arithmetic critic with a reasoning one on a different model family, so the proposer isn't grading its own work. One extra LLM call per iteration; falls back to the heuristic critic on any failure and labels the verdict when it does |
| `API_KEY` | — | **Required.** Shared secret for Bearer auth (Phase 23B) — see [Authentication](#authentication) |
| `RATE_LIMIT_ENABLED` | `true` | Global kill switch for rate limiting (Phase 23C) — see [Rate limiting](#rate-limiting) |
| `RATE_LIMIT_SEARCH_PER_MINUTE` | `100` | `/search/incidents`, `/search/debug` |
| `RATE_LIMIT_AGENT_PER_MINUTE` | `20` | `/agent/investigate` |
| `RATE_LIMIT_EVALUATION_QUERY_PER_MINUTE` | `20` | `POST /evaluation/query` |
| `RATE_LIMIT_EVALUATION_RETRIEVAL_PER_MINUTE` | `5` | `POST /evaluation/retrieval` |
| `RATE_LIMIT_EVALUATION_REASONING_PER_MINUTE` | `5` | `POST /evaluation/reasoning` |
| `RATE_LIMIT_EVALUATION_FULL_PER_MINUTE` | `2` | `POST /evaluation/full` (the most expensive single call in the API) |
| `RATE_LIMIT_INTERACTIVE_EVALUATION_PER_MINUTE` | `20` | `preview`/`by-title`/`{session_id}` routes |
| `RATE_LIMIT_INCIDENTS_PER_MINUTE` | `100` | Not in the original spec's suggested defaults — added so this router isn't left unlimited |
| `RATE_LIMIT_INGESTION_PER_MINUTE` | `10` | Not in the original spec's suggested defaults — ingestion triggers external HTTP calls, the platform's most abuse-prone surface |
| `RATE_LIMIT_EVALUATION_RUNS_PER_MINUTE` | `60` | Not in the original spec's suggested defaults — covers the read-only `GET /evaluation/runs*`, `/stats` views |
| `RATE_LIMIT_ALERTS_PER_MINUTE` | `20` | `POST /alerts/webhook` — same cost class as `/agent/investigate` (it calls the same orchestrator internally), so it shares that default |

## Authentication

Every business endpoint (`/incidents`, `/ingestion`, `/search`, `/agent`, `/evaluation`) requires a
Bearer API key. `/health`, `/health/ready`, and the docs routes (`/docs`, `/redoc`,
`/openapi.json`) are intentionally left open — they're liveness/readiness probes and API
documentation, not business data.

This is deliberately **not** a user-management system — no accounts, passwords, sessions, JWTs,
OAuth, refresh tokens, roles, or a login endpoint. The platform is meant to run as an internal
service, behind an API gateway or inside a trusted network, so a single shared secret compared on
every request is the right amount of mechanism for that deployment model — see
`app/api/auth.py`'s module docstring for the fuller rationale.

**Making a request:**

```bash
curl -H "Authorization: Bearer $API_KEY" http://localhost:8000/incidents
```

Every failure mode — missing header, malformed header, wrong key — returns the same `401` with the
same generic `{"detail": "Not authenticated."}` body and a `WWW-Authenticate: Bearer` header, so a
response never reveals which part of the request was wrong.

**Using Swagger:** open `http://localhost:8000/docs`, click the **Authorize** button (top right),
paste the raw key (no `Bearer` prefix — the UI adds it), and click **Authorize**. Every subsequent
"Try it out" request against a protected endpoint then carries the header automatically; public
endpoints show no lock icon and need no authorization.

**Configuration:** set `API_KEY` in `.env` (see `.env.example`) or as an environment variable — for
example, generate one with `openssl rand -hex 32`. There is no default: if `API_KEY` is unset, every
protected request is rejected (fail-closed), never silently allowed through.

The dependency (`app/api/auth.py`'s `require_api_key`) is centralized and applied once per
router — via each `APIRouter(dependencies=[Depends(require_api_key)])` — not repeated inside
individual endpoints, so a new route under a protected router is authenticated automatically with
no extra code.

## Rate limiting

Every business endpoint has a per-minute request limit, sized to the endpoint's actual cost —
cheap reads (`/incidents`, `/search`) allow far more traffic than expensive LLM-backed calls
(`/evaluation/full`, which can run retrieval + reasoning + judging + generation in one request,
allows only 2/minute). `/health` and `/health/ready` are always unlimited. Limits are per **caller
identity** (the presented Bearer token, or client IP as a fallback) and per **endpoint group** —
exhausting one endpoint's quota never affects another's, and different callers never share a quota.

**On exceeding a limit:**

```json
HTTP/1.1 429 Too Many Requests
Retry-After: 37
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1735689660

{"detail": "Rate limit exceeded for 'agent': 20 requests per 60 seconds. Retry after 37 seconds."}
```

Every response from a rate-limited endpoint — successful or not — carries `X-RateLimit-Limit`,
`X-RateLimit-Remaining`, and `X-RateLimit-Reset` so a well-behaved client can back off before
hitting the limit, not just after.

**Implementation:** a fixed 60-second window, counted in-process (no Redis — the platform runs
single-process today; see `app/api/rate_limit.py`'s module docstring for why this is a deliberate,
documented tradeoff). The counting logic sits behind a `RateLimitBackend` abstraction so a
distributed backend could replace the in-memory one later without touching any route. Like auth,
it's wired once per router/route via `dependencies=[Depends(<group>_rate_limit)]` — never
duplicated inside an endpoint body.

**Configuration:** every limit is a `Settings` field (see the Configuration table above), overridable
via environment variable, e.g. `RATE_LIMIT_SEARCH_PER_MINUTE=200`. Set `RATE_LIMIT_ENABLED=false` to
disable rate limiting entirely (health remains unaffected either way).

**In Swagger:** every protected route documents a `429` response (see the **Responses** section of
each endpoint at `/docs`) alongside its usual success responses.

## API surface

22 endpoints across 7 routers (plus FastAPI's auto-generated `/docs`, `/redoc`, `/openapi.json`).
Reduced from 27 in Phase 23A by removing duplicate/legacy routes — one canonical endpoint per
business capability; see [Phase 23A: API surface consolidation](#phase-23a-api-surface-consolidation)
below.

**Health** — `GET /health` (liveness, unconditional), `GET /health/ready` (readiness, checks DB —
returns 503 when unreachable)

**Incidents** — `GET /incidents` (paginated list), `GET /incidents/{incident_id}` (UUID-validated)

**Ingestion** — `POST /ingestion/github`, `POST /ingestion/jira` (both return 502 on upstream
failure rather than crashing)

**Search** — `POST /search/incidents`, `POST /search/debug` — two genuinely distinct capabilities,
not duplicates: `/incidents` is plain filtered retrieval (arbitrary limit, full incident objects,
no LLM); `/debug` is the LLM-expanded + LLM-reranked variant (fixed at 5 results, lightweight
response shape, echoes the resolved filters) — currently the *only* way to invoke query expansion
and reranking over the API, despite the "debug" name.

**Agent** — `POST /agent/investigate` — the single canonical investigation endpoint, backed by the
full planner/hypothesis/critic/orchestrator loop. (Phase 23A retired `/investigate-advanced` and
the separate `/investigate-orchestrated` path — three routes for one business capability at three
generations of sophistication became one.)

**Alerts** (Phase 25) — `POST /alerts/webhook` — accepts a Prometheus Alertmanager or PagerDuty
webhook payload as-is, normalizes it, and runs the exact same orchestrator `/agent/investigate`
calls. See [Alert-triggered investigation, deployment history, and an execution policy
engine](#alert-triggered-investigation-deployment-history-and-an-execution-policy-engine) below.

**Evaluation** (12 endpoints) — `POST /evaluation/query|retrieval|reasoning|full`,
`GET /evaluation/runs`, `/runs/latest`, `/runs/{run_id}` (now includes failed-query/
failed-reasoning/judge-disagreement views, plus an opt-in `?include_diagnostics=true` diagnostics
dashboard — Phase 23A folded four separate GET routes into this one), `/stats`, plus a
human-friendly interactive flow: `POST /evaluation/query/preview`, `/by-title`,
`/{session_id}/evaluate`, `GET /evaluation/query/{session_id}` — full detail in
[`docs/architecture/22`](docs/architecture/22_evaluation_api.md).

## Phase 23A: API surface consolidation

A dedicated refactoring pass, on top of the feature-complete, production-hardened platform: same
capabilities, fewer routes to secure/document/maintain. Reviewed every endpoint group; consolidated
only genuine duplication.

- **Agent (3 → 1):** `/investigate`, `/investigate-advanced`, `/investigate-orchestrated` were three
  successive implementations of "investigate this problem and report a root cause," not three
  capabilities — the orchestrated one was already documented as canonical. Removed the other two
  routes; `POST /agent/investigate` now serves the orchestrator directly. The underlying single-shot
  agent classes (`InvestigationAgent`, `AdvancedInvestigationAgent`) are untouched and still
  independently unit-tested — only their HTTP routes were retired.
- **Evaluation run views (4 → 0 extra routes):** `/runs/{id}/failed-queries`, `/failed-reasoning`,
  `/judge-disagreements`, `/diagnostics` each did nothing but load the same run and return one
  filtered view of it. Folded into `GET /runs/{run_id}` — the three failure/disagreement lists are
  free (precomputed at save time, always included); diagnostics is real computation, so it stays
  opt-in via `?include_diagnostics=true`.
- **Search — reviewed, kept both.** `/search/incidents` and `/search/debug` looked redundant by
  name but aren't: different retrieval behavior (plain vs. LLM-expanded+reranked), different
  response shapes. Consolidating would have meant either losing a capability or growing
  `/search/incidents`'s contract — out of scope for a routes-only refactor.
- **Evaluation benchmarks and the interactive query workflow — reviewed, kept as-is.**
  `/query`, `/retrieval`, `/reasoning`, `/full` have different response shapes and scopes (not
  provably interchangeable without touching evaluation logic, which this phase didn't). The
  interactive `preview` → `evaluate` flow and the one-shot `by-title` endpoint serve two different
  callers — a human reviewing results before committing vs. a caller that already knows the answer
  — not one workflow expressed three ways.

## Running tests

```bash
python -m pytest              # full suite — 1,606 tests
python -m pytest tests/unit    # unit tests only (no HTTP layer)
python -m pytest tests/api     # FastAPI route tests (TestClient, no real DB/LLM)
python -m ruff check .         # lint
```

Every test runs against fakes/mocks for the database, LLM, and embedding backends — no live
Postgres or OpenAI credentials are needed. Two settings must be *present* though not valid, because
`LLMService` and Bearer auth both fail closed at construction on an empty value; CI supplies dummy
values for `OPENAI_API_KEY` and `API_KEY`, and a local `.env` covers it otherwise.

**Latest verified result: 1,606 passed, 0 failures.**

The long-standing failure in
`tests/api/test_production_hardening.py::test_evaluation_run_id_with_dots_only_rejected` is fixed.
It had been recorded here as an environment artifact — "depends on a local file this environment's
history doesn't have" — and that diagnosis was wrong. `validate_safe_identifier`'s allow-list
permits `.` and `-`, so `..`, `...`, `....`, `.` and `-` all passed validation, and
`history_root / ".."` resolves one directory *above* the intended root. The 500 was the visible
symptom; where the target file happens to exist, the endpoint would have served data from outside
the run directory instead of erroring. The validator now additionally requires at least one letter
or digit, which rejects every all-punctuation identifier at once. Regression tests:
`tests/unit/test_api_validation.py`.

## Load testing and performance profiling

Added in the production-hardening pass (Phase 23):

```bash
python scripts/load_test.py         # 10/50/100 concurrent users; latency/throughput/error rate
python scripts/profile_performance.py  # retrieval, routing, generation, evaluation, orchestration
```

Both scripts run in-process against the real FastAPI app (via `httpx.ASGITransport`, no server
process needed) and write JSON reports to `.benchmarks/phase23_load_test/` and
`.benchmarks/phase23_performance/`. They measure real platform overhead (routing, validation,
serialization) — see each script's own docstring for what they do and don't cover without a live
database/LLM backend.

## Production hardening (Phase 23)

A dedicated validation-and-hardening pass, on top of the feature-complete platform:

- **Input validation** — every endpoint bounds string length, list size, and format (UUIDs
  validated before hitting the database; malformed input returns 422, not a silent 404 or a crash).
- **Graceful degradation** — a platform-wide exception handler converts unhandled failures (DB
  connection drop, LLM timeout, malformed LLM JSON response) into clean typed errors instead of
  leaking stack traces or internal details (connection strings, API key fragments) to the client.
  Typed errors: `LLMResponseError`, `EmbeddingServiceError`; DB failures → 503; upstream
  ingestion failures → 502.
- **Security testing** — SQL-injection-shaped and prompt-injection-shaped input verified to pass
  through as inert text (no server-side string concatenation); path-traversal attempts on
  filesystem-backed identifiers (`run_id`) rejected both at the API layer and via defense-in-depth
  containment checks in the repository layer itself.
- **Resilience** — corrupted evaluation-run files on disk are skipped (logged, not crashed);
  DB/LLM/embedding client timeouts configured (previously unbounded, could hang indefinitely).
- **Deployment readiness** — `Dockerfile` (multi-stage, non-root user, healthcheck) and
  `docker-compose.yml` added (the README referenced them before they existed); a `/health/ready`
  probe and app lifespan (startup DB check, clean shutdown) added.

1,200+ tests (at the time of that phase) cover empty/malformed/oversized/unicode input, invalid
UUIDs, missing resources, simulated DB/LLM/embedding/upstream failures, and corrupted evaluation
data.

## Production hardening (Phase 24)

A second hardening pass, split into two sub-phases, focused on configuration correctness and
failure-handling gaps that only surface under real operating conditions rather than input
validation:

**Phase 24A — configuration & environment:**
- An explicit `ENVIRONMENT` setting (`development`/`production`, see the Configuration table above).
  Development behavior is completely unchanged; `production` fails fast at process startup — not at
  first request — if `API_KEY` is unset or still the placeholder value, or if `DATABASE_URL` still
  points at localhost.
- `Field(gt=0)` validation on every rate-limit setting and `EMBEDDING_DIMENSIONS`, so an invalid
  value (e.g. `0` or negative from a typo'd environment variable) is rejected at startup instead of
  producing undefined behavior at runtime.
- `/docs`, `/redoc`, and `/openapi.json` (public and unauthenticated by FastAPI's own default) are
  disabled outright when `ENVIRONMENT=production`.

**Phase 24B — application reliability & error handling:** an audit-then-fix pass across API error
handling, database failures, LLM/embedding failures, the multi-agent investigation path, evaluation
failures, and resource lifecycle. Most of the platform's existing failure handling held up under
audit (typed exceptions, no secret/traceback leakage, per-query/per-metric fault isolation in the
evaluation harnesses); five concrete gaps were fixed:
- **Embedding model caching** — `EmbeddingService` is now cached process-wide (mirroring the
  existing BM25 index cache), so the SentenceTransformer model is no longer reloaded from scratch on
  every `/search/*` and `/agent/investigate` request.
- **`/evaluation/full` persistence failures** — a failed run-save previously returned a bare
  `run_id: null` with no explanation and nothing in the server logs. It now surfaces in the
  response's `errors` list and is logged, matching how `/evaluation/retrieval` and
  `/evaluation/reasoning` already handle the identical failure.
- **Silent `_build_*` helper failures** — four internal helpers in the evaluation API that degrade
  gracefully to `None` (missing OpenAI key, etc.) now log the exception they swallow, instead of
  giving zero server-side trace of why generation/grounding was skipped.
- **DB exception classification** — the platform-wide database exception handler was broadened from
  `OperationalError` to its common base, `SQLAlchemyError`, so connection-pool exhaustion and
  embedding-dimension-mismatch errors return `503` (service temporarily unavailable) instead of a
  generic `500`.
- **Typed persistence errors** — writing an evaluation run to disk (`ExperimentRepository`,
  `FileRunRepositoryMixin`) previously had no error handling on the write path at all (only reads
  were hardened); a disk-full or permission failure now raises one typed `PersistenceError` instead
  of a raw filesystem exception.

Two related, real gaps were identified but deliberately **not** fixed in this pass, since fixing
them means changing behavior rather than just handling failure more cleanly: the multi-agent
orchestrator has no fault isolation between hypotheses or iterations (one LLM failure partway
through discards all completed work for that investigation), and there is no retry/backoff for
transient LLM failures (a single rate-limit or timeout fails the whole call). Both are documented,
scoped, and left for a future phase rather than bundled in here.

72 new tests cover both sub-phases (48 for 24A's configuration validation, 24 for 24B's failure
paths); combined with everything before it, the full suite is 1,606 tests (see
[Running tests](#running-tests)).

## Alert-triggered investigation, deployment history, and an execution policy engine

Phase 25 asks a different question than everything above: not "how good is a diagnosis" but "what
happens once you have one." Three additions, each independently useful, wired together end to end.

**`POST /alerts/webhook` accepts a monitoring provider's alert as-is.** Prometheus Alertmanager and
PagerDuty webhook payloads have nothing in common syntactically — this normalizes either shape
(`app/services/alert_normalization.py`, structured the same way as the ingestion layer's
`SourceAdapter` — a common ABC, tried by `matches()`, first match wins) into a plain problem
statement, then runs it through the *unmodified* `MultiAgentInvestigationOrchestrator`. A human
typing the same problem into `/agent/investigate` gets an identical investigation; the webhook
changes only where the problem statement comes from.

**`DeploymentHistoryClient` answers "what deployed near this failure window."** Given a repository
and a failure timestamp, it returns GitHub commits in an asymmetric window — wide lookback (a
deploy precedes the failure it causes by minutes to hours), narrow lookahead (only to absorb clock
skew). This is a correlation-in-time lookup, not a causal claim: it returns candidates for a human
(or a future hypothesis-scoring step) to weigh, exactly the same posture `HypothesisEvaluator`
already takes toward its own supporting evidence. Like other read-only diagnostics in this
codebase, it fails open — a timeout or network error returns an empty list with a warning logged,
rather than failing the investigation that requested context.

**`ActionPolicyEngine` is the actual point of this phase.** Everything upstream of it produces a
*diagnosis* — a root cause and a confidence score. None of it should be trusted to decide when an
*action* is safe, because confidence measures how sure a model is that it found the right problem,
not how reversible a proposed fix is. The engine classifies a proposed action into exactly two
categories and applies exactly one rule per category: `READ_ONLY` may proceed automatically;
`STATE_CHANGING` **always** requires human approval, with no threshold and no confidence override —
a rollback proposed at confidence 0.99 gets the identical gate as one proposed at 0.40. An action
type the engine has never seen fails closed to `STATE_CHANGING` rather than being assumed safe. This
is the same deterministic, one-sentence-explainable posture as `RuleBasedPlanner` and
`HeuristicCriticAgent` elsewhere in this codebase — and it is deliberately *not* tunable, because the
one failure this module exists to prevent is a sufficiently confident diagnosis buying its way past
a human.

The webhook ties the three together: after an investigation returns, a small heuristic (authored by
inspection, not validated against real incident/deployment correlation data — flagged as exactly
that in the code) suggests `rollback_deployment` when a deployment landed in the failure window and
confidence is HIGH, or `gather_diagnostics` otherwise. Whatever it suggests is then run through the
policy engine, which is the part that actually matters: the suggestion can be as wrong as it likes,
because a state-changing suggestion never executes — this project implements no execution path at
all, on purpose. `suggested_action`/`policy_decision` are null whenever the investigation abstained,
since there is nothing to act on.

**An MCP server exposes four of these capabilities as tools** (`app/mcp/server.py`, run with
`python -m app.mcp.server`): `investigate`, `search_incidents`, `check_action_policy`,
`deployment_history`. Each tool is a thin adapter over an already-tested service — no new logic
lives in the MCP layer, so a wrong answer is a bug in the underlying service, not in the adapter.
There is deliberately no `rollback` or `restart_service` tool: giving an MCP client a tool that
skips `check_action_policy` would defeat the reason the policy engine exists. Verified against the
real protocol, not just as Python functions — a subprocess spawned over stdio with the actual `mcp`
client SDK correctly lists all four tools and returns `requires_approval: true` for a
`rollback_deployment` call made at confidence 0.99.

98 new tests (policy engine, deployment history, alert normalization, the webhook route, the MCP
adapters); combined with everything before it, the full suite is 1,606 tests.

## Evaluation findings

Phase 22's evaluation work went beyond running the metrics — it stress-tested whether the
platform's confidence signal actually means what it's assumed to mean, and found a real gap:

**Dense-retrieval confidence is not a validated abstention mechanism.** The existing
`classify_confidence` thresholds (0.40/0.55) were originally calibrated against 4 negative-control
queries that were all clearly out-of-domain (payments, mobile push, spreadsheets, VPN). A follow-up
evaluation built 20 **hard negatives** — queries that deliberately share vocabulary or a general
topic with something already in the corpus (e.g. a Helm values-merging question near a real Helm
values bug) without describing the same underlying incident — and re-ran calibration against the
live corpus. Result: top-1 dense similarity alone does **not** cleanly separate genuine matches from
these harder negatives. 11 of 20 hard negatives scored in the "HIGH" confidence band — the same band
genuine matches occupy — and no single threshold in the tested range achieved both good precision
and good recall against them, unlike the clean separation the original 4-negative calibration
showed. One genuine, verified positive match in the current corpus scores *below* the current LOW
boundary, which is the concrete shape of the precision/recall tension: raising the threshold to
catch more hard negatives would also start rejecting real matches.

**The production investigation path narrows this gap but does not close it.** Running the full
`/agent/investigate` orchestrator against the same 20 hard negatives — with the 8 easy negatives and
6 verified genuine matches as controls — it abstains (`is_uncertain: true`, no root cause) on every
query whose top-1 retrieval lands in the MEDIUM or LOW band: 9 of 20 hard negatives and 8 of 8 easy
negatives. That's the composite-confidence floor working as designed: when the per-hypothesis
evidence search finds nothing, every hypothesis is rejected. But all 11 hard negatives in the HIGH
band (top-1 ≥ 0.55) came back with a confident root cause (mean confidence 0.80), and the rule-based
critic approved 9 of them — the same confidence and verdict the genuine matches received, so nothing
in the response distinguishes a real answer from a plausible wrong one. On the other side, 2 of the
6 verified positives (both paraphrase-style, where keyword validation fails) were wrongly abstained.
Probe script: `scripts/probe_orchestrator_hard_negatives.py`; per-query results:
`.benchmarks/orchestrator_hard_negative_probe.json`.

**A candidate second signal looks promising, and is not yet shipped.** Since the gap is bounded to
the HIGH band, the natural experiment is a targeted one: a single cheap LLM call asking whether the
top-1 retrieved incident describes the *same underlying problem* as the query — aimed at exactly
what embeddings cannot see (same topic, different problem). Measured on all 35 HIGH-band queries
(11 hard negatives, 24 genuine matches): **11/11 hard negatives correctly rejected** (against a
baseline of 0/11 — today every one gets a confident answer) and **20/21 genuine matches correctly
kept**, where "genuine match" means retrieval actually surfaced the expected incident. Precision
1.00, recall 0.95, at roughly one second and one cheap LLM call per query.

**Its run-to-run stability has since been measured, and it is exact.** Re-running the identical
experiment 5 times (175 LLM calls) using this project's own evaluator-stability tooling
(`measure_stability`, the same HIGH/MEDIUM/LOW variance bands applied to grounding metrics):
**zero queries changed decision** across all repetitions. Precision, recall, and FPR each had a
standard deviation of exactly 0.0 (evaluator confidence HIGH on every query). The single false
rejection is stable rather than stochastic — `v2-multi-04` was rejected in all 5 runs, with a
consistent stated reason, so it's a genuine limitation on multi-concept queries where top-1 covers
only part of the question, not noise. Notably the decisions are also **not** explainable as a
re-parameterized similarity threshold: the gate rejected a hard negative scoring 0.780 while keeping
a genuine match scoring 0.646, so it separates on something orthogonal to the score it's meant to
supplement.

It is nonetheless **still not wired in**, because stability is not validity. Two limits remain, and
the second is the serious one: n=35 is small, and the hard negatives were authored by finding a real
incident and writing a query that shares its topic but differs in mechanism — which is precisely the
distinction the gate's prompt asks it to make. Comparing the gate's stated reasons against the
dataset's own authoring notes shows them converging on nearly the same sentences ("a values file set
to an empty object, not multiple values files with the same key"). The gate never sees those notes —
it receives only the query and the retrieved incident's title/status/resolution, so there is no
literal leakage — but the test set was *constructed* around the criterion being tested, so this
result demonstrates the gate works on hard negatives of this construction, not that it generalizes to
naturally-occurring queries. Closing that requires hard negatives authored independently of the
gate's criterion. Probes: `scripts/probe_llm_relevance_gate.py`,
`scripts/probe_llm_relevance_gate_stability.py`; results:
`.benchmarks/llm_relevance_gate_probe.json`, `.benchmarks/llm_relevance_gate_stability.json`.

**The pipeline could be confidently wrong even when retrieval was perfect, and the cause was a
planner bug.** Tracing a single investigation end to end (`scripts/trace_investigation.py`) against
a query that was the exact title of a real incident — retrieved at rank 1, top-1 0.9301, HIGH
confidence — returned the root cause *"Network latency due to misconfigured routing paths"* at
confidence 0.80, critic APPROVED. `RuleBasedPlanner` was concatenating the problem statement with
all ten retrieved incidents into one keyword-matching haystack, so a ~10-word problem made up well
under 1% of the matched text. With NETWORK second in the priority order holding generic keywords
(`timeout`, `latency`, `socket`, `proxy`), almost any real result set dragged the plan to NETWORK,
and since the plan steers hypothesis generation the whole investigation followed it. Fixed by
matching the problem alone first and consulting evidence only when the problem names no strategy;
keyword matching also now accepts plurals (`pods` → `pod`), which had been dropping genuine
infrastructure matches. On the probe's positive controls two answers went from wrong to correct
(`v2-multi-03` "DNS resolution issues" → "Insufficient memory allocation for BestEffort pods";
`v2-multi-06` "Misconfigured OAuth scopes" → "Incorrect handling of relative paths in stack trace
resolution"). One case is deliberately unfixed: `v2-lex-08`'s problem text matches no keyword at all,
so it still falls back to evidence and still lands on NETWORK.

**Evidence evaluation was structurally incapable of finding contradiction.** `HypothesisEvaluator`
searched using the *hypothesis's own* validation keywords and then labelled results by cosine
similarity to that same query — at or above 0.40 "supporting", below it "contradicting". A
hypothesis therefore always found agreement with itself, and a low-similarity result is irrelevant
rather than contradicting. Across the 34-investigation probe this produced **78 supporting evidence
items against 2**, only 1 of 34 investigations found any contradicting evidence at all, and the
critic's `rejected` verdict — which fires on a contradiction ratio ≥ 0.5 — **never fired once**. The
critic is arithmetic over these counts, so its rejection path was effectively dead code.

**A local cross-encoder separates genuine matches from topical neighbours where cosine cannot.**
Phase 22F's finding was that top-1 similarity cannot make this distinction. Scoring the (query,
incident) pair jointly with `cross-encoder/ms-marco-MiniLM-L6-v2` — which runs locally beside the
existing SentenceTransformer, needs no API key, and is deterministic — can. Calibrated on 56 gold
queries (28 positives, 8 easy negatives, 20 hard negatives) via the production
`retrieve(expand=True, rerank=True)` path, counting a positive only when retrieval actually
surfaced its expected incident:

| gate | recall | FPR (hard negatives) | precision |
| --- | --- | --- | --- |
| cosine ≥ 0.40 (current default) | 1.000 | 0.900 | 0.553 |
| cosine ≥ 0.60 | 0.846 | 0.450 | 0.710 |
| cosine ≥ 0.70 | 0.654 | 0.200 | 0.810 |
| **cross-encoder ≥ 2.0** | **0.885** | **0.150** | **0.885** |

It dominates rather than trades: at higher recall than cosine at 0.60 it has a third of the
false-accept rate, and cosine only reaches a comparable FPR by collapsing recall to 0.654. Wired as
a pre-generation gate in the orchestrator (`RETRIEVAL_GATE_ENABLED`, **off by default**), the
end-to-end probe moved hard-negative abstention from 8/20 to 16/20 while rejecting **zero** of the
six positive controls — the two that still abstain do so through `max_iterations`/`no_progress`,
exactly as before. Four of 28 negatives still slip through, a 0.143 false-accept rate matching the
offline calibration's prediction. Because the gate runs before hypothesis generation, 24 of 34 probe
queries cost **zero LLM calls** instead of 3-5. Calibration runner:
`scripts/calibrate_retrieval_gate.py`; results: `.benchmarks/retrieval_gate_calibration.json`.

Two limits are load-bearing here. The threshold was read off this curve rather than fitted on a
held-out split, and it rests on one corpus, one run, and 20 hard negatives. And the gate only stops
the system answering questions the corpus cannot answer — it does nothing about answering the wrong
way a question the corpus *can* answer, which `v2-lex-08` still demonstrates.

**A second reasoning model catches what arithmetic cannot.** The rule-based critic decides by
contradiction ratio, and since evidence is retrieved using each hypothesis's own keywords that ratio
is pinned near zero — which is why it has never once rejected anything. `LLMCriticAgent`
(`LLM_CRITIC_ENABLED`, **off by default**) replaces that arithmetic with one call to a different
model family, asking the question no rule can: *does this root cause actually address the problem as
stated?* On the exact case that motivated this work — the Zookeeper resource-leak query the
heuristic critic approved at confidence 0.8:

| critic | verdict | reason |
| --- | --- | --- |
| Heuristic (arithmetic) | `approved`, 0.8 | contradiction ratio 0.00, below the 0.50 threshold |
| Gemini (reasoning) | `alternative_hypothesis_plausible`, 0.8 | "addresses network latency, while the problem statement describes a resource management/lifecycle issue" |

That the proposer (`OPENAI_MODEL`) and the challenger (`GEMINI_MODEL`) are different vendors is the
design, not a detail: a model grading its own output tells you it is self-consistent, not that it is
right. The critic's prompt is also told explicitly that supporting evidence was gathered using the
hypothesis's own keywords, so apparent agreement is weak by construction.

It ships off for two reasons. It costs one extra LLM call per iteration, and Gemini's free tier
allows 20 requests per day *per model*, so quota exhaustion is routine rather than exceptional.
Every failure path — missing key, quota, overload, unparseable response — falls back to the
heuristic critic and records that it did so in the returned `findings`, so a reader can always tell
which critic produced a verdict. An abstained decision spends no call at all, since there is no root
cause to challenge. This has been demonstrated on one case, not measured across the gold set; that
run is still pending.

**Applying the same scorer to the evidence split measures the drift rather than fixing it.**
Re-scoring supporting/contradicting against the *problem* instead of the hypothesis's keywords
inverts the counts almost exactly — 78/2 becomes 2/78, and critic approvals drop from 16 to 0. That
is not the threshold being harsh: it says that when a hypothesis searches for confirmation of
itself, what it finds has almost nothing to do with the question actually asked. Fixing it requires
changing the retrieval query, not just the scoring, so `EVIDENCE_RELEVANCE_ENABLED` ships **off** and
the flaw is documented rather than half-patched.

This does not mean confidence scoring is broken or unused — it's real, computed, and already
threaded through the investigation agents and (as of Phase 22D) gates the generation step against
the clearest failures (very low similarity). What it means is narrower and more honest: **top-1
similarity alone should not be presented as a solved or fully validated "don't answer when
unsure" mechanism.** It catches the easy case reliably; it does not reliably catch a topically
adjacent but substantively different incident. Whether a second signal (a cheap LLM relevance check,
score-distribution shape, corpus-relative ranking) closes this gap is an open, explicitly
unresolved question — two candidate signals (rank-1-to-rank-2 gap, source diversity) were tested and
did not cleanly separate the two classes either. The orchestrator probe above does narrow where such
a signal is needed: only the HIGH band, since the composite floor already handles MEDIUM and LOW. See
`docs/architecture/14_confidence_calibration.md` for the original calibration this finding builds
on; the hard-negative dataset is `tests/eval/gold_queries_hard_negatives_v1.json` (each entry
records the nearest real incident it was probed against and why it isn't a genuine match), the
calibration runner is `tests/eval/run_phase22f_confidence_calibration.py`, and the full per-query
results are in `tests/eval/results/phase22f_confidence_calibration.json`.

## Security & operational notes

Consolidated pointer to what's implemented — each is covered in more depth in its own section above:

- **Authentication** — Bearer API-key auth on every business endpoint, fail-closed if unconfigured.
  See [Authentication](#authentication).
- **Rate limiting** — per-endpoint-group, per-caller limits with `Retry-After`/`X-RateLimit-*`
  headers. See [Rate limiting](#rate-limiting).
- **Environment-aware configuration** — `ENVIRONMENT=production` enforces a real `API_KEY` and a
  non-localhost `DATABASE_URL` at startup, and disables `/docs`/`/redoc`/`/openapi.json`. See
  [Configuration](#configuration) and [Production hardening (Phase 24)](#production-hardening-phase-24).
- **Non-root Docker runtime** — the `Dockerfile` runs the application as a dedicated non-root
  `appuser`, multi-stage build (no build toolchain in the runtime image), with a container
  healthcheck. See [Production hardening (Phase 23)](#production-hardening-phase-23).
- **No secrets in the repository** — `.env` is gitignored; only `.env.example` (placeholder values)
  is committed.

## Corpus (at time of writing)

~8,000 incidents: ~6,500 from GitHub (16 repositories) + ~1,500 from Jira (KAFKA, SPARK, CASSANDRA).

## Current limitations

See [`docs/architecture/16`](docs/architecture/16_current_limitations.md) for the full list and
[`docs/architecture/17`](docs/architecture/17_future_roadmap.md) for the prioritized roadmap. Notable
ones: BM25/Hybrid retrieval requires a process restart to see newly-ingested incidents (the index is
built once and cached, not incrementally updated); the evaluation platform is reachable via its REST
API and CLI scripts but not run automatically by CI (CI runs the unit/API suite, not evaluation
runs, which need a live corpus and API credentials); rate limiting (Phase 23C, see
[Rate limiting](#rate-limiting)) is in-memory and process-local, correct only as long as the
deployment stays single-process (the current `Dockerfile`/`docker-compose.yml` already do); top-1
dense-similarity confidence does not reliably separate genuine matches from topically-adjacent hard
negatives and should not be treated as a validated abstention mechanism (see
[Evaluation findings](#evaluation-findings)); the multi-agent investigation orchestrator has no
fault isolation between hypotheses/iterations and there is no retry/backoff for transient LLM
failures (both identified in the Phase 24B audit, deliberately not fixed in that pass — see
[Production hardening (Phase 24)](#production-hardening-phase-24)); evidence evaluation retrieves
using each hypothesis's own keywords, so it cannot surface evidence *against* a hypothesis and the
`contradicting_evidence` field is a misnomer (see [Evaluation findings](#evaluation-findings)); the
cross-encoder retrieval gate is measured but ships disabled, with a threshold read off its own
calibration curve rather than fitted on a held-out split; the Anthropic/Gemini judge clients are
implemented and tested but not constructed by any evaluation run, so the LLM-as-judge circularity
caveat (answers written and graded by the same model family) remains open in practice; and Phase
25's action-suggestion heuristic (rollback vs. gather-diagnostics) is authored by inspection, not
validated against real incident/deployment correlation data — the policy engine that gates whatever
it suggests is the load-bearing part, the heuristic itself is not.

## Project status

**Core engineering implementation is substantially complete through Phase 25.** Ingestion, hybrid
retrieval with adaptive routing, the four-agent investigation loop, alert-triggered investigation
with deployment-history context and an execution policy engine, an MCP server exposing the platform
as tools, the evaluation platform (retrieval/reasoning/generation/grounding metrics, LLM-as-judge,
diagnostics), and two hardening passes (Phase 23's input validation/graceful-degradation/
load-testing, Phase 24's environment configuration and failure-handling audit) are all implemented
and covered by the current 1,606-test suite (all passing — see [Running tests](#running-tests)).

Since that checkpoint, a diagnostic pass found and fixed a planner bug that made the pipeline
confidently wrong even on perfect retrieval, and established that a local cross-encoder separates
genuine matches from hard negatives where top-1 cosine cannot — cutting the hard-negative
false-accept rate from 0.900 to 0.150 at 0.885 recall, for no API cost. Both the retrieval gate and
the cross-encoder evidence split are implemented, measured, and **shipped disabled**, pending a
held-out validation split.

What's genuinely open, not glossed over: the confidence-calibration gap above is narrowed but not
closed, and the measured gate is unshipped rather than solved; evidence evaluation still cannot find
contradicting evidence by construction, so the rule-based critic has never once returned `rejected`
and the LLM critic that catches this is demonstrated on a single case rather than measured across the
gold set; two failure-handling gaps in the investigation orchestrator (noted above) were identified
and deliberately deferred rather than fixed; the LLM-as-judge circularity caveat is unresolved because
no evaluation run constructs the second-model judge clients that now exist; Phase 25's
rollback-vs-diagnostics suggestion is a hand-authored heuristic, not something measured against real
deployment/incident correlation, and this project implements no execution path for any action —
policy decisions are recommendations, nothing actually rolls back or restarts anything; and the
platform has not been deployed anywhere beyond local Docker Compose — no staging/production
environment, no live deployment verification. CI now runs the full suite on every push and pull
request (`.github/workflows/ci.yml`); linting runs alongside it but is deliberately advisory rather
than gating, because the codebase has ~418 ruff findings (mostly docstring line-length and import
ordering, none behavioural) and a permanently red build trains people to ignore CI.

The work is intentionally paused here, before deployment, specifically to keep an accurate and
honest snapshot in place first.

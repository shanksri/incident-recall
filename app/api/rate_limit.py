"""Phase 23C: centralized, endpoint-aware rate limiting.

One dependency factory (`_group_dependency`) producing one named,
importable dependency per endpoint-cost group (`search_rate_limit`,
`agent_rate_limit`, ...), wired onto routers/routes exactly like Phase
23B's `require_api_key` — never re-implemented inside a route body. Limits
are per-group *and* per-caller: each group has its own bucket, and within
a group each caller identity (API key, or client IP if no key was
presented — see `_resolve_identity`) has its own independent counter.

# Why identity is "whatever Bearer token was presented," not "validated key"

Every rate-limited router already requires authentication (Phase 23B), so
by the time a request reaches business logic it necessarily carried a
*valid* key — `require_api_key` already rejected it with 401 otherwise.
Re-deriving "the API key" here by re-validating would just duplicate that
check. Instead, identity is resolved directly from the raw `Authorization`
header: a well-formed `Bearer <token>` uses that token string as the
bucket key (regardless of whether the platform's single shared key
happens to equal it — see Remaining risks in the Phase 23C report for why
a single shared key limits how much this actually discriminates between
callers today); anything else (no header, wrong scheme) falls back to
`request.client.host`. This keeps the dependency generically reusable on a
future endpoint that doesn't require auth, per the phase's "reusable
dependency" requirement, even though under the *current* wiring every
rate-limited request is already authenticated and the IP-fallback branch
is not reachable over HTTP (it is still directly unit-tested).

# Why a fixed window, not a sliding log

A fixed 60-second window (bucket = `floor(now / 60) * 60`) is the simplest
implementation that satisfies every required test case (below/at/above
limit, reset after the window rolls over, independent buckets) without
needing to store or prune per-request timestamps. It has the well-known
fixed-window edge case (a burst straddling a window boundary can
momentarily allow up to ~2x the limit) — accepted here as a reasonable
tradeoff for a "protect against accidental/malicious abuse" objective, not
a hard billing-grade guarantee; documented as a remaining risk.

# Why the backend is abstracted at all

`RateLimitBackend` (ABC) has exactly one implementation
(`InMemoryRateLimitBackend`) — in-memory, process-local, matching the
platform's current single-process deployment (explicitly required: no
Redis, no distributed infrastructure). The abstraction exists so a future
distributed backend (Redis `INCR`+`EXPIRE`, etc.) can be dropped in behind
the same `check()`/`reset()` interface without touching a single route or
dependency call site — the dependency factory below only ever talks to
`RateLimitBackend`, never to `InMemoryRateLimitBackend` directly.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import HTTPException, Request, Response

from app.core.config import settings

_WINDOW_SECONDS = 60


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_at: float  # epoch seconds when the current window ends


class RateLimitBackend(ABC):
    """Storage interface for rate-limit counters. See module docstring's
    "Why the backend is abstracted at all" for why this exists with only
    one implementation today.
    """

    @abstractmethod
    def check(self, key: str, *, limit: int, window_seconds: int) -> RateLimitDecision:
        """Record one request against `key` and return whether it's
        within `limit` for the current `window_seconds`-wide window.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all counters. Test-only — never called from production
        request handling.
        """


class InMemoryRateLimitBackend(RateLimitBackend):
    """Fixed-window counter keyed by an opaque string (already composed as
    ``f"{group}:{identity}"`` by the caller — this class knows nothing
    about groups or identities). A ``threading.Lock`` guards the shared
    dict because FastAPI's sync route/dependency functions run in a
    bounded threadpool (see the Phase 23 performance findings), so
    concurrent requests can genuinely race on the same counter.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._counters: dict[str, tuple[int, int]] = {}  # key -> (window_start, count)

    def check(self, key: str, *, limit: int, window_seconds: int) -> RateLimitDecision:
        now = self._clock()
        window_start = int(now // window_seconds) * window_seconds
        with self._lock:
            stored_start, count = self._counters.get(key, (window_start, 0))
            if stored_start != window_start:
                stored_start, count = window_start, 0
            count += 1
            self._counters[key] = (stored_start, count)
        reset_at = float(stored_start + window_seconds)
        return RateLimitDecision(
            allowed=count <= limit,
            limit=limit,
            remaining=max(0, limit - count),
            reset_at=reset_at,
        )

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()


# Process-local singleton — see module docstring. `reset_rate_limits()` is
# the test-only entry point that clears it (tests/api/conftest.py calls
# this before every test for isolation; see that file for why).
_backend: RateLimitBackend = InMemoryRateLimitBackend()


def reset_rate_limits() -> None:
    """Test-only: clear all counters. Never call this from request-serving
    code.
    """
    _backend.reset()


def _resolve_identity(request: Request) -> str:
    """See module docstring's "Why identity is ..." section."""
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        return f"key:{token}"
    client = request.client
    host = client.host if client is not None else "unknown"
    return f"ip:{host}"


def _group_dependency(group: str, limit_getter: Callable[[], int | None]) -> Callable:
    """Build the FastAPI dependency for one endpoint-cost group. `limit_getter`
    is read on every call (not captured once) so changing `Settings` at
    runtime — e.g. in a test via `monkeypatch.setattr(settings, ...)` —
    takes effect immediately, matching "configurable through Settings."
    """

    def _dependency(request: Request, response: Response) -> None:
        if not settings.rate_limit_enabled:
            return
        limit = limit_getter()
        if limit is None:  # explicit "unlimited" for this group
            return

        identity = _resolve_identity(request)
        decision = _backend.check(f"{group}:{identity}", limit=limit, window_seconds=_WINDOW_SECONDS)

        response.headers["X-RateLimit-Limit"] = str(decision.limit)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(decision.reset_at))

        if not decision.allowed:
            retry_after = max(0, int(decision.reset_at - time.time()))
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded for {group!r}: {limit} requests per "
                    f"{_WINDOW_SECONDS} seconds. Retry after {retry_after} seconds."
                ),
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(decision.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(decision.reset_at)),
                },
            )

    return _dependency


# ── One named, importable dependency per endpoint-cost group ────────────────
#
# Named module-level objects (not inline `Depends(_group_dependency(...))`
# at each call site) so every router/route wiring below references the
# exact same callable — required for the OpenAPI/dependency machinery to
# treat repeated use across routes as "the same dependency," and so tests
# have a single stable object to reason about per group.

search_rate_limit = _group_dependency("search", lambda: settings.rate_limit_search_per_minute)
agent_rate_limit = _group_dependency("agent", lambda: settings.rate_limit_agent_per_minute)
evaluation_query_rate_limit = _group_dependency(
    "evaluation_query", lambda: settings.rate_limit_evaluation_query_per_minute
)
evaluation_retrieval_rate_limit = _group_dependency(
    "evaluation_retrieval", lambda: settings.rate_limit_evaluation_retrieval_per_minute
)
evaluation_reasoning_rate_limit = _group_dependency(
    "evaluation_reasoning", lambda: settings.rate_limit_evaluation_reasoning_per_minute
)
evaluation_full_rate_limit = _group_dependency(
    "evaluation_full", lambda: settings.rate_limit_evaluation_full_per_minute
)
evaluation_runs_rate_limit = _group_dependency(
    "evaluation_runs", lambda: settings.rate_limit_evaluation_runs_per_minute
)
interactive_evaluation_rate_limit = _group_dependency(
    "interactive_evaluation", lambda: settings.rate_limit_interactive_evaluation_per_minute
)
incidents_rate_limit = _group_dependency("incidents", lambda: settings.rate_limit_incidents_per_minute)
ingestion_rate_limit = _group_dependency("ingestion", lambda: settings.rate_limit_ingestion_per_minute)
alerts_rate_limit = _group_dependency("alerts", lambda: settings.rate_limit_alerts_per_minute)

# Documented, reusable OpenAPI fragment for the 429 response — attached to
# every rate-limited router via `APIRouter(responses=RATE_LIMIT_RESPONSES)`
# so Swagger shows the possible 429 without repeating this per route.
RATE_LIMIT_RESPONSES: dict[int | str, dict] = {
    429: {
        "description": (
            "Rate limit exceeded for this endpoint group. See the "
            "`Retry-After` and `X-RateLimit-*` response headers."
        ),
    }
}

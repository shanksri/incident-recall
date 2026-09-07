from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.routes import (
    agent,
    alerts,
    evaluation,
    evaluation_interactive,
    health,
    incidents,
    ingestion,
    search,
)
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: best-effort DB connectivity probe, logged either way — never
    blocks the app from starting (a DB that isn't up yet, e.g. during a
    rolling deploy, shouldn't prevent the container from becoming live;
    ``/health/ready`` is what a load balancer/orchestrator should gate
    traffic on). Shutdown: dispose the connection pool cleanly rather than
    letting connections leak on process exit.
    """
    from app.db.session import engine

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("startup: database connectivity OK")
    except Exception:  # noqa: BLE001 — startup must never crash on this
        logger.warning("startup: database not reachable yet (will retry per-request)")
    yield
    engine.dispose()
    logger.info("shutdown: database connection pool disposed")


# Phase 24A: /docs, /redoc, and /openapi.json are public and unauthenticated
# by FastAPI's own default (they aren't behind any router's require_api_key
# dependency). That's fine in development, but in production it exposes the
# full API surface/schema to anyone unauthenticated, so all three are
# disabled outright rather than gated some other way. Every business route's
# actual auth/rate-limiting is unaffected either way. Extracted as a pure
# function (rather than inlined) so it's directly unit-testable without
# reloading this module or its FastAPI app.
def _docs_urls(environment: str) -> dict[str, str | None]:
    if environment == "production":
        return {"docs_url": None, "redoc_url": None, "openapi_url": None}
    return {"docs_url": "/docs", "redoc_url": "/redoc", "openapi_url": "/openapi.json"}


app = FastAPI(
    title="Enterprise Incident Intelligence Platform",
    version="0.1.0",
    description="Phase 1 GitHub incident ingestion and semantic incident search.",
    lifespan=lifespan,
    **_docs_urls(settings.environment),
)
app.include_router(health.router)
app.include_router(agent.router)
app.include_router(alerts.router)
app.include_router(incidents.router)
app.include_router(ingestion.router)
app.include_router(search.router)
app.include_router(evaluation.router)
app.include_router(evaluation_interactive.router)


# ── Phase 23: platform-wide failure handling ─────────────────────────────────
#
# Neither handler changes any route's behavior on success. Both exist so that
# a failure the route itself didn't anticipate (a DB connection drop, an
# unguarded service exception) degrades to a clean, typed JSON error instead
# of an unhandled-exception traceback — the same "log full detail server-side,
# return a generic message client-side" discipline already used by the
# evaluation API's ``_build_search_service``/``_build_orchestrator`` helpers,
# applied platform-wide as a safety net.


@app.exception_handler(SQLAlchemyError)
async def database_unavailable_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    # Phase 24B: broadened from OperationalError (DB connection/network
    # failures only) to SQLAlchemyError, its common base — OperationalError
    # is NOT the base of every DB-layer failure that should read as "service
    # temporarily unavailable" rather than a generic bug: pool-exhaustion
    # timeouts (sqlalchemy.exc.TimeoutError) and embedding-dimension
    # mismatches (sqlalchemy.exc.DataError) were previously falling through
    # to the generic 500 handler below instead of this one. Audited first
    # (see Phase 24B report): OperationalError is the only SQLAlchemy
    # exception type referenced anywhere else in this codebase or its
    # tests, so no existing behavior intentionally depended on any other
    # subtype reaching the generic 500 handler instead of this one.
    logger.exception("Database operation failed for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=503,
        content={"detail": "Database is temporarily unavailable. Please retry shortly."},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred."},
    )

"""
src/api/main.py
────────────────────────────────────────────────────────────────
FastAPI application — entry point for the Clinical NLP API.

Start the server
────────────────
  Development (auto-reload on code changes):
    uvicorn src.api.main:app --reload --port 8000

  Production (multiple workers):
    uvicorn src.api.main:app --workers 4 --port 8000

API documentation
─────────────────
  Swagger UI : http://localhost:8000/docs
  ReDoc      : http://localhost:8000/redoc
  OpenAPI    : http://localhost:8000/openapi.json

Environment variables
─────────────────────
  DATABASE_URL  — SQLAlchemy DB URL (default: SQLite)
  API_DEBUG     — enable /docs and /redoc (default: true)
  API_PORT      — server port (used by the run helper)
  LOG_LEVEL     — DEBUG / INFO / WARNING (default: INFO)
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import entities, icd, model, notes
from src.api.schemas import HealthResponse
from src.db.connection import check_connection, create_all_tables
from src.utils.config import APIConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────
# FastAPI's lifespan replaces the older @app.on_event pattern.
# Code before `yield` runs at startup; code after runs at shutdown.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown tasks for the application."""

    # ── Startup ───────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("  Clinical NLP API  v%s", APIConfig.version)
    logger.info("=" * 50)

    # Ensure all database tables exist before the first request
    try:
        create_all_tables()
        db_status = "connected" if check_connection() else "unreachable"
    except Exception as exc:
        logger.error("Database setup failed: %s", exc)
        db_status = "unreachable"

    # Attach db_status to app.state so the health endpoint can read it
    app.state.db_status = db_status
    logger.info("Database: %s", db_status)

    # Eagerly load the NER pipeline, ICD-10 mapper, and classifier now,
    # rather than lazily on whichever request happens to be first.
    # Most deployment platforms hold off routing traffic until the
    # health check passes, so this cost is absorbed into startup time
    # rather than a real user's request.
    #
    # Skippable via WARM_UP_MODELS=false -- on a memory-constrained host
    # (e.g. Render's free 512Mi tier) the hybrid NER pipeline + classifier
    # can get OOM-killed by the OS at boot, which no try/except below can
    # catch or recover from since it isn't a Python exception. Skipping
    # warm-up at least lets the API start and serve DB-backed endpoints;
    # NLP endpoints will still need enough memory when actually called.
    if APIConfig.warm_up_models:
        try:
            notes.warm_up()
        except Exception:
            # Non-fatal: whatever didn't warm up here will load lazily on
            # whichever request needs it. exc_info=True so a resource-
            # contention failure (e.g. an import failing under low memory)
            # is actually diagnosable instead of logging an empty message.
            logger.error("Model warm-up failed — falling back to lazy loading", exc_info=True)
    else:
        logger.info("WARM_UP_MODELS=false — skipping eager model load, will load lazily per-request")

    logger.info("API ready at http://%s:%s", APIConfig.host, APIConfig.port)

    yield   # application is running

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("Shutting down Clinical NLP API...")


# ── Application ───────────────────────────────────────────────────

app = FastAPI(
    title       = APIConfig.title,
    version     = APIConfig.version,
    description = APIConfig.description,
    docs_url    = "/docs"    if APIConfig.debug else None,
    redoc_url   = "/redoc"   if APIConfig.debug else None,
    lifespan    = lifespan,
)

# CORS — allow the Streamlit dashboard (and any other origin in dev)
# to call the API from the browser.
# In production, restrict origins to your actual dashboard domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Routers ───────────────────────────────────────────────────────

app.include_router(notes.router)
app.include_router(entities.router)
app.include_router(icd.router)
app.include_router(model.router)


# ── Health check ──────────────────────────────────────────────────

@app.get(
    "/health",
    response_model = HealthResponse,
    tags           = ["Health"],
    summary        = "API health check",
)
def health() -> HealthResponse:
    """Return the current health status of the API.

    Used by load balancers, monitoring tools, and the Streamlit
    dashboard to confirm the API is running before making requests.

    Returns:
        :class:`HealthResponse` with status, version, DB state, and
        whether semantic ICD-10 matching is available or degraded.
    """
    db_status = getattr(app.state, "db_status", "unknown")
    return HealthResponse(
        status   = "ok",
        version  = APIConfig.version,
        database = db_status,
        icd10_embedding_available = notes.icd10_embedding_available(),
    )


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

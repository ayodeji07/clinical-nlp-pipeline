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

from src.api.routes import entities, icd, notes
from src.api.schemas import HealthResponse
from src.db.connection import check_connection, create_all_tables
from src.utils.config import APIConfig, settings
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
        :class:`HealthResponse` with status, version, and DB state.
    """
    db_status = getattr(app.state, "db_status", "unknown")
    return HealthResponse(
        status   = "ok",
        version  = APIConfig.version,
        database = db_status,
    )


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

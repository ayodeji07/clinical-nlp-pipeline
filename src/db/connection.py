"""
src/db/connection.py
────────────────────────────────────────────────────────────────
Database connection and session management.

This module is the only place in the codebase that knows which
database engine is in use.  Everything above it (ETL, API,
repositories) calls get_session() and works with the result —
they never build connection strings or reference a dialect.

How the backend is selected
────────────────────────────
The DATABASE_URL environment variable drives the choice:

  Not set / .env missing
      → SQLite file at project root (clinical_nlp.db)
      → Zero configuration; works out of the box

  DATABASE_URL=sqlite:///./clinical_nlp.db
      → Same SQLite file, explicit

  DATABASE_URL=postgresql://user:pass@host:5432/dbname
      → PostgreSQL (Supabase, AWS RDS, local Postgres, anything)

One variable.  No code changes.

SQLite vs PostgreSQL quirks
────────────────────────────
SQLAlchemy handles most dialect differences transparently, but
two things need special handling:

  1. Connection pool: SQLite is file-based and single-writer;
     the NullPool prevents "database is locked" errors when
     multiple threads try to connect simultaneously.

  2. check_same_thread=False: Required for SQLite when used
     with FastAPI (which runs handlers in a thread pool).

Both are applied automatically based on the URL.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from src.utils.config import DatabaseConfig, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_engine(url: str) -> Engine:
    """Create a SQLAlchemy engine appropriate for the given URL.

    Applies connection pool settings that work correctly for both
    SQLite (development) and PostgreSQL (staging / production).

    Args:
        url: SQLAlchemy-compatible database URL.

    Returns:
        Configured :class:`sqlalchemy.engine.Engine` instance.
    """
    is_sqlite = url.startswith("sqlite")

    if is_sqlite:
        # NullPool avoids the "database is locked" error that occurs
        # when SQLite is accessed from multiple threads (e.g. FastAPI).
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
        )
        # Enable WAL mode for better concurrent read performance
        @event.listens_for(engine, "connect")
        def set_wal_mode(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")

        logger.debug("SQLite engine created: %s", url)

    else:
        # PostgreSQL — use connection pooling for efficiency
        engine = create_engine(
            url,
            pool_size    = DatabaseConfig.pool_size,
            max_overflow = DatabaseConfig.max_overflow,
            pool_timeout = DatabaseConfig.pool_timeout,
            # Recycle connections after 30 minutes to avoid
            # "server closed connection" errors on long-running apps
            pool_recycle = 1800,
        )
        logger.debug("PostgreSQL engine created")

    return engine


# ── Module-level singletons ───────────────────────────────────────
# Created once at import time.  Tests can call _reset() to swap
# in an in-memory SQLite database without restarting the process.

_engine: Engine | None         = None
_SessionFactory: sessionmaker | None = None


def get_engine() -> Engine:
    """Return the module-level database engine, creating it if needed.

    Returns:
        The active :class:`~sqlalchemy.engine.Engine`.
    """
    global _engine
    if _engine is None:
        _engine = _build_engine(DatabaseConfig.url)
    return _engine


def get_session_factory() -> sessionmaker:
    """Return the module-level session factory, creating it if needed.

    Returns:
        A :class:`~sqlalchemy.orm.sessionmaker` bound to the engine.
    """
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind        = get_engine(),
            autocommit  = False,
            autoflush   = False,
            expire_on_commit = False,   # safer for async contexts
        )
    return _SessionFactory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional database session as a context manager.

    Commits on clean exit; rolls back and re-raises on any exception.
    Always closes the session when the block exits.

    Yields:
        An active :class:`~sqlalchemy.orm.Session`.

    Example::

        with get_session() as session:
            note = session.get(ClinicalNote, note_id)
            note.severity = "urgent"
        # committed automatically on clean exit

        # Exception example:
        with get_session() as session:
            session.add(bad_record)
        # → rolls back; exception propagates to caller
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session per request.

    Designed for use with FastAPI's ``Depends()``.  Closes the
    session after the response is sent, even on errors.

    Yields:
        An active :class:`~sqlalchemy.orm.Session`.

    Example::

        @router.get("/notes/{note_id}")
        def read_note(note_id: int, db: Session = Depends(get_db_session)):
            return db.get(ClinicalNote, note_id)
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all_tables() -> None:
    """Create all database tables defined in the ORM models.

    Safe to call multiple times — uses ``checkfirst=True`` so
    existing tables are not dropped or modified.

    Typically called once at application startup.
    """
    from src.db.models import Base  # imported here to avoid circular imports
    Base.metadata.create_all(bind=get_engine(), checkfirst=True)
    logger.info("Database tables created (or already exist)")


def check_connection() -> bool:
    """Verify that the database is reachable and responding.

    Returns:
        True if the connection succeeds; False otherwise.

    Example::

        if not check_connection():
            raise RuntimeError("Database unreachable at startup")
    """
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection verified ✓")
        return True
    except Exception as exc:
        logger.error("Database connection failed: %s", exc)
        return False


def _reset_for_testing(url: str = "sqlite:///:memory:") -> None:
    """Replace the engine with a fresh in-memory database.

    Only intended for use in the test suite.  Do not call in
    production code.

    Args:
        url: Database URL for the test engine.
            Defaults to an in-memory SQLite database.
    """
    global _engine, _SessionFactory
    if _engine:
        _engine.dispose()
    _engine         = _build_engine(url)
    _SessionFactory = sessionmaker(
        bind=_engine, autocommit=False, autoflush=False
    )
    logger.debug("Test database engine reset: %s", url)

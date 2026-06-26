"""
src/db/repository.py
────────────────────────────────────────────────────────────────
Repository layer — all database queries live here.

The repository pattern keeps SQL out of the API routes and
business logic.  Routes call repository functions; they never
build queries directly.  This makes the API easy to test
(swap in a fake repository) and easy to change (queries are
in one place, not scattered across the codebase).

One class per table, following a consistent interface:
  get(id)           → single record or None
  list(filters)     → list of records
  create(data)      → new record
  update(id, data)  → modified record
  delete(id)        → bool (True if found and deleted)

All methods accept a Session argument so the caller controls
the transaction boundary — the repository never opens its own
session.  This is intentional: it lets the API layer wrap
multiple repository calls in a single transaction.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from src.db.models import ClinicalNote, Entity, ICD10Mapping, ModelRun
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NoteRepository:
    """Query interface for the clinical_notes table."""

    @staticmethod
    def get(session: Session, note_id: int) -> ClinicalNote | None:
        """Fetch a single note by primary key.

        Args:
            session: Active SQLAlchemy session.
            note_id: Primary key of the note.

        Returns:
            The matching :class:`ClinicalNote`, or None if not found.
        """
        return session.get(ClinicalNote, note_id)

    @staticmethod
    def get_by_source_id(
        session: Session, source_id: str
    ) -> ClinicalNote | None:
        """Fetch a note by its source dataset identifier.

        Used by the ETL loader to check for duplicates before inserting.

        Args:
            session: Active SQLAlchemy session.
            source_id: The stable identifier from the source dataset.

        Returns:
            Matching note or None.
        """
        return (
            session.query(ClinicalNote)
            .filter_by(source_id=source_id)
            .first()
        )

    @staticmethod
    def list_by_specialty(
        session: Session,
        specialty: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ClinicalNote]:
        """Return notes filtered by medical specialty.

        Args:
            session: Active SQLAlchemy session.
            specialty: Specialty string to filter on (case-insensitive).
            limit: Maximum records to return.
            offset: Number of records to skip (for pagination).

        Returns:
            List of :class:`ClinicalNote` objects.
        """
        return (
            session.query(ClinicalNote)
            .filter(
                func.lower(ClinicalNote.specialty) == specialty.lower()
            )
            .order_by(ClinicalNote.id)
            .limit(limit)
            .offset(offset)
            .all()
        )

    @staticmethod
    def list_by_severity(
        session: Session,
        severity: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ClinicalNote]:
        """Return notes filtered by severity label.

        Args:
            session: Active SQLAlchemy session.
            severity: One of ``"routine"``, ``"urgent"``, ``"critical"``.
            limit: Maximum records to return.
            offset: Records to skip.

        Returns:
            List of :class:`ClinicalNote` objects.
        """
        return (
            session.query(ClinicalNote)
            .filter_by(severity=severity)
            .order_by(ClinicalNote.id)
            .limit(limit)
            .offset(offset)
            .all()
        )

    @staticmethod
    def count_by_severity(session: Session) -> dict[str, int]:
        """Return the count of notes per severity label.

        Used by the dashboard to show distribution charts.

        Args:
            session: Active SQLAlchemy session.

        Returns:
            Dict mapping severity label to count,
            e.g. ``{"routine": 3200, "urgent": 1400, "critical": 399}``.
        """
        rows = (
            session.query(ClinicalNote.severity, func.count(ClinicalNote.id))
            .group_by(ClinicalNote.severity)
            .all()
        )
        return {row[0]: row[1] for row in rows if row[0]}

    @staticmethod
    def count_by_specialty(session: Session) -> dict[str, int]:
        """Return the count of notes per medical specialty.

        Args:
            session: Active SQLAlchemy session.

        Returns:
            Dict mapping specialty to note count, sorted descending.
        """
        rows = (
            session.query(ClinicalNote.specialty, func.count(ClinicalNote.id))
            .group_by(ClinicalNote.specialty)
            .order_by(func.count(ClinicalNote.id).desc())
            .all()
        )
        return {row[0]: row[1] for row in rows if row[0]}

    @staticmethod
    def total(session: Session) -> int:
        """Return the total number of notes in the database.

        Args:
            session: Active SQLAlchemy session.

        Returns:
            Integer count.
        """
        return session.query(func.count(ClinicalNote.id)).scalar() or 0


class EntityRepository:
    """Query interface for the entities table."""

    @staticmethod
    def get(session: Session, entity_id: int) -> Entity | None:
        """Fetch a single entity by primary key.

        Args:
            session: Active SQLAlchemy session.
            entity_id: Primary key.

        Returns:
            The matching :class:`Entity`, or None.
        """
        return session.get(Entity, entity_id)

    @staticmethod
    def list_by_note(
        session: Session,
        note_id: int,
    ) -> list[Entity]:
        """Return all entities extracted from a specific note.

        Eager-loads ICD-10 mappings so callers don't trigger
        additional queries per entity.

        Args:
            session: Active SQLAlchemy session.
            note_id: Primary key of the parent note.

        Returns:
            List of :class:`Entity` objects with mappings pre-loaded.
        """
        return (
            session.query(Entity)
            .options(joinedload(Entity.icd10_mappings))
            .filter_by(note_id=note_id)
            .order_by(Entity.start_char)
            .all()
        )

    @staticmethod
    def list_by_label(
        session: Session,
        label: str,
        limit: int = 500,
    ) -> list[Entity]:
        """Return entities filtered by entity type label.

        Args:
            session: Active SQLAlchemy session.
            label: Entity type to filter on (e.g. ``"DISEASE"``).
            limit: Maximum records to return.

        Returns:
            List of :class:`Entity` objects.
        """
        return (
            session.query(Entity)
            .filter(func.upper(Entity.label) == label.upper())
            .limit(limit)
            .all()
        )

    @staticmethod
    def top_entities(
        session: Session,
        label: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, int]]:
        """Return the most frequently extracted entity texts.

        Used by the dashboard frequency chart.

        Args:
            session: Active SQLAlchemy session.
            label: If provided, filter to this entity type only.
            limit: Number of top entities to return.

        Returns:
            List of ``(entity_text, count)`` tuples, sorted by count
            descending.

        Example::

            top = EntityRepository.top_entities(session, label="DISEASE", limit=10)
            # → [("hypertension", 412), ("diabetes", 389), ...]
        """
        q = (
            session.query(Entity.text, func.count(Entity.id).label("n"))
            .group_by(Entity.text)
            .order_by(func.count(Entity.id).desc())
        )
        if label:
            q = q.filter(func.upper(Entity.label) == label.upper())

        return [(row.text, row.n) for row in q.limit(limit).all()]

    @staticmethod
    def cooccurrence_pairs(
        session: Session,
        label: str = "DISEASE",
        min_count: int = 5,
        limit: int = 200,
    ) -> list[tuple[str, str, int]]:
        """Return entity pairs that appear together in the same note.

        Used to build the co-occurrence network graph.

        Args:
            session: Active SQLAlchemy session.
            label: Entity type to consider.
            min_count: Minimum co-occurrence count to include a pair.
            limit: Maximum number of pairs to return.

        Returns:
            List of ``(entity_a, entity_b, count)`` tuples.
        """
        # Self-join on note_id to find pairs within the same note
        e1 = Entity.__table__.alias("e1")
        e2 = Entity.__table__.alias("e2")

        stmt = (
            select(
                e1.c.text.label("entity_a"),
                e2.c.text.label("entity_b"),
                func.count().label("n"),
            )
            .select_from(e1)
            .join(e2, (e1.c.note_id == e2.c.note_id) & (e1.c.text < e2.c.text))
            .where(
                func.upper(e1.c.label) == label.upper(),
                func.upper(e2.c.label) == label.upper(),
            )
            .group_by(e1.c.text, e2.c.text)
            .having(func.count() >= min_count)
            .order_by(func.count().desc())
            .limit(limit)
        )

        rows = session.execute(stmt).fetchall()
        return [(r.entity_a, r.entity_b, r.n) for r in rows]


class ICD10Repository:
    """Query interface for the icd10_mappings table."""

    @staticmethod
    def list_by_entity(
        session: Session,
        entity_id: int,
    ) -> list[ICD10Mapping]:
        """Return all ICD-10 mappings for a given entity.

        Results are ordered by rank (best match first).

        Args:
            session: Active SQLAlchemy session.
            entity_id: Primary key of the entity.

        Returns:
            List of :class:`ICD10Mapping` objects.
        """
        return (
            session.query(ICD10Mapping)
            .filter_by(entity_id=entity_id)
            .order_by(ICD10Mapping.rank)
            .all()
        )

    @staticmethod
    def top_codes(
        session: Session,
        limit: int = 20,
    ) -> list[tuple[str, str, int]]:
        """Return the most frequently assigned ICD-10 codes.

        Used by the dashboard to show which diagnoses appear most
        often across the dataset.

        Args:
            session: Active SQLAlchemy session.
            limit: Number of top codes to return.

        Returns:
            List of ``(icd10_code, description, count)`` tuples.
        """
        rows = (
            session.query(
                ICD10Mapping.icd10_code,
                ICD10Mapping.description,
                func.count(ICD10Mapping.id).label("n"),
            )
            .group_by(ICD10Mapping.icd10_code, ICD10Mapping.description)
            .order_by(func.count(ICD10Mapping.id).desc())
            .limit(limit)
            .all()
        )
        return [(r.icd10_code, r.description, r.n) for r in rows]


class ModelRunRepository:
    """Query interface for the model_runs table."""

    @staticmethod
    def create(
        session: Session,
        model_name: str,
        task: str,
        **kwargs,
    ) -> ModelRun:
        """Record a new classifier training run.

        Args:
            session: Active SQLAlchemy session.
            model_name: Base model used for fine-tuning.
            task: Classification task name.
            **kwargs: Optional columns — ``val_accuracy``, ``val_f1``,
                ``training_samples``, ``epochs``, ``run_notes``.

        Returns:
            The newly created :class:`ModelRun` record.
        """
        run = ModelRun(model_name=model_name, task=task, **kwargs)
        session.add(run)
        session.flush()   # populate run.id without committing
        logger.info(
            "Model run recorded: %s / %s (id=%d)", model_name, task, run.id
        )
        return run

    @staticmethod
    def get_deployed(session: Session, task: str) -> ModelRun | None:
        """Return the currently deployed model run for a task.

        Args:
            session: Active SQLAlchemy session.
            task: Task name to look up (e.g. ``"severity"``).

        Returns:
            The deployed :class:`ModelRun`, or None if none is marked.
        """
        return (
            session.query(ModelRun)
            .filter_by(task=task, is_deployed=True)
            .order_by(ModelRun.created_at.desc())
            .first()
        )

    @staticmethod
    def list_by_task(
        session: Session,
        task: str,
        limit: int = 10,
    ) -> list[ModelRun]:
        """Return recent training runs for a task, newest first.

        Args:
            session: Active SQLAlchemy session.
            task: Task name.
            limit: Maximum runs to return.

        Returns:
            List of :class:`ModelRun` objects.
        """
        return (
            session.query(ModelRun)
            .filter_by(task=task)
            .order_by(ModelRun.created_at.desc())
            .limit(limit)
            .all()
        )

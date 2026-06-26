"""
src/db/models.py
────────────────────────────────────────────────────────────────
SQLAlchemy ORM models — one class per database table.

Table overview
──────────────
  clinical_notes   : processed MTSamples / MIMIC-III notes
  entities         : NER extractions (one row per entity span)
  icd10_mappings   : entity → ICD-10 code matches
  model_runs       : audit trail for classifier training runs

Design decisions
────────────────
  - All primary keys are auto-incrementing integers.  This is
    simpler than UUIDs and sufficient for this scale.

  - created_at is set by the database server, not the application,
    so it is consistent regardless of timezone or clock skew.

  - Text columns use Text() not String(N) — clinical notes can
    be arbitrarily long and we do not want silent truncation.

  - Relationships are defined with lazy="select" (the default)
    so they do not cause N+1 query problems in the API routes.
    Callers that need related data should use joinedload() or
    subqueryload() explicitly.

  - Every table has an updated_at column maintained by an
    onupdate trigger so we can audit changes without extra code.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models.

    Every table inherits from this so SQLAlchemy can discover
    them all when create_all_tables() is called.
    """
    pass


class ClinicalNote(Base):
    """A single processed clinical note.

    Populated by the ETL pipeline from MTSamples or MIMIC-III.
    One row per unique transcription.

    Columns
    ───────
    source_id       : stable identifier from the source dataset
                      (e.g. "mtsamples_42").  Used to detect
                      duplicate loads on re-runs.
    transcription   : cleaned note text
    specialty       : medical specialty (e.g. "Cardiology")
    note_type       : document type (e.g. "Discharge Summary")
    severity        : weak-supervision label — routine / urgent / critical
    word_count      : pre-computed for fast filtering
    data_source     : which dataset this came from (mtsamples / mimic3)
    """

    __tablename__ = "clinical_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    source_id: Mapped[str] = mapped_column(
        String(120), unique=True, nullable=False, index=True,
        comment="Stable ID from the source dataset — prevents duplicate loads",
    )
    transcription: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="Cleaned clinical note text",
    )
    specialty: Mapped[str | None] = mapped_column(
        String(120), nullable=True, index=True,
        comment="Medical specialty (e.g. Cardiology, Neurology)",
    )
    note_type: Mapped[str | None] = mapped_column(
        String(120), nullable=True,
        comment="Document type (e.g. Discharge Summary, Consult Note)",
    )
    severity: Mapped[str | None] = mapped_column(
        String(20), nullable=True, index=True,
        comment="Weak-supervision severity label: routine / urgent / critical",
    )
    word_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
        comment="Pre-computed word count for fast length filtering",
    )
    data_source: Mapped[str] = mapped_column(
        String(30), nullable=False, default="mtsamples",
        comment="Source dataset identifier (mtsamples or mimic3)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(),
        onupdate=func.now(), nullable=False,
    )

    # A note can have many extracted entities
    entities: Mapped[list[Entity]] = relationship(
        "Entity", back_populates="note", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"ClinicalNote(id={self.id}, specialty={self.specialty!r}, "
            f"severity={self.severity!r}, words={self.word_count})"
        )


class Entity(Base):
    """A single named entity extracted from a clinical note.

    One row per entity span.  A note with ten extracted entities
    produces ten rows in this table.

    Columns
    ───────
    note_id     : foreign key to clinical_notes
    text        : the exact extracted text (e.g. "hypertension")
    label       : entity type from the NER model
                  (DISEASE, MEDICATION, PROCEDURE, SYMPTOM, ANATOMY)
    start_char  : character offset in the cleaned transcription
    end_char    : character offset end
    confidence  : model confidence score (0.0 – 1.0) where available
    """

    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    note_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("clinical_notes.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    text: Mapped[str] = mapped_column(
        String(500), nullable=False,
        comment="Extracted entity text as it appears in the note",
    )
    label: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True,
        comment="Entity type: DISEASE, MEDICATION, PROCEDURE, SYMPTOM, ANATOMY",
    )
    start_char: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="Start character offset in the transcription",
    )
    end_char: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="End character offset in the transcription",
    )
    confidence: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="NER model confidence score (0.0–1.0)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False,
    )

    # Back-reference to the parent note
    note: Mapped[ClinicalNote] = relationship(
        "ClinicalNote", back_populates="entities",
    )

    # An entity can have multiple ICD-10 candidate mappings
    icd10_mappings: Mapped[list[ICD10Mapping]] = relationship(
        "ICD10Mapping", back_populates="entity", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"Entity(id={self.id}, text={self.text!r}, "
            f"label={self.label!r}, conf={self.confidence:.2f})"
            if self.confidence is not None
            else f"Entity(id={self.id}, text={self.text!r}, label={self.label!r})"
        )


class ICD10Mapping(Base):
    """An ICD-10 code matched to an extracted entity.

    One entity can have multiple candidate codes (top-k results).
    The best match is the one with the highest confidence.

    Columns
    ───────
    entity_id    : foreign key to entities
    icd10_code   : ICD-10-CM code (e.g. "I10", "J18.9")
    description  : human-readable description of the code
    match_method : how the match was made — "lookup" (fuzzy string
                   match) or "embedding" (cosine similarity fallback)
    confidence   : match confidence (0.0 – 1.0)
    rank         : position in the top-k results (1 = best match)
    """

    __tablename__ = "icd10_mappings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    icd10_code: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True,
        comment="ICD-10-CM diagnostic code",
    )
    description: Mapped[str] = mapped_column(
        String(500), nullable=False, default="",
        comment="Human-readable ICD-10 code description",
    )
    match_method: Mapped[str] = mapped_column(
        String(20), nullable=False, default="lookup",
        comment="How the match was found: lookup or embedding",
    )
    confidence: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Match confidence score (0.0–1.0)",
    )
    rank: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1,
        comment="Position in top-k candidates (1 = best match)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False,
    )

    # Back-reference to the parent entity
    entity: Mapped[Entity] = relationship(
        "Entity", back_populates="icd10_mappings",
    )

    def __repr__(self) -> str:
        return (
            f"ICD10Mapping(code={self.icd10_code!r}, "
            f"method={self.match_method!r}, conf={self.confidence:.2f})"
        )


class ModelRun(Base):
    """Audit record for a classifier training run.

    Tracks which model was trained, when, on how many samples,
    and what performance metrics it achieved.  Useful for
    comparing successive fine-tuning runs.

    Columns
    ───────
    model_name    : HuggingFace model ID used as base
    task          : classification task (severity / readmission)
    training_samples : number of training examples
    val_accuracy  : validation accuracy at end of training
    val_f1        : weighted F1 score on validation set
    epochs        : number of training epochs completed
    notes         : free-text notes about the run
    """

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    model_name: Mapped[str] = mapped_column(
        String(200), nullable=False,
        comment="Base model used for fine-tuning",
    )
    task: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True,
        comment="Classification task: severity or readmission",
    )
    training_samples: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )
    val_accuracy: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="Validation accuracy (0.0–1.0)",
    )
    val_f1: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="Weighted F1 score on validation set",
    )
    epochs: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )
    is_deployed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
        comment="Whether this run is the currently deployed model",
    )
    run_notes: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Free-text notes about training conditions or observations",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"ModelRun(id={self.id}, task={self.task!r}, "
            f"val_f1={self.val_f1}, deployed={self.is_deployed})"
        )

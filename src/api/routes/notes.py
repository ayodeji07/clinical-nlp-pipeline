"""
src/api/routes/notes.py
────────────────────────────────────────────────────────────────
Endpoints for clinical note analysis and retrieval.

Routes
──────
  POST /notes/analyse   — analyse a pasted clinical note
  GET  /notes           — list stored notes (paginated)
  GET  /notes/{id}      — retrieve one note with its entities
  GET  /notes/stats     — aggregate counts for the dashboard
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.schemas import (
    AnalyseRequest,
    AnalyseResponse,
    EntityResponse,
    ICD10MatchResponse,
    NoteDetail,
    NoteListResponse,
    NoteSummary,
    SeverityResponse,
    StatsResponse,
    SeverityDistribution,
    SpecialtyCount,
    TopEntityItem,
)
from src.db.connection import get_db_session
from src.db.repository import EntityRepository, ICD10Repository, NoteRepository
from src.utils.logger import get_logger
from src.etl.transform import prepare_for_inference
from src.utils.text_utils import word_count

logger    = get_logger(__name__)
router    = APIRouter(prefix="/notes", tags=["Notes"])

# NER pipeline and ICD mapper are initialised once at module level
# so the model is not reloaded on every request.
# They are loaded lazily — the first request will be slower.
_ner_pipeline = None
_icd_mapper   = None
_classifier   = None


def _get_ner():
    """Return the module-level NER pipeline, loading it if needed."""
    global _ner_pipeline
    if _ner_pipeline is None:
        from src.nlp.ner import build_ner_pipeline
        _ner_pipeline = build_ner_pipeline()
    return _ner_pipeline


def _get_mapper():
    """Return the module-level ICD-10 mapper, loading it if needed."""
    global _icd_mapper
    if _icd_mapper is None:
        from src.nlp.icd_mapper import ICD10Mapper
        try:
            _icd_mapper = ICD10Mapper()
        except FileNotFoundError:
            logger.warning(
                "ICD-10 reference file not found — "
                "ICD-10 mapping will return empty results."
            )
    return _icd_mapper


def icd10_embedding_available() -> bool:
    """Whether semantic (embedding-based) ICD-10 matching is usable.

    Used by the ``/health`` endpoint to surface degraded-mode state
    rather than leaving it silently indistinguishable from "no match
    found". ``True`` if the mapper hasn't loaded yet (not yet known to
    be broken) or has loaded successfully; ``False`` only after a real
    load attempt has failed.
    """
    mapper = _icd_mapper  # don't trigger a load just to check status
    if mapper is None:
        return True
    return mapper.embedding_available


def _get_classifier():
    """Return the module-level classifier, loading it if needed."""
    global _classifier
    if _classifier is None:
        from src.nlp.classifier import ClinicalClassifier
        clf = ClinicalClassifier()
        try:
            clf.load()
            _classifier = clf
        except Exception as exc:
            logger.warning(
                "Classifier could not be loaded (%s) — "
                "severity prediction will be skipped.", exc
            )
    return _classifier


def warm_up() -> None:
    """Eagerly load the NER pipeline, ICD-10 mapper, and classifier.

    Call this during application startup (see ``src/api/main.py``'s
    lifespan) rather than letting them load lazily on first request.
    Combined model-loading cost is ~110-140s — deferring it means
    whichever user sends the first request eats that entire delay, and
    most HTTP clients (including the dashboard) have a much shorter
    timeout and will report a false "API did not respond" error well
    before loading actually finishes. Paying this cost once at startup
    means no real request is ever the slow one.
    """
    logger.info("Warming up NLP pipeline (NER, ICD-10 mapper, classifier)...")
    t_start = time.perf_counter()

    # Constructing these wrapper objects is cheap -- the actual heavy
    # loading (spaCy's .load(), the sentence-transformer embedding model
    # + 74k-description index) only happens lazily on first real use.
    # Force that now with a real dummy call, rather than just
    # constructing the wrappers and leaving the expensive part for
    # whichever request happens to be first.
    ner = _get_ner()
    if ner:
        ner.extract("Patient has hypertension.")

    mapper = _get_mapper()
    if mapper:
        # Deliberately obscure text guaranteed not to exact- or
        # fuzzy-match any ICD-10 description, forcing the embedding
        # fallback (and therefore the embedding model + index load)
        # to actually run now instead of on the first real request.
        mapper.map("zzqxv nonexistent fictional placeholder condition")

    _get_classifier()

    logger.info("Warm-up complete in %.1fs", time.perf_counter() - t_start)


# ── Analyse ───────────────────────────────────────────────────────

@router.post(
    "/analyse",
    response_model = AnalyseResponse,
    summary        = "Analyse a clinical note",
    description    = (
        "Extract named entities, map to ICD-10 codes, and classify "
        "severity from a pasted clinical note.  Does not persist to "
        "the database — for real-time demo use."
    ),
)
def analyse_note(
    payload: AnalyseRequest,
) -> AnalyseResponse:
    """Analyse a single clinical note in one round trip.

    Runs the NER pipeline, optionally the ICD-10 mapper, and
    optionally the severity classifier.  Results are returned
    immediately and are not stored in the database.

    Args:
        payload: :class:`AnalyseRequest` with the note text and
                 feature flags.

    Returns:
        :class:`AnalyseResponse` with entities, ICD-10 mappings,
        and severity prediction.
    """
    t_start = time.perf_counter()

    # Step 1: clean the input text
    cleaned = prepare_for_inference(payload.text)

    # Step 2: run NER
    ner        = _get_ner()
    raw_entities = ner.extract(cleaned) if ner else []

    # Step 3: ICD-10 mapping (optional)
    icd_results: dict[str, list] = {}
    if payload.include_icd10 and raw_entities:
        mapper = _get_mapper()
        if mapper:
            icd_results = mapper.map_entities(raw_entities)

    # Step 4: severity classification (optional)
    severity_out = None
    if payload.include_severity:
        clf = _get_classifier()
        if clf:
            try:
                result       = clf.predict(cleaned)
                severity_out = SeverityResponse(
                    label         = result.label,
                    confidence    = result.confidence,
                    probabilities = result.probabilities,
                    task          = result.task,
                )
            except Exception as exc:
                logger.warning("Severity prediction failed: %s", exc)

    # Step 5: shape the response
    entity_responses = []
    entity_counts: dict[str, int] = {}

    for ent in raw_entities:
        icd_matches = [
            ICD10MatchResponse(**m.to_dict())
            for m in icd_results.get(ent.text, [])
        ]
        entity_responses.append(EntityResponse(
            text          = ent.text,
            label         = ent.label,
            start         = ent.start,
            end           = ent.end,
            confidence    = ent.confidence,
            icd10_matches = icd_matches,
        ))
        entity_counts[ent.label] = entity_counts.get(ent.label, 0) + 1

    processing_ms = (time.perf_counter() - t_start) * 1000

    return AnalyseResponse(
        text_length    = len(cleaned),
        word_count     = word_count(cleaned),
        entities       = entity_responses,
        entity_counts  = entity_counts,
        severity       = severity_out,
        processing_ms  = round(processing_ms, 1),
    )


# ── List notes ────────────────────────────────────────────────────

@router.get(
    "",
    response_model = NoteListResponse,
    summary        = "List stored clinical notes",
)
def list_notes(
    specialty: str | None = Query(None, description="Filter by specialty"),
    severity:  str | None = Query(None, description="Filter by severity"),
    limit:     int        = Query(50, ge=1, le=500),
    offset:    int        = Query(0, ge=0),
    db: Session           = Depends(get_db_session),
) -> NoteListResponse:
    """Return a paginated list of stored clinical notes.

    Optionally filtered by specialty or severity label.
    """
    if specialty:
        notes = NoteRepository.list_by_specialty(
            db, specialty, limit=limit, offset=offset
        )
    elif severity:
        notes = NoteRepository.list_by_severity(
            db, severity, limit=limit, offset=offset
        )
    else:
        # No filter — return all notes paginated
        from src.db.models import ClinicalNote
        notes = (
            db.query(ClinicalNote)
            .order_by(ClinicalNote.id)
            .offset(offset)
            .limit(limit)
            .all()
        )

    total = NoteRepository.total(db)

    return NoteListResponse(
        total  = total,
        limit  = limit,
        offset = offset,
        items  = [NoteSummary.model_validate(n) for n in notes],
    )


# ── Note detail ───────────────────────────────────────────────────

@router.get(
    "/{note_id}",
    response_model = NoteDetail,
    summary        = "Retrieve a single note",
)
def get_note(
    note_id: int,
    db: Session = Depends(get_db_session),
) -> NoteDetail:
    """Fetch a single stored note by its database ID.

    Args:
        note_id: Integer primary key.

    Raises:
        HTTPException 404: If no note with that ID exists.
    """
    note = NoteRepository.get(db, note_id)
    if note is None:
        raise HTTPException(
            status_code = 404,
            detail      = f"Note {note_id} not found",
        )
    return NoteDetail.model_validate(note)


# ── Aggregate stats ───────────────────────────────────────────────

@router.get(
    "/stats/overview",
    response_model = StatsResponse,
    summary        = "Aggregate statistics for the dashboard",
)
def get_stats(
    db: Session = Depends(get_db_session),
) -> StatsResponse:
    """Return aggregate counts and distributions for the dashboard.

    Used by the overview page to populate summary cards and charts.
    """
    severity_dist = NoteRepository.count_by_severity(db)
    specialty_counts = NoteRepository.count_by_specialty(db)
    top_diseases  = EntityRepository.top_entities(db, label="DISEASE", limit=10)
    top_icd10     = ICD10Repository.top_codes(db, limit=10)

    total_notes = NoteRepository.total(db)
    total_ents  = db.execute(
        __import__("sqlalchemy").text("SELECT COUNT(*) FROM entities")
    ).scalar() or 0

    return StatsResponse(
        total_notes    = total_notes,
        total_entities = total_ents,
        severity_distribution = SeverityDistribution(
            routine  = severity_dist.get("routine",  0),
            urgent   = severity_dist.get("urgent",   0),
            critical = severity_dist.get("critical", 0),
            total    = total_notes,
        ),
        top_specialties = [
            SpecialtyCount(specialty=k, count=v)
            for k, v in list(specialty_counts.items())[:10]
        ],
        top_diseases = [
            TopEntityItem(text=t, count=c) for t, c in top_diseases
        ],
        top_icd10_codes = [
            ICD10MatchResponse(
                icd10_code   = code,
                description  = desc,
                confidence   = 1.0,
                match_method = "lookup",
                rank         = i + 1,
            )
            for i, (code, desc, _) in enumerate(top_icd10)
        ],
    )

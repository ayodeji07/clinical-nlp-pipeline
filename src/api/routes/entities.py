"""
src/api/routes/entities.py
────────────────────────────────────────────────────────────────
Endpoints for entity frequency and co-occurrence data.

Routes
──────
  GET /entities              — list entities for a note
  GET /entities/top          — top-N most frequent entities
  GET /entities/cooccurrence — co-occurrence pairs for the graph
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.schemas import EntitySummary, TopEntitiesResponse, TopEntityItem
from src.db.connection import get_db_session
from src.db.repository import EntityRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/entities", tags=["Entities"])


@router.get(
    "",
    response_model = list[EntitySummary],
    summary        = "List entities for a stored note",
)
def list_entities_for_note(
    note_id: int     = Query(..., description="Note ID to fetch entities for"),
    db: Session      = Depends(get_db_session),
) -> list[EntitySummary]:
    """Return all entities extracted from a specific stored note.

    Entities are ordered by their position in the note text
    (start character offset ascending).

    Args:
        note_id: Primary key of the parent note.

    Returns:
        List of :class:`EntitySummary` objects.
    """
    entities = EntityRepository.list_by_note(db, note_id)
    return [EntitySummary.model_validate(e) for e in entities]


@router.get(
    "/top",
    response_model = TopEntitiesResponse,
    summary        = "Most frequently extracted entities",
)
def top_entities(
    label: str | None = Query(
        None,
        description = "Filter to this entity type (DISEASE, MEDICATION, etc.)",
    ),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db_session),
) -> TopEntitiesResponse:
    """Return the top-N most frequently extracted entity texts.

    Used by the dashboard frequency bar chart.

    Args:
        label: Optional entity type filter.
        limit: Number of top entities to return.

    Returns:
        :class:`TopEntitiesResponse` with ranked entity texts and counts.
    """
    rows = EntityRepository.top_entities(db, label=label, limit=limit)
    return TopEntitiesResponse(
        label = label,
        limit = limit,
        items = [TopEntityItem(text=t, count=c) for t, c in rows],
    )


@router.get(
    "/cooccurrence",
    response_model = list[dict],
    summary        = "Entity co-occurrence pairs",
)
def cooccurrence_pairs(
    label:     str = Query("DISEASE", description="Entity type to analyse"),
    min_count: int = Query(5, ge=1, description="Minimum co-occurrence count"),
    limit:     int = Query(100, ge=1, le=500),
    db: Session    = Depends(get_db_session),
) -> list[dict]:
    """Return entity pairs that frequently co-occur in the same note.

    Used by the dashboard to build the co-occurrence network graph.
    Each item in the response is an edge in the graph with source,
    target, and weight (co-occurrence count).

    Args:
        label:     Entity type — usually DISEASE for clinical insight.
        min_count: Minimum co-occurrences for a pair to be included.
        limit:     Maximum number of pairs to return.

    Returns:
        List of dicts: ``{"source": str, "target": str, "weight": int}``.
    """
    pairs = EntityRepository.cooccurrence_pairs(
        db, label=label, min_count=min_count, limit=limit
    )
    return [
        {"source": a, "target": b, "weight": w}
        for a, b, w in pairs
    ]

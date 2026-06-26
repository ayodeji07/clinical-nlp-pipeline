"""
src/api/routes/icd.py
────────────────────────────────────────────────────────────────
Endpoints for ICD-10 code mapping.

Routes
──────
  POST /icd/lookup   — map a free-text entity to ICD-10 codes
  GET  /icd/top      — most frequently assigned codes in the dataset
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.schemas import ICD10LookupRequest, ICD10LookupResponse, ICD10MatchResponse
from src.db.connection import get_db_session
from src.db.repository import ICD10Repository
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/icd", tags=["ICD-10"])

# Module-level singleton — loaded once, reused per request
_mapper = None


def _get_mapper():
    """Return the ICD-10 mapper, loading it if needed."""
    global _mapper
    if _mapper is None:
        from src.nlp.icd_mapper import ICD10Mapper
        try:
            _mapper = ICD10Mapper()
        except FileNotFoundError as exc:
            logger.warning("ICD-10 mapper unavailable: %s", exc)
    return _mapper


@router.post(
    "/lookup",
    response_model = ICD10LookupResponse,
    summary        = "Map clinical text to ICD-10 codes",
)
def lookup_icd10(payload: ICD10LookupRequest) -> ICD10LookupResponse:
    """Map a free-text clinical entity to ICD-10-CM codes.

    Tries exact → fuzzy → embedding matching in order.
    Returns the top-k candidates with confidence scores.

    Args:
        payload: :class:`ICD10LookupRequest` with the entity text.

    Returns:
        :class:`ICD10LookupResponse` with ranked ICD-10 candidates.
    """
    mapper = _get_mapper()
    if mapper is None:
        return ICD10LookupResponse(query=payload.text, matches=[])

    # Override top_k from the request
    mapper._top_k = payload.top_k
    matches       = mapper.map(payload.text)

    return ICD10LookupResponse(
        query   = payload.text,
        matches = [ICD10MatchResponse(**m.to_dict()) for m in matches],
    )


@router.get(
    "/top",
    response_model = list[ICD10MatchResponse],
    summary        = "Most frequently assigned ICD-10 codes",
)
def top_icd10_codes(
    limit:  int     = 20,
    db: Session     = Depends(get_db_session),
) -> list[ICD10MatchResponse]:
    """Return the most frequently assigned ICD-10 codes in the dataset.

    Used by the dashboard to show which diagnoses dominate the corpus.

    Args:
        limit: Number of top codes to return (default 20).

    Returns:
        List of :class:`ICD10MatchResponse` objects ranked by frequency.
    """
    rows = ICD10Repository.top_codes(db, limit=limit)
    return [
        ICD10MatchResponse(
            icd10_code   = code,
            description  = desc,
            confidence   = 1.0,
            match_method = "lookup",
            rank         = i + 1,
        )
        for i, (code, desc, _) in enumerate(rows)
    ]

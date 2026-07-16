"""
src/api/routes/model.py
────────────────────────────────────────────────────────────────
Endpoints for classifier training run metadata.

Routes
──────
  GET /model/metrics   — metrics for the currently deployed classifier
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import ModelMetricsResponse
from src.db.connection import get_db_session
from src.db.repository import ModelRunRepository
from src.utils.config import ClassifierConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/model", tags=["Model"])


@router.get(
    "/metrics",
    response_model = ModelMetricsResponse,
    summary        = "Metrics for the currently deployed severity classifier",
)
def get_model_metrics(
    task: str       = ClassifierConfig.task,
    db:   Session    = Depends(get_db_session),
) -> ModelMetricsResponse:
    """Return training metrics for the currently deployed classifier run.

    Args:
        task: Classification task to look up (default: the globally
            configured task, currently "severity").

    Returns:
        :class:`ModelMetricsResponse` for the deployed run.

    Raises:
        HTTPException: 404 if no run has been recorded for this task
            yet (e.g. before the first `train_severity_classifier.py`
            run since this feature was added).
    """
    run = ModelRunRepository.get_deployed(db, task)
    if run is None:
        raise HTTPException(
            status_code = 404,
            detail = (
                f"No deployed model run recorded for task={task!r} yet. "
                "Run scripts/train_severity_classifier.py to populate this."
            ),
        )
    return ModelMetricsResponse.model_validate(run)

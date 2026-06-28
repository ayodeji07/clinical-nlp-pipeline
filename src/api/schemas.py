"""
src/api/schemas.py
────────────────────────────────────────────────────────────────
Pydantic models for API request validation and response shaping.

Every route uses these schemas — nothing in the routes builds
or returns raw dicts.  This gives us:
  - Automatic request validation with clear error messages
  - Auto-generated OpenAPI documentation at /docs
  - Type safety between the API and the dashboard client

Naming convention
─────────────────
  *Request  — payload coming IN  (POST body, query params)
  *Response — payload going OUT  (what the client receives)
  *Summary  — a lightweight response used in list endpoints
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ════════════════════════════════════════════════════════════════
# SHARED / BASE
# ════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Response from the health-check endpoint."""

    status:   str = "ok"
    version:  str
    database: str = Field(description="'connected' or 'unreachable'")
    icd10_embedding_available: bool = Field(
        default=True,
        description=(
            "False if semantic ICD-10 matching has failed to load and "
            "is disabled for this process. Exact/fuzzy ICD-10 matching "
            "is unaffected either way — this only flags the fallback "
            "used for entity text that doesn't match any ICD-10 "
            "description lexically."
        ),
    )


# ════════════════════════════════════════════════════════════════
# NOTES
# ════════════════════════════════════════════════════════════════

class AnalyseRequest(BaseModel):
    """Request body for POST /notes/analyse.

    The user pastes or sends a clinical note and receives
    entities, ICD-10 mappings, and a severity prediction
    in one round trip.
    """

    text: str = Field(
        ...,
        min_length  = 10,
        description = "Clinical note text to analyse.",
        examples    = [
            "Patient presents with acute chest pain and shortness "
            "of breath. History of hypertension and diabetes."
        ],
    )
    include_icd10:      bool = Field(
        default     = True,
        description = "Whether to run ICD-10 mapping on extracted entities.",
    )
    include_severity:   bool = Field(
        default     = True,
        description = "Whether to run the severity classifier.",
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        """Reject payloads that are whitespace-only."""
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class EntityResponse(BaseModel):
    """A single extracted named entity."""

    text:         str
    label:        str = Field(
        description="DISEASE, MEDICATION, PROCEDURE, SYMPTOM, or ANATOMY"
    )
    start:        int
    end:          int
    confidence:   Optional[float] = None
    icd10_matches: list[ICD10MatchResponse] = Field(default_factory=list)


class ICD10MatchResponse(BaseModel):
    """One ICD-10 code candidate for an entity."""

    icd10_code:   str
    description:  str
    confidence:   float
    match_method: str = Field(
        description="'exact', 'fuzzy', or 'embedding'"
    )
    rank:         int


class SeverityResponse(BaseModel):
    """Severity classification result for a clinical note."""

    label:         str = Field(
        description="'routine', 'urgent', or 'critical'"
    )
    confidence:    float
    probabilities: dict[str, float]
    task:          str


class AnalyseResponse(BaseModel):
    """Full analysis response for a single clinical note.

    Returned by POST /notes/analyse.
    """

    text_length:    int
    word_count:     int
    entities:       list[EntityResponse]
    entity_counts:  dict[str, int] = Field(
        description="Count of entities per label type"
    )
    severity:       Optional[SeverityResponse] = None
    processing_ms:  Optional[float]            = None


class NoteSummary(BaseModel):
    """Lightweight note record for list responses."""

    id:          int
    specialty:   Optional[str]
    note_type:   Optional[str]
    severity:    Optional[str]
    word_count:  Optional[int]
    data_source: str
    created_at:  datetime

    model_config = {"from_attributes": True}


class NoteDetail(NoteSummary):
    """Full note record including transcription text."""

    transcription: str

    model_config = {"from_attributes": True}


class NoteListResponse(BaseModel):
    """Paginated list of notes."""

    total:  int
    limit:  int
    offset: int
    items:  list[NoteSummary]


# ════════════════════════════════════════════════════════════════
# ENTITIES
# ════════════════════════════════════════════════════════════════

class EntitySummary(BaseModel):
    """Lightweight entity record for list responses."""

    id:         int
    text:       str
    label:      str
    confidence: Optional[float]
    note_id:    int

    model_config = {"from_attributes": True}


class TopEntityItem(BaseModel):
    """One item in a top-N entity frequency list."""

    text:  str
    count: int


class TopEntitiesResponse(BaseModel):
    """Top-N most frequent entities, optionally filtered by label."""

    label:   Optional[str]
    limit:   int
    items:   list[TopEntityItem]


# ════════════════════════════════════════════════════════════════
# ICD-10
# ════════════════════════════════════════════════════════════════

class ICD10LookupRequest(BaseModel):
    """Request body for POST /icd/lookup."""

    text: str = Field(
        ...,
        min_length  = 2,
        description = "Entity text to map to ICD-10 codes.",
        examples    = ["hypertension", "type 2 diabetes mellitus"],
    )
    top_k: int = Field(
        default     = 3,
        ge          = 1,
        le          = 10,
        description = "Maximum number of candidate codes to return.",
    )


class ICD10LookupResponse(BaseModel):
    """ICD-10 mapping result for a single entity text."""

    query:   str
    matches: list[ICD10MatchResponse]


# ════════════════════════════════════════════════════════════════
# STATS / DASHBOARD
# ════════════════════════════════════════════════════════════════

class SeverityDistribution(BaseModel):
    """Count of notes per severity label."""

    routine:  int = 0
    urgent:   int = 0
    critical: int = 0
    total:    int = 0


class SpecialtyCount(BaseModel):
    """Note count for one specialty."""

    specialty: str
    count:     int


class StatsResponse(BaseModel):
    """Aggregate statistics for the dashboard overview page."""

    total_notes:          int
    total_entities:       int
    severity_distribution: SeverityDistribution
    top_specialties:      list[SpecialtyCount]
    top_diseases:         list[TopEntityItem]
    top_icd10_codes:      list[ICD10MatchResponse]


# ════════════════════════════════════════════════════════════════
# MODEL RUNS
# ════════════════════════════════════════════════════════════════

class ModelRunSummary(BaseModel):
    """Summary of a classifier training run."""

    id:               int
    model_name:       str
    task:             str
    val_accuracy:     Optional[float]
    val_f1:           Optional[float]
    test_accuracy:    Optional[float] = None
    test_f1:          Optional[float] = None
    training_samples: Optional[int]
    epochs:           Optional[int]
    is_deployed:      bool
    created_at:       datetime

    model_config = {"from_attributes": True}

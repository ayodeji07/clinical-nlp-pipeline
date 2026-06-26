"""
dashboard/api_client.py
────────────────────────────────────────────────────────────────
Typed HTTP client that the Streamlit pages use to talk to the
FastAPI backend.

Why a dedicated client module
──────────────────────────────
Streamlit pages should not contain raw requests.get() calls —
that scatters URL construction, error handling, and response
parsing across the codebase.  This module is the single place
that knows the API's URL structure.

The pages import functions like get_stats() or analyse_note()
and receive typed dicts back.  If the API changes, only this
file changes.

Configuration
─────────────
Set API_BASE_URL in Streamlit secrets or as an environment
variable.  The default points to a local FastAPI instance for
development.

  [secrets.toml]
  API_BASE_URL = "https://your-api.railway.app"

Error handling
──────────────
All functions return None (or an empty structure) on failure
and log the error rather than raising — this keeps the
dashboard alive even when the API is temporarily unavailable.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests
import streamlit as st

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Base URL ──────────────────────────────────────────────────────
# Read from Streamlit secrets first (production), then environment
# variable, then fall back to localhost for development.
def _base_url() -> str:
    """Return the FastAPI base URL from secrets or environment."""
    try:
        return st.secrets.get("API_BASE_URL", "http://localhost:8000")
    except Exception:
        return os.getenv("API_BASE_URL", "http://localhost:8000")


# Default timeout for all requests (seconds)
_TIMEOUT = 30


# ── Health ────────────────────────────────────────────────────────

def check_health() -> dict[str, Any]:
    """Check whether the API backend is reachable.

    Returns:
        Health response dict, or ``{"status": "unreachable"}`` on error.
    """
    try:
        resp = requests.get(
            f"{_base_url()}/health", timeout=5
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("API health check failed: %s", exc)
        return {"status": "unreachable", "database": "unknown"}


# ── Note analysis ─────────────────────────────────────────────────

def analyse_note(
    text:             str,
    include_icd10:    bool = True,
    include_severity: bool = True,
) -> Optional[dict[str, Any]]:
    """Send a clinical note to the API for full analysis.

    Args:
        text             : Clinical note text to analyse.
        include_icd10    : Whether to run ICD-10 mapping.
        include_severity : Whether to run severity classification.

    Returns:
        AnalyseResponse dict, or None on error.
    """
    try:
        resp = requests.post(
            f"{_base_url()}/notes/analyse",
            json    = {
                "text":             text,
                "include_icd10":    include_icd10,
                "include_severity": include_severity,
            },
            timeout = _TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        logger.error("analyse_note timed out after %ds", _TIMEOUT)
        return None
    except Exception as exc:
        logger.error("analyse_note failed: %s", exc)
        return None


# ── Stats ─────────────────────────────────────────────────────────

def get_stats() -> Optional[dict[str, Any]]:
    """Fetch aggregate statistics for the dashboard overview.

    Returns:
        StatsResponse dict, or None on error.
    """
    try:
        resp = requests.get(
            f"{_base_url()}/notes/stats/overview",
            timeout = _TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("get_stats failed: %s", exc)
        return None


# ── Notes list ────────────────────────────────────────────────────

def list_notes(
    specialty: Optional[str] = None,
    severity:  Optional[str] = None,
    limit:     int            = 50,
    offset:    int            = 0,
) -> Optional[dict[str, Any]]:
    """Fetch a paginated list of stored notes.

    Args:
        specialty : Filter by medical specialty.
        severity  : Filter by severity label.
        limit     : Max records per page.
        offset    : Pagination offset.

    Returns:
        NoteListResponse dict, or None on error.
    """
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if specialty:
        params["specialty"] = specialty
    if severity:
        params["severity"] = severity

    try:
        resp = requests.get(
            f"{_base_url()}/notes",
            params  = params,
            timeout = _TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("list_notes failed: %s", exc)
        return None


# ── Entities ──────────────────────────────────────────────────────

def get_top_entities(
    label: Optional[str] = None,
    limit: int            = 20,
) -> list[dict[str, Any]]:
    """Fetch the most frequently extracted entities.

    Args:
        label : Filter to one entity type (e.g. ``"DISEASE"``).
        limit : Number of top entities to return.

    Returns:
        List of ``{"text": str, "count": int}`` dicts.
    """
    params: dict[str, Any] = {"limit": limit}
    if label:
        params["label"] = label

    try:
        resp = requests.get(
            f"{_base_url()}/entities/top",
            params  = params,
            timeout = _TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])
    except Exception as exc:
        logger.error("get_top_entities failed: %s", exc)
        return []


def get_cooccurrence_pairs(
    label:     str = "DISEASE",
    min_count: int = 5,
    limit:     int = 100,
) -> list[dict[str, Any]]:
    """Fetch entity co-occurrence pairs for the network graph.

    Args:
        label     : Entity type to analyse.
        min_count : Minimum co-occurrence count for inclusion.
        limit     : Maximum pairs to return.

    Returns:
        List of ``{"source": str, "target": str, "weight": int}`` dicts.
    """
    try:
        resp = requests.get(
            f"{_base_url()}/entities/cooccurrence",
            params  = {
                "label":     label,
                "min_count": min_count,
                "limit":     limit,
            },
            timeout = _TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("get_cooccurrence_pairs failed: %s", exc)
        return []


# ── ICD-10 ────────────────────────────────────────────────────────

def lookup_icd10(text: str, top_k: int = 3) -> Optional[dict[str, Any]]:
    """Map a free-text entity to ICD-10 candidates.

    Args:
        text  : Entity text to map (e.g. ``"hypertension"``).
        top_k : Number of candidate codes to return.

    Returns:
        ICD10LookupResponse dict, or None on error.
    """
    try:
        resp = requests.post(
            f"{_base_url()}/icd/lookup",
            json    = {"text": text, "top_k": top_k},
            timeout = _TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("lookup_icd10 failed: %s", exc)
        return None

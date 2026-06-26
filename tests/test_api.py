"""
tests/test_api.py
────────────────────────────────────────────────────────────────
Integration tests for the FastAPI application.

Uses FastAPI's TestClient (which runs requests in-process, no
server needed) with an in-memory SQLite database so tests are
fully isolated and fast.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.db.connection import _reset_for_testing, create_all_tables


@pytest.fixture(scope="module", autouse=True)
def test_database():
    """Set up an isolated in-memory SQLite database for the test module."""
    _reset_for_testing("sqlite:///:memory:")
    create_all_tables()
    yield
    _reset_for_testing("sqlite:///:memory:")


@pytest.fixture(scope="module")
def client():
    """Return a TestClient for the FastAPI app."""
    from src.api.main import app
    return TestClient(app, raise_server_exceptions=True)


# ── Health check ──────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"]  == "ok"
        assert "version"  in data
        assert "database" in data

    def test_root_redirects_to_docs(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert "/docs" in resp.headers.get("location", "")


# ── Notes list ────────────────────────────────────────────────────

class TestNotesList:

    def test_list_notes_returns_paginated_response(self, client):
        resp = client.get("/notes")
        assert resp.status_code == 200
        data = resp.json()
        assert "total"  in data
        assert "items"  in data
        assert "limit"  in data
        assert "offset" in data

    def test_list_notes_default_pagination(self, client):
        resp = client.get("/notes")
        data = resp.json()
        assert data["limit"]  == 50
        assert data["offset"] == 0

    def test_list_notes_invalid_limit_rejected(self, client):
        resp = client.get("/notes?limit=0")
        assert resp.status_code == 422

    def test_note_not_found_returns_404(self, client):
        resp = client.get("/notes/99999")
        assert resp.status_code == 404


# ── Note analysis ─────────────────────────────────────────────────

class TestAnalyse:

    def _mock_ner_pipeline(self):
        from src.nlp.ner import Entity
        mock = MagicMock()
        mock.extract.return_value = [
            Entity(text="hypertension", label="DISEASE",
                   start=10, end=22, confidence=None),
            Entity(text="metformin", label="MEDICATION",
                   start=30, end=39, confidence=None),
        ]
        return mock

    def test_analyse_returns_expected_shape(self, client):
        with patch("src.api.routes.notes._get_ner",
                   return_value=self._mock_ner_pipeline()),              patch("src.api.routes.notes._get_mapper",  return_value=None),              patch("src.api.routes.notes._get_classifier", return_value=None):

            resp = client.post("/notes/analyse", json={
                "text": "Patient has hypertension and takes metformin.",
                "include_icd10":    False,
                "include_severity": False,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert "entities"       in data
        assert "entity_counts"  in data
        assert "word_count"     in data
        assert "text_length"    in data

    def test_analyse_short_text_rejected(self, client):
        resp = client.post("/notes/analyse", json={"text": "hi"})
        assert resp.status_code == 422

    def test_analyse_blank_text_rejected(self, client):
        resp = client.post("/notes/analyse", json={"text": "    "})
        assert resp.status_code == 422

    def test_analyse_entity_counts_match_entities(self, client):
        with patch("src.api.routes.notes._get_ner",
                   return_value=self._mock_ner_pipeline()),              patch("src.api.routes.notes._get_mapper",     return_value=None),              patch("src.api.routes.notes._get_classifier", return_value=None):

            resp = client.post("/notes/analyse", json={
                "text": "Patient has hypertension and takes metformin daily.",
                "include_icd10":    False,
                "include_severity": False,
            })

        data   = resp.json()
        counts = data["entity_counts"]
        total  = sum(counts.values())
        assert total == len(data["entities"])


# ── ICD-10 lookup ─────────────────────────────────────────────────

class TestICD10:

    def test_icd_lookup_missing_mapper_returns_empty(self, client):
        with patch("src.api.routes.icd._get_mapper", return_value=None):
            resp = client.post("/icd/lookup", json={
                "text": "hypertension", "top_k": 3
            })
        assert resp.status_code == 200
        assert resp.json()["matches"] == []

    def test_icd_lookup_short_text_rejected(self, client):
        resp = client.post("/icd/lookup", json={"text": "x"})
        assert resp.status_code == 422

    def test_icd_top_returns_list(self, client):
        resp = client.get("/icd/top")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ── Entities ──────────────────────────────────────────────────────

class TestEntities:

    def test_top_entities_returns_expected_shape(self, client):
        resp = client.get("/entities/top")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "limit" in data

    def test_top_entities_label_filter_accepted(self, client):
        resp = client.get("/entities/top?label=DISEASE&limit=10")
        assert resp.status_code == 200

    def test_cooccurrence_returns_list(self, client):
        resp = client.get("/entities/cooccurrence")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

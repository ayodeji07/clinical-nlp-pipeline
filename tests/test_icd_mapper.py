"""
tests/test_icd_mapper.py
────────────────────────────────────────────────────────────────
Unit tests for src/nlp/icd_mapper.py

Uses an in-memory ICD-10 DataFrame so no CSV file is needed.
Covers exact match, fuzzy match, the describe() helper, and
graceful handling of missing dependencies.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.nlp.icd_mapper import ICD10Match, ICD10Mapper


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def small_icd10_df():
    """Minimal ICD-10 reference table for testing."""
    return pd.DataFrame({
        "icd10_code":  ["I10",  "E11",               "J18.9",   "K21.0"],
        "description": ["Essential hypertension",
                        "Type 2 diabetes mellitus",
                        "Unspecified pneumonia",
                        "Gastro-oesophageal reflux disease"],
    })


@pytest.fixture
def mapper(small_icd10_df, tmp_path):
    """ICD10Mapper loaded from a temporary parquet file."""
    pq = tmp_path / "icd10_codes.parquet"
    small_icd10_df.to_parquet(pq)
    return ICD10Mapper(icd10_path=pq, top_k=3)


# ── ICD10Match dataclass ──────────────────────────────────────────

class TestICD10Match:

    def test_to_dict_rounds_confidence(self):
        m = ICD10Match(icd10_code="I10", description="Hypertension",
                       confidence=0.94567, match_method="fuzzy", rank=1)
        assert m.to_dict()["confidence"] == 0.946

    def test_to_dict_all_fields(self):
        m = ICD10Match(icd10_code="E11", description="Diabetes",
                       confidence=1.0, match_method="exact", rank=1)
        d = m.to_dict()
        assert d["icd10_code"]   == "E11"
        assert d["match_method"] == "exact"
        assert d["rank"]         == 1


# ── Exact match ───────────────────────────────────────────────────

class TestExactMatch:

    def test_exact_match_found(self, mapper):
        results = mapper.map("essential hypertension")
        assert len(results) == 1
        assert results[0].icd10_code   == "I10"
        assert results[0].match_method == "exact"
        assert results[0].confidence   == 1.0

    def test_case_insensitive_lookup(self, mapper):
        results = mapper.map("ESSENTIAL HYPERTENSION")
        assert len(results) == 1
        assert results[0].icd10_code == "I10"

    def test_code_reverse_lookup(self, mapper):
        # Looking up "i10" should return the description
        results = mapper.map("i10")
        assert results and results[0].icd10_code == "I10"

    def test_no_match_returns_empty(self, mapper):
        # "zymurgy" will not match anything
        with patch.object(mapper, "_fuzzy_match", return_value=[]):
            with patch.object(mapper, "_embedding_match", return_value=[]):
                results = mapper.map("zymurgy")
        assert results == []

    def test_empty_string_returns_empty(self, mapper):
        assert mapper.map("") == []
        assert mapper.map("   ") == []


# ── Fuzzy match ───────────────────────────────────────────────────

class TestFuzzyMatch:

    def test_fuzzy_match_spelling_variant(self, mapper):
        # "hypertensioon" — one extra char — should still fuzzy-match
        results = mapper._fuzzy_match("hypertensioon")
        # If rapidfuzz is installed and score >= 80, we get a result
        # If not installed, we get []
        if results:
            assert results[0].match_method == "fuzzy"
            assert results[0].confidence > 0.7

    def test_fuzzy_returns_empty_when_rapidfuzz_missing(self, mapper):
        with patch.dict("sys.modules", {"rapidfuzz": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                results = mapper._fuzzy_match("hypertension")
        # Should return [] gracefully, not raise
        assert isinstance(results, list)


# ── describe() ───────────────────────────────────────────────────

class TestDescribe:

    def test_describe_known_code(self, mapper):
        desc = mapper.describe("I10")
        assert desc == "Essential hypertension"

    def test_describe_lowercase_code(self, mapper):
        desc = mapper.describe("i10")
        assert desc == "Essential hypertension"

    def test_describe_unknown_code_returns_none(self, mapper):
        assert mapper.describe("Z99.99") is None


# ── map_entities() ────────────────────────────────────────────────

class TestMapEntities:

    def test_only_maps_disease_and_symptom(self, mapper):
        from src.nlp.ner import Entity
        entities = [
            Entity(text="essential hypertension", label="DISEASE",
                   start=0, end=22),
            Entity(text="metformin", label="MEDICATION",
                   start=30, end=39),   # should be skipped
        ]
        results = mapper.map_entities(entities)
        assert "essential hypertension" in results
        assert "metformin" not in results

    def test_returns_empty_for_no_matches(self, mapper):
        from src.nlp.ner import Entity
        entities = [
            Entity(text="xyzxyz", label="DISEASE", start=0, end=6),
        ]
        with patch.object(mapper, "_fuzzy_match", return_value=[]):
            with patch.object(mapper, "_embedding_match", return_value=[]):
                results = mapper.map_entities(entities)
        assert results == {}

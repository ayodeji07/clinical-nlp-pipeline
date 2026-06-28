"""
tests/test_ner.py
────────────────────────────────────────────────────────────────
Unit tests for src/nlp/ner.py

Covers: Entity dataclass, label normalisation, SpacyNERPipeline
interface (mocked model), factory function, edge cases.

The scispaCy model is never loaded — spacy.load() is mocked so
the suite runs in seconds without the 500 MB model file.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.nlp.ner import (
    BaseNERPipeline,
    Entity,
    HybridNERPipeline,
    SpacyNERPipeline,
    _normalise_label,
    build_ner_pipeline,
)


class TestEntity:
    """Entity dataclass serialisation."""

    def test_to_dict_all_fields(self):
        ent = Entity(text="hypertension", label="DISEASE",
                     start=10, end=22, confidence=0.95, note_id=7)
        d = ent.to_dict()
        assert d["text"]       == "hypertension"
        assert d["label"]      == "DISEASE"
        assert d["start"]      == 10
        assert d["end"]        == 22
        assert d["confidence"] == 0.95
        assert d["note_id"]    == 7

    def test_to_dict_none_confidence(self):
        ent = Entity(text="pain", label="SYMPTOM", start=0, end=4)
        assert ent.to_dict()["confidence"] is None

    def test_to_dict_rounds_confidence(self):
        ent = Entity(text="aspirin", label="MEDICATION",
                     start=0, end=7, confidence=0.94567)
        assert ent.to_dict()["confidence"] == 0.946

    def test_note_id_defaults_to_none(self):
        ent = Entity(text="fever", label="SYMPTOM", start=0, end=5)
        assert ent.note_id is None


class TestNormaliseLabel:
    """_normalise_label() mapping logic."""

    @pytest.mark.parametrize("raw,expected", [
        ("DISEASE",   "DISEASE"),
        ("CHEMICAL",  "MEDICATION"),
        ("DRUG",      "MEDICATION"),
        ("ANATOMY",   "ANATOMY"),
        ("ORGAN",     "ANATOMY"),
        ("PROCEDURE", "PROCEDURE"),
    ])
    def test_direct_label_map(self, raw, expected):
        assert _normalise_label(raw, "entity") == expected

    def test_procedure_keyword_match(self):
        assert _normalise_label("UNKNOWN", "knee surgery")    == "PROCEDURE"
        assert _normalise_label("UNKNOWN", "biopsy of liver") == "PROCEDURE"
        assert _normalise_label("UNKNOWN", "CT scan")         == "PROCEDURE"

    def test_anatomy_keyword_match(self):
        assert _normalise_label("UNKNOWN", "left ventricle") == "ANATOMY"
        assert _normalise_label("UNKNOWN", "right kidney")   == "ANATOMY"

    def test_default_to_symptom(self):
        assert _normalise_label("UNKNOWN", "fatigue") == "SYMPTOM"
        assert _normalise_label("XYZ",     "nausea")  == "SYMPTOM"

    def test_case_insensitive_raw_label(self):
        # "pain" is deliberately excluded here -- it's in the SYMPTOM
        # fast-path regardless of raw label (bc5cdr over-tags things like
        # "knee pain"/"cough" as DISEASE; gold-standard review confirmed
        # the fast-path must win). Use a term with no fast-path/keyword
        # collision so this only tests raw-label case-insensitivity.
        assert _normalise_label("disease",  "hypertension") == "DISEASE"
        assert _normalise_label("Chemical", "aspirin")      == "MEDICATION"

    def test_procedure_before_anatomy(self):
        # "heart surgery" — surgery keyword wins over heart (anatomy)
        assert _normalise_label("UNKNOWN", "heart surgery") == "PROCEDURE"


def _mock_spacy(ents):
    """Return a minimal spaCy model mock producing the given entities."""
    mock_doc  = MagicMock()
    mock_ents = []
    for text, start, end, label in ents:
        e = MagicMock()
        e.text       = text
        e.start_char = start
        e.end_char   = end
        e.label_     = label
        mock_ents.append(e)
    mock_doc.ents   = mock_ents
    mock_nlp        = MagicMock()
    mock_nlp.return_value = mock_doc
    mock_nlp.pipe   = MagicMock(return_value=iter([mock_doc]))
    return mock_nlp


class TestSpacyNERPipeline:
    """SpacyNERPipeline with mocked model."""

    def test_extract_returns_entities(self):
        pipeline      = SpacyNERPipeline()
        pipeline._nlp = _mock_spacy([
            ("hypertension", 10, 22, "DISEASE"),
            ("metformin",    30, 39, "CHEMICAL"),
        ])
        ents = pipeline.extract("Patient has hypertension. Takes metformin.")
        assert len(ents) == 2
        assert ents[0].label == "DISEASE"
        assert ents[1].label == "MEDICATION"

    def test_extract_sorts_by_start(self):
        pipeline      = SpacyNERPipeline()
        pipeline._nlp = _mock_spacy([
            ("diabetes", 30, 38, "DISEASE"),
            ("aspirin",   5, 12, "CHEMICAL"),
        ])
        ents = pipeline.extract("Takes aspirin. Has diabetes.")
        assert ents[0].start < ents[1].start

    def test_extract_skips_short_tokens(self):
        pipeline      = SpacyNERPipeline()
        pipeline._nlp = _mock_spacy([
            ("BP", 0, 2, "DISEASE"),
            ("hypertension", 10, 22, "DISEASE"),
        ])
        ents = pipeline.extract("BP 140. Hypertension noted.")
        assert len(ents) == 1
        assert ents[0].text == "hypertension"

    def test_extract_empty_text(self):
        pipeline      = SpacyNERPipeline()
        pipeline._nlp = MagicMock()
        assert pipeline.extract("") == []
        assert pipeline.extract("   ") == []
        pipeline._nlp.assert_not_called()

    def test_extract_batch_uses_pipe(self):
        pipeline      = SpacyNERPipeline()
        pipeline._nlp = _mock_spacy([("hypertension", 0, 12, "DISEASE")])
        pipeline.extract_batch(["Note one."], batch_size=8)
        pipeline._nlp.pipe.assert_called_once()

    def test_extract_batch_empty_input(self):
        pipeline      = SpacyNERPipeline()
        pipeline._nlp = MagicMock()
        assert pipeline.extract_batch([]) == []

    def test_load_error_raises_oserror(self):
        pipeline = SpacyNERPipeline(model_name="nonexistent")
        with patch("spacy.load", side_effect=OSError("not found")):
            with pytest.raises(OSError, match="not installed"):
                pipeline._load_model()

    def test_is_subclass_of_base(self):
        assert issubclass(SpacyNERPipeline, BaseNERPipeline)


class TestBuildNERPipeline:
    """build_ner_pipeline() factory."""

    def test_returns_base_pipeline(self):
        assert isinstance(build_ner_pipeline(model_name="en_core_sci_lg"),
                          BaseNERPipeline)

    def test_uses_config_default(self):
        # Default is "hybrid" (HybridNERPipeline has no single _model_name --
        # it wraps two SpacyNERPipeline instances under _fine/_broad).
        from src.utils.config import ModelConfig
        pipeline = build_ner_pipeline()
        if ModelConfig.ner_model == "hybrid":
            assert isinstance(pipeline, HybridNERPipeline)
            assert pipeline.model_name.startswith("hybrid(")
        else:
            assert pipeline._model_name == ModelConfig.ner_model

    def test_custom_model_name(self):
        pipeline = build_ner_pipeline(model_name="en_ner_bc5cdr_md")
        assert pipeline._model_name == "en_ner_bc5cdr_md"

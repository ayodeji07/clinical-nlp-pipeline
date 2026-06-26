"""
tests/test_classifier.py
────────────────────────────────────────────────────────────────
Unit tests for src/nlp/classifier.py

The HuggingFace model and tokenizer are mocked throughout so
the test suite runs without GPU or internet access.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.nlp.classifier import ClassificationResult, ClinicalClassifier


class TestClassificationResult:
    """ClassificationResult serialisation."""

    def test_to_dict_rounds_values(self):
        result = ClassificationResult(
            label         = "urgent",
            confidence    = 0.845678,
            probabilities = {"routine": 0.1, "urgent": 0.845678, "critical": 0.05},
            task          = "severity",
        )
        d = result.to_dict()
        assert d["label"]      == "urgent"
        assert d["confidence"] == 0.846
        assert d["task"]       == "severity"
        assert all(isinstance(v, float) for v in d["probabilities"].values())

    def test_repr(self):
        result = ClassificationResult("routine", 0.9, {}, "severity")
        assert "routine" in repr(result)
        assert "0.900" in repr(result)


class TestClinicalClassifier:
    """ClinicalClassifier with mocked transformers."""

    def _make_classifier(self, task="severity"):
        clf = ClinicalClassifier(task=task)
        return clf

    def test_init_sets_labels_from_config(self):
        clf = self._make_classifier(task="severity")
        assert clf._labels == ["routine", "urgent", "critical"]
        assert clf._label2id["routine"] == 0
        assert clf._id2label[2]         == "critical"

    def test_init_readmission_task(self):
        clf = self._make_classifier(task="readmission")
        assert clf._labels == ["not_readmitted", "readmitted"]

    def test_unknown_task_raises_on_labels(self):
        from src.utils.config import ClassifierConfig
        original = ClassifierConfig.task
        ClassifierConfig.task = "unknown_task"
        try:
            with pytest.raises(ValueError, match="Unknown classifier task"):
                ClassifierConfig.active_labels()
        finally:
            ClassifierConfig.task = original

    def test_predict_raises_if_not_loaded(self):
        clf = self._make_classifier()
        with pytest.raises(RuntimeError, match="not loaded"):
            clf.predict("Some clinical text.")

    def test_predict_returns_classification_result(self):
        """predict() with mocked model returns a ClassificationResult."""
        import torch
        import numpy as np

        clf = self._make_classifier(task="severity")

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_encoding  = {
            "input_ids":      torch.zeros(1, 512, dtype=torch.long),
            "attention_mask": torch.ones(1, 512, dtype=torch.long),
        }
        mock_tokenizer.return_value = mock_encoding
        clf._tokenizer = mock_tokenizer

        # Mock model — returns logits [critical, urgent, routine] scores
        mock_model  = MagicMock()
        mock_output = MagicMock()
        # logits: critical has highest score
        mock_output.logits = torch.tensor([[0.1, 0.2, 3.5]])
        mock_model.return_value = mock_output
        mock_model.parameters   = MagicMock(
            return_value=iter([torch.zeros(1)])
        )
        clf._model = mock_model

        result = clf.predict("Patient in ICU following cardiac arrest.")
        assert isinstance(result, ClassificationResult)
        assert result.task == "severity"
        assert result.label in ["routine", "urgent", "critical"]
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_batch_empty_list(self):
        clf = self._make_classifier()
        assert clf.predict_batch([]) == []

    def test_save_raises_if_not_loaded(self):
        clf = self._make_classifier()
        with pytest.raises(RuntimeError, match="No model to save"):
            clf.save()

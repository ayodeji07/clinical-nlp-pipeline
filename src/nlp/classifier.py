"""
src/nlp/classifier.py
────────────────────────────────────────────────────────────────
Clinical text classifier — fine-tunes Bio_ClinicalBERT for
severity classification (routine / urgent / critical).

Design
──────
The classifier is task-agnostic: the ``task`` config parameter
controls which label set is used.  Swapping to readmission risk
(Phase 2) is a one-line change in config.py.

Two modes of operation:
  Training  — fine-tunes the base model on MTSamples with
              weak-supervision severity labels.  Saves the
              fine-tuned weights to data/models/.
  Inference — loads the fine-tuned weights and classifies
              a single note or a batch.

Training takes ~20 minutes on CPU for MTSamples (~4,000 notes
after filtering).  On a free Colab GPU it takes under 5 minutes.
The fine-tuned model is saved and reused on subsequent runs.

The training loop is written from scratch using the HuggingFace
Trainer API.  This is more explicit than using a Pipeline object
and gives full control over evaluation and checkpoint saving.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import (
    ClassifierConfig,
    ModelConfig,
    TrainingConfig,
    force_offline_hf_env,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _from_pretrained_offline_first(loader_cls, model_name_or_path, **kwargs):
    """Load a HuggingFace tokenizer/model from the local cache first.

    HuggingFace libraries make a handful of network round-trips (HEAD
    requests checking for config/adapter file updates) even when a
    model is fully cached locally — a transient network blip during
    one of those checks crashes the whole load. Trying
    ``local_files_only=True`` first avoids the network entirely when
    the model is already cached, and only falls back to a live
    download when nothing is cached yet.

    Args:
        loader_cls: A HuggingFace ``...from_pretrained`` class, e.g.
            ``AutoTokenizer`` or ``AutoModelForSequenceClassification``.
        model_name_or_path: HuggingFace Hub model ID or local path.
        **kwargs: Forwarded to ``from_pretrained``.
    """
    try:
        with force_offline_hf_env():
            return loader_cls.from_pretrained(
                model_name_or_path, local_files_only=True, **kwargs
            )
    except Exception:
        logger.info(
            "%s not found in local cache — downloading: %s",
            loader_cls.__name__, model_name_or_path,
        )
        return loader_cls.from_pretrained(model_name_or_path, **kwargs)


# ── Output dataclass ──────────────────────────────────────────────

class ClassificationResult:
    """The output of classifying a single clinical note.

    Attributes:
        label       : Predicted class label (e.g. ``"urgent"``).
        confidence  : Softmax probability of the predicted class.
        probabilities: Full probability distribution over all classes.
        task        : Which classification task produced this result.
    """

    def __init__(
        self,
        label:         str,
        confidence:    float,
        probabilities: dict[str, float],
        task:          str = ClassifierConfig.task,
    ) -> None:
        self.label         = label
        self.confidence    = confidence
        self.probabilities = probabilities
        self.task          = task

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary for JSON responses."""
        return {
            "label":         self.label,
            "confidence":    round(self.confidence, 3),
            "probabilities": {
                k: round(v, 3) for k, v in self.probabilities.items()
            },
            "task": self.task,
        }

    def __repr__(self) -> str:
        return (
            f"ClassificationResult(label={self.label!r}, "
            f"confidence={self.confidence:.3f})"
        )


# ── Dataset ───────────────────────────────────────────────────────

class ClinicalNoteDataset:
    """PyTorch Dataset wrapper for tokenised clinical notes.

    Handles tokenisation and encoding in __getitem__ rather than
    upfront, which keeps memory usage manageable for large datasets.

    Args:
        texts     : List of clinical note strings.
        labels    : Integer label indices parallel to texts.
        tokenizer : HuggingFace tokenizer instance.
        max_length: Maximum token length (default 512 for BERT).
    """

    def __init__(
        self,
        texts:      list[str],
        labels:     list[int],
        tokenizer,
        max_length: int = TrainingConfig.max_length,
    ) -> None:
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """Return tokenised input for one example.

        Args:
            idx: Index into the dataset.

        Returns:
            Dict with ``input_ids``, ``attention_mask``,
            ``token_type_ids``, and ``labels`` tensors.
        """
        import torch

        encoding = self.tokenizer(
            self.texts[idx],
            max_length      = self.max_length,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Classifier ────────────────────────────────────────────────────

class ClinicalClassifier:
    """Fine-tune and serve Bio_ClinicalBERT for clinical text classification.

    Args:
        task        : Classification task.  One of ``"severity"``
                      (Phase 1) or ``"readmission"`` (Phase 2).
        model_name  : HuggingFace model ID for the base model.
        output_dir  : Where to save fine-tuned weights.

    Example::

        # Train
        clf = ClinicalClassifier(task="severity")
        clf.train(train_df)

        # Infer
        clf = ClinicalClassifier(task="severity")
        clf.load()
        result = clf.predict("Patient admitted to ICU following cardiac arrest.")
        print(result.label, result.confidence)
        # → critical  0.94
    """

    def __init__(
        self,
        task:       str            = ClassifierConfig.task,
        model_name: str            = ModelConfig.classifier_model,
        output_dir: Optional[Path] = None,
    ) -> None:
        self._task       = task
        self._model_name = model_name
        self._output_dir = output_dir or ModelConfig.fine_tuned_dir
        self._labels     = ClassifierConfig.active_labels(self._task)
        self._label2id   = {lbl: i for i, lbl in enumerate(self._labels)}
        self._id2label   = {i: lbl for i, lbl in enumerate(self._labels)}
        self._model      = None
        self._tokenizer  = None

        logger.info(
            "ClinicalClassifier init: task=%s, model=%s, labels=%s",
            self._task, self._model_name, self._labels,
        )

    # ── Model I/O ─────────────────────────────────────────────────

    def _load_tokenizer(self):
        """Load the tokenizer, downloading it if necessary."""
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer
        logger.info("Loading tokenizer: %s", self._model_name)
        self._tokenizer = _from_pretrained_offline_first(
            AutoTokenizer, self._model_name
        )
        return self._tokenizer

    def load(self) -> None:
        """Load fine-tuned weights, preferring local disk over HF Hub
        over the untrained base model, in that order.

        Loads from ``output_dir`` if it contains a saved checkpoint.
        Otherwise falls back to ``ModelConfig.classifier_hf_hub_fallback``
        — a copy of the fine-tuned weights on HuggingFace Hub, needed
        because a fresh deployment container won't have ``output_dir``
        locally (the weights are too large to commit to git). Only if
        *that* also fails does this fall back to the untrained base
        ``model_name`` — silently doing so without trying the HF Hub
        copy first would produce confident-looking but meaningless
        severity predictions with no error raised.

        Call this before predict() if you have already trained.

        Raises:
            FileNotFoundError: If no source has a valid model and
                HuggingFace cannot fetch the base either.
        """
        from transformers import AutoModelForSequenceClassification

        if self._output_dir.exists():
            source = str(self._output_dir)
        elif ModelConfig.classifier_hf_hub_fallback:
            source = ModelConfig.classifier_hf_hub_fallback
            logger.warning(
                "No local checkpoint at %s — loading fine-tuned weights "
                "from HF Hub fallback (%s) instead of the untrained "
                "base model.", self._output_dir, source,
            )
        else:
            source = self._model_name
            logger.warning(
                "No local checkpoint and no HF Hub fallback configured "
                "— loading UNTRAINED base model %s. Severity "
                "predictions will be meaningless.", source,
            )
        logger.info("Loading classifier from: %s", source)

        self._model = _from_pretrained_offline_first(
            AutoModelForSequenceClassification,
            source,
            num_labels = len(self._labels),
            id2label   = self._id2label,
            label2id   = self._label2id,
        )
        self._model.eval()
        self._load_tokenizer()
        logger.info("Classifier loaded OK")

    def save(self) -> None:
        """Save the fine-tuned model and tokenizer to disk.

        Also saves a label mapping JSON so the model can be reloaded
        without the original config.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No model to save — train or load first.")

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(self._output_dir)
        self._tokenizer.save_pretrained(self._output_dir)

        # Save label mapping for reloading without config
        label_map_path = self._output_dir / "label_map.json"
        label_map_path.write_text(
            json.dumps({"id2label": self._id2label, "task": self._task},
                       indent=2)
        )
        logger.info("Model saved to %s", self._output_dir)

    # ── Training ──────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        resume: bool = True,
        weighted_loss: bool = False,
    ) -> dict[str, float]:
        """Fine-tune the base model on labelled clinical notes.

        Splits data into train / validation / test sets, trains
        for the configured number of epochs, evaluates on the
        validation set after each epoch, and saves the best
        checkpoint.

        Args:
            df: DataFrame with ``transcription`` and ``severity``
                columns (or ``readmission`` for Phase 2).
                Run the ETL pipeline first to produce this.
            resume: If ``True`` (default) and a previously fine-tuned
                checkpoint already exists at ``output_dir``, continue
                training from those weights instead of the base
                pretrained model. This is a warm start, not a true
                resume — optimizer/scheduler state and epoch count
                are not restored, so the LR schedule still runs the
                full configured epoch count, but a re-run after an
                interrupted session does not throw away prior
                fine-tuning progress. Pass ``False`` to force training
                from the base model regardless of any existing checkpoint.
            weighted_loss: If ``True``, weight the cross-entropy loss
                inversely to each class's frequency in the training
                split (``n_samples / (n_classes * n_samples_c)``, the
                standard "balanced" formula). Use this when one class
                is under-represented (e.g. "critical" at ~15% of
                severity data) and the model is under-predicting it.

        Returns:
            Dict with final evaluation metrics: ``val_accuracy``,
            ``val_f1``, ``test_accuracy``, ``test_f1``, and
            ``test_report`` (a full per-class precision/recall/F1
            breakdown from ``sklearn.metrics.classification_report``).

        Raises:
            KeyError: If the required columns are not present.
        """
        label_col = self._task
        if label_col not in df.columns:
            raise KeyError(
                f"Column '{label_col}' not found. "
                f"Available: {list(df.columns)}"
            )

        logger.info("Starting fine-tuning: task=%s", self._task)

        import torch
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from transformers import (
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )

        tokenizer = self._load_tokenizer()
        cfg       = TrainingConfig

        # ── Encode labels ─────────────────────────────────────────
        valid_mask = df[label_col].isin(self._labels)
        df         = df[valid_mask].copy()

        texts  = df["transcription"].tolist()
        labels = [self._label2id[lbl] for lbl in df[label_col]]

        logger.info(
            "Training data: %d notes, label dist=%s",
            len(texts),
            {lbl: labels.count(i) for lbl, i in self._label2id.items()},
        )

        # ── Train / val / test split ──────────────────────────────
        # First cut off the test set, then split remaining into train/val
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            texts, labels,
            test_size    = cfg.test_split,
            random_state = cfg.random_seed,
            stratify     = labels,
        )
        val_ratio = cfg.val_split / (1 - cfg.test_split)
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval, y_trainval,
            test_size    = val_ratio,
            random_state = cfg.random_seed,
            stratify     = y_trainval,
        )

        logger.info(
            "Split: train=%d  val=%d  test=%d",
            len(x_train), len(x_val), len(x_test),
        )

        # ── Datasets and loaders ──────────────────────────────────
        train_ds = ClinicalNoteDataset(x_train, y_train, tokenizer)
        val_ds   = ClinicalNoteDataset(x_val,   y_val,   tokenizer)
        test_ds  = ClinicalNoteDataset(x_test,  y_test,  tokenizer)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size)

        # ── Model ─────────────────────────────────────────────────
        # Warm-start from an existing checkpoint if one is present —
        # avoids discarding prior fine-tuning progress on a re-run
        # after an interrupted/restarted session.
        checkpoint_exists = (self._output_dir / "config.json").exists()
        resuming = resume and checkpoint_exists
        model_source = str(self._output_dir) if resuming else self._model_name

        if resume and not checkpoint_exists:
            logger.info(
                "No existing checkpoint at %s — training from base model %s",
                self._output_dir, self._model_name,
            )
        elif resuming:
            logger.info("Resuming fine-tuning from checkpoint: %s", self._output_dir)

        self._model = _from_pretrained_offline_first(
            AutoModelForSequenceClassification,
            model_source,
            num_labels = len(self._labels),
            id2label   = self._id2label,
            label2id   = self._label2id,
        )

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        if device.type == "cpu":
            torch.set_num_threads(os.cpu_count())

        logger.info(
            "Training on device: %s (%d threads)",
            device, torch.get_num_threads(),
        )

        total_steps  = len(train_loader) * cfg.epochs
        warmup_steps = total_steps // 10   # 10% warmup

        optimizer = AdamW(self._model.parameters(), lr=cfg.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = warmup_steps,
            num_training_steps = total_steps,
        )

        # ── Class weights ─────────────────────────────────────────
        # Balanced formula: n_samples / (n_classes * n_samples_c).
        # Computed from the actual training split, not the full
        # dataset, since that's the distribution the model sees.
        loss_weight = None
        if weighted_loss:
            n_train, n_classes = len(y_train), len(self._labels)
            counts = [max(y_train.count(i), 1) for i in range(n_classes)]
            weights = [n_train / (n_classes * c) for c in counts]
            loss_weight = torch.tensor(weights, dtype=torch.float32, device=device)
            logger.info(
                "Weighted loss enabled: %s",
                dict(zip(self._labels, (round(w, 3) for w in weights))),
            )

        # ── Training loop ─────────────────────────────────────────
        # When resuming, evaluate the loaded checkpoint first so a
        # worse early epoch doesn't overwrite a better prior result.
        if resuming:
            baseline = self._evaluate(val_loader, device)
            best_val_f1, best_val_acc = baseline["f1"], baseline["accuracy"]
            logger.info(
                "Resumed checkpoint baseline: val_acc=%.4f  val_f1=%.4f",
                best_val_acc, best_val_f1,
            )
        else:
            best_val_f1   = 0.0
            best_val_acc  = 0.0
        metrics_history = []

        for epoch in range(1, cfg.epochs + 1):
            # Train
            self._model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                batch     = {k: v.to(device) for k, v in batch.items()}
                outputs   = self._model(**batch)
                # Always compute loss manually via F.cross_entropy rather
                # than relying on outputs.loss (the model's internal loss
                # is unweighted) -- passing weight=None reproduces the
                # model's default behaviour exactly when weighted_loss
                # is off, so this is a single code path either way.
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits, batch["labels"], weight=loss_weight
                )
                epoch_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_loss = epoch_loss / len(train_loader)

            # Validate
            val_metrics = self._evaluate(val_loader, device)
            logger.info(
                "Epoch %d/%d  loss=%.4f  val_acc=%.4f  val_f1=%.4f",
                epoch, cfg.epochs,
                avg_loss,
                val_metrics["accuracy"],
                val_metrics["f1"],
            )
            metrics_history.append({
                "epoch":    epoch,
                "loss":     avg_loss,
                "accuracy": val_metrics["accuracy"],
                "f1":       val_metrics["f1"],
            })

            # Save the best checkpoint
            if val_metrics["f1"] > best_val_f1:
                best_val_f1  = val_metrics["f1"]
                best_val_acc = val_metrics["accuracy"]
                self.save()
                logger.info("  → New best model saved (val_f1=%.4f)", best_val_f1)

        # ── Final evaluation on test set ──────────────────────────
        logger.info("Loading best checkpoint for test evaluation...")
        self.load()
        # load() always places the model on CPU -- move it back to the
        # training device or GPU runs crash here with a device mismatch
        # against the CUDA-resident evaluation batches.
        self._model.to(device)
        test_metrics = self._evaluate(test_loader, device)

        from sklearn.metrics import classification_report, confusion_matrix
        test_report = classification_report(
            test_metrics["labels"], test_metrics["preds"],
            target_names = self._labels,
            labels       = list(range(len(self._labels))),
            output_dict  = True,
            zero_division = 0,
        )
        per_class = {
            lbl: {
                "precision": test_report[lbl]["precision"],
                "recall":    test_report[lbl]["recall"],
                "f1":        test_report[lbl]["f1-score"],
                "support":   test_report[lbl]["support"],
            }
            for lbl in self._labels
        }
        for lbl, m in per_class.items():
            logger.info(
                "  %-10s precision=%.3f  recall=%.3f  f1=%.3f  (n=%d)",
                lbl, m["precision"], m["recall"], m["f1"], m["support"],
            )

        cm = confusion_matrix(
            test_metrics["labels"], test_metrics["preds"],
            labels = list(range(len(self._labels))),
        ).tolist()

        results = {
            "val_accuracy":     best_val_acc,
            "val_f1":           best_val_f1,
            "test_accuracy":    test_metrics["accuracy"],
            "test_f1":          test_metrics["f1"],
            "per_class":        per_class,
            "confusion_matrix": cm,
            "history":          metrics_history,
        }
        logger.info(
            "Training complete — test_acc=%.4f  test_f1=%.4f",
            results["test_accuracy"], results["test_f1"],
        )

        self._save_metrics(results)
        return results

    def _save_metrics(self, results: dict) -> None:
        """Persist training metrics to output_dir/training_metrics.json.

        Read by the dashboard's Model Metrics page (via the API, not
        directly -- the dashboard process never has this file locally).

        Args:
            results: The dict returned by train().
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self._output_dir / "training_metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2))
        logger.info("Training metrics saved to %s", metrics_path)

    def _evaluate(self, data_loader, device) -> dict[str, float]:
        """Run inference on a DataLoader and return accuracy and F1.

        Args:
            data_loader: PyTorch DataLoader with labelled examples.
            device:      torch.device for inference.

        Returns:
            Dict with ``"accuracy"`` and ``"f1"`` keys.
        """
        import torch
        from sklearn.metrics import accuracy_score, f1_score

        self._model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                labels = batch.pop("labels").numpy()
                batch  = {k: v.to(device) for k, v in batch.items()}
                logits = self._model(**batch).logits
                preds  = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1":       f1_score(
                all_labels, all_preds,
                average="weighted", zero_division=0,
            ),
            "labels": all_labels,
            "preds":  all_preds,
        }

    # ── Inference ─────────────────────────────────────────────────

    def predict(self, text: str) -> ClassificationResult:
        """Classify a single clinical note.

        Args:
            text: Cleaned clinical note text.  Pass raw text through
                  :func:`src.utils.text_utils.prepare_for_inference`
                  before calling this method.

        Returns:
            :class:`ClassificationResult` with the predicted label,
            confidence, and full probability distribution.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "Model not loaded. Call load() before predict()."
            )

        import torch

        device   = next(self._model.parameters()).device
        encoding = self._tokenizer(
            text,
            max_length     = TrainingConfig.max_length,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt",
        )

        self._model.eval()
        with torch.no_grad():
            logits = self._model(
                input_ids      = encoding["input_ids"].to(device),
                attention_mask = encoding["attention_mask"].to(device),
            ).logits

        probs       = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_idx    = int(np.argmax(probs))
        pred_label  = self._id2label[pred_idx]
        confidence  = float(probs[pred_idx])
        prob_dict   = {
            self._id2label[i]: float(p) for i, p in enumerate(probs)
        }

        return ClassificationResult(
            label         = pred_label,
            confidence    = confidence,
            probabilities = prob_dict,
            task          = self._task,
        )

    def predict_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """Classify multiple clinical notes.

        More efficient than calling predict() in a loop because
        the tokeniser and model run in batches.

        Args:
            texts: List of cleaned clinical note strings.

        Returns:
            List of :class:`ClassificationResult` objects, one per input.
        """
        if not texts:
            return []
        # For simplicity, loop — a production implementation would
        # use DataLoader batching here.
        return [self.predict(t) for t in texts]

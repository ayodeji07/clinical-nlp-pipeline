"""
scripts/train_severity_classifier.py
────────────────────────────────────────────────────────────────
Retrains the severity classifier with the two fixes that were
deferred on 2026-06-25 (see project memory):

  1. Class-weighted loss -- "critical" is ~15% of the data and
     was getting recall 0.486. ClinicalClassifier.train() now
     supports weighted_loss=True to counter this.
  2. Corrected weak-supervision labels -- src/etl/transform.py's
     `\\bacute\\b` rule was matching negated phrases like "no acute
     complaints". Fixed to `(?<!no )\\bacute\\b`, but the model was
     never retrained on the corrected labels. This script re-derives
     severity fresh via derive_severity_labels() rather than trusting
     the (possibly stale) severity column already stored in Supabase,
     so the fix is guaranteed to be reflected in training data
     regardless of when each note was originally loaded.

Trains from the base model (resume=False) rather than warm-starting
from the old checkpoint, since the loss function, label corrections,
and (on a GPU) batch/sequence length are all different from whatever
produced the existing checkpoint -- a clean run gives an unambiguous
before/after comparison.

Usage
─────
    python scripts/train_severity_classifier.py                # full run
    python scripts/train_severity_classifier.py --limit 300     # quick smoke test
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from src.db.connection import get_session
from src.db.models import ClinicalNote, ModelRun
from src.db.repository import ModelRunRepository
from src.etl.transform import derive_severity_labels
from src.nlp.classifier import ClinicalClassifier
from src.utils.config import ModelConfig, Paths
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Smoke-test runs (--limit) default to this scratch directory instead of
# the real ModelConfig.fine_tuned_dir -- train() saves a checkpoint on
# the very first epoch that beats val_f1=0.0, which is nearly guaranteed,
# so a real run's checkpoint must never share a path with a test run.
_SMOKE_TEST_DIR = Paths.models / "severity_classifier_smoketest"


def _load_notes(limit: int | None) -> pd.DataFrame:
    """Fetch note text from Supabase and derive fresh severity labels.

    Args:
        limit: Maximum number of notes to fetch, or None for all.

    Returns:
        DataFrame with ``transcription`` and freshly-derived
        ``severity`` columns.
    """
    with get_session() as session:
        stmt = select(ClinicalNote.transcription)
        if limit:
            stmt = stmt.limit(limit)
        rows = session.execute(stmt).all()

    df = pd.DataFrame({"transcription": [r.transcription for r in rows]})
    logger.info("Loaded %d notes from Supabase", len(df))

    df = derive_severity_labels(df)
    logger.info(
        "Severity distribution (freshly derived): %s",
        df["severity"].value_counts().to_dict(),
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the severity classifier with class-weighted loss and corrected labels."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of notes to train on (default: all notes). "
             "Implies --output-dir points at a scratch directory unless "
             "overridden, so a smoke test can never touch the real checkpoint.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to save the checkpoint (default: the real model dir for "
             "a full run, or a smoke-test scratch dir when --limit is set)",
    )
    args = parser.parse_args()

    df = _load_notes(args.limit)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None and args.limit is not None:
        output_dir = _SMOKE_TEST_DIR
        logger.warning(
            "--limit set without --output-dir -- saving to scratch dir %s "
            "instead of the real checkpoint.", output_dir,
        )

    clf = ClinicalClassifier(task="severity", output_dir=output_dir)
    results = clf.train(df, resume=False, weighted_loss=True)

    logger.info("=" * 60)
    logger.info(
        "Final: val_acc=%.4f val_f1=%.4f test_acc=%.4f test_f1=%.4f",
        results["val_accuracy"], results["val_f1"],
        results["test_accuracy"], results["test_f1"],
    )
    critical = results["per_class"]["critical"]
    logger.info(
        "Critical class: precision=%.3f recall=%.3f f1=%.3f",
        critical["precision"], critical["recall"], critical["f1"],
    )

    # Record this run in the DB -- what the dashboard's Model Metrics
    # page and the /model/metrics API endpoint actually read from.
    # Skipped for smoke-test runs (--limit set): a tiny/partial run
    # shouldn't overwrite the real deployed model's recorded metrics.
    if args.limit is None:
        with get_session() as session:
            session.query(ModelRun).filter_by(
                task="severity", is_deployed=True
            ).update({"is_deployed": False})
            ModelRunRepository.create(
                session,
                model_name       = ModelConfig.classifier_model,
                task             = "severity",
                training_samples = len(df),
                val_accuracy     = results["val_accuracy"],
                val_f1           = results["val_f1"],
                test_accuracy    = results["test_accuracy"],
                test_f1          = results["test_f1"],
                per_class        = results["per_class"],
                confusion_matrix = results["confusion_matrix"],
                history          = results["history"],
                epochs           = len(results["history"]),
                is_deployed      = True,
                run_notes        = "Class-weighted loss, corrected acute-negation labels",
            )
        logger.info("Run recorded in model_runs (is_deployed=True)")
    else:
        logger.info("--limit set -- skipping model_runs DB record (smoke test)")


if __name__ == "__main__":
    main()

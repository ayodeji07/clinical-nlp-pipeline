"""
src/etl/pipeline.py
────────────────────────────────────────────────────────────────
ETL pipeline orchestrator.

Ties together extract → transform → load into one callable
entry point.  Can be run from the command line or imported
and called programmatically.

Usage from command line:
    python -m src.etl.pipeline
    python -m src.etl.pipeline --dry-run
    python -m src.etl.pipeline --source mimic

Usage from Python:
    from src.etl.pipeline import run_pipeline
    result = run_pipeline()
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

from src.etl.extract import load_clinical_notes, load_icd10_codes
from src.etl.transform import prepare_for_training
from src.etl.load import load_clinical_notes as persist_notes
from src.utils.config import Paths, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Summary of a completed pipeline run.

    Attributes:
        notes_extracted: Raw notes loaded from source.
        notes_transformed: Notes remaining after cleaning.
        notes_loaded: New records written to the database.
        icd10_codes_loaded: Rows in the ICD-10 reference table.
        duration_seconds: Wall-clock time for the full run.
        errors: Any non-fatal errors encountered.
    """

    notes_extracted:    int = 0
    notes_transformed:  int = 0
    notes_loaded:       int = 0
    icd10_codes_loaded: int = 0
    duration_seconds:   float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable pipeline summary string."""
        lines = [
            "-" * 50,
            "  ETL PIPELINE SUMMARY",
            "-" * 50,
            f"  Extracted   : {self.notes_extracted:,} notes",
            f"  Transformed : {self.notes_transformed:,} notes",
            f"  Loaded      : {self.notes_loaded:,} new records",
            f"  ICD-10 codes: {self.icd10_codes_loaded:,}",
            f"  Duration    : {self.duration_seconds:.1f}s",
        ]
        if self.errors:
            lines.append(f"  Errors      : {len(self.errors)}")
            for err in self.errors:
                lines.append(f"    - {err}")
        lines.append("-" * 50)
        return "\n".join(lines)


def run_pipeline(
    dry_run: bool = False,
    data_source: str = "mtsamples",
) -> PipelineResult:
    """Run the full Extract → Transform → Load pipeline.

    Args:
        dry_run: If True, run extract and transform but skip the
            database write.  Useful for validating data quality
            before committing.
        data_source: Which source to load.  ``"mtsamples"`` (default)
            or ``"mimic"`` (requires MIMIC-III access).

    Returns:
        :class:`PipelineResult` with counts and timing.

    Example::

        result = run_pipeline(dry_run=True)
        print(result.summary())
    """
    start    = time.perf_counter()
    result   = PipelineResult()

    Paths.ensure_all()

    logger.info("=" * 50)
    logger.info("  CLINICAL NLP PIPELINE")
    logger.info("  dry_run=%s  source=%s", dry_run, data_source)
    logger.info("=" * 50)

    # ── Stage 1: Extract ─────────────────────────────────────────
    logger.info("Stage 1: Extract")
    try:
        import os
        os.environ["DATA_SOURCE"] = data_source
        raw_df = load_clinical_notes()
        result.notes_extracted = len(raw_df)
        logger.info("  Loaded %d raw notes", result.notes_extracted)
    except FileNotFoundError as exc:
        logger.error("Extract failed: %s", exc)
        result.errors.append(str(exc))
        result.duration_seconds = time.perf_counter() - start
        return result

    # ICD-10 reference table (best-effort — pipeline continues without it)
    try:
        icd_df = load_icd10_codes()
        result.icd10_codes_loaded = len(icd_df)
        # Persist to processed directory for fast access by the mapper
        icd_out = Paths.processed / "icd10_codes.parquet"
        icd_df.to_parquet(icd_out, index=False)
        logger.info("  ICD-10 table saved to %s", icd_out.name)
    except FileNotFoundError:
        logger.warning(
            "ICD-10 reference file not found — "
            "ICD-10 mapping will be disabled."
        )
        result.errors.append("ICD-10 reference file missing")

    # ── Stage 2: Transform ────────────────────────────────────────
    logger.info("Stage 2: Transform")
    processed_df = prepare_for_training(raw_df)
    result.notes_transformed = len(processed_df)

    # Save to parquet for use by the NLP training notebooks
    out_path = Paths.processed / "clinical_notes.parquet"
    processed_df.to_parquet(out_path, index=False)
    logger.info("  Processed notes saved to %s", out_path.name)

    # ── Stage 3: Load ─────────────────────────────────────────────
    if dry_run:
        logger.info("Stage 3: Load (SKIPPED — dry run)")
        logger.info(
            "  Would insert up to %d records", result.notes_transformed
        )
    else:
        logger.info("Stage 3: Load")
        try:
            from src.db.connection import create_all_tables
            create_all_tables()
            result.notes_loaded = persist_notes(processed_df)
        except Exception as exc:
            logger.error("Load failed: %s", exc)
            result.errors.append(f"Load error: {exc}")

    result.duration_seconds = time.perf_counter() - start
    logger.info(result.summary())
    return result


# ── CLI entry point ───────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clinical NLP Pipeline — ETL runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and transform only; do not write to database.",
    )
    parser.add_argument(
        "--source",
        choices=["mtsamples", "mimic"],
        default="mtsamples",
        help="Data source to load (default: mtsamples).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from src.utils.logger import set_log_level

    args = _parse_args()
    set_log_level(args.log_level)

    result = run_pipeline(
        dry_run     = args.dry_run,
        data_source = args.source,
    )

    import sys
    sys.exit(0 if not result.errors else 1)

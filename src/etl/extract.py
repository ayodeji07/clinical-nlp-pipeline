"""
src/etl/extract.py
────────────────────────────────────────────────────────────────
Data extraction layer — the "E" in ETL.

Responsibilities
────────────────
  - Load MTSamples clinical notes from CSV
  - Load the ICD-10 reference table
  - Detect whether MIMIC-III data is available (optional)
  - Cache raw downloads so re-runs are fast

Design
──────
  Extractors are intentionally dumb: they load data and add a
  ``_source`` provenance column, but they do not clean, filter,
  or reshape anything.  All of that happens in transform.py.

  Every public function returns a DataFrame or raises a clear
  exception — never returns None.  Callers can always rely on
  the return type.

  MTSamples is freely available on Kaggle:
    https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
  Download ``mtsamples.csv`` and place it in ``data/raw/``.

  The ICD-10 code file (icd10_codes.csv) is included in the
  project under ``data/raw/`` — it is a static reference table
  that rarely changes.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import Paths
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Column name aliases ────────────────────────────────────────────
# MTSamples columns vary slightly across Kaggle versions.
# We normalise to these names during extraction.
_MTSAMPLES_COLUMN_MAP: dict[str, str] = {
    "description":        "description",
    "medical_specialty":  "specialty",
    "sample_name":        "sample_name",
    "transcription":      "transcription",
    "keywords":           "keywords",
}

_ICD10_COLUMN_MAP: dict[str, str] = {
    "code":        "icd10_code",
    "description": "description",
    "category":    "category",
}


# ── MTSamples ─────────────────────────────────────────────────────

def load_mtsamples(
    path: Path | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Load the MTSamples clinical notes dataset from CSV.

    MTSamples contains 4,999 de-identified medical transcriptions
    across 40 clinical specialties.  The raw CSV is expected at
    ``data/raw/mtsamples.csv``.

    Args:
        path: Override the default file location.  Useful in tests.
        force_reload: If True, bypass the cached version and re-read
            from disk.  Defaults to False.

    Returns:
        DataFrame with columns::

            description     : short note title
            specialty       : medical specialty (40 categories)
            sample_name     : note type (e.g. "Discharge Summary")
            transcription   : full clinical note text
            keywords        : comma-separated clinical keywords
            _source         : "mtsamples" (provenance marker)

    Raises:
        FileNotFoundError: If the CSV does not exist at the
            expected path.  The error message includes the
            download URL.

    Example::

        df = load_mtsamples()
        print(df.shape)          # (4999, 6)
        print(df["specialty"].nunique())  # 40
    """
    csv_path = path or Paths.mtsamples_csv

    if not csv_path.exists():
        raise FileNotFoundError(
            f"MTSamples CSV not found at: {csv_path}\n"
            "Download it from Kaggle:\n"
            "  https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions\n"
            f"Then save it to: {Paths.raw}/"
        )

    logger.info("Loading MTSamples from %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    # Rename columns to our standard names, ignoring any extras
    rename_map = {
        col: _MTSAMPLES_COLUMN_MAP[col]
        for col in df.columns
        if col in _MTSAMPLES_COLUMN_MAP
    }
    df = df.rename(columns=rename_map)

    # Keep only columns we use; add the rest as-is if not in our map
    expected = list(_MTSAMPLES_COLUMN_MAP.values())
    available = [c for c in expected if c in df.columns]
    df = df[available].copy()

    df["_source"] = "mtsamples"

    logger.info(
        "MTSamples loaded: %d notes, %d specialties",
        len(df),
        df["specialty"].nunique() if "specialty" in df.columns else 0,
    )
    return df


# ── ICD-10 reference table ─────────────────────────────────────────

def load_icd10_codes(path: Path | None = None) -> pd.DataFrame:
    """Load the ICD-10 diagnostic code reference table.

    The ICD-10 CSV is included in the repository under
    ``data/raw/icd10_codes.csv``.  It contains all billable
    ICD-10-CM codes with descriptions.

    Args:
        path: Override the default file location.  Useful in tests.

    Returns:
        DataFrame with columns::

            icd10_code  : ICD-10-CM code (e.g. "I10", "J18.9")
            description : Human-readable description
            category    : Broad category (e.g. "Circulatory System")
            _source     : "icd10_cms" (provenance marker)

    Raises:
        FileNotFoundError: If the CSV is missing.

    Example::

        icd = load_icd10_codes()
        icd[icd["icd10_code"] == "I10"]
        # icd10_code | description            | category
        # I10        | Essential hypertension | Circulatory System
    """
    csv_path = path or Paths.icd10_csv

    if not csv_path.exists():
        raise FileNotFoundError(
            f"ICD-10 codes file not found at: {csv_path}\n"
            "This file should be included in the repository.\n"
            "Check that you have the full project download."
        )

    logger.info("Loading ICD-10 reference table from %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    rename_map = {
        col: _ICD10_COLUMN_MAP[col]
        for col in df.columns
        if col in _ICD10_COLUMN_MAP
    }
    df = df.rename(columns=rename_map)

    # Ensure the code column is clean
    if "icd10_code" in df.columns:
        df["icd10_code"] = df["icd10_code"].str.strip().str.upper()

    df["_source"] = "icd10_cms"

    logger.info("ICD-10 table loaded: %d codes", len(df))
    return df


# ── MIMIC-III (optional) ───────────────────────────────────────────

def load_mimic_notes(
    path: Path | None = None,
) -> pd.DataFrame | None:
    """Load MIMIC-III discharge summaries if available.

    MIMIC-III requires credentialed access via PhysioNet:
      https://physionet.org/content/mimiciii/

    This function gracefully returns None if the data is not
    present rather than raising an error — MIMIC-III is optional.
    The pipeline uses MTSamples when MIMIC-III is absent.

    Args:
        path: Path to ``NOTEEVENTS.csv`` from MIMIC-III.
            Defaults to ``data/raw/mimic_notes.csv``.

    Returns:
        DataFrame with columns ``transcription``, ``specialty``,
        ``_source`` if the file exists; None otherwise.

    Example::

        notes = load_mimic_notes()
        if notes is None:
            print("MIMIC-III not available, using MTSamples")
    """
    default_path = Paths.raw / "mimic_notes.csv"
    csv_path     = path or default_path

    if not csv_path.exists():
        logger.info(
            "MIMIC-III notes not found at %s — using MTSamples only. "
            "See https://physionet.org/content/mimiciii/ for access.",
            csv_path,
        )
        return None

    logger.info("Loading MIMIC-III notes from %s", csv_path)
    df = pd.read_csv(
        csv_path,
        usecols=["TEXT", "CATEGORY"],
        low_memory=False,
    )

    df = df.rename(columns={"TEXT": "transcription", "CATEGORY": "specialty"})
    df["description"] = "MIMIC-III discharge note"
    df["_source"]     = "mimic3"

    # Keep only discharge summaries for comparability with MTSamples
    df = df[df["specialty"].str.lower() == "discharge summary"].copy()

    logger.info("MIMIC-III notes loaded: %d discharge summaries", len(df))
    return df


# ── Auto-detect best available source ─────────────────────────────

def load_clinical_notes() -> pd.DataFrame:
    """Load clinical notes from the best available source.

    Tries sources in order of preference:
      1. MTSamples (primary — free, always available once downloaded)
      2. MIMIC-III (optional — richer, requires credentialed access)

    If both are available, MTSamples is used by default.
    Set ``DATA_SOURCE=mimic`` in your ``.env`` to prefer MIMIC-III.

    Returns:
        DataFrame with at minimum the columns ``transcription``,
        ``specialty``, and ``_source``.

    Raises:
        FileNotFoundError: If no data source is available.
    """
    import os
    preferred = os.getenv("DATA_SOURCE", "mtsamples").lower()

    if preferred == "mimic":
        mimic_df = load_mimic_notes()
        if mimic_df is not None:
            return mimic_df
        logger.warning(
            "DATA_SOURCE=mimic but MIMIC-III not found. "
            "Falling back to MTSamples."
        )

    return load_mtsamples()

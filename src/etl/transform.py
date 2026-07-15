"""
src/etl/transform.py
────────────────────────────────────────────────────────────────
Data transformation layer — the "T" in ETL.

Responsibilities
────────────────
  - Clean and validate raw clinical notes
  - Derive severity labels using weak supervision
  - Engineer features needed by the NLP models
  - Produce the final, analysis-ready DataFrame

Weak supervision for severity labels
─────────────────────────────────────
MTSamples has no severity labels, so we derive them using a set
of keyword rules applied to the transcription text.  This is
"weak supervision" — the labels are noisy but good enough to
train a classifier that learns to generalise beyond the rules.

The three severity tiers:
  critical  — ICU, ventilator, cardiac arrest, septic shock,
               stroke, code, emergent, life-threatening
  urgent    — emergency, acute, admitted, inpatient, fever,
               infection, pain scale > 6, SOB, unstable
  routine   — elective, outpatient, follow-up, stable, chronic
               management, wellness, screening

Notes that match no keywords default to "routine".
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re

import pandas as pd

from src.utils.logger import get_logger
from src.utils.text_utils import clean_clinical_text, word_count

logger = get_logger(__name__)


# ── Severity keyword rules ────────────────────────────────────────
# Evaluated in order: critical → urgent → routine.
# The first tier that matches wins.
_SEVERITY_RULES: dict[str, list[str]] = {
    "critical": [
        r"\bicu\b", r"\bintensive care\b", r"\bventilat",
        r"\bcardiac arrest\b", r"\bseptic shock\b",
        r"\brespiratory failure\b", r"\bmulti.?organ failure\b",
        r"\bcode blue\b", r"\bemergent intubat",
        r"\blife.?threatening\b", r"\bcritically ill\b",
        r"\bstroke\b", r"\bcva\b", r"\binfarction\b",
        r"\bpulseless\b", r"\bunresponsive\b",
    ],
    "urgent": [
        r"\bemergency\b", r"\bemer\b", r"(?<!no )\bacute\b",
        r"\badmitted\b", r"\binpatient\b",
        r"\bhigh fever\b", r"\btemperature of [3-4]\d\.",
        r"\bsevere pain\b", r"\bpain.{0,20}[7-9][/\s]10\b",
        r"\bshortness of breath\b", r"\bsob\b",
        r"\bunstable\b", r"\brapidly\b",
        r"\binfection\b", r"\bsepsis\b",
        r"\btrauma\b", r"\bfracture\b",
        r"\bhemorrhage\b", r"\bbleeding\b",
        r"\bchest pain\b",
    ],
    "routine": [
        r"\belective\b", r"\boutpatient\b",
        r"\bfollow.?up\b", r"\bstable\b",
        r"\bchronic\b", r"\bwellness\b",
        r"\bscreening\b", r"\bpreventive\b",
        r"\bmanagement of\b", r"\broutine\b",
        r"\bno acute\b", r"\bnormal\b",
    ],
}


def clean_notes(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate a DataFrame of clinical notes.

    Applies the following steps:
      1. Drop rows with missing or empty transcription text
      2. Normalise the transcription through the text cleaner
      3. Drop duplicates (same transcription appearing twice)
      4. Reset the index

    Args:
        df: Raw DataFrame from the extraction layer.  Must contain
            a ``transcription`` column.

    Returns:
        Cleaned DataFrame.  Row count will be <= input row count.

    Raises:
        KeyError: If ``transcription`` column is absent.
    """
    if "transcription" not in df.columns:
        raise KeyError(
            "DataFrame must have a 'transcription' column. "
            f"Got: {list(df.columns)}"
        )

    original_count = len(df)

    # Drop rows where the note text is missing or whitespace-only
    df = df.dropna(subset=["transcription"]).copy()
    df = df[df["transcription"].str.strip().astype(bool)].copy()

    after_drop = len(df)
    if after_drop < original_count:
        logger.info(
            "Dropped %d rows with missing transcription text",
            original_count - after_drop,
        )

    # Apply clinical text cleaning
    df["transcription"] = df["transcription"].apply(clean_clinical_text)

    # Remove duplicates — same note appearing more than once
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["transcription"]).copy()
    if len(df) < before_dedup:
        logger.info(
            "Removed %d duplicate notes", before_dedup - len(df)
        )

    df = df.reset_index(drop=True)
    logger.info("Clean notes: %d rows remaining", len(df))
    return df


def derive_severity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a severity label to each clinical note.

    Uses weak supervision (keyword matching on transcription text)
    to derive one of three labels: ``routine``, ``urgent``,
    ``critical``.

    Labels are noisy — they are intended as training signal, not
    ground truth.  The classifier learns to generalise beyond the
    keyword patterns.

    Args:
        df: DataFrame with a ``transcription`` column.

    Returns:
        DataFrame with a new ``severity`` column appended.
        Distribution is logged at INFO level.

    Note:
        Notes that match no keywords are labelled ``"routine"``
        (the most common clinical scenario).
    """
    def _assign_label(text: str) -> str:
        """Classify one note by matching against keyword tiers."""
        if not isinstance(text, str):
            return "routine"

        lower = text.lower()

        for tier in ("critical", "urgent", "routine"):
            patterns = _SEVERITY_RULES[tier]
            if any(re.search(p, lower) for p in patterns):
                return tier

        # Default: routine
        return "routine"

    logger.info("Deriving severity labels via weak supervision...")
    df = df.copy()
    df["severity"] = df["transcription"].apply(_assign_label)

    # Log the distribution so the user can spot class imbalance
    dist = df["severity"].value_counts()
    for label, count in dist.items():
        pct = count / len(df) * 100
        logger.info("  %-10s %4d  (%.1f%%)", label, count, pct)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used by downstream models and analysis.

    Features added:
      - ``word_count``    : number of words in the transcription
      - ``char_count``    : character count
      - ``has_keywords``  : whether the keywords column is populated
      - ``note_type``     : normalised note type (lowercase)
      - ``specialty_clean``: specialty with whitespace normalised

    Args:
        df: DataFrame with at minimum a ``transcription`` column.

    Returns:
        DataFrame with additional feature columns.
    """
    df = df.copy()

    df["word_count"] = df["transcription"].apply(word_count)
    df["char_count"] = df["transcription"].str.len().fillna(0).astype(int)

    if "keywords" in df.columns:
        df["has_keywords"] = (
            df["keywords"].notna() & df["keywords"].str.strip().astype(bool)
        )
    else:
        df["has_keywords"] = False

    if "sample_name" in df.columns:
        df["note_type"] = (
            df["sample_name"]
            .fillna("unknown")
            .str.lower()
            .str.strip()
        )

    if "specialty" in df.columns:
        df["specialty_clean"] = (
            df["specialty"]
            .fillna("unknown")
            .str.strip()
            .str.title()
        )

    return df


def filter_minimum_length(
    df: pd.DataFrame,
    min_words: int = 30,
) -> pd.DataFrame:
    """Remove notes that are too short to be clinically useful.

    Very short notes (< 30 words by default) tend to be incomplete
    transcriptions or metadata entries rather than real clinical notes.
    Including them degrades model training.

    Args:
        df: DataFrame with a ``word_count`` column.  Call
            :func:`engineer_features` first if needed.
        min_words: Minimum word count to retain a note.

    Returns:
        Filtered DataFrame.
    """
    if "word_count" not in df.columns:
        df = engineer_features(df)

    before = len(df)
    df = df[df["word_count"] >= min_words].copy()
    dropped = before - len(df)

    if dropped:
        logger.info(
            "Filtered %d notes shorter than %d words", dropped, min_words
        )

    return df.reset_index(drop=True)


def prepare_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full transformation pipeline for model training.

    Convenience wrapper that runs all transformation steps in the
    correct order.

    Steps:
      1. :func:`clean_notes`
      2. :func:`engineer_features`
      3. :func:`filter_minimum_length`
      4. :func:`derive_severity_labels`

    Args:
        df: Raw DataFrame from the extraction layer.

    Returns:
        Fully transformed, analysis-ready DataFrame.
    """
    logger.info("Running full transformation pipeline...")

    df = clean_notes(df)
    df = engineer_features(df)
    df = filter_minimum_length(df)
    df = derive_severity_labels(df)

    logger.info(
        "Transformation complete: %d notes ready for training", len(df)
    )
    return df


def prepare_for_inference(text: str) -> str:
    """Prepare a single raw text string for model inference.

    Used by the API and Streamlit demo when a user pastes clinical
    text.  Applies the same cleaning as training data without the
    label derivation.

    Args:
        text: Raw clinical note text from the user.

    Returns:
        Cleaned text string ready for NER and classification.
    """
    return clean_clinical_text(text, expand_abbreviations=True)

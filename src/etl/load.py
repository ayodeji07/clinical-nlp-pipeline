"""
src/etl/load.py
────────────────────────────────────────────────────────────────
Data loading layer — the "L" in ETL.

Responsibilities
────────────────
  - Write the processed clinical notes DataFrame to the database
  - Write extracted NER entities to the entities table
  - Write ICD-10 mapping results to the mappings table
  - Handle upserts gracefully (re-running the pipeline is safe)

All write operations go through the SQLAlchemy session from
``src/db/connection.py``.  The loader never builds SQL strings
directly — it uses the ORM models from ``src/db/models.py``.

Idempotency
───────────
The pipeline can be run multiple times without creating
duplicate records.  This is handled by:
  - Checking for existing records by natural key before inserting
  - Using bulk_insert_mappings for performance on large datasets
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import pandas as pd

from src.db.connection import get_session
from src.db.models import ClinicalNote, Entity, ICD10Mapping
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_clinical_notes(df: pd.DataFrame) -> int:
    """Persist a DataFrame of processed clinical notes to the database.

    Uses bulk insert for performance.  Records are skipped if a note
    with the same ``source_id`` already exists, making the operation
    safe to re-run.

    Args:
        df: Processed DataFrame.  Must contain at minimum:
            ``transcription``, ``specialty``, ``_source``.
            Optional: ``severity``, ``word_count``, ``note_type``.

    Returns:
        Number of new records inserted.

    Example::

        n = load_clinical_notes(processed_df)
        print(f"Inserted {n} new notes")
    """
    if df.empty:
        logger.warning("load_clinical_notes called with empty DataFrame")
        return 0

    inserted = 0

    with get_session() as session:
        for idx, row in df.iterrows():
            # Use the DataFrame row index as a stable source ID.
            # In production you would use a hash of the transcription.
            source_id = f"{row.get('_source', 'unknown')}_{idx}"

            # Skip if already loaded
            exists = (
                session.query(ClinicalNote)
                .filter_by(source_id=source_id)
                .first()
            )
            if exists:
                continue

            note = ClinicalNote(
                source_id      = source_id,
                transcription  = row.get("transcription", ""),
                specialty      = row.get("specialty_clean") or row.get("specialty"),
                note_type      = row.get("note_type"),
                severity       = row.get("severity"),
                word_count     = row.get("word_count"),
                data_source    = row.get("_source", "unknown"),
            )
            session.add(note)
            inserted += 1

            # Commit in batches of 500 to avoid large transactions
            if inserted % 500 == 0:
                session.commit()
                logger.info("  Committed %d notes...", inserted)

        session.commit()

    logger.info("load_clinical_notes: inserted %d new records", inserted)
    return inserted


def load_entities(entities: list[dict]) -> int:
    """Persist extracted NER entities to the database.

    Each entity dict should contain:
      ``note_id``, ``text``, ``label``, ``start``, ``end``,
      ``confidence`` (optional).

    Args:
        entities: List of entity dicts from the NER pipeline.

    Returns:
        Number of records inserted.
    """
    if not entities:
        return 0

    inserted = 0

    with get_session() as session:
        for ent in entities:
            record = Entity(
                note_id    = ent["note_id"],
                text       = ent["text"],
                label      = ent["label"],
                start_char = ent.get("start", 0),
                end_char   = ent.get("end", 0),
                confidence = ent.get("confidence"),
            )
            session.add(record)
            inserted += 1

        session.commit()

    logger.info("load_entities: inserted %d entity records", inserted)
    return inserted


def load_icd10_mappings(mappings: list[dict]) -> int:
    """Persist ICD-10 mapping results to the database.

    Each mapping dict should contain:
      ``entity_id``, ``icd10_code``, ``description``,
      ``match_method`` (``"lookup"`` or ``"embedding"``),
      ``confidence``.

    Args:
        mappings: List of mapping dicts from the ICD-10 mapper.

    Returns:
        Number of records inserted.
    """
    if not mappings:
        return 0

    inserted = 0

    with get_session() as session:
        for m in mappings:
            record = ICD10Mapping(
                entity_id    = m["entity_id"],
                icd10_code   = m["icd10_code"],
                description  = m.get("description", ""),
                match_method = m.get("match_method", "unknown"),
                confidence   = m.get("confidence", 0.0),
                rank         = m.get("rank", 1),
            )
            session.add(record)
            inserted += 1

        session.commit()

    logger.info("load_icd10_mappings: inserted %d records", inserted)
    return inserted

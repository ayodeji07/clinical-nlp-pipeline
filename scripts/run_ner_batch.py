"""
scripts/run_ner_batch.py
────────────────────────────────────────────────────────────────
Batch NER runner — the missing link between the ETL loader and
the entities table. Reads stored clinical notes from the
database (Supabase Postgres or local SQLite, whichever
DATABASE_URL points at), runs the configured NER pipeline
(src/nlp/ner.py) over notes that don't have entities yet, and
writes the results back via load_entities() (src/etl/load.py).

Idempotent / resumable: a note is only processed if it has zero
rows in the entities table, so interrupting the run (Ctrl+C,
crash, overnight power loss) and re-running later just picks up
where it left off — no duplicate entities.

Usage
─────
    python scripts/run_ner_batch.py                  # all pending notes
    python scripts/run_ner_batch.py --limit 250       # quick demo subset
    python scripts/run_ner_batch.py --batch-size 16   # smaller spaCy batches
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import time

from sqlalchemy import select

from src.db.connection import get_session
from src.db.models import ClinicalNote, Entity as EntityRow
from src.etl.load import load_entities
from src.nlp.ner import build_ner_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _fetch_pending_notes(limit: int | None) -> list[tuple[int, str]]:
    """Return (id, transcription) pairs for notes with no entities yet.

    Args:
        limit: Maximum number of notes to return, or None for all
            pending notes.

    Returns:
        List of (note_id, transcription) tuples, ordered by note id
        so a --limit run always covers the same notes on re-run.
    """
    with get_session() as session:
        stmt = (
            select(ClinicalNote.id, ClinicalNote.transcription)
            .outerjoin(EntityRow, EntityRow.note_id == ClinicalNote.id)
            .where(EntityRow.id.is_(None))
            .order_by(ClinicalNote.id)
        )
        if limit:
            stmt = stmt.limit(limit)
        return [(row.id, row.transcription) for row in session.execute(stmt)]


def run(limit: int | None, batch_size: int) -> None:
    """Run NER over pending notes and save the extracted entities.

    Args:
        limit: Maximum number of pending notes to process, or None
            for every note that doesn't have entities yet.
        batch_size: Number of notes per spaCy nlp.pipe() batch, and
            per load_entities() write.
    """
    notes = _fetch_pending_notes(limit)
    total = len(notes)
    if total == 0:
        logger.info("No pending notes — every note already has entities.")
        return

    logger.info("Running NER on %d notes (batch_size=%d)...", total, batch_size)
    pipeline = build_ner_pipeline()

    start = time.time()
    processed = 0
    total_entities = 0

    for chunk_start in range(0, total, batch_size):
        chunk = notes[chunk_start : chunk_start + batch_size]
        ids = [note_id for note_id, _ in chunk]
        texts = [text or "" for _, text in chunk]

        batches = pipeline.extract_batch(texts, batch_size=batch_size)

        entity_dicts = []
        for note_id, entities in zip(ids, batches):
            for ent in entities:
                d = ent.to_dict()
                d["note_id"] = note_id
                entity_dicts.append(d)

        total_entities += load_entities(entity_dicts)
        processed += len(chunk)

        elapsed = time.time() - start
        rate = processed / elapsed if elapsed > 0 else 0
        eta_seconds = (total - processed) / rate if rate > 0 else float("inf")
        logger.info(
            "  %d/%d notes | %d entities so far | %.0fs elapsed | ETA %.0fs",
            processed, total, total_entities, elapsed, eta_seconds,
        )

    logger.info(
        "Done: %d notes processed, %d entities inserted in %.0fs",
        processed, total_entities, time.time() - start,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NER over stored clinical notes and save the extracted entities."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of pending notes to process (default: all pending notes)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Notes per spaCy nlp.pipe() batch (default: 32)",
    )
    args = parser.parse_args()

    run(limit=args.limit, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

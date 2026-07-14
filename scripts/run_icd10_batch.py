"""
scripts/run_icd10_batch.py
────────────────────────────────────────────────────────────────
Batch ICD-10 mapper — completes the src/nlp/icd_mapper.py wiring.
Reads DISEASE/SYMPTOM entities that don't have ICD-10 mappings
yet, maps each *unique* entity text once (exact -> fuzzy ->
embedding, see ICD10Mapper.map), and fans the result out to every
entity row that shares that text. Persists via load_icd10_mappings().

Only DISEASE and SYMPTOM entities are eligible -- medications,
procedures, and anatomy terms use different code systems (see
ICD10Mapper.map_entities). Caching by unique text matters here:
206k eligible entity rows reduce to ~43k unique texts, so the
cache avoids redundant fuzzy/embedding lookups for repeated terms
like "hypertension".

Idempotent / resumable: an entity is only processed if it has
zero rows in icd10_mappings, so re-running after an interruption
never creates duplicates.

icd10_mappings only ever stores *found* matches -- an entity text
that the mapper couldn't confidently map has no row and therefore
still looks "pending" on the next run. Without a separate record
of what's already been tried, every restart would re-run the slow
fuzzy/embedding matching on every previously-checked no-match text.
A small on-disk cache (data/processed/icd10_mapping_cache.json,
keyed by entity text) closes that gap: any text seen before --
matched or not -- is skipped straight from the cache on restart.

Usage
─────
    python scripts/run_icd10_batch.py                # all pending entities
    python scripts/run_icd10_batch.py --limit 500     # quick demo subset
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import time

from sqlalchemy import select

from src.db.connection import get_session
from src.db.models import Entity, ICD10Mapping
from src.etl.load import load_icd10_mappings
from src.nlp.icd_mapper import ICD10Mapper
from src.utils.config import Paths
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ELIGIBLE_LABELS = ("DISEASE", "SYMPTOM")
_CACHE_PATH = Paths.processed / "icd10_mapping_cache.json"


def _load_cache() -> dict[str, list[dict]]:
    """Load the on-disk text -> match-dicts cache, or an empty dict."""
    if not _CACHE_PATH.exists():
        return {}
    with open(_CACHE_PATH, encoding="utf-8") as f:
        cache = json.load(f)
    logger.info("Loaded ICD-10 mapping cache: %d entity texts already tried", len(cache))
    return cache


def _save_cache(cache: dict[str, list[dict]]) -> None:
    """Write the text -> match-dicts cache to disk."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _fetch_pending_entities(limit: int | None) -> list[tuple[int, str]]:
    """Return (id, text) for DISEASE/SYMPTOM entities with no ICD-10 mapping yet.

    Args:
        limit: Maximum number of entity rows to return, or None for
            every pending entity.

    Returns:
        List of (entity_id, text) tuples, ordered by entity id so a
        --limit run always covers the same entities on re-run.
    """
    with get_session() as session:
        stmt = (
            select(Entity.id, Entity.text)
            .outerjoin(ICD10Mapping, ICD10Mapping.entity_id == Entity.id)
            .where(
                Entity.label.in_(_ELIGIBLE_LABELS),
                ICD10Mapping.id.is_(None),
            )
            .order_by(Entity.id)
        )
        if limit:
            stmt = stmt.limit(limit)
        return [(row.id, row.text) for row in session.execute(stmt)]


def run(limit: int | None, commit_every: int) -> None:
    """Map pending entities to ICD-10 codes and save the results.

    Args:
        limit: Maximum number of pending entity rows to process, or
            None for every DISEASE/SYMPTOM entity without a mapping.
        commit_every: Write to the database once this many mapping
            rows have been queued in memory.
    """
    pending = _fetch_pending_entities(limit)
    total_entities = len(pending)
    if total_entities == 0:
        logger.info(
            "No pending entities -- every DISEASE/SYMPTOM entity "
            "already has an ICD-10 mapping."
        )
        return

    by_text: dict[str, list[int]] = {}
    for entity_id, text in pending:
        by_text.setdefault(text, []).append(entity_id)

    unique_texts = list(by_text.keys())
    logger.info(
        "Mapping %d unique entity texts (%d entity rows) to ICD-10 codes...",
        len(unique_texts), total_entities,
    )

    mapper = ICD10Mapper()
    cache = _load_cache()
    cache_hits = 0

    start = time.time()
    processed_entities = 0
    inserted_total = 0
    pending_dicts: list[dict] = []

    for i, text in enumerate(unique_texts, start=1):
        if text in cache:
            match_dicts = cache[text]
            cache_hits += 1
        else:
            match_dicts = [m.to_dict() for m in mapper.map(text)]
            cache[text] = match_dicts

        entity_ids = by_text[text]
        for entity_id in entity_ids:
            for m in match_dicts:
                d = dict(m)
                d["entity_id"] = entity_id
                pending_dicts.append(d)

        processed_entities += len(entity_ids)

        if len(pending_dicts) >= commit_every or i == len(unique_texts):
            inserted_total += load_icd10_mappings(pending_dicts)
            pending_dicts = []
            _save_cache(cache)

            elapsed = time.time() - start
            rate = processed_entities / elapsed if elapsed > 0 else 0
            eta_seconds = (total_entities - processed_entities) / rate if rate > 0 else float("inf")
            logger.info(
                "  %d/%d entities | %d/%d unique texts (%d cache hits) | %d mappings so far | "
                "%.0fs elapsed | ETA %.0fs",
                processed_entities, total_entities, i, len(unique_texts), cache_hits,
                inserted_total, elapsed, eta_seconds,
            )

    logger.info(
        "Done: %d entities processed, %d mappings inserted in %.0fs",
        processed_entities, inserted_total, time.time() - start,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map DISEASE/SYMPTOM entities to ICD-10 codes and save the results."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of pending entity rows to process (default: all pending entities)",
    )
    parser.add_argument(
        "--commit-every", type=int, default=500,
        help="Queue this many mapping rows before writing to the database (default: 500)",
    )
    args = parser.parse_args()

    run(limit=args.limit, commit_every=args.commit_every)


if __name__ == "__main__":
    main()

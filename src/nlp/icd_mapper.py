"""
src/nlp/icd_mapper.py
────────────────────────────────────────────────────────────────
ICD-10 code mapping module.

Strategy — lookup-first, embedding fallback
────────────────────────────────────────────
This is the industry-standard approach for clinical coding:

  Step 1 — Exact match
    Direct dictionary lookup on the normalised entity text.
    Fast and unambiguous.  Most common clinical terms hit here.

  Step 2 — Fuzzy match (rapidfuzz)
    Levenshtein-based string matching against ICD-10 descriptions.
    Catches spelling variants and minor OCR errors.
    Returns a match when score >= ICD10Config.fuzzy_threshold (80).

  Step 3 — Embedding match (sentence-transformers)
    Encode both the entity and all ICD-10 descriptions into dense
    vectors, then find the closest by cosine similarity.
    Handles genuinely novel phrasing that fuzzy matching misses.
    Returns a match when similarity >= ICD10Config.embedding_threshold (0.75).
    This step is expensive (~100ms per entity) so it only runs
    when steps 1 and 2 produce no confident result.

  Step 4 — No match
    Returns an empty list with a LOW_CONFIDENCE flag so the
    caller can decide how to handle unknown entities.

Each step returns the top-k candidates, not just the best match.
This lets the dashboard show alternative codes and lets a human
reviewer choose the most appropriate one.

Confidence scores
─────────────────
  Exact match   → 1.0
  Fuzzy match   → fuzzy_score / 100.0
  Embedding     → cosine similarity score (0.0–1.0)
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import ICD10Config, Paths, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _strip_parens(text: str) -> str:
    """Remove parenthetical qualifiers and collapse whitespace.

    ICD-10 descriptions often carry a parenthetical qualifier, e.g.
    "Essential (primary) hypertension" — about 8% of the ~74k codes.
    Without stripping these, common, simply-phrased clinical terms
    ("essential hypertension") miss the exact match entirely and fall
    through to fuzzy matching, which can rank an unrelated same-length
    term (e.g. "Neonatal hypertension") above the correct one purely
    because it scores closer on string length.
    """
    stripped = re.sub(r"\s*\([^)]*\)", "", text)
    stripped = re.sub(r"\s{2,}", " ", stripped)
    return stripped.strip()


# Words that qualify/modify a clinical term without naming a distinct
# condition. Stripped out before computing "extra word" counts so that
# e.g. "Type 2 diabetes mellitus without complications" and "...with
# other specified complication" are correctly recognised as equally
# generic relative to the query "type 2 diabetes mellitus".
_GENERIC_QUALIFIER_WORDS: frozenset[str] = frozenset({
    "unspecified", "specified", "other", "type", "without", "with",
    "complication", "complications", "due", "to", "of", "in", "and",
    "or", "the", "a", "an", "not", "elsewhere", "classified",
    "organism", "agent", "cause", "origin", "site", "side",
})


def _content_words(text: str) -> set[str]:
    """Extract the clinically meaningful words from a description.

    Strips generic qualifier words (see :data:`_GENERIC_QUALIFIER_WORDS`)
    so that two phrasings of the same underlying condition compare equal
    regardless of which specific qualifier each one uses.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _GENERIC_QUALIFIER_WORDS and len(w) > 1}


def _is_unsafe_type_assertion(query: str, description: str) -> bool:
    """True if `description` asserts "type 1"/"type 2" but `query` doesn't.

    `_content_words()` deliberately strips single-character tokens, so
    "1" and "2" are invisible to the generic-vs-specific preference
    logic above — "diabetes mellitus", "type 1 diabetes mellitus", and
    "type 2 diabetes mellitus" all reduce to the same content-word set.
    That means a bare, type-less query is equally "close" to both Type 1
    and Type 2 ICD-10 entries, and whichever one a fuzzy/embedding
    matcher ranks first is essentially arbitrary — asserting the wrong
    type is a different diagnosis, not just a less precise one, so this
    blocks that specific failure mode rather than relying on string
    closeness to pick correctly.
    """
    desc_lower = description.lower()
    has_type1 = "type 1" in desc_lower or "type i " in desc_lower
    has_type2 = "type 2" in desc_lower or "type ii" in desc_lower
    if not (has_type1 or has_type2):
        return False

    query_lower = query.lower()
    query_has_type1 = "type 1" in query_lower or "type i " in f"{query_lower} "
    query_has_type2 = "type 2" in query_lower or "type ii" in query_lower

    if has_type1 and query_has_type1:
        return False
    if has_type2 and query_has_type2:
        return False
    return True


def _is_external_cause_code(icd10_code: str) -> bool:
    """True if `icd10_code` is an ICD-10 "external causes" code (V00-Y99).

    Chapter 20 codes describe HOW an injury happened (a fall, a
    collision, an accident) and are only ever valid as a supplementary
    code paired with an actual injury diagnosis — never a standalone
    match for a SYMPTOM or DISEASE entity, which is all this mapper is
    ever asked to map (see ``map_entities``). Fuzzy/embedding matching
    can still rank one highly off an unrelated word that happens to
    appear in its description (e.g. "stairs", from "worse with
    stairs", matching "Fall on stairs" — fabricating a fall that was
    never mentioned).
    """
    if not icd10_code:
        return False
    return icd10_code[0].upper() in "VWXY"


# ── Output dataclass ──────────────────────────────────────────────

@dataclass
class ICD10Match:
    """A single ICD-10 code candidate for an extracted entity.

    Attributes:
        icd10_code   : ICD-10-CM code (e.g. "I10").
        description  : Human-readable description.
        confidence   : Match confidence (0.0–1.0).
        match_method : How the match was found:
                       "exact", "fuzzy", "embedding", or "none".
        rank         : Position in the top-k list (1 = best).
    """

    icd10_code:   str
    description:  str
    confidence:   float
    match_method: str
    rank:         int = 1

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary for JSON responses."""
        return {
            "icd10_code":   self.icd10_code,
            "description":  self.description,
            "confidence":   round(self.confidence, 3),
            "match_method": self.match_method,
            "rank":         self.rank,
        }


class ICD10Mapper:
    """Maps clinical entity text to ICD-10-CM codes.

    Loads the ICD-10 reference table once at construction time.
    The embedding index is built lazily on first use — it takes
    a few seconds and is cached in memory for subsequent calls.

    Args:
        icd10_path : Path to the ICD-10 CSV file.
                     Defaults to the processed parquet if available,
                     falls back to the raw CSV.
        top_k      : Maximum number of candidate codes to return
                     per entity.

    Example::

        mapper   = ICD10Mapper()
        matches  = mapper.map("hypertension")
        for m in matches:
            print(m.icd10_code, m.description, m.confidence)
        # → I10  Essential (primary) hypertension  1.0
    """

    def __init__(
        self,
        icd10_path: Optional[Path] = None,
        top_k: int = ICD10Config.top_k,
    ) -> None:
        self._top_k       = top_k
        self._df          = self._load_reference_table(icd10_path)
        self._lookup, self._stripped_lookup = self._build_lookup(self._df)
        # Lowercased once up front — fuzz.token_set_ratio is case-sensitive,
        # and the query is always lowercased before matching, so comparing
        # against original-case descriptions silently degraded every score.
        self._descriptions_lower = self._df["description"].str.lower().tolist()
        self._embeddings  = None   # built lazily
        self._embed_model = None   # loaded lazily
        # Set True only after a real load attempt fails (import error,
        # or no local cache + no network). False until then — covers
        # both "never attempted yet" and "loaded successfully".
        self._embedding_unavailable = False
        logger.info(
            "ICD10Mapper ready: %d codes loaded", len(self._df)
        )

    @property
    def embedding_available(self) -> bool:
        """Whether semantic (embedding-based) ICD-10 matching is usable.

        ``True`` until a real load attempt fails. Exact and fuzzy
        matching are unaffected either way — this only governs the
        fallback used for entity text that doesn't match any ICD-10
        description lexically.
        """
        return not self._embedding_unavailable

    # ── Data loading ──────────────────────────────────────────────

    def _load_reference_table(self, path: Optional[Path]) -> pd.DataFrame:
        """Load the ICD-10 reference table from parquet or CSV.

        The ETL pipeline saves a parquet copy to data/processed/
        which is faster to load.  Falls back to the raw CSV if
        the processed version is not yet available.

        Args:
            path: Explicit path override, or None to auto-detect.

        Returns:
            DataFrame with columns ``icd10_code`` and ``description``.

        Raises:
            FileNotFoundError: If no ICD-10 file can be found.
        """
        # Priority: explicit path > processed parquet > raw CSV
        candidates = [
            p for p in [
                path,
                Paths.processed / "icd10_codes.parquet",
                Paths.icd10_csv,
            ]
            if p is not None and p.exists()
        ]

        if not candidates:
            raise FileNotFoundError(
                "ICD-10 reference file not found.\n"
                "Run the ETL pipeline first:\n"
                "  python -m src.etl.pipeline"
            )

        chosen = candidates[0]
        logger.debug("Loading ICD-10 from %s", chosen)

        df = (
            pd.read_parquet(chosen)
            if chosen.suffix == ".parquet"
            else pd.read_csv(chosen, dtype=str)
        )

        # Accept 'code' (from fetch_icd10.py) or 'icd10_code' (processed parquet)
        if "code" in df.columns and "icd10_code" not in df.columns:
            df = df.rename(columns={"code": "icd10_code"})

        # Ensure we have the columns we need
        required = {"icd10_code", "description"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"ICD-10 file is missing columns: {missing}. "
                f"Got: {list(df.columns)}"
            )

        # Normalise: strip whitespace, uppercase codes
        df["icd10_code"]   = df["icd10_code"].str.strip().str.upper()
        df["description"]  = df["description"].str.strip()
        df = df.dropna(subset=["icd10_code", "description"])

        return df.reset_index(drop=True)

    def _build_lookup(
        self, df: pd.DataFrame
    ) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
        """Build O(1) exact-match dictionaries from the ICD-10 table.

        Builds two lookups:
          - Primary: lowercase description → (code, original_description).
            Also indexes the code itself so "I10" → hypertension works.
          - Stripped: same, but with parenthetical qualifiers removed
            (see :func:`_strip_parens`).  Only populated when stripping
            actually changes the key, and on collision keeps the
            shortest original description (usually the more general /
            primary code).

        Args:
            df: ICD-10 reference DataFrame.

        Returns:
            Tuple of ``(primary_lookup, stripped_lookup)``.
        """
        lookup = {}
        stripped_lookup: dict[str, tuple[str, str]] = {}

        for _, row in df.iterrows():
            code = row["icd10_code"]
            desc = row["description"]
            key  = desc.lower().strip()

            lookup[key] = (code, desc)
            # Also index by code for reverse lookups
            lookup[code.lower()] = (code, desc)

            stripped_key = _strip_parens(key)
            if stripped_key and stripped_key != key:
                existing = stripped_lookup.get(stripped_key)
                if existing is None or len(desc) < len(existing[1]):
                    stripped_lookup[stripped_key] = (code, desc)

        return lookup, stripped_lookup

    # ── Mapping methods ───────────────────────────────────────────

    def map(self, entity_text: str) -> list[ICD10Match]:
        """Map an entity text string to ICD-10 candidates.

        Tries exact → fuzzy → embedding in order, stopping at the
        first method that returns a result above its threshold.

        Args:
            entity_text: Extracted entity text (e.g. "hypertension").
                         Cleaned text produces better results.

        Returns:
            List of :class:`ICD10Match` objects ordered by confidence.
            Empty list if no match is found above any threshold.
        """
        if not entity_text or not entity_text.strip():
            return []

        normalised = entity_text.lower().strip()

        # Step 1: exact match — O(1), always try first
        exact = self._exact_match(normalised)
        exact = [m for m in exact if not _is_external_cause_code(m.icd10_code)]
        if exact:
            return exact

        # Step 2: fuzzy match — ~10ms, good for spelling variants
        fuzzy = self._fuzzy_match(normalised)
        fuzzy = [
            m for m in fuzzy
            if not _is_unsafe_type_assertion(normalised, m.description)
            and not _is_external_cause_code(m.icd10_code)
        ]
        if fuzzy:
            return fuzzy

        # Step 3: embedding match — ~100ms, for novel phrasing
        embedding = self._embedding_match(normalised)
        embedding = [
            m for m in embedding
            if not _is_unsafe_type_assertion(normalised, m.description)
            and not _is_external_cause_code(m.icd10_code)
        ]
        return embedding

    def map_entities(
        self, entities: list
    ) -> dict[str, list[ICD10Match]]:
        """Map a list of Entity objects to ICD-10 candidates.

        Convenience wrapper around map() that processes a list
        of Entity objects from the NER pipeline.

        Args:
            entities: List of :class:`src.nlp.ner.Entity` objects.

        Returns:
            Dict mapping entity text → list of ICD10Match objects.
        """
        results = {}
        for entity in entities:
            # Only map disease and symptom entities — medications,
            # procedures, and anatomy terms have their own code systems
            if entity.label in ("DISEASE", "SYMPTOM"):
                matches = self.map(entity.text)
                if matches:
                    results[entity.text] = matches
        return results

    # ── Match implementations ─────────────────────────────────────

    def _exact_match(self, text: str) -> list[ICD10Match]:
        """Direct dictionary lookup for exact description matches.

        Tries the primary lookup first, then falls back to matching
        against descriptions with parenthetical qualifiers stripped
        (see :func:`_strip_parens`) — this lets common, simply-phrased
        terms like "essential hypertension" match ``I10`` directly
        instead of falling through to noisy fuzzy matching.

        Args:
            text: Lowercase, stripped entity text.

        Returns:
            List with one match at confidence=1.0, or empty list.
        """
        result = self._lookup.get(text) or self._stripped_lookup.get(_strip_parens(text))
        if result:
            code, description = result
            return [ICD10Match(
                icd10_code   = code,
                description  = description,
                confidence   = 1.0,
                match_method = "exact",
                rank         = 1,
            )]
        return []

    def _fuzzy_match(self, text: str) -> list[ICD10Match]:
        """Fuzzy string matching against ICD-10 descriptions.

        Uses rapidfuzz's token_set_ratio, which scores a query as a
        perfect match against any description that contains all of
        the query's words plus extras — exactly the "base condition
        plus complication/qualifier" pattern ICD-10 uses heavily
        (e.g. "Type 2 diabetes mellitus {without complications,
        with hyperglycemia, with foot ulcer, ...}"). That produces
        many tied top-scoring candidates, so the actual selection
        happens via two tiebreakers, applied in order:

          1. Fewest "extra" clinically-meaningful words beyond the
             query (see :func:`_content_words`) — prefers the
             general/default code over a specific complication when
             the query didn't mention one.
          2. Shortest description — a final tiebreak between
             phrasings that reduce to the same extra-word count
             (e.g. "without complications" vs "with other specified
             complication" both strip to zero extra words).

        Without tiebreaker 1, "essential hypertension" or "unspecified
        pneumonia" rank an unrelated same-length term above the
        correct code purely because it scores closer on raw string
        similarity. Without tiebreaker 2, "type 2 diabetes mellitus"
        picks an arbitrary complication code instead of E11.9.

        A full scan over all descriptions is used rather than a small
        top-N candidate pool — ties at the top score can number in
        the dozens (86 for the diabetes example above), and a small
        pool risks truncating before reaching the correct candidate.
        rapidfuzz is fast enough that this costs well under a second
        even across 74k descriptions.

        Args:
            text: Lowercase, stripped entity text.

        Returns:
            Top-k matches above the fuzzy threshold, or empty list
            (callers should fall through to the embedding match).
        """
        try:
            from rapidfuzz import process, fuzz
        except ImportError:
            logger.warning(
                "rapidfuzz not installed — fuzzy matching disabled. "
                "Run: pip install rapidfuzz"
            )
            return []

        query_words = _content_words(text)

        candidates = process.extract(
            text,
            self._descriptions_lower,
            scorer = fuzz.token_set_ratio,
            limit  = len(self._descriptions_lower),
        )

        qualifying = []
        for desc, score, idx in candidates:
            if score < ICD10Config.fuzzy_threshold:
                break  # sorted descending — nothing further clears the bar
            desc_words = _content_words(desc)
            if query_words and not query_words.issubset(desc_words):
                continue  # candidate doesn't actually contain the query's terms
            extra = len(desc_words - query_words)
            qualifying.append((extra, len(desc), -score, idx, score))

        if not qualifying:
            return []

        qualifying.sort()

        # Only auto-resolve when the best candidate is a true generic/
        # default equivalent (zero extra clinically-meaningful words).
        # If even the closest candidate still requires adding a genuinely
        # distinguishing word (e.g. bare "hypertension" only ever appears
        # alongside "portal", "renal", "essential" etc. — ICD-10 has no
        # plain "Hypertension, unspecified" code), guessing one of those
        # specific variants would silently produce a wrong diagnosis code.
        # Defer to the embedding step instead, same as if fuzzy found
        # nothing at all.
        if qualifying[0][0] != 0:
            return []

        matches = []
        for extra, _len, _neg_score, idx, score in qualifying[: self._top_k]:
            row = self._df.iloc[idx]
            matches.append(ICD10Match(
                icd10_code   = row["icd10_code"],
                description  = row["description"],
                confidence   = score / 100.0,
                match_method = "fuzzy",
                rank         = len(matches) + 1,
            ))

        return matches

    def _embedding_cache_path(self) -> Path:
        """Return the on-disk cache path for the ICD-10 embedding index."""
        return Paths.processed / "icd10_embeddings.pt"

    def _load_or_build_embedding_index(self):
        """Load the cached ICD-10 embedding index from disk, or build it.

        Encoding 74k descriptions through BERT on CPU takes 15-40+
        minutes, so the result is cached to disk and only rebuilt if
        the ICD-10 table size changes (e.g. a new CMS release).

        Returns:
            A tensor of shape (n_descriptions, embedding_dim).
        """
        import torch

        cache_path = self._embedding_cache_path()
        if cache_path.exists():
            cached = torch.load(cache_path)
            if cached.shape[0] == len(self._df):
                logger.info(
                    "Loaded cached ICD-10 embedding index from %s (%d vectors)",
                    cache_path, cached.shape[0],
                )
                return cached
            logger.info(
                "Cached embedding index size (%d) does not match current "
                "ICD-10 table (%d) — rebuilding.",
                cached.shape[0], len(self._df),
            )

        # Try downloading a precomputed copy from HF Hub before resorting
        # to a 15-40+ minute CPU rebuild — e.g. a fresh deployment
        # container that doesn't bundle data/processed/ locally (the
        # file is too large to commit to git).
        from src.utils.config import ModelConfig
        hub_repo = ModelConfig.icd10_embeddings_hf_hub_fallback
        if hub_repo:
            try:
                from huggingface_hub import hf_hub_download
                logger.info(
                    "No local embedding cache — downloading precomputed "
                    "index from HF Hub (%s) instead of rebuilding from "
                    "scratch.", hub_repo,
                )
                downloaded_path = hf_hub_download(
                    repo_id  = hub_repo,
                    repo_type = "dataset",
                    filename = "icd10_embeddings.pt",
                )
                cached = torch.load(downloaded_path)
                if cached.shape[0] == len(self._df):
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(cached, cache_path)
                    logger.info(
                        "Loaded embedding index from HF Hub (%d vectors), "
                        "cached locally to %s", cached.shape[0], cache_path,
                    )
                    return cached
                logger.warning(
                    "HF Hub embedding index size (%d) does not match "
                    "current ICD-10 table (%d) — rebuilding instead.",
                    cached.shape[0], len(self._df),
                )
            except Exception as exc:
                logger.warning(
                    "Could not download embedding index from HF Hub "
                    "(%s) — rebuilding from scratch instead.", exc,
                )

        logger.info(
            "Building ICD-10 embedding index (%d descriptions) — "
            "one-time cost on CPU, can take 15-40+ minutes. "
            "Progress bar below; result will be cached to disk.",
            len(self._df),
        )
        embeddings = self._embed_model.encode(
            self._df["description"].tolist(),
            batch_size        = 128,
            show_progress_bar = True,
            convert_to_tensor = True,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_path)
        logger.info("Embedding index ready and cached to %s ✓", cache_path)
        return embeddings

    def _embedding_match(self, text: str) -> list[ICD10Match]:
        """Semantic similarity matching using sentence embeddings.

        Encodes the entity text and all ICD-10 descriptions into
        dense vectors, then finds the closest descriptions by
        cosine similarity.

        The embedding index is built once and cached both in memory
        and on disk (see :meth:`_load_or_build_embedding_index`).
        First call ever takes 15-40+ minutes; every call after that
        (including in future notebook sessions) loads from disk in
        under a second.

        Args:
            text: Lowercase, stripped entity text.

        Returns:
            Top-k matches above the embedding threshold, or empty list.
        """
        # Once a real attempt has failed, don't keep retrying a known-
        # broken import/load on every single call — fail fast instead.
        if self._embedding_unavailable:
            return []

        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer, util

            # Load the embedding model once and cache it.
            # Try the local cache first — sentence-transformers otherwise
            # always does a handful of network round-trips (HEAD requests
            # for adapter_config.json etc.) even when the model is fully
            # cached, which crashes the whole pipeline on any transient
            # network blip. Only fall back to a live download if nothing
            # is cached yet.
            if self._embed_model is None:
                from src.utils.config import ModelConfig, force_offline_hf_env
                model_name = ModelConfig.embedding_model
                try:
                    with force_offline_hf_env():
                        self._embed_model = SentenceTransformer(
                            model_name, local_files_only=True
                        )
                    logger.info(
                        "Loaded embedding model from local cache (offline): %s",
                        model_name,
                    )
                except Exception:
                    logger.info(
                        "Embedding model not found in local cache — "
                        "downloading from Hugging Face Hub: %s (first call only)",
                        model_name,
                    )
                    self._embed_model = SentenceTransformer(model_name)

            # Build (or load from disk cache) the ICD-10 embedding index.
            # The cache is saved from whatever device originally built it
            # (often CPU) -- move it onto the embedding model's device so
            # cos_sim() below never sees a CPU/CUDA tensor mismatch.
            if self._embeddings is None:
                self._embeddings = self._load_or_build_embedding_index()
                self._embeddings = self._embeddings.to(self._embed_model.device)

            query_embedding = self._embed_model.encode(
                text, convert_to_tensor=True
            )
            scores = util.cos_sim(query_embedding, self._embeddings)[0]
            top_indices = scores.argsort(descending=True)[: self._top_k * 2]

            matches = []
            for idx in top_indices:
                score = float(scores[idx])
                if score < ICD10Config.embedding_threshold:
                    break
                row = self._df.iloc[int(idx)]
                matches.append(ICD10Match(
                    icd10_code   = row["icd10_code"],
                    description  = row["description"],
                    confidence   = score,
                    match_method = "embedding",
                    rank         = len(matches) + 1,
                ))
                if len(matches) >= self._top_k:
                    break

            return matches

        except ImportError:
            self._embedding_unavailable = True
            logger.warning(
                "sentence-transformers not installed — "
                "embedding fallback disabled. "
                "Run: pip install sentence-transformers"
            )
            return []
        except Exception as exc:
            # Never let an embedding-fallback failure crash the whole
            # /notes/analyse request — exact/fuzzy matching still works.
            # Mark unavailable so we don't keep retrying a known-broken
            # load (e.g. resource contention during import) on every call.
            self._embedding_unavailable = True
            logger.warning(
                "Embedding fallback failed and is now disabled for this "
                "process (%s: %s). Exact/fuzzy ICD-10 matching is "
                "unaffected; restart the process to retry.",
                type(exc).__name__, exc,
            )
            return []

    # ── Utility ───────────────────────────────────────────────────

    def describe(self, icd10_code: str) -> Optional[str]:
        """Return the description for an ICD-10 code.

        Args:
            icd10_code: ICD-10-CM code (case-insensitive).

        Returns:
            Description string, or None if the code is not found.
        """
        result = self._lookup.get(icd10_code.upper().strip())
        if result:
            return result[1]
        # Try the lower-case key
        result = self._lookup.get(icd10_code.lower().strip())
        return result[1] if result else None

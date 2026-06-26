"""
src/utils/text_utils.py
────────────────────────────────────────────────────────────────
Text cleaning and preprocessing utilities for clinical notes.

Clinical text is messy in ways that general-purpose NLP tools
do not handle well:
  - Transcription artefacts ("XXXX" for redacted names)
  - Section headers in all-caps ("HISTORY OF PRESENT ILLNESS:")
  - Non-standard whitespace from copy-paste out of EMR systems
  - Mixed abbreviations ("h/o", "s/p", "c/o", "w/")
  - Fragmented sentences and bullet points

These helpers normalise text before it enters the NER pipeline
or the classifier, without destroying clinical meaning.

All functions are pure (no side effects) and return new strings
rather than modifying in place.  This makes them safe to use
in parallel pipelines.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import unicodedata


# ── Common clinical abbreviation expansions ───────────────────────
# Expand before NER so the model sees full words, not shorthand.
# Keys are lowercase; matching is case-insensitive.
_ABBREVIATIONS: dict[str, str] = {
    r"\bh/o\b":   "history of",
    r"\bs/p\b":   "status post",
    r"\bc/o\b":   "complains of",
    r"\bw/\b":    "with",
    r"\bw/o\b":   "without",
    r"\bp/w\b":   "presenting with",
    r"\bpt\b":    "patient",
    r"\bpts\b":   "patients",
    r"\bhtn\b":   "hypertension",
    r"\bdm\b":    "diabetes mellitus",
    r"\bcad\b":   "coronary artery disease",
    r"\bcopd\b":  "chronic obstructive pulmonary disease",
    r"\buti\b":   "urinary tract infection",
    r"\bcva\b":   "cerebrovascular accident",
    r"\bmi\b":    "myocardial infarction",
    r"\bgerd\b":  "gastroesophageal reflux disease",
    r"\bsob\b":   "shortness of breath",
    r"\bcp\b":    "chest pain",
    r"\bn/v\b":   "nausea and vomiting",
    r"\bf/u\b":   "follow up",
    # optional periods between letters so "q.d.", "b.i.d." etc. also match,
    # not just the period-free forms "qd", "bid"
    r"\bq\.?d\.?\b":    "once daily",
    r"\bb\.?i\.?d\.?\b": "twice daily",
    r"\bt\.?i\.?d\.?\b": "three times daily",
    r"\bq\.?i\.?d\.?\b": "four times daily",
    r"\bprn\b":   "as needed",
    r"\bpo\b":    "by mouth",
    r"\biv\b":    "intravenous",
    r"\bim\b":    "intramuscular",
    r"\bsc\b":    "subcutaneous",
    r"\bnpo\b":   "nothing by mouth",
    r"\bicu\b":   "intensive care unit",
    r"\bed\b":    "emergency department",
    # NOTE: deliberately no "or" -> "operating room" entry. "or" is
    # overwhelmingly the common English conjunction ("headache or sore
    # throat") in clinical text; blindly expanding it corrupted notes
    # system-wide (e.g. "headache or sore throat" -> "headache operating
    # room sore throat"). The rare genuine OR-abbreviation usage is
    # almost always written in caps and is not worth this tradeoff.
}

# Placeholder patterns that indicate redacted or missing information.
# Replace with a neutral token rather than deleting (deletion shifts
# character offsets and breaks entity span extraction).
_REDACTION_PATTERNS: list[str] = [
    r"X{2,}",        # "XXXX" redacted names
    r"\[.*?\]",      # [REDACTED], [NAME], [DATE]
    r"\*{2,}",       # **** redacted fields
    r"_{3,}",        # ___ blank fields
]


def clean_clinical_text(text: str, expand_abbreviations: bool = True) -> str:
    """Normalise a raw clinical note for downstream NLP processing.

    Steps applied in order:
      1. Decode unicode characters to ASCII equivalents where possible
      2. Normalise whitespace (tabs, multiple spaces, Windows line endings)
      3. Replace redaction placeholders with a neutral token
      4. Optionally expand common clinical abbreviations
      5. Fix common transcription quirks

    Args:
        text: Raw clinical note text as a string.
        expand_abbreviations: Whether to expand clinical shorthand
            (e.g. "h/o" → "history of").  Disable when you need the
            original wording preserved (e.g. for exact string matching).

    Returns:
        Cleaned text string.  Length may differ from input due to
        whitespace normalisation and abbreviation expansion.

    Example::

        raw = "Pt c/o SOB and CP. H/o HTN. XXXX denies N/V."
        clean_clinical_text(raw)
        # → "Patient complains of shortness of breath and chest pain.
        #    History of hypertension. [ANON] denies nausea and vomiting."
    """
    if not text or not text.strip():
        return ""

    # Step 1: Unicode normalisation — convert curly quotes, em-dashes etc.
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # Step 2: Normalise line endings and tabs to single spaces,
    # but preserve paragraph breaks (double newlines) as sentence
    # boundaries for the segmenter.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse 3+ newlines to 2
    text = re.sub(r"[ \t]+", " ", text)      # collapse horizontal whitespace
    text = text.strip()

    # Step 3: Replace redaction placeholders
    for pattern in _REDACTION_PATTERNS:
        text = re.sub(pattern, "[ANON]", text, flags=re.IGNORECASE)

    # Step 4: Expand abbreviations
    if expand_abbreviations:
        text = expand_clinical_abbreviations(text)

    # Step 5: Fix double-punctuation from expansion side effects
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def expand_clinical_abbreviations(text: str) -> str:
    """Replace clinical shorthand with full words.

    Matches are case-insensitive and word-boundary aware so "pt"
    in "patient" is not incorrectly expanded.

    Args:
        text: Text containing clinical abbreviations.

    Returns:
        Text with abbreviations replaced by their expansions.
    """
    for pattern, expansion in _ABBREVIATIONS.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


def segment_into_sections(text: str) -> dict[str, str]:
    """Split a clinical note into its named sections.

    Common section headers (CHIEF COMPLAINT, HISTORY OF PRESENT
    ILLNESS, MEDICATIONS, etc.) are detected by their all-caps
    formatting followed by a colon.

    Args:
        text: A full clinical note, potentially multi-section.

    Returns:
        Dictionary mapping section name (lowercase, underscored)
        to section text.  An ``"unsectioned"`` key holds any text
        that appears before the first header.

    Example::

        note = "CHIEF COMPLAINT: chest pain\\nHISTORY: ..."
        sections = segment_into_sections(note)
        sections["chief_complaint"]  # → "chest pain"
        sections["history"]          # → "..."
    """
    # Match headers like "CHIEF COMPLAINT:" or "PAST MEDICAL HISTORY:"
    header_pattern = re.compile(
        r"^([A-Z][A-Z\s/\-]{2,50}):\s*", re.MULTILINE
    )

    sections: dict[str, str] = {}
    last_key = "unsectioned"
    last_end = 0

    for match in header_pattern.finditer(text):
        # Save everything from the previous header to this one
        sections[last_key] = text[last_end: match.start()].strip()
        last_key = _normalise_section_name(match.group(1))
        last_end = match.end()

    # Save the final section
    sections[last_key] = text[last_end:].strip()

    # Remove empty sections
    return {k: v for k, v in sections.items() if v}


def _normalise_section_name(raw: str) -> str:
    """Convert a section header to a clean dictionary key.

    Args:
        raw: Header text as matched (e.g. ``'CHIEF COMPLAINT'``).

    Returns:
        Lowercase, underscore-separated key
        (e.g. ``'chief_complaint'``).
    """
    return re.sub(r"[\s/\-]+", "_", raw.strip().lower())


def truncate_to_tokens(
    text: str,
    max_tokens: int = 512,
    words_per_token: float = 0.75,
) -> str:
    """Truncate text to approximately `max_tokens` BERT tokens.

    BERT-family models have a 512-token limit.  Rather than
    letting the tokeniser silently truncate (which loses the end
    of the note), this function truncates at a sentence boundary
    close to the limit.

    The ``words_per_token`` ratio (default 0.75) is an empirical
    approximation for clinical English — clinical text tends to
    have more medical terminology, which tokenises into more
    subword pieces than ordinary English.

    Args:
        text: Input text to truncate.
        max_tokens: Maximum number of BERT tokens (default 512).
        words_per_token: Approximate words-per-token ratio.

    Returns:
        Truncated text ending at the nearest sentence boundary.
    """
    max_words = int(max_tokens * words_per_token)
    words     = text.split()

    if len(words) <= max_words:
        return text

    # Truncate to max_words and find the last sentence boundary
    truncated = " ".join(words[:max_words])
    last_stop = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
    )

    if last_stop > len(truncated) * 0.5:
        # A sentence boundary exists in the second half — use it
        return truncated[: last_stop + 1].strip()

    # No clean boundary found — fall back to hard word truncation
    return truncated.strip()


def extract_numeric_values(text: str) -> list[dict[str, str]]:
    """Extract numeric measurements from clinical text.

    Finds values like "120/80 mmHg", "98.6°F", "Hb 11.2 g/dL",
    "SpO2 94%".

    Args:
        text: Clinical note text.

    Returns:
        List of dicts with keys ``'value'``, ``'unit'``, ``'context'``.
        Context is the surrounding 20 characters for disambiguation.

    Example::

        extract_numeric_values("BP was 130/85 mmHg, temp 37.2 C")
        # → [
        #     {"value": "130/85", "unit": "mmHg", "context": "BP was 130/85 mmHg"},
        #     {"value": "37.2",   "unit": "C",    "context": "temp 37.2 C"},
        #   ]
    """
    pattern = re.compile(
        r"(\d+(?:[./]\d+)?)\s*"          # numeric value, possibly fractional
        r"(mmHg|bpm|mg/dL|g/dL|mmol/L|"  # common clinical units
        r"mEq/L|IU/L|U/L|ng/mL|mcg/mL|"
        r"%|°?[CF]|°?C|kg|lbs?|cm|mm|"
        r"mL|L|mg|mcg|g)\b",
        re.IGNORECASE,
    )

    results = []
    for match in pattern.finditer(text):
        start   = max(0, match.start() - 10)
        end     = min(len(text), match.end() + 10)
        results.append({
            "value":   match.group(1),
            "unit":    match.group(2),
            "context": text[start:end].strip(),
        })

    return results


def word_count(text: str) -> int:
    """Return the number of whitespace-delimited words in text.

    Args:
        text: Any string.

    Returns:
        Integer word count.
    """
    return len(text.split()) if text and text.strip() else 0

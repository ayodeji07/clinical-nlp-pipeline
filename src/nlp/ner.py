"""
src/nlp/ner.py
────────────────────────────────────────────────────────────────
Named Entity Recognition pipeline for clinical text.

Architecture
────────────
A thin interface (BaseNERPipeline) sits in front of the actual
implementation (SpacyNERPipeline).  This makes the NER model
swappable without touching the ETL, API, or dashboard.

The scispaCy model (en_core_sci_lg) is trained on PubMed
abstracts and the CRAFT corpus.  It recognises biomedical
entities but does not distinguish between sub-types out of
the box.  We apply a post-processing step that maps the raw
entity labels to our five clinical categories:

  Raw scispaCy label → Our label
  ───────────────────────────────
  DISEASE             → DISEASE
  CHEMICAL            → MEDICATION
  Any entity matching a procedure keyword list → PROCEDURE
  Any entity matching an anatomy term list     → ANATOMY
  Remaining entities                           → SYMPTOM

This mapping is deliberately simple and can be refined once
real labelled data is available.

Installation
────────────
  pip install spacy scispacy
  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/ \
      releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from src.utils.config import ModelConfig, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Output dataclass ──────────────────────────────────────────────

@dataclass
class Entity:
    """A single named entity extracted from clinical text.

    Attributes:
        text       : The extracted text as it appears in the note.
        label      : Normalised entity type — one of DISEASE,
                     MEDICATION, PROCEDURE, SYMPTOM, ANATOMY.
        start      : Start character offset in the source text.
        end        : End character offset in the source text.
        confidence : Model confidence score (0.0–1.0).
                     None when the model does not provide scores.
        note_id    : Optional database ID of the parent note.
                     Populated when processing stored notes.
    """

    text:       str
    label:      str
    start:      int
    end:        int
    confidence: Optional[float] = None
    note_id:    Optional[int]   = None

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary for JSON responses.

        Returns:
            Dict with all fields; confidence is rounded to 3 d.p.
            if present.
        """
        return {
            "text":       self.text,
            "label":      self.label,
            "start":      self.start,
            "end":        self.end,
            "confidence": round(self.confidence, 3) if self.confidence else None,
            "note_id":    self.note_id,
        }


# ── Label mapping ─────────────────────────────────────────────────
# Maps scispaCy's raw entity labels to our five clinical categories.
# en_core_sci_lg emits only "ENTITY"; specialised models (bc5cdr, etc.)
# emit DISEASE / CHEMICAL — both cases handled below.
_LABEL_MAP: dict[str, str] = {
    "DISEASE":   "DISEASE",
    "CHEMICAL":  "MEDICATION",
    "DRUG":      "MEDICATION",
    "PROCEDURE": "PROCEDURE",
    "ANATOMY":   "ANATOMY",
    "ORGAN":     "ANATOMY",
    "CELL":      "ANATOMY",
    "GENE":      "DISEASE",
    "SPECIES":   "DISEASE",
}

# Standalone fragments that are only meaningful as part of a longer phrase.
# e.g. bc5cdr tags "mellitus" from "type 2 diabetes mellitus" as a separate span.
_FRAGMENT_TERMS: frozenset[str] = frozenset({
    "mellitus", "magna", "vera", "simplex", "complex",
    "syndrome", "disease", "disorder", "condition",
})

# A span starting with one of these describes the ABSENCE of a finding
# ("denies obesity", "no evidence of fracture") — reporting it as DISEASE
# would claim the patient HAS the condition they explicitly don't.
_NEGATION_PREFIXES: tuple[str, ...] = (
    "denies", "denied", "no evidence of", "no history of",
    "without", "negative for", "ruled out", "r/o",
    "not consistent with", "absence of",
)

# A span starting with one of these is a leaked section-header / reporting
# fragment from an imperfect NER span boundary ("complaint of allergies"),
# not the entity itself.
_GENERIC_REPORT_PREFIXES: tuple[str, ...] = (
    "complaint of", "complaints of", "complains of", "history of", "c/o",
)

# Terms that are never clinically useful entities (section headers,
# demographic words, generic document words, dosing qualifiers,
# directional / generic anatomical qualifiers).
_SKIP_TERMS: frozenset[str] = frozenset({
    # demographics / section headers
    "patient", "patients", "male", "female", "man", "woman",
    "history", "impression", "assessment", "plan", "findings",
    "medications", "medication", "allergies", "allergy",
    "subjective", "objective", "note", "notes", "discharge",
    "summary", "report",
    # dosing qualifiers
    "once", "twice", "daily", "weekly",
    "twice daily", "once daily", "three times", "four times",
    "per day", "per week", "tab", "tabs", "tablet", "tablets",
    "capsule", "capsules", "mg", "mcg", "units",
    # generic / directional words that look like entities but aren't
    "area", "region", "site", "location", "level", "side",
    "right", "left", "right side", "left side", "bilateral",
    "proximal", "distal", "anterior", "posterior",
    "superior", "inferior", "medial", "lateral",
    # generic clinical document words
    "procedure", "procedures", "technique", "approach",
    "identified", "noted", "seen", "given", "placed", "performed",
    "stable condition", "stable", "normal", "negative",
    "positive", "anon", "anonymous",
    # vague clinical descriptors that appear as isolated entities
    "attention", "complication", "complications", "elevated",
    "day", "days", "week", "weeks", "month", "months",
    "local", "general", "incidental", "significant",
    "increased", "decreased", "moderate", "mild", "severe",
    "mild to moderate", "moderate to severe",
    "port", "head", "operative site", "operative", "postoperative",
    "intraoperative", "perioperative",
    # quality/appearance descriptors — not entities on their own
    "purulent", "serous", "serosanguinous", "sanguineous",
    "hemorrhagic", "haemorrhagic", "fibrinous",
    # negative findings — absence of a symptom, not a reportable entity
    "afebrile", "asymptomatic", "nontender", "non-tender",
    # bare adjectives / incomplete fragments (real usage always pairs
    # these with a noun, e.g. "undescended testis", "allergic reaction",
    # "diabetic neuropathy", "pitting edema", "H. pylori" with "H." split
    # off as a false sentence-end abbreviation)
    "allergic", "undescended", "diabetic", "pitting", "pylori",
    # generic verbs / state words picked up as spurious entities —
    # found via gold-standard sample review (2026-06-26): these carry
    # no clinical meaning on their own ("the airway would improve",
    # "before she leaves", "any problem with her going home")
    "improve", "improves", "improved", "episodes", "infiltrated",
    "outside", "seated", "admission", "leaves", "problem", "problems",
    "instantaneous", "integrity", "unimproved", "insertion",
    "warm", "clear", "healthy-appearing", "years", "smoke", "smoking",
    # section-header fragments leaking through as bare entities
    "operation", "operations", "systems",
    # measurement units (already have mg/mcg/units; mmHg was missing)
    "mmhg",
    # equipment / supply names, not clinical entities
    "cuff", "webril", "hand reamer",
    # lab-test / panel names and social-history substances that bc5cdr
    # also tags CHEMICAL, which would otherwise be trusted as MEDICATION
    # at step 4 — these are not medications being administered
    "alcohol", "tobacco", "nicotine", "cholesterol", "glucose", "ana",
    "creatinine", "hemoglobin", "sodium", "potassium", "calcium",
    "triglycerides", "bilirubin",
    # anatomy abbreviation bc5cdr sometimes mistags as CHEMICAL
    "mca",
    # more generic verbs/fragments/connectors found in second review pass
    # (2026-06-26) — meaningless as standalone entities
    "this", "that", "insidious", "evaluate", "team", "study", "decision",
    "distribution", "patient lives", "prescription", "consistent",
    "consistent with", "caring", "dear doctor",
    # lab test name (not a symptom or medication)
    "inr",
    # exam/test names that don't fit DISEASE/SYMPTOM/PROCEDURE cleanly
    "romberg",
    # medical supplies/materials, not clinical entities
    "stockinette", "sterile saline", "stitch",
    # social-history item, not a clinical finding
    "tattoos", "tattoo",
    # section-header fragment
    "past surgical history",
    # dosing-frequency abbreviations with periods (the word-boundary
    # regex in text_utils.py's abbreviation expander only matches
    # period-free forms like "qd"; "q.d" with periods slips through)
    "q.d", "b.i.d", "t.i.d", "q.i.d", "q.d.", "b.i.d.", "t.i.d.", "q.i.d.",
    "p.o", "p.o.",
    # role/department abbreviations, not clinical entities
    "pcp", "appointment", "emergency department",
})

# Keywords that signal a procedure mention.
_PROCEDURE_KEYWORDS: frozenset[str] = frozenset({
    # surgical actions
    "surgery", "surgical", "operation", "procedure", "resection",
    "excision", "biopsy", "osteotomy",
    "incision", "dissection", "closure", "suture", "ligat",
    "anastomosis", "hemostasis", "electrocautery", "cauterization",
    "angioplasty", "stenting", "intubation", "ventilation",
    "dialysis", "transfusion", "transplant", "repair",
    "reconstruction", "amputation", "debridement", "drainage",
    "aspiration", "lavage", "catheterization", "catheterisation",
    "appendectomy", "cholecystectomy", "hysterectomy",
    "mastectomy", "colostomy", "tracheostomy", "orchiectomy",
    # anaesthesia / sedation
    "anesthesia", "anaesthesia", "anesthetic", "anaesthetic",
    "sedation", "intubation",
    # devices placed during procedures
    "stent", "catheter", "drain", "cannula",
    # imaging / diagnostics
    "ct scan", "mri", "x-ray", "ultrasound", "echocardiogram",
    "electrocardiogram", "ekg", "ecg", "eeg",
    # interventional cardiology / common abbreviations
    "pci", "cabg", "ptca", "ercp", "cath",
    "endoscopy", "colonoscopy", "bronchoscopy", "laparoscopy",
})

# Disease/diagnosis abbreviations and phrases the model often misses.
_DISEASE_TERMS: frozenset[str] = frozenset({
    "stemi", "nstemi", "acs", "mi", "chf", "hf",
    "copd", "ckd", "esrd", "afib", "af", "vt", "vf",
    "dvt", "pe", "uti", "cad", "dm", "htn",
    "gerd", "ibd", "ibs", "ms", "als", "ra", "sle", "hiv",
    "st elevation", "st depression", "st changes",
})

# Common drug-name suffixes — catch most small-molecule medications.
_MEDICATION_SUFFIXES: tuple[str, ...] = (
    "metformin", "insulin",
    "statin", "vastatin",        # atorvastatin, simvastatin …
    "pril",                      # lisinopril, enalapril …
    "sartan",                    # losartan, valsartan …
    "olol",                      # metoprolol, atenolol …
    "dipine",                    # amlodipine, nifedipine …
    "azole",                     # fluconazole, omeprazole …
    "mycin", "cillin", "cycline",# antibiotics
    "afil",                      # sildenafil, tadalafil …
    "mab", "umab", "ximab",      # monoclonal antibodies
    "tide", "tide",              # peptide drugs
    "aspirin", "warfarin", "heparin", "clopidogrel",
    "prednisone", "prednisolone", "dexamethasone",
    "morphine", "codeine", "oxycodone", "fentanyl",
    "amoxicillin", "azithromycin", "ciprofloxacin",
    "furosemide", "spironolactone", "hydrochlorothiazide",
    "albuterol", "salbutamol", "budesonide",
    "levothyroxine", "thyroxine",
)

# Medication-form / dosage-form words. A mention combining one of these
# with anything else ("cough syrup", "eye drops") is describing a
# medication being taken, regardless of what label the NER model assigned —
# bc5cdr occasionally mistags an OTC remedy as DISEASE in context.
_MEDICATION_FORM_WORDS: tuple[str, ...] = (
    # NOTE: deliberately no "capsule" — it's genuinely ambiguous in
    # clinical text (medication form vs. anatomical structure: joint
    # capsule, lens capsule, renal capsule), and the false-positive risk
    # outweighs the narrow benefit here.
    "syrup", "lozenge", "ointment",
    "cream", "lotion", "suspension", "drops", "inhaler",
)

# Anatomical term signals
_ANATOMY_KEYWORDS: frozenset[str] = frozenset({
    "heart", "lung", "liver", "kidney", "brain", "spine",
    "abdomen", "abdominal", "thorax", "pelvis", "femur", "tibia", "fibula",
    "conjunctiva", "conjunctival", "cornea", "corneal", "retina", "retinal",
    "artery", "vein", "aorta", "ventricle", "atrium",
    "cortex", "cerebellum", "cerebrum", "frontal", "parietal",
    "temporal", "occipital", "trachea", "bronchus", "alveoli",
    "esophagus", "oesophagus", "stomach", "duodenum", "colon",
    "rectum", "bladder", "ureter", "urethra", "prostate",
    "uterus", "ovary", "testis", "arm", "leg", "chest",
    "shoulder", "knee", "hip", "ankle", "wrist", "elbow",
    # additional limb / joint / extremity terms (same "X + pain" trap —
    # ICD-10 codes these under musculoskeletal M-codes, not R-codes)
    "back", "neck", "foot", "feet", "hand", "hands",
    "finger", "fingers", "toe", "toes", "groin", "flank",
    "joint", "joints", "limb", "limbs", "forearm",
    "calf", "thigh", "heel", "sole", "palm", "molar",
    "quadriceps", "metatarsal", "metatarsophalangeal",
    # additional surface / soft-tissue structures
    "skin", "scrotum", "penis", "vulva", "perineum",
    "muscle", "muscles", "tissue", "tissues", "fascia",
    "nerve", "nerves", "tendon", "ligament", "cartilage",
    "bone", "bones", "marrow", "vessel", "vessels",
    "lymph", "lymph node", "lymph nodes", "gland", "glands",
    "thyroid", "parathyroid", "adrenal", "pancreas", "spleen",
    "gallbladder", "bile duct", "appendix", "diaphragm",
    "peritoneum", "pleura", "pericardium", "meninges",
    "wound", "wounds",
    # surgical / hernia anatomy terms that bc5cdr mis-tags as DISEASE
    "cord", "ring", "canal", "oblique", "inguinal",
    "vas", "deferens", "spermatic", "epididymis",
    "sac", "pouch", "fossa", "sulcus",
    # head / neck / oral / ENT structures (same mis-tagging pattern)
    "palate", "throat", "tooth", "teeth", "gum", "gums",
    "tongue", "tonsil", "tonsils", "adenoid", "adenoids",
    "uvula", "pharynx", "larynx", "epiglottis",
    "sinus", "sinuses", "nasal", "nostril", "nostrils",
    "jaw", "mandible", "maxilla", "ear", "eardrum",
})


# ── ICD-10 chapter classifier ─────────────────────────────────────

class _ICD10Classifier:
    """Classify entities as DISEASE or SYMPTOM via ICD-10 chapter lookup.

    Lazily loads ``data/raw/icd10_codes.csv``, then uses rapidfuzz to
    fuzzy-match entity text against the 74k ICD-10 descriptions.
    The matched code's first character determines the label:

      R...   → SYMPTOM   (Chapter 18: Signs, symptoms, abnormal findings)
      V–Z    → None      (External causes / admin codes — defer to fallback)
      A–Q, S–T → DISEASE

    Every result is cached so each unique entity string is only looked
    up once per process lifetime.
    """

    THRESHOLD = 85  # minimum WRatio score to accept a match

    def __init__(self) -> None:
        # dict[first_letter] -> ([descriptions], [codes])
        self._index: dict[str, tuple[list[str], list[str]]] = {}
        self._cache: dict[str, str | None] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import pandas as pd
            from src.utils.config import Paths
            df = pd.read_csv(Paths.icd10_csv, dtype=str).dropna(
                subset=["description", "code"]
            )
            descs = df["description"].str.lower().tolist()
            codes = df["code"].tolist()
            # Group by first letter of description for fast pre-filtering.
            # Reduces each fuzzy search from 74k to ~3-5k candidates.
            for desc, code in zip(descs, codes):
                key = desc[0] if desc else "_"
                bucket = self._index.setdefault(key, ([], []))
                bucket[0].append(desc)
                bucket[1].append(code)
            logger.debug(
                "ICD-10 classifier ready: %d codes across %d buckets",
                len(descs), len(self._index),
            )
        except Exception as exc:
            logger.warning("ICD-10 classifier unavailable: %s", exc)

    def classify(self, entity_text: str) -> str | None:
        """Return DISEASE, SYMPTOM, or None (no confident match).

        None means the caller should fall back to keyword / model-label rules.
        """
        if entity_text in self._cache:
            return self._cache[entity_text]

        self._load()
        label = None

        text_lower = entity_text.lower().strip()
        key = text_lower[0] if text_lower else "_"
        bucket = self._index.get(key)

        if bucket:
            from rapidfuzz import fuzz, process
            descs, codes = bucket
            # Get multiple candidates rather than just the top one — WRatio
            # frequently ties several descriptions at the same score for a
            # short query (e.g. "cough" scores 90 against "cough variant
            # asthma", "cough syncope", AND "cough, unspecified" alike).
            # extractOne returns whichever ties first in iteration order,
            # which is not necessarily the right one. Break ties by
            # preferring the shortest description — closest to an exact
            # match, least likely to be an unrelated compound condition
            # that merely contains the query as a substring.
            candidates = process.extract(
                text_lower,
                descs,
                scorer=fuzz.WRatio,
                score_cutoff=self.THRESHOLD,
                limit=10,
            )
            if candidates:
                _desc, _score, idx = max(
                    candidates, key=lambda c: (c[1], -len(c[0]))
                )
                first = (codes[idx] or " ")[0].upper()
                if first == "R":
                    label = "SYMPTOM"
                elif first not in "VWXYZ":
                    label = "DISEASE"

        self._cache[entity_text] = label
        return label


_icd10_clf = _ICD10Classifier()


_SYMPTOM_FAST: frozenset[str] = frozenset({
    "pain", "pains", "ache", "aches", "aching", "tenderness", "swelling", "bleeding",
    "blood loss", "discharge", "nausea", "vomiting", "diarrhea",
    "diarrhoea", "fatigue", "weakness", "numbness", "tingling",
    "dizziness", "shortness of breath", "dyspnea", "dyspnoea",
    "fever", "chills", "sweating", "palpitations", "syncope",
    "seizure", "confusion", "dysuria", "hematuria", "haematuria",
    "incontinence", "urinary frequency", "urinary retention",
    "nocturia", "retention", "chest pain", "chest tightness",
    "chest discomfort", "edema", "oedema", "ascites", "jaundice",
    "cyanosis", "tachycardia", "bradycardia", "hypotension",
    # weight / appetite signs (ICD-10 R63.x — stored as "Abnormal weight loss"
    # so first-letter bucket lookup misses the R-code; catch here explicitly)
    "weight loss", "weight gain", "anorexia", "appetite loss",
    "malaise", "lethargy", "myalgia", "arthralgia",
    "rash", "erythema", "pruritus", "urticaria",
    "haemoptysis", "hemoptysis", "epistaxis",
    "polyuria", "oliguria", "anuria",
    "constipation", "diarrhoea", "diarrhea",
    "bruising", "bruise", "bruised", "ecchymosis",
    "diplopia", "blurred vision", "tinnitus", "vertigo",
    "distention", "distension", "bloating",
})

# Pathology-indicating suffix words/morphemes. An anatomy keyword combined
# with one of these is a compound DISEASE NAME ("liver disease",
# "hyperthyroidism"), not a body-part reference — the anatomy word is the
# disease's root, not what the entity is actually about. Without this
# check, the anatomy-substring safety net (step 10) would intercept these
# before bc5cdr's own DISEASE tag ever gets a chance at ICD-10 arbitration,
# since "thyroid" is a literal substring of "hyperthyroidism" and "liver"
# of "liver disease".
_DISEASE_NAME_HINTS: tuple[str, ...] = (
    "disease", "failure", "itis", "osis", "oma", "pathy",
    "megaly", "emia", "uria", "algia",
)


def _looks_like_disease_name(text_lower: str) -> bool:
    """Heuristic: does this anatomy-containing text look like a disease name?

    Checks for an explicit pathology suffix word ("liver disease") or the
    hyper-/hypo- endocrine prefix pattern ("hyperthyroidism").
    """
    if any(hint in text_lower for hint in _DISEASE_NAME_HINTS):
        return True
    if text_lower.startswith("hyper") or text_lower.startswith("hypo"):
        return True
    return False


_SYMPTOM_QUALIFIERS: tuple[str, ...] = (
    "pain", "ache", "aching", "tenderness", "swelling",
    "discomfort", "numbness", "weakness", "fatigue", "nausea",
    # found via gold-standard sample review (2026-06-26): anatomy term +
    # one of these is a symptom/finding report, not an anatomy mention
    # ("bladder incontinence", "abdominal injury", "sore throat",
    # "vascular abnormalities")
    "incontinence", "injury", "injuries", "sore", "abnormal", "abnormality",
)

# Matches a run of 2+ consecutive punctuation marks — a reliable signal
# that a NER span has crossed a sentence/section boundary and appended a
# stray fragment (e.g. "gastroesophageal reflux disease.,past" — the
# ".,past" leaked from the next sentence, not part of the entity). A
# single period/comma is left alone since that can appear legitimately
# within a real phrase or abbreviation.
_SPAN_ARTIFACT_PATTERN = re.compile(r"[.,;:]{2,}")


def _clean_entity_text(text: str) -> str:
    """Strip a trailing punctuation+fragment span-boundary artifact.

    Args:
        text: Raw entity text as extracted by spaCy (``ent.text``).

    Returns:
        Cleaned text with any trailing merge artifact removed.
    """
    match = _SPAN_ARTIFACT_PATTERN.search(text)
    if match:
        text = text[:match.start()]
    return text.strip(" .,;:")


def _normalise_label(raw_label: str, entity_text: str) -> str | None:
    """Map a raw NER label to one of our five clinical categories.

    Architecture
    ────────────
    Two model types produce spans:
      • bc5cdr   → raw labels DISEASE or CHEMICAL (high precision, domain-specific)
      • en_core_sci_lg → raw label ENTITY (broad coverage, no sub-type)

    DISEASE classification is restricted to bc5cdr DISEASE spans.
    en_core_sci_lg ENTITY spans can only become PROCEDURE / ANATOMY /
    MEDICATION / SYMPTOM — never DISEASE — because WRatio fuzzy-matching
    ICD-10 against any arbitrary biomedical token produces too many false
    positives ("antibiotics" → "Adverse effect of antibiotics" T36.x → DISEASE).

    Priority order (same for both model types up to step 8):
      1.  Negation / span-artifact filter — "denies X", "complaint of X",
          spans containing a stray location word like "room"
      2.  Skip filter  — headers, demographics, severity qualifiers
      3.  Fragment filter — "mellitus", "syndrome" …
      4.  CHEMICAL / DRUG → MEDICATION
      5.  Disease abbreviations — STEMI, DVT, COPD … (too short for ICD-10)
      6.  Procedure keywords — incision, stent, anesthesia, MRI …
      7.  Medication suffixes / known drug class names
      8.  Anatomy exact match
      9.  Anatomy substring match ("cord structures", "external oblique" …)
      10. Symptom fast-path — known R-code terms
      11. bc5cdr DISEASE only → ICD-10 chapter lookup (DISEASE vs SYMPTOM)
          If ICD-10 has no opinion, keep as DISEASE (trust bc5cdr).
      12. Default: SYMPTOM

    Returns:
        One of ``"DISEASE"``, ``"MEDICATION"``, ``"PROCEDURE"``,
        ``"ANATOMY"``, ``"SYMPTOM"``, or ``None`` (discard entity).
    """
    text_lower = entity_text.lower().strip()
    label_up   = raw_label.upper()

    # 1. Negation / span-artifact filter. A span beginning with a negation
    #    word ("denies obesity") describes the ABSENCE of a finding, not a
    #    reportable entity — classifying it as DISEASE would claim the
    #    patient HAS the condition they explicitly don't. A span beginning
    #    with a generic reporting phrase ("complaint of allergies") is a
    #    leaked section-header fragment from imperfect span boundaries.
    #    "room" catches NER span-boundary errors that merge an unrelated
    #    location word with an adjacent clinical term (e.g. "emergency
    #    room" + "cancer" -> "room cancer") — not a real entity either way.
    if any(text_lower.startswith(p) for p in _NEGATION_PREFIXES):
        return None
    if any(text_lower.startswith(p) for p in _GENERIC_REPORT_PREFIXES):
        return None
    if "room" in text_lower:
        return None

    # 2. Skip filter
    if text_lower in _SKIP_TERMS:
        return None

    # 3. Fragment filter
    if text_lower in _FRAGMENT_TERMS:
        return None

    # 4. bc5cdr CHEMICAL / DRUG — high-precision medication label
    if label_up in ("CHEMICAL", "DRUG"):
        return "MEDICATION"

    # 5. Disease abbreviations (single tokens too short for ICD-10 fuzzy match)
    if text_lower in _DISEASE_TERMS:
        return "DISEASE"

    # 6. Symptom fast-path (known R-code surface forms) — EXACT match,
    #    checked before any substring-based rule. An exact match is an
    #    unambiguous signal and must win over accidental substring
    #    collisions in the keyword lists below (e.g. "distention" contains
    #    "stent" — a PROCEDURE device keyword — so without this ordering
    #    it would wrongly resolve to PROCEDURE before ever being checked
    #    against the symptom list).
    if text_lower in _SYMPTOM_FAST:
        return "SYMPTOM"

    # 7. Procedure keywords
    if any(kw in text_lower for kw in _PROCEDURE_KEYWORDS):
        return "PROCEDURE"

    # 8. Medication suffix / known drug class, or a medication-form word
    #    ("cough syrup", "eye drops") — these override even a bc5cdr
    #    DISEASE tag, since bc5cdr occasionally mistags an OTC remedy
    #    mention as a disease in context ("offered him a cough syrup").
    if any(text_lower.endswith(sfx) or sfx in text_lower for sfx in _MEDICATION_SUFFIXES):
        return "MEDICATION"
    if any(form in text_lower for form in _MEDICATION_FORM_WORDS):
        return "MEDICATION"

    # 9. Anatomy — exact single-word match
    if text_lower in _ANATOMY_KEYWORDS:
        return "ANATOMY"

    # 10. Anatomy — substring match for multi-word phrases ("cord structures").
    #     If a symptom qualifier is also present ("knee pain", "shoulder
    #     ache"), this is a symptom report, not an anatomy mention — resolve
    #     to SYMPTOM directly rather than falling through to ICD-10, which
    #     codes joint/limb pain under musculoskeletal M-codes (M25.5x), not
    #     R-codes, so the chapter heuristic would otherwise misclassify it
    #     as DISEASE.
    if any(kw in text_lower for kw in _ANATOMY_KEYWORDS):
        if any(q in text_lower for q in _SYMPTOM_QUALIFIERS):
            return "SYMPTOM"
        if not _looks_like_disease_name(text_lower):
            return "ANATOMY"
        # else: this is a compound disease name built on an anatomy root
        # ("liver disease", "hyperthyroidism") — fall through to ICD-10
        # arbitration instead of forcing ANATOMY.

    # 11. ICD-10 lookup — ONLY for bc5cdr DISEASE spans.
    #     Determines whether a confirmed biomedical disease entity is
    #     a true DISEASE or a SYMPTOM/sign (ICD-10 Chapter 18, R-codes).
    #     en_core_sci_lg ENTITY spans skip this step entirely.
    if label_up == "DISEASE":
        icd10_label = _icd10_clf.classify(entity_text)
        if icd10_label is not None:
            return icd10_label
        return "DISEASE"  # bc5cdr confident; ICD-10 had no strong match

    # 12. Default for ENTITY spans: SYMPTOM
    return "SYMPTOM"


# ── Base interface ────────────────────────────────────────────────

class BaseNERPipeline(ABC):
    """Abstract interface for NER pipelines.

    Concrete implementations (SpacyNERPipeline, and any future
    HuggingFace NER pipeline) must implement ``extract()``.
    All code that uses NER should type-hint against this base
    class, not the concrete implementation.
    """

    @abstractmethod
    def extract(self, text: str) -> list[Entity]:
        """Extract named entities from a single text string.

        Args:
            text: Cleaned clinical note text.

        Returns:
            List of :class:`Entity` objects sorted by start offset.
        """
        ...

    def extract_batch(self, texts: list[str]) -> list[list[Entity]]:
        """Extract entities from a list of texts.

        Default implementation calls extract() in a loop.
        Concrete subclasses may override with a more efficient
        batched implementation.

        Args:
            texts: List of clinical note strings.

        Returns:
            List of entity lists, one per input text.
        """
        return [self.extract(t) for t in texts]


# ── scispaCy implementation ───────────────────────────────────────

class SpacyNERPipeline(BaseNERPipeline):
    """NER pipeline backed by a scispaCy biomedical model.

    Loads the model once at construction time and reuses it for
    all subsequent calls.  The model is expensive to load (~2s)
    but fast to run (~10ms per note).

    Args:
        model_name: Name of the spaCy/scispaCy model to load.
            Defaults to the value in ModelConfig.

    Example::

        pipeline = SpacyNERPipeline()
        entities = pipeline.extract("Patient has hypertension and diabetes.")
        for ent in entities:
            print(ent.text, ent.label)
        # → hypertension  DISEASE
        # → diabetes       DISEASE
    """

    def __init__(self, model_name: str = ModelConfig.ner_model) -> None:
        self._model_name = model_name
        self._nlp        = None   # lazy-loaded on first use

    def _load_model(self) -> None:
        """Load the spaCy model into memory.

        Called lazily on the first extract() call so that importing
        this module does not trigger a slow model load at startup.

        Raises:
            OSError: If the model is not installed.  The error
                message includes the installation command.
        """
        try:
            import spacy
            logger.info("Loading NER model: %s", self._model_name)
            self._nlp = spacy.load(self._model_name)
            logger.info("NER model loaded ✓")
        except OSError:
            raise OSError(
                f"spaCy model '{self._model_name}' is not installed.\n"
                "Install it with:\n"
                f"  pip install https://s3-us-west-2.amazonaws.com/"
                f"ai2-s2-scispacy/releases/v0.5.3/"
                f"{self._model_name}-0.5.3.tar.gz"
            )

    @property
    def model_name(self) -> str:
        """Return the spaCy model name."""
        return self._model_name

    @property
    def nlp(self):
        """Return the loaded spaCy model, loading it if necessary."""
        if self._nlp is None:
            self._load_model()
        return self._nlp

    def extract(self, text: str) -> list[Entity]:
        """Extract named entities from a clinical note.

        Args:
            text: Cleaned clinical note text.  Run
                :func:`src.utils.text_utils.clean_clinical_text`
                on raw input before passing it here.

        Returns:
            List of :class:`Entity` objects sorted by start offset.
            Empty list if text is empty or no entities are found.
        """
        if not text or not text.strip():
            return []

        doc      = self.nlp(text)
        entities = []

        for ent in doc.ents:
            cleaned_text = _clean_entity_text(ent.text)

            # Skip very short tokens — usually OCR artefacts or initials
            if len(cleaned_text) < 3:
                continue

            label = _normalise_label(ent.label_, cleaned_text)
            if label is None:
                continue

            entities.append(Entity(
                text       = cleaned_text,
                label      = label,
                start      = ent.start_char,
                end        = ent.start_char + len(cleaned_text),
                # scispaCy does not expose per-entity scores natively;
                # we leave confidence as None rather than fabricating a value
                confidence = None,
            ))

        return sorted(entities, key=lambda e: e.start)

    def extract_batch(self, texts: list[str], batch_size: int = 32) -> list[list[Entity]]:
        """Extract entities from multiple texts using spaCy's pipe().

        More efficient than calling extract() in a loop because
        spaCy processes documents in parallel where possible.

        Args:
            texts:      List of clinical note strings.
            batch_size: Number of texts to process per batch.

        Returns:
            List of entity lists, one per input text.
        """
        if not texts:
            return []

        results  = []
        # spaCy's nlp.pipe is the idiomatic way to batch-process
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            entities = []
            for ent in doc.ents:
                cleaned_text = _clean_entity_text(ent.text)
                if len(cleaned_text) < 3:
                    continue
                label = _normalise_label(ent.label_, cleaned_text)
                if label is None:
                    continue
                entities.append(Entity(
                    text       = cleaned_text,
                    label      = label,
                    start      = ent.start_char,
                    end        = ent.start_char + len(cleaned_text),
                    confidence = None,
                ))
            results.append(sorted(entities, key=lambda e: e.start))

        return results


# ── Hybrid pipeline ───────────────────────────────────────────────

class HybridNERPipeline(BaseNERPipeline):
    """Combines two scispaCy models for best coverage and accuracy.

    ``fine_model`` (default: en_ner_bc5cdr_md) runs first and
    produces accurate DISEASE and MEDICATION labels from a model
    trained specifically on biomedical text.

    ``broad_model`` (default: en_core_sci_lg) runs second and
    contributes any spans that do not overlap with the fine model's
    output — typically PROCEDURE, ANATOMY, and SYMPTOM entities
    that bc5cdr was not trained to detect.

    Overlap rule: if a broad-model span shares any characters with
    a fine-model span, the broad span is dropped (fine wins).

    Args:
        fine_model:  Model name for precise DISEASE/MEDICATION detection.
        broad_model: Model name for broad entity coverage.
    """

    def __init__(
        self,
        fine_model:  str = "en_ner_bc5cdr_md",
        broad_model: str = "en_core_sci_lg",
    ) -> None:
        self._fine  = SpacyNERPipeline(fine_model)
        self._broad = SpacyNERPipeline(broad_model)

    @property
    def model_name(self) -> str:
        """Return a descriptive name for the hybrid pipeline."""
        return f"hybrid({self._fine.model_name} + {self._broad.model_name})"

    @staticmethod
    def _overlaps(a: Entity, b: Entity) -> bool:
        return max(a.start, b.start) < min(a.end, b.end)

    def extract(self, text: str) -> list[Entity]:
        fine_ents  = self._fine.extract(text)
        broad_ents = self._broad.extract(text)

        merged = list(fine_ents)
        for broad in broad_ents:
            if not any(self._overlaps(broad, f) for f in fine_ents):
                merged.append(broad)

        return sorted(merged, key=lambda e: e.start)

    def extract_batch(self, texts: list[str], batch_size: int = 32) -> list[list[Entity]]:
        fine_batches  = self._fine.extract_batch(texts, batch_size)
        broad_batches = self._broad.extract_batch(texts, batch_size)

        results = []
        for fine_ents, broad_ents in zip(fine_batches, broad_batches):
            merged = list(fine_ents)
            for broad in broad_ents:
                if not any(self._overlaps(broad, f) for f in fine_ents):
                    merged.append(broad)
            results.append(sorted(merged, key=lambda e: e.start))
        return results


# ── Factory function ──────────────────────────────────────────────

def build_ner_pipeline(
    model_name: str | None = None,
) -> BaseNERPipeline:
    """Construct and return the configured NER pipeline.

    Pass ``model_name="hybrid"`` (or set ``NER_MODEL=hybrid`` in
    ``.env``) to use :class:`HybridNERPipeline`, which combines
    ``en_ner_bc5cdr_md`` (accurate DISEASE/MEDICATION) with
    ``en_core_sci_lg`` (broad coverage for PROCEDURE/ANATOMY/SYMPTOM).

    Any other value is treated as a single spaCy model name and
    passed to :class:`SpacyNERPipeline`.

    Args:
        model_name: Override the default model from config.
            Use ``"hybrid"`` for the two-model pipeline.

    Returns:
        A ready-to-use :class:`BaseNERPipeline` instance.
    """
    name = model_name or ModelConfig.ner_model
    if name == "hybrid":
        logger.debug("Building hybrid NER pipeline (bc5cdr + en_core_sci_lg)")
        return HybridNERPipeline()
    logger.debug("Building NER pipeline with model: %s", name)
    return SpacyNERPipeline(model_name=name)

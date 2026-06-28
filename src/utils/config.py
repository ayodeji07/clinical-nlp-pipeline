"""
src/utils/config.py
────────────────────────────────────────────────────────────────
Central configuration for the Clinical NLP Pipeline.

All paths, model names, and tuneable parameters live here.
Nothing else in the codebase should hardcode a path or model name —
import from this module instead.

Design notes
────────────
- Settings are read from environment variables where appropriate,
  with sensible defaults so the project works out of the box.
- Model names are strings, not imports, so swapping a model is a
  one-line change that does not require touching pipeline code.
- The Paths class uses pathlib throughout; no os.path anywhere.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # populate os.getenv() from .env before any class is defined


@contextmanager
def force_offline_hf_env():
    """Temporarily force huggingface_hub/transformers into fully offline mode.

    ``local_files_only=True`` on an individual ``from_pretrained()`` call
    does not stop every internal network call some library versions
    make (e.g. commit-history or discussion metadata lookups). Those
    extra calls can hang indefinitely on a flaky network rather than
    failing fast — even though the model is fully cached locally —
    which previously caused multi-minute hangs on model load.

    ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` are huggingface_hub's
    documented mechanism for guaranteeing zero network calls; they're
    stronger than the per-call flag. Use this to wrap *only* the
    local-cache load attempt, so a genuine fallback download (model not
    yet cached) still works normally once the env is restored.

    Example::

        with force_offline_hf_env():
            try:
                return SentenceTransformer(name, local_files_only=True)
            except Exception:
                pass
        return SentenceTransformer(name)  # real download, env restored
    """
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── Project root ─────────────────────────────────────────────────
# Resolves correctly whether you run from the repo root, a
# subdirectory, or inside a Docker container.
ROOT = Path(__file__).resolve().parents[2]


class Paths:
    """Filesystem layout for the project.

    All directories are relative to the repository root so the
    project is portable across machines and containers.
    """

    # Source data lands here before any processing
    raw: Path = ROOT / "data" / "raw"

    # Cleaned and featurised outputs
    processed: Path = ROOT / "data" / "processed"

    # Saved model checkpoints and artefacts
    models: Path = ROOT / "data" / "models"

    # ICD-10 reference files
    icd10_csv: Path = raw / "icd10_codes.csv"

    # MTSamples clinical notes
    mtsamples_csv: Path = raw / "mtsamples.csv"

    # SQLite database (used when DATABASE_URL is not set)
    sqlite_db: Path = ROOT / "clinical_nlp.db"

    @classmethod
    def ensure_all(cls) -> None:
        """Create every data directory if it does not already exist.

        Safe to call multiple times — uses exist_ok=True throughout.
        """
        for attr in ("raw", "processed", "models"):
            getattr(cls, attr).mkdir(parents=True, exist_ok=True)


class ModelConfig:
    """Names and sources for every model used in the pipeline.

    Changing a model is a one-line edit here; nothing downstream
    needs to change.  All names are valid HuggingFace model IDs
    or spaCy model names.
    """

    # ── NER ──────────────────────────────────────────────────────
    # "hybrid" runs en_ner_bc5cdr_md (DISEASE/MEDICATION) + en_core_sci_lg
    # (PROCEDURE/ANATOMY/SYMPTOM) and merges, fine model taking priority.
    # Single-model alternatives: "en_ner_bc5cdr_md" or "en_core_sci_lg"
    ner_model: str = os.getenv("NER_MODEL", "hybrid")

    # ── Classifier ────────────────────────────────────────────────
    # Bio_ClinicalBERT — BERT pre-trained on MIMIC-III notes.
    # Best publicly available model for clinical text classification.
    classifier_model: str = os.getenv(
        "CLASSIFIER_MODEL",
        "emilyalsentzer/Bio_ClinicalBERT",
    )

    # ── Embeddings (ICD-10 fallback matching) ─────────────────────
    # Sentence-transformer variant of BioBERT fine-tuned on
    # medical NLI tasks — strong semantic similarity on clinical text.
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    )

    # Directory where fine-tuned weights are saved after training
    fine_tuned_dir: Path = Paths.models / "severity_classifier"


class ClassifierConfig:
    """Parameters for the severity classification task.

    Designed to be task-agnostic: swap `task` and `labels` to
    switch from severity to readmission risk (or anything else)
    without touching the model code.
    """

    # Current active task — change to "readmission" for Phase 2
    task: str = os.getenv("CLASSIFIER_TASK", "severity")

    # Severity labels (Phase 1)
    severity_labels: list[str] = ["routine", "urgent", "critical"]

    # Readmission labels (Phase 2 — needs MIMIC-III)
    readmission_labels: list[str] = ["not_readmitted", "readmitted"]

    @classmethod
    def active_labels(cls, task: str | None = None) -> list[str]:
        """Return the label set for the given (or globally configured) task.

        Args:
            task: Task name to look up. Defaults to ``cls.task`` (the
                global default) when not provided, so existing callers
                that rely on the env-configured default keep working.
                Pass explicitly to respect a per-instance task override
                (e.g. from ``ClinicalClassifier(task=...)``) rather than
                silently falling back to the global default.

        Returns:
            List of string class labels for the active task.

        Raises:
            ValueError: If `task` is not a recognised task name.
        """
        task = task or cls.task
        label_map = {
            "severity":    cls.severity_labels,
            "readmission": cls.readmission_labels,
        }
        if task not in label_map:
            raise ValueError(
                f"Unknown classifier task '{task}'. "
                f"Choose from: {list(label_map.keys())}"
            )
        return label_map[task]


class TrainingConfig:
    """Hyperparameters for fine-tuning Bio_ClinicalBERT.

    These defaults are tuned for MTSamples (~5k notes) on a CPU.
    Reduce batch_size if you run out of memory; increase epochs
    if validation loss is still falling at the end of training.
    """

    epochs: int      = int(os.getenv("TRAIN_EPOCHS", "3"))
    batch_size: int  = int(os.getenv("TRAIN_BATCH_SIZE", "16"))
    learning_rate: float = float(os.getenv("TRAIN_LR", "2e-5"))
    max_length: int  = int(os.getenv("TRAIN_MAX_LENGTH", "512"))
    val_split: float = float(os.getenv("TRAIN_VAL_SPLIT", "0.15"))
    test_split: float = float(os.getenv("TRAIN_TEST_SPLIT", "0.10"))
    random_seed: int = int(os.getenv("TRAIN_SEED", "42"))


class ICD10Config:
    """Settings for the ICD-10 code mapping module."""

    # Minimum fuzzy-match score (0–100) to accept a lookup result.
    # Below this threshold the embedding fallback is used instead.
    fuzzy_threshold: int = int(os.getenv("ICD10_FUZZY_THRESHOLD", "80"))

    # Minimum cosine similarity (0.0–1.0) to accept an embedding match.
    # Results below this are returned with a low-confidence flag.
    embedding_threshold: float = float(
        os.getenv("ICD10_EMBEDDING_THRESHOLD", "0.75")
    )

    # Maximum number of candidate ICD-10 codes returned per entity
    top_k: int = int(os.getenv("ICD10_TOP_K", "3"))


class APIConfig:
    """FastAPI application settings."""

    host: str  = os.getenv("API_HOST", "0.0.0.0")
    port: int  = int(os.getenv("API_PORT", "8000"))
    title: str = "Clinical NLP Pipeline API"
    version: str = "1.0.0"
    description: str = (
        "Named entity recognition, ICD-10 mapping, and severity "
        "classification for clinical text."
    )

    # Set to False in production — disables /docs and /redoc
    debug: bool = os.getenv("API_DEBUG", "true").lower() == "true"


class DatabaseConfig:
    """Database connection settings.

    SQLite is used when DATABASE_URL is not set — zero configuration
    required for local development.  Set DATABASE_URL to a PostgreSQL
    connection string (e.g. Supabase) for staging and production.
    """

    url: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{Paths.sqlite_db}",
    )

    # SQLAlchemy connection pool settings.
    # These are ignored for SQLite (which has no connection pool).
    pool_size: int     = int(os.getenv("DB_POOL_SIZE", "5"))
    max_overflow: int  = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    pool_timeout: int  = int(os.getenv("DB_POOL_TIMEOUT", "30"))

    @classmethod
    def is_sqlite(cls) -> bool:
        """Return True if the configured database is SQLite."""
        return cls.url.startswith("sqlite")


# ── Convenience re-exports ────────────────────────────────────────
# Import just `settings` anywhere in the codebase for quick access.
class Settings:
    """Single entry point for all configuration sections.

    Usage::

        from src.utils.config import settings

        model = settings.model.ner_model
        db    = settings.db.url
    """

    paths:      Paths           = Paths
    model:      ModelConfig     = ModelConfig
    classifier: ClassifierConfig = ClassifierConfig
    training:   TrainingConfig  = TrainingConfig
    icd10:      ICD10Config     = ICD10Config
    api:        APIConfig       = APIConfig
    db:         DatabaseConfig  = DatabaseConfig


settings = Settings()

# ── Builder stage ─────────────────────────────────────────────────
# Installs all dependencies including heavy ML packages.
# Kept separate so the final image is smaller.
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# --extra-index-url pulls the CPU-only torch build (matches the local
# dev env: torch==2.12.0+cpu). Without it, pip resolves requirements.txt's
# plain "torch>=2.1.0" against default PyPI and pulls the CUDA build —
# adding ~2GB of unused NVIDIA libraries (cudnn, cusparselt, etc.) that
# this project never needs (it runs CPU-only everywhere, see classifier.py).
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# scispaCy + its own deps (conllu/pysbd/nmslib-metabrainz).
# --no-deps avoids scispaCy's spacy<3.8.0 pin clobbering the
# spacy>=3.7.0,<3.8.0 already installed via requirements.txt.
RUN pip install --no-cache-dir --no-deps scispacy==0.5.5 && \
    pip install --no-cache-dir conllu pysbd "nmslib-metabrainz==2.1.3"

# en_core_sci_lg (~531 MB) — broad entity coverage.
# Separate layer so a network drop only retries this download,
# not the whole scispaCy install above.
# --timeout 300: S3 can stall on large files; 15s default is too short.
RUN pip install --no-cache-dir --no-deps --timeout 300 \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

# en_ner_bc5cdr_md (~100 MB) — DISEASE/CHEMICAL NER.
RUN pip install --no-cache-dir --no-deps --timeout 300 \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz


# ── Runtime stage ─────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Application code and the small public ICD-10 reference CSV.
# NOT copied: data/processed/ (cached embeddings) or data/models/
# (fine-tuned classifier) — both too large to ship in the image.
# ClinicalClassifier.load() and ICD10Mapper download them from
# HuggingFace Hub on first use instead (see ModelConfig fallback
# settings in src/utils/config.py).
COPY src/        ./src/
COPY dashboard/  ./dashboard/
COPY data/raw/   ./data/raw/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# The API port — Railway/Render assign this via $PORT at runtime
EXPOSE 8000

# Shell form so $PORT is actually expanded; falls back to 8000 for
# local `docker run` testing where $PORT isn't set.
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

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
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install the scispaCy NER model
RUN pip install --no-cache-dir \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz


# ── Runtime stage ─────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/        ./src/
COPY dashboard/  ./dashboard/
COPY data/raw/   ./data/raw/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# The API port — override with $PORT in Railway/Render
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Clinical NLP Pipeline — VSCode Setup & Run Guide

Everything you need to go from a fresh clone to a running
Streamlit demo in one sitting.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10+ | python.org |
| Git | any | git-scm.com |
| VSCode | any | code.visualstudio.com |
| Tesseract (optional) | 5.x | see Phase 0 |

---

## Phase 0 — Clone and virtual environment

```bash
git clone https://github.com/ayodeji07/clinical-nlp-pipeline.git
cd clinical-nlp-pipeline

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements-dev.txt
```

Copy the example env file:

```bash
cp .env.example .env
```

Open `.env` — for local development the defaults are fine.
`DATABASE_URL` will default to SQLite automatically.

---

## Phase 1 — Install NLP models

The scispaCy model is not on PyPI — install it directly:

```bash
pip install scispacy==0.5.5 --no-deps
pip install conllu pysbd "nmslib-metabrainz==2.1.3"
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz --no-deps
```

> **Note**: `--no-deps` is required on spaCy ≥ 3.8 because scispaCy pins `spacy<3.8.0`.
> The model loads and runs correctly despite the version mismatch.

Verify:

```bash
python -c "import spacy; nlp = spacy.load('en_core_sci_lg'); print('NER model OK')"
```

---

## Phase 2 — Download MTSamples

1. Go to https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
2. Download `mtsamples.csv`
3. Save it to `data/raw/mtsamples.csv`

Also download the ICD-10 reference file (included in the repo under `data/raw/`).
If it is missing, download from:
https://www.cms.gov/medicare/coding-billing/icd-10-codes

Save as `data/raw/icd10_codes.csv` with columns: `code`, `description`.

---

## Phase 3 — Run the ETL pipeline

Dry run first to validate data without writing to the database:

```bash
python -m src.etl.pipeline --dry-run
```

Full run:

```bash
python -m src.etl.pipeline
```

Expected output:
```
Stage 1: Extract — loaded 4,999 raw notes
Stage 2: Transform — 4,721 notes ready
Stage 3: Load — inserted 4,721 records
```

The ETL pipeline only loads notes — it doesn't run NER, ICD-10 mapping,
or classification against them. Populate those with the batch scripts:

```bash
python scripts/run_ner_batch.py                            # extract entities from stored notes
python scripts/run_icd10_batch.py                          # map DISEASE/SYMPTOM entities to ICD-10 codes
python scripts/train_severity_classifier.py --n-seeds 4    # fine-tune the severity classifier
```

All three are idempotent and resumable, so interrupting and re-running
is safe. Pass `--limit N` to any of them for a quick subset run instead
of processing the full dataset. The dashboard's Explorer and Model
Metrics pages need this step done first — without it there's nothing
to show beyond raw note counts.

`--n-seeds N` on the classifier script trains N different random seeds
and keeps only the checkpoint with the best critical-class F1, then
records it in the `model_runs` table (what `GET /model/metrics` and
the Model Metrics dashboard page read from). Fine-tuning a small
classification head on a small dataset is sensitive to random
init — an unseeded single run isn't reliably comparable across
retrains, so this matters more than it might look like it should.
Omit the flag (or use `--n-seeds 1`) for a single quick run.

If your Supabase database was created before this feature was added,
`model_runs` is missing the columns this needs (`create_all_tables()`
only creates missing tables, not missing columns on existing ones) —
run the migration in `sql/schema.sql` (the commented `ALTER TABLE`
statements near the bottom of the `model_runs` block) once via the
Supabase SQL Editor, or execute them directly:

```bash
python -c "
from sqlalchemy import text
from src.db.connection import get_session
stmts = [
    'ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS test_accuracy REAL',
    'ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS test_f1 REAL',
    'ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS per_class JSON',
    'ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS confusion_matrix JSON',
    'ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS history JSON',
]
with get_session() as session:
    for stmt in stmts:
        session.execute(text(stmt))
"
```

---

## Phase 4 — Run the notebooks

Open VSCode, install the Jupyter extension, then open notebooks in order:

```
notebooks/00_data_exploration.ipynb   ← understand the dataset
notebooks/01_ner_walkthrough.ipynb    ← try the NER pipeline
notebooks/02_icd_mapping.ipynb        ← try ICD-10 mapping
notebooks/03_classification.ipynb     ← train the classifier (~20 min CPU)
notebooks/04_visualisation.ipynb      ← build the charts
```

Select kernel: Python (``.venv``)

> **GPU tip**: For faster training, open notebook 03 in Google Colab.
> Upload the notebook, mount your Drive, and run — takes ~5 minutes on T4.

---

## Phase 5 — Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Open http://localhost:8000/docs to see the Swagger UI.

Test the health endpoint:

```bash
curl http://localhost:8000/health
# → {"status":"ok","version":"1.0.0","database":"connected"}
```

Test the analyse endpoint:

```bash
curl -X POST http://localhost:8000/notes/analyse \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient has hypertension and takes metformin.", "include_icd10": true}'
```

---

## Phase 6 — Start the Streamlit dashboard

In a second terminal (keep the API running in the first):

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501

You should see:
- 🔬 **Demo** page — paste a note and click Analyse
- 📊 **Explorer** — entity frequency + co-occurrence graph
- 🧠 **Metrics** — classifier performance (after training notebook 03)

---

## Phase 7 — Run the tests

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

Expected: 65 tests pass, ~0 failures.

---

## Phase 8 — Deploy to Streamlit Cloud

### Step 1: Create a Supabase database

1. Go to https://supabase.com → New project
2. Choose a region close to your users
3. Copy the connection string from Settings → Database — but use the
   **Pooler** connection string (Transaction or Session mode), not the
   direct one. It looks like:
   `postgresql://postgres.<ref>:<password>@aws-<region>.pooler.supabase.com:5432/postgres`

   **Do not use** the direct-connection host
   (`db.<ref>.supabase.co`) for a deployed app — it resolves to an
   IPv6-only address, and several free-tier hosts (Render's included)
   don't support IPv6 egress. You'll get a "Network is unreachable"
   error at startup that looks like a code bug but is actually just
   the wrong host. The pooler host is IPv4-compatible and works
   everywhere.

### Step 2: Deploy the API

The API needs to be publicly accessible for Streamlit Cloud to reach it.

- **Hugging Face Spaces** (recommended for this project) —
  https://huggingface.co/new-space, SDK: **Docker**. Push this repo to
  the Space's git remote and it builds from the existing `Dockerfile`.
  The free CPU tier has enough RAM (historically ~16GB) to run the full
  hybrid NER pipeline + ICD-10 embeddings + classifier together without
  issue — confirmed working in production for this project. Sleeps
  after 48h of inactivity, not 15 minutes, so it stays warm for normal
  demo traffic.

- **Railway** — https://railway.app (free tier)
  ```bash
  railway login
  railway up
  ```
  Copy the public URL Railway gives you.

- **Render** — https://render.com (free tier) — **be aware its free
  tier caps out at 512Mi RAM**, which is not enough to run this
  project's full hybrid NER pipeline (`en_ner_bc5cdr_md` +
  `en_core_sci_lg`) together with the classifier — a real
  `/notes/analyse` request will get OOM-killed (502 Bad Gateway),
  even though `/health` and DB-backed endpoints work fine. If you use
  Render anyway:
  - Set `WARM_UP_MODELS=false` so the app can at least boot instead of
    crash-looping on eager model load at startup.
  - Set `NER_MODEL=en_ner_bc5cdr_md` (single model instead of
    `hybrid`) to meaningfully cut memory, at the cost of losing
    `PROCEDURE`/`ANATOMY`/most `SYMPTOM` entity coverage (bc5cdr alone
    still detects `DISEASE`/`MEDICATION` — the more important two).
  - Or upgrade to a paid instance size.

Set `DATABASE_URL` (the pooler string from Step 1) as an environment
variable on whichever platform you choose.

### Step 3: Deploy the Streamlit app

1. Go to https://share.streamlit.io → New app
2. Connect your GitHub repo
3. Set **Main file path**: `dashboard/app.py`
4. Click **Advanced settings** → **Secrets** and add:

```toml
API_BASE_URL = "https://<your-username>-<space-name>.hf.space"
DATABASE_URL = "postgresql://postgres.<ref>:<password>@aws-<region>.pooler.supabase.com:5432/postgres"
```

(Swap `API_BASE_URL` for your Railway/Render URL if you went that
route instead.)

5. Click **Deploy**

Your app will be live at `https://<your-app>.streamlit.app`


---

## Phase 9 — Docker (optional)

Run the full stack in containers without installing anything locally
except Docker Desktop.

```bash
# Build and start API + dashboard
docker-compose up --build

# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

Stop everything:

```bash
docker-compose down
```

The `data/` directory is mounted as a volume so your database and
processed files persist between container restarts.

---

## Supabase — applying the schema

When you create a new Supabase project the database is empty.
SQLAlchemy will create the tables automatically on first API startup,
but you can also apply the schema manually for review or migration:

1. Open your Supabase project dashboard
2. Click **SQL Editor** in the left sidebar
3. Click **New query**
4. Open `sql/schema.sql` from this repo and paste the contents
5. Click **Run** (Ctrl+Enter)

You should see all four tables appear in the **Table Editor**:
`clinical_notes`, `entities`, `icd10_mappings`, `model_runs`.

> **Note**: The schema uses `AUTOINCREMENT` which is SQLite syntax.
> For Supabase (PostgreSQL), replace `INTEGER PRIMARY KEY AUTOINCREMENT`
> with `SERIAL PRIMARY KEY` in the SQL editor. SQLAlchemy handles
> this automatically when creating tables via `create_all_tables()`.

---

## Setting up Streamlit secrets locally

Streamlit Cloud reads secrets from `.streamlit/secrets.toml`.
For local development, create this file (it is in `.gitignore`):

```bash
mkdir -p .streamlit
cat > .streamlit/secrets.toml << 'EOF'
API_BASE_URL = "http://localhost:8000"
# DATABASE_URL = "postgresql://..."  # only needed for cloud
EOF
```

The dashboard will read `API_BASE_URL` from here automatically.
---

## Troubleshooting

**`OSError: en_core_sci_lg not found`**
```bash
pip install scispacy==0.5.5 --no-deps
pip install conllu pysbd "nmslib-metabrainz==2.1.3"
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz --no-deps
```

**`FileNotFoundError: mtsamples.csv not found`**
Download from Kaggle and save to `data/raw/mtsamples.csv`.

**`ModuleNotFoundError: No module named 'src'`**
Run commands from the repo root, or add it to PYTHONPATH:
```bash
export PYTHONPATH=$(pwd)
```

**`RuntimeError: Model not loaded`**
Run notebook 03 to fine-tune and save the classifier before
calling `clf.predict()`.

**Dashboard shows "API offline"**
Ensure `uvicorn src.api.main:app --reload` is running in a
separate terminal and `API_BASE_URL` in `.env` points to it.

**Supabase connection timeout / "Network is unreachable" on a deployed host**
Make sure you're using the *pooler* connection string, not the direct
one — the direct host (`db.<ref>.supabase.co`) is IPv6-only and
unreachable from several free-tier hosts (confirmed on Render's free
tier). Use the pooler host instead, with `?sslmode=require` appended:
```
postgresql://postgres.<ref>:<pass>@aws-<region>.pooler.supabase.com:5432/postgres?sslmode=require
```

---

## Project structure (quick reference)

```
src/utils/      config, logger, text cleaning
src/etl/        extract, transform, load, pipeline
src/nlp/        ner, icd_mapper, classifier, cooccurrence
src/db/         connection, models, repository
src/api/        FastAPI app + routes
dashboard/      Streamlit app + pages
notebooks/      00–04 walkthrough notebooks
tests/          pytest test suite (65 tests)
```

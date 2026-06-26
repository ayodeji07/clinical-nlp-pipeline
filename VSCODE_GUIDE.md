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
git clone https://github.com/<your-username>/clinical-nlp-pipeline.git
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

Expected: 61 tests pass, ~0 failures.

---

## Phase 8 — Deploy to Streamlit Cloud

### Step 1: Create a Supabase database

1. Go to https://supabase.com → New project
2. Choose a region close to your users
3. Copy the **Connection string** from Settings → Database → URI
   It looks like: `postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres`

### Step 2: Deploy the API

The API needs to be publicly accessible for Streamlit Cloud to reach it.
Recommended free options:

- **Railway** — https://railway.app (easiest, free tier)
  ```bash
  railway login
  railway up
  ```
  Copy the public URL Railway gives you.

- **Render** — https://render.com (free tier, cold starts)
  Connect your GitHub repo and set start command:
  ```
  uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
  ```

Set the `DATABASE_URL` environment variable to your Supabase URL in the
Railway/Render dashboard.

### Step 3: Deploy the Streamlit app

1. Go to https://share.streamlit.io → New app
2. Connect your GitHub repo
3. Set **Main file path**: `dashboard/app.py`
4. Click **Advanced settings** → **Secrets** and add:

```toml
API_BASE_URL = "https://your-api.railway.app"
DATABASE_URL = "postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres"
```

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

**Supabase connection timeout on Streamlit Cloud**
Add `?sslmode=require` to the end of your DATABASE_URL:
```
postgresql://postgres:<pass>@db.<ref>.supabase.co:5432/postgres?sslmode=require
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
tests/          pytest test suite (61 tests)
```

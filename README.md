# Clinical NLP Pipeline

**NLP · Named Entity Recognition · Clinical Text Mining · BERT Fine-tuning**

A production-grade pipeline that extracts structured clinical knowledge
from unstructured medical notes using state-of-the-art biomedical NLP models.

[![CI](https://github.com/<your-username>/clinical-nlp-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/clinical-nlp-pipeline/actions)

---

## What it does

| Component | What it does |
|-----------|-------------|
| **NER** | Extracts diagnoses, medications, procedures, symptoms, and anatomical terms using scispaCy (`en_core_sci_lg`) |
| **ICD-10 mapping** | Maps extracted entities to ICD-10-CM codes via exact → fuzzy → embedding matching |
| **Severity classifier** | Fine-tunes Bio_ClinicalBERT to classify notes as `routine`, `urgent`, or `critical` |
| **Co-occurrence graph** | Builds an interactive network of entity pairs that appear together in clinical notes |
| **FastAPI** | REST API serving all NLP functionality |
| **Streamlit demo** | Live public demo — paste any clinical note, get results in real time |

---

## Live demo

🔗 [clinical-nlp.streamlit.app](https://clinical-nlp.streamlit.app) *(deploy your own — see VSCODE_GUIDE.md)*

---

## Quick start

```bash
git clone https://github.com/<your-username>/clinical-nlp-pipeline
cd clinical-nlp-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Install the scispaCy NER model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz

# Download MTSamples from Kaggle → data/raw/mtsamples.csv
# Then run the pipeline
python -m src.etl.pipeline --dry-run
uvicorn src.api.main:app --reload &
streamlit run dashboard/app.py
```

Full step-by-step instructions: **[VSCODE_GUIDE.md](VSCODE_GUIDE.md)**

---

## Tech stack

| Layer | Library |
|-------|---------|
| NER | spaCy + scispaCy `en_core_sci_lg` |
| Classification | HuggingFace Transformers + `Bio_ClinicalBERT` |
| ICD-10 fuzzy | rapidfuzz |
| ICD-10 embeddings | sentence-transformers |
| API | FastAPI + Pydantic |
| Database | SQLAlchemy (SQLite local / PostgreSQL cloud) |
| Dashboard | Streamlit |
| Visualisation | Plotly + NetworkX + pyvis |
| Testing | pytest (61 tests) |
| CI/CD | GitHub Actions |

---

## Project structure

```
src/
  utils/       config, logger, text cleaning utilities
  etl/         extract → transform → load pipeline
  nlp/         ner, icd_mapper, classifier, cooccurrence
  db/           connection, ORM models, repository layer
  api/          FastAPI app, Pydantic schemas, route handlers
dashboard/
  app.py        Streamlit entry point
  api_client.py typed HTTP client for the API
  pages/        demo, explorer, model_metrics
notebooks/
  00_data_exploration.ipynb
  01_ner_walkthrough.ipynb
  02_icd_mapping.ipynb
  03_classification.ipynb   ← fine-tune Bio_ClinicalBERT
  04_visualisation.ipynb
tests/          pytest test suite — 61 tests, 0 dependencies on GPU
sql/            schema.sql for Supabase migration
```

---

## Dataset

**MTSamples** — 4,999 de-identified medical transcriptions across 40 specialties.
Download free from [Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions).

MIMIC-III discharge summaries are optionally supported (requires PhysioNet credentialing).

---

## Severity labels

MTSamples has no severity labels, so we derive them using weak supervision:

| Label | Signal |
|-------|--------|
| `critical` | ICU, ventilator, cardiac arrest, stroke, respiratory failure |
| `urgent` | Emergency, acute, admitted, infection, chest pain, unstable |
| `routine` | Elective, outpatient, follow-up, stable, screening |

The classifier learns to generalise beyond these keyword rules.
Expected performance: ~75–80% accuracy, ~0.75 weighted F1 on MTSamples.

---

## Deployment

See **[VSCODE_GUIDE.md](VSCODE_GUIDE.md)** — Phase 8 covers:
- Supabase (free PostgreSQL for the database)
- Railway or Render (free API hosting)
- Streamlit Cloud (free dashboard hosting)

Total cloud cost for a portfolio demo: **£0/month**.

---

## Running tests

```bash
pytest                              # all 61 tests
pytest --cov=src                    # with coverage
pytest tests/test_ner.py -v         # one file
```

---

## Author

Built by Ayodeji as part of a HealthTech Data Engineering portfolio.

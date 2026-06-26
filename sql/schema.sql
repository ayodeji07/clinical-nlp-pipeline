-- sql/schema.sql
-- Clinical NLP Pipeline — database schema
--
-- Compatible with both SQLite (local) and PostgreSQL (Supabase).
-- SQLAlchemy creates these tables automatically via create_all_tables()
-- but this file lets you inspect or manually apply the schema.
--
-- To apply to Supabase:
--   1. Go to Supabase project → SQL Editor
--   2. Paste this file and click Run

CREATE TABLE IF NOT EXISTS clinical_notes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id     TEXT    NOT NULL UNIQUE,
    transcription TEXT    NOT NULL,
    specialty     TEXT,
    note_type     TEXT,
    severity      TEXT,
    word_count    INTEGER,
    data_source   TEXT    NOT NULL DEFAULT 'mtsamples',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_notes_specialty ON clinical_notes (specialty);
CREATE INDEX IF NOT EXISTS ix_notes_severity  ON clinical_notes (severity);
CREATE INDEX IF NOT EXISTS ix_notes_source_id ON clinical_notes (source_id);

CREATE TABLE IF NOT EXISTS entities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id    INTEGER NOT NULL REFERENCES clinical_notes(id) ON DELETE CASCADE,
    text       TEXT    NOT NULL,
    label      TEXT    NOT NULL,
    start_char INTEGER NOT NULL DEFAULT 0,
    end_char   INTEGER NOT NULL DEFAULT 0,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_entities_note_id ON entities (note_id);
CREATE INDEX IF NOT EXISTS ix_entities_label   ON entities (label);

CREATE TABLE IF NOT EXISTS icd10_mappings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id    INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    icd10_code   TEXT    NOT NULL,
    description  TEXT    NOT NULL DEFAULT '',
    match_method TEXT    NOT NULL DEFAULT 'lookup',
    confidence   REAL    NOT NULL DEFAULT 0.0,
    rank         INTEGER NOT NULL DEFAULT 1,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_icd10_entity_id ON icd10_mappings (entity_id);
CREATE INDEX IF NOT EXISTS ix_icd10_code      ON icd10_mappings (icd10_code);

CREATE TABLE IF NOT EXISTS model_runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name       TEXT    NOT NULL,
    task             TEXT    NOT NULL,
    training_samples INTEGER,
    val_accuracy     REAL,
    val_f1           REAL,
    epochs           INTEGER,
    is_deployed      INTEGER NOT NULL DEFAULT 0,
    run_notes        TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_model_runs_task ON model_runs (task);

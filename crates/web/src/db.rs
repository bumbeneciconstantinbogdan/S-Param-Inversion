//! SQLite database for dataset and training run persistence.

use std::fmt;
use std::path::Path;

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use sparam_data::generation::PermittivitySample;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS datasets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    train_samples   INTEGER NOT NULL,
    val_samples     INTEGER NOT NULL,
    test_samples    INTEGER NOT NULL,
    config_json     TEXT NOT NULL,
    data_dir        TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dataset_samples (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id       INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    split            TEXT    NOT NULL,
    row_idx          INTEGER NOT NULL,
    s11_real         REAL    NOT NULL,
    s11_imag         REAL    NOT NULL,
    s21_real         REAL    NOT NULL,
    s21_imag         REAL    NOT NULL,
    eps_prime        REAL    NOT NULL,
    eps_double_prime REAL    NOT NULL,
    is_dense_patch   INTEGER NOT NULL DEFAULT 0  -- 1 = dense low-ε patch overlay
);
CREATE INDEX IF NOT EXISTS idx_dataset_samples_lookup
    ON dataset_samples(dataset_id, split, row_idx);

CREATE TABLE IF NOT EXISTS training_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id          INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    status              TEXT NOT NULL DEFAULT 'pending',
    model_name          TEXT NOT NULL DEFAULT '',
    config_json         TEXT NOT NULL,
    metrics_json        TEXT,
    history_json        TEXT,
    eval_charts_json    TEXT,
    weights_blob        BLOB,
    best_epoch          INTEGER,
    final_epoch         INTEGER,
    stopped_early       INTEGER,
    training_time_secs  REAL,
    output_dir          TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at        TEXT
);

CREATE TABLE IF NOT EXISTS hpo_studies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    model_type      TEXT NOT NULL DEFAULT 'real',
    status          TEXT NOT NULL DEFAULT 'pending',
    config_json     TEXT NOT NULL,
    n_trials        INTEGER NOT NULL,
    n_completed     INTEGER NOT NULL DEFAULT 0,
    n_feasible      INTEGER NOT NULL DEFAULT 0,
    pareto_size     INTEGER NOT NULL DEFAULT 0,
    total_time_secs REAL,
    results_json    TEXT,
    output_dir      TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at    TEXT
);

CREATE TABLE IF NOT EXISTS hpo_trials (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    study_id        INTEGER NOT NULL REFERENCES hpo_studies(id) ON DELETE CASCADE,
    trial_number    INTEGER NOT NULL,
    status          TEXT NOT NULL,
    model_type      TEXT NOT NULL DEFAULT '',
    params_json     TEXT NOT NULL DEFAULT '{}',
    metrics_json    TEXT,
    objectives_json TEXT,
    is_feasible     INTEGER NOT NULL DEFAULT 1,
    is_pareto       INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"#;

pub fn open_database(path: &Path) -> rusqlite::Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch("PRAGMA journal_mode = WAL;")?;
    conn.execute_batch("PRAGMA foreign_keys = ON;")?;

    migrate(&conn)?;

    Ok(conn)
}

fn migrate(conn: &Connection) -> rusqlite::Result<()> {
    // MVP schema had a `projects` FK; nuke any leftover MVP DB.
    let has_projects: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='projects')",
        [],
        |row| row.get(0),
    )?;
    if has_projects {
        conn.execute_batch(
            "DROP TABLE IF EXISTS training_runs;
             DROP TABLE IF EXISTS datasets;
             DROP TABLE IF EXISTS projects;",
        )?;
    }

    conn.execute_batch(SCHEMA)?;

    let has_model_name: bool = conn
        .prepare("SELECT model_name FROM training_runs LIMIT 0")
        .is_ok();
    if !has_model_name {
        conn.execute_batch(
            "ALTER TABLE training_runs ADD COLUMN model_name TEXT NOT NULL DEFAULT '';",
        )?;
    }

    let has_history_json: bool = conn
        .prepare("SELECT history_json FROM training_runs LIMIT 0")
        .is_ok();
    if !has_history_json {
        conn.execute_batch("ALTER TABLE training_runs ADD COLUMN history_json TEXT;")?;
    }
    let has_eval_charts_json: bool = conn
        .prepare("SELECT eval_charts_json FROM training_runs LIMIT 0")
        .is_ok();
    if !has_eval_charts_json {
        conn.execute_batch("ALTER TABLE training_runs ADD COLUMN eval_charts_json TEXT;")?;
    }

    let has_weights_blob: bool = conn
        .prepare("SELECT weights_blob FROM training_runs LIMIT 0")
        .is_ok();
    if !has_weights_blob {
        conn.execute_batch("ALTER TABLE training_runs ADD COLUMN weights_blob BLOB;")?;
    }

    let has_ensemble_weights: bool = conn
        .prepare("SELECT ensemble_weights_json FROM training_runs LIMIT 0")
        .is_ok();
    if !has_ensemble_weights {
        conn.execute_batch(
            "ALTER TABLE training_runs ADD COLUMN ensemble_weights_json TEXT;",
        )?;
    }

    let has_is_dense_patch: bool = conn
        .prepare("SELECT is_dense_patch FROM dataset_samples LIMIT 0")
        .is_ok();
    if !has_is_dense_patch {
        conn.execute_batch(
            "ALTER TABLE dataset_samples
                ADD COLUMN is_dense_patch INTEGER NOT NULL DEFAULT 0;",
        )?;
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRow {
    pub id: i64,
    pub train_samples: i64,
    pub val_samples: i64,
    pub test_samples: i64,
    pub data_dir: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRunRow {
    pub id: i64,
    pub dataset_id: i64,
    pub status: String,
    pub model_name: String,
    pub best_epoch: Option<i64>,
    pub final_epoch: Option<i64>,
    pub training_time_secs: Option<f64>,
    pub output_dir: String,
    pub created_at: String,
    pub completed_at: Option<String>,
}

// Dataset queries

pub fn list_datasets(conn: &Connection) -> rusqlite::Result<Vec<DatasetRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, train_samples, val_samples, test_samples, data_dir, created_at
         FROM datasets ORDER BY created_at DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(DatasetRow {
            id: row.get(0)?,
            train_samples: row.get(1)?,
            val_samples: row.get(2)?,
            test_samples: row.get(3)?,
            data_dir: row.get(4)?,
            created_at: row.get(5)?,
        })
    })?;
    rows.collect()
}

pub fn get_dataset(conn: &Connection, id: i64) -> rusqlite::Result<DatasetRow> {
    conn.query_row(
        "SELECT id, train_samples, val_samples, test_samples, data_dir, created_at
         FROM datasets WHERE id = ?1",
        params![id],
        |row| {
            Ok(DatasetRow {
                id: row.get(0)?,
                train_samples: row.get(1)?,
                val_samples: row.get(2)?,
                test_samples: row.get(3)?,
                data_dir: row.get(4)?,
                created_at: row.get(5)?,
            })
        },
    )
}

pub fn insert_dataset(
    conn: &Connection,
    train_samples: usize,
    val_samples: usize,
    test_samples: usize,
    config_json: &str,
    data_dir: &str,
) -> rusqlite::Result<i64> {
    conn.execute(
        "INSERT INTO datasets (train_samples, val_samples, test_samples, config_json, data_dir)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![train_samples as i64, val_samples as i64, test_samples as i64, config_json, data_dir],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn delete_dataset(conn: &Connection, id: i64) -> rusqlite::Result<String> {
    let data_dir: String = conn.query_row(
        "SELECT data_dir FROM datasets WHERE id = ?1",
        params![id],
        |row| row.get(0),
    )?;
    conn.execute("DELETE FROM datasets WHERE id = ?1", params![id])?;
    Ok(data_dir)
}

// Training run queries

pub fn list_training_runs(conn: &Connection) -> rusqlite::Result<Vec<TrainingRunRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, dataset_id, status, model_name, best_epoch, final_epoch,
                training_time_secs, output_dir, created_at, completed_at
         FROM training_runs ORDER BY created_at DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(TrainingRunRow {
            id: row.get(0)?,
            dataset_id: row.get(1)?,
            status: row.get(2)?,
            model_name: row.get(3)?,
            best_epoch: row.get(4)?,
            final_epoch: row.get(5)?,
            training_time_secs: row.get(6)?,
            output_dir: row.get(7)?,
            created_at: row.get(8)?,
            completed_at: row.get(9)?,
        })
    })?;
    rows.collect()
}

pub fn get_training_run(conn: &Connection, id: i64) -> rusqlite::Result<TrainingRunRow> {
    conn.query_row(
        "SELECT id, dataset_id, status, model_name, best_epoch, final_epoch,
                training_time_secs, output_dir, created_at, completed_at
         FROM training_runs WHERE id = ?1",
        params![id],
        |row| {
            Ok(TrainingRunRow {
                id: row.get(0)?,
                dataset_id: row.get(1)?,
                status: row.get(2)?,
                model_name: row.get(3)?,
                best_epoch: row.get(4)?,
                final_epoch: row.get(5)?,
                training_time_secs: row.get(6)?,
                output_dir: row.get(7)?,
                created_at: row.get(8)?,
                completed_at: row.get(9)?,
            })
        },
    )
}

pub fn list_completed_runs(conn: &Connection) -> rusqlite::Result<Vec<TrainingRunRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, dataset_id, status, model_name, best_epoch, final_epoch,
                training_time_secs, output_dir, created_at, completed_at
         FROM training_runs WHERE status = 'completed' ORDER BY created_at DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(TrainingRunRow {
            id: row.get(0)?,
            dataset_id: row.get(1)?,
            status: row.get(2)?,
            model_name: row.get(3)?,
            best_epoch: row.get(4)?,
            final_epoch: row.get(5)?,
            training_time_secs: row.get(6)?,
            output_dir: row.get(7)?,
            created_at: row.get(8)?,
            completed_at: row.get(9)?,
        })
    })?;
    rows.collect()
}

pub fn insert_training_run(
    conn: &Connection,
    dataset_id: i64,
    model_name: &str,
    config_json: &str,
    output_dir: &str,
) -> rusqlite::Result<i64> {
    conn.execute(
        "INSERT INTO training_runs (dataset_id, status, model_name, config_json, output_dir)
         VALUES (?1, 'running', ?2, ?3, ?4)",
        params![dataset_id, model_name, config_json, output_dir],
    )?;
    Ok(conn.last_insert_rowid())
}

// `stopped_early` lives inside `metrics_json` via `TrainMetricsSummary`;
// the legacy column stays NULL for new writes.
pub fn complete_training_run(
    conn: &Connection,
    run_id: i64,
    metrics_json: &str,
    best_epoch: usize,
    final_epoch: usize,
    training_time_secs: f64,
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE training_runs SET
            status = 'completed', metrics_json = ?2, best_epoch = ?3, final_epoch = ?4,
            training_time_secs = ?5, completed_at = datetime('now')
         WHERE id = ?1",
        params![run_id, metrics_json, best_epoch as i64, final_epoch as i64, training_time_secs],
    )?;
    Ok(())
}

pub fn fail_training_run(conn: &Connection, run_id: i64, error: &str) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE training_runs SET status = 'failed', metrics_json = ?2, completed_at = datetime('now')
         WHERE id = ?1",
        params![run_id, format!(r#"{{"error":"{}"}}"#, error.replace('"', "\\\""))],
    )?;
    Ok(())
}

pub fn delete_training_run(conn: &Connection, id: i64) -> rusqlite::Result<String> {
    let output_dir: String = conn.query_row(
        "SELECT output_dir FROM training_runs WHERE id = ?1",
        params![id],
        |row| row.get(0),
    )?;
    conn.execute("DELETE FROM training_runs WHERE id = ?1", params![id])?;
    Ok(output_dir)
}

// HPO study queries

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HpoStudyRow {
    pub id: i64,
    pub name: String,
    pub dataset_id: i64,
    pub model_type: String,
    pub status: String,
    pub config_json: String,
    pub n_trials: i64,
    pub n_completed: i64,
    pub n_feasible: i64,
    pub pareto_size: i64,
    pub total_time_secs: Option<f64>,
    pub output_dir: String,
    pub created_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HpoTrialRow {
    pub id: i64,
    pub study_id: i64,
    pub trial_number: i64,
    pub status: String,
    pub model_type: String,
    pub params_json: String,
    pub metrics_json: Option<String>,
    pub objectives_json: Option<String>,
    pub is_feasible: bool,
    pub is_pareto: bool,
    pub created_at: String,
}

pub fn list_hpo_studies(conn: &Connection) -> rusqlite::Result<Vec<HpoStudyRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, name, dataset_id, model_type, status, COALESCE(config_json,'{}'), n_trials, n_completed,
                n_feasible, pareto_size, total_time_secs, output_dir, created_at, completed_at
         FROM hpo_studies ORDER BY created_at DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(HpoStudyRow {
            id: row.get(0)?,
            name: row.get(1)?,
            dataset_id: row.get(2)?,
            model_type: row.get(3)?,
            status: row.get(4)?,
            config_json: row.get(5)?,
            n_trials: row.get(6)?,
            n_completed: row.get(7)?,
            n_feasible: row.get(8)?,
            pareto_size: row.get(9)?,
            total_time_secs: row.get(10)?,
            output_dir: row.get(11)?,
            created_at: row.get(12)?,
            completed_at: row.get(13)?,
        })
    })?;
    rows.collect()
}

pub fn get_hpo_study(conn: &Connection, id: i64) -> rusqlite::Result<HpoStudyRow> {
    conn.query_row(
        "SELECT id, name, dataset_id, model_type, status, COALESCE(config_json,'{}'), n_trials, n_completed,
                n_feasible, pareto_size, total_time_secs, output_dir, created_at, completed_at
         FROM hpo_studies WHERE id = ?1",
        params![id],
        |row| {
            Ok(HpoStudyRow {
                id: row.get(0)?,
                name: row.get(1)?,
                dataset_id: row.get(2)?,
                model_type: row.get(3)?,
                status: row.get(4)?,
                config_json: row.get(5)?,
                n_trials: row.get(6)?,
                n_completed: row.get(7)?,
                n_feasible: row.get(8)?,
                pareto_size: row.get(9)?,
                total_time_secs: row.get(10)?,
                output_dir: row.get(11)?,
                created_at: row.get(12)?,
                completed_at: row.get(13)?,
            })
        },
    )
}

pub fn insert_hpo_study(
    conn: &Connection,
    name: &str,
    dataset_id: i64,
    model_type: &str,
    config_json: &str,
    n_trials: usize,
    output_dir: &str,
) -> rusqlite::Result<i64> {
    conn.execute(
        "INSERT INTO hpo_studies (name, dataset_id, model_type, status, config_json, n_trials, output_dir)
         VALUES (?1, ?2, ?3, 'running', ?4, ?5, ?6)",
        params![name, dataset_id, model_type, config_json, n_trials as i64, output_dir],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn update_hpo_progress(
    conn: &Connection,
    study_id: i64,
    n_completed: usize,
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE hpo_studies SET n_completed = ?2 WHERE id = ?1",
        params![study_id, n_completed as i64],
    )?;
    Ok(())
}

pub fn complete_hpo_study(
    conn: &Connection,
    study_id: i64,
    n_feasible: usize,
    pareto_size: usize,
    total_time_secs: f64,
    results_json: &str,
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE hpo_studies SET status='completed', n_feasible=?2, pareto_size=?3,
         total_time_secs=?4, results_json=?5, completed_at=datetime('now') WHERE id=?1",
        params![study_id, n_feasible as i64, pareto_size as i64, total_time_secs, results_json],
    )?;
    Ok(())
}

pub fn fail_hpo_study(conn: &Connection, study_id: i64, error: &str) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE hpo_studies SET status='failed', results_json=?2, completed_at=datetime('now') WHERE id=?1",
        params![study_id, format!(r#"{{"error":"{}"}}"#, error.replace('"', "\\\""))],
    )?;
    Ok(())
}

pub fn delete_hpo_study(conn: &Connection, id: i64) -> rusqlite::Result<()> {
    conn.execute("DELETE FROM hpo_studies WHERE id = ?1", params![id])?;
    Ok(())
}

pub fn insert_hpo_trial(
    conn: &Connection,
    study_id: i64,
    trial_number: usize,
    status: &str,
    model_type: &str,
    params_json: &str,
    metrics_json: &str,
    objectives_json: &str,
    is_feasible: bool,
) -> rusqlite::Result<i64> {
    conn.execute(
        "INSERT INTO hpo_trials (study_id, trial_number, status, model_type, params_json, metrics_json, objectives_json, is_feasible)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![study_id, trial_number as i64, status, model_type, params_json, metrics_json, objectives_json, is_feasible as i64],
    )?;
    Ok(conn.last_insert_rowid())
}

/// `model_type` scopes the update; running `both` produces twin
/// studies that share `trial_number` and would otherwise collide.
pub fn mark_trial_pareto(
    conn: &Connection,
    study_id: i64,
    model_type: &str,
    trial_number: usize,
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE hpo_trials SET is_pareto = 1
         WHERE study_id = ?1 AND model_type = ?2 AND trial_number = ?3",
        params![study_id, model_type, trial_number as i64],
    )?;
    Ok(())
}

pub fn list_hpo_trials(conn: &Connection, study_id: i64) -> rusqlite::Result<Vec<HpoTrialRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, study_id, trial_number, status, model_type, params_json, metrics_json,
                objectives_json, is_feasible, is_pareto, created_at
         FROM hpo_trials WHERE study_id = ?1 ORDER BY trial_number ASC",
    )?;
    let rows = stmt.query_map(params![study_id], |row| {
        Ok(HpoTrialRow {
            id: row.get(0)?,
            study_id: row.get(1)?,
            trial_number: row.get(2)?,
            status: row.get(3)?,
            model_type: row.get(4)?,
            params_json: row.get(5)?,
            metrics_json: row.get(6)?,
            objectives_json: row.get(7)?,
            is_feasible: row.get::<_, i64>(8)? != 0,
            is_pareto: row.get::<_, i64>(9)? != 0,
            created_at: row.get(10)?,
        })
    })?;
    rows.collect()
}

// Dataset samples

pub fn insert_dataset_samples(
    conn: &mut Connection,
    dataset_id: i64,
    split: &str,
    samples: &[PermittivitySample],
) -> rusqlite::Result<()> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO dataset_samples
                (dataset_id, split, row_idx, s11_real, s11_imag, s21_real, s21_imag, eps_prime, eps_double_prime, is_dense_patch)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        )?;
        for (idx, s) in samples.iter().enumerate() {
            stmt.execute(params![
                dataset_id,
                split,
                idx as i64,
                s.s11_real,
                s.s11_imag,
                s.s21_real,
                s.s21_imag,
                s.eps_prime,
                s.eps_double_prime,
                if s.is_dense_patch { 1_i64 } else { 0 },
            ])?;
        }
    }
    tx.commit()
}

/// Returns `(train, val, test)` for one dataset.
pub fn load_dataset_splits(
    conn: &Connection,
    dataset_id: i64,
) -> rusqlite::Result<(Vec<PermittivitySample>, Vec<PermittivitySample>, Vec<PermittivitySample>)> {
    let train = load_dataset_samples(conn, dataset_id, "train")?;
    let val = load_dataset_samples(conn, dataset_id, "val")?;
    let test = load_dataset_samples(conn, dataset_id, "test")?;
    Ok((train, val, test))
}

pub fn load_dataset_samples(
    conn: &Connection,
    dataset_id: i64,
    split: &str,
) -> rusqlite::Result<Vec<PermittivitySample>> {
    let mut stmt = conn.prepare(
        "SELECT s11_real, s11_imag, s21_real, s21_imag, eps_prime, eps_double_prime, is_dense_patch
         FROM dataset_samples
         WHERE dataset_id = ?1 AND split = ?2
         ORDER BY row_idx ASC",
    )?;
    let rows = stmt.query_map(params![dataset_id, split], |row| {
        let dense_flag: i64 = row.get(6).unwrap_or(0);
        Ok(PermittivitySample {
            s11_real: row.get(0)?,
            s11_imag: row.get(1)?,
            s21_real: row.get(2)?,
            s21_imag: row.get(3)?,
            eps_prime: row.get(4)?,
            eps_double_prime: row.get(5)?,
            is_dense_patch: dense_flag != 0,
        })
    })?;
    rows.collect()
}

pub fn update_training_run_artifacts(
    conn: &Connection,
    run_id: i64,
    metrics_json: Option<&str>,
    history_json: Option<&str>,
    config_json: Option<&str>,
) -> rusqlite::Result<()> {
    if metrics_json.is_some() || history_json.is_some() || config_json.is_some() {
        conn.execute(
            "UPDATE training_runs SET
                metrics_json = COALESCE(?1, metrics_json),
                history_json = COALESCE(?2, history_json),
                config_json  = COALESCE(?3, config_json)
             WHERE id = ?4",
            params![metrics_json, history_json, config_json, run_id],
        )?;
    }
    Ok(())
}

pub fn update_training_run_eval_charts(
    conn: &Connection,
    run_id: i64,
    eval_charts_json: &str,
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE training_runs SET eval_charts_json = ?1 WHERE id = ?2",
        params![eval_charts_json, run_id],
    )?;
    Ok(())
}

pub fn get_training_run_config(conn: &Connection, run_id: i64) -> rusqlite::Result<String> {
    conn.query_row(
        "SELECT config_json FROM training_runs WHERE id = ?1",
        params![run_id],
        |row| row.get(0),
    )
}

pub fn get_dataset_config(conn: &Connection, dataset_id: i64) -> rusqlite::Result<String> {
    conn.query_row(
        "SELECT config_json FROM datasets WHERE id = ?1",
        params![dataset_id],
        |row| row.get(0),
    )
}

pub fn get_training_run_history(conn: &Connection, run_id: i64) -> rusqlite::Result<Option<String>> {
    conn.query_row(
        "SELECT history_json FROM training_runs WHERE id = ?1",
        params![run_id],
        |row| row.get(0),
    )
}

pub fn get_training_run_eval_charts(
    conn: &Connection,
    run_id: i64,
) -> rusqlite::Result<Option<String>> {
    conn.query_row(
        "SELECT eval_charts_json FROM training_runs WHERE id = ?1",
        params![run_id],
        |row| row.get(0),
    )
}

pub fn get_training_run_metrics(
    conn: &Connection,
    run_id: i64,
) -> rusqlite::Result<Option<String>> {
    conn.query_row(
        "SELECT metrics_json FROM training_runs WHERE id = ?1",
        params![run_id],
        |row| row.get(0),
    )
}

pub fn update_training_run_weights(
    conn: &Connection,
    run_id: i64,
    weights: &[u8],
) -> rusqlite::Result<()> {
    conn.execute(
        "UPDATE training_runs SET weights_blob = ?1 WHERE id = ?2",
        params![weights, run_id],
    )?;
    Ok(())
}

/// Returns `Ok(None)` when the column is NULL (still flushing).
pub fn get_training_run_weights(
    conn: &Connection,
    run_id: i64,
) -> rusqlite::Result<Option<Vec<u8>>> {
    conn.query_row(
        "SELECT weights_blob FROM training_runs WHERE id = ?1",
        params![run_id],
        |row| row.get(0),
    )
}

/// Stores the stack non-primary members as JSON `[base64, ...]`.
/// Primary lives in `weights_blob`. Empty Vec → empty JSON array.
pub fn update_training_run_ensemble_weights(
    conn: &Connection,
    run_id: i64,
    members: &[Vec<u8>],
) -> rusqlite::Result<()> {
    use base64::Engine;
    let encoded: Vec<String> = members
        .iter()
        .map(|m| base64::engine::general_purpose::STANDARD.encode(m))
        .collect();
    let json = serde_json::to_string(&encoded).map_err(|e| {
        rusqlite::Error::ToSqlConversionFailure(Box::new(e))
    })?;
    conn.execute(
        "UPDATE training_runs SET ensemble_weights_json = ?1 WHERE id = ?2",
        params![json, run_id],
    )?;
    Ok(())
}

/// Returns `Ok(vec![])` for single-model runs (NULL / empty column).
pub fn get_training_run_ensemble_weights(
    conn: &Connection,
    run_id: i64,
) -> rusqlite::Result<Vec<Vec<u8>>> {
    use base64::Engine;
    let json: Option<String> = conn.query_row(
        "SELECT ensemble_weights_json FROM training_runs WHERE id = ?1",
        params![run_id],
        |row| row.get(0),
    )?;
    let Some(json) = json else { return Ok(Vec::new()) };
    let encoded: Vec<String> = serde_json::from_str(&json).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Text,
            Box::new(e),
        )
    })?;
    let mut out = Vec::with_capacity(encoded.len());
    for s in encoded {
        let bytes = base64::engine::general_purpose::STANDARD.decode(&s).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                Box::new(e),
            )
        })?;
        out.push(bytes);
    }
    Ok(out)
}

/// Bundle of everything `/evaluate` and `/infer` need to replay a run.
#[derive(Debug)]
pub struct RunBundle {
    pub run_id: i64,
    pub model_name: String,
    pub dataset_id: i64,
    /// Kept raw because `run_neural_net_inference` re-parses it.
    pub cfg_text: String,
    pub cfg_val: serde_json::Value,
    pub weights: Vec<u8>,
    /// `extra_members` is already populated from `ensemble_weights_json`.
    pub stack: sparam_app::workflows::config::StackedInferenceConfig,
}

#[derive(Debug, Clone, Copy)]
pub enum RunLoadError {
    RunNotFound,
    DatasetNotFound,
    /// Race window between SSE Complete and the BLOB write.
    WeightsNotFlushed,
    ConfigInvalid,
}

impl fmt::Display for RunLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::RunNotFound => "Run not found",
            Self::DatasetNotFound => "Dataset not found",
            Self::WeightsNotFlushed => {
                "Training run has no weights yet — the background writer may \
                 still be flushing. Try again in a moment."
            }
            Self::ConfigInvalid => "Training run config is not valid JSON",
        })
    }
}

pub fn load_run_bundle(
    conn: &Connection,
    run_id: i64,
) -> Result<RunBundle, RunLoadError> {
    let run = get_training_run(conn, run_id).map_err(|_| RunLoadError::RunNotFound)?;
    if get_dataset(conn, run.dataset_id).is_err() {
        return Err(RunLoadError::DatasetNotFound);
    }
    let cfg_text =
        get_training_run_config(conn, run_id).map_err(|_| RunLoadError::ConfigInvalid)?;
    let cfg_val: serde_json::Value =
        serde_json::from_str(&cfg_text).map_err(|_| RunLoadError::ConfigInvalid)?;
    let weights = get_training_run_weights(conn, run_id)
        .map_err(|_| RunLoadError::WeightsNotFlushed)?
        .ok_or(RunLoadError::WeightsNotFlushed)?;
    let mut stack = sparam_app::workflows::config::StackedInferenceConfig::from_config_json(&cfg_val, 42);
    stack.extra_members = get_training_run_ensemble_weights(conn, run_id).unwrap_or_default();
    Ok(RunBundle {
        run_id,
        model_name: run.model_name,
        dataset_id: run.dataset_id,
        cfg_text,
        cfg_val,
        weights,
        stack,
    })
}

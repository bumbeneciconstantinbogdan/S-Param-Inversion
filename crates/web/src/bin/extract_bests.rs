//! extract_bests — pull headline best-trial configs out of `study.db`
//! (Optuna NSGA-II, real-valued) and `sparam.db` (Rust NSGA-III,
//! complex-valued, two seeds), and write them as JSON files into
//! `poster/studies/methodology/configs/`.
//!
//! The methodology volume's M01..M06 sub-studies refer to these JSON
//! files when populating the "Reproduce" recipes.  This binary is the
//! single source of truth for which trial counts as "best" per study;
//! re-running it after the DBs change keeps the configs in sync.
//!
//! Usage:
//!   target/release/extract_bests
//!     [--real-db <path>] [--complex-db <path>] [--out-dir <path>]
//!     [--pareto-table]
//!
//! With `--pareto-table` the output additionally includes the per-H
//! Pareto-best rows that M02 prints in its tables.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use rusqlite::{Connection, OpenFlags};
use serde::Serialize;
use serde_json::Value;
use sparam_web::study_common::arg_owned as arg;

const DEFAULT_REAL_DB: &str =
    ".claude/worktrees/happy-ptolemy-8f8690/artifacts/study.db";
const DEFAULT_COMPLEX_DB: &str = "sparam_data/sparam.db";
const DEFAULT_OUT_DIR: &str = "poster/studies/methodology/configs";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let real_db = arg(&args, "--real-db").unwrap_or(DEFAULT_REAL_DB.to_string());
    let complex_db = arg(&args, "--complex-db").unwrap_or(DEFAULT_COMPLEX_DB.to_string());
    let out_dir = PathBuf::from(arg(&args, "--out-dir").unwrap_or(DEFAULT_OUT_DIR.to_string()));
    let pareto_table = args.iter().any(|a| a == "--pareto-table");

    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("create out_dir {}: {}", out_dir.display(), e);
        return ExitCode::FAILURE;
    }

    if let Err(e) = extract_real(&real_db, &out_dir, pareto_table) {
        eprintln!("real extraction failed: {e}");
        return ExitCode::FAILURE;
    }
    if let Err(e) = extract_complex(&complex_db, &out_dir, pareto_table) {
        eprintln!("complex extraction failed: {e}");
        return ExitCode::FAILURE;
    }

    eprintln!("wrote configs to {}", out_dir.display());
    ExitCode::SUCCESS
}


// ---------------------------------------------------------------------
//  Real (Optuna): study.db
// ---------------------------------------------------------------------

#[derive(Serialize)]
struct RealBest {
    db: String,
    study_id: i64,
    trial_id: i64,
    trial_number: i64,
    objectives: Vec<f64>,
    objective_names: Vec<&'static str>,
    params: HashMap<String, Value>,
    user_attrs: HashMap<String, Value>,
}

#[derive(Serialize)]
struct RealParetoTable {
    rows: Vec<RealParetoRow>,
}

#[derive(Serialize)]
struct RealParetoRow {
    h: i64,
    best_ok1_pct: f64,
    trial_id: i64,
}

fn extract_real(
    db_path: &str,
    out_dir: &PathBuf,
    pareto_table: bool,
) -> rusqlite::Result<()> {
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;

    // The Optuna 2-objective study uses `trial_values` keyed on
    // `objective` (0 = OK@1%, 1 = H).  Pick the trial with the
    // highest OK@1%.
    let (trial_id, ok1, h_obj): (i64, f64, f64) = conn
        .query_row(
            "SELECT t.trial_id, tv0.value, tv1.value
             FROM trials t
             JOIN trial_values tv0 ON tv0.trial_id=t.trial_id AND tv0.objective=0
             JOIN trial_values tv1 ON tv1.trial_id=t.trial_id AND tv1.objective=1
             WHERE t.state='COMPLETE'
             ORDER BY tv0.value DESC
             LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;

    let trial_number: i64 = conn.query_row(
        "SELECT number FROM trials WHERE trial_id=?",
        [trial_id],
        |row| row.get(0),
    )?;

    // Decode each param using its categorical/float distribution JSON.
    let mut stmt = conn.prepare(
        "SELECT param_name, param_value, distribution_json
         FROM trial_params WHERE trial_id=?",
    )?;
    let mut params: HashMap<String, Value> = HashMap::new();
    let rows = stmt.query_map([trial_id], |row| {
        let name: String = row.get(0)?;
        let value: f64 = row.get(1)?;
        let dist: String = row.get(2)?;
        Ok((name, value, dist))
    })?;
    for r in rows {
        let (name, value, dist) = r?;
        let decoded = decode_optuna_param(value, &dist);
        params.insert(name, decoded);
    }

    // user_attrs (e.g. max_err)
    let mut stmt = conn.prepare(
        "SELECT key, value_json FROM trial_user_attributes WHERE trial_id=?",
    )?;
    let mut user_attrs: HashMap<String, Value> = HashMap::new();
    let rows = stmt.query_map([trial_id], |row| {
        let key: String = row.get(0)?;
        let json: String = row.get(1)?;
        Ok((key, json))
    })?;
    for r in rows {
        let (k, j) = r?;
        let v: Value = serde_json::from_str(&j).unwrap_or(Value::String(j));
        user_attrs.insert(k, v);
    }

    let best = RealBest {
        db: db_path.to_string(),
        study_id: 1,
        trial_id,
        trial_number,
        objectives: vec![ok1, h_obj],
        objective_names: vec!["ok1_pct", "hidden_size"],
        params,
        user_attrs,
    };
    let path = out_dir.join("real_best.json");
    fs::write(&path, serde_json::to_string_pretty(&best).unwrap())
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    eprintln!("real_best.json: trial {trial_id} (number {trial_number}), OK@1%={ok1:.2}, H={h_obj}");

    if pareto_table {
        // Per-H Pareto-best (max OK@1% per hidden_size).
        let mut rows = Vec::new();
        for h_idx in [0, 1, 2, 3] {
            let h_value = match h_idx {
                0 => 8,
                1 => 16,
                2 => 32,
                3 => 64,
                _ => unreachable!(),
            };
            let q = conn.query_row(
                "SELECT t.trial_id, tv0.value
                 FROM trials t
                 JOIN trial_params tp ON tp.trial_id=t.trial_id AND tp.param_name='hidden_size' AND tp.param_value=?1
                 JOIN trial_values tv0 ON tv0.trial_id=t.trial_id AND tv0.objective=0
                 WHERE t.state='COMPLETE'
                 ORDER BY tv0.value DESC LIMIT 1",
                [h_idx as f64],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?)),
            );
            if let Ok((tid, ok1)) = q {
                rows.push(RealParetoRow { h: h_value, best_ok1_pct: ok1, trial_id: tid });
            }
        }
        let path = out_dir.join("real_pareto_per_h.json");
        fs::write(
            &path,
            serde_json::to_string_pretty(&RealParetoTable { rows }).unwrap(),
        )
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    }

    Ok(())
}

/// Decode an Optuna param: float distributions return the raw f64,
/// categorical distributions look up the value as an index in the
/// stored choices array.
fn decode_optuna_param(value: f64, distribution_json: &str) -> Value {
    let dist: Value = match serde_json::from_str(distribution_json) {
        Ok(v) => v,
        Err(_) => return Value::from(value),
    };
    let attrs = match dist.get("attributes") {
        Some(a) => a,
        None => return Value::from(value),
    };
    let kind = dist.get("name").and_then(|v| v.as_str()).unwrap_or("");
    if kind == "CategoricalDistribution" {
        if let Some(arr) = attrs.get("choices").and_then(|c| c.as_array()) {
            let idx = value as usize;
            if let Some(choice) = arr.get(idx) {
                return choice.clone();
            }
        }
    }
    // FloatDistribution / IntDistribution → just store the float.
    Value::from(value)
}

// ---------------------------------------------------------------------
//  Complex (Rust NSGA-III): sparam.db
// ---------------------------------------------------------------------

#[derive(Serialize)]
struct ComplexBest {
    db: String,
    study_id: i64,
    study_name: String,
    seed: String,
    trial_id: i64,
    trial_number: i64,
    objectives: Vec<f64>,
    objective_names: Vec<&'static str>,
    params: Value,
}

#[derive(Serialize)]
struct ComplexParetoTable {
    seed: String,
    rows: Vec<ComplexParetoRow>,
}

#[derive(Serialize)]
struct ComplexParetoRow {
    h: i64,
    best_ok1_pct: f64,
    params: i64,
    max_err_pct: f64,
    trial_id: i64,
}

fn extract_complex(
    db_path: &str,
    out_dir: &PathBuf,
    pareto_table: bool,
) -> rusqlite::Result<()> {
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;

    let mut stmt = conn.prepare("SELECT id, name FROM hpo_studies ORDER BY id")?;
    let studies: Vec<(i64, String)> = stmt
        .query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)))?
        .collect::<rusqlite::Result<_>>()?;

    for (study_id, study_name) in studies {
        let seed_label = study_name
            .strip_prefix("hpo_study_complex_seed_")
            .unwrap_or(&study_name)
            .to_string();

        let (best_trial_id, best_trial_number, objectives, params_json) = conn
            .query_row(
                "SELECT id, trial_number, objectives_json, params_json
                 FROM hpo_trials
                 WHERE study_id=?1 AND is_pareto=1
                 ORDER BY CAST(json_extract(objectives_json, '$[0]') AS REAL) DESC
                 LIMIT 1",
                [study_id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                },
            )
            .or_else(|_| {
                // Fallback for SQLite without json_extract path support:
                // pull all Pareto rows and pick best in-process.
                let mut stmt = conn.prepare(
                    "SELECT id, trial_number, objectives_json, params_json
                     FROM hpo_trials WHERE study_id=?1 AND is_pareto=1",
                )?;
                let rows = stmt
                    .query_map([study_id], |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, i64>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                        ))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                let best = rows
                    .into_iter()
                    .max_by(|a, b| {
                        let pa = parse_first_objective(&a.2);
                        let pb = parse_first_objective(&b.2);
                        pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .ok_or_else(|| {
                        rusqlite::Error::QueryReturnedNoRows
                    })?;
                Ok::<_, rusqlite::Error>(best)
            })?;

        let objectives_arr: Vec<f64> =
            serde_json::from_str(&objectives).unwrap_or_default();
        let params_val: Value =
            serde_json::from_str(&params_json).unwrap_or(Value::Null);

        let best = ComplexBest {
            db: db_path.to_string(),
            study_id,
            study_name: study_name.clone(),
            seed: seed_label.clone(),
            trial_id: best_trial_id,
            trial_number: best_trial_number,
            objectives: objectives_arr.clone(),
            objective_names: vec!["ok1_pct", "param_count", "max_err_pct"],
            params: params_val,
        };
        let path =
            out_dir.join(format!("complex_best_seed{}.json", seed_label));
        fs::write(&path, serde_json::to_string_pretty(&best).unwrap())
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        eprintln!(
            "complex_best_seed{}.json: trial {} (number {}), OK@1%={:.2}, params={}, max_err={:.2}",
            seed_label,
            best_trial_id,
            best_trial_number,
            objectives_arr.first().copied().unwrap_or(f64::NAN),
            objectives_arr.get(1).copied().unwrap_or(f64::NAN),
            objectives_arr.get(2).copied().unwrap_or(f64::NAN),
        );

        if pareto_table {
            let mut stmt = conn.prepare(
                "SELECT id, params_json, objectives_json
                 FROM hpo_trials WHERE study_id=?1 AND is_pareto=1",
            )?;
            let rows = stmt
                .query_map([study_id], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            let mut by_h: HashMap<i64, ComplexParetoRow> = HashMap::new();
            for (tid, params_json, obj_json) in rows {
                let p: Value = serde_json::from_str(&params_json).unwrap_or(Value::Null);
                let h = p.get("hidden_size").and_then(|v| v.as_i64()).unwrap_or(0);
                let objs: Vec<f64> = serde_json::from_str(&obj_json).unwrap_or_default();
                let ok1 = objs.first().copied().unwrap_or(f64::NAN);
                let pc = objs.get(1).copied().unwrap_or(f64::NAN) as i64;
                let me = objs.get(2).copied().unwrap_or(f64::NAN);
                by_h
                    .entry(h)
                    .and_modify(|cur| {
                        if ok1 > cur.best_ok1_pct {
                            cur.best_ok1_pct = ok1;
                            cur.params = pc;
                            cur.max_err_pct = me;
                            cur.trial_id = tid;
                        }
                    })
                    .or_insert(ComplexParetoRow {
                        h,
                        best_ok1_pct: ok1,
                        params: pc,
                        max_err_pct: me,
                        trial_id: tid,
                    });
            }
            let mut rows: Vec<ComplexParetoRow> = by_h.into_values().collect();
            rows.sort_by_key(|r| r.h);
            let path = out_dir.join(format!(
                "complex_pareto_per_h_seed{}.json",
                seed_label
            ));
            fs::write(
                &path,
                serde_json::to_string_pretty(&ComplexParetoTable {
                    seed: seed_label,
                    rows,
                })
                .unwrap(),
            )
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        }
    }
    Ok(())
}

fn parse_first_objective(json: &str) -> f64 {
    serde_json::from_str::<Vec<f64>>(json)
        .ok()
        .and_then(|v| v.first().copied())
        .unwrap_or(f64::NAN)
}

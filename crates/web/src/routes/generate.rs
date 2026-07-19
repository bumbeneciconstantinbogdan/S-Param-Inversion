//! Data generation and dataset preview routes.

use std::sync::Arc;

use askama::Template;
use axum::extract::{Path, State};
use axum::response::Html;
use axum::Form;
use serde::Deserialize;

use sparam_app::workflows::generate::run_generate_in_memory;
use sparam_data::generation::{DataGenerationConfig, PermittivitySample};

use crate::db;
use crate::error::{WebError, WebResult, render};
use crate::state::AppState;

#[derive(Template)]
#[template(path = "generate_form.html")]
pub struct GenerateFormTemplate;

#[derive(Template)]
#[template(path = "generate_result.html")]
pub struct GenerateResultTemplate {
    pub dataset_id: i64,
    pub train_samples: usize,
    pub val_samples: usize,
    pub test_samples: usize,
}

/// Per-column statistics for dataset preview.
pub struct ColumnStats {
    pub name: String,
    pub min: String,
    pub max: String,
    pub mean: String,
    pub std: String,
}

/// Stats for one data split.
pub struct SplitStats {
    pub name: String,
    pub count: usize,
    pub columns: Vec<ColumnStats>,
}

#[derive(Template)]
#[template(path = "dataset_preview.html")]
pub struct DatasetPreviewTemplate {
    pub dataset_id: i64,
    pub splits: Vec<SplitStats>,
    pub sample_rows: Vec<SampleRow>,
}

pub struct SampleRow {
    pub s11_real: String,
    pub s11_imag: String,
    pub s21_real: String,
    pub s21_imag: String,
    pub eps_prime: String,
    pub eps_secund: String,
}

#[derive(Deserialize)]
pub struct GenerateForm {
    #[serde(default = "default_grid_nx")]
    pub grid_nx: usize,
    #[serde(default = "default_grid_ny")]
    pub grid_ny: usize,
    #[serde(default = "default_eps_prime_min")]
    pub eps_prime_min: f64,
    #[serde(default = "default_eps_prime_max")]
    pub eps_prime_max: f64,
    #[serde(default = "default_eps_secund_min")]
    pub eps_secund_min: f64,
    #[serde(default = "default_eps_secund_max")]
    pub eps_secund_max: f64,

    // Dense low-ε patch dims (override the previously hardcoded 10/5).
    // 0 disables the patch in the corresponding split.
    #[serde(default = "default_train_patch_n")]
    pub train_patch_nx: usize,
    #[serde(default = "default_train_patch_n")]
    pub train_patch_ny: usize,
    #[serde(default = "default_eval_patch_n")]
    pub eval_patch_nx: usize,
    #[serde(default = "default_eval_patch_n")]
    pub eval_patch_ny: usize,

    // Optional "dense test grid" — when ANY of these are non-zero, the
    // route switches to the two-call generation pattern from
    // `gen_dense_eval.rs`: train + val use the form's standard
    // `grid_nx/ny + train_patch`, while the test split is replaced by
    // the bulk train output of a SECOND generation at the dense
    // dimensions below.  Lets the user create a stress-test eval grid
    // (e.g. 2000 × 1000 + 80 × 80) without paying for a 16 M-row
    // training set.  Leave at 0 to keep the legacy single-call path.
    #[serde(default)]
    pub dense_test_grid_nx: usize,
    #[serde(default)]
    pub dense_test_grid_ny: usize,
    #[serde(default)]
    pub dense_test_patch_nx: usize,
    #[serde(default)]
    pub dense_test_patch_ny: usize,
}

fn default_grid_nx() -> usize { 30 }
fn default_grid_ny() -> usize { 15 }
fn default_eps_prime_min() -> f64 { 1.0 }
fn default_eps_prime_max() -> f64 { 200.0 }
fn default_eps_secund_min() -> f64 { 0.0 }
fn default_eps_secund_max() -> f64 { 100.0 }
fn default_train_patch_n() -> usize { 10 }
fn default_eval_patch_n() -> usize { 5 }

pub async fn form(_state: State<Arc<AppState>>) -> WebResult<Html<String>> {
    render(GenerateFormTemplate)
}

pub async fn run(
    State(state): State<Arc<AppState>>,
    Form(form): Form<GenerateForm>,
) -> WebResult<Html<String>> {
    // Dense low-ε patch overlay enabled by default for new datasets:
    // a (configurable) train / val / test patch over `[1, 2.5] × [0, 1.75]`,
    // with small linear offsets between val/test and train so no
    // points coincide. The bulk grid keeps its usual coverage; the
    // patch concentrates samples in the near-vacuum corner where the
    // bulk grid is sparsest and where models tend to underperform.
    // Patch samples carry `is_dense_patch=true` so the evaluate UI
    // can split them into a separate (toggleable) classification map.
    // `DataGenerationConfig::default()` itself keeps the patch off
    // so internal tests / Python parity fixtures still pass — this
    // route opts in explicitly.
    let standard_config = DataGenerationConfig {
        n_eps_prime_train: form.grid_nx,
        n_eps_double_prime_train: form.grid_ny,
        eps_prime_range: (form.eps_prime_min, form.eps_prime_max),
        eps_double_prime_range: (form.eps_secund_min, form.eps_secund_max),
        // Narrowed from [1, 5] × [0, 5] after observing that worst-
        // case errors cluster in the very-near-vacuum corner; the
        // tighter window keeps the same train/eval densities so
        // cells are ~3× finer (≈ 0.17 in ε', ≈ 0.19 in ε'') than
        // the wider patch produced.
        dense_patch_eps_prime_range: (1.0, 2.5),
        dense_patch_eps_double_prime_range: (0.0, 1.75),
        dense_patch_n_train: (form.train_patch_nx, form.train_patch_ny),
        dense_patch_n_eval: (form.eval_patch_nx, form.eval_patch_ny),
        dense_patch_val_offset: (0.05, 0.025),
        dense_patch_test_offset: (0.10, 0.05),
        ..DataGenerationConfig::default()
    };

    // Two-call dense-test-grid path — same trick `gen_dense_eval.rs`
    // uses.  Activates as soon as ANY of the four `dense_test_*`
    // dimensions is non-zero so the user can ask for a stress-test
    // eval grid without paying for a `0.125 × n_test` training set.
    let use_dense_test = form.dense_test_grid_nx > 0
        || form.dense_test_grid_ny > 0
        || form.dense_test_patch_nx > 0
        || form.dense_test_patch_ny > 0;

    let (train, val, test, used_config_json) = if use_dense_test {
        // Fallback for any dimension the user left blank: use the
        // standard `grid_*` / `train_patch_*` value (so a partial
        // form — e.g. user only typed the bulk dims — still works).
        // Note this is "use the entered value verbatim if non-zero,
        // else fall back" — NOT `.max`, since a smaller dense grid is
        // a valid request (e.g. for a quick smoke test).
        let test_grid_nx = if form.dense_test_grid_nx > 0 {
            form.dense_test_grid_nx
        } else {
            form.grid_nx
        };
        let test_grid_ny = if form.dense_test_grid_ny > 0 {
            form.dense_test_grid_ny
        } else {
            form.grid_ny
        };
        let test_patch_nx = if form.dense_test_patch_nx > 0 {
            form.dense_test_patch_nx
        } else {
            form.train_patch_nx
        };
        let test_patch_ny = if form.dense_test_patch_ny > 0 {
            form.dense_test_patch_ny
        } else {
            form.train_patch_ny
        };
        let dense_config = DataGenerationConfig {
            n_eps_prime_train: test_grid_nx,
            n_eps_double_prime_train: test_grid_ny,
            dense_patch_n_train: (test_patch_nx, test_patch_ny),
            ..standard_config.clone()
        };

        // Run BOTH generations on the blocking pool — the dense one
        // can build 2 M+ samples and would otherwise stall the async
        // runtime.  Done sequentially (each grabs the same Cpu) so we
        // don't double-allocate memory.
        let std_cfg = standard_config.clone();
        let dense_cfg = dense_config.clone();
        let (standard, dense) = tokio::task::spawn_blocking(move || {
            let s = run_generate_in_memory(&std_cfg)?;
            let d = run_generate_in_memory(&dense_cfg)?;
            Ok::<_, candle_core::Error>((s, d))
        })
        .await
        .map_err(|e| WebError::Computation(format!("Task failed: {e}")))?
        .map_err(WebError::from)?;

        // Persist BOTH configs so the dataset is fully reproducible.
        let cfg_json = serde_json::to_string(&serde_json::json!({
            "train_val_config": &standard_config,
            "test_config": &dense_config,
            "note": "two-call generation: standard config produced train+val, dense config's TRAIN output became this dataset's test split (so test points sit at offset (0,0), not test_offset)",
        }))?;
        // `dense.train` becomes our test split — see comment above and
        // the matching helper in `gen_dense_eval.rs`.
        (standard.train, standard.validation, dense.train, cfg_json)
    } else {
        // Single-call legacy path — train/val/test all derived from
        // one config, test_offset applies to the test split.
        let dc = standard_config.clone();
        let result = tokio::task::spawn_blocking(move || run_generate_in_memory(&dc))
            .await
            .map_err(|e| WebError::Computation(format!("Task failed: {e}")))?
            .map_err(WebError::from)?;
        let cfg_json = serde_json::to_string(&standard_config)?;
        (result.train, result.validation, result.test, cfg_json)
    };

    let train_len = train.len();
    let val_len = val.len();
    let test_len = test.len();

    let mut conn = state.db.lock().await;
    let dataset_id = db::insert_dataset(
        &conn,
        train_len,
        val_len,
        test_len,
        &used_config_json,
        "", // data_dir is legacy/unused — samples live in dataset_samples
    )?;
    db::insert_dataset_samples(&mut conn, dataset_id, "train", &train)?;
    db::insert_dataset_samples(&mut conn, dataset_id, "val", &val)?;
    db::insert_dataset_samples(&mut conn, dataset_id, "test", &test)?;
    drop(conn);

    render(GenerateResultTemplate {
        dataset_id,
        train_samples: train_len,
        val_samples: val_len,
        test_samples: test_len,
    })
}

pub async fn preview(
    State(state): State<Arc<AppState>>,
    Path(dataset_id): Path<i64>,
) -> WebResult<Html<String>> {
    let conn = state.db.lock().await;
    let _dataset = db::get_dataset(&conn, dataset_id)
        .map_err(|_| WebError::NotFound(format!("Dataset {dataset_id} not found")))?;
    let train = db::load_dataset_samples(&conn, dataset_id, "train")?;
    let val = db::load_dataset_samples(&conn, dataset_id, "val")?;
    let test = db::load_dataset_samples(&conn, dataset_id, "test")?;
    drop(conn);

    let splits = vec![
        SplitStats { name: "Training".into(), count: train.len(), columns: compute_column_stats(&train) },
        SplitStats { name: "Validation".into(), count: val.len(), columns: compute_column_stats(&val) },
        SplitStats { name: "Test".into(), count: test.len(), columns: compute_column_stats(&test) },
    ];

    let sample_rows: Vec<SampleRow> = train
        .iter()
        .take(10)
        .map(|s| SampleRow {
            s11_real: format!("{:.6}", s.s11_real),
            s11_imag: format!("{:.6}", s.s11_imag),
            s21_real: format!("{:.6}", s.s21_real),
            s21_imag: format!("{:.6}", s.s21_imag),
            eps_prime: format!("{:.4}", s.eps_prime),
            eps_secund: format!("{:.4}", s.eps_double_prime),
        })
        .collect();

    render(DatasetPreviewTemplate {
        dataset_id,
        splits,
        sample_rows,
    })
}

fn compute_column_stats(samples: &[PermittivitySample]) -> Vec<ColumnStats> {
    if samples.is_empty() {
        return Vec::new();
    }
    let n = samples.len() as f64;

    let columns: Vec<(&str, Vec<f64>)> = vec![
        ("S11 real", samples.iter().map(|s| s.s11_real).collect()),
        ("S11 imag", samples.iter().map(|s| s.s11_imag).collect()),
        ("S21 real", samples.iter().map(|s| s.s21_real).collect()),
        ("S21 imag", samples.iter().map(|s| s.s21_imag).collect()),
        ("eps'", samples.iter().map(|s| s.eps_prime).collect()),
        ("eps''", samples.iter().map(|s| s.eps_double_prime).collect()),
    ];

    columns
        .into_iter()
        .map(|(name, vals)| {
            let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean = vals.iter().sum::<f64>() / n;
            let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            let std = variance.sqrt();
            ColumnStats {
                name: name.to_string(),
                min: format!("{min:.6}"),
                max: format!("{max:.6}"),
                mean: format!("{mean:.6}"),
                std: format!("{std:.6}"),
            }
        })
        .collect()
}


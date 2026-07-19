//! Evaluation routes: evaluate a trained model on test data with plots.

use std::sync::Arc;

use askama::Template;
use axum::extract::State;
use axum::response::Html;
use axum::Form;
use serde::Deserialize;

use sparam_app::workflows::config::{CheckpointSource, EvaluateConfig, NrwEvaluateConfig};
use sparam_app::workflows::evaluate::{run_evaluation, run_nrw_evaluation};
use sparam_app::workflows::stacked_training::synthesise_train_overlay;
use sparam_app::workflows::train_form::empty_string_as_none;

use crate::db::{self, DatasetRow, TrainingRunRow};
use crate::error::{WebError, WebResult, render};
use crate::state::{AppState, ScalerCacheKey};

#[derive(Template)]
#[template(path = "evaluate_form.html")]
pub struct EvaluateFormTemplate {
    pub runs: Vec<TrainingRunRow>,
    pub datasets: Vec<DatasetRow>,
}

#[derive(Template)]
#[template(path = "evaluate_nrw_form.html")]
pub struct EvaluateNrwFormTemplate {}

pub struct PlotFile {
    pub title: String,
    pub url: String,
}

#[derive(Template)]
#[template(path = "evaluate_result.html")]
pub struct EvaluateResultTemplate {
    pub model_name: String,
    pub test_samples: usize,
    pub r2_real: String,
    pub r2_imag: String,
    pub ok_1pct: String,
    pub ok_10pct: String,
    pub mean_error: String,
    pub max_error: String,
    pub min_error: String,
    pub std_error: String,
    pub plots: Vec<PlotFile>,
    pub charts_json: String,
}

#[derive(Deserialize)]
pub struct EvaluateForm {
    pub run_id: i64,
    /// Empty = the run's original dataset.
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub dataset_id: Option<i64>,
    /// Opt-in toggle for the inference-time NRW refinement step
    /// (Stage 5 of the M09/M10 stack).  When the checkbox is unchecked,
    /// the field is absent from the form post and `apply_refinement`
    /// stays `None`; the route handler treats `None` as "off" and
    /// zeroes `stack.refine_steps` regardless of what the run config
    /// says.  Default behaviour: training-only stack
    /// (overlay + ensemble + median); refinement runs only when the
    /// user explicitly opts in.
    #[serde(default)]
    pub apply_refinement: Option<String>,
}

pub async fn form(
    State(state): State<Arc<AppState>>,
) -> WebResult<Html<String>> {
    let (runs, datasets) = {
        let conn = state.db.lock().await;
        (db::list_completed_runs(&conn)?, db::list_datasets(&conn)?)
    };
    render(EvaluateFormTemplate { runs, datasets })
}

pub async fn run(
    State(state): State<Arc<AppState>>,
    Form(form): Form<EvaluateForm>,
) -> WebResult<Html<String>> {
    let bundle = {
        let conn = state.db.lock().await;
        db::load_run_bundle(&conn, form.run_id).map_err(|e| match e {
            db::RunLoadError::RunNotFound | db::RunLoadError::DatasetNotFound => {
                WebError::NotFound(e.to_string())
            }
            db::RunLoadError::WeightsNotFlushed => WebError::BadRequest(e.to_string()),
            db::RunLoadError::ConfigInvalid => WebError::BadRequest(format!(
                "Training run {}: {e}",
                form.run_id
            )),
        })?
    };
    let db::RunBundle {
        run_id: _,
        model_name,
        dataset_id: default_dataset_id,
        cfg_text: _,
        cfg_val,
        weights,
        stack,
    } = bundle;
    // Opt-in refinement: zero out refine_steps unless the user
    // explicitly checked the form checkbox.  The training-only stack
    // (overlay + ensemble) runs regardless.
    let apply_refinement = form.apply_refinement.is_some();
    let mut stack = stack;
    if !apply_refinement {
        stack.refine_steps = 0;
    }

    let model_config: sparam_app::workflows::config::ModelConfig =
        serde_json::from_value(cfg_val.clone()).map_err(|e| {
            WebError::BadRequest(format!(
                "Training run {}: invalid model config — {e}",
                form.run_id
            ))
        })?;
    // parameter_count is only stamped after a run completes — a
    // missing value usually means the run is still in progress.
    let parameter_count = require_u64(&cfg_val, "parameter_count", form.run_id)? as usize;

    let eval_dataset_id = form.dataset_id.unwrap_or(default_dataset_id);

    // `test_samples` come from the (possibly overridden) eval dataset;
    // `scaler_samples` MUST come from the model's ORIGINAL training
    // dataset so the StandardScaler stats match.
    let (test_samples, scaler_samples) = {
        let conn = state.db.lock().await;
        if eval_dataset_id != default_dataset_id {
            db::get_dataset(&conn, eval_dataset_id).map_err(|_| {
                WebError::NotFound(format!("Eval dataset {eval_dataset_id} not found"))
            })?;
        }
        let test = db::load_dataset_samples(&conn, eval_dataset_id, "test")?;
        let mut train = db::load_dataset_samples(&conn, default_dataset_id, "train")?;
        if stack.overlay_density > 0 {
            let overlay = synthesise_train_overlay(stack.overlay_density, stack.seed)
                .map_err(WebError::from)?;
            train.extend(overlay);
        }
        (test, train)
    };

    let scaler_samples_arc: std::sync::Arc<[sparam_data::generation::PermittivitySample]> =
        scaler_samples.into();
    let cache_key = ScalerCacheKey {
        dataset_id: default_dataset_id,
        is_complex: matches!(model_config, sparam_app::workflows::config::ModelConfig::Complex { .. }),
        overlay_density: stack.overlay_density,
        overlay_seed: stack.seed,
    };
    let scaler_for_fit = std::sync::Arc::clone(&scaler_samples_arc);
    let model_type_for_fit = model_config.model_type();
    let pre_fitted = state
        .get_or_fit_scalers(cache_key, move || {
            fit_scalers_blocking(model_type_for_fit, &scaler_for_fit)
        })
        .await
        .map_err(WebError::from)?;

    let config = EvaluateConfig {
        model: CheckpointSource::Bytes(weights),
        model_config,
        parameter_count,
        test_samples: test_samples.into(),
        scaler_samples: scaler_samples_arc,
        thresholds: (1.0, 10.0),
        generate_plots: true,
        quiet: true,
        stack,
        pre_fitted_scalers: Some(pre_fitted),
    };

    let result = tokio::task::spawn_blocking(move || run_evaluation(&config))
        .await
        .map_err(|e| WebError::Computation(format!("Task failed: {e}")))?
        .map_err(WebError::from)?;

    let charts_json = result
        .charts_json
        .as_ref()
        .map(|v| serde_json::to_string(v).unwrap_or_default())
        .unwrap_or_default();
    let metrics_json = serde_json::to_string(&result.output.metrics).unwrap_or_default();
    {
        let conn = state.db.lock().await;
        if !charts_json.is_empty() {
            let _ = db::update_training_run_eval_charts(&conn, form.run_id, &charts_json);
        }
        let _ = db::update_training_run_artifacts(&conn, form.run_id, Some(&metrics_json), None, None);
    }

    let plots: Vec<PlotFile> = Vec::new();
    let m = &result.output.metrics;

    render(EvaluateResultTemplate {
        model_name,
        test_samples: result.output.test_samples,
        r2_real: format!("{:.5}", m.r2_real),
        r2_imag: format!("{:.5}", m.r2_imag),
        ok_1pct: format!("{:.1}", m.ok_at_1pct),
        ok_10pct: format!("{:.1}", m.ok_at_10pct),
        mean_error: format!("{:.3}", m.mean_error),
        max_error: format!("{:.3}", m.max_error),
        min_error: format!("{:.6}", m.min_error),
        std_error: format!("{:.3}", m.std_error),
        plots,
        charts_json,
    })
}

// Strict config-field accessors. Missing or wrong-typed fields are
// hard errors — silently defaulting would produce a model whose
// architecture mismatches the saved weights BLOB.
fn missing_field_error(run_id: i64, field: &str, type_hint: &str) -> WebError {
    WebError::BadRequest(format!(
        "Training run #{run_id} config is missing or malformed — \
         required field `{field}` ({type_hint}) not found. \
         The run's model cannot be reconstructed; retrain or re-import the run."
    ))
}

fn require_u64(
    cfg: &serde_json::Value, field: &str, run_id: i64,
) -> Result<u64, WebError> {
    cfg.get(field)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| missing_field_error(run_id, field, "non-negative integer"))
}

/// Fit feature + target StandardScalers on `samples` for the given
/// model family. Used as the cache-miss path for `AppState::scaler_cache`.
pub(crate) fn fit_scalers_blocking(
    model_type: sparam_models::ModelType,
    samples: &[sparam_data::generation::PermittivitySample],
) -> Result<sparam_data::scaling::FittedScalers, candle_core::Error> {
    use sparam_data::scaling::{Scaler, StandardScaler};
    let ds = sparam_data::tensor_bridges::samples_to_tensors(model_type, samples, false)?;
    let mut feature = StandardScaler::new();
    feature.fit(&ds.features)?;
    let mut target = StandardScaler::new();
    target.fit(&ds.targets)?;
    Ok(sparam_data::scaling::FittedScalers { feature, target })
}

// NRW analytical round-trip — closed-form baseline, no model needed.
#[derive(Deserialize)]
pub struct NrwEvaluateForm {
    pub d_mm: f64,
    pub a_mm: f64,
    pub frequency_ghz: f64,
    pub eps_prime_min: f64,
    pub eps_prime_max: f64,
    pub eps_double_prime_min: f64,
    pub eps_double_prime_max: f64,
    pub n_eps_prime: usize,
    pub n_eps_double_prime: usize,
}

pub async fn nrw_form() -> WebResult<Html<String>> {
    render(EvaluateNrwFormTemplate {})
}

pub async fn nrw_run(
    Form(form): Form<NrwEvaluateForm>,
) -> WebResult<Html<String>> {
    if form.n_eps_prime == 0 || form.n_eps_double_prime == 0 {
        return Err(WebError::BadRequest(
            "Grid dimensions must be positive (n_eps_prime, n_eps_double_prime)".into(),
        ));
    }
    if form.eps_prime_min >= form.eps_prime_max
        || form.eps_double_prime_min >= form.eps_double_prime_max
    {
        return Err(WebError::BadRequest(
            "Permittivity ranges must satisfy min < max".into(),
        ));
    }
    if form.d_mm <= 0.0 || form.a_mm <= 0.0 || form.frequency_ghz <= 0.0 {
        return Err(WebError::BadRequest(
            "d, a, and frequency must be positive".into(),
        ));
    }

    let config = NrwEvaluateConfig {
        d: form.d_mm * 1e-3,
        a: form.a_mm * 1e-3,
        frequency: form.frequency_ghz * 1e9,
        eps_prime_range: (form.eps_prime_min, form.eps_prime_max),
        eps_double_prime_range: (form.eps_double_prime_min, form.eps_double_prime_max),
        n_eps_prime: form.n_eps_prime,
        n_eps_double_prime: form.n_eps_double_prime,
        thresholds: (1.0, 10.0),
        generate_plots: true,
        quiet: true,
    };

    let result = tokio::task::spawn_blocking(move || run_nrw_evaluation(&config))
        .await
        .map_err(|e| WebError::Computation(format!("Task failed: {e}")))?
        .map_err(WebError::from)?;

    let charts_json = result
        .charts_json
        .as_ref()
        .map(|v| serde_json::to_string(v).unwrap_or_default())
        .unwrap_or_default();

    let m = &result.output.metrics;
    let plots: Vec<PlotFile> = Vec::new();

    render(EvaluateResultTemplate {
        model_name: result.output.model_path.clone(),
        test_samples: result.output.test_samples,
        r2_real: format!("{:.5}", m.r2_real),
        r2_imag: format!("{:.5}", m.r2_imag),
        ok_1pct: format!("{:.1}", m.ok_at_1pct),
        ok_10pct: format!("{:.1}", m.ok_at_10pct),
        mean_error: format!("{:.3}", m.mean_error),
        max_error: format!("{:.3}", m.max_error),
        min_error: format!("{:.6}", m.min_error),
        std_error: format!("{:.3}", m.std_error),
        plots,
        charts_json,
    })
}


//! Multi-model inference comparison route. HTTP glue around
//! [`sparam_app::workflows::infer`].

use std::sync::Arc;

use askama::Template;
use axum::extract::State;
use axum::response::Html;
use axum::Form;
use serde::Deserialize;
use sparam_app::workflows::train_form::empty_string_as_none;

use sparam_app::workflows::infer::{
    NeuralNetInferRequest, run_neural_net_inference, run_nrw_inversion,
};

use crate::db::{self, TrainingRunRow};
use crate::error::{WebError, WebResult, render};
use crate::json_config;
use crate::routes::evaluate::fit_scalers_blocking;
use crate::state::{AppState, ScalerCacheKey};

/// `Some(msg)` if `(ep, edp)` falls outside the dataset's
/// `eps_*_range`. Older runs stored ε″ under `eps_secund_range`, so
/// both keys are checked.
fn extrapolation_warning(
    nrw_ep: f64,
    nrw_edp: f64,
    dataset_cfg: &serde_json::Value,
) -> Option<String> {
    let pair = |key: &str, default: (f64, f64)| -> (f64, f64) {
        dataset_cfg
            .get(key)
            .and_then(|v| v.as_array())
            .and_then(|a| Some((a.first()?.as_f64()?, a.get(1)?.as_f64()?)))
            .unwrap_or(default)
    };
    let (ep_min, ep_max) = pair("eps_prime_range", (1.0, 200.0));
    let (edp_min, edp_max) = dataset_cfg
        .get("eps_double_prime_range")
        .or_else(|| dataset_cfg.get("eps_secund_range"))
        .and_then(|v| v.as_array())
        .and_then(|a| Some((a.first()?.as_f64()?, a.get(1)?.as_f64()?)))
        .unwrap_or((0.0, 100.0));

    let in_range = (ep_min..=ep_max).contains(&nrw_ep)
        && (edp_min..=edp_max).contains(&nrw_edp);
    if in_range {
        return None;
    }
    Some(format!(
        "NRW result ({nrw_ep:.2}, {nrw_edp:.2}) is outside training range \
         [{ep_min:.0}-{ep_max:.0}, {edp_min:.0}-{edp_max:.0}]. \
         NN predictions may be unreliable (extrapolation)."
    ))
}

#[derive(Template)]
#[template(path = "infer_form.html")]
pub struct InferFormTemplate {
    pub runs: Vec<TrainingRunRow>,
    /// Set when arriving from an evaluate-grid cell click; `None` for
    /// manual S-param entry.
    pub prefill_s11_real: Option<f64>,
    pub prefill_s11_imag: Option<f64>,
    pub prefill_s21_real: Option<f64>,
    pub prefill_s21_imag: Option<f64>,
    /// Direct-NRW ground truth carried through hidden form fields
    /// from an evaluate-grid click; reference for the result page's
    /// status column. `None` for manual entry.
    pub prefill_eps_prime_true: Option<f64>,
    pub prefill_eps_double_prime_true: Option<f64>,
}

#[derive(Debug, Deserialize, Default)]
pub struct InferFormPrefill {
    pub s11_real: Option<f64>,
    pub s11_imag: Option<f64>,
    pub s21_real: Option<f64>,
    pub s21_imag: Option<f64>,
    pub eps_prime_true: Option<f64>,
    pub eps_double_prime_true: Option<f64>,
}

pub struct InferenceResult {
    pub method: String,
    pub eps_prime: String,
    pub eps_double_prime: String,
    pub status: String,
    pub warning: String,
    /// Relative % error vs the direct-NRW truth (only set when the
    /// user came in from an evaluate-grid click).
    pub rel_err_pct: String,
    /// Raw numeric ε' for the Re/Im comparison plot. `None` when
    /// inference failed or returned non-finite output, which the
    /// template skips when emitting Plotly traces.
    pub eps_prime_num: Option<f64>,
    pub eps_double_prime_num: Option<f64>,
}

impl InferenceResult {
    /// Run/dataset missing, weights still flushing, etc.
    pub fn not_found(method: &str, reason: &str) -> Self {
        Self {
            method: method.to_string(),
            eps_prime: "N/A".into(),
            eps_double_prime: "N/A".into(),
            status: reason.to_string(),
            warning: String::new(),
            rel_err_pct: String::new(),
            eps_prime_num: None,
            eps_double_prime_num: None,
        }
    }

    /// Inference itself errored (e.g. candle kernel error, NaN out).
    /// model returned NaNs or candle hit a kernel error).  `err` is
    /// shown verbatim in the warning column so the user can copy/paste.
    pub fn computation_failed(method: &str, err: &str) -> Self {
        Self {
            method: method.to_string(),
            eps_prime: "N/A".into(),
            eps_double_prime: "N/A".into(),
            status: "Computation error".into(),
            warning: err.to_string(),
            rel_err_pct: String::new(),
            eps_prime_num: None,
            eps_double_prime_num: None,
        }
    }
}

#[derive(Template)]
#[template(path = "infer_result.html")]
pub struct InferResultTemplate {
    pub results: Vec<InferenceResult>,
    pub input_summary: String,
    pub global_warning: String,
    /// JSON payload `{"truth": {...}|null, "preds": [...]}` consumed
    /// by the Re/Im plot. Pre-serialised in Rust so the template can
    /// embed it verbatim without relying on an askama JSON filter
    /// (which needs an extra feature flag).
    pub chart_data_json: String,
}

#[derive(Deserialize)]
pub struct InferForm {
    pub s11_real: f64,
    pub s11_imag: f64,
    pub s21_real: f64,
    pub s21_imag: f64,
    #[serde(default = "default_frequency")]
    pub frequency_ghz: f64,
    #[serde(default = "default_thickness")]
    pub thickness_mm: f64,
    #[serde(default)]
    pub use_nrw: Option<String>,
    /// Opt-in toggle for the inference-time NRW refinement step
    /// (Stage 5 of the M09/M10 stack).  Default off: just the trained
    /// ensemble (median across members) without the per-sample Adam
    /// loop.  Checkbox checked = run the refinement.
    #[serde(default)]
    pub apply_refinement: Option<String>,
    /// Comma-separated (serde_urlencoded can't deserialize repeated fields to Vec).
    #[serde(default)]
    pub run_ids: String,
    /// ε″ uses the dataset convention (positive loss factor), not the
    /// Complex MLP internal `-ε″`.  Both fields accept an empty string
    /// when the user does not provide a ground truth — the
    /// `empty_string_as_none` deserializer maps that to `None` and the
    /// downstream `eps_prime_true.zip(eps_double_prime_true)` skips
    /// the `Δ vs true ε` column.
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub eps_prime_true: Option<f64>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub eps_double_prime_true: Option<f64>,
}

fn default_frequency() -> f64 { 8.2 }
fn default_thickness() -> f64 { 1.5 }

pub async fn form(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(prefill): axum::extract::Query<InferFormPrefill>,
) -> WebResult<Html<String>> {
    let conn = state.db.lock().await;
    let runs = db::list_completed_runs(&conn)?;
    render(InferFormTemplate {
        runs,
        prefill_s11_real: prefill.s11_real,
        prefill_s11_imag: prefill.s11_imag,
        prefill_s21_real: prefill.s21_real,
        prefill_s21_imag: prefill.s21_imag,
        prefill_eps_prime_true: prefill.eps_prime_true,
        prefill_eps_double_prime_true: prefill.eps_double_prime_true,
    })
}

pub async fn predict(
    State(state): State<Arc<AppState>>,
    Form(form): Form<InferForm>,
) -> WebResult<Html<String>> {
    let mut results = Vec::new();
    let mut global_warning = String::new();
    let input_summary = format!(
        "S11 = {:.4} + {:.4}j, S21 = {:.4} + {:.4}j, f = {} GHz, d = {} mm",
        form.s11_real, form.s11_imag, form.s21_real, form.s21_imag,
        form.frequency_ghz, form.thickness_mm
    );

    // Direct-NRW ground truth, present only when the request came from
    // an evaluate-grid click. Used to score every inversion method
    // against the ε that generated the S, not against another inversion.
    let truth_eps: Option<(f64, f64)> =
        form.eps_prime_true.zip(form.eps_double_prime_true);
    let rel_err_pct = |pred_ep: f64, pred_edp: f64| -> String {
        match truth_eps {
            Some((ep_t, edp_t)) => {
                let de_re = pred_ep - ep_t;
                let de_im = pred_edp - edp_t;
                let abs_diff = (de_re * de_re + de_im * de_im).sqrt();
                let abs_true = (ep_t * ep_t + edp_t * edp_t).sqrt().max(1e-12);
                format!("{:.2}%", abs_diff / abs_true * 100.0)
            }
            None => String::new(),
        }
    };
    let status_for_pred = |pred_ep: f64, pred_edp: f64| -> String {
        match truth_eps {
            Some((ep_t, edp_t)) => {
                let de_re = pred_ep - ep_t;
                let de_im = pred_edp - edp_t;
                let abs_diff = (de_re * de_re + de_im * de_im).sqrt();
                let abs_true = (ep_t * ep_t + edp_t * edp_t).sqrt().max(1e-12);
                let pct = abs_diff / abs_true * 100.0;
                if pct < 1.0 {
                    "OK".into()
                } else if pct < 10.0 {
                    format!("≈ {pct:.2}% off")
                } else {
                    format!("{pct:.1}% off")
                }
            }
            None => "OK".into(),
        }
    };

    // Parse run IDs from comma-separated string.
    let run_id_list: Vec<i64> = form
        .run_ids
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();

    // NRW Inverse (always run if checkbox present)
    let mut nrw_eps: Option<(f64, f64)> = None;
    if form.use_nrw.is_some() {
        let nrw = run_nrw_inversion(
            form.s11_real, form.s11_imag, form.s21_real, form.s21_imag,
            form.frequency_ghz * 1e9, form.thickness_mm * 1e-3,
        );
        nrw_eps = nrw;

        let (status, warning) = match nrw {
            Some((ep, edp)) if ep < 1.0 || edp < 0.0 => (
                "Non-physical".into(),
                format!(
                    "eps'={ep:.2} < 1 or eps''={edp:.2} < 0 — physically impossible for passive dielectrics. \
                     The input S-parameters likely don't correspond to a valid material."
                ),
            ),
            // When ground truth is available the status is bound to
            // the ε-space relative error against it; otherwise we
            // still surface "OK" to mean "physically valid".
            Some((ep, edp)) => (status_for_pred(ep, edp), String::new()),
            None => ("OK".into(), String::new()),
        };
        if !warning.is_empty() {
            global_warning = warning.clone();
        }

        results.push(match nrw {
            Some((ep, edp)) => InferenceResult {
                method: "NRW Inverse".into(),
                eps_prime: format!("{ep:.4}"),
                eps_double_prime: format!("{edp:.4}"),
                status,
                warning,
                rel_err_pct: rel_err_pct(ep, edp),
                eps_prime_num: Some(ep),
                eps_double_prime_num: Some(edp),
            },
            None => InferenceResult::computation_failed(
                "NRW Inverse",
                "The S-parameters produced a non-finite NRW result.",
            ),
        });
    }

    // NN models. RunLoadError becomes a row-level "not found"; other
    // selected runs keep going.
    for &run_id in &run_id_list {
        let (bundle, dataset_cfg, train_samples) = {
            let conn = state.db.lock().await;
            let bundle = match db::load_run_bundle(&conn, run_id) {
                Ok(b) => b,
                Err(e) => {
                    results.push(InferenceResult::not_found(
                        &format!("Run #{run_id}"),
                        &e.to_string(),
                    ));
                    continue;
                }
            };
            let train_samples = db::load_dataset_samples(&conn, bundle.dataset_id, "train")
                .unwrap_or_default();
            let dataset_cfg_json = db::get_dataset_config(&conn, bundle.dataset_id)
                .unwrap_or_else(|_| "{}".into());
            let dataset_cfg = json_config::parse_or_default(&dataset_cfg_json);
            (bundle, dataset_cfg, train_samples)
        };
        let db::RunBundle {
            run_id: _,
            model_name,
            dataset_id: bundle_dataset_id,
            cfg_text,
            cfg_val: _,
            weights,
            stack,
        } = bundle;
        // Opt-in refinement: zero out refine_steps unless the user
        // explicitly checked the form checkbox.  The training-only
        // stack (overlay + ensemble) still runs.
        let apply_refinement = form.apply_refinement.is_some();
        let mut stack = stack;
        if !apply_refinement {
            stack.refine_steps = 0;
        }

        let s_params = sparam_core::s_params::S11S21::new(
            form.s11_real, form.s11_imag, form.s21_real, form.s21_imag,
        );

        let cfg_text_clone = cfg_text.clone();
        // `train_samples` was loaded fresh inside this iteration's DB
        // block — move it in (no clone). Extend in-place only when M10
        // overlay is active; otherwise we keep the original Vec.
        let mut samples_owned = train_samples;
        if stack.overlay_density > 0
            && let Ok(overlay) =
                sparam_app::workflows::stacked_training::synthesise_train_overlay(stack.overlay_density, stack.seed)
        {
            samples_owned.extend(overlay);
        }
        let stack_active = !stack.is_passthrough();

        // `Arc<[T]>` so the same allocation is shared between the
        // (possibly-async) cache-fit closure and the blocking workflow
        // call below — only Arc refcount bumps, no Vec deep clone.
        let samples_arc: std::sync::Arc<[sparam_data::generation::PermittivitySample]> =
            samples_owned.into();

        let model_config_for_key: sparam_app::workflows::config::ModelConfig =
            serde_json::from_str(&cfg_text_clone).map_err(|e| {
                WebError::BadRequest(format!("Run #{run_id}: invalid model config — {e}"))
            })?;
        let model_type_for_key = model_config_for_key.model_type();
        let cache_key = ScalerCacheKey {
            dataset_id: bundle_dataset_id,
            is_complex: matches!(
                model_config_for_key,
                sparam_app::workflows::config::ModelConfig::Complex { .. }
            ),
            overlay_density: stack.overlay_density,
            overlay_seed: stack.seed,
        };
        let samples_for_fit = std::sync::Arc::clone(&samples_arc);
        let pre_fitted = state
            .get_or_fit_scalers(cache_key, move || {
                fit_scalers_blocking(model_type_for_key, &samples_for_fit)
            })
            .await
            .map_err(WebError::from)?;

        let samples_for_workflow = std::sync::Arc::clone(&samples_arc);
        let nn_result = tokio::task::spawn_blocking(move || {
            let req = NeuralNetInferRequest {
                checkpoint_bytes: &weights,
                training_config_json: &cfg_text_clone,
                train_samples: &samples_for_workflow,
                s_params,
                pre_fitted_scalers: Some(&pre_fitted),
            };
            if stack_active {
                sparam_app::workflows::infer::run_neural_net_inference_stacked(
                    req,
                    &stack.extra_members,
                    stack.refine_steps,
                    stack.refine_lr,
                )
                .map_err(|e| e.to_string())
            } else {
                run_neural_net_inference(req).map_err(|e| e.to_string())
            }
        })
        .await
        .map_err(|e| WebError::Computation(format!("Task failed: {e}")))?;

        let range_warning = nrw_eps
            .and_then(|(ep, edp)| extrapolation_warning(ep, edp, &dataset_cfg))
            .unwrap_or_default();
        if !range_warning.is_empty() {
            global_warning = range_warning.clone();
        }

        results.push(match nn_result {
            Ok((ep, edp)) => InferenceResult {
                method: format!("NN: {model_name}"),
                eps_prime: format!("{ep:.4}"),
                eps_double_prime: format!("{edp:.4}"),
                status: status_for_pred(ep, edp),
                warning: range_warning,
                rel_err_pct: rel_err_pct(ep, edp),
                eps_prime_num: Some(ep),
                eps_double_prime_num: Some(edp),
            },
            Err(e) => {
                let mut row = InferenceResult::computation_failed(
                    &format!("NN: {model_name}"),
                    "",
                );
                row.status = e;
                row
            }
        });
    }

    if results.is_empty() {
        return Err(WebError::BadRequest(
            "Select at least one method (NRW or a trained model)".into(),
        ));
    }

    let chart_preds: Vec<serde_json::Value> = results
        .iter()
        .filter_map(|r| {
            let ep = r.eps_prime_num?;
            let edp = r.eps_double_prime_num?;
            Some(serde_json::json!({
                "method": r.method,
                "ep": ep,
                "edp": edp,
            }))
        })
        .collect();
    let chart_truth = truth_eps.map(|(ep, edp)| {
        serde_json::json!({ "ep": ep, "edp": edp })
    });
    let chart_data_json = serde_json::json!({
        "preds": chart_preds,
        "truth": chart_truth,
    })
    .to_string();
    render(InferResultTemplate {
        results,
        input_summary,
        global_warning,
        chart_data_json,
    })
}


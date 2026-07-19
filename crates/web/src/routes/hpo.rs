//! HPO (Hyperparameter Optimization) routes.

use std::convert::Infallible;
use std::sync::Arc;

use askama::Template;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Form;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use sparam_app::workflows::hpo::{HpoTrialEvent, run_hpo_with_callback};

use crate::db::{self, DatasetRow, HpoStudyRow, HpoTrialRow};
use crate::error::{WebError, WebResult, render};
use crate::state::{AppState, HpoBroadcast, HpoSseEvent};

// ── Templates ────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "hpo_dashboard.html")]
pub struct HpoDashboardTemplate {
    pub studies: Vec<HpoStudyRow>,
}

#[derive(Template)]
#[template(path = "hpo_form.html")]
pub struct HpoFormTemplate {
    pub datasets: Vec<DatasetRow>,
}

#[derive(Template)]
#[template(path = "hpo_progress.html")]
pub struct HpoProgressTemplate {
    pub study_id: i64,
    pub name: String,
    pub n_trials: usize,
    pub model_type: String,
}

/// One parameter's importance score for the bar chart.
pub struct ImportanceEntry {
    pub name: String,
    pub importance: String, // pre-formatted percentage
    pub bar_width: String,  // CSS width for bar
    /// Model type this importance was computed for ("real", "complex",
    /// or empty string for single-type studies).
    pub model_type: String,
}

/// Per-objective importance bars for a single (model_type, objective)
/// pair. A study optimizing three objectives produces three of these
/// per model type — see [`ImportanceGroup`].
pub struct ObjectiveImportance {
    /// Machine name (`"ok_at_1pct"` / `"param_count"` / `"max_error"`).
    pub objective: String,
    /// Human label for the section heading (e.g. `"OK@1%"`).
    pub objective_label: String,
    pub entries: Vec<ImportanceEntry>,
    /// fANOVA interaction residual for this objective, as a
    /// pre-formatted percentage string (e.g. `"49.0"`). `Some(_)` only
    /// when the summary JSON carried a `param_interactions_per_objective`
    /// entry for this objective key — i.e. fANOVA succeeded and we're
    /// not in the Spearman-fallback branch. Legacy studies serialised
    /// before the field existed yield `None` and the template skips
    /// the interactions bar entirely.
    pub interactions_pct: Option<String>,
}

/// Grouped importance data for the chart: one group per model type
/// ("real" / "complex"), each containing per-objective sub-groups
/// (one per NSGA-III objective). If the stored study_json is missing
/// the new `param_importance_per_objective` field (legacy rows),
/// this degrades to a single synthetic "overall" objective populated
/// from the legacy `param_importance` field so old studies still
/// render.
pub struct ImportanceGroup {
    pub model_type: String,
    pub objectives: Vec<ObjectiveImportance>,
}

/// One point on the Pareto front for the detailed table.
pub struct ParetoEntry {
    pub trial: i64,
    /// "real" or "complex" — so a "both" study disambiguates trial numbers
    /// shared between the two per-type runs.
    pub model_type: String,
    pub ok_at_1pct: String,
    /// Trained-model parameter count (the NSGA-II complexity objective).
    pub param_count: String,
    pub hidden_size: String,
    pub val_loss: String,
    pub max_error: String,
    pub lr: String,
    pub activation: String,
    pub optimizer: String,
}

#[derive(Template)]
#[template(path = "hpo_detail.html")]
pub struct HpoDetailTemplate {
    pub study: HpoStudyRow,
    pub trials: Vec<HpoTrialRow>,
    pub importance_groups: Vec<ImportanceGroup>,
    pub pareto_entries: Vec<ParetoEntry>,
    pub best_trial_summary: String,
    pub study_config_json: String,
    pub dataset_id: i64,
}

// ── Form ─────────────────────────────────────────────────────────

// `HpoForm` + its defaults live in `sparam_app::workflows::hpo_form`
// so the form schema and the form → `HpoWorkflowConfig` mapping are
// a single source of truth. The route re-exports the type locally so
// Axum's `Form<HpoForm>` extractor sees the same name.
use sparam_app::workflows::hpo_form::HpoForm;

/// NaN / ±Inf → JSON `null` (JSON spec forbids non-finite floats).
#[inline]
fn f64_or_null(v: f64) -> serde_json::Value {
    if v.is_finite() {
        serde_json::Value::from(v)
    } else {
        serde_json::Value::Null
    }
}

// ── Handlers ─────────────────────────────────────────────────────

pub async fn dashboard(
    State(state): State<Arc<AppState>>,
) -> WebResult<Html<String>> {
    let studies = {
        let conn = state.db.lock().await;
        db::list_hpo_studies(&conn)?
    };
    render(HpoDashboardTemplate { studies })
}

pub async fn form(
    State(state): State<Arc<AppState>>,
) -> WebResult<Html<String>> {
    let datasets = {
        let conn = state.db.lock().await;
        db::list_datasets(&conn)?
    };
    render(HpoFormTemplate { datasets })
}

pub async fn launch(
    State(state): State<Arc<AppState>>,
    Form(form): Form<HpoForm>,
) -> WebResult<Html<String>> {
    // Load train + val + test samples from the DB.
    let dataset_id = form.dataset_id;
    let (train_samples, val_samples, test_samples) = {
        let conn = state.db.lock().await;
        db::get_dataset(&conn, dataset_id)
            .map_err(|_| WebError::NotFound("Dataset not found".into()))?;
        db::load_dataset_splits(&conn, dataset_id)?
    };

    // Delegate the full form → workflow-config mapping to the shared
    // `HpoForm::build` — single source of truth for defaults, field
    // names, and `total_trials` = `n_trials × (2 if "both" else 1)`.
    let built = form
        .build(sparam_app::workflows::hpo_form::HpoSplits {
            train: train_samples.into(),
            val: val_samples.into(),
            test: test_samples.into(),
        })
        .map_err(|e| WebError::BadRequest(format!("invalid HPO submission: {e}")))?;

    let sparam_app::workflows::hpo_form::BuiltHpoSubmission {
        model_type,
        study_name,
        dataset_id: built_dataset_id,
        n_trials: _,
        total_trials,
        config,
        config_json,
    } = built;

    let study_id = {
        let conn = state.db.lock().await;
        db::insert_hpo_study(
            &conn,
            &study_name,
            built_dataset_id,
            &model_type,
            &config_json,
            total_trials,
            "", // output_dir is legacy — HPO artifacts live in the DB now.
        )?
    };

    // Create broadcast channel
    let (tx, _) = tokio::sync::broadcast::channel::<HpoSseEvent>(512);
    {
        let mut channels = state.hpo_channels.write().await;
        channels.insert(study_id, Arc::new(HpoBroadcast { sender: tx.clone() }));
    }

    // Dedicated DB-writer task: receives each trial event over an mpsc channel
    // and persists it immediately. This keeps the HPO workers non-blocking
    // (the send is cheap) while every trial lands in the DB as soon as it
    // completes, so the detail page can read progress live from the DB.
    let (db_tx, mut db_rx) = tokio::sync::mpsc::unbounded_channel::<HpoTrialEvent>();
    let db_state = Arc::clone(&state);
    let writer_sid = study_id;
    let writer_handle = tokio::spawn(async move {
        let mut written: usize = 0;
        while let Some(event) = db_rx.recv().await {
            let metrics_payload = serde_json::json!({
                "val_loss": f64_or_null(event.val_loss),
                "ok_at_1pct": f64_or_null(event.ok_at_1pct),
                "max_error": f64_or_null(event.max_error),
                "param_count": event.param_count,
                "training_time_secs": f64_or_null(event.training_time_secs),
            });
            let metrics = serde_json::to_string(&metrics_payload).unwrap_or_else(|e| {
                eprintln!(
                    "[hpo:{writer_sid}] metrics_json serialization failed for trial {}: {e}",
                    event.trial_number,
                );
                "{}".to_string()
            });
            // `[OK@1%, param_count, max_error]` — drives the Pareto plot.
            let objectives_payload = serde_json::json!([
                f64_or_null(event.ok_at_1pct),
                event.param_count as f64,
                f64_or_null(event.max_error),
            ]);
            let objectives = serde_json::to_string(&objectives_payload).unwrap_or_else(|e| {
                eprintln!(
                    "[hpo:{writer_sid}] objectives_json serialization failed for trial {}: {e}",
                    event.trial_number,
                );
                "[]".to_string()
            });

            // Scope each DB lock to one rusqlite call so concurrent
            // HPOs can interleave writes (SQLite WAL handles it).
            let trial_num = event.trial_number;
            {
                let conn = db_state.db.lock().await;
                if let Err(e) = db::insert_hpo_trial(
                    &conn, writer_sid, event.trial_number, &event.status,
                    &event.model_type, &event.params_json, &metrics, &objectives,
                    event.is_feasible,
                ) {
                    eprintln!(
                        "[hpo:{writer_sid}] insert_hpo_trial({trial_num}) failed: {e}",
                    );
                    continue;
                }
            }
            written += 1;
            {
                let conn = db_state.db.lock().await;
                if let Err(e) = db::update_hpo_progress(&conn, writer_sid, written) {
                    eprintln!(
                        "[hpo:{writer_sid}] update_hpo_progress({trial_num}, \
                         n_completed={written}) failed: {e}",
                    );
                }
            }
        }
    });
    let writer_handle = std::sync::Mutex::new(Some(writer_handle));

    // Stash copies for the final template render — the spawn_blocking
    // closure takes ownership of `study_name`, `model_type`, `config`.
    let template_study_name = study_name.clone();
    let template_model_type = model_type.clone();

    // Spawn HPO task
    let state_clone = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        let tx_clone = tx.clone();
        let sid = study_id;

        let trial_count = std::sync::atomic::AtomicUsize::new(0);

        let result = run_hpo_with_callback(&config, |event: HpoTrialEvent| {
            let n = trial_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

            let _ = tx_clone.send(HpoSseEvent::TrialComplete {
                trial: n,
                total: total_trials,
                model_type: event.model_type.clone(),
                status: event.status.clone(),
                val_loss: event.val_loss,
                ok_at_1pct: event.ok_at_1pct,
                hidden_size: event.hidden_size,
                param_count: event.param_count,
                max_error: event.max_error,
                is_feasible: event.is_feasible,
            });

            // Send to dedicated DB writer — non-blocking.
            let _ = db_tx.send(event);
        });

        // Close the DB-writer channel so it can flush and exit after the final
        // trial is persisted.
        drop(db_tx);

        // Wait for the DB writer to drain every queued trial before we mark
        // the study complete. This guarantees the `results_json` / Pareto
        // marks don't race with in-flight per-trial inserts.
        match tokio::runtime::Handle::try_current() {
            Ok(rt) => {
                let handle = writer_handle.lock().ok().and_then(|mut g| g.take());
                if let Some(h) = handle {
                    rt.block_on(async {
                        let _ = h.await;
                    });
                }
            }
            Err(e) => {
                // Without a runtime handle we cannot drain the writer
                // nor persist the final DB update. The study would
                // permanently show as 'running' — surface loudly.
                eprintln!(
                    "[hpo:{sid}] FATAL: Handle::try_current() failed: {e}. \
                     Study will remain 'running' in the dashboard."
                );
                return;
            }
        }

        match result {
            Ok(run_result) => {
                let _ = tx.send(HpoSseEvent::StudyComplete { study_id: sid });

                let results_json =
                    serde_json::to_string(&run_result.summaries).unwrap_or_default();
                let n_feasible: usize =
                    run_result.summaries.iter().map(|s| s.feasible_trials).sum();
                let pareto_size: usize =
                    run_result.summaries.iter().map(|s| s.pareto_front_size).sum();
                let time = run_result.total_time_secs;
                // Collect Pareto (model_type, trial_number) from each
                // summary. Trial numbers collide across real / complex
                // studies (both seeded identically), so the mark must be
                // scoped by model type too.
                let pareto_marks: Vec<(String, usize)> = run_result
                    .summaries
                    .iter()
                    .flat_map(|s| {
                        let mt = s.model_type.clone();
                        s.best_configs.iter().map(move |c| (mt.clone(), c.trial))
                    })
                    .collect();

                match tokio::runtime::Handle::try_current() {
                    Ok(rt) => {
                        rt.block_on(async {
                            let conn = state_clone.db.lock().await;
                            if let Err(e) = db::complete_hpo_study(
                                &conn, sid, n_feasible, pareto_size, time, &results_json,
                            ) {
                                eprintln!(
                                    "[hpo:{sid}] complete_hpo_study failed: {e} \
                                     — study may remain in 'running' state"
                                );
                            }
                            // Mark Pareto trials in DB, scoped per model type.
                            for (model_type, trial_num) in &pareto_marks {
                                if let Err(e) =
                                    db::mark_trial_pareto(&conn, sid, model_type, *trial_num)
                                {
                                    eprintln!(
                                        "[hpo:{sid}] mark_trial_pareto({model_type}, \
                                         {trial_num}) failed: {e}"
                                    );
                                }
                            }
                        });
                    }
                    Err(e) => {
                        eprintln!(
                            "[hpo:{sid}] FATAL: Handle::try_current() for DB write failed: {e}"
                        );
                    }
                }
            }
            Err(e) => {
                let msg = e.to_string();
                let _ = tx.send(HpoSseEvent::StudyError { message: msg.clone() });
                if let Ok(rt) = tokio::runtime::Handle::try_current() {
                    rt.block_on(async {
                        let conn = state_clone.db.lock().await;
                        if let Err(err) = db::fail_hpo_study(&conn, sid, &msg) {
                            eprintln!("[hpo:{sid}] fail_hpo_study DB update failed: {err}");
                        }
                    });
                }
            }
        }

        // Channel cleanup runs on the async runtime so the blocking
        // thread is freed immediately.
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            let mut channels = state_clone.hpo_channels.write().await;
            channels.remove(&sid);
        });
    });

    render(HpoProgressTemplate {
        study_id,
        name: template_study_name,
        n_trials: total_trials,
        model_type: template_model_type,
    })
}

pub async fn sse_stream(
    State(state): State<Arc<AppState>>,
    Path(study_id): Path<i64>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = {
        let channels = state.hpo_channels.read().await;
        channels.get(&study_id).map(|b| b.sender.subscribe())
    };

    let stream = match rx {
        Some(rx) => {
            let s = BroadcastStream::new(rx).filter_map(|result| {
                result.ok().map(|event| {
                    let event_type = match &event {
                        HpoSseEvent::TrialComplete { .. } => "trial",
                        HpoSseEvent::StudyComplete { .. } => "complete",
                        HpoSseEvent::StudyError { .. } => "hpo_error",
                    };
                    Ok(Event::default()
                        .event(event_type)
                        .data(serde_json::to_string(&event).unwrap_or_default()))
                })
            });
            Box::pin(s) as std::pin::Pin<Box<dyn tokio_stream::Stream<Item = Result<Event, Infallible>> + Send>>
        }
        None => {
            let done = tokio_stream::once(Ok(
                Event::default()
                    .event("complete")
                    .data(format!(r#"{{"type":"StudyComplete","study_id":{study_id}}}"#))
            ));
            Box::pin(done)
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

pub async fn detail(
    State(state): State<Arc<AppState>>,
    Path(study_id): Path<i64>,
) -> WebResult<Html<String>> {
    let (study, trials, importance_groups) = {
        let conn = state.db.lock().await;
        let study = db::get_hpo_study(&conn, study_id)
            .map_err(|_| WebError::NotFound(format!("Study {study_id} not found")))?;
        let trials = db::list_hpo_trials(&conn, study_id)?;
        // Native per-model-type param importance from `results_json`
        // (computed via the optimizer crate's `Study::fanova()`).
        let importance_groups = read_importance_groups(&conn, study_id);
        (study, trials, importance_groups)
    };

    let pareto_entries = extract_pareto_entries(&trials);
    let best_trial_summary = if let Some(best) = pareto_entries.first() {
        format!("Trial #{} — OK@1%={}, H={}, Loss={}",
            best.trial, best.ok_at_1pct, best.hidden_size, best.val_loss)
    } else {
        "No feasible trials".into()
    };

    let study_config_json = study.config_json.clone();
    let dataset_id = study.dataset_id;
    render(HpoDetailTemplate { study, trials, importance_groups, pareto_entries, best_trial_summary, study_config_json, dataset_id })
}

/// Read per-objective parameter importances from the stored
/// `results_json`, keeping them separated by (model_type, objective).
/// The values are the raw output of the optimizer crate's
/// `Study::fanova()` (main-effect variance ratios) — one companion
/// study per NSGA-III objective, see
/// [`sparam_hpo::study::ImportanceStudies`]. No extra aggregation or
/// smoothing here.
///
/// Returned shape: one `ImportanceGroup` per model type ("real",
/// "complex"), each containing up to 3 per-objective sub-lists
/// ("ok_at_1pct", "param_count", "max_error"), each sorted by
/// descending importance and truncated to the top 15 parameters.
///
/// Backward compat: if the stored `results_json` lacks
/// `param_importance_per_objective` (studies completed before the
/// per-objective split), we synthesize a single "overall" pseudo-
/// objective from the legacy `param_importance` field so old studies
/// still render.
fn read_importance_groups(
    conn: &rusqlite::Connection,
    study_id: i64,
) -> Vec<ImportanceGroup> {
    let results_json: Option<String> = conn
        .query_row(
            "SELECT results_json FROM hpo_studies WHERE id = ?1",
            rusqlite::params![study_id],
            |row| row.get(0),
        )
        .ok()
        .flatten();

    let Some(json) = results_json else { return Vec::new(); };
    let summaries: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap_or_default();

    // Preferred display order + labels for the three NSGA-III
    // objectives. Matches `sparam_hpo::study::OBJECTIVE_NAMES` so the
    // template ordering is stable regardless of HashMap iteration.
    let objective_order: [(&str, &str); 3] = [
        ("ok_at_1pct", "OK@1%"),
        ("param_count", "Param count"),
        ("max_error", "Max error"),
    ];

    let mut groups: Vec<ImportanceGroup> = Vec::new();
    for summary in &summaries {
        let model_type = summary
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Per-objective interaction shares — the fANOVA residual
        // (`1 - Σ main_effects`). Absent in legacy studies + in
        // Spearman-fallback objectives. The template uses `Option`
        // to distinguish "no data" (hide the bar) from "zero"
        // (sub-threshold; hide the bar too).
        let interactions_map = summary.get("param_interactions_per_objective");

        // New path: `param_importance_per_objective` is a map from
        // objective name to a Vec<[name, score]>. Walk it in the
        // canonical order so the UI shows objectives consistently.
        let mut objective_sections: Vec<ObjectiveImportance> = Vec::new();
        if let Some(map) = summary
            .get("param_importance_per_objective")
            .and_then(|v| v.as_object())
        {
            for (key, label) in &objective_order {
                let Some(arr) = map.get(*key).and_then(|v| v.as_array()) else {
                    continue;
                };
                if let Some(entries) = build_importance_entries(arr, &model_type) {
                    // Look up the matching interaction share. Threshold
                    // `> 1e-4` matches the workflow's log threshold so
                    // chart and log agree on when to show the bar.
                    let interactions_pct = interactions_map
                        .and_then(|v| v.as_object())
                        .and_then(|m| m.get(*key))
                        .and_then(|v| v.as_f64())
                        .filter(|v| v.is_finite() && *v > 1e-4)
                        .map(|v| format!("{:.2}", v * 100.0));
                    objective_sections.push(ObjectiveImportance {
                        objective: (*key).to_string(),
                        objective_label: (*label).to_string(),
                        entries,
                        interactions_pct,
                    });
                }
            }
        }

        // Legacy fallback: studies completed before per-objective
        // importance existed only have `param_importance`. Treat it
        // as one generic "overall" section so the page still shows
        // something sensible.
        if objective_sections.is_empty() {
            if let Some(arr) = summary
                .get("param_importance")
                .and_then(|v| v.as_array())
            {
                if let Some(entries) = build_importance_entries(arr, &model_type) {
                    objective_sections.push(ObjectiveImportance {
                        objective: "overall".to_string(),
                        objective_label: "Overall (legacy study)".to_string(),
                        entries,
                        interactions_pct: None,
                    });
                }
            }
        }

        if !objective_sections.is_empty() {
            groups.push(ImportanceGroup {
                model_type,
                objectives: objective_sections,
            });
        }
    }

    groups
}

/// Parse one `Vec<[name, score]>` JSON array into a sorted
/// `Vec<ImportanceEntry>`. Returns `None` if the parsed list is
/// empty — the caller omits empty sub-groups from the render.
fn build_importance_entries(
    arr: &[serde_json::Value],
    model_type: &str,
) -> Option<Vec<ImportanceEntry>> {
    let mut pairs: Vec<(String, f64)> = arr
        .iter()
        .filter_map(|entry| {
            let pair = entry.as_array()?;
            let name = pair.first().and_then(|v| v.as_str())?.to_string();
            let score = pair.get(1).and_then(|v| v.as_f64())?;
            // Drop NaN / ±Infinity importance scores: they'd sort to
            // arbitrary positions (`partial_cmp` on NaN returns
            // `None`, previously defaulted to `Equal`) and their
            // percentage bars would render as `NaN%`. A finite-only
            // filter keeps the chart readable and hides parameters
            // that fANOVA produced no signal for.
            score.is_finite().then_some((name, score))
        })
        .collect();

    if pairs.is_empty() {
        return None;
    }
    // Safe to unwrap `partial_cmp` here: every score is finite (see
    // filter above).
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max_val = pairs.first().map(|(_, v)| *v).unwrap_or(1.0).max(1e-10);

    let entries: Vec<ImportanceEntry> = pairs
        .into_iter()
        .take(15)
        .map(|(name, score)| ImportanceEntry {
            name: prettify_param_name(&name),
            importance: format!("{:.2}%", score * 100.0),
            bar_width: format!("{:.0}%", score / max_val * 100.0),
            model_type: model_type.to_string(),
        })
        .collect();

    if entries.is_empty() { None } else { Some(entries) }
}

fn prettify_param_name(name: &str) -> String {
    match name {
        "hidden_size" => "Hidden Size".into(),
        "lr" => "Learning Rate".into(),
        "activation" => "Activation".into(),
        "optimizer" => "Optimizer".into(),
        "train_batch_size" => "Batch Size".into(),
        "weight_decay" => "Weight Decay".into(),
        "dropout_p" => "Dropout".into(),
        "norm" => "Normalization".into(),
        "loss" => "Loss Function".into(),
        "grad_clip_norm" => "Grad Clipping".into(),
        "input_noise_std" => "Input Noise".into(),
        "scheduler" => "Scheduler".into(),
        other => other.replace('_', " "),
    }
}

fn extract_pareto_entries(trials: &[db::HpoTrialRow]) -> Vec<ParetoEntry> {
    // Use the per-model-type `is_pareto` flag stamped into the DB by
    // `mark_trial_pareto` after each HPO run completes. This already reflects
    // NSGA-II's per-study Pareto front (so for a "both" study it's the union
    // of real+complex fronts, not a pooled re-computation that would drop
    // entries dominated across model types).
    let mut pareto: Vec<ParetoEntry> = trials.iter()
        .filter(|t| t.status == "completed" && t.is_pareto)
        .filter_map(|t| {
            let metrics: serde_json::Value = t.metrics_json.as_deref()
                .and_then(|m| serde_json::from_str(m).ok())?;
            let params: serde_json::Value = serde_json::from_str(&t.params_json).ok()?;
            let ok = metrics.get("ok_at_1pct")?.as_f64()?;
            let val_loss = metrics.get("val_loss")?.as_f64()?;
            let max_error = metrics.get("max_error")?.as_f64().unwrap_or(f64::INFINITY);
            // `param_count` is what NSGA-II actually sees as the
            // complexity objective, so show it explicitly rather than
            // letting the reader infer it from `hidden_size`.
            let param_count = metrics
                .get("param_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let hidden = params.get("hidden_size")?.as_i64().unwrap_or(0);
            let lr = params.get("lr")?.as_f64().unwrap_or(0.0);
            let act = params.get("activation")?.as_str().unwrap_or("-").to_string();
            let opt = params.get("optimizer")?.as_str().unwrap_or("-").to_string();
            Some(ParetoEntry {
                trial: t.trial_number,
                model_type: t.model_type.clone(),
                ok_at_1pct: format!("{ok:.1}"),
                param_count: param_count.to_string(),
                hidden_size: hidden.to_string(),
                val_loss: format!("{val_loss:.6}"),
                max_error: format!("{max_error:.1}"),
                lr: format!("{lr:.2e}"),
                activation: act,
                optimizer: opt,
            })
        })
        .collect();

    pareto.sort_by(|a, b| {
        a.model_type.cmp(&b.model_type).then(a.trial.cmp(&b.trial))
    });
    pareto
}

pub async fn delete(
    State(state): State<Arc<AppState>>,
    Path(study_id): Path<i64>,
) -> WebResult<Response> {
    let conn = state.db.lock().await;
    db::delete_hpo_study(&conn, study_id)?;
    Ok((StatusCode::OK, [("HX-Redirect", "/hpo")], "").into_response())
}


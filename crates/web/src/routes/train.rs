//! Training routes: full configuration form, launch, SSE stream, results.
//!
//! The training form schema (`TrainForm`) and the form → `TrainConfig`
//! mapping live in `sparam_app::workflows::train_form` so that adding a
//! field to `TrainConfig` only requires updating one place. This
//! route is pure HTTP glue — DB reads, `form.build()`, task spawn,
//! template render.

use std::convert::Infallible;
use std::sync::Arc;

use askama::Template;
use axum::extract::{Path, State};
use axum::response::Html;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Form;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use sparam_app::workflows::train_form::{TrainForm, TrainSplits};

use crate::db::{self, DatasetRow, TrainingRunRow};
use crate::error::{WebError, WebResult, render};
use crate::state::{AppState, RunBroadcast, RunId, SseEvent};
use crate::tasks;

#[derive(Template)]
#[template(path = "train_form.html")]
pub struct TrainFormTemplate {
    pub datasets: Vec<DatasetRow>,
}

#[derive(Template)]
#[template(path = "train_progress.html")]
pub struct TrainProgressTemplate {
    pub run_id: RunId,
    pub config_summary: String,
}

#[derive(Template)]
#[template(path = "train_result.html")]
pub struct TrainResultTemplate {
    pub run: TrainingRunRow,
}

/// Detail for a training run.
pub struct PlotFile {
    pub title: String,
    pub url: String,
}

#[derive(Template)]
#[template(path = "train_detail.html")]
pub struct TrainDetailTemplate {
    pub run: TrainingRunRow,
    pub config_json: String,
    pub metrics_json: String,
    pub history_json: String,
    pub param_count: String,
    pub plots: Vec<PlotFile>,
}

pub async fn form(
    State(state): State<Arc<AppState>>,
) -> WebResult<Html<String>> {
    let datasets = {
        let conn = state.db.lock().await;
        db::list_datasets(&conn)?
    };
    render(TrainFormTemplate { datasets })
}

pub async fn launch(
    State(state): State<Arc<AppState>>,
    Form(form): Form<TrainForm>,
) -> WebResult<Html<String>> {
    // Load dataset samples from DB (no filesystem access).
    let dataset_id = form.dataset_id;
    let (train_samples, val_samples, test_samples) = {
        let conn = state.db.lock().await;
        db::get_dataset(&conn, dataset_id)
            .map_err(|_| WebError::NotFound("Dataset not found".into()))?;
        db::load_dataset_splits(&conn, dataset_id)?
    };

    // Delegate the full form → workflow-config mapping (model config
    // parse, training hyperparams, JSON round-trip for DB persistence)
    // to the shared `TrainForm::build_stacked` — single source of
    // truth.  When the form's stack toggles are at defaults the
    // resulting `StackedTrainConfig` is a passthrough and downstream
    // training behaves identically to the legacy single-model path.
    let built = form
        .build_stacked(TrainSplits {
            train: train_samples.into(),
            val: val_samples.into(),
            test: test_samples.into(),
        })
        .map_err(|e| WebError::BadRequest(e.0))?;

    let stack_summary = if built.config.is_passthrough() {
        String::new()
    } else {
        let mut bits: Vec<String> = Vec::new();
        if built.config.train_overlay_density > 0 {
            bits.push(format!(
                "overlay {}×{}",
                built.config.train_overlay_density,
                built.config.train_overlay_density,
            ));
        }
        if built.config.ensemble_size > 1 {
            bits.push(format!("{}-median ensemble", built.config.ensemble_size));
        }
        if built.config.refine_steps > 0 {
            bits.push(format!("{}-step NRW refine", built.config.refine_steps));
        }
        format!(" | M10: {}", bits.join(" + "))
    };
    let config_summary = format!(
        "{} | {} | lr={} | {} | {} epochs{}",
        built.model_name,
        built.config.base.optimizer,
        built.config.base.lr,
        built.config.base.loss,
        built.config.base.max_epochs,
        stack_summary,
    );

    // Allocate the training_run DB row first so we know the run_id
    // before training — it's the key the weights-writer uses to target
    // the correct row when persisting the final safetensors BLOB.
    let run_id = {
        let conn = state.db.lock().await;
        db::insert_training_run(
            &conn,
            dataset_id,
            &built.model_name,
            &built.config_json,
            "", // output_dir is legacy — weights now live in the DB
        )?
    };

    let (tx, _) = tokio::sync::broadcast::channel::<SseEvent>(256);
    {
        let mut channels = state.run_channels.write().await;
        channels.insert(run_id, Arc::new(RunBroadcast { sender: tx.clone() }));
    }

    tasks::spawn_training(Arc::clone(&state), run_id, built.config, tx);

    render(TrainProgressTemplate { run_id, config_summary })
}

pub async fn sse_stream(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<RunId>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = {
        let channels = state.run_channels.read().await;
        channels.get(&run_id).map(|b| b.sender.subscribe())
    };

    let stream = match rx {
        Some(rx) => {
            let s = BroadcastStream::new(rx).filter_map(|result| {
                result.ok().map(|event| {
                    let event_type = match &event {
                        SseEvent::Epoch { .. } => "epoch",
                        SseEvent::Improvement { .. } => "improvement",
                        SseEvent::Stage { .. } => "stage",
                        SseEvent::Complete { .. } => "complete",
                        SseEvent::Error { .. } => "training_error",
                    };
                    Ok(Event::default()
                        .event(event_type)
                        .data(serde_json::to_string(&event).unwrap_or_default()))
                })
            });
            Box::pin(s) as std::pin::Pin<Box<dyn tokio_stream::Stream<Item = Result<Event, Infallible>> + Send>>
        }
        None => {
            // Channel gone — training already finished. Check DB for status.
            let event = {
                let conn = state.db.lock().await;
                match db::get_training_run(&conn, run_id) {
                    Ok(run) if run.status == "failed" => {
                        // Extract error message from metrics_json.
                        let msg = serde_json::from_str::<serde_json::Value>(
                            run.completed_at.as_deref().unwrap_or(""),
                        )
                        .ok()
                        .and_then(|v| v.get("error")?.as_str().map(String::from))
                        .or_else(|| {
                            // metrics_json stores the error for failed runs
                            let conn2 = &conn;
                            conn2
                                .query_row(
                                    "SELECT metrics_json FROM training_runs WHERE id = ?1",
                                    rusqlite::params![run_id],
                                    |row| row.get::<_, Option<String>>(0),
                                )
                                .ok()
                                .flatten()
                                .and_then(|mj| {
                                    serde_json::from_str::<serde_json::Value>(&mj)
                                        .ok()?
                                        .get("error")?
                                        .as_str()
                                        .map(String::from)
                                })
                        })
                        .unwrap_or_else(|| "Training failed (unknown error)".into());
                        Event::default()
                            .event("training_error")
                            .data(serde_json::to_string(&SseEvent::Error { message: msg }).unwrap_or_default())
                    }
                    _ => {
                        Event::default()
                            .event("complete")
                            .data(format!(r#"{{"type":"Complete","run_id":{run_id}}}"#))
                    }
                }
            };
            Box::pin(tokio_stream::once(Ok(event)))
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

pub async fn result(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<RunId>,
) -> WebResult<Html<String>> {
    let run = {
        let conn = state.db.lock().await;
        db::get_training_run(&conn, run_id)
            .map_err(|_| WebError::NotFound(format!("Run {run_id} not found")))?
    };
    render(TrainResultTemplate { run })
}

pub async fn detail(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<RunId>,
) -> WebResult<Html<String>> {
    let conn = state.db.lock().await;
    let run = db::get_training_run(&conn, run_id)
        .map_err(|_| WebError::NotFound(format!("Run {run_id} not found")))?;
    let config_json = db::get_training_run_config(&conn, run_id).unwrap_or_else(|_| "{}".into());
    let metrics_json = db::get_training_run_metrics(&conn, run_id)
        .ok()
        .flatten()
        .unwrap_or_else(|| "{}".into());
    let history_json = db::get_training_run_history(&conn, run_id)
        .ok()
        .flatten()
        .unwrap_or_else(|| "{}".into());
    drop(conn);

    // Extract parameter count from config.
    let param_count = serde_json::from_str::<serde_json::Value>(&config_json)
        .ok()
        .and_then(|v| v.get("parameter_count")?.as_u64())
        .map(|n| format!("{n}"))
        .unwrap_or_else(|| "-".into());

    // No PNG plot files anymore — charts are rendered client-side from JSON.
    let plots: Vec<PlotFile> = Vec::new();

    // Pretty-print JSON for display.
    let config_pretty = serde_json::from_str::<serde_json::Value>(&config_json)
        .and_then(|v| serde_json::to_string_pretty(&v))
        .unwrap_or(config_json);
    let metrics_pretty = serde_json::from_str::<serde_json::Value>(&metrics_json)
        .and_then(|v| serde_json::to_string_pretty(&v))
        .unwrap_or(metrics_json);

    render(TrainDetailTemplate {
        run,
        config_json: config_pretty,
        metrics_json: metrics_pretty,
        history_json,
        param_count,
        plots,
    })
}

// TrainForm deserialization tests live alongside the form definition
// in `sparam_app::workflows::train_form::tests`.

//! Background task management for long-running operations.

use std::sync::Arc;

use tokio::sync::broadcast;

use sparam_app::workflows::config::StackedTrainConfig;
use sparam_app::workflows::stacked_training::run_stacked_training_with_callback;
use sparam_app::workflows::train::{TrainRunResult, run_training_with_callback};

use crate::db;
use crate::sse::SseCallback;
use crate::state::{AppState, RunId, SseEvent};

/// Spawn a training run on a blocking thread, streaming progress via
/// SSE. Passthrough config → per-epoch SSE; stacked pipeline → per-member
/// epoch events tagged with `(member, ensemble_size)` plus stage
/// events for the post-training pipeline.
pub fn spawn_training(
    state: Arc<AppState>,
    run_id: RunId,
    config: StackedTrainConfig,
    tx: broadcast::Sender<SseEvent>,
) -> tokio::task::JoinHandle<()> {
    let state_clone = Arc::clone(&state);
    let tx_clone = tx.clone();

    tokio::task::spawn_blocking(move || {
        let stack_active = !config.is_passthrough();
        let mut stacked_metrics: Option<sparam_app::workflows::config::TrainMetricsSummary> = None;
        // Non-primary ensemble members; the first member's bytes go to
        // `weights_blob`, the rest to `ensemble_weights_json` for
        // /evaluate to replay the median.
        let mut extra_members: Vec<Vec<u8>> = Vec::new();
        let result: Result<TrainRunResult, candle_core::Error> = if stack_active {
            let tx_for_cb = tx_clone.clone();
            let tx_for_stage = tx_clone.clone();
            let stack_result = run_stacked_training_with_callback(
                &config,
                move |member, ensemble_size| {
                    Some(SseCallback::new(tx_for_cb.clone())
                        .with_member(member, ensemble_size))
                },
                move |msg| {
                    let _ = tx_for_stage.send(SseEvent::Stage { message: msg });
                },
            );
            match stack_result {
                Ok(stacked) => {
                    stacked_metrics = stacked.stacked_metrics;
                    if let Some(all_members) = stacked.ensemble_weights {
                        // skip(1): primary's bytes already go to `weights_blob`.
                        extra_members = all_members.into_iter().skip(1).collect();
                    }
                    Ok(stacked.primary)
                }
                Err(e) => Err(e),
            }
        } else {
            let callback = SseCallback::new(tx_clone.clone());
            run_training_with_callback(&config.base, callback)
        };

        match result {
            Ok(train_result) => {
                // Persist to SQLite BEFORE emitting Complete so the
                // client can't race a downstream Evaluate click against
                // an unflushed weights blob.
                let metrics_json =
                    serde_json::to_string(&train_result.metrics).unwrap_or_default();
                let history_json = serde_json::to_string(&serde_json::json!({
                    "train_losses": train_result.history.train_losses,
                    "val_losses": train_result.history.val_losses,
                    "learning_rates": train_result.history.learning_rates,
                })).unwrap_or_default();

                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    // Per-call lock scoping: each DB write takes the
                    // mutex briefly, then drops it so concurrent web
                    // requests aren't blocked across the whole sequence.
                    let weights_res = {
                        let conn = state_clone.db.lock().await;
                        db::update_training_run_weights(&conn, run_id, &train_result.weights)
                    };
                    if let Err(err) = weights_res {
                        eprintln!("failed to persist weights for run {run_id}: {err}");
                        let _ = tx_clone.send(SseEvent::Error {
                            message: format!(
                                "Training succeeded but weights could not be saved: {err}"
                            ),
                        });
                        return;
                    }

                    if !extra_members.is_empty() {
                        let conn = state_clone.db.lock().await;
                        if let Err(err) = db::update_training_run_ensemble_weights(
                            &conn, run_id, &extra_members,
                        ) {
                            eprintln!("failed to persist ensemble weights for run {run_id}: {err}");
                        }
                    }

                    // Enrich stored config with parameter_count + the M10
                    // stacked metrics so the train detail page can show
                    // both single-model and post-ensemble views.
                    let existing = {
                        let conn = state_clone.db.lock().await;
                        db::get_training_run_config(&conn, run_id).unwrap_or_default()
                    };
                    let mut cfg: serde_json::Value = serde_json::from_str(&existing)
                        .unwrap_or(serde_json::json!({}));
                    if let Some(obj) = cfg.as_object_mut() {
                        obj.insert(
                            "parameter_count".into(),
                            serde_json::json!(train_result.parameter_count),
                        );
                        if let Some(ref sm) = stacked_metrics {
                            obj.insert(
                                "m10_stacked_metrics".into(),
                                serde_json::to_value(sm).unwrap_or(serde_json::json!({})),
                            );
                        }
                    }
                    let enriched_config = serde_json::to_string(&cfg).unwrap_or(existing);

                    {
                        let conn = state_clone.db.lock().await;
                        if let Err(err) = db::update_training_run_artifacts(
                            &conn,
                            run_id,
                            Some(&metrics_json),
                            Some(&history_json),
                            Some(&enriched_config),
                        ) {
                            eprintln!("failed to persist artifacts for run {run_id}: {err}");
                        }
                        if let Err(err) = db::complete_training_run(
                            &conn,
                            run_id,
                            &metrics_json,
                            train_result.metrics.best_epoch,
                            train_result.metrics.final_epoch,
                            train_result.metrics.training_time_secs,
                        ) {
                            eprintln!("failed to mark run {run_id} completed: {err}");
                        }
                    }

                    let _ = tx_clone.send(SseEvent::Complete { run_id });
                });
            }
            Err(e) => {
                let msg = e.to_string();
                let _ = tx_clone.send(SseEvent::Error {
                    message: msg.clone(),
                });

                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    let conn = state_clone.db.lock().await;
                    if let Err(err) = db::fail_training_run(&conn, run_id, &msg) {
                        eprintln!("failed to mark run {run_id} failed: {err}");
                    }
                });
            }
        }

        // Channel cleanup runs on the async runtime so the blocking
        // thread is freed immediately for the next training run.
        let state_cleanup = Arc::clone(&state_clone);
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let mut channels = state_cleanup.run_channels.write().await;
            channels.remove(&run_id);
        });
    })
}

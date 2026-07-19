//! End-to-end training workflow.

use std::path::PathBuf;

use candle_core::{DType, Result};

use super::{
    config::{TrainConfig, TrainMetricsSummary},
    internal::{
        data_prep::{build_loader, parse_batch_size, prepare_splits_from_samples},
        evaluation::evaluate_model_predictions,
        model_factory::build_model,
        training::packed_loss_fn,
    },
};
use sparam_training::trainer::TrainingHistory;
use sparam_training::trainer::{
    EarlyStoppingConfig, NoopTrainingCallback, TrainerBuilder, TrainingCallback,
};
use sparam_training::losses::MlpLoss;
use sparam_training::optimizers::OptimizerConfig;
use sparam_training::checkpoint::{
    load_model_checkpoint, save_model_checkpoint, save_model_checkpoint_bytes,
};
use sparam_core::io::ensure_dir;
use sparam_training::logger::{LogMessage, LogSender, LogWorker};

/// Result returned from a completed training run.
/// The caller is responsible for persisting `metrics`, `history`, and
/// `weights` (e.g. SQLite BLOB for the web UI, `.safetensors` file for
/// the CLI).
#[derive(Debug, Clone)]
pub struct TrainRunResult {
    pub arch_description: String,
    pub parameter_count: usize,
    pub metrics: TrainMetricsSummary,
    pub history: TrainingHistory,
    /// On-disk safetensors path, populated only when
    /// [`TrainConfig::checkpoint_path`] was `Some`.
    pub checkpoint_path: Option<PathBuf>,
    /// Serialized safetensors bytes of the final model weights.
    /// Always produced — the web layer streams these to a SQLite BLOB
    /// instead of the filesystem, the CLI writes them to a file.
    pub weights: Vec<u8>,
}

/// Run the full training workflow with the default (no-op) callback.
///
/// This is the legacy single-model entry point.  For the stacked pipeline —
/// train-overlay synthesis, median ensembling, and inference-time NRW
/// refinement, all opt-in and stackable — use
/// [`crate::workflows::stacked_training::run_stacked_training`] with a
/// [`crate::workflows::config::StackedTrainConfig`].  When all the stack toggles
/// are at their defaults the stacked entry point degenerates to this
/// function, so existing callers are not affected.
pub fn run_training(config: &TrainConfig) -> Result<TrainRunResult> {
    run_training_with_callback(config, NoopTrainingCallback)
}

/// Run the full training workflow, forwarding per-epoch metrics to `callback`.
///
/// The callback receives [`EpochMetrics`][sparam_training::trainer::EpochMetrics]
/// events on every epoch, and improvement / early-stop notifications. This is
/// the extension point for web-UI SSE streaming.
pub fn run_training_with_callback<C: TrainingCallback>(
    config: &TrainConfig,
    callback: C,
) -> Result<TrainRunResult> {
    sparam_core::rng::set_global_seed(config.seed);

    // Ensure the parent directory of the checkpoint exists (only when
    // the caller asked for an on-disk sidecar file — web UI doesn't).
    if let Some(parent) = config
        .checkpoint_path
        .as_deref()
        .and_then(|p| p.parent())
    {
        ensure_dir(parent)?;
    }

    let mut _worker: Option<LogWorker> = None;
    let log: LogSender = if config.log_progress {
        let (s, w) = LogSender::new();
        _worker = Some(w);
        // Tag the sender with a train-run-specific context so two
        // concurrent trainings produce distinguishable stderr output.
        // Arch + seed is usually unique enough; callers running
        // identical arch-seed pairs concurrently can disambiguate via
        // the DB's `run_id` column.
        s.with_context(format!(
            "train:{}@seed-{}",
            config.model.arch_description(),
            config.seed,
        ))
    } else {
        LogSender::null()
    };

    log.send(LogMessage::Info(format!(
        "  Train: {} samples, Val: {} samples, Test: {} samples",
        config.train_samples.len(),
        config.val_samples.len(),
        config.test_samples.len()
    )));

    let prepared = prepare_splits_from_samples(
        &config.train_samples,
        &config.val_samples,
        &config.test_samples,
        config.model.model_type(),
        DType::F64,
        false,
    )?;

    let (model, mut varmap) = build_model(&config.model, prepared.dtype())?;
    let arch_description = config.model.arch_description();
    let parameter_count = model.parameter_count();

    // Log the configuration table first so subsequent per-init
    // diagnostics (`emit_varmap_hash`, resume notice) appear AFTER
    // their context instead of before it.
    log.send(LogMessage::Info(String::new()));
    log.send(LogMessage::Info("Training Configuration:".into()));
    log.send(LogMessage::MetricRow { label: "Model".into(), value: arch_description.clone() });
    log.send(LogMessage::MetricRow { label: "Parameters".into(), value: parameter_count.to_string() });
    log.send(LogMessage::MetricRow { label: "Optimizer".into(), value: format!("{} (lr={}, wd={})", config.optimizer, config.lr, config.weight_decay) });
    log.send(LogMessage::MetricRow { label: "Scheduler".into(), value: config.scheduler.clone() });
    log.send(LogMessage::MetricRow { label: "Epochs".into(), value: format!("{} (patience={})", config.max_epochs, config.patience) });
    log.send(LogMessage::MetricRow { label: "Batch size".into(), value: config.batch_size.clone() });
    log.send(LogMessage::MetricRow { label: "Loss".into(), value: config.loss.clone() });
    log.send(LogMessage::Info(String::new()));

    sparam_core::determinism::deterministic_reinit_varmap(&varmap, config.seed)?;
    sparam_core::determinism::emit_varmap_hash(
        format_args!("train post-reinit"),
        &varmap,
    );

    if let Some(checkpoint_path) = &config.checkpoint {
        load_model_checkpoint(&mut varmap, checkpoint_path)?;
        log.send(LogMessage::Info(format!(
            "Resumed weights from {}",
            checkpoint_path.display()
        )));
        // Honest debug trail: the `post-reinit` hash above is stale
        // once resume overwrites the VarMap. Emit the real starting
        // hash so SPARAM_DEBUG_HASH_VARMAP readers see what the
        // training loop actually consumes.
        sparam_core::determinism::emit_varmap_hash(
            format_args!("train post-resume"),
            &varmap,
        );
    }

    // Regularization applies to both Real and Complex models. Dropout
    // is carried on the model config (pair-wise for Complex); gradient
    // clipping runs in the optimizer-agnostic trainer step; input noise
    // is added to the packed `[Re, Im]` input before the forward pass.
    // Only `norm` stays Real-only — the Complex path has no
    // normalization layer.
    let (effective_grad_clip, effective_noise) =
        (config.grad_clip_norm, config.input_noise_std);

    // Use `*_with` builders when sub-params are supplied, else
    // `from_name` for the library defaults.
    let optimizer_config = match config.optimizer.to_ascii_lowercase().as_str() {
        "sgd" if config.sgd_momentum.is_some() || config.sgd_nesterov.is_some() => {
            OptimizerConfig::sgd_with(
                config.lr,
                config.weight_decay,
                config.sgd_momentum.unwrap_or(0.0),
                config.sgd_nesterov.unwrap_or(false),
            )
        }
        "rmsprop" if config.rmsprop_momentum.is_some() || config.rmsprop_alpha.is_some() => {
            OptimizerConfig::rmsprop_with(
                config.lr,
                config.weight_decay,
                config.rmsprop_momentum.unwrap_or(0.0),
                config.rmsprop_alpha.unwrap_or(0.99),
            )
        }
        _ => OptimizerConfig::from_name(&config.optimizer, config.lr, config.weight_decay)?,
    };
    let optimizer = optimizer_config.build(sparam_core::determinism::stable_all_vars(&varmap))?;

    // Share the alias parser + knob-aware builder with the HPO bridge so
    // a `/train` retrain from a trial reproduces its scheduler exactly.
    let scheduler_choice = sparam_hpo::SchedulerChoice::from_alias(&config.scheduler)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "unknown scheduler '{}', expected: none, cosine, plateau",
                config.scheduler
            ))
        })?;
    let scheduler_config = scheduler_choice.to_scheduler_config(
        config.max_epochs,
        config.cosine_eta_min,
        config.plateau_factor,
        config.plateau_patience,
        config.plateau_min_lr,
    );
    let scheduler = scheduler_config.build(config.lr)?;

    // Only enable input noise for strictly positive σ. Belt-and-suspenders:
    // matches the HPO evaluator's threshold and the trainer's own
    // `debug_assert!(std > 0.0)` inside `add_gaussian_noise`.
    let effective_input_noise = effective_noise.filter(|v| *v > 0.0);

    let mut trainer = TrainerBuilder::new(config.max_epochs)
        .with_early_stopping(Some(EarlyStoppingConfig {
            patience: config.patience,
            warmup_epochs: config.warmup_epochs,
            ..Default::default()
        }))
        .with_max_grad_norm(effective_grad_clip)
        .with_input_noise_std(effective_input_noise)
        .with_log_interval(1)
        .with_scheduler(scheduler)
        .with_log(log.clone())
        .build(optimizer)?
        .with_callback(callback);

    let batch_size = parse_batch_size(&config.batch_size)?;
    let shuffle_seed = config.shuffle_seed.unwrap_or(config.seed);
    let train_loader = build_loader(
        &prepared.train_features,
        &prepared.train_targets,
        batch_size,
        &prepared.feature_scaler,
        &prepared.target_scaler,
        true,
        shuffle_seed,
    )?;
    let val_loader = build_loader(
        &prepared.val_features,
        &prepared.val_targets,
        sparam_data::loader::BatchSize::All,
        &prepared.feature_scaler,
        &prepared.target_scaler,
        false,
        config.seed,
    )?;

    let loss = MlpLoss::from_name(&config.loss)?;
    let feature_scaler = Some(sparam_data::scaling::ScalerRef::from(&prepared.feature_scaler));
    let target_scaler = Some(sparam_data::scaling::ScalerRef::from(&prepared.target_scaler));
    let physics_ctx = sparam_physics::PhysicsContext::default_wr90(
        &candle_core::Device::Cpu,
    )?;
    // Per-trial λ when the form / HPO retrain supplies one; otherwise
    // fall back to the workflow default. Only consumed by the hybrid
    // PhysicsForward loss; MSE / SmoothL1 / EpsRelMse ignore it.
    let physics_lambda = config
        .physics_lambda
        .unwrap_or(crate::workflows::internal::training::DEFAULT_PHYSICS_LAMBDA);
    let loss_fn = packed_loss_fn(
        &model,
        loss,
        physics_lambda,
        feature_scaler,
        target_scaler,
        Some(&physics_ctx),
    );
    let result = trainer.fit(&model, &mut varmap, loss_fn, &train_loader, &val_loader)?;

    log.send(LogMessage::Info(format!(
        "\n  Training complete: best epoch {}/{}, stopped_early={}",
        result.best_epoch, result.final_epoch, result.stopped_early
    )));

    let evaluation = evaluate_model_predictions(
        &model,
        &prepared.test_features,
        &prepared.test_targets,
        &prepared.feature_scaler,
        &prepared.target_scaler,
        prepared.encoding,
    )?;

    log.send(LogMessage::Info("\nTest Metrics:".into()));
    log.send(LogMessage::MetricRow {
        label: "OK@1%".into(),
        value: format!(
            "{:.2}% ({}/{})",
            evaluation.ok_at_1.ok_percent,
            evaluation.ok_at_1.ok_count,
            evaluation.rel_error.n_samples
        ),
    });
    log.send(LogMessage::MetricRow {
        label: "OK@10%".into(),
        value: format!(
            "{:.2}% ({}/{})",
            evaluation.ok_at_10.ok_percent,
            evaluation.ok_at_10.ok_count,
            evaluation.rel_error.n_samples
        ),
    });
    log.send(LogMessage::MetricRow {
        label: "Mean error".into(),
        value: format!("{:.3}%", evaluation.rel_error.mean_error),
    });
    log.send(LogMessage::MetricRow {
        label: "Max error".into(),
        value: format!("{:.3}%", evaluation.rel_error.max_error),
    });
    log.send(LogMessage::MetricRow {
        label: "R²(ε')".into(),
        value: format!("{:.5}", evaluation.r2_scores.r2_real),
    });
    log.send(LogMessage::MetricRow {
        label: "R²(ε'')".into(),
        value: format!("{:.5}", evaluation.r2_scores.r2_imag),
    });

    // Always produce the in-memory safetensors buffer — that's what
    // the web UI streams to the DB. When the caller also asked for a
    // sidecar file (CLI), write it out too.
    let weights = save_model_checkpoint_bytes(&varmap)?;
    if let Some(path) = config.checkpoint_path.as_ref() {
        save_model_checkpoint(&varmap, path)?;
    }

    let metrics = TrainMetricsSummary {
        ok_at_1pct: evaluation.ok_at_1.ok_percent,
        ok_at_10pct: evaluation.ok_at_10.ok_percent,
        mean_error: evaluation.rel_error.mean_error,
        max_error: evaluation.rel_error.max_error,
        min_error: evaluation.rel_error.min_error,
        median_error: evaluation.rel_error.median_error,
        r2_real: evaluation.r2_scores.r2_real,
        r2_imag: evaluation.r2_scores.r2_imag,
        best_epoch: result.best_epoch,
        final_epoch: result.final_epoch,
        stopped_early: result.stopped_early,
        training_time_secs: result.training_time_secs,
    };

    Ok(TrainRunResult {
        arch_description,
        parameter_count,
        metrics,
        history: result.history,
        checkpoint_path: config.checkpoint_path.clone(),
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparam_data::generation::PermittivitySample;

    /// Tiny deterministic synthetic dataset for in-process training
    /// tests. `seed` offsets train/val/test so they carry distinct samples.
    fn toy_samples(n: usize, seed: u64) -> Vec<PermittivitySample> {
        (0..n)
            .map(|i| {
                let t = ((seed as f64) + (i as f64)) * 0.037;
                PermittivitySample {
                    s11_real: (t * 1.3).sin() * 0.5,
                    s11_imag: (t * 1.7).cos() * 0.5,
                    s21_real: (t * 2.1).sin() * 0.8,
                    s21_imag: (t * 2.3).cos() * 0.8,
                    eps_prime: 3.0 + (t * 0.5).sin() * 0.5,
                    eps_double_prime: 0.05 + (t * 0.7).cos() * 0.05,
                    is_dense_patch: false,
                }
            })
            .collect()
    }

    fn base_config(seed: u64) -> TrainConfig {
        use super::super::config::ModelConfig;
        TrainConfig {
            train_samples: toy_samples(64, seed + 1).into(),
            val_samples: toy_samples(16, seed + 2).into(),
            test_samples: toy_samples(16, seed + 3).into(),
            checkpoint_path: None,
            model: ModelConfig::real(8, "gelu", 0.0, "none").unwrap(),
            optimizer: "adamw".into(),
            lr: 1e-2,
            weight_decay: 1e-4,
            batch_size: "16".into(),
            max_epochs: 3,
            patience: 100,
            warmup_epochs: 0,
            scheduler: "none".into(),
            loss: "mse".into(),
            checkpoint: None,
            seed,
            log_progress: false,
            shuffle_seed: None,
            grad_clip_norm: None,
            input_noise_std: None,
            cosine_eta_min: None,
            plateau_factor: None,
            plateau_patience: None,
            plateau_min_lr: None,
            sgd_momentum: None,
            sgd_nesterov: None,
            rmsprop_momentum: None,
            rmsprop_alpha: None,
            pi_mape_beta: None,
            physics_lambda: None,
        }
    }

    /// `grad_clip_norm` reaches `TrainerBuilder::with_max_grad_norm`.
    #[test]
    fn grad_clip_norm_is_plumbed_into_trainer() {
        let seed = 777;
        let unclipped = run_training(&base_config(seed)).unwrap();

        let mut cfg = base_config(seed);
        cfg.grad_clip_norm = Some(1e-4);
        let clipped = run_training(&cfg).unwrap();

        assert_ne!(unclipped.weights, clipped.weights);
    }

    /// `input_noise_std` reaches `TrainerBuilder::with_input_noise_std`.
    #[test]
    fn input_noise_std_is_plumbed_into_trainer() {
        let seed = 888;
        let noiseless = run_training(&base_config(seed)).unwrap();

        let mut cfg = base_config(seed);
        cfg.input_noise_std = Some(0.1);
        let noisy = run_training(&cfg).unwrap();

        assert_ne!(noiseless.weights, noisy.weights);
    }

    /// `shuffle_seed` steers the DataLoader's batch order.
    #[test]
    fn shuffle_seed_steers_training_trajectory() {
        let seed = 4242;
        let mut cfg_a = base_config(seed);
        cfg_a.shuffle_seed = Some(100);
        let train_a = run_training(&cfg_a).unwrap();

        let mut cfg_b = base_config(seed);
        cfg_b.shuffle_seed = Some(200);
        let train_b = run_training(&cfg_b).unwrap();

        assert_ne!(train_a.weights, train_b.weights);
    }

    /// `shuffle_seed = None` falls back to `seed`.
    #[test]
    fn shuffle_seed_none_matches_seed_fallback() {
        let seed = 5151;
        let default_run = run_training(&base_config(seed)).unwrap();

        let mut explicit_cfg = base_config(seed);
        explicit_cfg.shuffle_seed = Some(seed);
        let explicit_run = run_training(&explicit_cfg).unwrap();

        assert_eq!(default_run.weights, explicit_run.weights);
    }

    /// `input_noise_std = Some(0.0)` is treated as no noise (matches HPO threshold).
    #[test]
    fn zero_input_noise_std_matches_none() {
        let seed = 999;
        let noise_none = run_training(&base_config(seed)).unwrap();

        let mut cfg = base_config(seed);
        cfg.input_noise_std = Some(0.0);
        let noise_zero = run_training(&cfg).unwrap();

        assert_eq!(
            noise_none.weights, noise_zero.weights,
            "input_noise_std = Some(0.0) should produce the same weights as None (the workflow filters <= 0)"
        );
    }
}

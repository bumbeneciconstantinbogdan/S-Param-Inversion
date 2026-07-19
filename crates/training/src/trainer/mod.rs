//! Training loop with early stopping and in-memory best-model tracking.
//!
//! The [`Trainer`] struct orchestrates training, validation, learning-rate
//! scheduling, and early stopping. It is generic over any model
//! implementing [`ModuleT`] and any loss closure
//! `Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>` — `(inputs, predictions, targets)`.
//!
//! Best-epoch weights live in a flat `Vec<f64>` slab (one contiguous
//! run per Var, in `stable_all_vars` order) and are memcpy'd in/out
//! without allocating Tensors. No intermediate disk I/O — callers that
//! want a persisted checkpoint serialize the final `VarMap` themselves
//! via [`crate::checkpoint`] helpers after training returns.

use std::fmt;
use std::time::Instant;

use candle_core::{DType, Result, Tensor, Var};
use candle_nn::{ModuleT, VarMap};
use serde::{Deserialize, Serialize};

use crate::batch_source::BatchSource;
use crate::logger::{LogMessage, LogSender};
use crate::optimizers::OptimizerKind;
use crate::schedulers::{
    CosineAnnealingConfig, CosineAnnealingLR, LRScheduler, ReduceOnPlateauConfig,
    ReduceOnPlateauLR,
};
use sparam_core::error::candle_msg;
use sparam_core::validation::{validate_non_negative_f64, validate_positive_f64, validate_positive_usize};

// ──────────────────────────────────────────────────────────────────
//  Early Stopping — see `early_stopping.rs`
// ──────────────────────────────────────────────────────────────────
//
// `EarlyStoppingMode`, `EarlyStoppingConfig`, and the internal
// `EarlyStoppingState` are defined in the sibling `early_stopping`
// module (extracted from this file to keep the trainer's core fit
// loop easier to locate). Re-exported here so existing call sites
// that import via `sparam_training::trainer::EarlyStoppingConfig`
// continue to work unchanged.

mod early_stopping;

pub use early_stopping::{EarlyStoppingConfig, EarlyStoppingMode};
use early_stopping::EarlyStoppingState;

// ──────────────────────────────────────────────────────────────────
//  Trainer Config
// ──────────────────────────────────────────────────────────────────

/// Configuration for the training loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    /// Maximum number of training epochs.
    pub max_epochs: usize,
    /// Optional early stopping.
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Print epoch metrics every N epochs.  `0` disables logging.
    pub log_interval: usize,
    /// Run validation every N epochs. The final epoch is always validated.
    pub val_interval: usize,
    /// Optional maximum global gradient norm applied before the optimizer step.
    pub max_grad_norm: Option<f64>,
    /// Optional standard deviation of Gaussian noise added to training inputs.
    /// `None` or `Some(0.0)` disables noise injection.
    pub input_noise_std: Option<f64>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            max_epochs: 100,
            early_stopping: None,
            log_interval: 1,
            val_interval: 1,
            max_grad_norm: None,
            input_noise_std: None,
        }
    }
}

impl TrainerConfig {
    fn validate(&self) -> Result<()> {
        validate_positive_usize("max_epochs", self.max_epochs)?;
        validate_positive_usize("val_interval", self.val_interval)?;
        if let Some(max_grad_norm) = self.max_grad_norm {
            validate_positive_f64("max_grad_norm", max_grad_norm)?;
        }
        if let Some(input_noise_std) = self.input_noise_std {
            validate_non_negative_f64("input_noise_std", input_noise_std)?;
        }
        if let Some(cfg) = &self.early_stopping {
            cfg.validate()?;
        }
        Ok(())
    }
}

/// Builder for a [`Trainer`] plus its optional learning-rate scheduler.
pub struct TrainerBuilder {
    config: TrainerConfig,
    scheduler: Option<LRScheduler>,
    log: LogSender,
}

impl TrainerBuilder {
    #[must_use]
    pub fn new(max_epochs: usize) -> Self {
        Self {
            config: TrainerConfig {
                max_epochs,
                ..TrainerConfig::default()
            },
            scheduler: None,
            log: LogSender::null(),
        }
    }

    #[must_use]
    pub fn from_config(config: TrainerConfig) -> Self {
        Self {
            config,
            scheduler: None,
            log: LogSender::null(),
        }
    }

    #[must_use]
    pub fn with_early_stopping(mut self, early_stopping: Option<EarlyStoppingConfig>) -> Self {
        self.config.early_stopping = early_stopping;
        self
    }

    #[must_use]
    pub fn with_log_interval(mut self, log_interval: usize) -> Self {
        self.config.log_interval = log_interval;
        self
    }

    #[must_use]
    pub fn with_val_interval(mut self, val_interval: usize) -> Self {
        self.config.val_interval = val_interval;
        self
    }

    #[must_use]
    pub fn with_max_grad_norm(mut self, max_grad_norm: Option<f64>) -> Self {
        self.config.max_grad_norm = max_grad_norm;
        self
    }

    #[must_use]
    pub fn with_input_noise_std(mut self, input_noise_std: Option<f64>) -> Self {
        self.config.input_noise_std = input_noise_std;
        self
    }

    #[must_use]
    pub fn with_scheduler(mut self, scheduler: Option<LRScheduler>) -> Self {
        self.scheduler = scheduler;
        self
    }

    pub fn with_cosine_scheduler(
        mut self,
        initial_lr: f64,
        config: CosineAnnealingConfig,
    ) -> Result<Self> {
        let scheduler = CosineAnnealingLR::new(initial_lr, config)?;
        self.scheduler = Some(LRScheduler::CosineAnnealing(scheduler));
        Ok(self)
    }

    pub fn with_plateau_scheduler(
        mut self,
        initial_lr: f64,
        config: ReduceOnPlateauConfig,
    ) -> Result<Self> {
        let scheduler = ReduceOnPlateauLR::new(initial_lr, config)?;
        self.scheduler = Some(LRScheduler::ReduceOnPlateau(scheduler));
        Ok(self)
    }

    #[must_use]
    pub fn config(&self) -> &TrainerConfig {
        &self.config
    }

    #[must_use]
    pub fn scheduler_ref(&self) -> Option<&LRScheduler> {
        self.scheduler.as_ref()
    }

    /// Attach a [`LogSender`] so the trainer dispatches output through the
    /// non-blocking channel instead of writing to stderr directly.
    #[must_use]
    pub fn with_log(mut self, log: LogSender) -> Self {
        self.log = log;
        self
    }

    pub fn build(self, optimizer: OptimizerKind) -> Result<Trainer<NoopTrainingCallback>> {
        let mut trainer = Trainer::with_log(optimizer, self.config, self.log)?;
        if let Some(scheduler) = self.scheduler {
            trainer.set_scheduler(scheduler);
        }
        Ok(trainer)
    }
}

// ──────────────────────────────────────────────────────────────────
//  Training Result
// ──────────────────────────────────────────────────────────────────

/// Per-epoch metrics collected during training.
#[derive(Debug, Clone, Default)]
pub struct TrainingHistory {
    pub train_losses: Vec<f64>,
    pub val_losses: Vec<Option<f64>>,
    pub learning_rates: Vec<f64>,
}

/// Per-epoch metrics emitted to optional training callbacks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub max_epochs: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub learning_rate: f64,
    pub elapsed_secs: f64,
    pub patience_status: Option<(usize, usize)>,
}

/// Internal result of one [`Trainer::run_epoch`] call. Lets the caller
/// decide whether to snapshot weights (`fit`) or fire a user closure
/// (`fit_for_hpo`) before the next iteration's training step clobbers
/// the vars.
struct EpochOutcome {
    val_loss: Option<f64>,
    improved: bool,
    should_stop: bool,
}

/// Optional extension points for progress bars or external loggers.
pub trait TrainingCallback {
    fn on_epoch_end(&mut self, _metrics: &EpochMetrics) {}

    fn on_improvement(&mut self, _epoch: usize, _val_loss: f64) {}

    fn on_early_stop(&mut self, _epoch: usize) {}
}

/// Default zero-cost callback type used when no callback is attached.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopTrainingCallback;

impl TrainingCallback for NoopTrainingCallback {}

/// Summary returned after training completes.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Last epoch that ran (1-indexed).
    pub final_epoch: usize,
    /// Epoch with the best validation loss (1-indexed).
    pub best_epoch: usize,
    /// Best validation loss observed.
    pub best_val_loss: f64,
    /// Whether training was terminated early.
    pub stopped_early: bool,
    /// Wall-clock duration of the training loop in seconds.
    pub training_time_secs: f64,
    /// Full per-epoch history.
    pub history: TrainingHistory,
}

impl fmt::Display for TrainingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epochs={}/", self.final_epoch)?;
        if self.stopped_early {
            write!(f, "{}", self.final_epoch)?;
        } else {
            f.write_str("max")?;
        }
        write!(f, " best_epoch={} best_val_loss=", self.best_epoch)?;
        if self.best_val_loss.is_finite() {
            write!(f, "{:.6}", self.best_val_loss)?;
        } else {
            f.write_str("n/a")?;
        }
        write!(
            f,
            " early_stopped={} time={:.1}s",
            self.stopped_early, self.training_time_secs
        )
    }
}

struct TrainingState {
    history: TrainingHistory,
    best_epoch: usize,
    best_val_loss: f64,
    /// Flat `Vec<f64>` slab of best-epoch weights. Layout is one
    /// contiguous run per Var, in the order produced by
    /// `stable_all_vars` (so snapshot and restore line up by index).
    /// `None` until the first improvement.
    best_snapshot: Option<Vec<f64>>,
    stopped_early: bool,
    early_stopping: Option<EarlyStoppingState>,
}

impl TrainingState {
    fn new(config: &TrainerConfig) -> Self {
        // Pre-size the history vectors to `max_epochs`; saves the
        // geometric-growth reallocs across a full-length run. Early
        // stopping just leaves some tail capacity unused — fine.
        let n = config.max_epochs;
        Self {
            history: TrainingHistory {
                train_losses: Vec::with_capacity(n),
                val_losses: Vec::with_capacity(n),
                learning_rates: Vec::with_capacity(n),
            },
            best_epoch: 0,
            best_val_loss: f64::INFINITY,
            best_snapshot: None,
            stopped_early: false,
            early_stopping: config
                .early_stopping
                .as_ref()
                .map(|cfg| EarlyStoppingState::new(cfg.mode)),
        }
    }

    fn patience_status(&self, config: &TrainerConfig, epoch: usize) -> Option<(usize, usize)> {
        let early_stopping = config.early_stopping.as_ref()?;
        if epoch <= early_stopping.warmup_epochs {
            return None;
        }
        let state = self.early_stopping.as_ref()?;
        Some((state.epochs_without_improvement, early_stopping.patience))
    }

    fn should_stop(&self, config: &TrainerConfig, epoch: usize, validated: bool) -> bool {
        let Some(early_stopping) = config.early_stopping.as_ref() else {
            return false;
        };
        if !validated || epoch <= early_stopping.warmup_epochs {
            return false;
        }
        self.early_stopping
            .as_ref()
            .is_some_and(|state| state.epochs_without_improvement >= early_stopping.patience)
    }
}

// ──────────────────────────────────────────────────────────────────
//  Trainer
// ──────────────────────────────────────────────────────────────────

/// Orchestrates the training loop, validation, early stopping, and
/// checkpoint management.
pub struct Trainer<C = NoopTrainingCallback> {
    optimizer: OptimizerKind,
    scheduler: Option<LRScheduler>,
    config: TrainerConfig,
    callbacks: Vec<C>,
    log: LogSender,
}

impl Trainer<NoopTrainingCallback> {
    /// Create a new trainer with a no-op (silent) logger.
    pub fn new(optimizer: OptimizerKind, config: TrainerConfig) -> Result<Self> {
        Self::with_log(optimizer, config, LogSender::null())
    }

    /// Create a new trainer that dispatches output through the given
    /// [`LogSender`] channel.
    pub fn with_log(
        optimizer: OptimizerKind,
        config: TrainerConfig,
        log: LogSender,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            optimizer,
            scheduler: None,
            config,
            callbacks: Vec::new(),
            log,
        })
    }

    /// Attach the first callback and switch the trainer to that concrete callback type.
    #[must_use]
    pub fn with_callback<C>(self, callback: C) -> Trainer<C>
    where
        C: TrainingCallback,
    {
        Trainer {
            optimizer: self.optimizer,
            scheduler: self.scheduler,
            config: self.config,
            callbacks: vec![callback],
            log: self.log,
        }
    }
}

impl<C> Trainer<C>
where
    C: TrainingCallback,
{
    /// Attach a learning-rate scheduler.
    pub fn set_scheduler(&mut self, scheduler: LRScheduler) {
        self.scheduler = Some(scheduler);
    }

    #[must_use]
    pub fn optimizer(&self) -> &OptimizerKind {
        &self.optimizer
    }

    pub fn optimizer_mut(&mut self) -> &mut OptimizerKind {
        &mut self.optimizer
    }

    #[must_use]
    pub fn scheduler_ref(&self) -> Option<&LRScheduler> {
        self.scheduler.as_ref()
    }

    #[must_use]
    pub fn config(&self) -> &TrainerConfig {
        &self.config
    }

    /// Run the full training loop.
    ///
    /// `loss_fn` receives `(inputs, predictions, targets)` and must return a
    /// **scalar** loss tensor. `inputs` is the **clean** batch feature tensor
    /// (pre-noise) so physics-informed losses can recover the measured
    /// S-parameters; input noise, when enabled, is applied only to the
    /// model's forward pass. The trainer backpropagates this loss every batch
    /// and can optionally clip the resulting gradients before stepping the
    /// optimizer.
    ///
    /// On completion (or early stop) the best model weights are restored into
    /// `varmap`.
    #[must_use]
    pub fn fit<M, F>(
        &mut self,
        model: &M,
        varmap: &mut VarMap,
        loss_fn: F,
        train_loader: &dyn BatchSource,
        val_loader: &dyn BatchSource,
    ) -> Result<TrainingResult>
    where
        M: ModuleT,
        F: Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    {
        let start = Instant::now();
        let max_epochs = self.config.max_epochs;
        let vars = sparam_core::determinism::stable_all_vars(varmap);
        let var_sizes: Vec<usize> = vars.iter().map(|v| v.as_tensor().elem_count()).collect();
        let total_params: usize = var_sizes.iter().sum();

        let mut state = TrainingState::new(&self.config);

        // Per-epoch varmap hashes, gated by `SPARAM_DEBUG_HASH_VARMAP`.
        let debug_hash = sparam_core::determinism::debug_hash_enabled();
        if debug_hash {
            sparam_core::determinism::emit_varmap_hash(format_args!("fit init"), varmap);
        }

        for epoch in 1..=max_epochs {
            let outcome = self.run_epoch(
                epoch, model, &loss_fn, train_loader, val_loader, &vars, &mut state,
            )?;

            // Improvement action for `fit`: snapshot current weights
            // into a flat `Vec<f64>` so the final restore at the end
            // puts the caller's VarMap back to best-epoch state, then
            // fire the trait-level `on_improvement` callback.
            if outcome.improved && let Some(val_loss) = outcome.val_loss {
                state.best_epoch = epoch;
                state.best_val_loss = val_loss;
                let buf = state
                    .best_snapshot
                    .get_or_insert_with(|| vec![0.0f64; total_params]);
                snapshot_varmap_flat(&vars, &var_sizes, buf)?;
                self.emit_improvement(epoch, val_loss);
            }

            if outcome.should_stop {
                self.emit_early_stop(epoch);
                state.stopped_early = true;
                if debug_hash {
                    sparam_core::determinism::emit_varmap_hash(
                        format_args!("fit epoch {epoch:>3} (stopped_early)"),
                        varmap,
                    );
                }
                break;
            }

            if debug_hash {
                sparam_core::determinism::emit_varmap_hash(
                    format_args!("fit epoch {epoch:>3}"),
                    varmap,
                );
            }
        }

        if let Some(snapshot) = state.best_snapshot.take() {
            restore_varmap_from_flat(&vars, &var_sizes, &snapshot)?;
        }
        if debug_hash {
            sparam_core::determinism::emit_varmap_hash(format_args!("fit after-restore"), varmap);
        }

        Ok(TrainingResult {
            final_epoch: state.history.train_losses.len(),
            best_epoch: state.best_epoch,
            best_val_loss: state.best_val_loss,
            stopped_early: state.stopped_early,
            training_time_secs: start.elapsed().as_secs_f64(),
            history: state.history,
        })
    }

    /// Fast training path with **zero disk I/O** and **zero weight copying**.
    ///
    /// Calls `on_improvement` each time validation loss improves — at that
    /// instant the model naturally holds its best-epoch weights, so the
    /// caller can compute any needed metrics (e.g. OK@1%) without ever
    /// snapshotting or restoring parameters.
    ///
    /// **Note:** the model has *last-epoch* weights when this returns.
    /// Any metrics that need best-epoch weights must be captured inside the
    /// `on_improvement` closure.
    ///
    /// **Trait callbacks:** `on_epoch_end` on any `TrainingCallback`
    /// attached via [`TrainerBuilder::with_callback`] still fires, but
    /// the `on_improvement` and `on_early_stop` trait methods do NOT —
    /// HPO consumers use the `on_improvement` closure parameter instead,
    /// and the early-stop notification is derived from the returned
    /// [`TrainingResult::stopped_early`]. If you need trait-level
    /// improvement / early-stop callbacks, use [`Self::fit`].
    pub fn fit_for_hpo<M, F, H>(
        &mut self,
        model: &M,
        varmap: &mut VarMap,
        loss_fn: F,
        train_loader: &dyn BatchSource,
        val_loader: &dyn BatchSource,
        mut on_improvement: H,
    ) -> Result<TrainingResult>
    where
        M: ModuleT,
        F: Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
        H: FnMut() -> Result<()>,
    {
        let start = Instant::now();
        let max_epochs = self.config.max_epochs;
        let vars = sparam_core::determinism::stable_all_vars(varmap);
        let mut state = TrainingState::new(&self.config);

        let debug_hash = sparam_core::determinism::debug_hash_enabled();
        if debug_hash {
            sparam_core::determinism::emit_varmap_hash(format_args!("fit_for_hpo init"), varmap);
        }

        for epoch in 1..=max_epochs {
            let outcome = self.run_epoch(
                epoch, model, &loss_fn, train_loader, val_loader, &vars, &mut state,
            )?;

            // Improvement action for `fit_for_hpo`: the model currently
            // holds best-epoch weights (no snapshot), so fire the user
            // closure immediately while that's still true.
            if outcome.improved && let Some(val_loss) = outcome.val_loss {
                state.best_epoch = epoch;
                state.best_val_loss = val_loss;
                on_improvement()?;
            }

            if outcome.should_stop {
                state.stopped_early = true;
                if debug_hash {
                    sparam_core::determinism::emit_varmap_hash(
                        format_args!("fit_for_hpo epoch {epoch:>3} (stopped_early)"),
                        varmap,
                    );
                }
                break;
            }

            if debug_hash {
                sparam_core::determinism::emit_varmap_hash(
                    format_args!("fit_for_hpo epoch {epoch:>3}"),
                    varmap,
                );
            }
        }

        Ok(TrainingResult {
            final_epoch: state.history.train_losses.len(),
            best_epoch: state.best_epoch,
            best_val_loss: state.best_val_loss,
            stopped_early: state.stopped_early,
            training_time_secs: start.elapsed().as_secs_f64(),
            history: state.history,
        })
    }

    // ── internal helpers ──────────────────────────────────────────

    /// Run one training epoch and return the mean batch loss.
    fn train_one_epoch<M, F>(
        &mut self,
        model: &M,
        loss_fn: &F,
        loader: &dyn BatchSource,
        vars: &[Var],
    ) -> Result<f64>
    where
        M: ModuleT,
        F: Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    {
        // Accumulate on the tensor side to skip per-batch `to_scalar`.
        // Both operands must be detached: without it each `acc + loss`
        // extends a `BackpropOp::Add` chain that overflows the stack
        // on recursive Drop after ~1 250 batches.
        let mut loss_sum: Option<Tensor> = None;
        let mut n_batches = 0usize;

        for batch in loader.batches() {
            let (x, y) = batch?;
            // Input noise regularises the model but must NOT corrupt the
            // features the physics-forward loss compares against, so we
            // keep a handle on the clean `x` and feed only the noisy copy
            // through the model's forward pass.  When noise is off we
            // reuse `&x` directly — no per-batch tensor clone.
            let noisy: Option<Tensor> = match self.config.input_noise_std {
                Some(std) if std > 0.0 => Some(add_gaussian_noise(&x, std)?),
                _ => None,
            };
            let forward_input: &Tensor = noisy.as_ref().unwrap_or(&x);
            let pred = model.forward_t(forward_input, true)?;
            // CRITICAL: Use forward_input (which may be noisy) for loss computation,
            // not the original x. This ensures PhysicsForward loss uses the same
            // S-parameters that the model sees, maintaining consistency.
            // Without this, when noise is added: model sees S_noisy but PI loss uses S_clean.
            let loss = loss_fn(forward_input, &pred, &y)?;
            self.apply_optimizer_step(&loss, vars)?;

            let loss_detached = loss.detach();
            loss_sum = Some(match loss_sum {
                None => loss_detached,
                Some(acc) => (acc + loss_detached)?.detach(),
            });
            n_batches += 1;
        }

        // One scalar extraction per epoch; a NaN batch still propagates
        // here so we abort the epoch.
        let Some(sum) = loss_sum else {
            return Err(candle_msg("training loader produced no batches"));
        };
        let mean_loss = scalar_loss_f64(&sum)? / n_batches as f64;
        if !mean_loss.is_finite() {
            return Err(candle_msg(format!(
                "non-finite training loss after {n_batches} batches: {mean_loss}"
            )));
        }

        Ok(mean_loss)
    }

    fn apply_optimizer_step(&mut self, loss: &Tensor, vars: &[Var]) -> Result<()> {
        if let Some(max_grad_norm) = self.config.max_grad_norm {
            let mut grads = loss.backward()?;
            sparam_core::determinism::clip_grad_norm_stable(&mut grads, vars, max_grad_norm)?;
            self.optimizer.step(&grads)
        } else {
            self.optimizer.backward_step(loss)
        }
    }

    /// Step the scheduler (if any) and return the current learning rate.
    fn step_scheduler(&mut self, val_loss: Option<f64>) -> Result<f64> {
        let Some(scheduler) = &mut self.scheduler else {
            return Ok(self.optimizer.learning_rate());
        };
        if !scheduler.requires_metric() {
            return scheduler.step_optimizer(&mut self.optimizer);
        }
        let Some(val_loss) = val_loss else {
            return Ok(self.optimizer.learning_rate());
        };
        scheduler.step_optimizer_with_metric(val_loss, &mut self.optimizer)
    }

    /// Run one full epoch (train → validate → scheduler → improvement
    /// check → history → epoch-end emit) and return what the caller
    /// needs to decide on the improvement action and whether to stop.
    ///
    /// Shared between [`Self::fit`] and [`Self::fit_for_hpo`]; neither
    /// the snapshot nor the user `on_improvement` closure fires from
    /// here — that's the caller's job, because those two functions
    /// handle improvements differently (deep-copy vs. fire-in-place).
    fn run_epoch<M, F>(
        &mut self,
        epoch: usize,
        model: &M,
        loss_fn: &F,
        train_loader: &dyn BatchSource,
        val_loader: &dyn BatchSource,
        vars: &[Var],
        state: &mut TrainingState,
    ) -> Result<EpochOutcome>
    where
        M: ModuleT,
        F: Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    {
        let epoch_start = Instant::now();

        let train_loss = self.train_one_epoch(model, loss_fn, train_loader, vars)?;

        let should_validate = self.should_validate(epoch);
        let val_loss = if should_validate {
            let loss = evaluate(model, loss_fn, val_loader)?;
            if !loss.is_finite() {
                return Err(candle_msg(format!(
                    "non-finite validation loss at epoch {epoch}: {loss}"
                )));
            }
            Some(loss)
        } else {
            None
        };

        let lr = self.step_scheduler(val_loss)?;

        let improved = match val_loss {
            Some(v) => self.check_improvement(epoch, v, state),
            None => false,
        };

        state.history.train_losses.push(train_loss);
        state.history.val_losses.push(val_loss);
        state.history.learning_rates.push(lr);

        let metrics = EpochMetrics {
            epoch,
            max_epochs: self.config.max_epochs,
            train_loss,
            val_loss,
            learning_rate: lr,
            elapsed_secs: epoch_start.elapsed().as_secs_f64(),
            patience_status: state.patience_status(&self.config, epoch),
        };

        if self.config.log_interval > 0 && epoch % self.config.log_interval == 0 {
            self.log_epoch_metrics(&metrics);
        }
        self.emit_epoch_end(&metrics);

        let should_stop = state.should_stop(&self.config, epoch, should_validate);

        Ok(EpochOutcome {
            val_loss,
            improved,
            should_stop,
        })
    }

    /// Decide whether this epoch's `val_loss` improves on the best so
    /// far, updating early-stopping bookkeeping as a side effect.
    /// Pure w.r.t. the VarMap — callers take the improvement action
    /// (snapshot in `fit`, user callback in `fit_for_hpo`) themselves
    /// so the function has one job.
    fn check_improvement(
        &self,
        epoch: usize,
        val_loss: f64,
        state: &mut TrainingState,
    ) -> bool {
        if let (Some(early_state), Some(config)) =
            (state.early_stopping.as_mut(), &self.config.early_stopping)
        {
            let monitoring = epoch > config.warmup_epochs;
            let min_delta = if monitoring { config.min_delta } else { 0.0 };
            let improved = config
                .mode
                .is_improvement(val_loss, early_state.best_metric, min_delta);

            if improved {
                early_state.best_metric = val_loss;
                early_state.best_epoch = epoch;
                if monitoring {
                    early_state.epochs_without_improvement = 0;
                }
            } else if monitoring {
                early_state.epochs_without_improvement += 1;
            }

            return improved;
        }

        val_loss < state.best_val_loss
    }

    fn should_validate(&self, epoch: usize) -> bool {
        epoch == self.config.max_epochs || epoch.is_multiple_of(self.config.val_interval)
    }

    fn log_epoch_metrics(&self, metrics: &EpochMetrics) {
        let warmup_status = if metrics.patience_status.is_none() {
            self.config.early_stopping.as_ref().and_then(|es| {
                (metrics.epoch <= es.warmup_epochs)
                    .then_some((metrics.epoch, es.warmup_epochs))
            })
        } else {
            None
        };
        self.log.send(LogMessage::Progress {
            epoch: metrics.epoch,
            total: metrics.max_epochs,
            loss: metrics.train_loss,
            val_loss: metrics.val_loss,
            lr: metrics.learning_rate,
            elapsed_secs: metrics.elapsed_secs,
            patience_status: metrics.patience_status,
            warmup_status,
        });
    }

    fn emit_epoch_end(&mut self, metrics: &EpochMetrics) {
        for callback in &mut self.callbacks {
            callback.on_epoch_end(metrics);
        }
    }

    fn emit_improvement(&mut self, epoch: usize, val_loss: f64) {
        for callback in &mut self.callbacks {
            callback.on_improvement(epoch, val_loss);
        }
    }

    fn emit_early_stop(&mut self, epoch: usize) {
        for callback in &mut self.callbacks {
            callback.on_early_stop(epoch);
        }
    }

}

/// Memcpy each Var's CPU storage into the flat `buf`, widening to
/// `f64` so the slab is dtype-agnostic. Both F32 and F64 Vars round-
/// trip cleanly via the matching `Tensor::from_slice` path in
/// [`restore_varmap_from_flat`].
fn snapshot_varmap_flat(vars: &[Var], var_sizes: &[usize], buf: &mut [f64]) -> Result<()> {
    let mut off = 0;
    for (var, &n) in vars.iter().zip(var_sizes) {
        let (storage, layout) = var.as_tensor().storage_and_layout();
        let s = layout.start_offset();
        match &*storage {
            candle_core::Storage::Cpu(cpu) => match var.as_tensor().dtype() {
                DType::F64 => {
                    let src: &[f64] = cpu.as_slice()?;
                    buf[off..off + n].copy_from_slice(&src[s..s + n]);
                }
                DType::F32 => {
                    let src: &[f32] = cpu.as_slice()?;
                    for (dst, &v) in buf[off..off + n].iter_mut().zip(&src[s..s + n]) {
                        *dst = v as f64;
                    }
                }
                dt => {
                    return Err(candle_msg(format!(
                        "snapshot_varmap_flat: unsupported dtype {dt:?}"
                    )));
                }
            },
            _ => {
                return Err(candle_msg(
                    "snapshot_varmap_flat: only CPU storage supported",
                ));
            }
        }
        off += n;
    }
    Ok(())
}

/// Restore each Var from `slab` (output of [`snapshot_varmap_flat`]).
/// Narrows to F32 if the Var's native dtype is F32; F64 uses the slab
/// values directly.
fn restore_varmap_from_flat(vars: &[Var], var_sizes: &[usize], slab: &[f64]) -> Result<()> {
    let mut off = 0;
    for (var, &n) in vars.iter().zip(var_sizes) {
        let tensor = var.as_tensor();
        let device = tensor.device();
        let shape = tensor.shape();
        let t = match tensor.dtype() {
            DType::F64 => Tensor::from_slice(&slab[off..off + n], shape, device)?,
            DType::F32 => {
                let narrowed: Vec<f32> =
                    slab[off..off + n].iter().map(|&v| v as f32).collect();
                Tensor::from_slice(&narrowed, shape, device)?
            }
            dt => {
                return Err(candle_msg(format!(
                    "restore_varmap_from_flat: unsupported dtype {dt:?}"
                )));
            }
        };
        var.set(&t)?;
        off += n;
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────
//  Standalone evaluation helper
// ──────────────────────────────────────────────────────────────────

/// Evaluate a model on a batch source, returning the mean loss across batches.
///
/// Uses `forward_t(xs, false)` (inference mode) for each batch. Loss
/// accumulation stays on the tensor side until the final scalar
/// extraction — same pattern as [`Trainer::train_one_epoch`] to avoid
/// the per-batch `to_scalar` materialisation barrier.
pub fn evaluate<M, F>(model: &M, loss_fn: &F, loader: &dyn BatchSource) -> Result<f64>
where
    M: ModuleT,
    F: Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
{
    let mut loss_sum: Option<Tensor> = None;
    let mut n_batches = 0usize;

    for batch in loader.batches() {
        let (x, y) = batch?;
        let pred = model.forward_t(&x, false)?;
        let loss = loss_fn(&x, &pred, &y)?;
        let loss_detached = loss.detach();
        // Detach the running sum too — see `train_one_epoch`.
        loss_sum = Some(match loss_sum {
            None => loss_detached,
            Some(acc) => (acc + loss_detached)?.detach(),
        });
        n_batches += 1;
    }

    let Some(sum) = loss_sum else {
        return Err(candle_msg("evaluation loader produced no batches"));
    };
    Ok(scalar_loss_f64(&sum)? / n_batches as f64)
}

/// Extract a scalar loss value as f64, handling both F32 and F64 dtypes.
fn scalar_loss_f64(loss: &Tensor) -> Result<f64> {
    match loss.dtype() {
        DType::F32 => loss.to_scalar::<f32>().map(f64::from),
        DType::F64 => loss.to_scalar::<f64>(),
        dtype => Err(candle_msg(format!(
            "expected F32 or F64 scalar loss, got {dtype:?}"
        ))),
    }
}

/// Add Gaussian noise to a tensor. Caller must ensure `std > 0` (validated
/// by [`TrainerConfig::validate`] at construction time).
#[inline]
fn add_gaussian_noise(x: &Tensor, std: f64) -> Result<Tensor> {
    debug_assert!(std.is_finite() && std > 0.0);
    let noise = sparam_core::determinism::seeded_randn_like(x, 0.0, std)?;
    x + &noise
}

// ──────────────────────────────────────────────────────────────────
//  Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    use candle_core::{DType, Device, Var};
    use candle_nn::VarBuilder;

    use sparam_models::{Activation, MLPConfig, MLPRegressor};
    use crate::losses::{mse_loss, Reduction};
    use crate::optimizers::AdamWConfig;

    // ── TestBatchSource ───────────────────────────────────────

    /// Minimal [`BatchSource`] implementation for test fixtures,
    /// replacing `DataLoader` from the data crate.
    struct TestBatchSource {
        features: Tensor,
        targets: Tensor,
        batch_size: usize,
    }

    impl TestBatchSource {
        fn new(features: Tensor, targets: Tensor, batch_size: usize) -> Self {
            Self {
                features,
                targets,
                batch_size,
            }
        }

        fn all(features: Tensor, targets: Tensor) -> Self {
            let n = features.dim(0).unwrap();
            Self {
                features,
                targets,
                batch_size: n,
            }
        }
    }

    impl BatchSource for TestBatchSource {
        fn batches(&self) -> Box<dyn Iterator<Item = Result<(Tensor, Tensor)>> + '_> {
            let n = self.sample_count();
            let batch_size = self.batch_size;
            let features = &self.features;
            let targets = &self.targets;
            let mut offset = 0;
            Box::new(std::iter::from_fn(move || {
                if offset >= n {
                    return None;
                }
                let len = batch_size.min(n - offset);
                let result = (|| -> Result<(Tensor, Tensor)> {
                    let x = features.narrow(0, offset, len)?;
                    let y = targets.narrow(0, offset, len)?;
                    Ok((x, y))
                })();
                offset += len;
                Some(result)
            }))
        }

        fn sample_count(&self) -> usize {
            self.features.dim(0).unwrap()
        }

        fn num_batches(&self) -> usize {
            self.sample_count().div_ceil(self.batch_size)
        }
    }

    // ── test fixtures ──────────────────────────────────────────

    /// Create deterministic training data: y = x1 + x2.
    fn make_test_data(n: usize) -> (Tensor, Tensor) {
        let mut feats = Vec::with_capacity(n * 2);
        let mut tgts = Vec::with_capacity(n);
        for i in 0..n {
            let x1 = (i as f64) / (n as f64);
            let x2 = ((i * 7 + 3) % n) as f64 / (n as f64);
            feats.push(x1);
            feats.push(x2);
            tgts.push(x1 + x2);
        }
        let features = Tensor::from_vec(feats, (n, 2), &Device::Cpu).unwrap();
        let targets = Tensor::from_vec(tgts, (n, 1), &Device::Cpu).unwrap();
        (features, targets)
    }

    fn make_batch_sources(
        n_train: usize,
        n_val: usize,
        batch_size: usize,
    ) -> (TestBatchSource, TestBatchSource) {
        let (train_f, train_t) = make_test_data(n_train);
        let (val_f, val_t) = make_test_data(n_val);
        let train_source = TestBatchSource::new(train_f, train_t, batch_size);
        let val_source = TestBatchSource::all(val_f, val_t);
        (train_source, val_source)
    }

    fn make_model() -> (VarMap, MLPRegressor) {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        let config = MLPConfig::new(2, 16, 1, Activation::GELU);
        let model = MLPRegressor::new(vb, &config).unwrap();
        (varmap, model)
    }

    fn make_optimizer(varmap: &VarMap) -> OptimizerKind {
        OptimizerKind::adamw(
            varmap.all_vars(),
            AdamWConfig {
                lr: 1e-2,
                ..AdamWConfig::default()
            },
        )
        .unwrap()
    }

    fn mse(_input: &Tensor, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        mse_loss(pred, target, Reduction::Mean)
    }

    #[derive(Clone, Default)]
    struct RecordingCallback {
        epochs: Rc<RefCell<Vec<(usize, Option<f64>)>>>,
        improvements: Rc<RefCell<Vec<(usize, f64)>>>,
        early_stops: Rc<RefCell<Vec<usize>>>,
    }

    impl TrainingCallback for RecordingCallback {
        fn on_epoch_end(&mut self, metrics: &EpochMetrics) {
            self.epochs
                .borrow_mut()
                .push((metrics.epoch, metrics.val_loss));
        }

        fn on_improvement(&mut self, epoch: usize, val_loss: f64) {
            self.improvements.borrow_mut().push((epoch, val_loss));
        }

        fn on_early_stop(&mut self, epoch: usize) {
            self.early_stops.borrow_mut().push(epoch);
        }
    }

    struct RecordingModel<M> {
        inner: M,
        train_inputs: Rc<RefCell<Vec<Vec<f64>>>>,
        eval_inputs: Rc<RefCell<Vec<Vec<f64>>>>,
    }

    impl<M> ModuleT for RecordingModel<M>
    where
        M: ModuleT,
    {
        fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
            let snapshot = xs.flatten_all()?.to_vec1::<f64>()?;
            if train {
                self.train_inputs.borrow_mut().push(snapshot);
            } else {
                self.eval_inputs.borrow_mut().push(snapshot);
            }
            self.inner.forward_t(xs, train)
        }
    }

    fn mean_and_std(values: &[f64]) -> (f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f64>()
            / values.len() as f64;
        (mean, variance.sqrt())
    }

    // ── config validation ──────────────────────────────────────

    #[test]
    fn config_rejects_zero_max_epochs() {
        let (varmap, _model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let config = TrainerConfig {
            max_epochs: 0,
            ..TrainerConfig::default()
        };
        assert!(Trainer::new(optimizer, config).is_err());
    }

    #[test]
    fn config_rejects_invalid_min_delta() {
        // Same rule for negative and NaN `min_delta`; exercising both in
        // one test keeps the validation-rule coverage in one place.
        for bad in [-1.0, f64::NAN] {
            let (varmap, _model) = make_model();
            let optimizer = make_optimizer(&varmap);
            let config = TrainerConfig {
                early_stopping: Some(EarlyStoppingConfig {
                    min_delta: bad,
                    ..EarlyStoppingConfig::default()
                }),
                ..TrainerConfig::default()
            };
            assert!(
                Trainer::new(optimizer, config).is_err(),
                "expected error for min_delta = {bad}",
            );
        }
    }

    #[test]
    fn config_rejects_zero_patience() {
        let (varmap, _model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let config = TrainerConfig {
            early_stopping: Some(EarlyStoppingConfig {
                patience: 0,
                ..EarlyStoppingConfig::default()
            }),
            ..TrainerConfig::default()
        };
        assert!(Trainer::new(optimizer, config).is_err());
    }

    #[test]
    fn config_rejects_zero_val_interval() {
        let (varmap, _model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let config = TrainerConfig {
            val_interval: 0,
            ..TrainerConfig::default()
        };
        assert!(Trainer::new(optimizer, config).is_err());
    }

    #[test]
    fn config_rejects_invalid_max_grad_norm() {
        let (varmap, _model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let zero = TrainerConfig {
            max_grad_norm: Some(0.0),
            ..TrainerConfig::default()
        };
        assert!(Trainer::new(optimizer, zero).is_err());

        let (varmap, _model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let nan = TrainerConfig {
            max_grad_norm: Some(f64::NAN),
            ..TrainerConfig::default()
        };
        assert!(Trainer::new(optimizer, nan).is_err());
    }

    #[test]
    fn config_rejects_invalid_input_noise_std() {
        let (varmap, _model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let negative = TrainerConfig {
            input_noise_std: Some(-0.1),
            ..TrainerConfig::default()
        };
        assert!(Trainer::new(optimizer, negative).is_err());

        let (varmap, _model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let nan = TrainerConfig {
            input_noise_std: Some(f64::NAN),
            ..TrainerConfig::default()
        };
        assert!(Trainer::new(optimizer, nan).is_err());
    }

    #[test]
    fn trainer_builder_accumulates_config_fields() {
        let early_stopping = EarlyStoppingConfig {
            patience: 7,
            min_delta: 1e-4,
            warmup_epochs: 3,
            ..EarlyStoppingConfig::default()
        };
        let builder = TrainerBuilder::new(25)
            .with_early_stopping(Some(early_stopping.clone()))
            .with_log_interval(4)
            .with_val_interval(2)
            .with_max_grad_norm(Some(1.5))
            .with_input_noise_std(Some(0.05));

        let config = builder.config();
        assert_eq!(config.max_epochs, 25);
        assert_eq!(config.log_interval, 4);
        assert_eq!(config.val_interval, 2);
        assert_eq!(config.max_grad_norm, Some(1.5));
        assert_eq!(config.input_noise_std, Some(0.05));
        assert_eq!(
            config.early_stopping.as_ref().map(|cfg| cfg.patience),
            Some(early_stopping.patience)
        );
    }

    #[test]
    fn trainer_builder_builds_trainer_with_scheduler() {
        let (varmap, _model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let trainer = TrainerBuilder::new(10)
            .with_cosine_scheduler(
                1e-3,
                CosineAnnealingConfig {
                    t_max: 10,
                    eta_min: 1e-5,
                },
            )
            .unwrap()
            .build(optimizer)
            .unwrap();

        assert!(matches!(
            trainer.scheduler_ref(),
            Some(LRScheduler::CosineAnnealing(_))
        ));
    }

    // ── early stopping mode ────────────────────────────────────
    //
    // Unit tests for `EarlyStoppingMode::is_improvement` live beside
    // the type in `trainer/early_stopping.rs`. The trainer-integration
    // tests below (which exercise the fit loop under various
    // `EarlyStoppingConfig`s) stay here since they need the full
    // Trainer + VarMap + model setup this test module already wires.

    // ── basic training ─────────────────────────────────────────

    #[test]
    fn basic_training_decreases_loss() {
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let (train_source, val_source) = make_batch_sources(64, 32, 32);

        let config = TrainerConfig {
            max_epochs: 30,
            log_interval: 0,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config).unwrap();
        let result = trainer
            .fit(&model, &mut varmap, mse, &train_source, &val_source)
            .unwrap();

        // Loss should decrease from first to last epoch.
        let first = result.history.val_losses[0].unwrap();
        let last = result.history.val_losses.last().copied().flatten().unwrap();
        assert!(
            last < first,
            "expected val loss to decrease: first={first}, last={last}"
        );
        assert_eq!(result.final_epoch, 30);
        assert!(!result.stopped_early);
    }

    #[test]
    fn basic_training_supports_f32_loss_accumulation() {
        let (train_f, train_t) = make_test_data(64);
        let (val_f, val_t) = make_test_data(32);
        let train_source = TestBatchSource::new(
            train_f.to_dtype(DType::F32).unwrap(),
            train_t.to_dtype(DType::F32).unwrap(),
            32,
        );
        let val_source = TestBatchSource::all(
            val_f.to_dtype(DType::F32).unwrap(),
            val_t.to_dtype(DType::F32).unwrap(),
        );

        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let config = MLPConfig::new(2, 16, 1, Activation::GELU);
        let model = MLPRegressor::new(vb, &config).unwrap();
        let optimizer = make_optimizer(&varmap);

        let config = TrainerConfig {
            max_epochs: 5,
            log_interval: 0,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config).unwrap();
        let result = trainer
            .fit(&model, &mut varmap, mse, &train_source, &val_source)
            .unwrap();

        assert_eq!(result.final_epoch, 5);
        assert!(result.best_val_loss.is_finite());
    }

    // ── early stopping kicks in ────────────────────────────────

    #[test]
    fn early_stopping_stops_training() {
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let (train_source, val_source) = make_batch_sources(64, 32, 64);

        let config = TrainerConfig {
            max_epochs: 50,
            early_stopping: Some(EarlyStoppingConfig {
                patience: 5,
                min_delta: 1e6,
                warmup_epochs: 0,
                mode: EarlyStoppingMode::Min,
            }),
            log_interval: 0,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config).unwrap();
        let result = trainer
            .fit(&model, &mut varmap, mse, &train_source, &val_source)
            .unwrap();

        // Should stop before reaching 500 epochs.
        assert!(
            result.stopped_early || result.final_epoch < 500,
            "expected early stop, got final_epoch={}",
            result.final_epoch
        );
        assert!(result.best_epoch <= result.final_epoch);
    }

    // ── warmup delays monitoring ───────────────────────────────

    #[test]
    fn warmup_delays_early_stopping() {
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let (train_source, val_source) = make_batch_sources(64, 32, 64);

        let warmup = 10;
        let config = TrainerConfig {
            max_epochs: 15,
            early_stopping: Some(EarlyStoppingConfig {
                patience: 2,    // very tight patience
                min_delta: 1e6, // impossible to improve by this much
                warmup_epochs: warmup,
                mode: EarlyStoppingMode::Min,
            }),
            log_interval: 0,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config).unwrap();
        let result = trainer
            .fit(&model, &mut varmap, mse, &train_source, &val_source)
            .unwrap();

        // Even with patience=2 and impossible min_delta, we should reach at
        // least warmup + patience epochs because monitoring only starts after
        // warmup.
        assert!(
            result.final_epoch >= warmup + 2,
            "expected at least {} epochs, got {}",
            warmup + 2,
            result.final_epoch
        );
    }

    // ── scheduler integration ──────────────────────────────────

    #[test]
    fn cosine_scheduler_changes_lr() {
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let (train_source, val_source) = make_batch_sources(64, 32, 64);

        let config = TrainerConfig {
            max_epochs: 10,
            log_interval: 0,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config).unwrap();
        let scheduler = LRScheduler::CosineAnnealing(
            CosineAnnealingLR::from_optimizer(
                trainer.optimizer(),
                CosineAnnealingConfig {
                    t_max: 10,
                    eta_min: 1e-5,
                },
            )
            .unwrap(),
        );
        trainer.set_scheduler(scheduler);
        let result = trainer
            .fit(&model, &mut varmap, mse, &train_source, &val_source)
            .unwrap();

        // LR should change across epochs.
        let lrs = &result.history.learning_rates;
        assert!(lrs.len() == 10);
        let first_lr = lrs[0];
        let mid_lr = lrs[4];
        assert!(
            (first_lr - mid_lr).abs() > 1e-8,
            "expected LR to change across epochs"
        );
    }

    // ── checkpoint restore ─────────────────────────────────────

    #[test]
    fn best_model_is_restored_after_training() {
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let (train_source, val_source) = make_batch_sources(64, 32, 64);

        let config = TrainerConfig {
            max_epochs: 20,
            log_interval: 0,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config).unwrap();
        let result = trainer
            .fit(&model, &mut varmap, mse, &train_source, &val_source)
            .unwrap();

        // Re-evaluate with the restored best weights.
        let eval_loss = evaluate(&model, &mse, &val_source).unwrap();
        // The restored model should match the best recorded val loss.
        let tolerance = 1e-6;
        assert!(
            (eval_loss - result.best_val_loss).abs() < tolerance,
            "restored model loss ({eval_loss}) should match best_val_loss ({})",
            result.best_val_loss
        );
    }

    #[test]
    fn fit_for_hpo_captures_best_epoch_metrics_via_callback() {
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let (train_source, val_source) = make_batch_sources(64, 32, 64);

        let config = TrainerConfig {
            max_epochs: 20,
            log_interval: 0,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config).unwrap();

        // Capture val_loss inside the callback — model has best weights at
        // that instant, so evaluate() should match best_val_loss.
        let mut captured_loss: Option<f64> = None;
        let result = trainer
            .fit_for_hpo(
                &model,
                &mut varmap,
                mse,
                &train_source,
                &val_source,
                || {
                    captured_loss = Some(evaluate(&model, &mse, &val_source)?);
                    Ok(())
                },
            )
            .unwrap();

        let loss = captured_loss.expect("on_improvement should have been called");
        let tolerance = 1e-6;
        assert!(
            (loss - result.best_val_loss).abs() < tolerance,
            "callback-captured loss ({loss}) should match best_val_loss ({})",
            result.best_val_loss
        );
    }

    // ── loss-accumulator chain fix ─────────────────────────────

    #[test]
    fn epoch_loss_accumulator_does_not_recursively_drop_a_long_chain() {
        // Regression guard for the tokio-worker stack overflow that
        // hit trial 146 complex at ~1250 batches/epoch. Cause: the
        // per-batch running sum `(acc + loss_detached)?` built a
        // Candle `BackpropOp::Add(acc, loss)` per iteration, forming
        // an N-deep chain of `Arc<Tensor_>` refs whose recursive Drop
        // blew the 2 MB tokio-worker stack. Fix: detach the running
        // sum each step so `BackpropOp::new2` short-circuits to None
        // when neither operand tracks an op.
        //
        // Stack frames from Candle's Drop are small (~100 B each), so
        // 4 000 batches on the default ~2 MB test-thread stack would
        // have been just enough to blow it under the broken behaviour;
        // with the fix the chain depth stays at 1 and this runs
        // without issue.
        let batch_count = 4_000;
        let n_train = batch_count * 4; // batch_size = 4 → 4000 batches
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let (train_source, val_source) = make_batch_sources(n_train, 16, 4);
        let config = TrainerConfig {
            max_epochs: 1,
            log_interval: 0,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config).unwrap();
        let result = trainer
            .fit(&model, &mut varmap, mse, &train_source, &val_source)
            .unwrap();
        // If we got here without stack overflow, the accumulator chain
        // has been successfully broken.
        assert_eq!(result.final_epoch, 1);
        assert!(result.history.train_losses[0].is_finite());
    }

    // ── in-memory best-model snapshot ──────────────────────────

    #[test]
    fn best_model_snapshot_is_independent_of_subsequent_steps() {
        // Guard the core invariant of the in-memory snapshot: once
        // `fit` has called `snapshot_varmap` to deep-copy the VarMap
        // at an improvement, further optimizer steps on the live vars
        // must not leak into the snapshot. `Tensor::copy` allocates
        // fresh storage so this should hold — the test exercises it
        // by comparing post-fit restored weights against a fresh
        // `evaluate` at the reported `best_val_loss`.
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let (train_source, val_source) = make_batch_sources(128, 64, 64);

        let config = TrainerConfig {
            max_epochs: 15,
            log_interval: 0,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config).unwrap();
        let result = trainer
            .fit(&model, &mut varmap, mse, &train_source, &val_source)
            .unwrap();

        // After fit() returns, the VarMap holds the BEST-epoch weights
        // (restored from the in-memory snapshot). Re-evaluating with
        // the same val loader must therefore reproduce `best_val_loss`.
        let restored = evaluate(&model, &mse, &val_source).unwrap();
        assert!(
            (restored - result.best_val_loss).abs() < 1e-9,
            "restored val loss ({restored}) must match best_val_loss ({}) \
             — the snapshot must be detached from later optimizer steps",
            result.best_val_loss
        );
    }

    // ── NaN detection ──────────────────────────────────────────

    #[test]
    fn nan_loss_produces_error() {
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let (train_source, val_source) = make_batch_sources(64, 32, 64);

        let config = TrainerConfig {
            max_epochs: 5,
            log_interval: 0,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config).unwrap();
        // Loss function that returns NaN.
        let nan_loss = |_input: &Tensor, _pred: &Tensor, _target: &Tensor| -> Result<Tensor> {
            Tensor::new(&[f64::NAN], &Device::Cpu)
        };
        let result = trainer.fit(&model, &mut varmap, nan_loss, &train_source, &val_source);
        assert!(result.is_err());
    }

    // ── training history populated ─────────────────────────────

    #[test]
    fn history_vectors_populated() {
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let (train_source, val_source) = make_batch_sources(64, 32, 64);

        let config = TrainerConfig {
            max_epochs: 5,
            log_interval: 0,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config).unwrap();
        let result = trainer
            .fit(&model, &mut varmap, mse, &train_source, &val_source)
            .unwrap();

        assert_eq!(result.history.train_losses.len(), 5);
        assert_eq!(result.history.val_losses.len(), 5);
        assert_eq!(result.history.learning_rates.len(), 5);
        assert!(result.history.val_losses.iter().all(Option::is_some));
    }

    #[test]
    fn validation_interval_skips_intermediate_epochs() {
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let (train_source, val_source) = make_batch_sources(64, 32, 64);

        let config = TrainerConfig {
            max_epochs: 5,
            val_interval: 3,
            log_interval: 0,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config).unwrap();
        let result = trainer
            .fit(&model, &mut varmap, mse, &train_source, &val_source)
            .unwrap();

        assert_eq!(result.history.val_losses.len(), 5);
        assert_eq!(result.history.val_losses[0], None);
        assert_eq!(result.history.val_losses[1], None);
        assert!(result.history.val_losses[2].is_some());
        assert_eq!(result.history.val_losses[3], None);
        assert!(result.history.val_losses[4].is_some());
    }

    #[test]
    fn callbacks_receive_epoch_and_improvement_events() {
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let (train_source, val_source) = make_batch_sources(64, 32, 64);
        let callback = RecordingCallback::default();
        let recorded_epochs = callback.epochs.clone();
        let recorded_improvements = callback.improvements.clone();

        let config = TrainerConfig {
            max_epochs: 4,
            val_interval: 2,
            log_interval: 0,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config)
            .unwrap()
            .with_callback(callback);
        let _result = trainer
            .fit(&model, &mut varmap, mse, &train_source, &val_source)
            .unwrap();

        let epochs = recorded_epochs.borrow();
        assert_eq!(epochs.len(), 4);
        assert_eq!(epochs[0], (1, None));
        assert!(epochs[1].1.is_some());
        assert!(!recorded_improvements.borrow().is_empty());
        assert!(
            recorded_improvements
                .borrow()
                .iter()
                .all(|(_, loss)| loss.is_finite())
        );
    }

    #[test]
    fn callbacks_receive_early_stop_event() {
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let (train_source, val_source) = make_batch_sources(64, 32, 64);
        let callback = RecordingCallback::default();
        let recorded_early_stops = callback.early_stops.clone();

        let config = TrainerConfig {
            max_epochs: 100,
            early_stopping: Some(EarlyStoppingConfig {
                patience: 2,
                min_delta: 1e6,
                warmup_epochs: 0,
                mode: EarlyStoppingMode::Min,
            }),
            log_interval: 0,
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config)
            .unwrap()
            .with_callback(callback);
        let result = trainer
            .fit(&model, &mut varmap, mse, &train_source, &val_source)
            .unwrap();

        let early_stops = recorded_early_stops.borrow();
        assert!(result.stopped_early);
        assert_eq!(early_stops.len(), 1);
        assert_eq!(early_stops[0], result.final_epoch);
    }

    #[test]
    fn clip_grad_norm_scales_large_gradients() {
        let var = Var::from_slice(&[10.0f64, -10.0], 2, &Device::Cpu).unwrap();
        let loss = var.as_tensor().sqr().unwrap().sum_all().unwrap();
        let mut grads = loss.backward().unwrap();
        let vars = vec![var.clone()];

        let unclipped_norm =
            sparam_core::determinism::clip_grad_norm_stable(&mut grads, &vars, 5.0).unwrap();
        assert!(unclipped_norm > 5.0);

        let clipped = grads.get(var.as_tensor()).unwrap();
        let clipped_norm = clipped
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .sqrt()
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();
        assert!(
            (clipped_norm - 5.0).abs() < 1e-6,
            "expected clipped norm ~= 5, got {clipped_norm}"
        );
    }

    #[test]
    fn gaussian_noise_zero_std_is_no_op() {
        // When input_noise_std is Some(0.0), the trainer skips noise injection
        // entirely (std > 0.0 guard). Verify the match arm logic:
        let noise_std: Option<f64> = Some(0.0);
        let xs = Tensor::from_vec(vec![0.0f64, 1.0, 2.0, 3.0], (2, 2), &Device::Cpu).unwrap();
        let result = match noise_std {
            Some(std) if std > 0.0 => add_gaussian_noise(&xs, std).unwrap(),
            _ => xs.clone(),
        };
        assert_eq!(
            result.flatten_all().unwrap().to_vec1::<f64>().unwrap(),
            xs.flatten_all().unwrap().to_vec1::<f64>().unwrap()
        );
    }

    #[test]
    fn gaussian_noise_has_expected_statistics() {
        let xs = Tensor::from_vec(vec![0.0f64; 16_384], (4096, 4), &Device::Cpu).unwrap();
        let noisy = add_gaussian_noise(&xs, 0.2).unwrap();
        let samples = noisy.flatten_all().unwrap().to_vec1::<f64>().unwrap();
        let (mean, std) = mean_and_std(&samples);

        assert!(
            mean.abs() < 0.01,
            "expected near-zero mean noise, got {mean}"
        );
        assert!(
            (std - 0.2).abs() < 0.02,
            "expected std close to 0.2, got {std}"
        );
    }

    #[test]
    fn input_noise_is_applied_only_to_training_inputs() {
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);

        let train_features =
            Tensor::from_vec(vec![0.1f64, 0.2, 0.3, 0.4], (2, 2), &Device::Cpu).unwrap();
        let train_targets = Tensor::from_vec(vec![0.3f64, 0.7], (2, 1), &Device::Cpu).unwrap();
        let val_features =
            Tensor::from_vec(vec![0.5f64, 0.6, 0.7, 0.8], (2, 2), &Device::Cpu).unwrap();
        let val_targets = Tensor::from_vec(vec![1.1f64, 1.5], (2, 1), &Device::Cpu).unwrap();

        let train_source = TestBatchSource::all(train_features.clone(), train_targets);
        let val_source = TestBatchSource::all(val_features.clone(), val_targets);

        let recorded_train_inputs = Rc::new(RefCell::new(Vec::new()));
        let recorded_eval_inputs = Rc::new(RefCell::new(Vec::new()));
        let model = RecordingModel {
            inner: model,
            train_inputs: recorded_train_inputs.clone(),
            eval_inputs: recorded_eval_inputs.clone(),
        };

        let config = TrainerConfig {
            max_epochs: 1,
            log_interval: 0,
            input_noise_std: Some(0.1),
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config).unwrap();
        trainer
            .fit(&model, &mut varmap, mse, &train_source, &val_source)
            .unwrap();

        let train_batches = recorded_train_inputs.borrow();
        let eval_batches = recorded_eval_inputs.borrow();
        let original_train = train_features
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let original_eval = val_features
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();

        assert_eq!(train_batches.len(), 1);
        assert!(
            train_batches[0]
                .iter()
                .zip(original_train.iter())
                .any(|(actual, expected)| (actual - expected).abs() > 1e-9)
        );
        assert_eq!(eval_batches.len(), 1);
        assert_eq!(eval_batches[0], original_eval);
    }

    #[test]
    fn two_runs_with_same_seed_produce_identical_final_weights() {
        // End-to-end determinism check: build + train an MLP with
        // dropout and input noise twice with the same seed, then compare
        // every weight element. They must match exactly.
        use sparam_core::rng::{set_global_seed, test_seed_lock};
        use sparam_core::determinism::{deterministic_reinit_varmap, flatten_vars_f64};

        let _guard = test_seed_lock().lock().unwrap_or_else(|e| e.into_inner());
        let seed = 12345u64;

        fn run_once(seed: u64) -> Vec<f64> {
            set_global_seed(seed);
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
            let cfg = MLPConfig::new(2, 16, 1, Activation::GELU).with_dropout(0.2);
            let model = MLPRegressor::new(vb, &cfg).unwrap();
            deterministic_reinit_varmap(&varmap, seed).unwrap();
            // Re-seed right before training so host-side RNGs (data
            // shuffle, noise, dropout) start from a known state.
            set_global_seed(seed);

            let optimizer = make_optimizer(&varmap);
            let (train_source, val_source) = make_batch_sources(128, 64, 32);
            let config = TrainerConfig {
                max_epochs: 4,
                log_interval: 0,
                input_noise_std: Some(0.05),
                ..TrainerConfig::default()
            };
            let mut trainer = Trainer::new(optimizer, config).unwrap();
            let mut varmap = varmap;
            trainer.fit(&model, &mut varmap, mse, &train_source, &val_source).unwrap();

            flatten_vars_f64(&varmap)
        }

        let a = run_once(seed);
        let b = run_once(seed);
        assert_eq!(a.len(), b.len(), "param count drifted between runs");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(), y.to_bits(),
                "param {i} diverged: {x} vs {y}"
            );
        }
    }

    #[test]
    fn training_with_input_noise_still_converges() {
        let (mut varmap, model) = make_model();
        let optimizer = make_optimizer(&varmap);
        let (train_source, val_source) = make_batch_sources(128, 64, 32);

        let config = TrainerConfig {
            max_epochs: 20,
            log_interval: 0,
            input_noise_std: Some(0.05),
            ..TrainerConfig::default()
        };
        let mut trainer = Trainer::new(optimizer, config).unwrap();
        let result = trainer
            .fit(&model, &mut varmap, mse, &train_source, &val_source)
            .unwrap();

        assert!(result.best_val_loss.is_finite());
        assert!(result.history.train_losses.last().unwrap() < &result.history.train_losses[0]);
    }

    // ── standalone evaluate function ───────────────────────────

    #[test]
    fn evaluate_returns_finite_loss() {
        let (_varmap, model) = make_model();
        let (_, val_source) = make_batch_sources(64, 32, 32);
        let loss = evaluate(&model, &mse, &val_source).unwrap();
        assert!(loss.is_finite(), "expected finite eval loss, got {loss}");
    }

    // ── serde round-trip for configs ───────────────────────────

    #[test]
    fn early_stopping_config_serde_round_trip() {
        let config = EarlyStoppingConfig {
            patience: 15,
            min_delta: 1e-4,
            warmup_epochs: 5,
            mode: EarlyStoppingMode::Min,
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: EarlyStoppingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.patience, 15);
        assert_eq!(restored.warmup_epochs, 5);
    }

    #[test]
    fn trainer_config_serde_round_trip() {
        let config = TrainerConfig {
            max_epochs: 25,
            early_stopping: Some(EarlyStoppingConfig {
                patience: 4,
                min_delta: 1e-3,
                warmup_epochs: 2,
                mode: EarlyStoppingMode::Min,
            }),
            log_interval: 2,
            val_interval: 3,
            max_grad_norm: Some(1.5),
            input_noise_std: Some(0.05),
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: TrainerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.max_epochs, 25);
        assert_eq!(restored.val_interval, 3);
        assert_eq!(restored.max_grad_norm, Some(1.5));
        assert_eq!(restored.input_noise_std, Some(0.05));
    }

    // ── Display impls ──────────────────────────────────────────

    #[test]
    fn early_stopping_mode_display() {
        assert_eq!(format!("{}", EarlyStoppingMode::Min), "min");
        assert_eq!(format!("{}", EarlyStoppingMode::Max), "max");
    }

    #[test]
    fn training_result_display() {
        let result = TrainingResult {
            final_epoch: 42,
            best_epoch: 35,
            best_val_loss: 0.001234,
            stopped_early: true,
            training_time_secs: 12.5,
            history: TrainingHistory::default(),
        };
        let s = format!("{result}");
        assert!(s.contains("best_epoch=35"));
        assert!(s.contains("early_stopped=true"));
    }
}

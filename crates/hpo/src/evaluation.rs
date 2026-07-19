//! Trial evaluation types and traits for HPO.
//!
//! Provides a generic evaluation pipeline:
//!
//! - [`TrialOutcome`] — result of a single trial.
//! - [`TrialMetrics`] — Copy-able metric struct with named fields.
//! - [`ModelBuilder`] — builds a model from [`HyperParams`].
//! - [`Evaluator`] — trains a model and returns [`TrialOutcome`].
//! - [`DataSource`] — provides train/validation data loaders.
//! - [`TrialRunner`] — orchestrates build → evaluate for each trial.

use candle_core::Result;
use std::time::Instant;

use crate::search_space::HyperParams;

// ---------------------------------------------------------------------------
// Outcome & metrics
// ---------------------------------------------------------------------------

/// Status of a completed trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialStatus {
    /// Training completed normally.
    Completed,
    /// Training diverged (NaN / Inf loss) or another recoverable error.
    Failed,
}

/// Fixed-size metrics collected during a trial.
///
/// This is a `Copy` type — no heap allocation.
#[must_use]
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct TrialMetrics {
    /// Best validation loss observed during training.
    #[serde(with = "crate::serde_helpers::special_f64")]
    pub best_val_loss: f64,
    /// Number of trainable parameters in the model.
    pub param_count: usize,
    /// Whether early stopping was triggered.
    pub stopped_early: bool,
    /// Epoch that achieved the best validation loss (1-indexed).
    pub best_epoch: usize,
    /// Total number of epochs that ran.
    pub final_epoch: usize,
    /// Wall-clock training time in seconds.
    pub training_time_secs: f64,
    /// Percentage of validation predictions within 1% relative error.
    #[serde(with = "crate::serde_helpers::special_f64")]
    pub ok_at_1pct: f64,
    /// Maximum validation relative error in percent.
    #[serde(with = "crate::serde_helpers::special_f64")]
    pub max_error: f64,
}

impl Default for TrialMetrics {
    fn default() -> Self {
        Self {
            best_val_loss: f64::INFINITY,
            param_count: 0,
            stopped_early: false,
            best_epoch: 0,
            final_epoch: 0,
            training_time_secs: 0.0,
            ok_at_1pct: f64::NAN,
            max_error: f64::INFINITY,
        }
    }
}

/// Result of a single HPO trial.
#[must_use]
#[derive(Debug, Clone)]
pub struct TrialOutcome {
    /// Objective values (e.g. `[val_loss]` or `[ok_at_1pct, hidden_size]`).
    pub objectives: Vec<f64>,
    /// Detailed metrics for logging / analysis.
    pub metrics: TrialMetrics,
    /// Trial status.
    pub status: TrialStatus,
}

impl TrialOutcome {
    /// Create a single-objective outcome.
    #[inline]
    pub fn single(objective: f64, metrics: TrialMetrics, status: TrialStatus) -> Self {
        Self {
            objectives: vec![objective],
            metrics,
            status,
        }
    }

    /// Create a dual-objective outcome.
    #[inline]
    pub fn dual(obj0: f64, obj1: f64, metrics: TrialMetrics, status: TrialStatus) -> Self {
        Self {
            objectives: vec![obj0, obj1],
            metrics,
            status,
        }
    }

    /// Create a triple-objective outcome.
    #[inline]
    pub fn triple(
        obj0: f64,
        obj1: f64,
        obj2: f64,
        metrics: TrialMetrics,
        status: TrialStatus,
    ) -> Self {
        Self {
            objectives: vec![obj0, obj1, obj2],
            metrics,
            status,
        }
    }

    /// Create the triple-objective outcome
    /// `[OK@1%, param_count, max_error]` used by the NSGA-III
    /// multi-objective study.
    ///
    /// Objective rationale:
    /// - **OK@1%** (maximise) — the accuracy metric we care about.
    /// - **param_count** (minimise) — architectural complexity /
    ///   deployment cost. `param_count` (trained-model weights) is
    ///   used instead of the raw `hidden_size` parameter so Real and
    ///   Complex trials at the same seed + same `hidden_size` report
    ///   genuinely different values (a complex linear layer stores 2×
    ///   the weights), letting NSGA evolve each type independently.
    /// - **max_error** (minimise) — the worst-case regression error on
    ///   the test grid. Previously used as a hard feasibility
    ///   constraint (drop trials with `max_error > threshold`), now an
    ///   objective: a trial with high max_error isn't disqualified, it's
    ///   just dominated on this axis, and the sampler's gradient
    ///   points away from bad-max-error regions naturally.
    #[inline]
    pub fn accuracy_complexity_error(
        ok_at_1pct: f64,
        param_count: usize,
        max_error: f64,
        metrics: TrialMetrics,
        status: TrialStatus,
    ) -> Self {
        Self::triple(ok_at_1pct, param_count as f64, max_error, metrics, status)
    }

    /// Create a failed outcome with sentinel values.
    pub fn failed() -> Self {
        Self {
            objectives: vec![f64::INFINITY],
            metrics: TrialMetrics::default(),
            status: TrialStatus::Failed,
        }
    }

    /// Returns the primary objective when present.
    #[inline]
    pub fn primary_objective(&self) -> Option<f64> {
        self.objectives.first().copied()
    }

    /// Returns the objective pair when this is a two-objective outcome.
    #[inline]
    pub fn objective_pair(&self) -> Option<[f64; 2]> {
        match self.objectives.as_slice() {
            [first, second] => Some([*first, *second]),
            _ => None,
        }
    }

    /// Returns the objective triple when this is a three-objective outcome.
    #[inline]
    pub fn objective_triple(&self) -> Option<[f64; 3]> {
        match self.objectives.as_slice() {
            [a, b, c] => Some([*a, *b, *c]),
            _ => None,
        }
    }

    /// Returns a slice over the active objectives.
    #[inline]
    pub fn objective_values(&self) -> &[f64] {
        &self.objectives
    }
}

// ---------------------------------------------------------------------------
// Traits (static dispatch via generics)
// ---------------------------------------------------------------------------

/// Builds a model from sampled hyperparameters.
///
/// Generic over the model type `M` to allow static dispatch.
pub trait ModelBuilder {
    /// The model type produced by this builder.
    type Model;

    /// Build a new model from the given hyperparameters.
    ///
    /// Returns `(model, varmap)` so the caller owns both.
    fn build(&self, params: &HyperParams) -> Result<(Self::Model, candle_nn::VarMap)>;
}

/// Provides train and validation [`DataLoader`](sparam_data::loader::DataLoader) views.
pub trait DataSource<'a> {
    /// Return a `(train_loader, val_loader)` pair.
    fn loaders(
        &'a self,
        params: &HyperParams,
    ) -> Result<(sparam_data::loader::DataLoader, sparam_data::loader::DataLoader)>;

    /// Optional test loader used to compute Pareto objectives (OK@1%, max_err)
    /// after training. When `None`, the evaluator falls back to the val loader.
    ///
    /// Matches the Python baseline, which evaluates the objective on the
    /// perturbed test grid rather than on validation.
    fn test_loader(
        &'a self,
        _params: &HyperParams,
    ) -> Result<Option<sparam_data::loader::DataLoader>> {
        Ok(None)
    }

    /// Return the target scaler associated with this data source, if any.
    fn target_scaler(&self) -> Option<sparam_data::scaling::ScalerRef<'a>> {
        None
    }

    /// Return the feature scaler associated with this data source, if any.
    /// Required by `LossChoice::PhysicsForward` to recover the batch's raw
    /// S-parameters from the scaled model input.
    fn feature_scaler(&self) -> Option<sparam_data::scaling::ScalerRef<'a>> {
        None
    }
}

/// Trains a model and returns a [`TrialOutcome`].
pub trait Evaluator<M> {
    /// Run one full training loop for the given model.
    fn evaluate(
        &self,
        model: &M,
        varmap: &mut candle_nn::VarMap,
        params: &HyperParams,
    ) -> Result<TrialOutcome>;
}

// ---------------------------------------------------------------------------
// TrialRunner — orchestrates build → data → evaluate
// ---------------------------------------------------------------------------

/// Generic trial runner that wires together a builder and evaluator
/// without any dynamic dispatch.
pub struct TrialRunner<'a, B, E> {
    builder: &'a B,
    evaluator: &'a E,
    log: sparam_training::logger::LogSender,
}

impl<'a, B, E> TrialRunner<'a, B, E> {
    pub fn new(builder: &'a B, evaluator: &'a E) -> Self {
        Self {
            builder,
            evaluator,
            log: sparam_training::logger::LogSender::null(),
        }
    }

    /// Attach a logger for trial error messages.
    pub fn with_log(mut self, log: sparam_training::logger::LogSender) -> Self {
        self.log = log;
        self
    }
}

impl<'a, B, E> TrialRunner<'a, B, E>
where
    B: ModelBuilder,
    E: Evaluator<B::Model>,
{
    /// Execute a single trial: build model → evaluate.
    pub fn run(&self, params: &HyperParams) -> TrialOutcome {
        let start = Instant::now();
        match self.run_inner(params) {
            Ok(mut outcome) => {
                // Ensure wall-clock time is captured even if the evaluator
                // already set it.
                if outcome.metrics.training_time_secs == 0.0 {
                    outcome.metrics.training_time_secs = start.elapsed().as_secs_f64();
                }
                outcome
            }
            Err(e) => {
                self.log.send(sparam_training::logger::LogMessage::Warning(
                    format!("Trial failed: {e}")
                ));
                TrialOutcome::failed()
            }
        }
    }

    fn run_inner(&self, params: &HyperParams) -> Result<TrialOutcome> {
        let (model, mut varmap) = self.builder.build(params)?;
        let seed = sparam_core::rng::get_global_seed();
        sparam_core::determinism::deterministic_reinit_varmap(&varmap, seed)?;
        sparam_core::determinism::emit_varmap_hash(
            format_args!("hpo-trial post-reinit (seed={seed})"),
            &varmap,
        );
        self.evaluator.evaluate(&model, &mut varmap, params)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trial_outcome_single_has_one_objective() {
        let m = TrialMetrics {
            best_val_loss: 0.01,
            param_count: 100,
            stopped_early: false,
            best_epoch: 5,
            final_epoch: 10,
            training_time_secs: 1.5,
            ..TrialMetrics::default()
        };
        let o = TrialOutcome::single(0.01, m, TrialStatus::Completed);
        assert_eq!(o.objective_values().len(), 1);
        assert_eq!(o.objective_values(), &[0.01]);
    }

    #[test]
    fn trial_outcome_dual_has_two_objectives() {
        let m = TrialMetrics {
            best_val_loss: 0.02,
            param_count: 200,
            stopped_early: true,
            best_epoch: 8,
            final_epoch: 20,
            training_time_secs: 3.0,
            ..TrialMetrics::default()
        };
        let o = TrialOutcome::dual(0.02, 0.5, m, TrialStatus::Completed);
        assert_eq!(o.objective_values().len(), 2);
        assert_eq!(o.objective_values(), &[0.02, 0.5]);
        assert_eq!(o.objective_pair(), Some([0.02, 0.5]));
    }

    #[test]
    fn trial_outcome_accuracy_complexity_error_has_three_objectives() {
        // 3-objective layout: [OK@1%, param_count, max_error].
        // param_count (trained weights) instead of sampled hidden_size
        // so Real/Complex diverge at the same seed; max_error is a
        // first-class objective (was a feasibility constraint).
        let metrics = TrialMetrics {
            ok_at_1pct: 97.2,
            max_error: 3.5,
            param_count: 1234,
            ..TrialMetrics::default()
        };

        let outcome = TrialOutcome::accuracy_complexity_error(
            metrics.ok_at_1pct,
            metrics.param_count,
            metrics.max_error,
            metrics,
            TrialStatus::Completed,
        );

        assert_eq!(outcome.objective_values(), &[97.2, 1234.0, 3.5]);
        assert_eq!(outcome.objective_triple(), Some([97.2, 1234.0, 3.5]));
    }

    #[test]
    fn trial_outcome_failed_has_sentinel_values() {
        let o = TrialOutcome::failed();
        assert_eq!(o.status, TrialStatus::Failed);
        assert!(o.metrics.best_val_loss.is_infinite());
        assert!(o.primary_objective().unwrap().is_infinite());
    }

    #[test]
    fn trial_metrics_is_copy() {
        let m = TrialMetrics {
            best_val_loss: 0.05,
            param_count: 50,
            stopped_early: false,
            best_epoch: 3,
            final_epoch: 10,
            training_time_secs: 0.5,
            ..TrialMetrics::default()
        };
        let m2 = m; // Copy
        assert_eq!(m.best_val_loss, m2.best_val_loss);
    }
}

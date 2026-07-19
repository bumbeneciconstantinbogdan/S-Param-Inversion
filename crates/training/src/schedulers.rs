//! Learning-rate schedulers for the unified training surface.
//!
//! Candle exposes optimizers but no built-in scheduler abstractions, so the
//! training module provides lightweight scheduler implementations that operate
//! on top of [`OptimizerKind`].

use std::{f64::consts::PI, fmt};

use candle_core::Result;
use serde::{Deserialize, Serialize};

use sparam_core::error::candle_msg;

use crate::optimizers::OptimizerKind;

/// Configuration for cosine annealing learning-rate scheduling.
///
/// The schedule follows the closed-form PyTorch reference:
///
/// `eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(t * π / t_max))`
///
/// where `eta_max` is the optimizer learning rate when the scheduler is
/// created. The cosine continues beyond `t_max` rather than clamping, matching
/// the closed-form behavior of PyTorch's `CosineAnnealingLR`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CosineAnnealingConfig {
    /// Number of steps in a half cosine cycle.
    pub t_max: usize,
    /// Minimum learning rate.
    pub eta_min: f64,
}

impl fmt::Display for CosineAnnealingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CosineAnnealingLR(t_max={}, eta_min={})",
            self.t_max, self.eta_min
        )
    }
}

/// Cosine annealing learning-rate scheduler.
///
/// The scheduler captures the optimizer's initial learning rate as `eta_max`
/// and advances one step at a time via `step()` or `step_optimizer()`. Step 0
/// retains the initial learning rate; the first call to `step()` advances to
/// step 1.
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    config: CosineAnnealingConfig,
    eta_max: f64,
    current_step: usize,
}

impl CosineAnnealingLR {
    /// Create a scheduler from an explicit starting learning rate.
    pub fn new(eta_max: f64, config: CosineAnnealingConfig) -> Result<Self> {
        validate_t_max(config.t_max)?;
        validate_eta(eta_max, "CosineAnnealingLR: eta_max")?;
        validate_eta(config.eta_min, "CosineAnnealingLR: eta_min")?;
        if config.eta_min > eta_max {
            return Err(candle_msg(format!(
                "CosineAnnealingLR: eta_min ({}) must be <= eta_max ({eta_max})",
                config.eta_min
            )));
        }
        Ok(Self {
            config,
            eta_max,
            current_step: 0,
        })
    }

    /// Create a scheduler from the optimizer's current learning rate.
    pub fn from_optimizer(
        optimizer: &OptimizerKind,
        config: CosineAnnealingConfig,
    ) -> Result<Self> {
        Self::new(optimizer.learning_rate(), config)
    }

    /// Return the configured half-cycle length.
    #[must_use]
    pub fn t_max(&self) -> usize {
        self.config.t_max
    }

    /// Return the configured minimum learning rate.
    #[must_use]
    pub fn eta_min(&self) -> f64 {
        self.config.eta_min
    }

    /// Return the captured base learning rate.
    #[must_use]
    pub fn eta_max(&self) -> f64 {
        self.eta_max
    }

    /// Return the current scheduler step.
    #[must_use]
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Return the current learning rate without advancing the scheduler.
    #[inline]
    #[must_use]
    pub fn current_lr(&self) -> f64 {
        self.lr_at(self.current_step)
    }

    /// Compute the scheduled learning rate at an arbitrary step.
    #[inline]
    #[must_use]
    pub fn lr_at(&self, step: usize) -> f64 {
        let cosine = ((step as f64) * PI / (self.config.t_max as f64)).cos();
        self.config.eta_min + 0.5 * (self.eta_max - self.config.eta_min) * (1.0 + cosine)
    }

    /// Advance the scheduler by one step and return the new learning rate.
    #[inline]
    pub fn step(&mut self) -> f64 {
        self.current_step = self.current_step.saturating_add(1);
        self.current_lr()
    }

    /// Advance the scheduler and apply the new learning rate to the optimizer.
    #[inline]
    pub fn step_optimizer(&mut self, optimizer: &mut OptimizerKind) -> f64 {
        let lr = self.step();
        optimizer.set_learning_rate(lr);
        lr
    }

    /// Reset the scheduler back to step 0.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

impl fmt::Display for CosineAnnealingLR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CosineAnnealingLR(step={}, eta_max={}, eta_min={}, t_max={})",
            self.current_step, self.eta_max, self.config.eta_min, self.config.t_max
        )
    }
}

/// Whether plateau detection should minimize or maximize the monitored metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ReduceOnPlateauMode {
    /// Lower metric values are better.
    #[default]
    #[serde(rename = "min")]
    Min,
    /// Higher metric values are better.
    #[serde(rename = "max")]
    Max,
}

impl fmt::Display for ReduceOnPlateauMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Min => write!(f, "min"),
            Self::Max => write!(f, "max"),
        }
    }
}

/// How the improvement threshold should be interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ReduceOnPlateauThresholdMode {
    /// Compare against the best metric using a relative delta.
    #[default]
    #[serde(rename = "rel")]
    Rel,
    /// Compare against the best metric using an absolute delta.
    #[serde(rename = "abs")]
    Abs,
}

impl fmt::Display for ReduceOnPlateauThresholdMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rel => write!(f, "rel"),
            Self::Abs => write!(f, "abs"),
        }
    }
}

/// Configuration for a ReduceLROnPlateau-style scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReduceOnPlateauConfig {
    /// Whether lower or higher metrics are better.
    pub mode: ReduceOnPlateauMode,
    /// Multiplicative factor applied when the learning rate is reduced.
    pub factor: f64,
    /// Number of consecutive non-improving metric updates tolerated before reducing.
    pub patience: usize,
    /// Minimum delta required to qualify as an improvement.
    pub threshold: f64,
    /// Interpret `threshold` as relative or absolute.
    pub threshold_mode: ReduceOnPlateauThresholdMode,
    /// Number of metric updates to ignore after a reduction.
    pub cooldown: usize,
    /// Lower floor for the learning rate.
    pub min_lr: f64,
    /// Minimum learning-rate change required to apply a reduction.
    pub eps: f64,
}

impl Default for ReduceOnPlateauConfig {
    fn default() -> Self {
        Self {
            mode: ReduceOnPlateauMode::Min,
            factor: 0.1,
            patience: 10,
            threshold: 1e-4,
            threshold_mode: ReduceOnPlateauThresholdMode::Rel,
            cooldown: 0,
            min_lr: 0.0,
            eps: 1e-8,
        }
    }
}

impl fmt::Display for ReduceOnPlateauConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ReduceOnPlateau(mode={}, factor={}, patience={}, threshold={}, threshold_mode={}, cooldown={}, min_lr={}, eps={})",
            self.mode,
            self.factor,
            self.patience,
            self.threshold,
            self.threshold_mode,
            self.cooldown,
            self.min_lr,
            self.eps
        )
    }
}

/// Reactive learning-rate scheduler that reduces LR when a monitored metric stalls.
#[derive(Debug, Clone)]
pub struct ReduceOnPlateauLR {
    config: ReduceOnPlateauConfig,
    initial_lr: f64,
    current_lr: f64,
    best: Option<f64>,
    bad_steps: usize,
    cooldown_counter: usize,
}

impl ReduceOnPlateauLR {
    /// Create a scheduler from an explicit initial learning rate.
    pub fn new(initial_lr: f64, config: ReduceOnPlateauConfig) -> Result<Self> {
        validate_eta(initial_lr, "ReduceOnPlateauLR: initial_lr")?;
        validate_factor(config.factor)?;
        validate_patience(config.patience)?;
        validate_eta(config.threshold, "ReduceOnPlateauLR: threshold")?;
        validate_eta(config.min_lr, "ReduceOnPlateauLR: min_lr")?;
        validate_eta(config.eps, "ReduceOnPlateauLR: eps")?;
        if config.min_lr > initial_lr {
            return Err(candle_msg(format!(
                "ReduceOnPlateauLR: min_lr ({}) must be <= initial_lr ({initial_lr})",
                config.min_lr
            )));
        }
        Ok(Self {
            config,
            initial_lr,
            current_lr: initial_lr,
            best: None,
            bad_steps: 0,
            cooldown_counter: 0,
        })
    }

    /// Create a scheduler from the optimizer's current learning rate.
    pub fn from_optimizer(
        optimizer: &OptimizerKind,
        config: ReduceOnPlateauConfig,
    ) -> Result<Self> {
        Self::new(optimizer.learning_rate(), config)
    }

    /// Return the current learning rate tracked by the scheduler.
    #[must_use]
    pub fn current_lr(&self) -> f64 {
        self.current_lr
    }

    /// Return the best metric seen so far, if any.
    #[must_use]
    pub fn best(&self) -> Option<f64> {
        self.best
    }

    /// Return the number of consecutive non-improving metric updates.
    #[must_use]
    pub fn bad_steps(&self) -> usize {
        self.bad_steps
    }

    /// Return the remaining cooldown steps.
    #[must_use]
    pub fn cooldown_counter(&self) -> usize {
        self.cooldown_counter
    }

    /// Return whether the scheduler is currently in cooldown.
    #[must_use]
    pub fn in_cooldown(&self) -> bool {
        self.cooldown_counter > 0
    }

    /// Update the scheduler state from a monitored metric and return the active LR.
    pub fn step(&mut self, metric: f64) -> Result<f64> {
        validate_finite_metric(metric)?;

        if self.in_cooldown() {
            self.cooldown_counter = self.cooldown_counter.saturating_sub(1);
            if self.best.is_none_or(|best| self.is_better(metric, best)) {
                self.best = Some(metric);
            }
            return Ok(self.current_lr);
        }

        let is_better = self.best.is_none_or(|best| self.is_better(metric, best));

        if is_better {
            self.best = Some(metric);
            self.bad_steps = 0;
        } else {
            self.bad_steps = self.bad_steps.saturating_add(1);
        }

        if self.bad_steps >= self.config.patience {
            self.maybe_reduce_lr();
            self.bad_steps = 0;
        }

        Ok(self.current_lr)
    }

    /// Update the scheduler and immediately apply the resulting LR to an optimizer.
    pub fn step_optimizer(&mut self, metric: f64, optimizer: &mut OptimizerKind) -> Result<f64> {
        let lr = self.step(metric)?;
        optimizer.set_learning_rate(lr);
        Ok(lr)
    }

    /// Reset the scheduler to its initial state.
    pub fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.best = None;
        self.bad_steps = 0;
        self.cooldown_counter = 0;
    }

    #[inline]
    fn is_better(&self, metric: f64, best: f64) -> bool {
        match (self.config.mode, self.config.threshold_mode) {
            (ReduceOnPlateauMode::Min, ReduceOnPlateauThresholdMode::Rel) => {
                metric < best * (1.0 - self.config.threshold)
            }
            (ReduceOnPlateauMode::Min, ReduceOnPlateauThresholdMode::Abs) => {
                metric < best - self.config.threshold
            }
            (ReduceOnPlateauMode::Max, ReduceOnPlateauThresholdMode::Rel) => {
                metric > best * (1.0 + self.config.threshold)
            }
            (ReduceOnPlateauMode::Max, ReduceOnPlateauThresholdMode::Abs) => {
                metric > best + self.config.threshold
            }
        }
    }

    #[inline]
    fn maybe_reduce_lr(&mut self) {
        let new_lr = (self.current_lr * self.config.factor).max(self.config.min_lr);
        if self.current_lr - new_lr > self.config.eps {
            self.current_lr = new_lr;
            self.cooldown_counter = self.config.cooldown;
        }
    }
}

impl fmt::Display for ReduceOnPlateauLR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ReduceOnPlateauLR(lr={}, best={:?}, bad_steps={}, cooldown_counter={})",
            self.current_lr, self.best, self.bad_steps, self.cooldown_counter
        )
    }
}

/// Unified scheduler enum for training loops that need to hold different
/// scheduler strategies behind a single type.
#[derive(Debug, Clone)]
pub enum LRScheduler {
    /// Deterministic cosine annealing scheduler.
    CosineAnnealing(CosineAnnealingLR),
    /// Reactive metric-driven plateau scheduler.
    ReduceOnPlateau(ReduceOnPlateauLR),
}

/// Tagged scheduler configuration for config-driven trainer construction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "params", rename_all = "snake_case")]
pub enum SchedulerConfig {
    None,
    CosineAnnealing(CosineAnnealingConfig),
    ReduceOnPlateau(ReduceOnPlateauConfig),
}

impl SchedulerConfig {
    #[must_use]
    pub fn none() -> Self {
        Self::None
    }

    #[must_use]
    pub fn cosine(t_max: usize, eta_min: f64) -> Self {
        Self::CosineAnnealing(CosineAnnealingConfig { t_max, eta_min })
    }

    #[must_use]
    pub fn plateau(config: ReduceOnPlateauConfig) -> Self {
        Self::ReduceOnPlateau(config)
    }

    #[must_use]
    pub fn plateau_with(factor: f64, patience: usize, min_lr: f64) -> Self {
        Self::ReduceOnPlateau(ReduceOnPlateauConfig {
            factor,
            patience,
            min_lr,
            ..Default::default()
        })
    }

    pub fn from_name(name: &str, max_epochs: usize) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "none" => Ok(Self::none()),
            "cosine" => Ok(Self::cosine(max_epochs, 1e-6)),
            "plateau" => Ok(Self::plateau(ReduceOnPlateauConfig::default())),
            other => Err(candle_msg(format!(
                "unknown scheduler '{other}', expected: none, cosine, plateau"
            ))),
        }
    }

    pub fn build(self, initial_lr: f64) -> Result<Option<LRScheduler>> {
        match self {
            Self::None => Ok(None),
            Self::CosineAnnealing(config) => Ok(Some(LRScheduler::CosineAnnealing(
                CosineAnnealingLR::new(initial_lr, config)?,
            ))),
            Self::ReduceOnPlateau(config) => Ok(Some(LRScheduler::ReduceOnPlateau(
                ReduceOnPlateauLR::new(initial_lr, config)?,
            ))),
        }
    }
}

impl fmt::Display for SchedulerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::CosineAnnealing(config) => write!(f, "{config}"),
            Self::ReduceOnPlateau(config) => write!(f, "{config}"),
        }
    }
}

impl LRScheduler {
    /// Return the current learning rate tracked by the active scheduler.
    #[must_use]
    pub fn current_lr(&self) -> f64 {
        match self {
            Self::CosineAnnealing(scheduler) => scheduler.current_lr(),
            Self::ReduceOnPlateau(scheduler) => scheduler.current_lr(),
        }
    }

    /// Return whether the active scheduler requires a monitored metric.
    #[must_use]
    pub fn requires_metric(&self) -> bool {
        matches!(self, Self::ReduceOnPlateau(_))
    }

    /// Human-readable name of the active scheduler variant.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::CosineAnnealing(_) => "CosineAnnealing",
            Self::ReduceOnPlateau(_) => "ReduceOnPlateau",
        }
    }

    /// Advance a scheduler that does not require a metric.
    pub fn step(&mut self) -> Result<f64> {
        match self {
            Self::CosineAnnealing(scheduler) => Ok(scheduler.step()),
            Self::ReduceOnPlateau(_) => Err(candle_msg(
                "LRScheduler::ReduceOnPlateau requires a metric; use step_with_metric",
            )),
        }
    }

    /// Advance the scheduler using a monitored metric when needed.
    ///
    /// For `CosineAnnealing`, the metric is ignored (schedule is deterministic).
    /// For `ReduceOnPlateau`, the metric drives the reduction decision.
    pub fn step_with_metric(&mut self, metric: f64) -> Result<f64> {
        match self {
            Self::CosineAnnealing(scheduler) => {
                let _ = metric;
                Ok(scheduler.step())
            }
            Self::ReduceOnPlateau(scheduler) => scheduler.step(metric),
        }
    }

    /// Advance a scheduler that does not require a metric and apply the new LR.
    pub fn step_optimizer(&mut self, optimizer: &mut OptimizerKind) -> Result<f64> {
        match self {
            Self::CosineAnnealing(scheduler) => Ok(scheduler.step_optimizer(optimizer)),
            Self::ReduceOnPlateau(_) => Err(candle_msg(
                "LRScheduler::ReduceOnPlateau requires a metric; use step_optimizer_with_metric",
            )),
        }
    }

    /// Advance the scheduler, passing a monitored metric when needed, and apply the new LR.
    pub fn step_optimizer_with_metric(
        &mut self,
        metric: f64,
        optimizer: &mut OptimizerKind,
    ) -> Result<f64> {
        match self {
            Self::CosineAnnealing(scheduler) => {
                let _ = metric;
                Ok(scheduler.step_optimizer(optimizer))
            }
            Self::ReduceOnPlateau(scheduler) => scheduler.step_optimizer(metric, optimizer),
        }
    }

    /// Reset the active scheduler to its initial state.
    pub fn reset(&mut self) {
        match self {
            Self::CosineAnnealing(scheduler) => scheduler.reset(),
            Self::ReduceOnPlateau(scheduler) => scheduler.reset(),
        }
    }
}

impl fmt::Display for LRScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CosineAnnealing(scheduler) => write!(f, "{scheduler}"),
            Self::ReduceOnPlateau(scheduler) => write!(f, "{scheduler}"),
        }
    }
}

fn validate_t_max(t_max: usize) -> Result<()> {
    sparam_core::validation::validate_positive_usize("CosineAnnealingLR: t_max", t_max)?;
    Ok(())
}

fn validate_eta(value: f64, name: &str) -> Result<()> {
    sparam_core::validation::validate_non_negative_f64(name, value)?;
    Ok(())
}

fn validate_factor(factor: f64) -> Result<()> {
    if !factor.is_finite() || factor <= 0.0 || factor >= 1.0 {
        return Err(candle_msg(format!(
            "ReduceOnPlateauLR: factor must be finite and in (0, 1), got {factor}"
        )));
    }
    Ok(())
}

fn validate_patience(patience: usize) -> Result<()> {
    sparam_core::validation::validate_positive_usize("ReduceOnPlateauLR: patience", patience)?;
    Ok(())
}

fn validate_finite_metric(metric: f64) -> Result<()> {
    if !metric.is_finite() {
        return Err(candle_msg(format!(
            "ReduceOnPlateauLR: metric must be finite, got {metric}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::{AdamConfig, OptimizerKind};

    fn assert_close(actual: f64, expected: f64) {
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-12,
            "expected {expected}, got {actual} (diff={diff})"
        );
    }

    #[test]
    fn config_display_and_serde() {
        let config = CosineAnnealingConfig {
            t_max: 10,
            eta_min: 1e-4,
        };
        assert!(config.to_string().contains("CosineAnnealingLR"));

        let json = serde_json::to_string(&config).unwrap();
        let round_trip: CosineAnnealingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip, config);
    }

    #[test]
    fn scheduler_config_from_name_parses_supported_values() {
        assert_eq!(
            SchedulerConfig::from_name("none", 10).unwrap(),
            SchedulerConfig::None
        );
        assert_eq!(
            SchedulerConfig::from_name("cosine", 25).unwrap(),
            SchedulerConfig::CosineAnnealing(CosineAnnealingConfig {
                t_max: 25,
                eta_min: 1e-6,
            })
        );
        assert!(matches!(
            SchedulerConfig::from_name("plateau", 25).unwrap(),
            SchedulerConfig::ReduceOnPlateau(_)
        ));
    }

    #[test]
    fn scheduler_config_from_name_rejects_unknown_values() {
        let error = SchedulerConfig::from_name("step", 10)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown scheduler"));
    }

    #[test]
    fn scheduler_config_builds_expected_variant() {
        assert!(SchedulerConfig::none().build(1e-3).unwrap().is_none());

        assert!(matches!(
            SchedulerConfig::cosine(20, 1e-6).build(1e-3).unwrap(),
            Some(LRScheduler::CosineAnnealing(_))
        ));
        assert!(matches!(
            SchedulerConfig::plateau_with(0.5, 5, 1e-6)
                .build(1e-3)
                .unwrap(),
            Some(LRScheduler::ReduceOnPlateau(_))
        ));
    }

    #[test]
    fn rejects_invalid_config() {
        assert!(
            CosineAnnealingLR::new(
                1e-3,
                CosineAnnealingConfig {
                    t_max: 0,
                    eta_min: 0.0,
                }
            )
            .is_err()
        );

        assert!(
            CosineAnnealingLR::new(
                1e-3,
                CosineAnnealingConfig {
                    t_max: 10,
                    eta_min: 2e-3,
                }
            )
            .is_err()
        );
    }

    #[test]
    fn matches_reference_points() {
        let scheduler = CosineAnnealingLR::new(
            1.0,
            CosineAnnealingConfig {
                t_max: 10,
                eta_min: 0.1,
            },
        )
        .unwrap();

        assert_close(scheduler.current_lr(), 1.0);
        assert_close(scheduler.lr_at(5), 0.55);
        assert_close(scheduler.lr_at(10), 0.1);
    }

    #[test]
    fn continues_beyond_t_max() {
        let scheduler = CosineAnnealingLR::new(
            1.0,
            CosineAnnealingConfig {
                t_max: 4,
                eta_min: 0.0,
            },
        )
        .unwrap();

        assert_close(scheduler.lr_at(0), 1.0);
        assert_close(scheduler.lr_at(4), 0.0);
        assert_close(scheduler.lr_at(8), 1.0);
        assert!(scheduler.lr_at(5) > 0.0);
    }

    #[test]
    fn step_advances_scheduler() {
        let mut scheduler = CosineAnnealingLR::new(
            1.0,
            CosineAnnealingConfig {
                t_max: 4,
                eta_min: 0.0,
            },
        )
        .unwrap();

        assert_eq!(scheduler.current_step(), 0);
        assert_close(scheduler.current_lr(), 1.0);

        let lr_1 = scheduler.step();
        assert_eq!(scheduler.current_step(), 1);
        assert_close(lr_1, 0.8535533905932737);

        let lr_2 = scheduler.step();
        assert_eq!(scheduler.current_step(), 2);
        assert_close(lr_2, 0.5);
    }

    #[test]
    fn step_optimizer_updates_optimizer_lr() {
        let mut optimizer = OptimizerKind::adam(
            vec![],
            AdamConfig {
                lr: 1.0,
                ..Default::default()
            },
        )
        .unwrap();
        let mut scheduler = CosineAnnealingLR::from_optimizer(
            &optimizer,
            CosineAnnealingConfig {
                t_max: 4,
                eta_min: 0.0,
            },
        )
        .unwrap();

        assert_close(optimizer.learning_rate(), 1.0);
        let lr = scheduler.step_optimizer(&mut optimizer);
        assert_close(lr, optimizer.learning_rate());
        assert_close(lr, 0.8535533905932737);
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut scheduler = CosineAnnealingLR::new(
            1.0,
            CosineAnnealingConfig {
                t_max: 4,
                eta_min: 0.0,
            },
        )
        .unwrap();

        scheduler.step();
        scheduler.step();
        scheduler.reset();

        assert_eq!(scheduler.current_step(), 0);
        assert_close(scheduler.current_lr(), 1.0);
    }

    #[test]
    fn reduce_on_plateau_config_defaults_and_serde() {
        let config = ReduceOnPlateauConfig::default();
        assert_eq!(config.mode, ReduceOnPlateauMode::Min);
        assert_eq!(config.factor, 0.1);
        assert_eq!(config.patience, 10);
        assert_eq!(config.threshold_mode, ReduceOnPlateauThresholdMode::Rel);
        assert!(config.to_string().contains("ReduceOnPlateau"));

        let json = serde_json::to_string(&config).unwrap();
        let round_trip: ReduceOnPlateauConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip, config);
    }

    #[test]
    fn reduce_on_plateau_mode_and_threshold_mode_serde() {
        let mode_json = serde_json::to_string(&ReduceOnPlateauMode::Max).unwrap();
        assert_eq!(mode_json, "\"max\"");
        let threshold_json = serde_json::to_string(&ReduceOnPlateauThresholdMode::Abs).unwrap();
        assert_eq!(threshold_json, "\"abs\"");
    }

    #[test]
    fn reduce_on_plateau_rejects_invalid_config() {
        assert!(
            ReduceOnPlateauLR::new(
                1.0,
                ReduceOnPlateauConfig {
                    factor: 1.0,
                    ..Default::default()
                },
            )
            .is_err()
        );

        assert!(
            ReduceOnPlateauLR::new(
                1.0,
                ReduceOnPlateauConfig {
                    patience: 0,
                    ..Default::default()
                },
            )
            .is_err()
        );

        assert!(
            ReduceOnPlateauLR::new(
                1.0,
                ReduceOnPlateauConfig {
                    min_lr: 2.0,
                    ..Default::default()
                },
            )
            .is_err()
        );
    }

    #[test]
    fn reduce_on_plateau_min_mode_reduces_after_patience() {
        let mut scheduler = ReduceOnPlateauLR::new(
            1.0,
            ReduceOnPlateauConfig {
                factor: 0.5,
                patience: 2,
                threshold: 0.0,
                eps: 0.0,
                ..Default::default()
            },
        )
        .unwrap();

        assert_close(scheduler.step(1.0).unwrap(), 1.0);
        assert_eq!(scheduler.best(), Some(1.0));
        assert_eq!(scheduler.bad_steps(), 0);

        assert_close(scheduler.step(1.1).unwrap(), 1.0);
        assert_eq!(scheduler.bad_steps(), 1);

        assert_close(scheduler.step(1.2).unwrap(), 0.5);
        assert_eq!(scheduler.bad_steps(), 0);
    }

    #[test]
    fn reduce_on_plateau_relative_threshold_requires_meaningful_improvement() {
        let mut scheduler = ReduceOnPlateauLR::new(
            1.0,
            ReduceOnPlateauConfig {
                threshold: 0.1,
                threshold_mode: ReduceOnPlateauThresholdMode::Rel,
                patience: 2,
                ..Default::default()
            },
        )
        .unwrap();

        scheduler.step(1.0).unwrap();
        scheduler.step(0.95).unwrap();
        assert_eq!(scheduler.best(), Some(1.0));
        assert_eq!(scheduler.bad_steps(), 1);

        scheduler.step(0.89).unwrap();
        assert_eq!(scheduler.best(), Some(0.89));
        assert_eq!(scheduler.bad_steps(), 0);
    }

    #[test]
    fn reduce_on_plateau_max_mode_absolute_threshold() {
        let mut scheduler = ReduceOnPlateauLR::new(
            1.0,
            ReduceOnPlateauConfig {
                mode: ReduceOnPlateauMode::Max,
                threshold: 0.1,
                threshold_mode: ReduceOnPlateauThresholdMode::Abs,
                patience: 2,
                ..Default::default()
            },
        )
        .unwrap();

        scheduler.step(1.0).unwrap();
        scheduler.step(1.05).unwrap();
        assert_eq!(scheduler.best(), Some(1.0));
        assert_eq!(scheduler.bad_steps(), 1);

        scheduler.step(1.11).unwrap();
        assert_eq!(scheduler.best(), Some(1.11));
        assert_eq!(scheduler.bad_steps(), 0);
    }

    #[test]
    fn reduce_on_plateau_cooldown_prevents_immediate_reductions() {
        let mut scheduler = ReduceOnPlateauLR::new(
            1.0,
            ReduceOnPlateauConfig {
                factor: 0.5,
                patience: 1,
                cooldown: 2,
                threshold: 0.0,
                eps: 0.0,
                ..Default::default()
            },
        )
        .unwrap();

        scheduler.step(1.0).unwrap();
        assert_close(scheduler.step(1.1).unwrap(), 0.5);
        assert_eq!(scheduler.cooldown_counter(), 2);

        assert_close(scheduler.step(1.2).unwrap(), 0.5);
        assert!(scheduler.in_cooldown());
        assert_eq!(scheduler.cooldown_counter(), 1);

        assert_close(scheduler.step(1.3).unwrap(), 0.5);
        assert!(!scheduler.in_cooldown());
        assert_eq!(scheduler.cooldown_counter(), 0);

        assert_close(scheduler.step(1.4).unwrap(), 0.25);
    }

    #[test]
    fn reduce_on_plateau_respects_min_lr_floor() {
        let mut scheduler = ReduceOnPlateauLR::new(
            1.0,
            ReduceOnPlateauConfig {
                factor: 0.5,
                patience: 1,
                min_lr: 0.3,
                threshold: 0.0,
                eps: 0.0,
                ..Default::default()
            },
        )
        .unwrap();

        scheduler.step(1.0).unwrap();
        assert_close(scheduler.step(1.1).unwrap(), 0.5);
        assert_close(scheduler.step(1.2).unwrap(), 0.3);
        assert_close(scheduler.step(1.3).unwrap(), 0.3);
    }

    #[test]
    fn reduce_on_plateau_eps_prevents_tiny_reductions() {
        let mut scheduler = ReduceOnPlateauLR::new(
            0.4,
            ReduceOnPlateauConfig {
                factor: 0.95,
                patience: 1,
                threshold: 0.0,
                eps: 0.05,
                ..Default::default()
            },
        )
        .unwrap();

        scheduler.step(1.0).unwrap();
        assert_close(scheduler.step(1.1).unwrap(), 0.4);
    }

    #[test]
    fn reduce_on_plateau_step_optimizer_updates_optimizer_lr() {
        let mut optimizer = OptimizerKind::adam(
            vec![],
            AdamConfig {
                lr: 1.0,
                ..Default::default()
            },
        )
        .unwrap();
        let mut scheduler = ReduceOnPlateauLR::from_optimizer(
            &optimizer,
            ReduceOnPlateauConfig {
                factor: 0.5,
                patience: 2,
                threshold: 0.0,
                eps: 0.0,
                ..Default::default()
            },
        )
        .unwrap();

        scheduler.step_optimizer(1.0, &mut optimizer).unwrap();
        scheduler.step_optimizer(1.1, &mut optimizer).unwrap();
        let lr = scheduler.step_optimizer(1.2, &mut optimizer).unwrap();
        assert_close(lr, 0.5);
        assert_close(optimizer.learning_rate(), 0.5);
    }

    #[test]
    fn reduce_on_plateau_reset_restores_initial_state() {
        let mut scheduler = ReduceOnPlateauLR::new(
            1.0,
            ReduceOnPlateauConfig {
                factor: 0.5,
                patience: 1,
                threshold: 0.0,
                eps: 0.0,
                ..Default::default()
            },
        )
        .unwrap();

        scheduler.step(1.0).unwrap();
        scheduler.step(1.1).unwrap();
        scheduler.reset();

        assert_close(scheduler.current_lr(), 1.0);
        assert_eq!(scheduler.best(), None);
        assert_eq!(scheduler.bad_steps(), 0);
        assert_eq!(scheduler.cooldown_counter(), 0);
    }

    #[test]
    fn reduce_on_plateau_rejects_non_finite_metric() {
        let mut scheduler = ReduceOnPlateauLR::new(1.0, ReduceOnPlateauConfig::default()).unwrap();
        assert!(scheduler.step(f64::NAN).is_err());
    }

    #[test]
    fn unified_scheduler_name_and_metric_requirement() {
        let cosine = LRScheduler::CosineAnnealing(
            CosineAnnealingLR::new(
                1.0,
                CosineAnnealingConfig {
                    t_max: 4,
                    eta_min: 0.0,
                },
            )
            .unwrap(),
        );
        let plateau = LRScheduler::ReduceOnPlateau(
            ReduceOnPlateauLR::new(1.0, ReduceOnPlateauConfig::default()).unwrap(),
        );

        assert_eq!(cosine.name(), "CosineAnnealing");
        assert!(!cosine.requires_metric());
        assert_eq!(plateau.name(), "ReduceOnPlateau");
        assert!(plateau.requires_metric());
    }

    #[test]
    fn unified_scheduler_cosine_step_and_reset() {
        let mut scheduler = LRScheduler::CosineAnnealing(
            CosineAnnealingLR::new(
                1.0,
                CosineAnnealingConfig {
                    t_max: 4,
                    eta_min: 0.0,
                },
            )
            .unwrap(),
        );

        assert_close(scheduler.current_lr(), 1.0);
        assert_close(scheduler.step().unwrap(), 0.8535533905932737);
        assert_close(scheduler.step_with_metric(42.0).unwrap(), 0.5);
        scheduler.reset();
        assert_close(scheduler.current_lr(), 1.0);
    }

    #[test]
    fn unified_scheduler_plateau_requires_metric_for_plain_step() {
        let mut scheduler = LRScheduler::ReduceOnPlateau(
            ReduceOnPlateauLR::new(1.0, ReduceOnPlateauConfig::default()).unwrap(),
        );
        let mut optimizer = OptimizerKind::adam(
            vec![],
            AdamConfig {
                lr: 1.0,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(scheduler.step().is_err());
        assert!(scheduler.step_optimizer(&mut optimizer).is_err());
    }

    #[test]
    fn unified_scheduler_plateau_step_optimizer_with_metric_updates_optimizer() {
        let mut scheduler = LRScheduler::ReduceOnPlateau(
            ReduceOnPlateauLR::new(
                1.0,
                ReduceOnPlateauConfig {
                    factor: 0.5,
                    patience: 2,
                    threshold: 0.0,
                    eps: 0.0,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let mut optimizer = OptimizerKind::adam(
            vec![],
            AdamConfig {
                lr: 1.0,
                ..Default::default()
            },
        )
        .unwrap();

        scheduler
            .step_optimizer_with_metric(1.0, &mut optimizer)
            .unwrap();
        scheduler
            .step_optimizer_with_metric(1.1, &mut optimizer)
            .unwrap();
        let lr = scheduler
            .step_optimizer_with_metric(1.2, &mut optimizer)
            .unwrap();

        assert_close(lr, 0.5);
        assert_close(optimizer.learning_rate(), 0.5);
    }
}

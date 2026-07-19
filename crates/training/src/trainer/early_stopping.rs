//! Early-stopping mode, configuration, and internal state.
//!
//! Extracted from [`crate::trainer`] because the three concepts form a
//! self-contained unit with their own public API contract
//! (`EarlyStoppingMode` + `EarlyStoppingConfig`) plus one private
//! helper (`EarlyStoppingState`) — keeping them in one 100-line file
//! rather than the 2000+-line `trainer/mod.rs` makes the early-
//! stopping behaviour easy to locate and review. No logic change from
//! the previous inline version; the only visibility shift is that
//! `EarlyStoppingState` is now `pub(super)` so the parent `trainer`
//! module can still construct it.

use std::fmt;

use candle_core::Result;
use serde::{Deserialize, Serialize};

use sparam_core::validation::{validate_non_negative_f64, validate_positive_usize};

/// Whether improvement means lower or higher metric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EarlyStoppingMode {
    /// Lower is better (e.g. loss).
    Min,
    /// Higher is better (e.g. accuracy).
    Max,
}

impl EarlyStoppingMode {
    /// Returns `true` when `current` is strictly better than `best` by at
    /// least `min_delta`.
    #[inline]
    pub(super) fn is_improvement(self, current: f64, best: f64, min_delta: f64) -> bool {
        match self {
            Self::Min => current < best - min_delta,
            Self::Max => current > best + min_delta,
        }
    }

    #[inline]
    pub(super) fn initial_best(self) -> f64 {
        match self {
            Self::Min => f64::INFINITY,
            Self::Max => f64::NEG_INFINITY,
        }
    }
}

impl fmt::Display for EarlyStoppingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Min => f.write_str("min"),
            Self::Max => f.write_str("max"),
        }
    }
}

/// Configuration for early stopping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Epochs without improvement before stopping.
    pub patience: usize,
    /// Minimum metric change to count as improvement.
    pub min_delta: f64,
    /// Epochs to train before monitoring begins.
    pub warmup_epochs: usize,
    /// Whether lower or higher metric is better.
    pub mode: EarlyStoppingMode,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            patience: 10,
            min_delta: 0.0,
            warmup_epochs: 0,
            mode: EarlyStoppingMode::Min,
        }
    }
}

impl EarlyStoppingConfig {
    pub(super) fn validate(&self) -> Result<()> {
        validate_positive_usize("patience", self.patience)?;
        validate_non_negative_f64("min_delta", self.min_delta)?;
        Ok(())
    }
}

/// Internal mutable state for early-stopping tracking.
pub(super) struct EarlyStoppingState {
    pub(super) best_metric: f64,
    pub(super) best_epoch: usize,
    pub(super) epochs_without_improvement: usize,
}

impl EarlyStoppingState {
    pub(super) fn new(mode: EarlyStoppingMode) -> Self {
        Self {
            best_metric: mode.initial_best(),
            best_epoch: 0,
            epochs_without_improvement: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_mode_improvement() {
        let mode = EarlyStoppingMode::Min;
        // 0.5 < 1.0 - 0.1 → true
        assert!(mode.is_improvement(0.5, 1.0, 0.1));
        // 0.95 < 1.0 - 0.1 = 0.9 → false
        assert!(!mode.is_improvement(0.95, 1.0, 0.1));
    }

    #[test]
    fn max_mode_improvement() {
        let mode = EarlyStoppingMode::Max;
        // 1.5 > 1.0 + 0.1 → true
        assert!(mode.is_improvement(1.5, 1.0, 0.1));
        // 1.05 > 1.0 + 0.1 = 1.1 → false
        assert!(!mode.is_improvement(1.05, 1.0, 0.1));
    }
}

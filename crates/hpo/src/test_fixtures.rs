//! Shared `HyperParams` fixtures used by tests across the crate.
//!
//! Before consolidation, each test module (`mlp_evaluator`, `pareto`,
//! `summary`, `storage::json`, `storage::sqlite`) carried its own
//! copy-paste of `sample_params(hidden_size)` — five near-identical
//! builders that had to be kept in sync by hand. Centralizing them
//! here means adding a new `HyperParams` field updates one file, and
//! the Real/Complex defaults match everywhere.
//!
//! Compiled only under `#[cfg(test)]` — adds nothing to release binaries.

#![cfg(test)]

use sparam_models::{Activation, ComplexActivation};

use crate::search_space::{
    BatchSizeChoice, GradClipChoice, HyperParams, LossChoice, LossHyperParams, ModelKind,
    NormChoice, OptimizerChoice, OptimizerHyperParams, SchedulerChoice, SchedulerHyperParams,
};

/// Default Real-variant `HyperParams` at the given `hidden_size`.
/// Fields match what individual test modules used to define
/// locally — stable across callers so refactors don't require
/// simultaneous updates to 5 test files.
pub(crate) fn sample_real_params(hidden_size: i64) -> HyperParams {
    HyperParams {
        hidden_size,
        optimizer: OptimizerChoice::Adam,
        lr: 1e-3,
        train_batch_size: BatchSizeChoice::All,
        weight_decay: 0.0,
        loss: LossChoice::Mse,
        loss_params: LossHyperParams::default(),
        scheduler: SchedulerChoice::None,
        scheduler_params: SchedulerHyperParams::default(),
        optimizer_params: OptimizerHyperParams::default(),
        kind: ModelKind::Real {
            activation: Activation::ReLU,
            dropout_p: 0.0,
            norm: NormChoice::None,
            grad_clip_norm: GradClipChoice::None,
            input_noise_std: 0.0,
        },
    }
}

/// Default Complex-variant `HyperParams` at the given `hidden_size`.
///
/// `#[allow(dead_code)]`: currently only `sample_real_params` is
/// imported by the storage / pareto tests. This counterpart exists so
/// future Complex-path tests (HPO retrain integration, summary cell
/// rendering for Complex trials) can pull a ready-made fixture
/// instead of open-coding another copy.
#[allow(dead_code)]
pub(crate) fn sample_complex_params(hidden_size: i64) -> HyperParams {
    HyperParams {
        hidden_size,
        optimizer: OptimizerChoice::AdamW,
        lr: 1e-3,
        train_batch_size: BatchSizeChoice::All,
        weight_decay: 0.01,
        // Complex MLPs train exclusively against the physics-forward
        // loss under the current partition — no loss-specific knobs.
        loss: LossChoice::PhysicsForward,
        loss_params: LossHyperParams::default(),
        scheduler: SchedulerChoice::None,
        scheduler_params: SchedulerHyperParams::default(),
        optimizer_params: OptimizerHyperParams::default(),
        kind: ModelKind::Complex {
            activation: ComplexActivation::CReLU,
            dropout_p: 0.0,
            grad_clip_norm: GradClipChoice::None,
            input_noise_std: 0.0,
        },
    }
}

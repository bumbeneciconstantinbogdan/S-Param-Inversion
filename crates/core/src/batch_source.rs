//! Trait abstraction for batch iteration, decoupling training from data loading.
//!
//! The [`BatchSource`] trait allows the trainer to consume batches without
//! knowing the concrete data-loading implementation.

use candle_core::{Result, Tensor};

/// A source of `(features, targets)` mini-batches for supervised training.
///
/// Implementors yield one epoch's worth of batches per `batches()` call.
pub trait BatchSource {
    /// Yield batches for one epoch.
    fn batches(&self) -> Box<dyn Iterator<Item = Result<(Tensor, Tensor)>> + '_>;

    /// Total number of samples in the dataset.
    fn sample_count(&self) -> usize;

    /// Number of batches per epoch.
    fn num_batches(&self) -> usize;
}

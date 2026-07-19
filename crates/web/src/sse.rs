//! SSE bridge: `TrainingCallback` implementation that forwards epoch metrics
//! into a `tokio::sync::broadcast` channel for SSE streaming.

use tokio::sync::broadcast;

use sparam_training::trainer::{EpochMetrics, TrainingCallback};

use crate::state::SseEvent;

/// A [`TrainingCallback`] that converts epoch metrics into [`SseEvent`]s
/// and sends them on a broadcast channel.
///
/// This type is `Send` (broadcast::Sender is Send+Sync) so it can be
/// moved into a `spawn_blocking` closure.
///
/// `member` / `ensemble_size` are set by the stacked pipeline training path
/// to identify which ensemble member the per-epoch event belongs to;
/// for single-model runs they're left `None` and the JS chart treats
/// the events as one unbroken series.
pub struct SseCallback {
    tx: broadcast::Sender<SseEvent>,
    member: Option<usize>,
    ensemble_size: Option<usize>,
}

impl SseCallback {
    pub fn new(tx: broadcast::Sender<SseEvent>) -> Self {
        Self { tx, member: None, ensemble_size: None }
    }

    /// Tag every epoch event this callback emits with the ensemble
    /// member index (1-based) and total ensemble size.  Used by the
    /// stacked pipeline path so the per-epoch chart can label which member
    /// it's currently drawing.
    pub fn with_member(mut self, member: usize, ensemble_size: usize) -> Self {
        self.member = Some(member);
        self.ensemble_size = Some(ensemble_size);
        self
    }
}

impl TrainingCallback for SseCallback {
    fn on_epoch_end(&mut self, metrics: &EpochMetrics) {
        // Ignore send errors (no active subscribers = event is dropped).
        let _ = self.tx.send(SseEvent::Epoch {
            epoch: metrics.epoch,
            max_epochs: metrics.max_epochs,
            train_loss: metrics.train_loss,
            val_loss: metrics.val_loss,
            learning_rate: metrics.learning_rate,
            elapsed_secs: metrics.elapsed_secs,
            patience_status: metrics.patience_status,
            member: self.member,
            ensemble_size: self.ensemble_size,
        });
    }

    fn on_improvement(&mut self, epoch: usize, val_loss: f64) {
        let _ = self.tx.send(SseEvent::Improvement { epoch, val_loss });
    }

    fn on_early_stop(&mut self, _epoch: usize) {
        // Early stop is communicated via the Complete event after fit() returns.
    }
}

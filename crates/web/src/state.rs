//! Shared application state.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rusqlite::Connection;
use sparam_data::scaling::FittedScalers;
use tokio::sync::{Mutex, RwLock, broadcast};

/// Unique identifier for a training run (matches DB row id).
pub type RunId = i64;

/// Events sent over the SSE channel during training.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum SseEvent {
    Epoch {
        epoch: usize,
        max_epochs: usize,
        train_loss: f64,
        val_loss: Option<f64>,
        learning_rate: f64,
        elapsed_secs: f64,
        patience_status: Option<(usize, usize)>,
        /// 1-based ensemble member index for the stacked pipeline path; `None`
        /// for single-model runs.  Lets the client clear or annotate
        /// the per-epoch chart when a new member starts.
        member: Option<usize>,
        /// Total members in the ensemble (`Some(N)` together with
        /// `member` for M10; `None` otherwise).
        ensemble_size: Option<usize>,
    },
    Improvement {
        epoch: usize,
        val_loss: f64,
    },
    /// stacked pipeline milestone — coarse progress signal between members
    /// and during the post-training NRW refinement phase.  Lets the UI
    /// show "Member 2 of 3" / "Computing ensemble metrics" / "Running
    /// 100-step NRW refinement…" instead of a frozen graph.
    Stage {
        message: String,
    },
    Complete {
        run_id: RunId,
    },
    Error {
        message: String,
    },
}

/// Per-run SSE broadcast channel.
pub struct RunBroadcast {
    pub sender: broadcast::Sender<SseEvent>,
}

/// Events sent over SSE during HPO.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum HpoSseEvent {
    TrialComplete {
        trial: usize,
        total: usize,
        model_type: String,
        status: String,
        val_loss: f64,
        ok_at_1pct: f64,
        hidden_size: i64,
        param_count: usize,
        max_error: f64,
        is_feasible: bool,
    },
    StudyComplete {
        study_id: i64,
    },
    StudyError {
        message: String,
    },
}

/// Per-study SSE broadcast channel.
pub struct HpoBroadcast {
    pub sender: broadcast::Sender<HpoSseEvent>,
}

/// Cache key for fitted feature/target scalers. Captures every input
/// the StandardScaler fit depends on: the dataset (training samples),
/// the model family (Real vs Complex packing), and the stack overlay
/// (which augments the training samples before fit).
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ScalerCacheKey {
    pub dataset_id: i64,
    pub is_complex: bool,
    pub overlay_density: usize,
    pub overlay_seed: u64,
}

/// Shared application state, wrapped in `Arc` and passed to all handlers.
pub struct AppState {
    /// SQLite connection (serialized access via Mutex since rusqlite
    /// Connection is not Sync).
    pub db: Mutex<Connection>,

    /// Root directory for all project data.
    pub project_root: PathBuf,

    /// Active SSE broadcast channels for training runs.
    pub run_channels: RwLock<HashMap<RunId, Arc<RunBroadcast>>>,

    /// Active SSE broadcast channels for HPO studies.
    pub hpo_channels: RwLock<HashMap<i64, Arc<HpoBroadcast>>>,

    /// Cross-request scaler cache. Keyed by everything the fit
    /// depends on; values are `Arc`-shared so requests can pass
    /// borrowed refs into the workflows without cloning.
    pub scaler_cache: RwLock<HashMap<ScalerCacheKey, Arc<FittedScalers>>>,
}

impl AppState {
    pub fn new(db: Connection, project_root: PathBuf) -> Self {
        Self {
            db: Mutex::new(db),
            project_root,
            run_channels: RwLock::new(HashMap::new()),
            hpo_channels: RwLock::new(HashMap::new()),
            scaler_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Get the root data directory.
    pub fn data_dir(&self) -> &Path {
        &self.project_root
    }

    /// Look up cached scalers for `key`, or fit-and-insert via `fit_fn`
    /// on miss. Concurrent requests for the same key may both fit; the
    /// first inserted value wins and is returned to all callers.
    pub async fn get_or_fit_scalers<F>(
        &self,
        key: ScalerCacheKey,
        fit_fn: F,
    ) -> Result<Arc<FittedScalers>, candle_core::Error>
    where
        F: FnOnce() -> Result<FittedScalers, candle_core::Error>,
    {
        if let Some(v) = self.scaler_cache.read().await.get(&key) {
            return Ok(Arc::clone(v));
        }
        let fitted = Arc::new(fit_fn()?);
        let mut cache = self.scaler_cache.write().await;
        Ok(Arc::clone(cache.entry(key).or_insert(fitted)))
    }
}

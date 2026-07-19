//! Public configuration and result types for workflow entry points.

use std::path::PathBuf;

use candle_core::Result;
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};

use sparam_models::{
    Activation, ComplexActivation, ComplexMLPConfig, ComplexMLPRegressor, ComplexNormChoice,
    MLPConfig, MLPRegressor, MlpModel, ModelType, Normalization,
};

/// Model architecture config. Activations are typed per family so a
/// complex name like `"modrelu"` can't end up in a Real config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "model_type", rename_all = "lowercase")]
pub enum ModelConfig {
    Real {
        hidden_size: usize,
        #[serde(with = "sparam_models::activation::as_name")]
        activation: Activation,
        dropout: f64,
        norm: String,
    },
    Complex {
        hidden_size: usize,
        complex_activation: ComplexActivation,
        #[serde(default)]
        dropout: f64,
        /// `None` keeps pre-LN-default checkpoints loadable — without
        /// it, old weights without `model.norm.*` tensors would feed
        /// through an untrained `(γ=1, β=0)` LayerNorm.
        #[serde(default)]
        norm: ComplexNormChoice,
    },
}

impl ModelConfig {
    pub fn real(
        hidden_size: usize,
        activation: impl AsRef<str>,
        dropout: f64,
        norm: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self::Real {
            hidden_size,
            activation: Activation::from_name(activation.as_ref())?,
            dropout,
            norm: norm.into(),
        })
    }

    pub fn complex(
        hidden_size: usize,
        complex_activation: impl AsRef<str>,
        dropout: f64,
    ) -> Result<Self> {
        Ok(Self::Complex {
            hidden_size,
            complex_activation: ComplexActivation::from_name(
                complex_activation.as_ref(),
            )?,
            dropout,
            norm: ComplexNormChoice::LayerNorm,
        })
    }

    #[must_use]
    pub fn is_complex(&self) -> bool {
        matches!(self, Self::Complex { .. })
    }

    #[must_use]
    pub fn model_type(&self) -> ModelType {
        match self {
            Self::Real { .. } => ModelType::Real,
            Self::Complex { .. } => ModelType::Complex,
        }
    }

    #[must_use]
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Real { hidden_size, .. } | Self::Complex { hidden_size, .. } => *hidden_size,
        }
    }

    #[must_use]
    pub fn arch_description(&self) -> String {
        match self {
            Self::Complex { hidden_size, .. } => format!("Complex MLP (2c-{hidden_size}c-1c)"),
            Self::Real { hidden_size, .. } => format!("Real MLP (4-{hidden_size}-2)"),
        }
    }

    pub fn build_model(&self, vb: VarBuilder) -> Result<MlpModel> {
        match self {
            Self::Real { hidden_size, activation, dropout, norm } => {
                let norm = Normalization::from_name(norm)?;
                let config = MLPConfig::permittivity(*hidden_size, *activation)
                    .with_dropout(*dropout as f32)
                    .with_norm(norm);
                Ok(MlpModel::Real(MLPRegressor::new(vb, &config)?))
            }
            Self::Complex { hidden_size, complex_activation, dropout, norm } => {
                let config = ComplexMLPConfig::permittivity(*hidden_size, *complex_activation)
                    .with_dropout(*dropout as f32)
                    .with_norm(*norm);
                Ok(MlpModel::Complex(ComplexMLPRegressor::new(vb, &config)?))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// `Arc<[_]>` so the web handler can hand the same backing
    /// allocation to train + evaluate without a clone.
    pub train_samples: std::sync::Arc<[sparam_data::generation::PermittivitySample]>,
    pub val_samples: std::sync::Arc<[sparam_data::generation::PermittivitySample]>,
    pub test_samples: std::sync::Arc<[sparam_data::generation::PermittivitySample]>,
    /// `Some(path)` writes a sidecar `.safetensors` (CLI); `None`
    /// skips disk and the caller persists the returned bytes (web UI).
    pub checkpoint_path: Option<PathBuf>,
    pub model: ModelConfig,
    pub optimizer: String,
    pub lr: f64,
    pub weight_decay: f64,
    pub batch_size: String,
    pub max_epochs: usize,
    pub patience: usize,
    pub warmup_epochs: usize,
    pub scheduler: String,
    pub loss: String,
    pub checkpoint: Option<PathBuf>,
    pub seed: u64,
    pub log_progress: bool,

    /// HPO pins this so every trial sees the same batch order;
    /// `None` falls back to `seed`.
    pub shuffle_seed: Option<u64>,

    // HPO-parity fields. Required for "retrain from trial" to reproduce
    // bit-for-bit. `None` = library default.
    pub grad_clip_norm: Option<f64>,
    pub input_noise_std: Option<f64>,
    pub cosine_eta_min: Option<f64>,
    pub plateau_factor: Option<f64>,
    pub plateau_patience: Option<usize>,
    pub plateau_min_lr: Option<f64>,
    pub sgd_momentum: Option<f64>,
    pub sgd_nesterov: Option<bool>,
    pub rmsprop_momentum: Option<f64>,
    pub rmsprop_alpha: Option<f64>,
    pub pi_mape_beta: Option<f64>,
    /// λ for hybrid PhysicsForward loss (`L = base + λ · NRW_residual`).
    pub physics_lambda: Option<f64>,
}

/// Five-stage stacked pipeline wrapper around [`TrainConfig`]. Each toggle is
/// independent; set to the listed defaults to disable. See
/// `poster/studies/methodology/M10_tail_sweep.tex`.
#[derive(Debug, Clone)]
pub struct StackedTrainConfig {
    pub base: TrainConfig,
    /// `N×N` near-vacuum patch synthesised at train time only (eval
    /// untouched). `0` = off. Sweet spot `8`.
    pub train_overlay_density: usize,
    /// Train N models with seeds `(base.seed..)`, take per-element
    /// median. `1` = single model. Sweet spot `3` (smallest odd N
    /// so the median is one of the actual predictions).
    pub ensemble_size: usize,
    /// Adam steps minimising `|NRW(ε̂) − S_measured|²` in physical
    /// space, starting from the MLP prediction. `0` = off. Sweet spot
    /// `100`.
    pub refine_steps: usize,
    pub refine_lr: f64,
}

impl StackedTrainConfig {
    pub fn from_base(base: TrainConfig) -> Self {
        Self {
            base,
            train_overlay_density: 0,
            ensemble_size: 1,
            refine_steps: 0,
            refine_lr: 1e-3,
        }
    }

    /// winning recipe at matched-budget (Complex H=48 / Real H=64):
    /// overlay 8 + 3-median + 100 refine. Caller still picks
    /// activation (`worelu`/`leaky_relu`) and `eps_rel_mse` loss.
    pub fn winning_recipe(base: TrainConfig) -> Self {
        Self {
            base,
            train_overlay_density: 8,
            ensemble_size: 3,
            refine_steps: 100,
            refine_lr: 1e-3,
        }
    }

    pub fn is_passthrough(&self) -> bool {
        self.train_overlay_density == 0
            && self.ensemble_size <= 1
            && self.refine_steps == 0
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainMetricsSummary {
    pub ok_at_1pct: f64,
    pub ok_at_10pct: f64,
    pub mean_error: f64,
    pub max_error: f64,
    pub min_error: f64,
    pub median_error: f64,
    pub r2_real: f64,
    pub r2_imag: f64,
    pub best_epoch: usize,
    pub final_epoch: usize,
    pub stopped_early: bool,
    pub training_time_secs: f64,
}

#[derive(Debug, Clone)]
pub enum CheckpointSource {
    Path(PathBuf),
    Bytes(Vec<u8>),
}

/// stacked toggles bundled into one struct. Each stage is
/// independently disabled via the default-zero / empty-vec value.
#[derive(Debug, Clone, Default)]
pub struct StackedInferenceConfig {
    pub overlay_density: usize,
    /// Seed for overlay synthesis; must match the run's training seed
    /// so overlay is bit-identical between train and eval.
    pub seed: u64,
    /// Non-primary ensemble members from `training_runs.ensemble_weights_json`.
    pub extra_members: Vec<Vec<u8>>,
    pub refine_steps: usize,
    pub refine_lr: f64,
}

impl StackedInferenceConfig {
    pub fn is_passthrough(&self) -> bool {
        self.overlay_density == 0
            && self.extra_members.is_empty()
            && self.refine_steps == 0
    }

    /// Pull stack toggles from a parsed `training_runs.config_json`.
    /// `seed_default` covers runs that pre-date the stack fields.
    /// Does NOT populate `extra_members` — caller loads those from the
    /// `ensemble_weights_json` column.
    pub fn from_config_json(cfg: &serde_json::Value, seed_default: u64) -> Self {
        let overlay_density = cfg
            .get("m10_train_overlay_density")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let seed = cfg.get("seed").and_then(|v| v.as_u64()).unwrap_or(seed_default);
        let refine_steps = cfg
            .get("m10_refine_steps")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let refine_lr = cfg
            .get("m10_refine_lr")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.001);
        Self {
            overlay_density,
            seed,
            extra_members: Vec::new(),
            refine_steps,
            refine_lr,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvaluateConfig {
    pub model: CheckpointSource,
    pub model_config: ModelConfig,
    pub parameter_count: usize,
    pub test_samples: std::sync::Arc<[sparam_data::generation::PermittivitySample]>,
    /// Empty → fall back to `test_samples`.
    pub scaler_samples: std::sync::Arc<[sparam_data::generation::PermittivitySample]>,
    pub thresholds: (f64, f64),
    pub generate_plots: bool,
    pub quiet: bool,
    pub stack: StackedInferenceConfig,
    /// Pre-fitted scaler stats from a higher-level cache. When `Some`,
    /// the workflow skips the StandardScaler fit on `scaler_samples`.
    pub pre_fitted_scalers: Option<std::sync::Arc<sparam_data::scaling::FittedScalers>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluateOutput {
    pub model_path: String,
    pub test_samples: usize,
    pub metrics: EvaluateMetrics,
    pub thresholds: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvaluateMetrics {
    pub ok_at_1pct: f64,
    pub ok_at_10pct: f64,
    pub mean_error: f64,
    pub max_error: f64,
    pub min_error: f64,
    pub std_error: f64,
    pub r2_real: f64,
    pub r2_imag: f64,
}

#[derive(Debug, Clone)]
pub struct ThresholdSummary {
    pub threshold: f64,
    pub ok_count: usize,
    pub ok_percent: f64,
}

#[derive(Debug, Clone)]
pub struct EvaluateRunResult {
    pub output: EvaluateOutput,
    pub classifications: Vec<ThresholdSummary>,
    /// Chart data JSON (plotly-ready). Caller persists to DB.
    pub charts_json: Option<serde_json::Value>,
}

/// Closed-form NRW round-trip evaluation: forward to S-params, invert
/// back to ε, compare against the input grid. Baseline for the NN
/// inverse. μ_r is fixed at `1 + 0j` (non-magnetic).
#[derive(Debug, Clone)]
pub struct NrwEvaluateConfig {
    /// Sample thickness (m).
    pub d: f64,
    /// Waveguide width (m).
    pub a: f64,
    /// Operating frequency (Hz).
    pub frequency: f64,
    pub eps_prime_range: (f64, f64),
    pub eps_double_prime_range: (f64, f64),
    pub n_eps_prime: usize,
    pub n_eps_double_prime: usize,
    /// `(ok@1%, ok@10%)` thresholds.
    pub thresholds: (f64, f64),
    pub generate_plots: bool,
    pub quiet: bool,
}

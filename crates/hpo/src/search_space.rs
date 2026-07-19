//! HPO search space and choice enums.

use std::fmt;

use candle_core::Result;
use candle_nn::VarBuilder;
use optimizer::Categorical;
use optimizer::prelude::*;
use serde::{Deserialize, Serialize};
use sparam_data::loader::BatchSize;
use sparam_models::{
    Activation, ComplexActivation, ComplexMLPConfig, ComplexMLPRegressor, ComplexNormChoice,
    MLPConfig, MLPRegressor, MlpModel, ModelType, Normalization,
};
use sparam_training::optimizers::OptimizerConfig;
use sparam_training::schedulers::{LRScheduler, SchedulerConfig};
use sparam_training::trainer::{EarlyStoppingConfig, TrainerBuilder, TrainerConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Categorical)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum ActivationChoice {
    Relu,
    Tanh,
    Gelu,
    LeakyRelu,
    Elu,
}

impl ActivationChoice {
    const ALL: [Self; 5] = [Self::Relu, Self::Tanh, Self::Gelu, Self::LeakyRelu, Self::Elu];

    #[inline]
    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.get(index).copied()
    }

    #[inline]
    pub fn label(self) -> &'static str {
        match self {
            Self::Relu => "relu",
            Self::Tanh => "tanh",
            Self::Gelu => "gelu",
            Self::LeakyRelu => "lrelu",
            Self::Elu => "elu",
        }
    }
}

impl fmt::Display for ActivationChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Categorical)]
#[repr(u8)]
pub enum OptimizerChoice {
    Adam,
    AdamW,
    RMSprop,
    SGD,
}

impl OptimizerChoice {
    const ALL: [Self; 4] = [Self::Adam, Self::AdamW, Self::RMSprop, Self::SGD];

    #[inline]
    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.get(index).copied()
    }

    #[inline]
    pub fn label(self) -> &'static str {
        match self {
            Self::Adam => "adam",
            Self::AdamW => "adamw",
            Self::RMSprop => "rmspr",
            Self::SGD => "sgd",
        }
    }

    pub fn from_alias(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "adam" => Some(Self::Adam),
            "adamw" | "adam_w" => Some(Self::AdamW),
            "rmsprop" | "rms_prop" => Some(Self::RMSprop),
            "sgd" => Some(Self::SGD),
            _ => None,
        }
    }
}

impl fmt::Display for OptimizerChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Categorical)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum NormChoice {
    None,
    LayerNorm,
    BatchNorm,
}

impl NormChoice {
    const ALL: [Self; 3] = [Self::None, Self::LayerNorm, Self::BatchNorm];

    #[inline]
    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.get(index).copied()
    }

    #[inline]
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::LayerNorm => "layr",
            Self::BatchNorm => "btch",
        }
    }

    pub fn from_alias(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" => Some(Self::None),
            "layernorm" | "layer_norm" => Some(Self::LayerNorm),
            "batchnorm" | "batch_norm" => Some(Self::BatchNorm),
            _ => None,
        }
    }
}

impl fmt::Display for NormChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Categorical)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum LossChoice {
    Mse,
    SmoothL1,
    EpsRelMse,
    /// NRW forward-model loss; Complex-MLP only.
    /// Legacy JSON `pi_mape` deserializes here via `#[serde(alias)]`.
    #[serde(alias = "pi_mape")]
    PhysicsForward,
}

impl LossChoice {
    pub const ALL: [Self; 4] = [Self::Mse, Self::SmoothL1, Self::EpsRelMse, Self::PhysicsForward];

    pub const REAL_COMPATIBLE: [Self; 3] = [Self::Mse, Self::SmoothL1, Self::EpsRelMse];

    pub const COMPLEX_COMPATIBLE: [Self; 3] =
        [Self::Mse, Self::SmoothL1, Self::PhysicsForward];

    #[inline]
    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.get(index).copied()
    }

    #[inline]
    pub fn is_complex_only(self) -> bool {
        matches!(self, Self::PhysicsForward)
    }

    #[inline]
    pub fn is_real_compatible(self) -> bool {
        Self::REAL_COMPATIBLE.contains(&self)
    }

    #[inline]
    pub fn is_complex_compatible(self) -> bool {
        Self::COMPLEX_COMPATIBLE.contains(&self)
    }

    #[inline]
    pub fn label(self) -> &'static str {
        match self {
            Self::Mse => "mse",
            Self::SmoothL1 => "smth",
            Self::EpsRelMse => "ermse",
            Self::PhysicsForward => "pforw",
        }
    }

    pub fn from_alias(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mse" => Some(Self::Mse),
            "smooth_l1" | "smoothl1" | "smooth-l1" => Some(Self::SmoothL1),
            "eps_rel_mse" | "epsrel" | "eps_rel" => Some(Self::EpsRelMse),
            "physics_forward" | "physicsforward" | "physics-forward" | "nrw" | "pi_mape"
            | "pimape" | "pi-mape" => Some(Self::PhysicsForward),
            _ => None,
        }
    }
}

impl fmt::Display for LossChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Categorical)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerChoice {
    None,
    Plateau,
    Cosine,
}

impl SchedulerChoice {
    const ALL: [Self; 3] = [Self::None, Self::Plateau, Self::Cosine];

    #[inline]
    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.get(index).copied()
    }

    #[inline]
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Plateau => "plat",
            Self::Cosine => "cos",
        }
    }

    pub fn from_alias(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" => Some(Self::None),
            "cosine" | "cos" => Some(Self::Cosine),
            "plateau" | "reduce_on_plateau" | "plat" => Some(Self::Plateau),
            _ => None,
        }
    }

    pub fn to_scheduler_config(
        self,
        max_epochs: usize,
        cosine_eta_min: Option<f64>,
        plateau_factor: Option<f64>,
        plateau_patience: Option<usize>,
        plateau_min_lr: Option<f64>,
    ) -> SchedulerConfig {
        match self {
            Self::None => SchedulerConfig::none(),
            Self::Cosine => SchedulerConfig::cosine(
                max_epochs,
                cosine_eta_min.unwrap_or(1e-6),
            ),
            Self::Plateau => SchedulerConfig::plateau_with(
                plateau_factor.unwrap_or(0.1),
                plateau_patience.unwrap_or(10),
                plateau_min_lr.unwrap_or(1e-6),
            ),
        }
    }
}

impl fmt::Display for SchedulerChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Categorical)]
#[repr(u8)]
pub enum BatchSizeChoice {
    #[serde(rename = "8")]
    B8,
    #[serde(rename = "16")]
    B16,
    #[serde(rename = "32")]
    B32,
    #[serde(rename = "64")]
    B64,
    #[serde(rename = "128")]
    B128,
    #[serde(rename = "256")]
    B256,
    #[serde(rename = "512")]
    B512,
    #[serde(rename = "ALL")]
    All,
}

impl BatchSizeChoice {
    const ALL: [Self; 8] = [
        Self::B8,
        Self::B16,
        Self::B32,
        Self::B64,
        Self::B128,
        Self::B256,
        Self::B512,
        Self::All,
    ];

    #[inline]
    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.get(index).copied()
    }

    #[inline]
    pub fn label(self) -> &'static str {
        match self {
            Self::B8 => "8",
            Self::B16 => "16",
            Self::B32 => "32",
            Self::B64 => "64",
            Self::B128 => "128",
            Self::B256 => "256",
            Self::B512 => "512",
            Self::All => "all",
        }
    }

    /// `None` = full-batch training.
    #[inline]
    pub fn to_usize(self) -> Option<usize> {
        match self {
            Self::B8 => Some(8),
            Self::B16 => Some(16),
            Self::B32 => Some(32),
            Self::B64 => Some(64),
            Self::B128 => Some(128),
            Self::B256 => Some(256),
            Self::B512 => Some(512),
            Self::All => None,
        }
    }

    pub fn from_alias(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "8" => Some(Self::B8),
            "16" => Some(Self::B16),
            "32" => Some(Self::B32),
            "64" => Some(Self::B64),
            "128" => Some(Self::B128),
            "256" => Some(Self::B256),
            "512" => Some(Self::B512),
            "all" | "full" => Some(Self::All),
            _ => None,
        }
    }
}

impl fmt::Display for BatchSizeChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// `None` is the default; lets `#[serde(default)]` on Complex
/// `ModelKind` round-trip legacy JSON without grad-clipping.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, Categorical,
)]
#[repr(u8)]
pub enum GradClipChoice {
    #[default]
    None,
    #[serde(rename = "0.5")]
    Clip05,
    #[serde(rename = "1.0")]
    Clip10,
    #[serde(rename = "2.0")]
    Clip20,
    #[serde(rename = "5.0")]
    Clip50,
}

impl GradClipChoice {
    /// `Clip20` is retained on the enum for legacy JSON but
    /// excluded from sampling — top-200 trials never picked it.
    pub const ALL: [Self; 4] = [
        Self::None,
        Self::Clip05,
        Self::Clip10,
        Self::Clip50,
    ];

    #[inline]
    pub fn to_f64(self) -> Option<f64> {
        match self {
            Self::None => None,
            Self::Clip05 => Some(0.5),
            Self::Clip10 => Some(1.0),
            Self::Clip20 => Some(2.0),
            Self::Clip50 => Some(5.0),
        }
    }

    pub fn from_alias(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "0" => Some(Self::None),
            "0.5" => Some(Self::Clip05),
            "1" | "1.0" => Some(Self::Clip10),
            "2" | "2.0" => Some(Self::Clip20),
            "5" | "5.0" => Some(Self::Clip50),
            _ => None,
        }
    }
}

/// `model_type` selects which `HyperParams` variant `suggest`
/// emits; Complex trials skip the Real-only regularization fields.
#[derive(Debug, Clone)]
pub struct MlpSearchSpace {
    pub model_type: ModelType,

    pub hidden_size: CategoricalParam<i64>,
    pub activation: CategoricalParam<String>,

    pub optimizer: CategoricalParam<OptimizerChoice>,
    pub lr: FloatParam,
    pub train_batch_size: CategoricalParam<BatchSizeChoice>,

    pub weight_decay: FloatParam,
    pub loss: CategoricalParam<LossChoice>,

    // Real-only regularization; ignored on Complex.
    pub dropout_p: FloatParam,
    pub norm: CategoricalParam<NormChoice>,
    pub grad_clip_norm: CategoricalParam<GradClipChoice>,
    pub input_noise_std: FloatParam,

    pub scheduler: CategoricalParam<SchedulerChoice>,
    pub plateau_factor: FloatParam,
    pub plateau_patience: IntParam,
    pub plateau_min_lr: FloatParam,
    pub cosine_eta_min: FloatParam,

    pub sgd_momentum: FloatParam,
    pub sgd_nesterov: BoolParam,
    pub rmsprop_momentum: FloatParam,
    pub rmsprop_alpha: FloatParam,
    pub physics_lambda: FloatParam,
}


impl Default for MlpSearchSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl MlpSearchSpace {
    pub fn new() -> Self {
        Self::for_real()
    }

    pub fn for_real() -> Self {
        Self::build_base_space(
            ModelType::Real,
            vec![8i64, 16, 32, 64],
            vec![
                "relu".into(), "tanh".into(), "gelu".into(),
                "leaky_relu".into(), "elu".into(),
            ],
        )
    }

    /// Complex search space with the full activation menu so HPO can
    /// rediscover whether the prior 3-activation winner set still
    /// dominates as the dataset / loss / overlay evolve.
    pub fn for_complex() -> Self {
        Self::build_base_space(
            ModelType::Complex,
            vec![6i64, 12, 24, 48],
            vec![
                "crelu".into(),
                "cgelu".into(),
                "cardioid".into(),
                "modrelu".into(),
                "hybrid_cardioid_gelu".into(),
                "lpma".into(),
                "cswish_phase".into(),
                "worelu".into(),
            ],
        )
    }

    /// Complex tightens continuous ranges to the dense quantiles of
    /// the Real-MLP top-200 (shared optimiser / scheduler stack).
    fn build_base_space(
        model_type: ModelType,
        hidden_sizes: Vec<i64>,
        activations: Vec<String>,
    ) -> Self {
        let is_complex = matches!(model_type, ModelType::Complex);

        // Continuous range bounds: Real (broad) vs Complex (narrowed).
        let lr_range = if is_complex { (3e-3, 2.5e-2) } else { (1e-4, 5e-2) };
        let wd_range = if is_complex { (1e-8, 1e-5) } else { (1e-10, 1e-2) };
        let dropout_range = if is_complex { (0.0, 0.05) } else { (0.0, 0.4) };
        let noise_range = if is_complex { (0.0, 0.008) } else { (0.0, 0.01) };
        let cos_eta_min_range = if is_complex { (1e-6, 5e-5) } else { (1e-6, 1e-4) };

        Self {
            model_type,
            hidden_size: CategoricalParam::new(hidden_sizes).name("hidden_size"),
            activation: CategoricalParam::new(activations).name("activation"),

            optimizer: CategoricalParam::new(OptimizerChoice::ALL.to_vec()).name("optimizer"),
            lr: FloatParam::new(lr_range.0, lr_range.1).log_scale().name("lr"),
            train_batch_size: CategoricalParam::new(BatchSizeChoice::ALL.to_vec())
                .name("train_batch_size"),

            weight_decay: FloatParam::new(wd_range.0, wd_range.1)
                .log_scale()
                .name("weight_decay"),
            dropout_p: FloatParam::new(dropout_range.0, dropout_range.1).name("dropout_p"),
            norm: CategoricalParam::new(NormChoice::ALL.to_vec()).name("norm"),
            // Real → REAL_COMPATIBLE; Complex → COMPLEX_COMPATIBLE.
            loss: CategoricalParam::new(match model_type {
                ModelType::Real => LossChoice::REAL_COMPATIBLE.to_vec(),
                ModelType::Complex => LossChoice::COMPLEX_COMPATIBLE.to_vec(),
            })
            .name("loss"),
            grad_clip_norm: CategoricalParam::new(GradClipChoice::ALL.to_vec())
                .name("grad_clip_norm"),
            input_noise_std: FloatParam::new(noise_range.0, noise_range.1).name("input_noise_std"),

            scheduler: CategoricalParam::new(SchedulerChoice::ALL.to_vec()).name("scheduler"),

            plateau_factor: FloatParam::new(0.2, 0.8).name("plateau_factor"),
            plateau_patience: IntParam::new(2, 8).name("plateau_patience"),
            plateau_min_lr: FloatParam::new(1e-6, 1e-4)
                .log_scale()
                .name("plateau_min_lr"),

            cosine_eta_min: FloatParam::new(cos_eta_min_range.0, cos_eta_min_range.1)
                .log_scale()
                .name("cosine_eta_min"),

            sgd_momentum: FloatParam::new(0.0, 0.95).name("sgd_momentum"),
            sgd_nesterov: BoolParam::new().name("sgd_nesterov"),

            rmsprop_momentum: FloatParam::new(0.0, 0.95).name("rmsprop_momentum"),
            rmsprop_alpha: FloatParam::new(0.85, 0.99).name("rmsprop_alpha"),

            physics_lambda: FloatParam::new(1e-3, 1.0)
                .log_scale()
                .name("physics_lambda"),
        }
    }

    /// Best-effort variant of [`suggest`]: per-field tolerant,
    /// falls back to placeholders so the dashboard still shows the
    /// attempted config when the sampler errored.
    pub fn suggest_best_effort(
        &self,
        trial: &mut Trial,
    ) -> (HyperParams, Option<optimizer::Error>) {
        let placeholder_kind = match self.model_type {
            ModelType::Real => ModelKind::real_default(),
            ModelType::Complex => ModelKind::complex_default(),
        };
        let mut params = HyperParams::placeholder(placeholder_kind);
        let mut first_err: Option<optimizer::Error> = None;
        let mut record = |err: optimizer::Error| {
            if first_err.is_none() {
                first_err = Some(err);
            }
        };

        match self.hidden_size.suggest(trial) {
            Ok(v) => params.hidden_size = v,
            Err(e) => record(e),
        }
        // String → typed enum per ModelKind.
        match self.activation.suggest(trial) {
            Ok(v) => match &mut params.kind {
                ModelKind::Real { activation, .. } => {
                    if let Ok(act) = Activation::from_name(&v) {
                        *activation = act;
                    }
                }
                ModelKind::Complex { activation, .. } => {
                    if let Ok(act) = ComplexActivation::from_name(&v) {
                        *activation = act;
                    }
                }
            },
            Err(e) => record(e),
        }
        match self.optimizer.suggest(trial) {
            Ok(v) => params.optimizer = v,
            Err(e) => record(e),
        }
        match self.lr.suggest(trial) {
            Ok(v) => params.lr = v,
            Err(e) => record(e),
        }
        match self.train_batch_size.suggest(trial) {
            Ok(v) => params.train_batch_size = v,
            Err(e) => record(e),
        }
        // WD, dropout, grad-clip, input-noise apply to both families;
        // only `norm` is Real-only.
        match self.weight_decay.suggest(trial) {
            Ok(v) => params.weight_decay = v,
            Err(e) => record(e),
        }
        let dropout_sample = self.dropout_p.suggest(trial);
        let grad_clip_sample = self.grad_clip_norm.suggest(trial);
        let noise_sample = self.input_noise_std.suggest(trial);
        match &mut params.kind {
            ModelKind::Real {
                dropout_p,
                norm,
                grad_clip_norm,
                input_noise_std,
                ..
            } => {
                match dropout_sample {
                    Ok(v) => *dropout_p = v,
                    Err(e) => record(e),
                }
                match self.norm.suggest(trial) {
                    Ok(v) => *norm = v,
                    Err(e) => record(e),
                }
                match grad_clip_sample {
                    Ok(v) => *grad_clip_norm = v,
                    Err(e) => record(e),
                }
                match noise_sample {
                    Ok(v) => *input_noise_std = v,
                    Err(e) => record(e),
                }
            }
            ModelKind::Complex {
                dropout_p,
                grad_clip_norm,
                input_noise_std,
                ..
            } => {
                match dropout_sample {
                    Ok(v) => *dropout_p = v,
                    Err(e) => record(e),
                }
                match grad_clip_sample {
                    Ok(v) => *grad_clip_norm = v,
                    Err(e) => record(e),
                }
                match noise_sample {
                    Ok(v) => *input_noise_std = v,
                    Err(e) => record(e),
                }
            }
        }
        match self.loss.suggest(trial) {
            Ok(v) => params.loss = v,
            Err(e) => record(e),
        }
        match self.scheduler.suggest(trial) {
            Ok(v) => params.scheduler = v,
            Err(e) => record(e),
        }

        // Sample unconditionally; only retain matching variant below.
        let plateau_factor = self.plateau_factor.suggest(trial).ok();
        let plateau_patience = self.plateau_patience.suggest(trial).ok();
        let plateau_min_lr = self.plateau_min_lr.suggest(trial).ok();
        let cosine_eta_min = self.cosine_eta_min.suggest(trial).ok();
        let sgd_momentum = self.sgd_momentum.suggest(trial).ok();
        let sgd_nesterov = self.sgd_nesterov.suggest(trial).ok();
        let rmsprop_momentum = self.rmsprop_momentum.suggest(trial).ok();
        let rmsprop_alpha = self.rmsprop_alpha.suggest(trial).ok();

        params.loss_params = LossHyperParams::default();

        params.scheduler_params = match params.scheduler {
            SchedulerChoice::Plateau => SchedulerHyperParams {
                plateau_factor,
                plateau_patience,
                plateau_min_lr,
                cosine_eta_min: None,
            },
            SchedulerChoice::Cosine => SchedulerHyperParams {
                plateau_factor: None,
                plateau_patience: None,
                plateau_min_lr: None,
                cosine_eta_min,
            },
            SchedulerChoice::None => SchedulerHyperParams::default(),
        };

        params.optimizer_params = match params.optimizer {
            OptimizerChoice::SGD => OptimizerHyperParams {
                sgd_momentum,
                sgd_nesterov,
                rmsprop_momentum: None,
                rmsprop_alpha: None,
            },
            OptimizerChoice::RMSprop => OptimizerHyperParams {
                sgd_momentum: None,
                sgd_nesterov: None,
                rmsprop_momentum,
                rmsprop_alpha,
            },
            _ => OptimizerHyperParams::default(),
        };

        (params, first_err)
    }

    pub fn suggest(&self, trial: &mut Trial) -> optimizer::Result<HyperParams> {
        let hidden_size = self.hidden_size.suggest(trial)?;
        let activation_name = self.activation.suggest(trial)?;

        let optimizer = self.optimizer.suggest(trial)?;
        let lr = self.lr.suggest(trial)?;
        let train_batch_size = self.train_batch_size.suggest(trial)?;

        let weight_decay = self.weight_decay.suggest(trial)?;
        let loss = self.loss.suggest(trial)?;

        // Activation strings from `self.activation` parse into typed
        // enums per family; unknown names error out loudly.
        let kind = match self.model_type {
            ModelType::Real => ModelKind::Real {
                activation: Activation::from_name(&activation_name).map_err(|e| {
                    optimizer::Error::ParameterConflict {
                        name: "activation".into(),
                        reason: format!("unknown real activation '{activation_name}': {e}"),
                    }
                })?,
                dropout_p: self.dropout_p.suggest(trial)?,
                norm: self.norm.suggest(trial)?,
                grad_clip_norm: self.grad_clip_norm.suggest(trial)?,
                input_noise_std: self.input_noise_std.suggest(trial)?,
            },
            ModelType::Complex => ModelKind::Complex {
                activation: ComplexActivation::from_name(&activation_name).map_err(
                    |e| optimizer::Error::ParameterConflict {
                        name: "activation".into(),
                        reason: format!(
                            "unknown complex activation '{activation_name}': {e}"
                        ),
                    },
                )?,
                dropout_p: self.dropout_p.suggest(trial)?,
                grad_clip_norm: self.grad_clip_norm.suggest(trial)?,
                input_noise_std: self.input_noise_std.suggest(trial)?,
            },
        };

        let scheduler = self.scheduler.suggest(trial)?;

        // Sample everything unconditionally so NSGA-II sees a fixed
        // dimension; conditional retention happens below.
        let plateau_factor = self.plateau_factor.suggest(trial)?;
        let plateau_patience = self.plateau_patience.suggest(trial)?;
        let plateau_min_lr = self.plateau_min_lr.suggest(trial)?;
        let cosine_eta_min = self.cosine_eta_min.suggest(trial)?;
        let sgd_momentum = self.sgd_momentum.suggest(trial)?;
        let sgd_nesterov = self.sgd_nesterov.suggest(trial)?;
        let rmsprop_momentum = self.rmsprop_momentum.suggest(trial)?;
        let rmsprop_alpha = self.rmsprop_alpha.suggest(trial)?;
        let physics_lambda = self.physics_lambda.suggest(trial)?;

        let scheduler_params = match scheduler {
            SchedulerChoice::Plateau => SchedulerHyperParams {
                plateau_factor: Some(plateau_factor),
                plateau_patience: Some(plateau_patience),
                plateau_min_lr: Some(plateau_min_lr),
                cosine_eta_min: None,
            },
            SchedulerChoice::Cosine => SchedulerHyperParams {
                plateau_factor: None,
                plateau_patience: None,
                plateau_min_lr: None,
                cosine_eta_min: Some(cosine_eta_min),
            },
            SchedulerChoice::None => SchedulerHyperParams::default(),
        };

        let optimizer_params = match optimizer {
            OptimizerChoice::SGD => OptimizerHyperParams {
                sgd_momentum: Some(sgd_momentum),
                sgd_nesterov: Some(sgd_nesterov),
                rmsprop_momentum: None,
                rmsprop_alpha: None,
            },
            OptimizerChoice::RMSprop => OptimizerHyperParams {
                sgd_momentum: None,
                sgd_nesterov: None,
                rmsprop_momentum: Some(rmsprop_momentum),
                rmsprop_alpha: Some(rmsprop_alpha),
            },
            _ => OptimizerHyperParams::default(),
        };

        let loss_params = match loss {
            LossChoice::PhysicsForward => LossHyperParams {
                physics_lambda: Some(physics_lambda),
                ..LossHyperParams::default()
            },
            _ => LossHyperParams::default(),
        };

        Ok(HyperParams {
            hidden_size,
            optimizer,
            lr,
            train_batch_size,
            weight_decay,
            loss,
            loss_params,
            scheduler,
            scheduler_params,
            optimizer_params,
            kind,
        })
    }
}

/// Activation is typed per family; only `norm` stays Real-only.
/// Complex defaults zero so legacy JSON round-trips via `#[serde(default)]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "model_type", rename_all = "lowercase")]
pub enum ModelKind {
    Real {
        #[serde(with = "sparam_models::activation::as_name")]
        activation: Activation,
        dropout_p: f64,
        norm: NormChoice,
        grad_clip_norm: GradClipChoice,
        input_noise_std: f64,
    },
    Complex {
        activation: ComplexActivation,
        #[serde(default)]
        dropout_p: f64,
        #[serde(default)]
        grad_clip_norm: GradClipChoice,
        #[serde(default)]
        input_noise_std: f64,
    },
}

impl ModelKind {
    pub fn real_default() -> Self {
        Self::Real {
            activation: Activation::GELU,
            dropout_p: 0.0,
            norm: NormChoice::None,
            grad_clip_norm: GradClipChoice::None,
            input_noise_std: 0.0,
        }
    }

    pub fn complex_default() -> Self {
        Self::Complex {
            activation: ComplexActivation::default(),
            dropout_p: 0.0,
            grad_clip_norm: GradClipChoice::None,
            input_noise_std: 0.0,
        }
    }

    pub fn is_complex(&self) -> bool {
        matches!(self, Self::Complex { .. })
    }

    #[must_use]
    pub fn activation_name(&self) -> &'static str {
        match self {
            Self::Real { activation, .. } => activation.name(),
            Self::Complex { activation, .. } => activation.name(),
        }
    }
}

/// A concrete sampled config. `kind` carries the family-specific
/// typed activation + regularization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperParams {
    pub hidden_size: i64,

    pub optimizer: OptimizerChoice,
    pub lr: f64,
    pub train_batch_size: BatchSizeChoice,

    pub weight_decay: f64,
    pub loss: LossChoice,

    #[serde(default)]
    pub loss_params: LossHyperParams,

    pub scheduler: SchedulerChoice,
    pub scheduler_params: SchedulerHyperParams,

    pub optimizer_params: OptimizerHyperParams,

    #[serde(flatten)]
    pub kind: ModelKind,
}

impl HyperParams {
    /// All-zero placeholder so sampler-error trials still occupy a
    /// slot in the dashboard.
    pub fn placeholder(kind: ModelKind) -> Self {
        let loss = if kind.is_complex() {
            LossChoice::PhysicsForward
        } else {
            LossChoice::Mse
        };
        let loss_params = LossHyperParams::default();
        Self {
            hidden_size: 0,
            optimizer: OptimizerChoice::Adam,
            lr: 0.0,
            train_batch_size: BatchSizeChoice::All,
            weight_decay: 0.0,
            loss,
            loss_params,
            scheduler: SchedulerChoice::None,
            scheduler_params: SchedulerHyperParams::default(),
            optimizer_params: OptimizerHyperParams::default(),
            kind,
        }
    }

    pub fn is_complex(&self) -> bool {
        self.kind.is_complex()
    }

    #[must_use]
    pub fn activation_name(&self) -> &'static str {
        self.kind.activation_name()
    }
}

/// **Persisted wire format** — keep nullable + flat, or old runs stop
/// replaying. Construct via `cosine()`/`plateau()`/`none()` and read via
/// [`Self::view`] for typed access.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerHyperParams {
    pub plateau_factor: Option<f64>,
    pub plateau_patience: Option<i64>,
    pub plateau_min_lr: Option<f64>,

    pub cosine_eta_min: Option<f64>,
}

impl SchedulerHyperParams {
    #[must_use]
    pub fn cosine(eta_min: f64) -> Self {
        Self { cosine_eta_min: Some(eta_min), ..Self::default() }
    }

    /// Build a Plateau config with reduce-factor + patience epochs +
    /// learning-rate floor.  Other variants' fields stay `None`.
    #[must_use]
    pub fn plateau(factor: f64, patience: i64, min_lr: f64) -> Self {
        Self {
            plateau_factor: Some(factor),
            plateau_patience: Some(patience),
            plateau_min_lr: Some(min_lr),
            cosine_eta_min: None,
        }
    }

    /// Empty config for `SchedulerChoice::None`.
    #[must_use]
    pub fn none() -> Self { Self::default() }

    /// Pair the flat hyperparam bag with a [`SchedulerChoice`] tag and
    /// return a typed [`SchedulerView`] for consumers.  Replaces the
    /// `match scheduler { Cosine => params.cosine_eta_min.unwrap_or(...) }`
    /// pattern with a single match that returns the right scalars.
    ///
    #[must_use]
    pub fn view(&self, choice: SchedulerChoice) -> SchedulerView {
        match choice {
            SchedulerChoice::None => SchedulerView::None,
            SchedulerChoice::Cosine => SchedulerView::Cosine {
                eta_min: self.cosine_eta_min.unwrap_or(1e-6),
            },
            SchedulerChoice::Plateau => SchedulerView::Plateau {
                factor: self.plateau_factor.unwrap_or(0.5),
                patience: self.plateau_patience.unwrap_or(5),
                min_lr: self.plateau_min_lr.unwrap_or(1e-7),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SchedulerView {
    None,
    Cosine { eta_min: f64 },
    Plateau { factor: f64, patience: i64, min_lr: f64 },
}

/// Same persistence + type-safety story as [`SchedulerHyperParams`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizerHyperParams {
    pub sgd_momentum: Option<f64>,
    pub sgd_nesterov: Option<bool>,

    pub rmsprop_momentum: Option<f64>,
    pub rmsprop_alpha: Option<f64>,
}

impl OptimizerHyperParams {
    #[must_use]
    pub fn sgd(momentum: f64, nesterov: bool) -> Self {
        Self {
            sgd_momentum: Some(momentum),
            sgd_nesterov: Some(nesterov),
            ..Self::default()
        }
    }

    #[must_use]
    pub fn rmsprop(momentum: f64, alpha: f64) -> Self {
        Self {
            rmsprop_momentum: Some(momentum),
            rmsprop_alpha: Some(alpha),
            ..Self::default()
        }
    }

    /// Adam/AdamW carry only `weight_decay`, stored at the top level.
    #[must_use]
    pub fn adam_like() -> Self { Self::default() }

    #[must_use]
    pub fn view(&self, choice: OptimizerChoice) -> OptimizerView {
        match choice {
            OptimizerChoice::AdamW => OptimizerView::AdamW,
            OptimizerChoice::Adam => OptimizerView::Adam,
            OptimizerChoice::SGD => OptimizerView::SGD {
                momentum: self.sgd_momentum.unwrap_or(0.0),
                nesterov: self.sgd_nesterov.unwrap_or(false),
            },
            OptimizerChoice::RMSprop => OptimizerView::RMSprop {
                momentum: self.rmsprop_momentum.unwrap_or(0.0),
                alpha: self.rmsprop_alpha.unwrap_or(0.99),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizerView {
    AdamW,
    Adam,
    SGD { momentum: f64, nesterov: bool },
    RMSprop { momentum: f64, alpha: f64 },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LossHyperParams {
    /// Legacy PI-MAPE β weight, read-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pi_mape_beta: Option<f64>,

    /// PhysicsForward mixing weight `λ`, sampled from log [1e-3, 1].
    /// Loss factory defaults to `1e-3` when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physics_lambda: Option<f64>,
}

impl std::fmt::Display for HyperParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HyperParams {{ hidden={}, act={}, opt={:?}, lr={:.6}, bs={:?} }}",
            self.hidden_size,
            self.activation_name(),
            self.optimizer,
            self.lr,
            self.train_batch_size,
        )
    }
}

// HyperParams → model / optimizer / scheduler / trainer bridges.

impl From<NormChoice> for Normalization {
    #[inline]
    fn from(choice: NormChoice) -> Self {
        match choice {
            NormChoice::None => Normalization::None,
            NormChoice::LayerNorm => Normalization::LayerNorm,
            NormChoice::BatchNorm => Normalization::BatchNorm,
        }
    }
}

impl From<BatchSizeChoice> for BatchSize {
    #[inline]
    fn from(choice: BatchSizeChoice) -> Self {
        match choice.to_usize() {
            Some(size) => BatchSize::Fixed(size),
            None => BatchSize::All,
        }
    }
}

impl HyperParams {
    /// `self.kind` selects the model family; callers don't dispatch.
    pub fn build_mlp(
        &self,
        input_size: usize,
        output_size: usize,
        vb: VarBuilder,
    ) -> Result<MlpModel> {
        match &self.kind {
            ModelKind::Real { activation, dropout_p, norm, .. } => {
                let config = MLPConfig::new(
                    input_size,
                    self.hidden_size as usize,
                    output_size,
                    *activation,
                )
                .with_dropout(*dropout_p as f32)
                .with_norm(Normalization::from(*norm));
                Ok(MlpModel::Real(MLPRegressor::new(vb, &config)?))
            }
            ModelKind::Complex { activation, dropout_p, .. } => {
                // LayerNorm always on: ablation showed ~13 pp gap on
                // ok@1% at the same parameter budget.
                let config = ComplexMLPConfig::new(
                    input_size,
                    self.hidden_size as usize,
                    output_size,
                    *activation,
                )
                .with_dropout(*dropout_p as f32)
                .with_norm(ComplexNormChoice::LayerNorm);
                Ok(MlpModel::Complex(ComplexMLPRegressor::new(vb, &config)?))
            }
        }
    }

    pub fn optimizer_config(&self) -> OptimizerConfig {
        let weight_decay = self.weight_decay;
        match self.optimizer_params.view(self.optimizer) {
            OptimizerView::Adam => OptimizerConfig::adam(self.lr),
            OptimizerView::AdamW => OptimizerConfig::adamw(self.lr, weight_decay),
            OptimizerView::SGD { momentum, nesterov } => {
                OptimizerConfig::sgd_with(self.lr, weight_decay, momentum, nesterov)
            }
            OptimizerView::RMSprop { momentum, alpha } => {
                OptimizerConfig::rmsprop_with(self.lr, weight_decay, momentum, alpha)
            }
        }
    }

    pub fn trainer_config(
        &self,
        max_epochs: usize,
        patience: usize,
        warmup_epochs: usize,
    ) -> TrainerConfig {
        let (grad_clip_norm, input_noise_std_value) = match &self.kind {
            ModelKind::Real { grad_clip_norm, input_noise_std, .. } => {
                (*grad_clip_norm, *input_noise_std)
            }
            ModelKind::Complex { grad_clip_norm, input_noise_std, .. } => {
                (*grad_clip_norm, *input_noise_std)
            }
        };
        let max_grad_norm = grad_clip_norm.to_f64();
        let input_noise_std = (input_noise_std_value > 0.0).then_some(input_noise_std_value);
        TrainerConfig {
            max_epochs,
            early_stopping: Some(EarlyStoppingConfig {
                patience,
                warmup_epochs,
                ..EarlyStoppingConfig::default()
            }),
            max_grad_norm,
            input_noise_std,
            log_interval: 0,
            ..Default::default()
        }
    }

    pub fn scheduler_config(&self, max_epochs: usize) -> SchedulerConfig {
        self.scheduler.to_scheduler_config(
            max_epochs,
            self.scheduler_params.cosine_eta_min,
            self.scheduler_params.plateau_factor,
            self.scheduler_params.plateau_patience.map(|p| p as usize),
            self.scheduler_params.plateau_min_lr,
        )
    }

    pub fn lr_scheduler(&self, max_epochs: usize) -> Result<Option<LRScheduler>> {
        self.scheduler_config(max_epochs).build(self.lr)
    }

    pub fn trainer_builder(
        &self,
        max_epochs: usize,
        patience: usize,
        warmup_epochs: usize,
    ) -> Result<TrainerBuilder> {
        Ok(TrainerBuilder::from_config(self.trainer_config(max_epochs, patience, warmup_epochs))
            .with_scheduler(self.lr_scheduler(max_epochs)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_display_matches_summary_labels() {
        assert_eq!(ActivationChoice::LeakyRelu.to_string(), "lrelu");
        assert_eq!(OptimizerChoice::RMSprop.to_string(), "rmspr");
        assert_eq!(SchedulerChoice::Plateau.to_string(), "plat");
        assert_eq!(BatchSizeChoice::All.to_string(), "all");
        assert_eq!(NormChoice::BatchNorm.to_string(), "btch");
        assert_eq!(LossChoice::EpsRelMse.to_string(), "ermse");
    }

    #[test]
    fn choice_from_index_matches_enum_order() {
        assert_eq!(ActivationChoice::from_index(2), Some(ActivationChoice::Gelu));
        assert_eq!(OptimizerChoice::from_index(3), Some(OptimizerChoice::SGD));
        assert_eq!(SchedulerChoice::from_index(2), Some(SchedulerChoice::Cosine));
        assert_eq!(BatchSizeChoice::from_index(7), Some(BatchSizeChoice::All));
        assert_eq!(NormChoice::from_index(1), Some(NormChoice::LayerNorm));
        assert_eq!(LossChoice::from_index(0), Some(LossChoice::Mse));
        assert_eq!(ActivationChoice::from_index(99), None);
    }

    fn run_trials(space: &MlpSearchSpace, n: usize) -> Vec<HyperParams> {
        let study: Study<f64> = Study::new(Direction::Minimize);
        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            let mut trial = study.ask();
            let params = space.suggest(&mut trial).unwrap();
            results.push(params);
            study.tell(trial, Ok::<_, &str>(0.0));
        }
        results
    }

    fn run_seeded_random_trials(space: &MlpSearchSpace, n: usize, seed: u64) -> Vec<HyperParams> {
        let mut study: Study<f64> = Study::new(Direction::Minimize);
        study.set_sampler(RandomSampler::with_seed(seed));

        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            let mut trial = study.ask();
            let params = space.suggest(&mut trial).unwrap();
            results.push(params);
            study.tell(trial, Ok::<_, &str>(0.0));
        }
        results
    }

    fn assert_same_params(lhs: &HyperParams, rhs: &HyperParams) {
        assert_eq!(lhs.hidden_size, rhs.hidden_size);
        assert_eq!(lhs.activation_name(), rhs.activation_name());
        assert_eq!(lhs.optimizer, rhs.optimizer);
        assert_eq!(lhs.train_batch_size, rhs.train_batch_size);
        assert_eq!(lhs.loss, rhs.loss);
        assert_eq!(lhs.scheduler, rhs.scheduler);
        assert_eq!(lhs.lr.to_bits(), rhs.lr.to_bits());
        assert_eq!(lhs.weight_decay.to_bits(), rhs.weight_decay.to_bits());
        match (&lhs.kind, &rhs.kind) {
            (
                ModelKind::Real {
                    activation: la,
                    dropout_p: ld, norm: ln, grad_clip_norm: lg, input_noise_std: li,
                },
                ModelKind::Real {
                    activation: ra,
                    dropout_p: rd, norm: rn, grad_clip_norm: rg, input_noise_std: ri,
                },
            ) => {
                assert_eq!(la, ra);
                assert_eq!(ln, rn);
                assert_eq!(lg, rg);
                assert_eq!(ld.to_bits(), rd.to_bits());
                assert_eq!(li.to_bits(), ri.to_bits());
            }
            (
                ModelKind::Complex {
                    activation: la,
                    dropout_p: ld,
                    grad_clip_norm: lg,
                    input_noise_std: li,
                },
                ModelKind::Complex {
                    activation: ra,
                    dropout_p: rd,
                    grad_clip_norm: rg,
                    input_noise_std: ri,
                },
            ) => {
                assert_eq!(la, ra);
                assert_eq!(ld.to_bits(), rd.to_bits());
                assert_eq!(lg, rg);
                assert_eq!(li.to_bits(), ri.to_bits());
            }
            _ => panic!("ModelKind variant mismatch between lhs/rhs"),
        }
        assert_eq!(
            lhs.scheduler_params.plateau_factor.map(f64::to_bits),
            rhs.scheduler_params.plateau_factor.map(f64::to_bits)
        );
        assert_eq!(
            lhs.scheduler_params.plateau_patience,
            rhs.scheduler_params.plateau_patience
        );
        assert_eq!(
            lhs.scheduler_params.plateau_min_lr.map(f64::to_bits),
            rhs.scheduler_params.plateau_min_lr.map(f64::to_bits)
        );
        assert_eq!(
            lhs.scheduler_params.cosine_eta_min.map(f64::to_bits),
            rhs.scheduler_params.cosine_eta_min.map(f64::to_bits)
        );
        assert_eq!(
            lhs.optimizer_params.sgd_momentum.map(f64::to_bits),
            rhs.optimizer_params.sgd_momentum.map(f64::to_bits)
        );
        assert_eq!(
            lhs.optimizer_params.sgd_nesterov,
            rhs.optimizer_params.sgd_nesterov
        );
        assert_eq!(
            lhs.optimizer_params.rmsprop_momentum.map(f64::to_bits),
            rhs.optimizer_params.rmsprop_momentum.map(f64::to_bits)
        );
        assert_eq!(
            lhs.optimizer_params.rmsprop_alpha.map(f64::to_bits),
            rhs.optimizer_params.rmsprop_alpha.map(f64::to_bits)
        );
    }

    #[test]
    fn suggest_returns_valid_hyperparams() {
        let space = MlpSearchSpace::new();
        let all = run_trials(&space, 10);
        for params in &all {
            assert!([8, 16, 32, 64].contains(&params.hidden_size));
            assert!(params.lr >= 1e-4 && params.lr <= 5e-2);
            let ModelKind::Real { dropout_p, .. } = &params.kind else {
                panic!("default search space is Real — expected ModelKind::Real");
            };
            assert!(*dropout_p >= 0.0 && *dropout_p <= 0.4);
        }
    }

    #[test]
    fn suggest_populates_all_conditional_params_for_sampled_variants() {
        let space = MlpSearchSpace::new();
        let all = run_trials(&space, 100);

        let (mut saw_sgd, mut saw_rmsprop) = (false, false);
        let (mut saw_plateau, mut saw_cosine) = (false, false);

        for params in &all {
            match params.optimizer {
                OptimizerChoice::SGD => {
                    assert!(params.optimizer_params.sgd_momentum.is_some());
                    assert!(params.optimizer_params.sgd_nesterov.is_some());
                    saw_sgd = true;
                }
                OptimizerChoice::RMSprop => {
                    assert!(params.optimizer_params.rmsprop_momentum.is_some());
                    assert!(params.optimizer_params.rmsprop_alpha.is_some());
                    saw_rmsprop = true;
                }
                _ => {}
            }
            match params.scheduler {
                SchedulerChoice::Plateau => {
                    assert!(params.scheduler_params.plateau_factor.is_some());
                    assert!(params.scheduler_params.plateau_patience.is_some());
                    assert!(params.scheduler_params.plateau_min_lr.is_some());
                    saw_plateau = true;
                }
                SchedulerChoice::Cosine => {
                    assert!(params.scheduler_params.cosine_eta_min.is_some());
                    saw_cosine = true;
                }
                _ => {}
            }
        }

        assert!(saw_sgd, "SGD was never sampled in 100 trials");
        assert!(saw_rmsprop, "RMSprop was never sampled in 100 trials");
        assert!(saw_plateau, "Plateau was never sampled in 100 trials");
        assert!(saw_cosine, "Cosine was never sampled in 100 trials");
    }

    #[test]
    fn seeded_random_sampling_covers_categorical_and_enum_choices() {
        let space = MlpSearchSpace::new();
        let all = run_seeded_random_trials(&space, 512, 42);

        for hidden_size in [8, 16, 32, 64] {
            assert!(
                all.iter().any(|params| params.hidden_size == hidden_size),
                "hidden_size={hidden_size} was never sampled"
            );
        }

        for activation in ["relu", "tanh", "gelu", "leaky_relu", "elu"] {
            assert!(
                all.iter().any(|params| params.activation_name() == activation),
                "activation variant {:?} was never sampled",
                activation
            );
        }

        for optimizer in [
            OptimizerChoice::Adam,
            OptimizerChoice::AdamW,
            OptimizerChoice::RMSprop,
            OptimizerChoice::SGD,
        ] {
            assert!(
                all.iter().any(|params| params.optimizer == optimizer),
                "optimizer variant {:?} was never sampled",
                optimizer
            );
        }

        for scheduler in [
            SchedulerChoice::None,
            SchedulerChoice::Plateau,
            SchedulerChoice::Cosine,
        ] {
            assert!(
                all.iter().any(|params| params.scheduler == scheduler),
                "scheduler variant {:?} was never sampled",
                scheduler
            );
        }
    }

    #[test]
    fn seeded_random_sampling_spans_numeric_ranges() {
        let space = MlpSearchSpace::new();
        let all = run_seeded_random_trials(&space, 1024, 7);

        let min_lr = all
            .iter()
            .map(|params| params.lr)
            .fold(f64::INFINITY, f64::min);
        let max_lr = all
            .iter()
            .map(|params| params.lr)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_weight_decay = all
            .iter()
            .map(|params| params.weight_decay)
            .fold(f64::INFINITY, f64::min);
        let max_weight_decay = all
            .iter()
            .map(|params| params.weight_decay)
            .fold(f64::NEG_INFINITY, f64::max);
        let dropout = |p: &HyperParams| match &p.kind {
            ModelKind::Real { dropout_p, .. } => *dropout_p,
            ModelKind::Complex { .. } => 0.0,
        };
        let min_dropout = all.iter().map(dropout).fold(f64::INFINITY, f64::min);
        let max_dropout = all.iter().map(dropout).fold(f64::NEG_INFINITY, f64::max);

        assert!(
            min_lr < 5e-4,
            "log-scale lr never reached lower end: {min_lr}"
        );
        assert!(
            max_lr > 1e-2,
            "log-scale lr never reached upper end: {max_lr}"
        );
        assert!(
            min_weight_decay < 1e-8,
            "log-scale weight_decay never reached lower orders: {min_weight_decay}"
        );
        assert!(
            max_weight_decay > 1e-3,
            "log-scale weight_decay never reached upper orders: {max_weight_decay}"
        );
        assert!(
            min_dropout < 0.05,
            "uniform dropout never reached lower end: {min_dropout}"
        );
        assert!(
            max_dropout > 0.35,
            "uniform dropout never reached upper end: {max_dropout}"
        );
    }

    #[test]
    fn seeded_random_sampling_exercises_integer_and_boolean_conditionals() {
        let space = MlpSearchSpace::new();
        let all = run_seeded_random_trials(&space, 1024, 99);

        let plateau_patiences: Vec<_> = all
            .iter()
            .filter(|params| params.scheduler == SchedulerChoice::Plateau)
            .filter_map(|params| params.scheduler_params.plateau_patience)
            .collect();
        let sgd_nesterovs: Vec<_> = all
            .iter()
            .filter(|params| params.optimizer == OptimizerChoice::SGD)
            .filter_map(|params| params.optimizer_params.sgd_nesterov)
            .collect();

        assert!(
            !plateau_patiences.is_empty(),
            "plateau scheduler was never sampled"
        );
        assert!(!sgd_nesterovs.is_empty(), "SGD optimizer was never sampled");
        assert!(plateau_patiences.iter().any(|&value| value <= 3));
        assert!(plateau_patiences.iter().any(|&value| value >= 7));
        assert!(sgd_nesterovs.iter().any(|&value| value));
        assert!(sgd_nesterovs.iter().any(|&value| !value));
    }

    #[test]
    fn seeded_random_sampling_is_reproducible() {
        let space = MlpSearchSpace::new();
        let run_a = run_seeded_random_trials(&space, 16, 123);
        let run_b = run_seeded_random_trials(&space, 16, 123);

        for (lhs, rhs) in run_a.iter().zip(run_b.iter()) {
            assert_same_params(lhs, rhs);
        }
    }

    #[test]
    fn hyperparams_serde_roundtrip() {
        let params = HyperParams {
            hidden_size: 32,
            optimizer: OptimizerChoice::AdamW,
            lr: 0.001,
            train_batch_size: BatchSizeChoice::B64,
            weight_decay: 1e-5,
            loss: LossChoice::Mse,
            loss_params: LossHyperParams::default(),
            scheduler: SchedulerChoice::Cosine,
            scheduler_params: SchedulerHyperParams {
                cosine_eta_min: Some(1e-5),
                ..Default::default()
            },
            optimizer_params: OptimizerHyperParams::default(),
            kind: ModelKind::Real {
                activation: Activation::GELU,
                dropout_p: 0.1,
                norm: NormChoice::LayerNorm,
                grad_clip_norm: GradClipChoice::Clip10,
                input_noise_std: 0.005,
            },
        };

        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains(r#""activation":"gelu""#));
        let restored: HyperParams = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.hidden_size, 32);
        assert_eq!(restored.activation_name(), "gelu");
        assert_eq!(restored.optimizer, OptimizerChoice::AdamW);
        assert!((restored.lr - 0.001).abs() < 1e-12);
        assert_eq!(restored.train_batch_size, BatchSizeChoice::B64);
        assert_eq!(restored.scheduler, SchedulerChoice::Cosine);
        assert!(restored.scheduler_params.cosine_eta_min.is_some());
        let ModelKind::Real { activation, dropout_p, .. } = restored.kind else {
            panic!("expected Real variant after roundtrip");
        };
        assert_eq!(activation, Activation::GELU);
        assert!((dropout_p - 0.1).abs() < 1e-12);
    }

    #[test]
    fn hyperparams_complex_serde_carries_shared_regularization() {
        let params = HyperParams {
            hidden_size: 24,
            optimizer: OptimizerChoice::AdamW,
            lr: 0.003,
            train_batch_size: BatchSizeChoice::B64,
            weight_decay: 1e-5,
            loss: LossChoice::Mse,
            loss_params: LossHyperParams::default(),
            scheduler: SchedulerChoice::None,
            scheduler_params: SchedulerHyperParams::default(),
            optimizer_params: OptimizerHyperParams::default(),
            kind: ModelKind::Complex {
                activation: ComplexActivation::CGELU,
                dropout_p: 0.25,
                grad_clip_norm: GradClipChoice::Clip10,
                input_noise_std: 0.01,
            },
        };

        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains(r#""model_type":"complex""#));
        assert!(json.contains(r#""activation":"cgelu""#));
        assert!(json.contains(r#""dropout_p":0.25"#));
        assert!(json.contains("grad_clip_norm"));
        assert!(json.contains(r#""input_noise_std":0.01"#));
        assert!(!json.contains("\"norm\""));

        let restored: HyperParams = serde_json::from_str(&json).unwrap();
        assert!(restored.is_complex());
        assert_eq!(restored.activation_name(), "cgelu");
        match restored.kind {
            ModelKind::Complex { dropout_p, grad_clip_norm, input_noise_std, .. } => {
                assert!((dropout_p - 0.25).abs() < 1e-12);
                assert_eq!(grad_clip_norm, GradClipChoice::Clip10);
                assert!((input_noise_std - 0.01).abs() < 1e-12);
            }
            _ => panic!("expected Complex variant"),
        }
    }

    #[test]
    fn legacy_complex_hyperparams_json_still_deserializes() {
        let legacy = r#"{
            "hidden_size": 24, "optimizer": "AdamW", "lr": 0.001,
            "train_batch_size": "64", "weight_decay": 0.0, "loss": "mse",
            "scheduler": "none", "scheduler_params": {},
            "optimizer_params": {},
            "model_type": "complex", "activation": "cgelu"
        }"#;
        let restored: HyperParams = serde_json::from_str(legacy).unwrap();
        match restored.kind {
            ModelKind::Complex { dropout_p, grad_clip_norm, input_noise_std, .. } => {
                assert_eq!(dropout_p, 0.0);
                assert_eq!(grad_clip_norm, GradClipChoice::None);
                assert_eq!(input_noise_std, 0.0);
            }
            _ => panic!("expected Complex variant"),
        }
    }

    #[test]
    fn batch_size_choice_to_usize() {
        assert_eq!(BatchSizeChoice::B8.to_usize(), Some(8));
        assert_eq!(BatchSizeChoice::B256.to_usize(), Some(256));
        assert_eq!(BatchSizeChoice::All.to_usize(), None);
    }

    #[test]
    fn grad_clip_choice_to_f64() {
        assert_eq!(GradClipChoice::None.to_f64(), None);
        assert_eq!(GradClipChoice::Clip10.to_f64(), Some(1.0));
        assert_eq!(GradClipChoice::Clip50.to_f64(), Some(5.0));
    }

    #[test]
    fn display_hyperparams() {
        let params = HyperParams {
            hidden_size: 32,
            optimizer: OptimizerChoice::AdamW,
            lr: 0.001,
            train_batch_size: BatchSizeChoice::B64,
            weight_decay: 1e-5,
            loss: LossChoice::Mse,
            loss_params: LossHyperParams::default(),
            scheduler: SchedulerChoice::None,
            scheduler_params: SchedulerHyperParams::default(),
            optimizer_params: OptimizerHyperParams::default(),
            kind: ModelKind::Real {
                activation: Activation::GELU,
                dropout_p: 0.1,
                norm: NormChoice::LayerNorm,
                grad_clip_norm: GradClipChoice::None,
                input_noise_std: 0.0,
            },
        };
        let s = format!("{params}");
        assert!(s.contains("hidden=32"));
        assert!(s.contains("gelu"));
    }

    #[test]
    fn no_conditional_params_for_adam() {
        let space = MlpSearchSpace::new();
        let all = run_trials(&space, 100);
        for params in &all {
            if params.optimizer == OptimizerChoice::Adam
                || params.optimizer == OptimizerChoice::AdamW
            {
                assert!(params.optimizer_params.sgd_momentum.is_none());
                assert!(params.optimizer_params.sgd_nesterov.is_none());
                assert!(params.optimizer_params.rmsprop_momentum.is_none());
                assert!(params.optimizer_params.rmsprop_alpha.is_none());
            }
        }
    }

    #[test]
    fn no_conditional_params_for_no_scheduler() {
        let space = MlpSearchSpace::new();
        let all = run_trials(&space, 100);
        for params in &all {
            if params.scheduler == SchedulerChoice::None {
                assert!(params.scheduler_params.plateau_factor.is_none());
                assert!(params.scheduler_params.cosine_eta_min.is_none());
            }
        }
    }

    #[test]
    fn complex_optimizer_config_honours_weight_decay() {
        let base = HyperParams {
            hidden_size: 24,
            optimizer: OptimizerChoice::AdamW,
            lr: 1e-3,
            train_batch_size: BatchSizeChoice::B64,
            weight_decay: 7.5e-4,
            loss: LossChoice::Mse,
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
        };

        match base.optimizer_config() {
            OptimizerConfig::AdamW(cfg) => assert_eq!(cfg.weight_decay, 7.5e-4),
            other => panic!("expected AdamW optimizer, got {other:?}"),
        }

        let sgd = HyperParams { optimizer: OptimizerChoice::SGD, ..base.clone() };
        match sgd.optimizer_config() {
            OptimizerConfig::Sgd(cfg) => assert_eq!(cfg.weight_decay, 7.5e-4),
            other => panic!("expected SGD, got {other:?}"),
        }

        let rms = HyperParams { optimizer: OptimizerChoice::RMSprop, ..base };
        match rms.optimizer_config() {
            OptimizerConfig::RMSprop(cfg) => assert_eq!(cfg.weight_decay, 7.5e-4),
            other => panic!("expected RMSprop, got {other:?}"),
        }
    }

    #[test]
    fn real_optimizer_config_honours_weight_decay() {
        let hp = HyperParams {
            hidden_size: 32,
            optimizer: OptimizerChoice::AdamW,
            lr: 1e-3,
            train_batch_size: BatchSizeChoice::B64,
            weight_decay: 7.5e-4,
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
        };

        match hp.optimizer_config() {
            OptimizerConfig::AdamW(cfg) => {
                assert!((cfg.weight_decay - 7.5e-4).abs() < f64::EPSILON);
            }
            other => panic!("expected AdamW, got {other:?}"),
        }
    }
}

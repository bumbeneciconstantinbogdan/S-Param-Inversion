//! US-10.2: `train` — parse CLI/TOML config and delegate orchestration to workflows.

use std::path::PathBuf;

use candle_core::Error as CandleError;
use clap::Args;
use serde::Deserialize;

use sparam_models::ModelType;

use crate::workflows::config::{ModelConfig, TrainConfig as WorkflowTrainConfig};
use crate::workflows::train::run_training;

use super::shared::load_toml_overlay;

/// Train command arguments.
///
/// All tunable fields are `Option<T>` so the same struct can be deserialized
/// from a TOML config file (absent fields → `None`) and parsed by Clap (absent
/// flags → `Some(default)`).  The TOML value takes precedence over the CLI
/// default; an explicit CLI flag still wins over an absent TOML key.
#[derive(Debug, Default, Args, Deserialize)]
pub struct TrainArgs {
    /// Directory containing train/val/test CSVs.
    #[arg(long, short = 'd', default_value = "./data")]
    #[serde(default)]
    pub data_dir: Option<PathBuf>,

    /// Directory to save model and logs.
    #[arg(long, short = 'o', default_value = "./artifacts")]
    #[serde(default)]
    pub output_dir: Option<PathBuf>,

    /// Model type: real or complex.
    #[arg(long, short = 'm', default_value = "real")]
    #[serde(default)]
    pub model_type: Option<String>,

    /// Hidden layer width.
    #[arg(long, short = 'H', default_value = "32")]
    #[serde(default)]
    pub hidden_size: Option<usize>,

    /// Activation function (for real models).
    #[arg(long, short = 'a', default_value = "gelu")]
    #[serde(default)]
    pub activation: Option<String>,

    /// Complex activation (for complex models).
    #[arg(long, default_value = "crelu")]
    #[serde(default)]
    pub complex_activation: Option<String>,

    /// Optimizer: adam, adamw, sgd, rmsprop.
    #[arg(long, default_value = "adamw")]
    #[serde(default)]
    pub optimizer: Option<String>,

    /// Learning rate.
    #[arg(long, default_value = "0.001")]
    #[serde(default)]
    pub lr: Option<f64>,

    /// Weight decay coefficient.
    #[arg(long, default_value = "0.0001")]
    #[serde(default)]
    pub weight_decay: Option<f64>,

    /// Training batch size (integer or ALL).
    #[arg(long, short = 'b', default_value = "256")]
    #[serde(default)]
    pub batch_size: Option<String>,

    /// Maximum training epochs.
    #[arg(long, short = 'e', default_value = "500")]
    #[serde(default)]
    pub max_epochs: Option<usize>,

    /// Early stopping patience.
    #[arg(long, default_value = "30")]
    #[serde(default)]
    pub patience: Option<usize>,

    /// Early stopping warmup epochs (delay monitoring).
    #[arg(long, default_value = "10")]
    #[serde(default)]
    pub warmup_epochs: Option<usize>,

    /// LR scheduler: none, cosine, plateau.
    #[arg(long, default_value = "cosine")]
    #[serde(default)]
    pub scheduler: Option<String>,

    /// Dropout probability (real AND complex — complex drops whole
    /// (real, imag) pairs pair-wise).
    #[arg(long, default_value = "0.0")]
    #[serde(default)]
    pub dropout: Option<f64>,

    /// Normalization (Real MLP only): none, layernorm, batchnorm.
    /// Ignored for `--model-type complex` — the Complex path has no
    /// normalization layer.
    #[arg(long, default_value = "layernorm")]
    #[serde(default)]
    pub norm: Option<String>,

    /// Loss function: mse, smooth_l1, eps_rel_mse, complex_rel_mse,
    /// log_mag_phase, log_huber_polar, pi_crl. Real MLPs must pick
    /// from the first three; Complex MLPs from the last four.
    #[arg(long, default_value = "mse")]
    #[serde(default)]
    pub loss: Option<String>,

    /// Max gradient norm for clipping. `None` / 0 disables clipping.
    /// Applies to both Real and Complex paths via the trainer's
    /// optimizer-agnostic clip step.
    #[arg(long)]
    #[serde(default)]
    pub grad_clip_norm: Option<f64>,

    /// Gaussian input-noise standard deviation. `None` / 0 disables.
    /// Applied to the packed `[Re, Im]` input before the forward
    /// pass — works identically for Real and Complex MLPs.
    #[arg(long)]
    #[serde(default)]
    pub input_noise_std: Option<f64>,

    /// Cosine scheduler minimum η. Only used when `--scheduler cosine`.
    #[arg(long)]
    #[serde(default)]
    pub cosine_eta_min: Option<f64>,

    /// Plateau scheduler decay factor. Only used when `--scheduler plateau`.
    #[arg(long)]
    #[serde(default)]
    pub plateau_factor: Option<f64>,

    /// Plateau scheduler patience (epochs). Only used when `--scheduler plateau`.
    #[arg(long)]
    #[serde(default)]
    pub plateau_patience: Option<usize>,

    /// Plateau scheduler minimum LR floor. Only used when `--scheduler plateau`.
    #[arg(long)]
    #[serde(default)]
    pub plateau_min_lr: Option<f64>,

    /// SGD momentum. Only used when `--optimizer sgd`.
    #[arg(long)]
    #[serde(default)]
    pub sgd_momentum: Option<f64>,

    /// Whether SGD uses Nesterov momentum. Only used when `--optimizer sgd`.
    #[arg(long)]
    #[serde(default)]
    pub sgd_nesterov: Option<bool>,

    /// RMSprop momentum. Only used when `--optimizer rmsprop`.
    #[arg(long)]
    #[serde(default)]
    pub rmsprop_momentum: Option<f64>,

    /// RMSprop α decay. Only used when `--optimizer rmsprop`.
    #[arg(long)]
    #[serde(default)]
    pub rmsprop_alpha: Option<f64>,

    /// PI-MAPE β — weight on the loss-tangent term. Only used when
    /// `--loss pi_mape` (Complex MLPs only). Range validated in
    /// [5, 100]; `None` falls back to the library default (30).
    #[arg(long)]
    #[serde(default)]
    pub pi_mape_beta: Option<f64>,

    /// Hybrid PhysicsForward mixing weight λ in
    /// `L = base + λ · NRW_residual`. HPO samples this from
    /// `log [1e-3, 1]` per trial. `None` falls back to the workflow
    /// default (`DEFAULT_PHYSICS_LAMBDA = 0.001`).
    #[arg(long)]
    #[serde(default)]
    pub physics_lambda: Option<f64>,

    /// Shuffle seed (DataLoader). Defaults to `--seed` when omitted.
    /// HPO trials pin this to the study seed; CLI `/train` runs that
    /// want bit-for-bit HPO parity should set it explicitly.
    #[arg(long)]
    #[serde(default)]
    pub shuffle_seed: Option<u64>,

    /// Experiment name for output sub-directory.
    #[arg(long, short = 'n')]
    #[serde(default)]
    pub name: Option<String>,

    /// Load settings from a TOML configuration file.
    #[arg(long, short = 'c')]
    #[serde(skip)]
    pub config: Option<PathBuf>,

    /// Resume from checkpoint.
    #[arg(long)]
    #[serde(skip)]
    pub checkpoint: Option<PathBuf>,
}

pub fn run(cli: &TrainArgs, seed: u64) -> Result<(), CandleError> {
    let toml: TrainArgs = load_toml_overlay(cli.config.as_deref(), "train")?;

    // TOML wins over CLI default; unwrap is safe because Clap always provides
    // a Some via default_value when the flag is absent.
    let data_dir = toml.data_dir.or(cli.data_dir.clone()).unwrap_or_else(|| "./data".into());
    let base_output_dir = toml.output_dir.or(cli.output_dir.clone()).unwrap_or_else(|| "./artifacts".into());
    let model_type = toml.model_type.or(cli.model_type.clone()).unwrap_or_else(|| "real".into());
    let hidden_size = toml.hidden_size.or(cli.hidden_size).unwrap_or(32);
    let activation = toml.activation.or(cli.activation.clone()).unwrap_or_else(|| "gelu".into());
    let complex_activation = toml.complex_activation.or(cli.complex_activation.clone()).unwrap_or_else(|| "crelu".into());
    let optimizer = toml.optimizer.or(cli.optimizer.clone()).unwrap_or_else(|| "adamw".into());
    let lr = toml.lr.or(cli.lr).unwrap_or(1e-3);
    let weight_decay = toml.weight_decay.or(cli.weight_decay).unwrap_or(1e-4);
    let batch_size = toml.batch_size.or(cli.batch_size.clone()).unwrap_or_else(|| "256".into());
    let max_epochs = toml.max_epochs.or(cli.max_epochs).unwrap_or(500);
    let patience = toml.patience.or(cli.patience).unwrap_or(30);
    let warmup_epochs = toml.warmup_epochs.or(cli.warmup_epochs).unwrap_or(10);
    let scheduler = toml.scheduler.or(cli.scheduler.clone()).unwrap_or_else(|| "cosine".into());
    let dropout = toml.dropout.or(cli.dropout).unwrap_or(0.0);
    let norm = toml.norm.or(cli.norm.clone()).unwrap_or_else(|| "layernorm".into());
    let loss = toml.loss.or(cli.loss.clone()).unwrap_or_else(|| "mse".into());
    // HPO-parity fields — each `.or(...)` lets TOML override the
    // matching CLI flag, and an explicit `None` at both levels falls
    // through to the trainer / optimizer defaults.
    let grad_clip_norm = toml.grad_clip_norm.or(cli.grad_clip_norm);
    let input_noise_std = toml.input_noise_std.or(cli.input_noise_std);
    let cosine_eta_min = toml.cosine_eta_min.or(cli.cosine_eta_min);
    let plateau_factor = toml.plateau_factor.or(cli.plateau_factor);
    let plateau_patience = toml.plateau_patience.or(cli.plateau_patience);
    let plateau_min_lr = toml.plateau_min_lr.or(cli.plateau_min_lr);
    let sgd_momentum = toml.sgd_momentum.or(cli.sgd_momentum);
    let sgd_nesterov = toml.sgd_nesterov.or(cli.sgd_nesterov);
    let rmsprop_momentum = toml.rmsprop_momentum.or(cli.rmsprop_momentum);
    let rmsprop_alpha = toml.rmsprop_alpha.or(cli.rmsprop_alpha);
    let pi_mape_beta = toml.pi_mape_beta.or(cli.pi_mape_beta);
    let physics_lambda = toml.physics_lambda.or(cli.physics_lambda);
    let shuffle_seed = toml.shuffle_seed.or(cli.shuffle_seed);

    let parsed_model_type: ModelType = model_type
        .parse()
        .map_err(|e: String| CandleError::Msg(e))?;

    let experiment_name = cli
        .name
        .as_deref()
        .or(toml.name.as_deref())
        .unwrap_or(match parsed_model_type {
            ModelType::Real => "real_train",
            ModelType::Complex => "complex_train",
        });

    // Load CSV samples from data_dir (CLI keeps CSV-based workflow).
    let train_samples = sparam_data::generation::load_samples_from_csv(
        &data_dir.join("data_train.csv"),
    )?;
    let val_samples = sparam_data::generation::load_samples_from_csv(
        &data_dir.join("data_val.csv"),
    )?;
    let test_samples = sparam_data::generation::load_samples_from_csv(
        &data_dir.join("data_test.csv"),
    )?;

    let out_dir = base_output_dir.join(experiment_name);
    std::fs::create_dir_all(&out_dir).ok();
    let checkpoint_path = out_dir.join("model.safetensors");

    let model = match parsed_model_type {
        // Complex CLI trains inherit the same `--dropout` value as
        // Real — the complex path applies it pair-wise after the
        // activation, so `--dropout 0.2` on a Complex model drops 20 %
        // of the `(real, imag)` pairs together.
        ModelType::Complex => ModelConfig::complex(hidden_size, complex_activation, dropout),
        ModelType::Real => ModelConfig::real(hidden_size, activation, dropout, norm),
    }?;

    let run_result = run_training(&WorkflowTrainConfig {
        train_samples: train_samples.into(),
        val_samples: val_samples.into(),
        test_samples: test_samples.into(),
        // CLI writes a `.safetensors` file alongside the experiment
        // directory; web mode passes `None` so only the DB BLOB is kept.
        checkpoint_path: Some(checkpoint_path.clone()),
        model,
        optimizer,
        lr,
        weight_decay,
        batch_size,
        max_epochs,
        patience,
        warmup_epochs,
        scheduler,
        loss,
        checkpoint: cli.checkpoint.clone(),
        seed,
        log_progress: true,
        // Every former "TODO" is now a real flag. The CLI has the
        // same tunable surface as the HPO-form retrain-prefill, so
        // reproducing an HPO trial by hand only needs the raw
        // hyperparameters — no TOML workaround.
        shuffle_seed,
        grad_clip_norm,
        input_noise_std,
        cosine_eta_min,
        plateau_factor,
        plateau_patience,
        plateau_min_lr,
        sgd_momentum,
        sgd_nesterov,
        rmsprop_momentum,
        rmsprop_alpha,
        pi_mape_beta,
        physics_lambda,
    })?;

    if let Some(path) = run_result.checkpoint_path.as_ref() {
        eprintln!("\nModel saved to {}", path.display());
    }
    Ok(())
}

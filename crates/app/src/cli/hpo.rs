//! US-10.4: `hpo` — parse CLI/TOML args and delegate to workflows.

use std::path::PathBuf;

use candle_core::Error as CandleError;
use clap::Args;
use serde::Deserialize;

use crate::workflows::hpo::{HpoWorkflowConfig, run_hpo};

use super::shared::load_toml_overlay;

/// HPO command arguments.
///
/// TOML-overridable fields are `Option<T>` so the same struct serves as both
/// the CLI parser and the TOML overlay target.  Fields that only make sense on
/// the CLI (`data_dir`, `output_dir`, `config`) are skipped by serde.
#[derive(Debug, Default, Args, Deserialize)]
pub struct HpoArgs {
    /// Directory containing data CSVs.
    #[arg(long, short = 'd', default_value = "./data")]
    #[serde(skip)]
    pub data_dir: PathBuf,

    /// Output directory for study results.
    #[arg(long, short = 'o', default_value = "./artifacts/hpo")]
    #[serde(skip)]
    pub output_dir: PathBuf,

    /// Model type: real, complex, or both.
    #[arg(long, short = 'm', default_value = "real")]
    #[serde(default)]
    pub model_type: Option<String>,

    /// Number of HPO trials.
    #[arg(long, short = 'n', default_value = "100")]
    #[serde(default)]
    pub n_trials: Option<usize>,

    /// Parallel workers (0 = auto-detect).
    #[arg(long, short = 'j', default_value = "0")]
    #[serde(default)]
    pub n_jobs: Option<usize>,

    /// Training epochs per trial.
    #[arg(long, short = 'e', default_value = "25")]
    #[serde(default)]
    pub max_epochs: Option<usize>,

    /// Early stopping patience per trial.
    #[arg(long, default_value = "5")]
    #[serde(default)]
    pub patience: Option<usize>,

    /// Early stopping warmup epochs per trial (delay monitoring).
    #[arg(long, default_value = "5")]
    #[serde(default)]
    pub warmup_epochs: Option<usize>,

    /// Study name.
    #[arg(long)]
    #[serde(default)]
    pub study_name: Option<String>,

    /// Load settings from a TOML configuration file.
    #[arg(long, short = 'c')]
    #[serde(skip)]
    pub config: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(cli: &HpoArgs, seed: u64) -> Result<(), CandleError> {
    let toml: HpoArgs = load_toml_overlay(cli.config.as_deref(), "hpo")?;

    // Load samples directly from CSVs for CLI use.
    let train_samples = sparam_data::generation::load_samples_from_csv(
        &cli.data_dir.join("data_train.csv"),
    )?;
    let val_samples = sparam_data::generation::load_samples_from_csv(
        &cli.data_dir.join("data_val.csv"),
    )?;
    let test_samples = sparam_data::generation::load_samples_from_csv(
        &cli.data_dir.join("data_test.csv"),
    )?;

    let config = HpoWorkflowConfig {
        train_samples: train_samples.into(),
        val_samples: val_samples.into(),
        test_samples: test_samples.into(),
        model_type: toml.model_type.or(cli.model_type.clone()).unwrap_or_else(|| "real".into()),
        n_trials: toml.n_trials.or(cli.n_trials).unwrap_or(100),
        n_jobs: toml.n_jobs.or(cli.n_jobs).unwrap_or(0),
        max_epochs: toml.max_epochs.or(cli.max_epochs).unwrap_or(25),
        patience: toml.patience.or(cli.patience).unwrap_or(5),
        warmup_epochs: toml.warmup_epochs.or(cli.warmup_epochs).unwrap_or(5),
        study_name: cli
            .study_name
            .as_deref()
            .or(toml.study_name.as_deref())
            .unwrap_or("hpo_study")
            .to_string(),
        seed,
        search_space_json: None,
    };

    let result = run_hpo(&config)?;
    eprintln!(
        "HPO complete: {} study summaries in {:.1}s",
        result.summaries.len(),
        result.total_time_secs,
    );
    Ok(())
}

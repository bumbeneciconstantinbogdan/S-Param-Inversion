//! US-10.1: `generate-data` — parse CLI/TOML args and delegate to workflows.

use std::path::PathBuf;

use candle_core::Error as CandleError;
use clap::Args;
use serde::Deserialize;

use crate::workflows::generate::{DataGenerationConfig, GenerateConfig, run_generate};

use super::shared::{load_toml_overlay, parse_range};

// ---------------------------------------------------------------------------
// CLI arguments (also used as TOML overlay target)
// ---------------------------------------------------------------------------

/// Generate-data command arguments.
///
/// TOML-overridable fields are `Option<T>`.  CLI-only fields (`output_dir`,
/// `quiet`, `config`) are skipped by serde.  Ranges are stored as `[f64; 2]`
/// arrays in TOML (e.g. `eps_prime_range = [1.0, 200.0]`) and as `(f64, f64)`
/// tuples on the CLI via the `parse_range` value-parser.
#[derive(Debug, Default, Args, Deserialize)]
pub struct GenerateDataArgs {
    /// Directory to save CSV files.
    #[arg(long, short = 'o', default_value = "./data")]
    #[serde(skip)]
    pub output_dir: PathBuf,

    /// Real permittivity range (lo,hi).
    #[arg(long, value_parser = parse_range, default_value = "1.0,200.0")]
    #[serde(default)]
    pub eps_prime_range: Option<(f64, f64)>,

    /// Imaginary permittivity range (lo,hi).
    #[arg(long, value_parser = parse_range, default_value = "0.0,100.0")]
    #[serde(default)]
    pub eps_secund_range: Option<(f64, f64)>,

    /// Grid points in ε' direction (train).
    #[arg(long, default_value = "200")]
    #[serde(default)]
    pub grid_nx: Option<usize>,

    /// Grid points in ε'' direction (train).
    #[arg(long, default_value = "100")]
    #[serde(default)]
    pub grid_ny: Option<usize>,

    /// Fraction for training data.
    #[arg(long, default_value = "0.8")]
    #[serde(default)]
    pub train_ratio: Option<f64>,

    /// Operating frequency in Hz.
    #[arg(long, short = 'f', default_value = "8200000000.0")]
    #[serde(default)]
    pub frequency: Option<f64>,

    /// Waveguide width in metres.
    #[arg(long, short = 'a', default_value = "0.02286")]
    #[serde(default)]
    pub waveguide_width: Option<f64>,

    /// Sample thickness in metres.
    #[arg(long, short = 'd', default_value = "0.0015")]
    #[serde(default)]
    pub sample_thickness: Option<f64>,

    /// Load settings from a TOML configuration file.
    #[arg(long, short = 'c')]
    #[serde(skip)]
    pub config: Option<PathBuf>,

    /// Suppress progress output.
    #[arg(long, short = 'q')]
    #[serde(skip)]
    pub quiet: bool,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(cli: &GenerateDataArgs, seed: u64) -> Result<(), CandleError> {
    let toml: GenerateDataArgs = load_toml_overlay(cli.config.as_deref(), "generate-data")?;

    let eps_prime = toml
        .eps_prime_range
        .or(cli.eps_prime_range)
        .unwrap_or((1.0, 200.0));
    let eps_secund = toml
        .eps_secund_range
        .or(cli.eps_secund_range)
        .unwrap_or((0.0, 100.0));

    let config = GenerateConfig {
        output_dir: cli.output_dir.clone(),
        data_config: DataGenerationConfig {
            d: toml.sample_thickness.or(cli.sample_thickness).unwrap_or(1.5e-3),
            a: toml.waveguide_width.or(cli.waveguide_width).unwrap_or(22.86e-3),
            frequency: toml.frequency.or(cli.frequency).unwrap_or(8.2e9),
            eps_prime_range: eps_prime,
            eps_double_prime_range: eps_secund,
            n_eps_prime_train: toml.grid_nx.or(cli.grid_nx).unwrap_or(200),
            n_eps_double_prime_train: toml.grid_ny.or(cli.grid_ny).unwrap_or(100),
            train_ratio: toml.train_ratio.or(cli.train_ratio).unwrap_or(0.8),
            seed,
            ..Default::default()
        },
        verbose: !cli.quiet,
    };

    let result = run_generate(&config)?;
    if !cli.quiet {
        eprintln!(
            "Generated {} train / {} val / {} test samples in {}",
            result.train_samples,
            result.val_samples,
            result.test_samples,
            cli.output_dir.display(),
        );
    }
    Ok(())
}

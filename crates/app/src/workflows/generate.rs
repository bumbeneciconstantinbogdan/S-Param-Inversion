//! End-to-end data generation workflow.

use std::path::PathBuf;

use candle_core::Result;
use serde::Serialize;

use sparam_data::generation::{generate_non_magnetic_data, print_data_generation_header, print_dataset_stats};

// Re-export so CLI can build configs without reaching into `data` directly.
pub use sparam_data::generation::DataGenerationConfig;
use sparam_core::io::ensure_dir;

use super::internal::writers::write_json_artifact;

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Configuration for the data generation workflow.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Output directory for CSV files and manifest.
    pub output_dir: PathBuf,
    /// Data generation parameters.
    pub data_config: DataGenerationConfig,
    /// Whether to print progress to stderr.
    pub verbose: bool,
}

// ---------------------------------------------------------------------------
// Public result
// ---------------------------------------------------------------------------

/// Outcome of a data-generation run.
#[derive(Debug, Clone)]
pub struct GenerateRunResult {
    pub train_samples: usize,
    pub val_samples: usize,
    pub test_samples: usize,
}

/// Outcome of a data-generation run that keeps the samples in memory
/// (no disk I/O). Used by the web UI which persists samples to SQLite
/// directly rather than writing CSVs.
pub struct GenerateInMemoryResult {
    pub train: Vec<sparam_data::generation::PermittivitySample>,
    pub validation: Vec<sparam_data::generation::PermittivitySample>,
    pub test: Vec<sparam_data::generation::PermittivitySample>,
}

/// Run the data generation workflow in memory — no files written to disk.
/// Caller is responsible for persisting the returned samples (e.g. to SQLite).
pub fn run_generate_in_memory(
    config: &DataGenerationConfig,
) -> Result<GenerateInMemoryResult> {
    let dataset = generate_non_magnetic_data(config)?;
    Ok(GenerateInMemoryResult {
        train: dataset.train,
        validation: dataset.validation,
        test: dataset.test,
    })
}

/// Snapshot of all effective parameters saved alongside the generated data.
#[derive(Debug, Serialize)]
pub struct GenerationManifest {
    pub eps_prime_range: (f64, f64),
    pub eps_secund_range: (f64, f64),
    pub grid_nx: usize,
    pub grid_ny: usize,
    pub train_ratio: f64,
    pub frequency: f64,
    pub waveguide_width: f64,
    pub sample_thickness: f64,
    pub seed: u64,
    pub train_samples: usize,
    pub val_samples: usize,
    pub test_samples: usize,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the data generation workflow.
pub fn run_generate(config: &GenerateConfig) -> Result<GenerateRunResult> {
    if config.verbose {
        print_data_generation_header(&config.data_config);
    }
    let dataset = generate_non_magnetic_data(&config.data_config)?;
    if config.verbose {
        print_dataset_stats(&dataset);
    }

    ensure_dir(&config.output_dir)?;
    dataset.save_to_directory(&config.output_dir)?;

    let manifest = GenerationManifest {
        eps_prime_range: config.data_config.eps_prime_range,
        eps_secund_range: config.data_config.eps_double_prime_range,
        grid_nx: config.data_config.n_eps_prime_train,
        grid_ny: config.data_config.n_eps_double_prime_train,
        train_ratio: config.data_config.train_ratio,
        frequency: config.data_config.frequency,
        waveguide_width: config.data_config.a,
        sample_thickness: config.data_config.d,
        seed: config.data_config.seed,
        train_samples: dataset.train.len(),
        val_samples: dataset.validation.len(),
        test_samples: dataset.test.len(),
    };
    write_json_artifact(&manifest, &config.output_dir.join("config.json"), "generation manifest")?;

    if config.verbose {
        eprintln!("\nDataset saved to {}", config.output_dir.display());
        eprintln!("  config.json written for reproducibility.");
    }

    Ok(GenerateRunResult {
        train_samples: dataset.train.len(),
        val_samples: dataset.validation.len(),
        test_samples: dataset.test.len(),
    })
}

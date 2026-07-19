
//! Synthetic non-magnetic dataset generation and CSV I/O.
//!
//! Generates permittivity grids, computes NRW forward S-parameters for each
//! sample, and writes/reads the standard 6-column CSV format:
//! `S11_real, S11_imag, S21_real, S21_imag, eps_prim, eps_secund`.

use std::{
    fmt::Write,
    fs,
    path::Path,
};

use candle_core::{DType, Device, Result, Tensor};
use csv::StringRecord;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use sparam_core::error::{ErrorContext, candle_msg};
use sparam_physics::nrw::{WaveguideConfig, nrw_direct_scalar_non_magnetic};

const TRAIN_CSV_FILE: &str = "data_train.csv";
const VALIDATION_CSV_FILE: &str = "data_val.csv";
const TEST_CSV_FILE: &str = "data_test.csv";
const SAMPLE_CSV_HEADERS: [&str; 6] = [
    "S11_real",
    "S11_imag",
    "S21_real",
    "S21_imag",
    "eps_prim",
    "eps_secund",
];

fn write_csv_field(buffer: &mut String, value: f64) {
    buffer.clear();
    // Writing into a String is infallible.
    write!(buffer, "{value}").expect("writing to a String should not fail");
}

/// Configuration for synthetic non-magnetic dataset generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataGenerationConfig {
    /// Sample thickness in meters.
    pub d: f64,
    /// Rectangular waveguide width in meters.
    pub a: f64,
    /// Operating frequency in hertz.
    pub frequency: f64,
    /// Inclusive range for the real permittivity component.
    pub eps_prime_range: (f64, f64),
    /// Inclusive range for the positive loss factor.
    pub eps_double_prime_range: (f64, f64),
    /// Number of training grid points along the eps_prime axis.
    pub n_eps_prime_train: usize,
    /// Number of training grid points along the eps_double_prime axis.
    pub n_eps_double_prime_train: usize,
    /// Target fraction of the total dataset assigned to training.
    pub train_ratio: f64,
    /// Validation-grid offsets applied to the lower/upper domain bounds.
    pub val_offset: (f64, f64),
    /// Test-grid offsets applied to the lower/upper domain bounds.
    pub test_offset: (f64, f64),
    /// Reserved for deterministic future stochastic operations.
    pub seed: u64,

    // ── Optional dense patch overlay ──────────────────────────────
    //
    // Adds a small extra grid concentrated in the low-ε region where
    // the bulk grid is sparse and where most model failures cluster.
    // Each split gets its own patch instance with an offset relative
    // to the training patch — same offset semantics as `val_offset` /
    // `test_offset`, but applied inside the patch sub-domain. The
    // patch samples are tagged via `PermittivitySample::is_dense_patch`
    // so the evaluate UI can render them as a separate (toggleable)
    // classification map alongside the bulk grid.
    //
    // All four fields use `#[serde(default)]` so legacy dataset JSON
    // (from before the patch existed) still round-trips, falling back
    // to the no-patch defaults below.
    /// `(ε', ε'')` window in which the dense patch lives.
    /// Default `(1.0, 2.5) × (0.0, 1.75)` — the very-near-vacuum
    /// corner adjacent to ε≈1+0j where the bulk linear grid has the
    /// sparsest coverage and where Complex MLPs concentrate their
    /// worst-case errors. Narrower than the original
    /// `(1, 5) × (0, 5)` window after observing that errors cluster
    /// closer to vacuum than that window suggested.
    #[serde(default = "default_dense_patch_eps_prime_range")]
    pub dense_patch_eps_prime_range: (f64, f64),
    #[serde(default = "default_dense_patch_eps_double_prime_range")]
    pub dense_patch_eps_double_prime_range: (f64, f64),
    /// Train-side patch density `(n_eps_prime, n_eps_double_prime)`.
    /// Default `(10, 10)` — `(0, 0)` disables the patch entirely.
    #[serde(default = "default_dense_patch_n_train")]
    pub dense_patch_n_train: (usize, usize),
    /// Val/Test-side patch density. Default `(5, 5)` — coarser than
    /// train so the eval surface stays readable when the second map
    /// toggle is on.
    #[serde(default = "default_dense_patch_n_eval")]
    pub dense_patch_n_eval: (usize, usize),
    /// Val patch offset *inside the patch sub-domain*. Default
    /// `(0.05, 0.025)` — small enough that no val point coincides
    /// with a train point but close enough that the points sit
    /// between train grid lines, not in unobserved regions. Matches
    /// the spirit of the bulk grid's `val_offset`.
    #[serde(default = "default_dense_patch_val_offset")]
    pub dense_patch_val_offset: (f64, f64),
    /// Test patch offset (slightly larger than val so test ≠ val).
    /// Default `(0.10, 0.05)`.
    #[serde(default = "default_dense_patch_test_offset")]
    pub dense_patch_test_offset: (f64, f64),
}

fn default_dense_patch_eps_prime_range() -> (f64, f64) { (1.0, 2.5) }
fn default_dense_patch_eps_double_prime_range() -> (f64, f64) { (0.0, 1.75) }
// Default densities are `(0, 0)` so `DataGenerationConfig::default()`
// reproduces the historical bulk-only dataset bit-for-bit. Callers
// that want the dense overlay set non-zero values explicitly — see
// `crates/web/src/routes/generate.rs` for the production wiring.
fn default_dense_patch_n_train() -> (usize, usize) { (0, 0) }
fn default_dense_patch_n_eval() -> (usize, usize) { (0, 0) }
fn default_dense_patch_val_offset() -> (f64, f64) { (0.05, 0.025) }
fn default_dense_patch_test_offset() -> (f64, f64) { (0.10, 0.05) }

impl Default for DataGenerationConfig {
    fn default() -> Self {
        Self {
            d: 1.5e-3,
            a: 22.86e-3,
            frequency: 8.2e9,
            eps_prime_range: (1.0, 200.0),
            eps_double_prime_range: (0.0, 100.0),
            n_eps_prime_train: 200,
            n_eps_double_prime_train: 100,
            train_ratio: 0.8,
            val_offset: (0.2, 0.1),
            test_offset: (0.5, 0.25),
            seed: 42,
            dense_patch_eps_prime_range: default_dense_patch_eps_prime_range(),
            dense_patch_eps_double_prime_range: default_dense_patch_eps_double_prime_range(),
            dense_patch_n_train: default_dense_patch_n_train(),
            dense_patch_n_eval: default_dense_patch_n_eval(),
            dense_patch_val_offset: default_dense_patch_val_offset(),
            dense_patch_test_offset: default_dense_patch_test_offset(),
        }
    }
}

/// A single synthetic non-magnetic sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PermittivitySample {
    /// Real part of the reflection coefficient S11.
    pub s11_real: f64,
    /// Imaginary part of the reflection coefficient S11.
    pub s11_imag: f64,
    /// Real part of the transmission coefficient S21.
    pub s21_real: f64,
    /// Imaginary part of the transmission coefficient S21.
    pub s21_imag: f64,
    /// Real part of the relative permittivity.
    pub eps_prime: f64,
    /// Positive loss factor stored with a positive sign.
    pub eps_double_prime: f64,
    /// `true` for samples generated as part of the optional dense
    /// low-ε patch overlay (see [`DataGenerationConfig::dense_patch_n_train`]).
    /// Lets the evaluate UI render the patch as a separate, smaller
    /// classification map alongside the bulk grid without false-
    /// positive matches between the two grids' axes.
    /// `#[serde(default)]` so legacy CSVs / JSON without this field
    /// deserialize as `false` (= bulk-only sample).
    #[serde(default)]
    pub is_dense_patch: bool,
}

/// Synthetic non-magnetic train/validation/test splits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[must_use = "dataset generation returns train/validation/test splits that should be used or saved"]
pub struct GeneratedDataset {
    pub train: Vec<PermittivitySample>,
    pub validation: Vec<PermittivitySample>,
    pub test: Vec<PermittivitySample>,
    pub config: DataGenerationConfig,
}

impl GeneratedDataset {
    /// Total number of generated samples across all splits.
    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.train.len() + self.validation.len() + self.test.len()
    }

    /// Save dataset splits to a directory using the Python-compatible CSV layout.
    pub fn save_to_directory(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir).context("failed to create dataset directory")?;
        save_samples_to_csv(&self.train, &dir.join(TRAIN_CSV_FILE))?;
        save_samples_to_csv(&self.validation, &dir.join(VALIDATION_CSV_FILE))?;
        save_samples_to_csv(&self.test, &dir.join(TEST_CSV_FILE))?;
        Ok(())
    }

    /// Load dataset splits from a directory containing the standard CSV filenames.
    pub fn load_from_directory(dir: &Path, config: DataGenerationConfig) -> Result<Self> {
        Ok(Self {
            train: load_samples_from_csv(&dir.join(TRAIN_CSV_FILE))?,
            validation: load_samples_from_csv(&dir.join(VALIDATION_CSV_FILE))?,
            test: load_samples_from_csv(&dir.join(TEST_CSV_FILE))?,
            config,
        })
    }
}

/// Generate synthetic non-magnetic train/validation/test datasets.
///
/// Samples are laid out in Python-compatible mesh order: the outer loop advances
/// along eps_prime and the inner loop advances along eps_double_prime.
#[must_use = "this function generates an in-memory dataset without mutating external state"]
pub fn generate_non_magnetic_data(config: &DataGenerationConfig) -> Result<GeneratedDataset> {
    validate_config(config)?;

    let train_axes = create_grid_vectors(
        config.eps_prime_range,
        config.eps_double_prime_range,
        config.n_eps_prime_train,
        config.n_eps_double_prime_train,
        (0.0, 0.0),
    )?;
    let (n_eps_prime_eval, n_eps_double_prime_eval) = derive_holdout_grid_dims(config)?;
    let validation_axes = create_grid_vectors(
        config.eps_prime_range,
        config.eps_double_prime_range,
        n_eps_prime_eval,
        n_eps_double_prime_eval,
        config.val_offset,
    )?;
    let test_axes = create_grid_vectors(
        config.eps_prime_range,
        config.eps_double_prime_range,
        n_eps_prime_eval,
        n_eps_double_prime_eval,
        config.test_offset,
    )?;

    let mut train = generate_samples(&train_axes.0, &train_axes.1, config)?;
    let mut validation = generate_samples(&validation_axes.0, &validation_axes.1, config)?;
    let mut test = generate_samples(&test_axes.0, &test_axes.1, config)?;

    // Dense low-ε patch — generated only when both densities are
    // non-zero. Each split gets its own patch instance with linearly
    // displaced axes inside the patch sub-domain (no shared points
    // between train/val/test patches but tightly clustered so val
    // and test still test interpolation, not extrapolation).
    let (n_tp, n_tdp) = config.dense_patch_n_train;
    let (n_ep, n_edp) = config.dense_patch_n_eval;
    let patch_enabled = n_tp > 0 && n_tdp > 0;
    if patch_enabled {
        let train_patch_axes = create_grid_vectors(
            config.dense_patch_eps_prime_range,
            config.dense_patch_eps_double_prime_range,
            n_tp,
            n_tdp,
            (0.0, 0.0),
        )?;
        let mut train_patch =
            generate_samples(&train_patch_axes.0, &train_patch_axes.1, config)?;
        for s in &mut train_patch {
            s.is_dense_patch = true;
        }
        train.extend(train_patch);

        if n_ep > 0 && n_edp > 0 {
            let val_patch_axes = create_grid_vectors(
                config.dense_patch_eps_prime_range,
                config.dense_patch_eps_double_prime_range,
                n_ep,
                n_edp,
                config.dense_patch_val_offset,
            )?;
            let mut val_patch =
                generate_samples(&val_patch_axes.0, &val_patch_axes.1, config)?;
            for s in &mut val_patch {
                s.is_dense_patch = true;
            }
            validation.extend(val_patch);

            let test_patch_axes = create_grid_vectors(
                config.dense_patch_eps_prime_range,
                config.dense_patch_eps_double_prime_range,
                n_ep,
                n_edp,
                config.dense_patch_test_offset,
            )?;
            let mut test_patch =
                generate_samples(&test_patch_axes.0, &test_patch_axes.1, config)?;
            for s in &mut test_patch {
                s.is_dense_patch = true;
            }
            test.extend(test_patch);
        }
    }

    Ok(GeneratedDataset {
        train,
        validation,
        test,
        config: config.clone(),
    })
}

/// Print a short summary of the data generation configuration to stdout.
pub fn print_data_generation_header(config: &DataGenerationConfig) {
    println!("Generating non-magnetic material dataset...");
    println!(
        "  Waveguide: a={:.2}mm, d={:.2}mm, f={:.2}GHz",
        config.a * 1e3,
        config.d * 1e3,
        config.frequency / 1e9
    );
    println!(
        "  Permittivity domain: eps' in [{:.1}, {:.1}], eps'' in [{:.1}, {:.1}]",
        config.eps_prime_range.0,
        config.eps_prime_range.1,
        config.eps_double_prime_range.0,
        config.eps_double_prime_range.1
    );
}

/// Print dataset split statistics to stdout.
pub fn print_dataset_stats(dataset: &GeneratedDataset) {
    let total = dataset.total_samples() as f64;
    println!("\nDataset statistics:");
    println!(
        "  Training:   {:>6} samples ({:.2}%)",
        dataset.train.len(),
        dataset.train.len() as f64 * 100.0 / total
    );
    println!(
        "  Validation: {:>6} samples ({:.2}%)",
        dataset.validation.len(),
        dataset.validation.len() as f64 * 100.0 / total
    );
    println!(
        "  Test:       {:>6} samples ({:.2}%)",
        dataset.test.len(),
        dataset.test.len() as f64 * 100.0 / total
    );
}

/// Save a tensor of shape `(N, 6)` to a Python-compatible CSV file.
///
/// Column order is fixed to:
/// `S11_real, S11_imag, S21_real, S21_imag, eps_prim, eps_secund`.
pub fn save_to_csv(data: &Tensor, path: &Path) -> Result<()> {
    let data = data.to_device(&Device::Cpu)?.to_dtype(DType::F64)?;
    let row_count = validate_csv_tensor_shape(data.dims())?;
    let flat_values = data.flatten_all()?.to_vec1::<f64>()?;

    ensure_parent_directory(path)?;
    let mut writer = csv::Writer::from_path(path).context("failed to create CSV")?;
    writer
        .write_record(SAMPLE_CSV_HEADERS)
        .context("failed to write CSV header")?;
    let mut fields: [String; 6] = std::array::from_fn(|_| String::new());
    debug_assert_eq!(flat_values.len(), row_count * SAMPLE_CSV_HEADERS.len());

    for row in flat_values.chunks_exact(SAMPLE_CSV_HEADERS.len()) {
        for (field, value) in fields.iter_mut().zip(row.iter()) {
            write_csv_field(field, *value);
        }
        writer
            .write_record(fields.iter().map(String::as_str))
            .context("failed to write CSV record")?;
    }

    writer.flush().context("failed to flush CSV")?;
    Ok(())
}

/// Resolved column indices for the standard 6-column CSV schema.
struct CsvColumns {
    s11_real: usize,
    s11_imag: usize,
    s21_real: usize,
    s21_imag: usize,
    eps_prim: usize,
    eps_secund: usize,
}

fn resolve_columns(headers: &StringRecord) -> Result<CsvColumns> {
    Ok(CsvColumns {
        s11_real: find_column(headers, "s11_real")?,
        s11_imag: find_column(headers, "s11_imag")?,
        s21_real: find_column(headers, "s21_real")?,
        s21_imag: find_column(headers, "s21_imag")?,
        eps_prim: find_column(headers, "eps_prim")?,
        eps_secund: find_column(headers, "eps_secund")?,
    })
}

/// Load a Python-compatible CSV file into a tensor of shape `(N, 6)` on CPU.
pub fn load_from_csv(path: &Path) -> Result<Tensor> {
    let mut reader = csv::Reader::from_path(path).context("failed to open CSV")?;
    let headers = reader
        .headers()
        .context("failed to read CSV headers")?
        .clone();
    let columns = resolve_columns(&headers)?;

    // Guess 20k rows (default training set); avoids ~17 reallocations.
    let mut values = Vec::with_capacity(20_000 * SAMPLE_CSV_HEADERS.len());
    let mut row_count = 0usize;
    for (row_index, record) in reader.records().enumerate() {
        let record = record.context("failed to read CSV record")?;
        values.push(parse_record_value(&record, columns.s11_real, row_index + 2, "S11_real")?);
        values.push(parse_record_value(&record, columns.s11_imag, row_index + 2, "S11_imag")?);
        values.push(parse_record_value(&record, columns.s21_real, row_index + 2, "S21_real")?);
        values.push(parse_record_value(&record, columns.s21_imag, row_index + 2, "S21_imag")?);
        values.push(parse_record_value(&record, columns.eps_prim, row_index + 2, "eps_prim")?);
        values.push(parse_record_value(
            &record,
            columns.eps_secund,
            row_index + 2,
            "eps_secund",
        )?);
        row_count += 1;
    }

    Tensor::from_vec(values, &[row_count, SAMPLE_CSV_HEADERS.len()], &Device::Cpu)
}

/// Save one split to a Python-compatible CSV file.
pub fn save_samples_to_csv(samples: &[PermittivitySample], path: &Path) -> Result<()> {
    ensure_parent_directory(path)?;
    let mut writer = csv::Writer::from_path(path).context("failed to create CSV")?;
    writer
        .write_record(SAMPLE_CSV_HEADERS)
        .context("failed to write CSV header")?;
    let mut fields: [String; 6] = std::array::from_fn(|_| String::new());

    for sample in samples {
        write_csv_field(&mut fields[0], sample.s11_real);
        write_csv_field(&mut fields[1], sample.s11_imag);
        write_csv_field(&mut fields[2], sample.s21_real);
        write_csv_field(&mut fields[3], sample.s21_imag);
        write_csv_field(&mut fields[4], sample.eps_prime);
        write_csv_field(&mut fields[5], sample.eps_double_prime);
        writer
            .write_record(fields.iter().map(String::as_str))
            .context("failed to write CSV record")?;
    }

    writer.flush().context("failed to flush CSV")?;
    Ok(())
}

/// Load one split from a Python-compatible CSV file.
pub fn load_samples_from_csv(path: &Path) -> Result<Vec<PermittivitySample>> {
    let mut reader = csv::Reader::from_path(path).context("failed to open CSV")?;
    let headers = reader
        .headers()
        .context("failed to read CSV headers")?
        .clone();
    let columns = resolve_columns(&headers)?;

    let mut samples = Vec::with_capacity(20_000);
    for (row_index, record) in reader.records().enumerate() {
        let record = record.context("failed to read CSV record")?;
        samples.push(PermittivitySample {
            s11_real: parse_record_value(&record, columns.s11_real, row_index + 2, "S11_real")?,
            s11_imag: parse_record_value(&record, columns.s11_imag, row_index + 2, "S11_imag")?,
            s21_real: parse_record_value(&record, columns.s21_real, row_index + 2, "S21_real")?,
            s21_imag: parse_record_value(&record, columns.s21_imag, row_index + 2, "S21_imag")?,
            eps_prime: parse_record_value(&record, columns.eps_prim, row_index + 2, "eps_prim")?,
            eps_double_prime: parse_record_value(&record, columns.eps_secund, row_index + 2, "eps_secund")?,
            // Legacy CSV schema has no is_dense_patch column; assume bulk.
            is_dense_patch: false,
        });
    }

    Ok(samples)
}

fn ensure_parent_directory(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).context("failed to create parent directories")?;
    }

    Ok(())
}

fn validate_csv_tensor_shape(dims: &[usize]) -> Result<usize> {
    if dims.len() != 2 || dims[1] != SAMPLE_CSV_HEADERS.len() {
        return Err(candle_msg(format!(
            "expected data with shape (N, 6), got {:?}",
            dims
        )));
    }

    Ok(dims[0])
}

fn validate_config(config: &DataGenerationConfig) -> Result<()> {
    use sparam_core::validation::{
        validate_non_negative_f64, validate_non_negative_range, validate_positive_f64,
        validate_positive_range, validate_positive_usize,
    };

    WaveguideConfig::new(config.d, config.a)
        .map_err(|e| candle_msg(format!("invalid waveguide config: {e}")))?;

    validate_positive_f64("operating frequency", config.frequency)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    validate_positive_range("eps_prime_range", config.eps_prime_range)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    validate_non_negative_range("eps_double_prime_range", config.eps_double_prime_range)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    validate_positive_usize("n_eps_prime_train", config.n_eps_prime_train)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    validate_positive_usize("n_eps_double_prime_train", config.n_eps_double_prime_train)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    if !config.train_ratio.is_finite() || config.train_ratio <= 0.0 || config.train_ratio >= 1.0 {
        return Err(candle_msg(format!(
            "train_ratio must be in the open interval (0, 1), got {}",
            config.train_ratio
        )));
    }

    validate_non_negative_f64("val_offset.eps_prime", config.val_offset.0)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    validate_non_negative_f64("val_offset.eps_double_prime", config.val_offset.1)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    validate_non_negative_f64("test_offset.eps_prime", config.test_offset.0)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    validate_non_negative_f64("test_offset.eps_double_prime", config.test_offset.1)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    Ok(())
}

fn derive_holdout_grid_dims(config: &DataGenerationConfig) -> Result<(usize, usize)> {
    let n_train = config
        .n_eps_prime_train
        .checked_mul(config.n_eps_double_prime_train)
        .ok_or_else(|| candle_msg("training grid size overflowed usize"))?;
    let val_ratio = (1.0 - config.train_ratio) / 2.0;
    let target_holdout = (val_ratio * n_train as f64 / config.train_ratio).trunc();
    if !target_holdout.is_finite() || target_holdout <= 0.0 {
        return Err(candle_msg(format!(
            "derived holdout size must be finite and positive, got {target_holdout}"
        )));
    }

    let n_eps_double_prime = ((target_holdout / 2.0).sqrt().trunc() as usize).max(1);
    let n_eps_prime = n_eps_double_prime
        .checked_mul(2)
        .ok_or_else(|| candle_msg("holdout grid width overflowed usize"))?;

    Ok((n_eps_prime, n_eps_double_prime))
}

fn create_grid_vectors(
    eps_prime_range: (f64, f64),
    eps_double_prime_range: (f64, f64),
    n_eps_prime: usize,
    n_eps_double_prime: usize,
    offset: (f64, f64),
) -> Result<(Vec<f64>, Vec<f64>)> {
    let eps_prime_start = eps_prime_range.0 + offset.0;
    let eps_prime_end = eps_prime_range.1 - offset.0;
    let eps_double_prime_start = eps_double_prime_range.0 + offset.1;
    let eps_double_prime_end = eps_double_prime_range.1 - offset.1;

    if eps_prime_start > eps_prime_end {
        return Err(candle_msg(format!(
            "eps_prime offset {} collapses the range ({}, {})",
            offset.0, eps_prime_range.0, eps_prime_range.1
        )));
    }
    if eps_double_prime_start > eps_double_prime_end {
        return Err(candle_msg(format!(
            "eps_double_prime offset {} collapses the range ({}, {})",
            offset.1, eps_double_prime_range.0, eps_double_prime_range.1
        )));
    }

    Ok((
        linspace_f32_compat(eps_prime_start, eps_prime_end, n_eps_prime),
        linspace_f32_compat(
            eps_double_prime_start,
            eps_double_prime_end,
            n_eps_double_prime,
        ),
    ))
}

/// Computes in `f32` precision to match the Python reference grid generation.
fn linspace_f32_compat(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![(start as f32) as f64];
    }

    let start = start as f32;
    let end = end as f32;
    let step = (end - start) / (n - 1) as f32;
    let midpoint = (n - 1) / 2;

    (0..n)
        .map(|index| {
            if index <= midpoint {
                (start + step * index as f32) as f64
            } else {
                (end - step * (n - 1 - index) as f32) as f64
            }
        })
        .collect()
}

fn generate_samples(
    eps_prime_axis: &[f64],
    eps_double_prime_axis: &[f64],
    config: &DataGenerationConfig,
) -> Result<Vec<PermittivitySample>> {
    let n_samples = eps_prime_axis
        .len()
        .checked_mul(eps_double_prime_axis.len())
        .ok_or_else(|| candle_msg("flattened grid size overflowed usize"))?;

    // Build pairs via Cartesian product, then compute NRW in parallel.
    // `flat_map` doesn't advertise an exact `size_hint`, so pre-size the
    // destination to `n_samples` to skip the geometric-growth reallocs.
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n_samples);
    pairs.extend(
        eps_prime_axis
            .iter()
            .flat_map(|&ep| eps_double_prime_axis.iter().map(move |&edp| (ep, edp))),
    );
    debug_assert_eq!(pairs.len(), n_samples);

    let samples: Vec<PermittivitySample> = pairs
        .par_iter()
        .map(|&(eps_prime, eps_double_prime)| {
            let (s11, s21) = nrw_direct_scalar_non_magnetic(
                config.d,
                config.a,
                config.frequency,
                eps_prime,
                eps_double_prime,
            );
            PermittivitySample {
                s11_real: s11.re,
                s11_imag: s11.im,
                s21_real: s21.re,
                s21_imag: s21.im,
                eps_prime,
                eps_double_prime,
                is_dense_patch: false,
            }
        })
        .collect();

    Ok(samples)
}

#[cfg(test)]
fn generate_samples_for_pairs(
    eps_prime_values: &[f64],
    eps_double_prime_values: &[f64],
    config: &DataGenerationConfig,
) -> Result<Vec<PermittivitySample>> {
    debug_assert_eq!(eps_prime_values.len(), eps_double_prime_values.len());
    if eps_prime_values.is_empty() {
        return Ok(Vec::new());
    }

    let samples: Vec<PermittivitySample> = eps_prime_values
        .par_iter()
        .zip(eps_double_prime_values.par_iter())
        .map(|(&eps_prime, &eps_double_prime)| {
            let (s11, s21) = nrw_direct_scalar_non_magnetic(
                config.d,
                config.a,
                config.frequency,
                eps_prime,
                eps_double_prime,
            );
            PermittivitySample {
                s11_real: s11.re,
                s11_imag: s11.im,
                s21_real: s21.re,
                s21_imag: s21.im,
                eps_prime,
                eps_double_prime,
                is_dense_patch: false,
            }
        })
        .collect();

    Ok(samples)
}

fn find_column(headers: &StringRecord, name: &str) -> Result<usize> {
    headers
        .iter()
        .position(|header| header.trim().eq_ignore_ascii_case(name))
        .ok_or_else(|| candle_msg(format!("missing required CSV column '{name}'")))
}

fn parse_record_value(
    record: &StringRecord,
    index: usize,
    row_number: usize,
    column_name: &str,
) -> Result<f64> {
    let raw_value = record.get(index).ok_or_else(|| {
        candle_msg(format!(
            "missing value for column '{column_name}' on row {row_number}"
        ))
    })?;
    raw_value.trim().parse::<f64>().map_err(|error| {
        candle_msg(format!(
            "failed to parse column '{column_name}' on row {row_number}: {error}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        env, fs,
        path::PathBuf,
        process,
        time::{SystemTime, UNIX_EPOCH},
    };

    use serde::Deserialize;

    use super::*;

    const DATA_GENERATION_PYTHON_PARITY_TOLERANCE: f64 = 1e-10;
    const DATA_GENERATION_COORDINATE_TOLERANCE: f64 = 1e-12;

    #[derive(Debug, Deserialize)]
    struct PythonParityRoot {
        parity: PythonParityCase,
    }

    #[derive(Debug, Deserialize)]
    struct PythonParityCase {
        a: f64,
        d: f64,
        frequencies_hz: Vec<f64>,
        eps_r_real: Vec<f64>,
        eps_r_imag: Vec<f64>,
        s11_real: Vec<f64>,
        s11_imag: Vec<f64>,
        s21_real: Vec<f64>,
        s21_imag: Vec<f64>,
    }

    fn assert_samples_match_python(
        actual: &[PermittivitySample],
        expected: &[PermittivitySample],
        split_name: &str,
    ) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{split_name} split length mismatch"
        );

        for (index, (actual_sample, expected_sample)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            let checks: &[(&str, f64, f64, f64)] = &[
                ("s11_real", actual_sample.s11_real, expected_sample.s11_real, DATA_GENERATION_PYTHON_PARITY_TOLERANCE),
                ("s11_imag", actual_sample.s11_imag, expected_sample.s11_imag, DATA_GENERATION_PYTHON_PARITY_TOLERANCE),
                ("s21_real", actual_sample.s21_real, expected_sample.s21_real, DATA_GENERATION_PYTHON_PARITY_TOLERANCE),
                ("s21_imag", actual_sample.s21_imag, expected_sample.s21_imag, DATA_GENERATION_PYTHON_PARITY_TOLERANCE),
                ("eps_prime", actual_sample.eps_prime, expected_sample.eps_prime, DATA_GENERATION_COORDINATE_TOLERANCE),
                ("eps_double_prime", actual_sample.eps_double_prime, expected_sample.eps_double_prime, DATA_GENERATION_COORDINATE_TOLERANCE),
            ];
            for &(name, a, e, tol) in checks {
                assert!(
                    (a - e).abs() < tol,
                    "{split_name} {name} mismatch at {index}: {a} vs {e}"
                );
            }
        }
    }

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(prefix: &str) -> Result<Self> {
            let unique = format!(
                "{}_{}_{}",
                prefix,
                process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system clock should be after UNIX_EPOCH")
                    .as_nanos()
            );
            let path = env::temp_dir().join(unique);
            fs::create_dir_all(&path).map_err(|e| candle_msg(format!("failed to create temporary test directory: {e}")))?;
            Ok(Self { path })
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn test_default_dataset_sizes_match_spec() -> Result<()> {
        let dataset = generate_non_magnetic_data(&DataGenerationConfig::default())?;

        assert_eq!(dataset.train.len(), 20_000);
        assert_eq!(dataset.validation.len(), 2_450);
        assert_eq!(dataset.test.len(), 2_450);
        assert_eq!(dataset.total_samples(), 24_900);

        Ok(())
    }

    #[test]
    fn test_training_samples_follow_python_meshgrid_order() -> Result<()> {
        let config = DataGenerationConfig {
            eps_prime_range: (1.0, 3.0),
            eps_double_prime_range: (0.0, 1.0),
            n_eps_prime_train: 3,
            n_eps_double_prime_train: 2,
            train_ratio: 0.5,
            val_offset: (0.2, 0.1),
            test_offset: (0.4, 0.2),
            ..DataGenerationConfig::default()
        };
        let dataset = generate_non_magnetic_data(&config)?;
        let coordinates: Vec<(f64, f64)> = dataset
            .train
            .iter()
            .map(|sample| (sample.eps_prime, sample.eps_double_prime))
            .collect();

        assert_eq!(
            coordinates,
            vec![
                (1.0, 0.0),
                (1.0, 1.0),
                (2.0, 0.0),
                (2.0, 1.0),
                (3.0, 0.0),
                (3.0, 1.0)
            ]
        );

        Ok(())
    }

    #[test]
    fn test_train_validation_and_test_grids_do_not_overlap() -> Result<()> {
        let dataset = generate_non_magnetic_data(&DataGenerationConfig::default())?;

        let train_points: HashSet<_> = dataset
            .train
            .iter()
            .map(|sample| {
                (
                    sample.eps_prime.to_bits(),
                    sample.eps_double_prime.to_bits(),
                )
            })
            .collect();
        let validation_points: HashSet<_> = dataset
            .validation
            .iter()
            .map(|sample| {
                (
                    sample.eps_prime.to_bits(),
                    sample.eps_double_prime.to_bits(),
                )
            })
            .collect();
        let test_points: HashSet<_> = dataset
            .test
            .iter()
            .map(|sample| {
                (
                    sample.eps_prime.to_bits(),
                    sample.eps_double_prime.to_bits(),
                )
            })
            .collect();

        assert!(train_points.is_disjoint(&validation_points));
        assert!(train_points.is_disjoint(&test_points));
        assert!(validation_points.is_disjoint(&test_points));

        Ok(())
    }

    #[test]
    fn test_generated_s_parameters_are_physically_valid() -> Result<()> {
        let dataset = generate_non_magnetic_data(&DataGenerationConfig::default())?;

        for sample in &dataset.train {
            let s11_mag = sample.s11_real.hypot(sample.s11_imag);
            let s21_mag = sample.s21_real.hypot(sample.s21_imag);

            assert!(s11_mag <= 1.0 + 1e-10, "S11 magnitude exceeds 1: {s11_mag}");
            assert!(s21_mag <= 1.0 + 1e-10, "S21 magnitude exceeds 1: {s21_mag}");
            assert!(
                s11_mag.powi(2) + s21_mag.powi(2) <= 1.0 + 1e-10,
                "passivity violated: |S11|^2 + |S21|^2 = {}",
                s11_mag.powi(2) + s21_mag.powi(2)
            );
        }

        Ok(())
    }



    #[test]
    fn test_generated_dataset_csv_round_trip() -> Result<()> {
        let config = DataGenerationConfig {
            n_eps_prime_train: 6,
            n_eps_double_prime_train: 4,
            train_ratio: 0.6,
            val_offset: (0.15, 0.05),
            test_offset: (0.35, 0.15),
            ..DataGenerationConfig::default()
        };
        let dataset = generate_non_magnetic_data(&config)?;
        let temp_dir = TestDir::new("cvnn_data_roundtrip")?;

        dataset.save_to_directory(&temp_dir.path)?;
        let header = fs::read_to_string(temp_dir.path.join(TRAIN_CSV_FILE))
            .map_err(|e| candle_msg(format!("failed to read generated training CSV: {e}")))?;
        assert_eq!(
            header
                .lines()
                .next()
                .expect("generated CSV should contain a header row"),
            "S11_real,S11_imag,S21_real,S21_imag,eps_prim,eps_secund"
        );

        let loaded = GeneratedDataset::load_from_directory(&temp_dir.path, config)?;
        assert_eq!(loaded, dataset);

        Ok(())
    }

    #[test]
    fn test_save_to_csv_creates_parent_directories_and_preserves_headers() -> Result<()> {
        let temp_dir = TestDir::new("cvnn_tensor_csv_save")?;
        let nested_path = temp_dir.path.join("nested/output/data.csv");
        let data = Tensor::from_vec(
            vec![0.1f64, -0.2, 0.3, -0.4, 2.5, 0.15],
            &[1, 6],
            &Device::Cpu,
        )?;

        save_to_csv(&data, &nested_path)?;

        let contents =
            fs::read_to_string(&nested_path).map_err(|e| candle_msg(format!("failed to read saved tensor CSV: {e}")))?;
        assert_eq!(
            contents
                .lines()
                .next()
                .expect("saved CSV should contain a header row"),
            "S11_real,S11_imag,S21_real,S21_imag,eps_prim,eps_secund"
        );

        Ok(())
    }

    #[test]
    fn test_load_from_csv_round_trips_tensor_data() -> Result<()> {
        let temp_dir = TestDir::new("cvnn_tensor_csv_roundtrip")?;
        let csv_path = temp_dir.path.join("tensor_roundtrip.csv");
        let data = Tensor::from_vec(
            vec![
                0.1f64, -0.2, 0.3, -0.4, 2.5, 0.15, -0.5, 0.6, -0.7, 0.8, 7.5, 1.25,
            ],
            &[2, 6],
            &Device::Cpu,
        )?;

        save_to_csv(&data, &csv_path)?;
        let loaded = load_from_csv(&csv_path)?;

        assert_eq!(loaded.dims(), &[2, 6]);
        assert_eq!(
            loaded.flatten_all()?.to_vec1::<f64>()?,
            data.flatten_all()?.to_vec1::<f64>()?
        );

        Ok(())
    }

    #[test]
    fn test_save_to_csv_rejects_non_matrix_or_wrong_column_count() -> Result<()> {
        let vector = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], &[3], &Device::Cpu)?;
        let wrong_columns = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu)?;
        let temp_dir = TestDir::new("cvnn_tensor_csv_invalid")?;

        let vector_error = save_to_csv(&vector, &temp_dir.path.join("vector.csv"))
            .expect_err("1D tensors should be rejected");
        let wrong_columns_error =
            save_to_csv(&wrong_columns, &temp_dir.path.join("wrong_columns.csv"))
                .expect_err("tensors without 6 columns should be rejected");

        assert!(vector_error.to_string().contains("shape (N, 6)"));
        assert!(wrong_columns_error.to_string().contains("shape (N, 6)"));

        Ok(())
    }

    #[test]
    fn test_print_functions_do_not_panic() -> Result<()> {
        let config = DataGenerationConfig {
            n_eps_prime_train: 8,
            n_eps_double_prime_train: 4,
            train_ratio: 0.5,
            ..DataGenerationConfig::default()
        };

        print_data_generation_header(&config);
        let dataset = generate_non_magnetic_data(&config)?;
        print_dataset_stats(&dataset);

        Ok(())
    }
}

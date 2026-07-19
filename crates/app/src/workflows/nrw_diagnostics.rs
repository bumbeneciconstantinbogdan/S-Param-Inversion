//! End-to-end NRW inverse diagnostics.
//!
//! Loads CSV test data, runs the NRW inverse algorithm, computes relative
//! error metrics and R² scores, performs OK/NotOK classification at
//! configurable thresholds, and optionally generates classification map
//! figures when the data lies on a regular permittivity grid.

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use csv::StringRecord;

use sparam_core::complex::Complex64;
use sparam_core::constants::{EPSILON_0, MU_0};
use sparam_core::error::{ErrorContext, Result, msg_error as analysis_error};
use sparam_core::grid::{GridStructure, detect_grid_structure};
use sparam_core::io::write_text_atomic;
use sparam_core::validation::{validate_non_negative_f64, validate_positive_f64};
use sparam_core::metrics::{
    classify_predictions, r2_score, relative_error_metrics_from_components,
};

/// Configuration for NRW inverse diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct NrwDiagnosticsConfig {
    /// Sample thickness in meters.
    pub d: f64,
    /// Waveguide width in meters.
    pub a: f64,
    /// Operating frequency in hertz.
    pub frequency: f64,
    /// Error thresholds in percent.
    pub thresholds: Vec<f64>,
    /// Whether classification maps should be written to disk.
    pub save_figures: bool,
    /// Output directory for generated figures.
    pub artifacts_dir: PathBuf,
}

impl Default for NrwDiagnosticsConfig {
    fn default() -> Self {
        Self {
            d: 1.5e-3,
            a: 22.86e-3,
            frequency: 8.2e9,
            thresholds: vec![10.0, 1.0],
            save_figures: true,
            artifacts_dir: PathBuf::from("artifacts"),
        }
    }
}

/// Threshold summary for an NRW diagnostics report.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ThresholdResult {
    pub threshold: f64,
    pub ok_count: usize,
    pub ok_percent: f64,
}

/// Complete diagnostics report for the NRW inverse algorithm.
#[must_use = "diagnostics report contains analysis results"]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NrwDiagnosticsReport {
    /// Number of test samples.
    pub n_samples: usize,
    /// Grid dimensions if the data forms a regular grid.
    pub grid_dims: Option<(usize, usize)>,
    /// Per-threshold OK/NotOK summary.
    pub threshold_results: Vec<ThresholdResult>,
    /// R^2 score for the real part of permittivity.
    pub r2_real: f64,
    /// R^2 score for the imaginary part of permittivity.
    pub r2_imag: f64,
    /// Mean relative error in percent.
    pub mean_error: f64,
    /// Maximum relative error in percent.
    pub max_error: f64,
    /// Minimum relative error in percent.
    pub min_error: f64,
    /// Paths to generated classification map figures.
    pub figure_paths: Vec<PathBuf>,
}

impl NrwDiagnosticsReport {
    /// Serialize the diagnostics report to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|error| {
            analysis_error(format!("Failed to serialize diagnostics report: {error}"))
        })
    }

    /// Save the diagnostics report as pretty-printed JSON.
    pub fn save_json(&self, path: &Path) -> Result<()> {
        let json = self.to_json()?;
        write_text_atomic(path, &json).context("Failed to write diagnostics report JSON")
    }
}

struct CsvColumns {
    s11_real: usize,
    s11_imag: usize,
    s21_real: usize,
    s21_imag: usize,
    eps_prime: usize,
    eps_double_prime: usize,
}

struct NrwCsvData {
    s11_real: Vec<f64>,
    s11_imag: Vec<f64>,
    s21_real: Vec<f64>,
    s21_imag: Vec<f64>,
    eps_prime: Vec<f64>,
    eps_double_prime: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NrwDiagnosticsFigure {
    pub(crate) threshold: f64,
    pub(crate) ordered_mask: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NrwDiagnosticsAnalysis {
    pub(crate) report: NrwDiagnosticsReport,
    pub(crate) grid: Option<GridStructure>,
    pub(crate) figures: Vec<NrwDiagnosticsFigure>,
}

/// Run NRW inverse diagnostics on CSV test data and return only the
/// report (no figures). Convenience wrapper over [`analyze_nrw_diagnostics`]
/// for callers that don't care about the rendered output paths.
pub fn run_nrw_diagnostics(
    data_file: &Path,
    config: &NrwDiagnosticsConfig,
) -> Result<NrwDiagnosticsReport> {
    analyze_nrw_diagnostics(data_file, config).map(|a| a.report)
}

/// Analyze NRW inverse diagnostics on CSV test data without rendering figures.
pub(crate) fn analyze_nrw_diagnostics(
    data_file: &Path,
    config: &NrwDiagnosticsConfig,
) -> Result<NrwDiagnosticsAnalysis> {
    validate_config(config)?;
    let csv_data = read_nrw_csv(data_file)?;
    let n_samples = csv_data.eps_prime.len();
    if n_samples == 0 {
        return Err(analysis_error(format!(
            "NRW diagnostics input '{}' does not contain any samples",
            data_file.display()
        )));
    }

    let true_imag_values: Vec<f64> = csv_data
        .eps_double_prime
        .iter()
        .map(|value| -*value)
        .collect();

    let mut predicted_real = Vec::with_capacity(n_samples);
    let mut predicted_imag = Vec::with_capacity(n_samples);
    for sample_index in 0..n_samples {
        let s11 = Complex64::new(
            csv_data.s11_real[sample_index],
            csv_data.s11_imag[sample_index],
        );
        let s21 = Complex64::new(
            csv_data.s21_real[sample_index],
            csv_data.s21_imag[sample_index],
        );
        let (eps_pred, _) =
            nrw_inverse_reference_single(config.d, config.a, config.frequency, s11, s21);
        predicted_real.push(eps_pred.re);
        predicted_imag.push(eps_pred.im);
    }
    let metrics = relative_error_metrics_from_components(
        &csv_data.eps_prime,
        &true_imag_values,
        &predicted_real,
        &predicted_imag,
    )?;
    let grid = detect_grid_structure(&csv_data.eps_prime, &csv_data.eps_double_prime);

    let mut threshold_results = Vec::with_capacity(config.thresholds.len());
    let mut figures = Vec::new();
    for threshold in &config.thresholds {
        let classification = classify_predictions(&metrics.errors, *threshold);
        threshold_results.push(ThresholdResult {
            threshold: *threshold,
            ok_count: classification.ok_count,
            ok_percent: classification.ok_percent,
        });

        if config.save_figures
            && let Some(grid) = &grid
        {
            let ordered_mask = reorder_mask_for_grid(
                &classification.ok_mask,
                grid,
                &csv_data.eps_prime,
                &csv_data.eps_double_prime,
            )?;
            figures.push(NrwDiagnosticsFigure {
                threshold: *threshold,
                ordered_mask,
            });
        }
    }

    let report = NrwDiagnosticsReport {
        n_samples,
        grid_dims: grid.as_ref().map(|structure| structure.dims),
        threshold_results,
        r2_real: r2_score(&csv_data.eps_prime, &predicted_real),
        r2_imag: r2_score(&true_imag_values, &predicted_imag),
        mean_error: metrics.mean_error,
        max_error: metrics.max_error,
        min_error: metrics.min_error,
        figure_paths: Vec::new(),
    };

    print!("{}", format_nrw_diagnostics_summary(&report, config));
    Ok(NrwDiagnosticsAnalysis {
        report,
        grid,
        figures,
    })
}

fn validate_config(config: &NrwDiagnosticsConfig) -> Result<()> {
    for (name, value) in [
        ("sample thickness d", config.d),
        ("waveguide width a", config.a),
        ("operating frequency", config.frequency),
    ] {
        validate_positive_f64(name, value)?;
    }

    if config.thresholds.is_empty() {
        return Err(analysis_error(
            "At least one NRW diagnostics threshold is required",
        ));
    }
    for threshold in &config.thresholds {
        validate_non_negative_f64("diagnostics threshold", *threshold)?;
    }
    Ok(())
}

fn nrw_inverse_reference_single(
    d: f64,
    a: f64,
    frequency: f64,
    s11: Complex64,
    s21: Complex64,
) -> (Complex64, Complex64) {
    let one = Complex64::new(1.0, 0.0);
    let v1 = s21.add(s11);
    let v2 = s21.sub(s11);
    let x = one
        .sub(v1.mul(v2))
        .div(v1.sub(v2).add(Complex64::new(1e-16, 0.0)));
    let sqrt_term = x.mul(x).sub(one).sqrt();
    let gamma_plus = x.add(sqrt_term);
    let gamma = if gamma_plus.abs() > 1.0 {
        one.div(gamma_plus)
    } else {
        gamma_plus
    };
    let propagation = v1.sub(gamma).div(one.sub(gamma.mul(v1)));
    let beta1_s = Complex64::new(0.0, 0.0)
        .sub(propagation.ln())
        .div(Complex64::new(0.0, d));

    let omega = 2.0 * PI * frequency;
    let k0_sq = omega.powi(2) * EPSILON_0 * MU_0;
    let kt_sq = (PI / a).powi(2);
    let beta1_e = Complex64::new(k0_sq - kt_sq, 0.0).sqrt();
    let mu_r = one.add(gamma).div(one.sub(gamma)).mul(beta1_s).div(beta1_e);
    let eps_r = beta1_s
        .mul(beta1_s)
        .add(Complex64::new(kt_sq, 0.0))
        .div(Complex64::new(k0_sq, 0.0).mul(mu_r));

    (eps_r, mu_r)
}

fn read_nrw_csv(data_file: &Path) -> Result<NrwCsvData> {
    let mut reader = csv::Reader::from_path(data_file).context("Failed to open diagnostics CSV")?;
    let headers = reader
        .headers()
        .context("Failed to read diagnostics CSV headers")?
        .clone();
    let columns = CsvColumns {
        s11_real: find_column(&headers, "s11_real")?,
        s11_imag: find_column(&headers, "s11_imag")?,
        s21_real: find_column(&headers, "s21_real")?,
        s21_imag: find_column(&headers, "s21_imag")?,
        eps_prime: find_column(&headers, "eps_prim")?,
        eps_double_prime: find_column(&headers, "eps_secund")?,
    };

    let mut data = NrwCsvData {
        s11_real: Vec::new(),
        s11_imag: Vec::new(),
        s21_real: Vec::new(),
        s21_imag: Vec::new(),
        eps_prime: Vec::new(),
        eps_double_prime: Vec::new(),
    };

    for (row_index, record) in reader.records().enumerate() {
        let record = record.context("Failed to read diagnostics CSV record")?;
        data.s11_real.push(parse_record_value(
            &record,
            columns.s11_real,
            row_index + 2,
            "S11_real",
        )?);
        data.s11_imag.push(parse_record_value(
            &record,
            columns.s11_imag,
            row_index + 2,
            "S11_imag",
        )?);
        data.s21_real.push(parse_record_value(
            &record,
            columns.s21_real,
            row_index + 2,
            "S21_real",
        )?);
        data.s21_imag.push(parse_record_value(
            &record,
            columns.s21_imag,
            row_index + 2,
            "S21_imag",
        )?);
        data.eps_prime.push(parse_record_value(
            &record,
            columns.eps_prime,
            row_index + 2,
            "eps_prim",
        )?);
        data.eps_double_prime.push(parse_record_value(
            &record,
            columns.eps_double_prime,
            row_index + 2,
            "eps_secund",
        )?);
    }

    Ok(data)
}

fn find_column(headers: &StringRecord, name: &str) -> Result<usize> {
    headers
        .iter()
        .position(|header| header.trim().eq_ignore_ascii_case(name))
        .ok_or_else(|| analysis_error(format!("Missing required diagnostics CSV column '{name}'")))
}

fn parse_record_value(
    record: &StringRecord,
    index: usize,
    row_number: usize,
    column_name: &str,
) -> Result<f64> {
    let raw_value = record.get(index).ok_or_else(|| {
        analysis_error(format!(
            "Missing value for column '{column_name}' on row {row_number}"
        ))
    })?;
    raw_value.trim().parse::<f64>().map_err(|error| {
        analysis_error(format!(
            "Failed to parse column '{column_name}' on row {row_number}: {error}"
        ))
    })
}

fn reorder_mask_for_grid(
    ok_mask: &[bool],
    grid: &GridStructure,
    eps_prime: &[f64],
    eps_double_prime: &[f64],
) -> Result<Vec<bool>> {
    if ok_mask.len() != eps_prime.len() || ok_mask.len() != eps_double_prime.len() {
        return Err(analysis_error(
            "Grid reordering requires matching sample counts",
        ));
    }

    let (nx, ny) = grid.dims;
    let mut ordered = vec![false; nx * ny];
    for sample_index in 0..ok_mask.len() {
        let x_index =
            find_axis_index(&grid.eps_prime_values, eps_prime[sample_index]).ok_or_else(|| {
                analysis_error(format!(
                    "eps' value {} does not belong to the detected grid",
                    eps_prime[sample_index]
                ))
            })?;
        let y_index = find_axis_index(
            &grid.eps_double_prime_values,
            eps_double_prime[sample_index],
        )
        .ok_or_else(|| {
            analysis_error(format!(
                "eps'' value {} does not belong to the detected grid",
                eps_double_prime[sample_index]
            ))
        })?;
        ordered[y_index * nx + x_index] = ok_mask[sample_index];
    }

    Ok(ordered)
}

fn find_axis_index(axis: &[f64], value: f64) -> Option<usize> {
    let tolerance = 1e-12_f64.max(value.abs() * 1e-12);
    axis.iter()
        .position(|candidate| (*candidate - value).abs() <= tolerance)
}

fn format_nrw_diagnostics_summary(
    report: &NrwDiagnosticsReport,
    config: &NrwDiagnosticsConfig,
) -> String {
    let mut lines = vec![
        "================================================================================"
            .to_owned(),
        "NRW INVERSE ALGORITHM DIAGNOSTICS".to_owned(),
        "================================================================================"
            .to_owned(),
        format!("Test samples: {}", report.n_samples),
        format!(
            "Waveguide: a={:.2}mm, d={:.2}mm, f={:.2}GHz",
            config.a * 1e3,
            config.d * 1e3,
            config.frequency / 1e9
        ),
        String::new(),
    ];

    for threshold_result in &report.threshold_results {
        lines.push(format!(
            "--- Threshold: {:.1}% ---",
            threshold_result.threshold
        ));
        lines.push("============================================================".to_owned());
        lines.push("Permittivity Error Analysis Summary".to_owned());
        lines.push("============================================================".to_owned());
        lines.push(format!("Total samples: {}", report.n_samples));
        match report.grid_dims {
            Some((nx, ny)) => lines.push(format!("Grid structure: {nx} x {ny}")),
            None => lines.push("Grid structure: not detected".to_owned()),
        }
        lines.push(format!(
            "Error threshold: {:.1}%",
            threshold_result.threshold
        ));
        lines.push(format!(
            "Samples within threshold: {}/{} ({:.2}%)",
            threshold_result.ok_count, report.n_samples, threshold_result.ok_percent
        ));
        lines.push(format!("Mean error: {:.4}%", report.mean_error));
        lines.push(format!("Max error: {:.4}%", report.max_error));
        lines.push(format!("Min error: {:.4}%", report.min_error));
        lines.push("============================================================".to_owned());
        lines.push(String::new());
    }

    lines.push("R^2 scores:".to_owned());
    lines.push(format!(" eps' (real): {:.5}", report.r2_real));
    lines.push(format!(" eps'' (imag): {:.5}", report.r2_imag));
    lines.push(String::new());
    lines.push(
        "================================================================================"
            .to_owned(),
    );
    lines.push("NRW DIAGNOSTICS COMPLETE".to_owned());
    lines.push(
        "================================================================================"
            .to_owned(),
    );
    lines.push(String::new());

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde::Deserialize;

    use super::*;

    #[derive(Debug, Deserialize)]
    struct PythonDiagnosticsRoot {
        diagnostics: PythonDiagnosticsReport,
    }

    #[derive(Debug, Deserialize)]
    struct PythonDiagnosticsReport {
        n_samples: usize,
        grid_dims: [usize; 2],
        mean_error: f64,
        max_error: f64,
        min_error: f64,
        r2_real: f64,
        r2_imag: f64,
        threshold_results: Vec<PythonThresholdResult>,
    }

    #[derive(Debug, Deserialize)]
    struct PythonThresholdResult {
        threshold: f64,
        ok_count: usize,
        ok_percent: f64,
    }


    #[test]
    fn test_nrw_diagnostics_report_json_round_trip() -> sparam_core::error::Result<()> {
        let temp_dir = std::env::temp_dir().join(format!(
            "nrw_diagnostics_report_json_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should be after unix epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&temp_dir).map_err(|error| {
            analysis_error(format!("Failed to create temp diagnostics dir: {error}"))
        })?;
        let report = NrwDiagnosticsReport {
            n_samples: 4,
            grid_dims: Some((2, 2)),
            threshold_results: vec![ThresholdResult {
                threshold: 10.0,
                ok_count: 3,
                ok_percent: 75.0,
            }],
            r2_real: 0.9,
            r2_imag: 0.8,
            mean_error: 5.0,
            max_error: 12.0,
            min_error: 0.1,
            figure_paths: vec![temp_dir.join("figure.png")],
        };
        let json = report.to_json()?;
        assert!(json.contains("\"n_samples\": 4"));
        assert!(json.contains("\"threshold\": 10.0"));

        let output_path = temp_dir.join("report.json");
        report.save_json(&output_path)?;
        let saved = fs::read_to_string(&output_path).map_err(|error| {
            analysis_error(format!(
                "Failed to read saved diagnostics report JSON: {error}"
            ))
        })?;
        let restored: NrwDiagnosticsReport = serde_json::from_str(&saved).map_err(|error| {
            analysis_error(format!(
                "Failed to deserialize diagnostics report JSON: {error}"
            ))
        })?;
        assert_eq!(restored, report);

        let _ = fs::remove_dir_all(&temp_dir);
        Ok(())
    }
}

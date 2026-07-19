//! Evaluation metrics for complex permittivity predictions.
//!
//! Provides:
//! - **Relative error** -- element-wise `|eps_pred - eps_true| / |eps_true|`
//!   with summary statistics (mean, median, min, max). The dataset's
//!   `ε' ≥ 1` invariant keeps the denominator bounded.
//! - **R² score** -- coefficient of determination for real-valued and complex-valued
//!   predictions (computed per component).
//! - **Classification** -- label predictions as OK/NotOK based on error thresholds.

use candle_core::Result;
use serde::Serialize;

use crate::complex_tensor::ComplexTensor;
use crate::error::candle_msg;

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// Classification result for a single error threshold.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ClassificationResult {
    /// Boolean mask: true = OK, false = NotOK.
    pub ok_mask: Vec<bool>,
    /// Count of OK samples.
    pub ok_count: usize,
    /// Percentage of OK samples.
    pub ok_percent: f64,
    /// Threshold used for classification.
    pub threshold: f64,
}

/// Classify predictions as OK when the relative error is within the threshold.
#[must_use]
pub fn classify_predictions(errors: &[f64], threshold: f64) -> ClassificationResult {
    classify_predictions_inner(errors, threshold)
}

/// Classify with multiple thresholds in a single pass over the errors.
#[must_use]
pub fn classify_multi_threshold(errors: &[f64], thresholds: &[f64]) -> Vec<ClassificationResult> {
    thresholds
        .iter()
        .map(|&threshold| classify_predictions_inner(errors, threshold))
        .collect()
}

fn classify_predictions_inner(errors: &[f64], threshold: f64) -> ClassificationResult {
    let ok_mask: Vec<bool> = errors
        .iter()
        .map(|error| error.is_finite() && *error <= threshold)
        .collect();
    let ok_count = ok_mask.iter().filter(|is_ok| **is_ok).count();
    let ok_percent = if errors.is_empty() {
        0.0
    } else {
        ok_count as f64 * 100.0 / errors.len() as f64
    };

    ClassificationResult {
        ok_mask,
        ok_count,
        ok_percent,
        threshold,
    }
}

// ---------------------------------------------------------------------------
// Relative error
// ---------------------------------------------------------------------------
//
// All public helpers compute `r_i = |ε̂_i − ε_i| / |ε_i|` with no floor
// on the denominator: the dataset enforces `ε' ≥ 1`, so `|ε_target| ≥ 1`
// and division is always well-conditioned.

/// Relative error statistics for complex permittivity predictions.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RelativeErrorMetrics {
    /// Per-sample relative error in percent.
    pub errors: Vec<f64>,
    /// Number of samples.
    pub n_samples: usize,
    /// Mean relative error in percent.
    pub mean_error: f64,
    /// Median relative error in percent.
    pub median_error: f64,
    /// Maximum relative error in percent.
    pub max_error: f64,
    /// Minimum relative error in percent.
    pub min_error: f64,
}

#[inline]
#[must_use]
fn complex_relative_error_percent_components(
    true_re: f64,
    true_im: f64,
    pred_re: f64,
    pred_im: f64,
) -> f64 {
    let abs_error = (pred_re - true_re).hypot(pred_im - true_im);
    let reference_magnitude = true_re.hypot(true_im);
    abs_error / reference_magnitude * 100.0
}

#[must_use]
fn summarize_relative_errors(errors: Vec<f64>) -> RelativeErrorMetrics {
    let n_samples = errors.len();
    let mean_error = if errors.is_empty() {
        0.0
    } else if errors.iter().any(|value| value.is_nan()) {
        f64::NAN
    } else if errors.iter().any(|value| value.is_infinite()) {
        f64::INFINITY
    } else {
        errors.iter().sum::<f64>() / errors.len() as f64
    };
    let median_error = if errors.is_empty() {
        0.0
    } else if errors.iter().any(|value| value.is_nan()) {
        f64::NAN
    } else {
        let mut sorted = errors.clone();
        sorted.sort_by(f64::total_cmp);
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) * 0.5
        } else {
            sorted[mid]
        }
    };
    let max_error = errors.iter().copied().max_by(f64::total_cmp).unwrap_or(0.0);
    let min_error = errors.iter().copied().min_by(f64::total_cmp).unwrap_or(0.0);

    RelativeErrorMetrics {
        errors,
        n_samples,
        mean_error,
        median_error,
        max_error,
        min_error,
    }
}

/// Compute relative error metrics from pre-extracted real/imaginary component slices.
pub fn relative_error_metrics_from_components(
    true_real: &[f64],
    true_imag: &[f64],
    pred_real: &[f64],
    pred_imag: &[f64],
) -> Result<RelativeErrorMetrics> {
    let len = true_real.len();
    if true_imag.len() != len || pred_real.len() != len || pred_imag.len() != len {
        return Err(candle_msg(format!(
            "Length mismatch for relative error components: true_real={}, true_imag={}, pred_real={}, pred_imag={}",
            true_real.len(),
            true_imag.len(),
            pred_real.len(),
            pred_imag.len()
        )));
    }

    let true_components = true_real.iter().copied().zip(true_imag.iter().copied());
    let pred_components = pred_real.iter().copied().zip(pred_imag.iter().copied());

    let mut errors = Vec::with_capacity(len);
    for ((true_re, true_im), (pred_re, pred_im)) in true_components.zip(pred_components) {
        errors.push(complex_relative_error_percent_components(
            true_re, true_im, pred_re, pred_im,
        ));
    }

    Ok(summarize_relative_errors(errors))
}

/// Compute relative error between true and predicted complex permittivity.
/// Returns `r_i = |ε̂_i − ε_i| / |ε_i|` in percent. The dataset's
/// `ε' ≥ 1` invariant keeps the denominator bounded away from zero.
pub fn compute_relative_error(
    eps_true: &ComplexTensor,
    eps_pred: &ComplexTensor,
) -> Result<RelativeErrorMetrics> {
    if eps_true.shape() != eps_pred.shape() {
        return Err(candle_msg(format!(
            "Shape mismatch for relative error: true {:?} vs pred {:?}",
            eps_true.shape(),
            eps_pred.shape()
        )));
    }

    let diff_mag = eps_pred.sub(eps_true)?.mag()?;
    let denominator = eps_true.mag()?;
    let mut errors = diff_mag
        .broadcast_div(&denominator)?
        .flatten_all()?
        .to_vec1::<f64>()?;
    for error in &mut errors {
        *error *= 100.0;
    }

    Ok(summarize_relative_errors(errors))
}

// ---------------------------------------------------------------------------
// R² score
// ---------------------------------------------------------------------------

/// Compute R^2 (coefficient of determination) for predictions.
#[must_use]
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return f64::NAN;
    }

    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_tot: f64 = y_true.iter().map(|value| (value - mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(true_value, pred_value)| (true_value - pred_value).powi(2))
        .sum();

    if ss_tot <= 0.0 {
        return f64::NAN;
    }
    1.0 - ss_res / ss_tot
}

/// R² scores for complex-valued predictions, computed separately per component.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ComplexR2Scores {
    /// R² for the real component (epsilon').
    pub r2_real: f64,
    /// R² for the imaginary component (epsilon'').
    pub r2_imag: f64,
}

/// Compute R² scores for real and imaginary components in a single
/// fused pass. Two passes over the four input slices total — one to
/// compute the means, one to accumulate ss_tot/ss_res for both
/// components simultaneously. Replaces 6 separate iterator passes.
#[must_use]
pub fn complex_r2_scores(
    true_real: &[f64],
    true_imag: &[f64],
    pred_real: &[f64],
    pred_imag: &[f64],
) -> ComplexR2Scores {
    let n = true_real.len();
    if n == 0
        || pred_real.len() != n
        || true_imag.len() != n
        || pred_imag.len() != n
    {
        return ComplexR2Scores {
            r2_real: f64::NAN,
            r2_imag: f64::NAN,
        };
    }
    let n_f = n as f64;

    // Pass 1: paired sums for both component means.
    let (sum_real, sum_imag) = true_real
        .iter()
        .zip(true_imag.iter())
        .fold((0.0_f64, 0.0_f64), |(sr, si), (&r, &i)| (sr + r, si + i));
    let mean_real = sum_real / n_f;
    let mean_imag = sum_imag / n_f;

    // Pass 2: ss_tot and ss_res for both components in one walk.
    let (ss_tot_r, ss_res_r, ss_tot_i, ss_res_i) = true_real
        .iter()
        .zip(true_imag.iter())
        .zip(pred_real.iter().zip(pred_imag.iter()))
        .fold(
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64),
            |(tr, rr, ti, ri), ((&yt_r, &yt_i), (&yp_r, &yp_i))| {
                let dr = yt_r - mean_real;
                let di = yt_i - mean_imag;
                let er = yt_r - yp_r;
                let ei = yt_i - yp_i;
                (tr + dr * dr, rr + er * er, ti + di * di, ri + ei * ei)
            },
        );

    let r2_real = if ss_tot_r > 0.0 { 1.0 - ss_res_r / ss_tot_r } else { f64::NAN };
    let r2_imag = if ss_tot_i > 0.0 { 1.0 - ss_res_i / ss_tot_i } else { f64::NAN };
    ComplexR2Scores { r2_real, r2_imag }
}

/// Compute R² scores from `ComplexTensor` predictions.
pub fn complex_r2_scores_from_tensors(
    eps_true: &ComplexTensor,
    eps_pred: &ComplexTensor,
) -> Result<ComplexR2Scores> {
    if eps_true.shape() != eps_pred.shape() {
        return Err(candle_msg(format!(
            "Shape mismatch for R² computation: true {:?} vs pred {:?}",
            eps_true.shape(),
            eps_pred.shape()
        )));
    }
    let true_real: Vec<f64> = eps_true.real.flatten_all()?.to_vec1()?;
    let true_imag: Vec<f64> = eps_true.imag.flatten_all()?.to_vec1()?;
    let pred_real: Vec<f64> = eps_pred.real.flatten_all()?.to_vec1()?;
    let pred_imag: Vec<f64> = eps_pred.imag.flatten_all()?.to_vec1()?;
    Ok(complex_r2_scores(
        &true_real, &true_imag, &pred_real, &pred_imag,
    ))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    #[test]
    fn test_ok_classification() {
        let errors = vec![0.5, 1.5, 5.0, 15.0, f64::INFINITY, f64::NAN];
        let result = classify_predictions(&errors, 10.0);

        assert_eq!(result.ok_mask, vec![true, true, true, false, false, false]);
        assert_eq!(result.ok_count, 3);
        assert!((result.ok_percent - 50.0).abs() < 1e-12);
        assert!((result.threshold - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_multi_threshold() {
        let errors = vec![0.5, 1.5, 5.0, 10.0];
        let results = classify_multi_threshold(&errors, &[1.0, 5.0, 10.0]);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].ok_count, 1);
        assert_eq!(results[1].ok_count, 3);
        assert_eq!(results[2].ok_count, 4);
    }

    #[test]
    fn test_multi_threshold_empty() {
        let results = classify_multi_threshold(&[0.5, 1.5], &[]);
        assert!(results.is_empty());

        let results = classify_multi_threshold(&[], &[1.0, 5.0]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].ok_count, 0);
        assert_eq!(results[0].ok_percent, 0.0);
    }

    #[test]
    fn test_relative_error_computation() -> Result<()> {
        let device = Device::Cpu;
        let eps_true = ComplexTensor::new(
            Tensor::new(&[2.5f64], &device)?,
            Tensor::new(&[-0.15f64], &device)?,
        )?;
        let eps_pred = ComplexTensor::new(
            Tensor::new(&[2.525f64], &device)?,
            Tensor::new(&[-0.152f64], &device)?,
        )?;

        let metrics = compute_relative_error(&eps_true, &eps_pred)?;

        assert_eq!(metrics.n_samples, 1);
        assert!((metrics.errors[0] - 1.0013940064519187).abs() < 1e-9);
        assert!((metrics.mean_error - metrics.errors[0]).abs() < 1e-12);
        assert!((metrics.median_error - metrics.errors[0]).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_relative_error_aggregates_batch_statistics() -> Result<()> {
        let metrics = relative_error_metrics_from_components(
            &[2.0, 3.0],
            &[-1.0, 4.0],
            &[2.0, 6.0],
            &[-1.0, 8.0],
        )?;

        assert_eq!(metrics.n_samples, 2);
        assert_eq!(metrics.errors.len(), 2);
        assert!((metrics.errors[0] - 0.0).abs() < 1e-12);
        assert!((metrics.errors[1] - 100.0).abs() < 1e-12);
        assert!((metrics.mean_error - 50.0).abs() < 1e-12);
        assert!((metrics.median_error - 50.0).abs() < 1e-12);
        assert!((metrics.max_error - 100.0).abs() < 1e-12);
        assert!((metrics.min_error - 0.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_relative_error_components_reject_length_mismatch() {
        let error = relative_error_metrics_from_components(&[1.0], &[0.0], &[1.0, 2.0], &[0.0])
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Length mismatch for relative error components")
        );
    }

    #[test]
    fn test_r2_score_perfect_prediction() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];

        assert!((r2_score(&y_true, &y_pred) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_r2_score_mean_prediction() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        assert!(r2_score(&y_true, &y_pred).abs() < 1e-10);
    }

    #[test]
    fn test_r2_score_negative() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![3.0, 2.0, 1.0];
        assert!(r2_score(&y_true, &y_pred) < 0.0);
    }

    #[test]
    fn test_complex_r2_scores() {
        let true_real = vec![1.0, 2.0, 3.0];
        let true_imag = vec![-0.1, -0.2, -0.3];
        let pred_real = vec![1.0, 2.0, 3.0];
        let pred_imag = vec![-0.15, -0.2, -0.25];

        let scores = complex_r2_scores(&true_real, &true_imag, &pred_real, &pred_imag);
        assert!((scores.r2_real - 1.0).abs() < 1e-10);
        assert!((scores.r2_imag - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_complex_r2_scores_from_tensors() -> Result<()> {
        let device = Device::Cpu;
        let eps_true = ComplexTensor::new(
            Tensor::new(&[1.0f64, 2.0, 3.0], &device)?,
            Tensor::new(&[-0.1f64, -0.2, -0.3], &device)?,
        )?;
        let eps_pred = ComplexTensor::new(
            Tensor::new(&[1.0f64, 2.0, 3.0], &device)?,
            Tensor::new(&[-0.15f64, -0.2, -0.25], &device)?,
        )?;

        let scores = complex_r2_scores_from_tensors(&eps_true, &eps_pred)?;
        assert!((scores.r2_real - 1.0).abs() < 1e-10);
        assert!((scores.r2_imag - 0.75).abs() < 1e-10);
        Ok(())
    }
}

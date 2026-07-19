//! Prediction evaluation: model output collection, error computation,
//! classification, and plot generation.

use candle_core::{DType, Result, Tensor};
use candle_nn::ModuleT;

use sparam_core::complex_tensor::ComplexTensor;
use sparam_data::loader::{BatchSize, DataLoader};
use sparam_data::physical_constraint::apply_physical_softplus_clamp;
use sparam_data::scaling::{Scaler, ScalerRef, StandardScaler};
use sparam_data::tensor_bridges::{PredictionEncoding, unpack_complex_features};
use sparam_core::error::candle_msg;
use sparam_core::grid::detect_grid_structure;
use sparam_models::MlpModel;
use sparam_core::metrics::{
    ClassificationResult, ComplexR2Scores, RelativeErrorMetrics, classify_predictions,
    complex_r2_scores, relative_error_metrics_from_components,
};
use sparam_data::generation::PermittivitySample;
use crate::workflows::config::EvaluateMetrics;

#[derive(Debug, Clone)]
pub struct PredictionArtifacts {
    pub errors: Vec<f64>,
    pub true_real: Vec<f64>,
    pub true_imag: Vec<f64>,
    pub pred_real: Vec<f64>,
    pub pred_imag: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PredictionEvaluation {
    pub r2_scores: ComplexR2Scores,
    pub rel_error: RelativeErrorMetrics,
    pub ok_at_1: ClassificationResult,
    pub ok_at_10: ClassificationResult,
    pub artifacts: PredictionArtifacts,
}

#[derive(Debug)]
struct PredictionVectors {
    true_real: Vec<f64>,
    true_imag: Vec<f64>,
    pred_real: Vec<f64>,
    pred_imag: Vec<f64>,
}

pub fn evaluate_model_predictions(
    model: &MlpModel,
    features: &Tensor,
    targets: &Tensor,
    feature_scaler: &StandardScaler,
    target_scaler: &StandardScaler,
    encoding: PredictionEncoding,
) -> Result<PredictionEvaluation> {
    let (predictions, scaled_targets) =
        collect_model_outputs(model, features, targets, feature_scaler, target_scaler)?;
    let predictions = target_scaler.inverse_transform(&predictions)?;
    let targets = target_scaler.inverse_transform(&scaled_targets)?;
    summarize_prediction_vectors(extract_prediction_vectors(
        &predictions,
        &targets,
        encoding,
    )?)
}

/// stacked-evaluate path: predictions are already produced (in physical
/// ε-space, as a [`ComplexTensor`] with the model's native imag-sign)
/// by [`crate::workflows::stacked_training::stacked_inference`].  Targets arrive in the
/// SAME physical space (the caller passes the unscaled targets from
/// `prepare_evaluation_data` directly).  We just repack the
/// predictions into the flat layout the `extract_prediction_vectors`
/// decoder expects, then summarise.
///
/// Keeps the same `PredictionEvaluation` output as the single-model
/// path so the plots / metrics code downstream is one shape.
pub(crate) fn evaluate_from_complex_predictions(
    predictions: &ComplexTensor,
    targets: &Tensor,
    encoding: PredictionEncoding,
) -> Result<PredictionEvaluation> {
    // Re-pack the (real, imag) ComplexTensor into the same flat tensor
    // shape `extract_prediction_vectors` reads — so we share the
    // existing `PackedComplex` / `RealColumns` decode logic instead of
    // forking it.  `pack_complex_features` is the canonical inverse of
    // `unpack_complex_features` (both narrow on the same axis).
    let packed = match encoding {
        PredictionEncoding::PackedComplex { .. }
        | PredictionEncoding::RealColumns { .. } => {
            Tensor::cat(&[&predictions.real, &predictions.imag], 1)?
        }
    };
    summarize_prediction_vectors(extract_prediction_vectors(
        &packed,
        targets,
        encoding,
    )?)
}

fn collect_model_outputs(
    model: &MlpModel,
    features: &Tensor,
    targets: &Tensor,
    feature_scaler: &StandardScaler,
    target_scaler: &StandardScaler,
) -> Result<(Tensor, Tensor)> {
    // Pre-scale once instead of per-batch — matches the training path.
    let scaled_features = feature_scaler.transform(features)?;
    let scaled_targets = target_scaler.transform(targets)?;
    let loader = DataLoader::new(scaled_features, scaled_targets, BatchSize::All)?;

    let n = loader.num_batches();
    let mut predictions = Vec::with_capacity(n);
    let mut all_targets = Vec::with_capacity(n);
    let is_complex = matches!(model, MlpModel::Complex(_));
    for batch in loader.iter() {
        let (inputs, batch_targets) = batch?;
        let raw = model.forward_t(&inputs, false)?;
        // Same architectural softplus clamp the model trained under,
        // applied here so the metrics are evaluated against the
        // model's actual production output (clamped to physical ε).
        let clamped =
            apply_physical_softplus_clamp(&raw, ScalerRef::Standard(target_scaler), is_complex)?;
        predictions.push(clamped);
        all_targets.push(batch_targets);
    }

    Ok((
        concat_tensors(predictions, "predictions")?,
        concat_tensors(all_targets, "targets")?,
    ))
}

fn concat_tensors(mut tensors: Vec<Tensor>, label: &str) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(candle_msg(format!("expected at least one {label} batch")));
    }
    if tensors.len() == 1 {
        return Ok(tensors.pop().expect("single tensor present"));
    }
    let refs: Vec<&Tensor> = tensors.iter().collect();
    Tensor::cat(&refs, 0)
}

fn extract_prediction_vectors(
    predictions: &Tensor,
    targets: &Tensor,
    encoding: PredictionEncoding,
) -> Result<PredictionVectors> {
    match encoding {
        PredictionEncoding::RealColumns { negate_imag } => {
            // Extract columns as flat vectors — O(1) allocations instead of O(N) rows.
            let imag_sign = if negate_imag { -1.0 } else { 1.0 };
            let pred_real_raw = col_to_f64(predictions, 0)?;
            let pred_imag_raw = col_to_f64(predictions, 1)?;
            let true_real_raw = col_to_f64(targets, 0)?;
            let true_imag_raw = col_to_f64(targets, 1)?;
            Ok(PredictionVectors {
                true_real: true_real_raw,
                true_imag: true_imag_raw
                    .into_iter()
                    .map(|value| imag_sign * value)
                    .collect(),
                pred_real: pred_real_raw,
                pred_imag: pred_imag_raw
                    .into_iter()
                    .map(|value| imag_sign * value)
                    .collect(),
            })
        }
        PredictionEncoding::PackedComplex {
            complex_feature_count,
            negate_imag,
        } => {
            let predictions = unpack_complex_features(predictions, complex_feature_count)?;
            let targets = unpack_complex_features(targets, complex_feature_count)?;
            let imag_sign = if negate_imag { -1.0 } else { 1.0 };
            Ok(PredictionVectors {
                true_real: targets.real.flatten_all()?.to_vec1::<f64>()?,
                true_imag: targets
                    .imag
                    .flatten_all()?
                    .to_vec1::<f64>()?
                    .into_iter()
                    .map(|value| imag_sign * value)
                    .collect(),
                pred_real: predictions.real.flatten_all()?.to_vec1::<f64>()?,
                pred_imag: predictions
                    .imag
                    .flatten_all()?
                    .to_vec1::<f64>()?
                    .into_iter()
                    .map(|value| imag_sign * value)
                    .collect(),
            })
        }
    }
}

/// Extract column `col` from a 2-D tensor as `Vec<f64>` without
/// materialising the whole 2-D row-jagged allocation.
fn col_to_f64(tensor: &Tensor, col: usize) -> Result<Vec<f64>> {
    let col_tensor = tensor.narrow(1, col, 1)?.flatten_all()?;
    match col_tensor.dtype() {
        DType::F64 => col_tensor.to_vec1::<f64>(),
        DType::F32 => Ok(col_tensor
            .to_vec1::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect()),
        dtype => Err(candle_msg(format!(
            "unsupported tensor dtype for prediction extraction: {dtype:?}"
        ))),
    }
}

fn summarize_prediction_vectors(vectors: PredictionVectors) -> Result<PredictionEvaluation> {
    let PredictionVectors {
        true_real,
        true_imag,
        pred_real,
        pred_imag,
    } = vectors;

    let r2_scores = complex_r2_scores(&true_real, &true_imag, &pred_real, &pred_imag);
    let rel_error = relative_error_metrics_from_components(
        &true_real,
        &true_imag,
        &pred_real,
        &pred_imag,
    )?;
    let errors = rel_error.errors.clone();
    let ok_at_1 = classify_predictions(&rel_error.errors, 1.0);
    let ok_at_10 = classify_predictions(&rel_error.errors, 10.0);

    Ok(PredictionEvaluation {
        r2_scores,
        rel_error,
        ok_at_1: ok_at_1.clone(),
        ok_at_10: ok_at_10.clone(),
        artifacts: PredictionArtifacts {
            errors,
            true_real,
            true_imag,
            pred_real,
            pred_imag,
        },
    })
}

pub(crate) fn find_threshold_percent(classifications: &[ClassificationResult], threshold: f64) -> f64 {
    classifications
        .iter()
        .find(|classification| (classification.threshold - threshold).abs() < 0.01)
        .map_or(0.0, |classification| classification.ok_percent)
}

pub(crate) fn generate_evaluation_plots(
    artifacts: &PredictionArtifacts,
    classifications: &[ClassificationResult],
    metrics: &EvaluateMetrics,
    test_samples: &[PermittivitySample],
) -> Result<serde_json::Value> {
    // Headline stats / scatter data are unconditionally over the
    // whole test set so the metric numbers shown above the maps
    // continue to match the bulk + patch combined run.
    let mut charts_data = serde_json::json!({
        "errors": &artifacts.errors,
        "true_real": &artifacts.true_real,
        "true_imag": &artifacts.true_imag,
        "pred_real": &artifacts.pred_real,
        "pred_imag": &artifacts.pred_imag,
        "r2_real": metrics.r2_real,
        "r2_imag": metrics.r2_imag,
    });

    // Partition into bulk vs dense-patch indices. Each subset is
    // expected to form its own complete (ε', ε'') grid because the
    // patch generator uses different axis values than the bulk grid
    // (or at minimum a different offset). When both subsets render
    // as proper grids, the evaluate UI gets two classification maps
    // — bulk on top (the unchanged headline view), dense patch
    // below as a toggleable zoom-in. When the patch is empty (legacy
    // datasets predating the patch overlay), only the bulk map ships.
    let (bulk_idx, patch_idx): (Vec<usize>, Vec<usize>) = (0..test_samples.len())
        .partition(|&k| !test_samples[k].is_dense_patch);

    let bulk_grid = build_grid_chart(&bulk_idx, test_samples, artifacts, classifications);
    let patch_grid = build_grid_chart(&patch_idx, test_samples, artifacts, classifications);

    // Bulk side keeps the historical key names (`classifications`,
    // `grid`) so the existing JS doesn't need to change. The patch
    // side rides under separate keys; the template gates its
    // rendering on their presence.
    if let Some(b) = bulk_grid {
        charts_data["classifications"] = b.classifications;
        charts_data["grid"] = b.grid;
    } else {
        // Falls back to the raw per-threshold summaries (no grid
        // available) — same fallback the pre-refactor code took.
        let masks: Vec<serde_json::Value> = classifications.iter().map(|c| {
            serde_json::json!({
                "threshold": c.threshold,
                "ok_count": c.ok_count,
                "ok_percent": c.ok_percent,
                "total": c.ok_mask.len(),
            })
        }).collect();
        charts_data["classifications"] = serde_json::json!(masks);
    }
    if let Some(p) = patch_grid {
        charts_data["patch_classifications"] = p.classifications;
        charts_data["patch_grid"] = p.grid;
    }

    Ok(charts_data)
}

struct GridChart {
    grid: serde_json::Value,
    classifications: serde_json::Value,
}

/// Build a grid + per-threshold classifications JSON chunk from a
/// subset of test samples (indexed by `subset_idx` into `samples` /
/// `artifacts.errors` / `classifications[*].ok_mask`).
///
/// Returns `None` when:
/// * the subset is empty (e.g. patch overlay disabled), or
/// * the subset's (ε', ε'') points don't form a complete Cartesian
///   product (no grid → can't render a heatmap).
fn build_grid_chart(
    subset_idx: &[usize],
    samples: &[PermittivitySample],
    artifacts: &PredictionArtifacts,
    classifications: &[ClassificationResult],
) -> Option<GridChart> {
    if subset_idx.is_empty() {
        return None;
    }
    let eps_prime: Vec<f64> = subset_idx.iter().map(|&k| samples[k].eps_prime).collect();
    let eps_double_prime: Vec<f64> =
        subset_idx.iter().map(|&k| samples[k].eps_double_prime).collect();
    let g = detect_grid_structure(&eps_prime, &eps_double_prime)?;
    let (nx, ny) = g.dims;

    let x_index: std::collections::HashMap<u64, usize> = g
        .eps_prime_values
        .iter()
        .enumerate()
        .map(|(i, v)| (v.to_bits(), i))
        .collect();
    let y_index: std::collections::HashMap<u64, usize> = g
        .eps_double_prime_values
        .iter()
        .enumerate()
        .map(|(i, v)| (v.to_bits(), i))
        .collect();

    let mut error_grid = vec![vec![0.0f64; nx]; ny];
    let mut s11_real_grid = vec![vec![0.0f64; nx]; ny];
    let mut s11_imag_grid = vec![vec![0.0f64; nx]; ny];
    let mut s21_real_grid = vec![vec![0.0f64; nx]; ny];
    let mut s21_imag_grid = vec![vec![0.0f64; nx]; ny];
    for &k in subset_idx {
        let sample = &samples[k];
        if let (Some(&xi), Some(&yi)) = (
            x_index.get(&sample.eps_prime.to_bits()),
            y_index.get(&sample.eps_double_prime.to_bits()),
        ) {
            error_grid[yi][xi] = artifacts.errors[k];
            s11_real_grid[yi][xi] = sample.s11_real;
            s11_imag_grid[yi][xi] = sample.s11_imag;
            s21_real_grid[yi][xi] = sample.s21_real;
            s21_imag_grid[yi][xi] = sample.s21_imag;
        }
    }

    // Re-compute per-threshold OK counts over THIS subset only —
    // the input `classifications` was computed over the full test
    // set, so its `ok_count` / `ok_percent` would mis-report when
    // shown next to a sub-map. Mask values are still pulled from
    // the original `ok_mask` (it's per-sample, indexable by `k`).
    let total = subset_idx.len();
    let masks_2d: Vec<serde_json::Value> = classifications
        .iter()
        .map(|c| {
            let mut grid_mask = vec![vec![false; nx]; ny];
            let mut subset_ok = 0usize;
            for &k in subset_idx {
                let sample = &samples[k];
                if let (Some(&xi), Some(&yi)) = (
                    x_index.get(&sample.eps_prime.to_bits()),
                    y_index.get(&sample.eps_double_prime.to_bits()),
                ) {
                    grid_mask[yi][xi] = c.ok_mask[k];
                    if c.ok_mask[k] {
                        subset_ok += 1;
                    }
                }
            }
            let pct = if total > 0 {
                subset_ok as f64 * 100.0 / total as f64
            } else {
                0.0
            };
            serde_json::json!({
                "threshold": c.threshold,
                "ok_count": subset_ok,
                "ok_percent": pct,
                "total": total,
                "grid_mask": grid_mask,
            })
        })
        .collect();

    Some(GridChart {
        grid: serde_json::json!({
            "eps_prime": &g.eps_prime_values,
            "eps_double_prime": &g.eps_double_prime_values,
            "nx": nx,
            "ny": ny,
            "error_grid": error_grid,
            "s11_real": s11_real_grid,
            "s11_imag": s11_imag_grid,
            "s21_real": s21_real_grid,
            "s21_imag": s21_imag_grid,
        }),
        classifications: serde_json::json!(masks_2d),
    })
}

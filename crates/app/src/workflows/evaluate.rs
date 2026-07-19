//! End-to-end evaluation workflow.

use candle_core::{DType, Device, Result, Tensor};

use super::{
    config::{
        CheckpointSource, EvaluateConfig, EvaluateMetrics, EvaluateOutput, EvaluateRunResult,
        NrwEvaluateConfig, ThresholdSummary,
    },
    internal::{
        data_prep::prepare_evaluation_data,
        evaluation::{
            PredictionArtifacts, evaluate_model_predictions, find_threshold_percent,
            generate_evaluation_plots,
        },
        model_factory::build_model,
    },
};
use sparam_core::complex_tensor::ComplexTensor;
use sparam_core::metrics::{
    classify_multi_threshold, complex_r2_scores, relative_error_metrics_from_components,
};
use sparam_data::generation::PermittivitySample;
use sparam_physics::nrw::{
    WaveguideConfig, nrw_direct_scalar_non_magnetic, nrw_inverse_with_config,
};
use sparam_training::checkpoint::{
    load_model_checkpoint, load_model_checkpoint_bytes, peek_checkpoint_tensor_names,
};
use sparam_training::logger::{LogMessage, LogSender, LogWorker};

pub fn run_evaluation(config: &EvaluateConfig) -> Result<EvaluateRunResult> {
    let mut model_config = config.model_config.clone();

    // Architecture autodetect — same rationale as `run_neural_net_inference`.
    // For Complex MLP runs that were trained between the moment LN
    // became always-on and the moment `ModelConfig::Complex.norm`
    // was added to the persisted JSON, the saved config is silent
    // about LN but the BLOB has `model.norm.*` tensors. Without this
    // override, eval would build a no-LN model and silently drop the
    // saved LN weights — producing classification maps that disagree
    // with the model's actual training-time predictions.
    let bytes_for_peek: Option<&[u8]> = match &config.model {
        CheckpointSource::Bytes(b) => Some(b.as_slice()),
        CheckpointSource::Path(_) => None,
    };
    if let Some(bytes) = bytes_for_peek
        && let Ok(names) = peek_checkpoint_tensor_names(bytes)
    {
        let has_complex_norm = names.iter().any(|n| n.starts_with("model.norm."));
        if let crate::workflows::config::ModelConfig::Complex { ref mut norm, .. } =
            model_config
        {
            use sparam_models::ComplexNormChoice;
            *norm = if has_complex_norm {
                ComplexNormChoice::LayerNorm
            } else {
                ComplexNormChoice::None
            };
        }
    }

    // Route status output through the tagged logger (same pattern as
    // `run_training_with_callback` and `run_hpo_with_callback`) so
    // two concurrent evaluations don't mix `"Model: …"` /
    // `"Test samples: …"` lines on a shared stderr. The context tag
    // includes the checkpoint source so each evaluation is
    // identifiable.
    let mut _worker: Option<LogWorker> = None;
    let log: LogSender = if config.quiet {
        LogSender::null()
    } else {
        let (sender, worker) = LogSender::new();
        _worker = Some(worker);
        let ctx = match &config.model {
            CheckpointSource::Path(p) => format!(
                "eval:{}",
                p.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_else(|| p.to_str().unwrap_or("unknown"))
            ),
            CheckpointSource::Bytes(b) => format!("eval:blob-{}b", b.len()),
        };
        sender.with_context(ctx)
    };

    log.send(LogMessage::Info(format!(
        "Model: {} ({} params)",
        model_config.arch_description(),
        config.parameter_count,
    )));
    log.send(LogMessage::Info(format!(
        "Test samples: {}",
        config.test_samples.len(),
    )));

    // Use the caller-provided scaler samples, or fall back to test samples.
    let scaler_samples: &[sparam_data::generation::PermittivitySample] =
        if config.scaler_samples.is_empty() {
            log.send(LogMessage::Warning(
                "no training data for scaler fitting, using test data.".into(),
            ));
            &config.test_samples
        } else {
            &config.scaler_samples
        };

    let prepared = prepare_evaluation_data(
        &config.test_samples,
        scaler_samples,
        model_config.model_type(),
        DType::F64,
        false,
        config.pre_fitted_scalers.as_deref(),
    )?;

    let (model, mut varmap) = build_model(&model_config, DType::F64)?;
    match &config.model {
        CheckpointSource::Path(path) => {
            load_model_checkpoint(&mut varmap, path)?;
        }
        CheckpointSource::Bytes(bytes) => {
            load_model_checkpoint_bytes(&mut varmap, bytes)?;
        }
    }

    // stacked pipeline at evaluate time: when the run was trained with extra
    // ensemble members and/or NRW refinement, replay both here so the
    // metrics + plots reflect what the user actually shipped — not just
    // the primary member's single-shot prediction.  `is_passthrough()`
    // covers the all-defaults case (legacy single-model runs); anything
    // else triggers the chunked stacked-inference path.
    let evaluation = if !config.stack.is_passthrough() {
        // Path branch needs a local Vec to keep the borrow alive for
        // the inference call; Bytes branch borrows directly from
        // `config.model`. Either way `primary_slice` is the active
        // borrow handed to `stacked_inference`.
        let path_bytes: Option<Vec<u8>> = match &config.model {
            CheckpointSource::Bytes(_) => None,
            CheckpointSource::Path(p) => Some(
                std::fs::read(p)
                    .map_err(|e| sparam_core::error::candle_msg(format!("read checkpoint: {e}")))?,
            ),
        };
        let primary_slice: &[u8] = match (&config.model, path_bytes.as_deref()) {
            (CheckpointSource::Bytes(b), _) => b.as_slice(),
            (CheckpointSource::Path(_), Some(bytes)) => bytes,
            (CheckpointSource::Path(_), None) => unreachable!(),
        };
        let mut all_members: Vec<&[u8]> =
            Vec::with_capacity(1 + config.stack.extra_members.len());
        all_members.push(primary_slice);
        all_members.extend(config.stack.extra_members.iter().map(|v| v.as_slice()));
        // Stacked predictions in physical ε space.  Returns a
        // ComplexTensor with the model's native imag-sign convention.
        // Reuse the scalers `prepare_evaluation_data` already fit so
        // we don't pay the StandardScaler cost twice per /evaluate.
        let within_request_fit = sparam_data::scaling::FittedScalers {
            feature: prepared.feature_scaler.clone(),
            target: prepared.target_scaler.clone(),
        };
        let predicted = crate::workflows::stacked_training::stacked_inference(
            &model_config,
            &all_members,
            scaler_samples,
            &config.test_samples,
            config.stack.refine_steps,
            config.stack.refine_lr,
            Some(&within_request_fit),
        )?;
        log.send(LogMessage::Info(format!(
            "stacked pipeline active: {} members, {} refine steps",
            all_members.len(),
            config.stack.refine_steps,
        )));
        crate::workflows::internal::evaluation::evaluate_from_complex_predictions(
            &predicted,
            &prepared.targets,
            prepared.encoding,
        )?
    } else {
        evaluate_model_predictions(
            &model,
            &prepared.features,
            &prepared.targets,
            &prepared.feature_scaler,
            &prepared.target_scaler,
            prepared.encoding,
        )?
    };

    let thresholds = vec![config.thresholds.0, config.thresholds.1];
    let classifications = classify_multi_threshold(&evaluation.rel_error.errors, &thresholds);
    let classification_summaries = classifications
        .iter()
        .map(|classification| ThresholdSummary {
            threshold: classification.threshold,
            ok_count: classification.ok_count,
            ok_percent: classification.ok_percent,
        })
        .collect::<Vec<_>>();

    let mean_error = evaluation.rel_error.mean_error;
    let variance = evaluation
        .rel_error
        .errors
        .iter()
        .map(|error| (error - mean_error).powi(2))
        .sum::<f64>()
        / evaluation.rel_error.n_samples as f64;
    let output = EvaluateOutput {
        model_path: match &config.model {
            CheckpointSource::Path(p) => p.display().to_string(),
            CheckpointSource::Bytes(b) => format!("<db blob, {} bytes>", b.len()),
        },
        test_samples: evaluation.rel_error.n_samples,
        metrics: EvaluateMetrics {
            ok_at_1pct: find_threshold_percent(&classifications, 1.0),
            ok_at_10pct: find_threshold_percent(&classifications, 10.0),
            mean_error,
            max_error: evaluation.rel_error.max_error,
            min_error: evaluation.rel_error.min_error,
            std_error: variance.sqrt(),
            r2_real: evaluation.r2_scores.r2_real,
            r2_imag: evaluation.r2_scores.r2_imag,
        },
        thresholds: thresholds.clone(),
    };

    let charts_json = if config.generate_plots {
        Some(generate_evaluation_plots(
            &evaluation.artifacts,
            &classifications,
            &output.metrics,
            &config.test_samples,
        )?)
    } else {
        None
    };

    Ok(EvaluateRunResult {
        output,
        classifications: classification_summaries,
        charts_json,
    })
}

/// Analytical NRW round-trip evaluation: forward `(ε_r, μ_r=1) → S`
/// via [`nrw_direct_scalar_non_magnetic`], then invert
/// `S → ε̂_r` via [`nrw_inverse_with_config`], compare against the
/// input grid. The output mirrors [`run_evaluation`]'s shape so the
/// `evaluate_result.html` template renders without changes — but no
/// model is loaded, so this runs even when the database has zero
/// trained runs. Use it to visualise the closed-form NRW failure
/// region (Step 1 instability near vacuum, Step 4 phase ambiguity for
/// thick samples) as a baseline against any trained NN.
pub fn run_nrw_evaluation(config: &NrwEvaluateConfig) -> Result<EvaluateRunResult> {
    if config.n_eps_prime == 0 || config.n_eps_double_prime == 0 {
        return Err(candle_core::Error::Msg(
            "grid dimensions must be positive".into(),
        ));
    }
    let waveguide = WaveguideConfig::new(config.d, config.a)?;

    let eps_prime_axis = linspace(config.eps_prime_range, config.n_eps_prime);
    let eps_double_prime_axis = linspace(config.eps_double_prime_range, config.n_eps_double_prime);

    let n_total = eps_prime_axis.len() * eps_double_prime_axis.len();
    let mut samples: Vec<PermittivitySample> = Vec::with_capacity(n_total);
    let mut s11_real = Vec::with_capacity(n_total);
    let mut s11_imag = Vec::with_capacity(n_total);
    let mut s21_real = Vec::with_capacity(n_total);
    let mut s21_imag = Vec::with_capacity(n_total);

    // Outer loop on ε′, inner on ε″ — same row-major order
    // `generate_samples` uses, so `detect_grid_structure` recognises
    // the resulting layout as a Cartesian grid for the heatmap.
    for &ep in &eps_prime_axis {
        for &edp in &eps_double_prime_axis {
            let (s11, s21) = nrw_direct_scalar_non_magnetic(
                config.d,
                config.a,
                config.frequency,
                ep,
                edp,
            );
            samples.push(PermittivitySample {
                s11_real: s11.re,
                s11_imag: s11.im,
                s21_real: s21.re,
                s21_imag: s21.im,
                eps_prime: ep,
                eps_double_prime: edp,
                is_dense_patch: false,
            });
            s11_real.push(s11.re);
            s11_imag.push(s11.im);
            s21_real.push(s21.re);
            s21_imag.push(s21.im);
        }
    }

    let device = Device::Cpu;
    let s11_tensor = ComplexTensor::new(
        Tensor::new(s11_real.as_slice(), &device)?,
        Tensor::new(s11_imag.as_slice(), &device)?,
    )?;
    let s21_tensor = ComplexTensor::new(
        Tensor::new(s21_real.as_slice(), &device)?,
        Tensor::new(s21_imag.as_slice(), &device)?,
    )?;
    let frequencies = Tensor::new(
        vec![config.frequency; n_total].as_slice(),
        &device,
    )?;

    let (eps_pred, _mu_pred) =
        nrw_inverse_with_config(&waveguide, &frequencies, &s11_tensor, &s21_tensor)?;

    // ε_r = ε′ − jε″ — NRW returns complex eps with `.imag` already
    // negative for physical (lossy) samples. Flip the sign so the
    // predicted ε″ uses the same positive-loss-factor convention as
    // the true axis (and as `PermittivitySample::eps_double_prime`).
    let pred_real: Vec<f64> = eps_pred.real.to_vec1::<f64>()?;
    let pred_imag: Vec<f64> = eps_pred
        .imag
        .to_vec1::<f64>()?
        .into_iter()
        .map(|v| -v)
        .collect();

    let true_real: Vec<f64> = samples.iter().map(|s| s.eps_prime).collect();
    let true_imag: Vec<f64> = samples.iter().map(|s| s.eps_double_prime).collect();

    let r2_scores = complex_r2_scores(&true_real, &true_imag, &pred_real, &pred_imag);
    let rel_error = relative_error_metrics_from_components(
        &true_real,
        &true_imag,
        &pred_real,
        &pred_imag,
    )?;

    let thresholds = vec![config.thresholds.0, config.thresholds.1];
    let classifications = classify_multi_threshold(&rel_error.errors, &thresholds);

    // Std deviation, NaN-safe (failure points have NaN errors which
    // would poison the running sum). Compute on finite values only.
    let finite: Vec<f64> = rel_error.errors.iter().copied().filter(|e| e.is_finite()).collect();
    let mean_finite = if finite.is_empty() {
        f64::NAN
    } else {
        finite.iter().sum::<f64>() / finite.len() as f64
    };
    let std_error = if finite.len() < 2 {
        0.0
    } else {
        (finite.iter().map(|e| (e - mean_finite).powi(2)).sum::<f64>() / finite.len() as f64).sqrt()
    };

    let metrics = EvaluateMetrics {
        ok_at_1pct: find_threshold_percent(&classifications, 1.0),
        ok_at_10pct: find_threshold_percent(&classifications, 10.0),
        mean_error: rel_error.mean_error,
        max_error: rel_error.max_error,
        min_error: rel_error.min_error,
        std_error,
        r2_real: r2_scores.r2_real,
        r2_imag: r2_scores.r2_imag,
    };

    let artifacts = PredictionArtifacts {
        errors: rel_error.errors.clone(),
        true_real,
        true_imag,
        pred_real,
        pred_imag,
    };

    let charts_json = if config.generate_plots {
        Some(generate_evaluation_plots(&artifacts, &classifications, &metrics, &samples)?)
    } else {
        None
    };

    let classification_summaries = classifications
        .iter()
        .map(|c| ThresholdSummary {
            threshold: c.threshold,
            ok_count: c.ok_count,
            ok_percent: c.ok_percent,
        })
        .collect();

    if !config.quiet {
        eprintln!(
            "NRW analytical round-trip: f={:.3} GHz, d={:.3} mm, a={:.3} mm, grid {}×{}, OK@1%={:.1}%, OK@10%={:.1}%",
            config.frequency / 1e9,
            config.d * 1e3,
            config.a * 1e3,
            config.n_eps_prime,
            config.n_eps_double_prime,
            metrics.ok_at_1pct,
            metrics.ok_at_10pct,
        );
    }

    Ok(EvaluateRunResult {
        output: EvaluateOutput {
            model_path: format!(
                "NRW analytical (no model) — f = {:.3} GHz, d = {:.3} mm, a = {:.3} mm, grid {}×{}",
                config.frequency / 1e9,
                config.d * 1e3,
                config.a * 1e3,
                config.n_eps_prime,
                config.n_eps_double_prime,
            ),
            test_samples: n_total,
            metrics,
            thresholds,
        },
        classifications: classification_summaries,
        charts_json,
    })
}

/// Float linspace over an inclusive range. Returns one or zero values
/// for `n <= 1` to match the analytical-grid use case (a single-point
/// sweep just samples at the lower bound).
fn linspace((start, end): (f64, f64), n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n as f64 - 1.0);
    (0..n).map(|i| start + step * i as f64).collect()
}

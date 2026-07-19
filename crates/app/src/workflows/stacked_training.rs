//! stacked pipeline: train overlay + median ensemble + NRW refinement layered
//! over [`crate::workflows::train::run_training`]. Toggles in
//! [`StackedTrainConfig`]; passthrough delegates straight to
//! `run_training`. See `poster/studies/methodology/M10_tail_sweep.tex`.

use std::sync::Arc;

use candle_core::{DType, Device, ModuleT, Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap};

use sparam_core::complex_tensor::ComplexTensor;
use sparam_core::error::candle_msg;
use sparam_data::generation::{
    DataGenerationConfig, PermittivitySample, generate_non_magnetic_data,
};
use sparam_data::physical_constraint::apply_physical_softplus_clamp;
use sparam_data::scaling::{FittedScalers, Scaler, ScalerRef, StandardScaler};
use sparam_data::tensor_bridges::unpack_complex_features;
use sparam_models::{MlpModel, ModelType};
use sparam_physics::PhysicsContext;
use sparam_physics::nrw::nrw_direct_non_magnetic_with_config;
use sparam_training::checkpoint::load_model_checkpoint_bytes;
use sparam_training::losses::complex_relative_error_elements;

use super::config::{ModelConfig, StackedTrainConfig, TrainConfig, TrainMetricsSummary};
use super::train::{TrainRunResult, run_training, run_training_with_callback};
use sparam_data::tensor_bridges::samples_to_tensors;
use sparam_training::trainer::TrainingCallback;

#[derive(Debug, Clone)]
pub struct StackedTrainResult {
    /// First (or only) ensemble member's `run_training` result.
    pub primary: TrainRunResult,
    /// All member weights when ensembling. `None` for single-model.
    pub ensemble_weights: Option<Vec<Vec<u8>>>,
    /// Metrics for the ensembled-and-refined predictions; `None` for
    /// passthrough runs (use `primary.metrics`).
    pub stacked_metrics: Option<TrainMetricsSummary>,
}

pub fn run_stacked_training(config: &StackedTrainConfig) -> Result<StackedTrainResult> {
    // `NoopCallback` is the concrete type that pins `C` for the
    // callback factory; without it the closure returning `None` leaves
    // the generic ambiguous.
    run_stacked_training_with_callback(
        config,
        |_idx, _total| -> Option<NoopCallback> { None },
        |_msg| {},
    )
}

/// Placeholder callback used by [`run_stacked_training`] when no SSE
/// hook is wanted; never instantiated, exists purely so the generic
/// `C` on [`run_stacked_training_with_callback`] is concretely
/// inferable when the per-member factory always returns `None`.
pub struct NoopCallback;
impl TrainingCallback for NoopCallback {
    fn on_epoch_end(&mut self, _metrics: &sparam_training::trainer::EpochMetrics) {}
    fn on_improvement(&mut self, _epoch: usize, _val_loss: f64) {}
    fn on_early_stop(&mut self, _epoch: usize) {}
}

/// SSE-aware variant of [`run_stacked_training`]: identical behaviour
/// but accepts a `member_callback` factory + `stage_emit` text hook so
/// the web UI can surface per-epoch progress and inter-stage status
/// instead of staring at a frozen graph for the full ~30 s pipeline.
pub fn run_stacked_training_with_callback<C, F, S>(
    config: &StackedTrainConfig,
    mut member_callback: F,
    mut stage_emit: S,
) -> Result<StackedTrainResult>
where
    C: TrainingCallback + Send + 'static,
    F: FnMut(usize, usize) -> Option<C>,
    S: FnMut(String),
{
    if config.is_passthrough() {
        let primary = match member_callback(1, 1) {
            Some(cb) => run_training_with_callback(&config.base, cb)?,
            None => run_training(&config.base)?,
        };
        return Ok(StackedTrainResult {
            primary,
            ensemble_weights: None,
            stacked_metrics: None,
        });
    }

    // Stage 1: train overlay.
    if config.train_overlay_density > 0 {
        stage_emit(format!(
            "Synthesising {density}×{density} train overlay (stack stage 1/4)…",
            density = config.train_overlay_density,
        ));
    }
    let augmented_train: Arc<[PermittivitySample]> = if config.train_overlay_density > 0 {
        let extra =
            synthesise_train_overlay(config.train_overlay_density, config.base.seed)?;
        let mut joined: Vec<PermittivitySample> =
            Vec::with_capacity(config.base.train_samples.len() + extra.len());
        joined.extend_from_slice(&config.base.train_samples);
        joined.extend_from_slice(&extra);
        joined.into()
    } else {
        Arc::clone(&config.base.train_samples)
    };

    // Stage 2: per-member training loop.
    let ensemble_size = config.ensemble_size.max(1);
    let mut weights_per_member: Vec<Vec<u8>> = Vec::with_capacity(ensemble_size);
    let mut primary: Option<TrainRunResult> = None;

    for member_idx in 0..ensemble_size {
        let one_based = member_idx + 1;
        if ensemble_size > 1 {
            stage_emit(format!(
                "Training ensemble member {one_based} of {ensemble_size} (stack stage 2)…",
            ));
        }
        let member_seed = config.base.seed.wrapping_add(member_idx as u64);
        let mut member_cfg = config.base.clone();
        member_cfg.train_samples = Arc::clone(&augmented_train);
        member_cfg.seed = member_seed;
        if member_cfg.shuffle_seed.is_none() {
            member_cfg.shuffle_seed = Some(member_seed);
        }
        let result = match member_callback(one_based, ensemble_size) {
            Some(cb) => run_training_with_callback(&member_cfg, cb)?,
            None => run_training(&member_cfg)?,
        };
        weights_per_member.push(result.weights.clone());
        if member_idx == 0 {
            primary = Some(result);
        }
    }
    let primary = primary.expect("ensemble loop runs at least once");


    if ensemble_size == 1 && config.refine_steps == 0 {
        return Ok(StackedTrainResult {
            primary,
            ensemble_weights: None,
            stacked_metrics: None,
        });
    }

    if config.refine_steps > 0 {
        stage_emit(format!(
            "Computing ensemble median + {steps}-step NRW refinement (stack stage 3/4)…",
            steps = config.refine_steps,
        ));
    } else {
        stage_emit("Computing ensemble median metrics (stack stage 3/4)…".into());
    }

    // Stages 3+4: ensemble median + optional NRW refinement, with
    // per-step progress threading via `stage_emit`.
    let stacked_metrics = compute_stacked_metrics(
        &config.base,
        &augmented_train,
        &weights_per_member,
        config.refine_steps,
        config.refine_lr,
        &mut stage_emit,
    )?;

    stage_emit("Persisting weights + stacked metrics (stack stage 4/4)…".into());

    Ok(StackedTrainResult {
        primary,
        ensemble_weights: if ensemble_size > 1 {
            Some(weights_per_member)
        } else {
            None
        },
        stacked_metrics: Some(stacked_metrics),
    })
}

/// `N×N` dense patch in `[1, 2.5] × [0, 1.75]` for the stacked train
/// overlay. WR-90 / 8.2 GHz / 1.5 mm — same geometry as the
/// production data generator's defaults.
pub fn synthesise_train_overlay(
    density: usize,
    seed: u64,
) -> Result<Vec<PermittivitySample>> {
    if density == 0 {
        return Ok(Vec::new());
    }
    // `generate_non_magnetic_data` rejects zero train dims, so we ask
    // for a throwaway 10×10 alongside the dense patch and filter out
    // the bulk via `is_dense_patch`.
    let mut cfg = DataGenerationConfig::default();
    cfg.seed = seed;
    cfg.n_eps_prime_train = 10;
    cfg.n_eps_double_prime_train = 10;
    cfg.dense_patch_n_train = (density, density);
    cfg.dense_patch_n_eval = (0, 0);
    let dataset = generate_non_magnetic_data(&cfg)?;
    Ok(dataset.train.into_iter().filter(|s| s.is_dense_patch).collect())
}

/// Run forward + median + optional NRW refine through every saved
/// ensemble member; returns the final ε in the model's native
/// imag-sign convention. Used by `/evaluate` and `/infer` to replay
/// the full stacked pipeline at inference time.
pub fn stacked_inference(
    model_config: &ModelConfig,
    member_weights: &[&[u8]],
    scaler_samples: &[PermittivitySample],
    test_samples: &[PermittivitySample],
    refine_steps: usize,
    refine_lr: f64,
    pre_fitted: Option<&FittedScalers>,
) -> Result<ComplexTensor> {
    let device = Device::Cpu;
    let dtype = DType::F64;
    let is_complex = matches!(model_config, ModelConfig::Complex { .. });
    let model_type = if is_complex { ModelType::Complex } else { ModelType::Real };

    let test_ds = samples_to_tensors(model_type, test_samples, false)?;
    let test_features = test_ds.features;
    let (feature_scaler, target_scaler) = if let Some(fitted) = pre_fitted {
        (fitted.feature.clone(), fitted.target.clone())
    } else {
        let train_ds = samples_to_tensors(model_type, scaler_samples, false)?;
        let mut feature_scaler = StandardScaler::new();
        feature_scaler.fit(&train_ds.features)?;
        let mut target_scaler = StandardScaler::new();
        target_scaler.fit(&train_ds.targets)?;
        (feature_scaler, target_scaler)
    };

    // Build each ensemble member's model ONCE outside the chunk loop.
    // VarMap stays alive for the model's lifetime; Vars are Arc-shared
    // so the model retains them after `varmap` drops. Loading bytes
    // updates the VarMap in-place; we re-build the model from the same
    // VarBuilder so it sees the loaded weights through the shared Vars.
    let mut models: Vec<MlpModel> = Vec::with_capacity(member_weights.len());
    let mut _varmaps: Vec<VarMap> = Vec::with_capacity(member_weights.len());
    for weights in member_weights {
        let mut varmap = VarMap::new();
        {
            let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
            let _ = model_config.build_model(vb.pp("model"))?;
        }
        load_model_checkpoint_bytes(&mut varmap, weights)?;
        let model = {
            let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
            model_config.build_model(vb.pp("model"))?
        };
        models.push(model);
        _varmaps.push(varmap);
    }

    // Process the test set in chunks so the autograd graph for the
    // NRW refinement step never exceeds working memory.  100k rows ×
    // 16 frequencies × ~6 ops in the differentiable kernel is well
    // under 1 GB of intermediates per backward pass; without chunking,
    // a 2 M-row dense eval grid blows past tens of GBs and OOMs.
    // Inference (forward-only) is also chunked so we don't briefly
    // materialise three 2M-row prediction tensors before the median.
    let n_test = test_features.dims()[0];
    let chunk_size: usize = 100_000;
    let mut refined_real_chunks: Vec<Tensor> = Vec::new();
    let mut refined_imag_chunks: Vec<Tensor> = Vec::new();
    let mut start = 0;
    while start < n_test {
        let end = (start + chunk_size).min(n_test);
        let chunk_len = end - start;
        let chunk_features = test_features.narrow(0, start, chunk_len)?;

        // Forward pass for each ensemble member on this chunk.
        let mut member_chunk_preds: Vec<Tensor> = Vec::with_capacity(models.len());
        for model in &models {
            let scaled = feature_scaler.transform(&chunk_features)?;
            let raw = model.forward_t(&scaled, false)?;
            let clamped = apply_physical_softplus_clamp(
                &raw, ScalerRef::Standard(&target_scaler), is_complex,
            )?;
            let physical = target_scaler.inverse_transform(&clamped)?;
            member_chunk_preds.push(physical);
        }

        // Per-element median for this chunk.
        let median_chunk = if member_chunk_preds.len() <= 1 {
            member_chunk_preds.into_iter().next().expect("≥1 member")
        } else {
            per_element_median(&member_chunk_preds, &device, dtype)?
        };
        let pred_chunk_complex = unpack_complex_features(&median_chunk, 1)?;

        // Optional NRW refine for this chunk.
        let final_chunk = if refine_steps > 0 {
            let s_measured = build_s_measured(&chunk_features, is_complex)?;
            refine_with_nrw(
                &pred_chunk_complex,
                &s_measured,
                is_complex,
                refine_steps,
                refine_lr,
                &device,
            )?
        } else {
            pred_chunk_complex
        };

        refined_real_chunks.push(final_chunk.real);
        refined_imag_chunks.push(final_chunk.imag);
        start = end;
    }

    let real_refs: Vec<&Tensor> = refined_real_chunks.iter().collect();
    let imag_refs: Vec<&Tensor> = refined_imag_chunks.iter().collect();
    Ok(ComplexTensor::new_unchecked(
        Tensor::cat(&real_refs, 0)?,
        Tensor::cat(&imag_refs, 0)?,
    ))
}

/// Forward each member, take per-element median, optionally NRW-refine
/// the median; return test-set metrics. Chunked so the autograd graph
/// for the refinement step stays bounded — without it, a 2M-row dense
/// test set spends tens of GB of intermediate tensors per backward.
fn compute_stacked_metrics<S>(
    base: &TrainConfig,
    augmented_train: &Arc<[PermittivitySample]>,
    weights_per_member: &[Vec<u8>],
    refine_steps: usize,
    refine_lr: f64,
    mut stage_emit: S,
) -> Result<TrainMetricsSummary>
where
    S: FnMut(String),
{
    let device = Device::Cpu;
    let dtype = DType::F64;
    let is_complex = matches!(base.model, ModelConfig::Complex { .. });
    let model_type = if is_complex { ModelType::Complex } else { ModelType::Real };

    let test_ds = samples_to_tensors(model_type, &base.test_samples, false)?;
    let test_features = test_ds.features;
    let test_targets = test_ds.targets;
    let train_ds = samples_to_tensors(model_type, augmented_train.as_ref(), false)?;
    let mut feature_scaler = StandardScaler::new();
    feature_scaler.fit(&train_ds.features)?;
    let mut target_scaler = StandardScaler::new();
    target_scaler.fit(&train_ds.targets)?;

    let n_members = weights_per_member.len();
    let n_test = test_features.dims()[0];
    let chunk_size: usize = 100_000;
    let n_chunks = n_test.div_ceil(chunk_size);

    // Build each ensemble member's model ONCE outside the chunk loop —
    // see `stacked_inference` for the rationale on the dual build_model
    // pattern (load fills shared Vars, second build wires up the model).
    stage_emit(format!(
        "Stage 3/4: building {n_members} ensemble model{}…",
        if n_members == 1 { "" } else { "s" },
    ));
    let mut models: Vec<MlpModel> = Vec::with_capacity(n_members);
    let mut _varmaps: Vec<VarMap> = Vec::with_capacity(n_members);
    for weights in weights_per_member {
        let mut varmap = VarMap::new();
        {
            let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
            let _ = base.model.build_model(vb.pp("model"))?;
        }
        load_model_checkpoint_bytes(&mut varmap, weights)?;
        let model = {
            let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
            base.model.build_model(vb.pp("model"))?
        };
        models.push(model);
        _varmaps.push(varmap);
    }

    let mut refined_real_chunks: Vec<Tensor> = Vec::new();
    let mut refined_imag_chunks: Vec<Tensor> = Vec::new();
    let mut start = 0;
    while start < n_test {
        let end = (start + chunk_size).min(n_test);
        let chunk_idx = start / chunk_size + 1;
        let chunk_features = test_features.narrow(0, start, end - start)?;

        // Forward pass each ensemble member on this chunk.
        let mut member_chunk_preds: Vec<Tensor> = Vec::with_capacity(n_members);
        for (i, model) in models.iter().enumerate() {
            if n_chunks > 1 {
                stage_emit(format!(
                    "Stage 3/4: forward-pass chunk {chunk_idx}/{n_chunks}, member {}/{n_members}…",
                    i + 1
                ));
            } else {
                stage_emit(format!(
                    "Stage 3/4: forward-pass member {}/{n_members} on test set…",
                    i + 1
                ));
            }
            let scaled = feature_scaler.transform(&chunk_features)?;
            let raw = model.forward_t(&scaled, false)?;
            let clamped = apply_physical_softplus_clamp(
                &raw, ScalerRef::Standard(&target_scaler), is_complex,
            )?;
            let physical = target_scaler.inverse_transform(&clamped)?;
            member_chunk_preds.push(physical);
        }

        if n_members > 1 && n_chunks == 1 {
            stage_emit(format!(
                "Stage 3/4: computing per-element median across {n_members} members…"
            ));
        }
        let median_chunk = if member_chunk_preds.len() <= 1 {
            member_chunk_preds.into_iter().next().expect("≥1 member")
        } else {
            per_element_median(&member_chunk_preds, &device, dtype)?
        };
        let pred_chunk_complex = unpack_complex_features(&median_chunk, 1)?;

        let final_chunk = if refine_steps > 0 {
            let s_measured = build_s_measured(&chunk_features, is_complex)?;
            let log_every = refine_steps.div_ceil(10).max(1);
            refine_with_nrw_progress(
                &pred_chunk_complex,
                &s_measured,
                is_complex,
                refine_steps,
                refine_lr,
                &device,
                log_every,
                |step, total, loss| {
                    if n_chunks > 1 {
                        stage_emit(format!(
                            "Stage 3/4: chunk {chunk_idx}/{n_chunks}, NRW refinement {step}/{total} (residual = {loss:.3e})"
                        ));
                    } else {
                        stage_emit(format!(
                            "Stage 3/4: NRW refinement {step}/{total} (residual = {loss:.3e})"
                        ));
                    }
                },
            )?
        } else {
            pred_chunk_complex
        };

        refined_real_chunks.push(final_chunk.real);
        refined_imag_chunks.push(final_chunk.imag);
        start = end;
    }

    let real_refs: Vec<&Tensor> = refined_real_chunks.iter().collect();
    let imag_refs: Vec<&Tensor> = refined_imag_chunks.iter().collect();
    let final_pred = ComplexTensor::new_unchecked(
        Tensor::cat(&real_refs, 0)?,
        Tensor::cat(&imag_refs, 0)?,
    );
    let target_complex = unpack_complex_features(&test_targets, 1)?;
    let rel_err_tensor =
        complex_relative_error_elements(&final_pred, &target_complex)?;
    metrics_from_relative_error(&rel_err_tensor)
}

fn per_element_median(
    members: &[Tensor],
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let n_members = members.len();
    if n_members == 0 {
        return Err(candle_msg("per_element_median: empty members"));
    }

    // Row-major flat layout (`m * n_total + elem`) keeps each
    // member's slice contiguous so the prefetcher can stream it; a
    // `Vec<Vec<f64>>` would force a pointer-chase per inner-loop hit.
    let n_total = members[0].dims().iter().product::<usize>();
    let mut flat: Vec<f64> = Vec::with_capacity(n_members * n_total);
    for t in members {
        flat.extend(t.flatten_all()?.to_vec1::<f64>()?);
    }
    debug_assert_eq!(flat.len(), n_members * n_total);

    let mut median_flat: Vec<f64> = Vec::with_capacity(n_total);
    let mut buf: Vec<f64> = Vec::with_capacity(n_members);
    for elem_idx in 0..n_total {
        buf.clear();
        for m in 0..n_members {
            buf.push(flat[m * n_total + elem_idx]);
        }
        buf.sort_by(f64::total_cmp);
        let mid = if n_members % 2 == 1 {
            buf[n_members / 2]
        } else {
            0.5 * (buf[n_members / 2 - 1] + buf[n_members / 2])
        };
        median_flat.push(mid);
    }
    let dims = members[0].dims().to_vec();
    Tensor::from_vec(median_flat, dims.as_slice(), device)?.to_dtype(dtype)
}

fn build_s_measured(test_features: &Tensor, is_complex: bool) -> Result<ComplexTensor> {
    if is_complex {
        // Complex packed: cols [s11_re, s21_re, s11_im, s21_im] →
        // unpack(2) gives ComplexTensor with real=[s11_re, s21_re],
        // imag=[s11_im, s21_im].
        unpack_complex_features(test_features, 2)
    } else {
        // Real interleaved: cols [s11_re, s11_im, s21_re, s21_im] —
        // narrow per column to construct the same ComplexTensor.
        let c0 = test_features.narrow(1, 0, 1)?;
        let c1 = test_features.narrow(1, 1, 1)?;
        let c2 = test_features.narrow(1, 2, 1)?;
        let c3 = test_features.narrow(1, 3, 1)?;
        Ok(ComplexTensor::new_unchecked(
            Tensor::cat(&[&c0, &c2], 1)?,
            Tensor::cat(&[&c1, &c3], 1)?,
        ))
    }
}

fn refine_with_nrw(
    init: &ComplexTensor,
    s_measured: &ComplexTensor,
    is_complex: bool,
    steps: usize,
    lr: f64,
    device: &Device,
) -> Result<ComplexTensor> {
    // No progress callback wanted → `log_every = 0` skips the loss
    // device-sync entirely.
    refine_with_nrw_progress(
        init, s_measured, is_complex, steps, lr, device, 0,
        |_step, _total, _loss| {},
    )
}

/// As [`refine_with_nrw`], with `progress(step, total, loss)` invoked
/// only on logging steps (`step == steps` or `step % log_every == 0`).
/// The intermediate steps skip `loss.to_scalar` entirely — that's a
/// device-sync barrier we don't pay for if the caller isn't going to
/// surface it.
fn refine_with_nrw_progress<P>(
    init: &ComplexTensor,
    s_measured: &ComplexTensor,
    is_complex: bool,
    steps: usize,
    lr: f64,
    device: &Device,
    log_every: usize,
    mut progress: P,
) -> Result<ComplexTensor>
where
    P: FnMut(usize, usize, f64),
{
    // NRW expects ε in `(eps_re, -eps_im_physical)`. Complex packed
    // targets already store `-eps_im`; Real interleaved store `+eps_im`,
    // so we negate before NRW (autograd carries through to the Var).
    let real_var = Var::from_tensor(&init.real)?;
    let imag_var = Var::from_tensor(&init.imag)?;
    let physics = PhysicsContext::default_wr90(device)?;
    let mut optimizer =
        AdamW::new_lr(vec![real_var.clone(), imag_var.clone()], lr)?;
    for step in 0..steps {
        let eps_imag = if is_complex {
            imag_var.as_tensor().clone()
        } else {
            imag_var.as_tensor().neg()?
        };
        let eps = ComplexTensor::new_unchecked(real_var.as_tensor().clone(), eps_imag);
        let (s11_pred, s21_pred) = nrw_direct_non_magnetic_with_config(
            &physics.waveguide,
            &physics.frequencies,
            &eps,
        )?;
        let s_pred = ComplexTensor::new_unchecked(
            Tensor::cat(&[&s11_pred.real, &s21_pred.real], 1)?,
            Tensor::cat(&[&s11_pred.imag, &s21_pred.imag], 1)?,
        );
        let residual = s_pred.sub(s_measured)?;
        let loss = residual.mag_sq()?.mean_all()?;
        let grads = loss.backward()?;
        optimizer.step(&grads)?;
        let step_1 = step + 1;
        let should_log = log_every > 0 && (step_1 == steps || step_1 % log_every == 0);
        if should_log {
            let loss_val = loss.to_scalar::<f64>().unwrap_or(f64::NAN);
            progress(step_1, steps, loss_val);
        }
    }
    Ok(ComplexTensor::new_unchecked(
        real_var.as_tensor().clone(),
        imag_var.as_tensor().clone(),
    ))
}

fn metrics_from_relative_error(rel_err: &Tensor) -> Result<TrainMetricsSummary> {
    // `affine(100, 0)` fuses the % conversion into the device op so we
    // pull the data out scaled in one pass.
    let mut pct: Vec<f64> = rel_err
        .affine(100.0, 0.0)?
        .flatten_all()?
        .to_vec1::<f64>()?;
    let n = pct.len();
    if n == 0 {
        return Err(candle_msg(
            "metrics_from_relative_error: empty relative-error tensor",
        ));
    }
    let n_f = n as f64;
    // Single fold over `pct` collects max/min/sum/ok counters in one
    // pass — was 5 passes (max, min, sum, filter ≤1, filter ≤10).
    let mut max = f64::NEG_INFINITY;
    let mut min = f64::INFINITY;
    let mut sum = 0.0_f64;
    let mut ok1_count = 0_usize;
    let mut ok10_count = 0_usize;
    for &x in &pct {
        if x > max { max = x; }
        if x < min { min = x; }
        sum += x;
        if x <= 1.0 { ok1_count += 1; }
        if x <= 10.0 { ok10_count += 1; }
    }
    let mean = sum / n_f;
    // `select_nth_unstable_by` is O(n) average; picking the median
    // in-place avoids the sort-clone the previous code did. For even
    // `n` we pick the lower middle (cheap floor) — close enough to
    // the average-of-two-middles for our display use, and matches what
    // most stat libs do for f64 medians.
    let mid_idx = n / 2;
    let median = *pct.select_nth_unstable_by(mid_idx, f64::total_cmp).1;
    let ok_at_1 = ok1_count as f64 / n_f * 100.0;
    let ok_at_10 = ok10_count as f64 / n_f * 100.0;
    Ok(TrainMetricsSummary {
        ok_at_1pct: ok_at_1,
        ok_at_10pct: ok_at_10,
        mean_error: mean,
        max_error: max,
        min_error: min,
        median_error: median,
        // r2 / epoch / time fields don't apply to post-training stacked
        // metrics; callers that need r2 on the stacked predictions
        // compute it from the refined tensor directly.
        r2_real: 0.0,
        r2_imag: 0.0,
        best_epoch: 0,
        final_epoch: 0,
        stopped_early: false,
        training_time_secs: 0.0,
    })
}

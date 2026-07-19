//! Single-sample NRW + neural inversion. Caller passes the trained
//! checkpoint, training samples for scaler fit, and the S-params to
//! invert; this module owns the assembly. No HTTP / DB / CLI here.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{ModuleT, VarBuilder, VarMap};

use sparam_core::complex_tensor::ComplexTensor;
use sparam_core::error::candle_msg;
use sparam_core::s_params::S11S21;
use sparam_data::generation::PermittivitySample;
use sparam_data::scaling::{Scaler, StandardScaler};
use sparam_physics::nrw::{WaveguideConfig, nrw_inverse_with_config};
use sparam_training::checkpoint::load_model_checkpoint_bytes;

use sparam_data::tensor_bridges::samples_to_tensors;

use super::config::ModelConfig;

/// Closed-form NRW inversion. Returns `(ε', ε'')` with imag sign-
/// flipped to match the `ε = ε' − j·ε''` convention; `None` on
/// non-finite (singular / out-of-range input).
#[must_use]
pub fn run_nrw_inversion(
    s11_real: f64,
    s11_imag: f64,
    s21_real: f64,
    s21_imag: f64,
    frequency_hz: f64,
    thickness_m: f64,
) -> Option<(f64, f64)> {
    let device = Device::Cpu;
    let freq = Tensor::new(&[frequency_hz], &device).ok()?;
    let s11 = ComplexTensor::new(
        Tensor::new(&[s11_real], &device).ok()?,
        Tensor::new(&[s11_imag], &device).ok()?,
    )
    .ok()?;
    let s21 = ComplexTensor::new(
        Tensor::new(&[s21_real], &device).ok()?,
        Tensor::new(&[s21_imag], &device).ok()?,
    )
    .ok()?;
    let config = WaveguideConfig::new(thickness_m, 22.86e-3).ok()?;

    let (eps_r, _mu_r) = nrw_inverse_with_config(&config, &freq, &s11, &s21).ok()?;
    let eps_real = eps_r.real.to_vec1::<f64>().ok()?;
    let eps_imag = eps_r.imag.to_vec1::<f64>().ok()?;

    if eps_real[0].is_finite() && eps_imag[0].is_finite() {
        Some((eps_real[0], -eps_imag[0]))
    } else {
        None
    }
}

#[derive(Debug)]
pub struct NeuralNetInferRequest<'a> {
    pub checkpoint_bytes: &'a [u8],
    pub training_config_json: &'a str,
    /// Caller must supply the same split the run was trained on so
    /// the scaler fit matches.
    pub train_samples: &'a [PermittivitySample],
    pub s_params: S11S21,
    /// Pre-fitted scaler stats from a higher-level cache. When `Some`,
    /// the function skips the StandardScaler fit on `train_samples`.
    pub pre_fitted_scalers: Option<&'a sparam_data::scaling::FittedScalers>,
}

/// Single-sample neural inversion. Returns `(ε', ε'')` in the project
/// convention — Complex model output (which stores imag as `-ε''`) is
/// flipped on the way out.
pub fn run_neural_net_inference(req: NeuralNetInferRequest<'_>) -> Result<(f64, f64)> {
    let device = Device::Cpu;

    let mut model_config: ModelConfig = serde_json::from_str(req.training_config_json)
        .map_err(|e| candle_msg(format!("invalid model config JSON: {e}")))?;
    let model_type = model_config.model_type();

    // Architecture autodetect: a Complex MLP trained before the
    // `norm` field landed in the persisted JSON has `model.norm.*`
    // tensors in the BLOB but no `norm` key in the config; without
    // this the loader silently skips the LN weights and predictions
    // come out scrambled.
    if let Ok(names) = sparam_training::checkpoint::peek_checkpoint_tensor_names(
        req.checkpoint_bytes,
    ) {
        let has_complex_norm = names.iter().any(|n| n.starts_with("model.norm."));
        if let ModelConfig::Complex { ref mut norm, .. } = model_config {
            use sparam_models::ComplexNormChoice;
            *norm = if has_complex_norm {
                ComplexNormChoice::LayerNorm
            } else {
                ComplexNormChoice::None
            };
        }
    }

    let (feature_scaler, target_scaler) = if let Some(fitted) = req.pre_fitted_scalers {
        (fitted.feature.clone(), fitted.target.clone())
    } else {
        if req.train_samples.is_empty() {
            return Err(candle_msg("no training samples available to fit scalers"));
        }
        let train_ds = samples_to_tensors(model_type, req.train_samples, false)?;
        let mut feature_scaler = StandardScaler::new();
        feature_scaler.fit(&train_ds.features)?;
        let mut target_scaler = StandardScaler::new();
        target_scaler.fit(&train_ds.targets)?;
        (feature_scaler, target_scaler)
    };

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F64, &device);

    // Real interleaves `[s11r, s11i, s21r, s21i]`; Complex packs
    // half-half `[s11r, s21r, s11i, s21i]` — `ComplexLinear::forward`
    // narrows cols 0..2 as Re, 2..4 as Im. Cross-feeding scrambles.
    let row: [f64; 4] = if model_type.is_complex() {
        req.s_params.as_complex_row()
    } else {
        req.s_params.as_real_row()
    };
    let input = Tensor::new(&row, &device)?.reshape((1, 4))?;
    let scaled_input = feature_scaler.transform(&input)?;

    let model = model_config.build_model(vb.pp("model"))?;
    load_model_checkpoint_bytes(&mut varmap, req.checkpoint_bytes)?;
    let scaled_output = model.forward_t(&scaled_input, false)?;
    let scaled_output = sparam_data::physical_constraint::apply_physical_softplus_clamp(
        &scaled_output,
        sparam_data::scaling::ScalerRef::Standard(&target_scaler),
        model_type.is_complex(),
    )?;

    let output = target_scaler.inverse_transform(&scaled_output)?;
    let result = output.to_vec2::<f64>()?;
    if result.is_empty() || result[0].len() < 2 {
        return Err(candle_msg("model output shape too small"));
    }

    let (ep, edp) = (result[0][0], result[0][1]);
    if model_type.is_complex() { Ok((ep, -edp)) } else { Ok((ep, edp)) }
}

/// stacked inference on a single sample. Wraps [`stacked_inference`]
/// with a 1-row test set built from the user's S-params (ε targets are
/// dummy; the function only forwards). The refinement target IS the
/// user's input S — no extra ground truth needed.
pub fn run_neural_net_inference_stacked(
    req: NeuralNetInferRequest<'_>,
    extra_member_weights: &[Vec<u8>],
    refine_steps: usize,
    refine_lr: f64,
) -> Result<(f64, f64)> {
    let mut model_config: ModelConfig = serde_json::from_str(req.training_config_json)
        .map_err(|e| candle_msg(format!("invalid model config JSON: {e}")))?;
    // Same norm-autodetect as `run_neural_net_inference`.
    if let Ok(names) = sparam_training::checkpoint::peek_checkpoint_tensor_names(
        req.checkpoint_bytes,
    ) {
        let has_complex_norm = names.iter().any(|n| n.starts_with("model.norm."));
        if let ModelConfig::Complex { ref mut norm, .. } = model_config {
            use sparam_models::ComplexNormChoice;
            *norm = if has_complex_norm {
                ComplexNormChoice::LayerNorm
            } else {
                ComplexNormChoice::None
            };
        }
    }
    let model_type = model_config.model_type();

    let user_sample = PermittivitySample {
        s11_real: req.s_params.s11_real,
        s11_imag: req.s_params.s11_imag,
        s21_real: req.s_params.s21_real,
        s21_imag: req.s_params.s21_imag,
        eps_prime: 0.0,
        eps_double_prime: 0.0,
        is_dense_patch: false,
    };

    let mut all_members: Vec<&[u8]> =
        Vec::with_capacity(1 + extra_member_weights.len());
    all_members.push(req.checkpoint_bytes);
    all_members.extend(extra_member_weights.iter().map(|v| v.as_slice()));

    let predicted = crate::workflows::stacked_training::stacked_inference(
        &model_config,
        &all_members,
        req.train_samples,
        std::slice::from_ref(&user_sample),
        refine_steps,
        refine_lr,
        req.pre_fitted_scalers,
    )?;

    let real_v = predicted.real.flatten_all()?.to_vec1::<f64>()?;
    let imag_v = predicted.imag.flatten_all()?.to_vec1::<f64>()?;
    if real_v.is_empty() || imag_v.is_empty() {
        return Err(candle_msg(
            "stacked_inference returned empty prediction tensor",
        ));
    }
    let ep = real_v[0];
    let edp = imag_v[0];
    if model_type.is_complex() { Ok((ep, -edp)) } else { Ok((ep, edp)) }
}

//! Loss functions for real-valued and complex-valued regression.
//!
//! Each loss is available in both a real variant (operating on [`Tensor`])
//! and a complex variant (operating on [`ComplexTensor`]).  All losses accept
//! a [`Reduction`] mode (mean, sum, or none).
//!
//! | Loss | Real | Complex |
//! |------|------|---------|
//! | MSE | [`mse_loss`] | [`complex_mse_loss`] |
//! | Smooth L1 | [`smooth_l1_loss`] | [`complex_smooth_l1_loss`] |
//! | Relative error (elements) | [`relative_error_elements`] | [`complex_relative_error_elements`] |

use std::{fmt, str::FromStr};

use candle_core::{Result, Tensor};
use serde::{Deserialize, Serialize};

use sparam_core::complex_tensor::ComplexTensor;
use sparam_core::config::normalize_config_name;
use sparam_core::error::candle_msg;
use sparam_core::validation::validate_positive_f64;
use sparam_physics::nrw::{WaveguideConfig, nrw_direct_non_magnetic_with_config};

/// Default beta used by Smooth L1 / Huber-style packed training loss selection.
pub const DEFAULT_SMOOTH_L1_BETA: f64 = 1.0;

pub use sparam_physics::PhysicsContext;

/// Loss choices for packed model outputs. `EpsRelMse` and
/// `PhysicsForward` both need the target-scaler inverse-transform
/// before the residual runs — dispatched by
/// `sparam_app::workflows::internal::training::packed_loss_fn`.
///
/// `PhysicsForward` is the sole Complex-MLP loss: it plugs the
/// predicted ε̂ through the NRW forward model
/// ([`nrw_direct_non_magnetic_with_config`]) to reconstruct S₁₁ and
/// S₂₁, then MSEs them against the same forward model evaluated on
/// the target ε (which, by dataset construction, reproduces the
/// measured S-parameters to numerical precision).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MlpLoss {
    Mse,
    SmoothL1 { beta: f64 },
    EpsRelMse,
    /// Physics-informed loss: `MSE(NRW(ε̂), NRW(ε))` on reconstructed
    /// S-parameters. Complex MLP only; requires scaler
    /// inverse-transform to run NRW in physical units.
    PhysicsForward,
}

impl MlpLoss {
    #[inline]
    #[must_use]
    pub fn is_eps_rel_mse(self) -> bool {
        matches!(self, Self::EpsRelMse)
    }

    /// True when the loss requires the scaler-inverse round-trip
    /// before computing residuals. `EpsRelMse` runs in physical
    /// data-space for the relative denominator; `PhysicsForward`
    /// needs physical ε to feed the NRW forward model. MSE / Smooth
    /// L1 stay in scaled space because their residuals are
    /// scale-equivariant.
    #[inline]
    #[must_use]
    pub fn needs_unscaled_targets(self) -> bool {
        matches!(self, Self::EpsRelMse | Self::PhysicsForward)
    }

    /// True for losses that only make sense on complex-valued outputs.
    /// Currently only `PhysicsForward` — the NRW forward model
    /// consumes a `ComplexTensor` ε and the reconstructed S-parameters
    /// are intrinsically complex.
    #[inline]
    #[must_use]
    pub fn is_complex_only(self) -> bool {
        matches!(self, Self::PhysicsForward)
    }

    pub fn from_name(name: &str) -> Result<Self> {
        let normalized = normalize_config_name(name);
        match normalized.as_str() {
            "mse" => Ok(Self::Mse),
            "smooth_l1" | "huber" => Ok(Self::SmoothL1 {
                beta: DEFAULT_SMOOTH_L1_BETA,
            }),
            "eps_rel_mse" | "epsrelmse" | "rel_mse" => Ok(Self::EpsRelMse),
            // `pi_mape` kept as a legacy alias so pre-existing HPO
            // studies / train_runs persisted with the old name still
            // deserialize and round-trip through this enum.
            "physics_forward" | "physicsforward" | "physics-forward" | "nrw" | "pi_mape"
            | "pimape" | "pi-mape" => Ok(Self::PhysicsForward),
            other => Err(candle_msg(format!(
                "unknown loss '{other}', expected: mse, smooth_l1, eps_rel_mse, physics_forward"
            ))),
        }
    }
}

impl fmt::Display for MlpLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mse => f.write_str("mse"),
            Self::SmoothL1 { .. } => f.write_str("smooth_l1"),
            Self::EpsRelMse => f.write_str("eps_rel_mse"),
            Self::PhysicsForward => f.write_str("physics_forward"),
        }
    }
}

impl FromStr for MlpLoss {
    type Err = candle_core::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_name(s)
    }
}

#[cold]
#[cfg(debug_assertions)]
fn shape_mismatch_error(
    loss_name: &str,
    pred_shape: &candle_core::Shape,
    target_shape: &candle_core::Shape,
) -> candle_core::Error {
    candle_msg(format!(
        "Shape mismatch for {loss_name}: pred {:?} vs target {:?}",
        pred_shape, target_shape
    ))
}

/// Reduction mode applied to an element-wise loss tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Reduction {
    /// Average over all elements.
    #[default]
    #[serde(rename = "mean", alias = "avg", alias = "average")]
    Mean,
    /// Sum over all elements.
    #[serde(rename = "sum")]
    Sum,
    /// Return the unreduced, element-wise loss tensor.
    #[serde(rename = "none", alias = "no_reduction")]
    None,
}

impl Reduction {
    /// Parse a reduction configuration string.
    pub fn from_name(name: &str) -> Result<Self> {
        let normalized = normalize_config_name(name);
        match normalized.as_str() {
            "mean" | "avg" | "average" => Ok(Self::Mean),
            "sum" => Ok(Self::Sum),
            "none" | "no_reduction" => Ok(Self::None),
            _ => Err(candle_msg(format!("unknown reduction: {name}"))),
        }
    }

    /// Reduce an element-wise loss tensor according to the selected mode.
    pub fn reduce(self, values: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mean => {
                if values.dims().iter().product::<usize>() == 0 {
                    return Err(candle_msg(
                        "Reduction::Mean requires at least one loss element",
                    ));
                }
                values.mean_all()
            }
            Self::Sum => values.sum_all(),
            Self::None => Ok(values.clone()),
        }
    }
}

impl fmt::Display for Reduction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mean => write!(f, "mean"),
            Self::Sum => write!(f, "sum"),
            Self::None => write!(f, "none"),
        }
    }
}

impl FromStr for Reduction {
    type Err = candle_core::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_name(s)
    }
}

/// Per-batch loss-input shape check, gated behind `debug_assertions`.
/// Every loss call hits this; the check compounds over an HPO
/// campaign. Release builds skip it and let Candle's `sub`/`sqr`
/// kernel raise the mismatch instead.
pub fn validate_same_shape(pred: &Tensor, target: &Tensor, loss_name: &str) -> Result<()> {
    #[cfg(debug_assertions)]
    {
        return validate_matching_shapes(pred.shape(), target.shape(), loss_name);
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (pred, target, loss_name);
        Ok(())
    }
}

/// Complex variant of [`validate_same_shape`]; same gating rationale.
pub fn validate_same_complex_shape(
    pred: &ComplexTensor,
    target: &ComplexTensor,
    loss_name: &str,
) -> Result<()> {
    #[cfg(debug_assertions)]
    {
        let pred_shape = pred.shape();
        let target_shape = target.shape();
        return validate_matching_shapes(&pred_shape, &target_shape, loss_name);
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (pred, target, loss_name);
        Ok(())
    }
}

#[cfg(debug_assertions)]
fn validate_matching_shapes(
    pred_shape: &candle_core::Shape,
    target_shape: &candle_core::Shape,
    loss_name: &str,
) -> Result<()> {
    if pred_shape != target_shape {
        return Err(shape_mismatch_error(loss_name, pred_shape, target_shape));
    }

    Ok(())
}

fn validate_beta(beta: f64) -> Result<()> {
    if !beta.is_finite() || beta <= 0.0 {
        return Err(candle_msg(format!(
            "smooth_l1 beta must be finite and positive, got {beta}"
        )));
    }
    Ok(())
}

/// Validate that an epsilon value is finite and positive.
pub fn validate_epsilon(epsilon: f64, context: &str) -> Result<()> {
    validate_positive_f64(&format!("{context} epsilon"), epsilon)?;
    Ok(())
}

fn abs_tensor(values: &Tensor) -> Result<Tensor> {
    values.abs()
}

/// Core Smooth L1 element-wise computation.  Caller must ensure `beta > 0`.
#[inline]
fn smooth_l1_elements(distance: &Tensor, beta: f64) -> Result<Tensor> {
    let quadratic = distance.clamp(0.0, beta)?;
    let linear = distance.broadcast_sub(&quadratic)?;

    quadratic
        .sqr()?
        .affine(0.5 / beta, 0.0)?
        .broadcast_add(&linear)
}

/// Element-wise relative error: `|pred − target| / |target|`.
///
/// The dataset enforces `ε' ≥ 1` so `|ε_target| ≥ 1` and the denominator
/// is always well above any floating-point cancellation threshold;
/// no `max(|target|, ε)` floor is applied.
pub fn relative_error_elements(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    let error_magnitude = abs_tensor(&pred.sub(target)?)?;
    let denominator = abs_tensor(target)?;
    error_magnitude.broadcast_div(&denominator)
}

/// Element-wise complex relative error: `|pred − target| / |target|`.
/// Same domain assumption as [`relative_error_elements`] — no floor.
pub fn complex_relative_error_elements(
    pred: &ComplexTensor,
    target: &ComplexTensor,
) -> Result<Tensor> {
    let error_magnitude = pred.sub(target)?.mag()?;
    let denominator = target.mag()?;
    error_magnitude.broadcast_div(&denominator)
}

/// Mean squared error for real-valued predictions.
///
/// Element-wise loss: $(\hat{y} - y)^2$
pub fn mse_loss(pred: &Tensor, target: &Tensor, reduction: Reduction) -> Result<Tensor> {
    validate_same_shape(pred, target, "mse_loss")?;

    let squared_error = pred.sub(target)?.sqr()?;
    reduction.reduce(&squared_error)
}

/// Mean squared error for complex-valued predictions.
///
/// Element-wise loss: $|\hat{z} - z|^2 = (\hat{z}_r - z_r)^2 + (\hat{z}_i - z_i)^2$
pub fn complex_mse_loss(
    pred: &ComplexTensor,
    target: &ComplexTensor,
    reduction: Reduction,
) -> Result<Tensor> {
    validate_same_complex_shape(pred, target, "complex_mse_loss")?;

    let squared_error = pred.sub(target)?.mag_sq()?;
    reduction.reduce(&squared_error)
}

/// Smooth L1 / Huber-style loss for real-valued predictions.
///
/// Element-wise loss:
/// - $0.5 (\hat{y} - y)^2 / \beta$ when $|\hat{y} - y| < \beta$
/// - $|\hat{y} - y| - 0.5\beta$ otherwise
pub fn smooth_l1_loss(
    pred: &Tensor,
    target: &Tensor,
    beta: f64,
    reduction: Reduction,
) -> Result<Tensor> {
    validate_same_shape(pred, target, "smooth_l1_loss")?;
    validate_beta(beta)?;

    let diff = pred.sub(target)?;
    let abs_diff = abs_tensor(&diff)?;
    let loss = smooth_l1_elements(&abs_diff, beta)?;
    reduction.reduce(&loss)
}

/// Smooth L1 / Huber-style loss for complex-valued predictions.
///
/// The complex error is reduced to its magnitude before applying the scalar
/// Smooth L1 formula.
pub fn complex_smooth_l1_loss(
    pred: &ComplexTensor,
    target: &ComplexTensor,
    beta: f64,
    reduction: Reduction,
) -> Result<Tensor> {
    validate_same_complex_shape(pred, target, "complex_smooth_l1_loss")?;
    validate_beta(beta)?;

    let distance = pred.sub(target)?.mag()?;
    let loss = smooth_l1_elements(&distance, beta)?;
    reduction.reduce(&loss)
}

/// Physics-informed loss for complex permittivity.
/// Complex-only.
///
/// Runs the NRW forward model on the predicted ε̂, then compares the
/// reconstructed `(Ŝ₁₁, Ŝ₂₁)` against the batch's **measured**
/// S-parameters — which we already have as the model input, so no
/// second NRW evaluation on `target_eps` is needed.
///
/// ```text
/// (Ŝ₁₁, Ŝ₂₁)  =  NRW(ε̂; cfg, f)
///
/// L  =  MSE(Ŝ₁₁ − S₁₁)  +  MSE(Ŝ₂₁ − S₂₁)
/// ```
///
/// `s_target` is a 2-column complex tensor carrying `[S₁₁, S₂₁]` per
/// batch element (column 0 = S₁₁, column 1 = S₂₁) in raw physical
/// units. `ε̂` is expected in physical units too; the caller is
/// responsible for the feature-scaler / target-scaler inverse round-
/// trips.
pub fn physics_forward_loss(
    pred_eps: &ComplexTensor,
    s_target: &ComplexTensor,
    waveguide: &WaveguideConfig,
    frequencies: &Tensor,
    reduction: Reduction,
) -> Result<Tensor> {
    let pred_dims = pred_eps.real.dims();
    let s_dims = s_target.real.dims();
    if pred_dims.len() != 2 || s_dims.len() != 2 {
        return Err(candle_msg(format!(
            "physics_forward_loss expects 2D complex tensors, got pred {:?} and s_target {:?}",
            pred_dims, s_dims
        )));
    }
    if pred_dims[0] != s_dims[0] {
        return Err(candle_msg(format!(
            "physics_forward_loss batch size mismatch: pred {} vs s_target {}",
            pred_dims[0], s_dims[0]
        )));
    }
    if s_dims[1] != 2 {
        return Err(candle_msg(format!(
            "physics_forward_loss expects s_target with 2 complex columns (S₁₁, S₂₁), got {}",
            s_dims[1]
        )));
    }

    let (s11_pred, s21_pred) =
        nrw_direct_non_magnetic_with_config(waveguide, frequencies, pred_eps)?;

    // Concat S₁₁ + S₂₁ predictions into a single 2-column complex
    // tensor and compute the residual in one shot — one sub, one
    // |·|², one reduction. `sum_keepdim(1)` collapses the two
    // S-parameter columns per sample, preserving the old
    // `MSE(S₁₁) + MSE(S₂₁)` semantic exactly (sum of the two
    // per-column means equals the batch-mean of the per-sample
    // column sums).
    let s_pred = ComplexTensor::new_unchecked(
        Tensor::cat(&[&s11_pred.real, &s21_pred.real], 1)?,
        Tensor::cat(&[&s11_pred.imag, &s21_pred.imag], 1)?,
    );
    let per_sample = s_pred.sub(s_target)?.mag_sq()?.sum_keepdim(1)?;
    reduction.reduce(&per_sample)
}


#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor, Var};

    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_tensor_close(actual: &Tensor, expected: &[f64], tol: f64) -> Result<()> {
        let values = actual.flatten_all()?.to_vec1::<f64>()?;
        assert_eq!(values.len(), expected.len());
        for (actual_value, expected_value) in values.iter().zip(expected.iter()) {
            assert_close(*actual_value, *expected_value, tol);
        }
        Ok(())
    }

    #[test]
    fn test_reduction_default_is_mean() {
        assert_eq!(Reduction::default(), Reduction::Mean);
    }

    #[test]
    fn test_reduction_from_name_accepts_aliases() -> Result<()> {
        assert_eq!(Reduction::from_name("mean")?, Reduction::Mean);
        assert_eq!(Reduction::from_name("avg")?, Reduction::Mean);
        assert_eq!(Reduction::from_name("Average")?, Reduction::Mean);
        assert_eq!(Reduction::from_name("sum")?, Reduction::Sum);
        assert_eq!(Reduction::from_name("no reduction")?, Reduction::None);
        assert_eq!(Reduction::from_str("none")?, Reduction::None);

        Ok(())
    }

    #[test]
    fn test_reduction_from_name_rejects_unknown_values() {
        let error = Reduction::from_name("median").unwrap_err().to_string();
        assert!(error.contains("unknown reduction: median"));
    }

    #[test]
    fn test_reduction_display_is_log_friendly() {
        assert_eq!(Reduction::Mean.to_string(), "mean");
        assert_eq!(Reduction::Sum.to_string(), "sum");
        assert_eq!(Reduction::None.to_string(), "none");
    }

    #[test]
    fn test_reduction_serde_roundtrip_and_aliases() -> Result<()> {
        let json = serde_json::to_string(&Reduction::Mean)
            .map_err(|error| candle_msg(error.to_string()))?;
        assert_eq!(json, "\"mean\"");

        let restored: Reduction =
            serde_json::from_str(&json).map_err(|error| candle_msg(error.to_string()))?;
        assert_eq!(restored, Reduction::Mean);

        let avg_alias: Reduction =
            serde_json::from_str("\"avg\"").map_err(|error| candle_msg(error.to_string()))?;
        assert_eq!(avg_alias, Reduction::Mean);

        let none_alias: Reduction = serde_json::from_str("\"no_reduction\"")
            .map_err(|error| candle_msg(error.to_string()))?;
        assert_eq!(none_alias, Reduction::None);

        Ok(())
    }

    #[test]
    fn test_mlp_loss_from_name_accepts_aliases() -> Result<()> {
        assert_eq!(MlpLoss::from_name("mse")?, MlpLoss::Mse);
        assert_eq!(
            MlpLoss::from_name("smooth_l1")?,
            MlpLoss::SmoothL1 {
                beta: DEFAULT_SMOOTH_L1_BETA,
            }
        );
        assert_eq!(
            MlpLoss::from_name("Huber")?,
            MlpLoss::SmoothL1 {
                beta: DEFAULT_SMOOTH_L1_BETA,
            }
        );
        assert_eq!(MlpLoss::from_str("mse")?, MlpLoss::Mse);
        Ok(())
    }

    #[test]
    fn test_mlp_loss_from_name_rejects_unknown_values() {
        let error = MlpLoss::from_name("relative_error")
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown loss 'relative_error'"));
    }

    #[test]
    fn test_mlp_loss_display_is_log_friendly() {
        assert_eq!(MlpLoss::Mse.to_string(), "mse");
        assert_eq!(
            MlpLoss::SmoothL1 {
                beta: DEFAULT_SMOOTH_L1_BETA,
            }
            .to_string(),
            "smooth_l1"
        );
    }

    #[test]
    fn test_mse_loss_real_reduction_modes() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], (2, 2), &device)?;
        let target = Tensor::from_vec(vec![0.0f64, 1.0, 2.0, 2.0], (2, 2), &device)?;

        let none = mse_loss(&pred, &target, Reduction::None)?;
        assert_eq!(none.dims(), &[2, 2]);
        assert_tensor_close(&none, &[1.0, 1.0, 1.0, 4.0], 1e-12)?;

        let sum = mse_loss(&pred, &target, Reduction::Sum)?;
        assert_close(sum.to_scalar::<f64>()?, 7.0, 1e-12);

        let mean = mse_loss(&pred, &target, Reduction::Mean)?;
        assert_close(mean.to_scalar::<f64>()?, 1.75, 1e-12);

        Ok(())
    }

    #[test]
    fn test_complex_mse_loss_reduction_modes() -> Result<()> {
        let device = Device::Cpu;
        let pred = ComplexTensor::new(
            Tensor::from_vec(vec![1.0f64, 2.0], (1, 2), &device)?,
            Tensor::from_vec(vec![1.0f64, 0.0], (1, 2), &device)?,
        )?;
        let target = ComplexTensor::new(
            Tensor::zeros((1, 2), DType::F64, &device)?,
            Tensor::zeros((1, 2), DType::F64, &device)?,
        )?;

        let none = complex_mse_loss(&pred, &target, Reduction::None)?;
        assert_eq!(none.dims(), &[1, 2]);
        assert_tensor_close(&none, &[2.0, 4.0], 1e-12)?;

        let sum = complex_mse_loss(&pred, &target, Reduction::Sum)?;
        assert_close(sum.to_scalar::<f64>()?, 6.0, 1e-12);

        let mean = complex_mse_loss(&pred, &target, Reduction::Mean)?;
        assert_close(mean.to_scalar::<f64>()?, 3.0, 1e-12);

        Ok(())
    }

    #[test]
    fn test_complex_mse_loss_is_zero_for_identical_inputs() -> Result<()> {
        let device = Device::Cpu;
        let pred = ComplexTensor::new(
            Tensor::from_vec(vec![1.0f64, -2.0], (1, 2), &device)?,
            Tensor::from_vec(vec![0.5f64, 3.0], (1, 2), &device)?,
        )?;

        let loss = complex_mse_loss(&pred, &pred, Reduction::Mean)?;
        assert_close(loss.to_scalar::<f64>()?, 0.0, 1e-12);

        Ok(())
    }

    #[test]
    fn test_complex_mse_loss_matches_componentwise_squared_error() -> Result<()> {
        let device = Device::Cpu;
        let pred = ComplexTensor::new(
            Tensor::from_vec(vec![1.0f64, 3.0, -1.0, 2.0], (2, 2), &device)?,
            Tensor::from_vec(vec![0.5f64, -1.0, 2.5, -3.0], (2, 2), &device)?,
        )?;
        let target = ComplexTensor::new(
            Tensor::from_vec(vec![0.0f64, 1.0, -2.0, 1.0], (2, 2), &device)?,
            Tensor::from_vec(vec![-0.5f64, -2.0, 1.5, -1.0], (2, 2), &device)?,
        )?;

        let expected = pred
            .real
            .sub(&target.real)?
            .sqr()?
            .add(&pred.imag.sub(&target.imag)?.sqr()?)?;
        let loss = complex_mse_loss(&pred, &target, Reduction::None)?;

        assert_tensor_close(&loss, &expected.flatten_all()?.to_vec1::<f64>()?, 1e-12)?;

        Ok(())
    }

    #[test]
    fn test_mse_loss_rejects_shape_mismatch() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::zeros((2, 2), DType::F64, &device)?;
        let target = Tensor::zeros((2, 1), DType::F64, &device)?;

        assert!(mse_loss(&pred, &target, Reduction::Mean).is_err());

        Ok(())
    }

    #[test]
    fn test_complex_mse_loss_rejects_shape_mismatch() -> Result<()> {
        let device = Device::Cpu;
        let pred = ComplexTensor::new(
            Tensor::zeros((2, 2), DType::F64, &device)?,
            Tensor::zeros((2, 2), DType::F64, &device)?,
        )?;
        let target = ComplexTensor::new(
            Tensor::zeros((2, 1), DType::F64, &device)?,
            Tensor::zeros((2, 1), DType::F64, &device)?,
        )?;

        assert!(complex_mse_loss(&pred, &target, Reduction::Mean).is_err());

        Ok(())
    }

    #[test]
    fn test_reduction_mean_rejects_empty_loss_tensor() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::zeros((0, 2), DType::F64, &device)?;
        let target = Tensor::zeros((0, 2), DType::F64, &device)?;

        assert!(mse_loss(&pred, &target, Reduction::Mean).is_err());

        let pred_complex = ComplexTensor::new(pred.clone(), target.clone())?;
        let target_complex = ComplexTensor::new(
            Tensor::zeros((0, 2), DType::F64, &device)?,
            Tensor::zeros((0, 2), DType::F64, &device)?,
        )?;
        assert!(complex_mse_loss(&pred_complex, &target_complex, Reduction::Mean).is_err());

        Ok(())
    }

    #[test]
    fn test_mse_loss_gradient_matches_mean_reduction_formula() -> Result<()> {
        let device = Device::Cpu;
        let pred = Var::from_tensor(&Tensor::from_vec(vec![1.0f64, -2.0], (1, 2), &device)?)?;
        let target = Tensor::zeros((1, 2), DType::F64, &device)?;

        let loss = mse_loss(pred.as_tensor(), &target, Reduction::Mean)?;
        let grads = loss.backward()?;
        let grad = grads
            .get(&pred)
            .cloned()
            .ok_or_else(|| candle_msg("missing gradient for mse pred"))?;

        assert_tensor_close(&grad, &[1.0, -2.0], 1e-12)?;

        Ok(())
    }

    #[test]
    fn test_complex_mse_loss_gradient_matches_mean_reduction_formula() -> Result<()> {
        let device = Device::Cpu;
        let pred_real = Var::from_tensor(&Tensor::from_vec(vec![1.0f64, -2.0], (1, 2), &device)?)?;
        let pred_imag = Var::from_tensor(&Tensor::from_vec(vec![0.5f64, 1.5], (1, 2), &device)?)?;
        let pred =
            ComplexTensor::new(pred_real.as_tensor().clone(), pred_imag.as_tensor().clone())?;
        let target = ComplexTensor::new(
            Tensor::zeros((1, 2), DType::F64, &device)?,
            Tensor::zeros((1, 2), DType::F64, &device)?,
        )?;

        let loss = complex_mse_loss(&pred, &target, Reduction::Mean)?;
        let grads = loss.backward()?;
        let grad_real = grads
            .get(&pred_real)
            .cloned()
            .ok_or_else(|| candle_msg("missing gradient for complex mse real part"))?;
        let grad_imag = grads
            .get(&pred_imag)
            .cloned()
            .ok_or_else(|| candle_msg("missing gradient for complex mse imaginary part"))?;

        assert_tensor_close(&grad_real, &[1.0, -2.0], 1e-12)?;
        assert_tensor_close(&grad_imag, &[0.5, 1.5], 1e-12)?;

        Ok(())
    }

    #[test]
    fn test_smooth_l1_loss_real_reduction_modes() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::from_vec(vec![0.5f64, -0.5, 2.0, -2.0], (2, 2), &device)?;
        let target = Tensor::zeros((2, 2), DType::F64, &device)?;

        let none = smooth_l1_loss(&pred, &target, 1.0, Reduction::None)?;
        assert_eq!(none.dims(), &[2, 2]);
        assert_tensor_close(&none, &[0.125, 0.125, 1.5, 1.5], 1e-12)?;

        let sum = smooth_l1_loss(&pred, &target, 1.0, Reduction::Sum)?;
        assert_close(sum.to_scalar::<f64>()?, 3.25, 1e-12);

        let mean = smooth_l1_loss(&pred, &target, 1.0, Reduction::Mean)?;
        assert_close(mean.to_scalar::<f64>()?, 0.8125, 1e-12);

        Ok(())
    }

    #[test]
    fn test_smooth_l1_loss_matches_boundary_value() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::from_vec(vec![1.0f64], 1, &device)?;
        let target = Tensor::zeros(1, DType::F64, &device)?;

        let loss = smooth_l1_loss(&pred, &target, 1.0, Reduction::Mean)?;
        assert_close(loss.to_scalar::<f64>()?, 0.5, 1e-12);

        Ok(())
    }

    #[test]
    fn test_complex_smooth_l1_loss_reduction_modes() -> Result<()> {
        let device = Device::Cpu;
        let pred = ComplexTensor::new(
            Tensor::from_vec(vec![0.3f64, 1.2], (1, 2), &device)?,
            Tensor::from_vec(vec![0.4f64, 1.6], (1, 2), &device)?,
        )?;
        let target = ComplexTensor::new(
            Tensor::zeros((1, 2), DType::F64, &device)?,
            Tensor::zeros((1, 2), DType::F64, &device)?,
        )?;

        let none = complex_smooth_l1_loss(&pred, &target, 1.0, Reduction::None)?;
        assert_eq!(none.dims(), &[1, 2]);
        assert_tensor_close(&none, &[0.125, 1.5], 1e-12)?;

        let sum = complex_smooth_l1_loss(&pred, &target, 1.0, Reduction::Sum)?;
        assert_close(sum.to_scalar::<f64>()?, 1.625, 1e-12);

        let mean = complex_smooth_l1_loss(&pred, &target, 1.0, Reduction::Mean)?;
        assert_close(mean.to_scalar::<f64>()?, 0.8125, 1e-12);

        Ok(())
    }

    #[test]
    fn test_complex_smooth_l1_loss_is_zero_for_identical_inputs() -> Result<()> {
        let device = Device::Cpu;
        let pred = ComplexTensor::new(
            Tensor::from_vec(vec![1.0f64, -2.0], (1, 2), &device)?,
            Tensor::from_vec(vec![0.5f64, 3.0], (1, 2), &device)?,
        )?;

        let loss = complex_smooth_l1_loss(&pred, &pred, 1.0, Reduction::Mean)?;
        assert_close(loss.to_scalar::<f64>()?, 0.0, 1e-12);

        Ok(())
    }

    #[test]
    fn test_smooth_l1_loss_rejects_invalid_beta() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::zeros((1, 1), DType::F64, &device)?;
        let target = Tensor::zeros((1, 1), DType::F64, &device)?;

        let error = smooth_l1_loss(&pred, &target, 0.0, Reduction::Mean)
            .unwrap_err()
            .to_string();
        assert!(error.contains("smooth_l1 beta must be finite and positive"));

        let complex_pred = ComplexTensor::new(pred.clone(), target.clone())?;
        let complex_target = ComplexTensor::new(
            Tensor::zeros((1, 1), DType::F64, &device)?,
            Tensor::zeros((1, 1), DType::F64, &device)?,
        )?;
        let complex_error = complex_smooth_l1_loss(
            &complex_pred,
            &complex_target,
            f64::INFINITY,
            Reduction::Mean,
        )
        .unwrap_err()
        .to_string();
        assert!(complex_error.contains("smooth_l1 beta must be finite and positive"));

        Ok(())
    }

    #[test]
    fn test_smooth_l1_loss_rejects_shape_mismatch() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::zeros((2, 2), DType::F64, &device)?;
        let target = Tensor::zeros((2, 1), DType::F64, &device)?;

        assert!(smooth_l1_loss(&pred, &target, 1.0, Reduction::Mean).is_err());

        Ok(())
    }

    #[test]
    fn test_complex_smooth_l1_loss_rejects_shape_mismatch() -> Result<()> {
        let device = Device::Cpu;
        let pred = ComplexTensor::new(
            Tensor::zeros((2, 2), DType::F64, &device)?,
            Tensor::zeros((2, 2), DType::F64, &device)?,
        )?;
        let target = ComplexTensor::new(
            Tensor::zeros((2, 1), DType::F64, &device)?,
            Tensor::zeros((2, 1), DType::F64, &device)?,
        )?;

        assert!(complex_smooth_l1_loss(&pred, &target, 1.0, Reduction::Mean).is_err());

        Ok(())
    }

    #[test]
    fn test_smooth_l1_mean_rejects_empty_tensor() -> Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::zeros((0, 2), DType::F64, &device)?;
        let target = Tensor::zeros((0, 2), DType::F64, &device)?;

        assert!(smooth_l1_loss(&pred, &target, 1.0, Reduction::Mean).is_err());

        let pred_complex = ComplexTensor::new(pred.clone(), target.clone())?;
        let target_complex = ComplexTensor::new(
            Tensor::zeros((0, 2), DType::F64, &device)?,
            Tensor::zeros((0, 2), DType::F64, &device)?,
        )?;
        assert!(
            complex_smooth_l1_loss(&pred_complex, &target_complex, 1.0, Reduction::Mean).is_err()
        );

        Ok(())
    }

    #[test]
    fn test_smooth_l1_loss_gradient_matches_piecewise_formula() -> Result<()> {
        let device = Device::Cpu;
        let pred = Var::from_tensor(&Tensor::from_vec(vec![0.5f64, -2.0], (1, 2), &device)?)?;
        let target = Tensor::zeros((1, 2), DType::F64, &device)?;

        let loss = smooth_l1_loss(pred.as_tensor(), &target, 1.0, Reduction::Mean)?;
        let grads = loss.backward()?;
        let grad = grads
            .get(&pred)
            .cloned()
            .ok_or_else(|| candle_msg("missing gradient for smooth_l1 pred"))?;

        assert_tensor_close(&grad, &[0.25, -0.5], 1e-12)?;

        Ok(())
    }

    #[test]
    fn test_complex_smooth_l1_loss_gradient_matches_piecewise_formula() -> Result<()> {
        let device = Device::Cpu;
        let pred_real = Var::from_tensor(&Tensor::from_vec(vec![0.3f64, 1.2], (1, 2), &device)?)?;
        let pred_imag = Var::from_tensor(&Tensor::from_vec(vec![0.4f64, 1.6], (1, 2), &device)?)?;
        let pred =
            ComplexTensor::new(pred_real.as_tensor().clone(), pred_imag.as_tensor().clone())?;
        let target = ComplexTensor::new(
            Tensor::zeros((1, 2), DType::F64, &device)?,
            Tensor::zeros((1, 2), DType::F64, &device)?,
        )?;

        let loss = complex_smooth_l1_loss(&pred, &target, 1.0, Reduction::Mean)?;
        let grads = loss.backward()?;
        let grad_real = grads
            .get(&pred_real)
            .cloned()
            .ok_or_else(|| candle_msg("missing gradient for complex smooth_l1 real part"))?;
        let grad_imag = grads.get(&pred_imag).cloned().ok_or_else(|| {
            candle_msg("missing gradient for complex smooth_l1 imaginary part")
        })?;

        assert_tensor_close(&grad_real, &[0.15, 0.3], 1e-12)?;
        assert_tensor_close(&grad_imag, &[0.2, 0.4], 1e-12)?;

        Ok(())
    }

    // ── Physics-forward loss ────────────────────────────────────────

    /// WR-90 waveguide used by [`sparam_data::generation`] by default
    /// (`a = 22.86 mm`, `d = 1.5 mm`). Tests build the config here
    /// once and reuse it.
    fn default_waveguide() -> WaveguideConfig {
        WaveguideConfig::new(1.5e-3, 22.86e-3).expect("valid waveguide geometry")
    }

    fn freq_tensor(freqs: &[f64], device: &Device) -> Result<Tensor> {
        Tensor::from_vec(freqs.to_vec(), (freqs.len(),), device)
    }

    /// Build the 2-column S-target tensor the loss now expects —
    /// concatenates NRW-derived S₁₁ and S₂₁ into a single (batch, 2)
    /// `ComplexTensor`. Tests that used to pass `target_eps` now run
    /// this helper first.
    fn s_target_from_eps(
        waveguide: &WaveguideConfig,
        frequencies: &Tensor,
        eps: &ComplexTensor,
    ) -> Result<ComplexTensor> {
        let (s11, s21) = nrw_direct_non_magnetic_with_config(waveguide, frequencies, eps)?;
        let real = Tensor::cat(&[&s11.real, &s21.real], 1)?;
        let imag = Tensor::cat(&[&s11.imag, &s21.imag], 1)?;
        Ok(ComplexTensor::new_unchecked(real, imag))
    }

    #[test]
    fn test_physics_forward_loss_is_zero_when_nrw_matches_s_target() -> Result<()> {
        // With s_target = NRW(pred), both column MSEs collapse to 0.
        let device = Device::Cpu;
        let pred = ComplexTensor::new(
            Tensor::from_vec(vec![10.0f64], (1, 1), &device)?,
            Tensor::from_vec(vec![-1.5f64], (1, 1), &device)?,
        )?;
        let waveguide = default_waveguide();
        let frequencies = freq_tensor(&[8.2e9], &device)?;
        let s_target = s_target_from_eps(&waveguide, &frequencies, &pred)?;
        let loss = physics_forward_loss(
            &pred,
            &s_target,
            &waveguide,
            &frequencies,
            Reduction::Mean,
        )?;
        assert_close(loss.to_scalar::<f64>()?, 0.0, 1e-18);
        Ok(())
    }

    #[test]
    fn test_physics_forward_loss_is_positive_when_pred_differs() -> Result<()> {
        // A small bias in ε̂ must produce a strictly positive MSE in
        // S-parameter space — this is the whole point of the loss.
        let device = Device::Cpu;
        let pred = ComplexTensor::new(
            Tensor::from_vec(vec![12.0f64], (1, 1), &device)?,
            Tensor::from_vec(vec![-1.8f64], (1, 1), &device)?,
        )?;
        let target = ComplexTensor::new(
            Tensor::from_vec(vec![10.0f64], (1, 1), &device)?,
            Tensor::from_vec(vec![-1.5f64], (1, 1), &device)?,
        )?;
        let waveguide = default_waveguide();
        let frequencies = freq_tensor(&[8.2e9], &device)?;
        let s_target = s_target_from_eps(&waveguide, &frequencies, &target)?;
        let loss = physics_forward_loss(
            &pred,
            &s_target,
            &waveguide,
            &frequencies,
            Reduction::Mean,
        )?
        .to_scalar::<f64>()?;
        assert!(loss > 0.0, "expected positive loss, got {loss}");
        assert!(loss.is_finite(), "expected finite loss, got {loss}");
        Ok(())
    }

    #[test]
    fn test_physics_forward_loss_equals_manual_s_parameter_mse() -> Result<()> {
        // Hand-verify by running NRW separately and computing the
        // componentwise MSE — the loss must match exactly.
        let device = Device::Cpu;
        let pred = ComplexTensor::new(
            Tensor::from_vec(vec![12.0f64], (1, 1), &device)?,
            Tensor::from_vec(vec![-1.8f64], (1, 1), &device)?,
        )?;
        let target = ComplexTensor::new(
            Tensor::from_vec(vec![10.0f64], (1, 1), &device)?,
            Tensor::from_vec(vec![-1.5f64], (1, 1), &device)?,
        )?;
        let waveguide = default_waveguide();
        let frequencies = freq_tensor(&[8.2e9], &device)?;
        let (s11_pred, s21_pred) =
            nrw_direct_non_magnetic_with_config(&waveguide, &frequencies, &pred)?;
        let (s11_target, s21_target) =
            nrw_direct_non_magnetic_with_config(&waveguide, &frequencies, &target)?;
        let expected = {
            let a = complex_mse_loss(&s11_pred, &s11_target, Reduction::Mean)?;
            let b = complex_mse_loss(&s21_pred, &s21_target, Reduction::Mean)?;
            a.broadcast_add(&b)?.to_scalar::<f64>()?
        };
        let s_target = s_target_from_eps(&waveguide, &frequencies, &target)?;
        let loss = physics_forward_loss(
            &pred,
            &s_target,
            &waveguide,
            &frequencies,
            Reduction::Mean,
        )?
        .to_scalar::<f64>()?;
        assert_close(loss, expected, 1e-18);
        Ok(())
    }

    #[test]
    fn test_physics_forward_loss_rejects_shape_mismatch() {
        // Batch-size mismatch between pred_eps and s_target must be
        // rejected before any NRW evaluation.
        let device = Device::Cpu;
        let pred = ComplexTensor::new(
            Tensor::zeros((1, 1), DType::F64, &device).unwrap(),
            Tensor::zeros((1, 1), DType::F64, &device).unwrap(),
        )
        .unwrap();
        let s_target = ComplexTensor::new(
            Tensor::zeros((2, 2), DType::F64, &device).unwrap(),
            Tensor::zeros((2, 2), DType::F64, &device).unwrap(),
        )
        .unwrap();
        let frequencies = freq_tensor(&[8.2e9], &device).unwrap();
        let err = physics_forward_loss(
            &pred,
            &s_target,
            &default_waveguide(),
            &frequencies,
            Reduction::Mean,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("physics_forward_loss"));
    }

    #[test]
    fn test_physics_forward_loss_rejects_wrong_s_target_columns() {
        // s_target must have exactly 2 complex columns (S₁₁, S₂₁).
        let device = Device::Cpu;
        let pred = ComplexTensor::new(
            Tensor::zeros((1, 1), DType::F64, &device).unwrap(),
            Tensor::zeros((1, 1), DType::F64, &device).unwrap(),
        )
        .unwrap();
        let s_target = ComplexTensor::new(
            Tensor::zeros((1, 3), DType::F64, &device).unwrap(),
            Tensor::zeros((1, 3), DType::F64, &device).unwrap(),
        )
        .unwrap();
        let frequencies = freq_tensor(&[8.2e9], &device).unwrap();
        let err = physics_forward_loss(
            &pred,
            &s_target,
            &default_waveguide(),
            &frequencies,
            Reduction::Mean,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("2 complex columns"));
    }

    #[test]
    fn test_physics_forward_loss_gradient_is_finite() -> Result<()> {
        // The NRW graph is a differentiable composition of exp/sqrt/
        // atan2 — end-to-end grads must stay finite on a typical
        // (ε̂, ε) pair close to the physical regime.
        let device = Device::Cpu;
        let pred_r = Var::from_tensor(&Tensor::from_vec(vec![11.0f64], (1, 1), &device)?)?;
        let pred_i = Var::from_tensor(&Tensor::from_vec(vec![-1.2f64], (1, 1), &device)?)?;
        let pred = ComplexTensor::new(pred_r.as_tensor().clone(), pred_i.as_tensor().clone())?;
        let target = ComplexTensor::new(
            Tensor::from_vec(vec![10.0f64], (1, 1), &device)?,
            Tensor::from_vec(vec![-1.0f64], (1, 1), &device)?,
        )?;
        let waveguide = default_waveguide();
        let frequencies = freq_tensor(&[8.2e9], &device)?;
        let s_target = s_target_from_eps(&waveguide, &frequencies, &target)?;
        let loss = physics_forward_loss(
            &pred,
            &s_target,
            &waveguide,
            &frequencies,
            Reduction::Mean,
        )?;
        let grads = loss.backward()?;
        let gr = grads.get(&pred_r).cloned().unwrap();
        let gi = grads.get(&pred_i).cloned().unwrap();
        for v in gr.flatten_all()?.to_vec1::<f64>()? {
            assert!(v.is_finite(), "real grad must be finite, got {v}");
        }
        for v in gi.flatten_all()?.to_vec1::<f64>()? {
            assert!(v.is_finite(), "imag grad must be finite, got {v}");
        }
        Ok(())
    }

    #[test]
    fn test_mlp_loss_from_name_accepts_physics_forward() -> Result<()> {
        assert!(matches!(
            MlpLoss::from_name("physics_forward")?,
            MlpLoss::PhysicsForward
        ));
        // Legacy alias retained so DB-persisted `pi_mape` configs
        // still deserialize (they now map to PhysicsForward).
        assert!(matches!(
            MlpLoss::from_name("pi_mape")?,
            MlpLoss::PhysicsForward
        ));
        assert!(MlpLoss::PhysicsForward.is_complex_only());
        assert!(MlpLoss::PhysicsForward.needs_unscaled_targets());
        assert!(!MlpLoss::Mse.is_complex_only());
        assert!(!MlpLoss::Mse.needs_unscaled_targets());
        Ok(())
    }
}

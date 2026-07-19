//! Architectural softplus clamp on the model output, applied unconditionally
//! at every call site that materialises a permittivity prediction.
//!
//! The model produces a 2-column tensor in **scaled** target space; this
//! helper inverse-scales it, applies a `softplus` so the result lands in
//! the physical regime (`Re(ε) ≥ 1`, `ε″ ≥ 0` for both Complex and Real conventions), then re-scales back.
//! Doing the clamp in physical space is what makes the constraint
//! meaningful — softplus on the scaled tensor would clamp at the dataset
//! mean, not at the vacuum boundary.
//!
//! Why architectural rather than a post-hoc filter:
//!
//! * Gradients flow through the softplus, so the model can never produce
//!   physically-invalid outputs, and the optimiser doesn't fight a
//!   constraint surface during training.
//! * Replaces the eval-only clamp that used to leak training-time
//!   gradients past the constraint into a regime the loss couldn't
//!   evaluate (`ε ≤ 0` blowing up the rel-err denominator).
//! * Pairs with the stable-sqrt NRW path (see
//!   [`sparam_physics::nrw::nrw_direct_non_magnetic_with_config`]):
//!   stable sqrt fixes the backward singularity at `ε ≈ 1 + 0j`, the
//!   softplus clamp prevents the forward pass from ever reaching
//!   `ε ≤ 0` where the inverse problem is undefined.

use candle_core::{Result, Tensor};

use crate::scaling::ScalerRef;

/// Numerically-stable `softplus(x) = ln(1 + e^x)`.
///
/// Computed as `max(x, 0) + ln(1 + e^{-|x|})` to avoid overflow when
/// `x ≫ 0` and underflow when `x ≪ 0`.
fn softplus(x: &Tensor) -> Result<Tensor> {
    let zero = x.zeros_like()?;
    let pos = x.maximum(&zero)?;
    let neg_abs = x.abs()?.neg()?;
    let log1p = neg_abs.exp()?.affine(1.0, 1.0)?.log()?;
    pos + log1p
}

/// Apply the architectural softplus clamp to a 2-column scaled
/// prediction tensor.
///
/// Layout: `pred_scaled[:, 0]` is the `ε′` channel, `pred_scaled[:, 1]`
/// is the `ε″` channel. Both Real and Complex models now use the same
/// convention (ε″ ≥ 0) after the sign convention fix — the `is_complex`
/// parameter is kept for backward compatibility but both branches apply
/// `softplus` to ensure non-negativity.
///
/// Output is in scaled space, ready to feed straight into the loss or
/// the metric helpers without re-scaling.
pub fn apply_physical_softplus_clamp(
    pred_scaled: &Tensor,
    target_scaler: ScalerRef<'_>,
    is_complex: bool,
) -> Result<Tensor> {
    // Methodology-study escape hatch: setting
    // `SPARAM_DISABLE_SOFTPLUS_CLAMP=1` returns the prediction
    // unchanged so a paired (clamp / no-clamp) comparison can be
    // made.  Production paths leave this unset; only the
    // `m04_clamp` binary toggles it.
    if std::env::var("SPARAM_DISABLE_SOFTPLUS_CLAMP")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return Ok(pred_scaled.clone());
    }
    let pred_physical = target_scaler.inverse_transform(pred_scaled)?;
    let raw_eps_prime = pred_physical.narrow(1, 0, 1)?;
    let raw_eps_dprime = pred_physical.narrow(1, 1, 1)?;

    // ε′ ≥ 1 always (Re of ε is at least the vacuum value).
    // Use identity for ε' ≥ 1, clamp to 1 for ε' < 1
    // Using a large upper bound to effectively clamp only the minimum
    let eps_prime = raw_eps_prime.clamp(1.0f64, 1e10f64)?;

    let eps_dprime = if is_complex {
        // Complex MLP target convention: ε'' ≥ 0 (same as Real).
        // softplus ensures the output is non-negative.
        softplus(&raw_eps_dprime)?
    } else {
        // Real MLP target convention: |ε″| ≥ 0 (loss tangent magnitude).
        softplus(&raw_eps_dprime)?
    };

    let constrained_physical = Tensor::cat(&[&eps_prime, &eps_dprime], 1)?;
    target_scaler.transform(&constrained_physical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scaling::{Scaler, StandardScaler};
    use candle_core::{DType, Device};

    fn fitted_scaler() -> StandardScaler {
        let device = Device::Cpu;
        // Simulate a target distribution: ε′ ∈ [5, 50], ε″ ∈ [0, 10] (Real).
        let stats = Tensor::new(&[[5.0_f64, 0.0], [50.0, 10.0]], &device)
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap();
        let mut scaler = StandardScaler::new();
        scaler.fit(&stats).unwrap();
        scaler
    }

    #[test]
    fn real_clamp_keeps_eps_prime_above_one() {
        // Even a wildly-negative raw prediction should map to ε′ ≥ 1.
        let scaler = fitted_scaler();
        let pred = Tensor::new(&[[-10.0_f64, -10.0], [-100.0, -100.0]], &Device::Cpu)
            .unwrap();
        let constrained = apply_physical_softplus_clamp(
            &pred,
            ScalerRef::Standard(&scaler),
            /*is_complex=*/ false,
        )
        .unwrap();
        let physical = scaler.inverse_transform(&constrained).unwrap();
        let v = physical.to_vec2::<f64>().unwrap();
        for row in &v {
            assert!(row[0] >= 1.0, "ε′ {} < 1", row[0]);
            assert!(row[1] >= 0.0, "ε″ {} < 0 (Real)", row[1]);
        }
    }

    #[test]
    fn complex_clamp_keeps_eps_dprime_non_negative() {
        let scaler = fitted_scaler();
        // For Complex the column-1 convention is ε'' ≥ 0 (same as Real, after sign fix).
        let pred = Tensor::new(&[[10.0_f64, -10.0], [-10.0, -10.0]], &Device::Cpu)
            .unwrap();
        let constrained = apply_physical_softplus_clamp(
            &pred,
            ScalerRef::Standard(&scaler),
            /*is_complex=*/ true,
        )
        .unwrap();
        let physical = scaler.inverse_transform(&constrained).unwrap();
        let v = physical.to_vec2::<f64>().unwrap();
        for row in &v {
            assert!(row[0] >= 1.0, "ε′ {} < 1", row[0]);
            assert!(row[1] >= 0.0, "ε'' {} < 0 (Complex)", row[1]);
        }
    }
}

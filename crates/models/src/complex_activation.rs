//! Complex-valued activation functions for CVNN layers.
//!
//! All activations preserve phase semantics where applicable and support
//! F64 gradients. Learnable variants own one or more parameter tensors;
//! stateless variants are zero-sized enum instances.

use std::f64::consts::FRAC_1_SQRT_2;
use std::fmt;
use std::str::FromStr;

use candle_core::{Result, Tensor};
use candle_nn::{Init, VarBuilder};
use serde::{Deserialize, Serialize};

use sparam_core::complex_tensor::{COMPLEX_STABILITY_EPS, ComplexTensor};
use sparam_core::config::normalize_config_name;
use sparam_core::error::candle_msg;

pub const DEFAULT_MOD_RELU_BIAS: f64 = -0.1;

// ── Unified complex-activation choice ────────────────────────────────────

/// Complete set of complex activations available in the HPO search
/// space, persisted configs, and the Complex MLP.
///
/// **Stateless** variants apply a fixed function with no learnable
/// parameters:
/// - `CReLU`: component-wise ReLU (simple, well-tested baseline).
/// - `CGELU`: polar GELU — `z · Φ(|z|)`, magnitude-gated and phase-preserving.
/// - `Cardioid`: `z · ½(1 + cos(arg z))` — phase-direction gate.
///
/// **Learnable** variants own one or more parameter tensors; picking
/// one in a config just selects the layer type, the actual parameters
/// are materialised at build time by the MLP via `ActivationLayer::new`:
/// - `ModReLU`: per-feature magnitude bias (H params).
/// - `HybridCardioidGelu`: one learnable λ scalar.
/// - `Lpma`: two per-feature biases (magnitude + phase), 2H params.
/// - `CSwishPhase`: one learnable α scalar.
/// - `Worelu`: phase-windowed ReLU with 3 learnable scalars (b, θ, k).
///
/// Serializes as a flat short name (`"crelu"`, `"modrelu"`, …) via the
/// per-variant `#[serde(rename = …)]`, so persisted JSON stays readable
/// and the HPO string → typed enum conversion is a one-to-one mapping.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ComplexActivation {
    // Stateless
    #[serde(rename = "crelu")]
    CReLU,
    #[serde(rename = "cgelu")]
    CGELU,
    #[serde(rename = "cardioid")]
    Cardioid,
    // Learnable
    #[serde(rename = "modrelu", alias = "mod_relu")]
    ModReLU,
    #[serde(rename = "hybrid_cardioid_gelu", alias = "cardioid_gelu")]
    HybridCardioidGelu,
    #[serde(rename = "lpma")]
    Lpma,
    #[serde(rename = "cswish_phase", alias = "cswishphase")]
    CSwishPhase,
    #[serde(rename = "worelu")]
    Worelu,
}

impl Default for ComplexActivation {
    fn default() -> Self {
        Self::CReLU
    }
}

impl fmt::Display for ComplexActivation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl ComplexActivation {
    /// Canonical short name used as the round-trip key in HPO search
    /// spaces and persisted configs.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::CReLU => "crelu",
            Self::CGELU => "cgelu",
            Self::Cardioid => "cardioid",
            Self::ModReLU => "modrelu",
            Self::HybridCardioidGelu => "hybrid_cardioid_gelu",
            Self::Lpma => "lpma",
            Self::CSwishPhase => "cswish_phase",
            Self::Worelu => "worelu",
        }
    }

    /// `true` for parameter-free activations (`CReLU`, `CGELU`,
    /// `Cardioid`); `false` for learnable ones that need `VarBuilder`
    /// allocation through `ActivationLayer::new`.
    #[must_use]
    pub fn is_stateless(&self) -> bool {
        matches!(self, Self::CReLU | Self::CGELU | Self::Cardioid)
    }

    /// Parse a complex activation from any of its accepted input
    /// names (canonical or alias — e.g. `"polar_gelu"` → `CGELU`,
    /// `"mod_relu"` → `ModReLU`). Covers all 8 variants.
    pub fn from_name(name: &str) -> Result<Self> {
        let normalized = normalize_config_name(name);
        match normalized.as_str() {
            "crelu" | "c_relu" | "complex_relu" => Ok(Self::CReLU),
            "cgelu" | "c_gelu" | "complex_gelu" | "polar_gelu" => Ok(Self::CGELU),
            "cardioid" => Ok(Self::Cardioid),
            "modrelu" | "mod_relu" => Ok(Self::ModReLU),
            "hybrid_cardioid_gelu" | "cardioid_gelu" => Ok(Self::HybridCardioidGelu),
            "lpma" => Ok(Self::Lpma),
            "cswish_phase" | "cswishphase" => Ok(Self::CSwishPhase),
            "worelu" => Ok(Self::Worelu),
            _ => Err(candle_msg(format!(
                "unknown complex activation: {name}"
            ))),
        }
    }

    /// Apply a **stateless** activation element-wise. Returns `Err`
    /// for learnable variants — they own tensor parameters and must
    /// be materialised as an `ActivationLayer` first.
    pub fn forward(&self, xs: &ComplexTensor) -> Result<ComplexTensor> {
        match self {
            Self::CReLU => Ok(ComplexTensor::new_unchecked(
                xs.real.relu()?,
                xs.imag.relu()?,
            )),
            Self::CGELU => {
                let gate = polar_gelu_gate(&xs.mag()?)?;
                xs.scale(&gate)
            }
            Self::Cardioid => apply_cardioid(xs),
            learnable => Err(candle_msg(format!(
                "{} is learnable; build an ActivationLayer via `ActivationLayer::new` instead",
                learnable.name()
            ))),
        }
    }
}

impl FromStr for ComplexActivation {
    type Err = candle_core::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_name(s)
    }
}

/// Factory helper retained for API compatibility with the previous
/// stateless-only split. New code should call `ComplexActivation::from_name`
/// directly.
pub fn make_complex_activation(name: &str) -> Result<ComplexActivation> {
    ComplexActivation::from_name(name)
}

/// Polar-GELU gate: `Φ(|z|) = ½(1 + erf(|z| / √2))`.
fn polar_gelu_gate(mag: &Tensor) -> Result<Tensor> {
    mag.affine(FRAC_1_SQRT_2, 0.0)?
        .erf()?
        .affine(0.5, 0.5)
}

/// Cardioid: `z · ½(1 + Re(z)/|z|)` with zero-guarded denominator.
fn apply_cardioid(xs: &ComplexTensor) -> Result<ComplexTensor> {
    let mag = xs.mag()?;
    let non_zero = mag.ge(COMPLEX_STABILITY_EPS)?;
    let safe_mag = non_zero.where_cond(&mag, &mag.ones_like()?)?;
    let gate = xs.real.broadcast_div(&safe_mag)?.affine(0.5, 0.5)?;
    let gate = non_zero.where_cond(&gate, &gate.zeros_like()?)?;
    xs.scale(&gate)
}

// ── ModReLU (legacy learnable, kept) ─────────────────────────────────────

/// Magnitude-gated complex ReLU with a learnable real bias per feature:
/// `modReLU(z) = ReLU(|z| + b) · z / |z|` (0 when |z|=0).
#[derive(Clone, Debug)]
pub struct ModReLU {
    bias: Tensor,
    features: usize,
}

impl ModReLU {
    pub fn new(features: usize, vb: VarBuilder) -> Result<Self> {
        if features == 0 {
            return Err(candle_msg("modrelu requires features > 0"));
        }
        let bias = vb.get_with_hints(features, "bias", Init::Const(DEFAULT_MOD_RELU_BIAS))?;
        Self::from_bias(bias)
    }

    pub fn from_bias(bias: Tensor) -> Result<Self> {
        let features = bias
            .dims1()
            .map_err(|_| candle_msg("modrelu bias must be rank-1"))?;
        if features == 0 {
            return Err(candle_msg("modrelu bias must be non-empty"));
        }
        Ok(Self { bias, features })
    }

    pub fn forward(&self, xs: &ComplexTensor) -> Result<ComplexTensor> {
        self.validate_last_dim(xs.real.dims(), "modrelu")?;
        let mag = xs.mag()?;
        let gated_mag = mag.broadcast_add(&self.bias)?.relu()?;
        let non_zero = mag.ge(COMPLEX_STABILITY_EPS)?;
        let safe_mag = non_zero.where_cond(&mag, &mag.ones_like()?)?;
        let scale = gated_mag.broadcast_div(&safe_mag)?;
        let scale = non_zero.where_cond(&scale, &scale.zeros_like()?)?;
        xs.scale(&scale)
    }

    #[cfg(any(test, doctest))]
    #[must_use]
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    #[cfg(any(test, doctest))]
    #[must_use]
    pub fn features(&self) -> usize {
        self.features
    }

    #[cfg(any(test, doctest))]
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.features
    }

    fn validate_last_dim(&self, dims: &[usize], name: &str) -> Result<()> {
        let Some(last) = dims.last().copied() else {
            return Err(candle_msg(format!(
                "{name} expects at least one input dim"
            )));
        };
        if last != self.features {
            return Err(candle_msg(format!(
                "{name} expected last dim {} but got {}",
                self.features, last
            )));
        }
        Ok(())
    }
}

impl fmt::Display for ModReLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "modrelu(features={})", self.features)
    }
}

// ── HybridCardioidGelu (top recommendation) ──────────────────────────────

/// Learnable blend of cardioid (phase gating) and polar GELU (magnitude
/// gating): `z · [λ · ½(1 + cos(arg z)) + (1−λ) · Φ(|z|)]`. One learnable
/// scalar `λ` per layer, initialised at 0.5.
#[derive(Clone, Debug)]
pub struct HybridCardioidGelu {
    lambda: Tensor,
}

impl HybridCardioidGelu {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let lambda = vb.get_with_hints(1, "lambda", Init::Const(0.5))?;
        Ok(Self { lambda })
    }

    pub fn forward(&self, xs: &ComplexTensor) -> Result<ComplexTensor> {
        let mag = xs.mag()?;
        let non_zero = mag.ge(COMPLEX_STABILITY_EPS)?;
        let safe_mag = non_zero.where_cond(&mag, &mag.ones_like()?)?;
        let cardioid_gate = xs.real.broadcast_div(&safe_mag)?.affine(0.5, 0.5)?;
        let cardioid_gate = non_zero.where_cond(&cardioid_gate, &cardioid_gate.zeros_like()?)?;
        let gelu_gate = polar_gelu_gate(&mag)?;
        let one_minus_lambda = self.lambda.affine(-1.0, 1.0)?;
        let blended = cardioid_gate
            .broadcast_mul(&self.lambda)?
            .broadcast_add(&gelu_gate.broadcast_mul(&one_minus_lambda)?)?;
        xs.scale(&blended)
    }

    #[cfg(any(test, doctest))]
    #[must_use]
    pub fn lambda(&self) -> &Tensor {
        &self.lambda
    }

    #[cfg(any(test, doctest))]
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        1
    }
}

impl fmt::Display for HybridCardioidGelu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "hybrid_cardioid_gelu(λ=learnable)")
    }
}

// ── LPMA (Learnable Phase-Magnitude Activation) ──────────────────────────

/// Learnable magnitude-and-phase gate: scales by `σ(|z| + b_m)` and
/// rotates by `φ = b_p · tanh(|z|)`. Two per-feature biases `b_m`, `b_p`.
/// Avoids `phase()` by applying the rotation directly as `z · e^(iφ)` —
/// the desired phase shift lands without ever extracting `arg z`.
#[derive(Clone, Debug)]
pub struct Lpma {
    b_mag: Tensor,
    b_phase: Tensor,
    features: usize,
}

impl Lpma {
    pub fn new(features: usize, vb: VarBuilder) -> Result<Self> {
        if features == 0 {
            return Err(candle_msg("lpma requires features > 0"));
        }
        let b_mag = vb.get_with_hints(features, "b_mag", Init::Const(DEFAULT_MOD_RELU_BIAS))?;
        let b_phase = vb.get_with_hints(features, "b_phase", Init::Const(0.0))?;
        Ok(Self { b_mag, b_phase, features })
    }

    pub fn forward(&self, xs: &ComplexTensor) -> Result<ComplexTensor> {
        self.validate_last_dim(xs.real.dims(), "lpma")?;
        let mag = xs.mag()?;
        let mag_scale = candle_nn::ops::sigmoid(&mag.broadcast_add(&self.b_mag)?)?;
        // phase rotation φ = b_phase · tanh(|z|)
        let phi = mag.tanh()?.broadcast_mul(&self.b_phase)?;
        let cos_phi = phi.cos()?;
        let sin_phi = phi.sin()?;
        let new_re = xs
            .real
            .mul(&cos_phi)?
            .sub(&xs.imag.mul(&sin_phi)?)?
            .mul(&mag_scale)?;
        let new_im = xs
            .real
            .mul(&sin_phi)?
            .add(&xs.imag.mul(&cos_phi)?)?
            .mul(&mag_scale)?;
        ComplexTensor::new(new_re, new_im)
    }

    #[cfg(any(test, doctest))]
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.features * 2
    }

    fn validate_last_dim(&self, dims: &[usize], name: &str) -> Result<()> {
        let Some(last) = dims.last().copied() else {
            return Err(candle_msg(format!(
                "{name} expects at least one input dim"
            )));
        };
        if last != self.features {
            return Err(candle_msg(format!(
                "{name} expected last dim {} but got {}",
                self.features, last
            )));
        }
        Ok(())
    }
}

impl fmt::Display for Lpma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lpma(features={})", self.features)
    }
}

// ── CSwishPhase (phase-corrected complex SiLU) ───────────────────────────

/// Phase-preserving Swish with an optional learnable phase rotation:
/// `z · σ(|z|) · exp(i · α · sin(2·arg(z)))`. At α=0 reduces to the plain
/// magnitude-gated SiLU. Uses the identity `sin(2·arg z) = 2·Re·Im / |z|²`
/// instead of `phase()` — exact, cheaper, and avoids the 0.22° approximation
/// error of the polynomial `atan2`.
#[derive(Clone, Debug)]
pub struct CSwishPhase {
    alpha: Tensor,
}

impl CSwishPhase {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get_with_hints(1, "alpha", Init::Const(0.0))?;
        Ok(Self { alpha })
    }

    pub fn forward(&self, xs: &ComplexTensor) -> Result<ComplexTensor> {
        let mag = xs.mag()?;
        let mag_scale = candle_nn::ops::sigmoid(&mag)?;
        // sin(2·arg(z)) = 2·Re·Im / |z|²  (no atan2 needed).
        // Uses the same `mag_sq + ε` floor that `mag()` uses under the sqrt
        // so the division is well-defined at z = 0 (gives 0/ε = 0, with
        // finite backward). The previous `where_cond(non_zero, …)` mask
        // was dead code after the `mag()` floor pushed `mag` above the
        // threshold — removing it also avoids the `0/0 = NaN` the active
        // branch was producing before the mask could select the zero path.
        let mag_sq_safe = (xs.mag_sq()? + COMPLEX_STABILITY_EPS)?;
        let sin_2arg = xs
            .real
            .mul(&xs.imag)?
            .affine(2.0, 0.0)?
            .div(&mag_sq_safe)?;
        let phi = sin_2arg.broadcast_mul(&self.alpha)?;
        let cos_phi = phi.cos()?;
        let sin_phi = phi.sin()?;
        let new_re = xs
            .real
            .mul(&cos_phi)?
            .sub(&xs.imag.mul(&sin_phi)?)?
            .mul(&mag_scale)?;
        let new_im = xs
            .real
            .mul(&sin_phi)?
            .add(&xs.imag.mul(&cos_phi)?)?
            .mul(&mag_scale)?;
        ComplexTensor::new(new_re, new_im)
    }

    #[cfg(any(test, doctest))]
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        1
    }
}

impl fmt::Display for CSwishPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cswish_phase(α=learnable)")
    }
}

// ── WoReLU (Wirtinger-Optimal ReLU, smooth form) ─────────────────────────

/// Smooth phase-windowed ReLU:
/// `f(z) = z · σ(k·(|z| − b)) · σ(k·(θ − |arg z|))`.
///
/// Phase extraction reuses the existing fast, differentiable
/// [`fast_complex_phase`](sparam_core::complex_tensor::fast_complex_phase)
/// — a polynomial `atan2` (~0.22° error) that stays on-device — rather
/// than reimplementing it. Three scalar learnable parameters per layer.
#[derive(Clone, Debug)]
pub struct Worelu {
    b: Tensor,     // magnitude threshold
    theta: Tensor, // phase-window half-width (radians)
    k: Tensor,     // sigmoid sharpness
}

impl Worelu {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let b = vb.get_with_hints(1, "b", Init::Const(0.0))?;
        // π/2 lets the network default to "keep right half-plane" = a soft
        // cardioid-like gate.
        let theta = vb.get_with_hints(1, "theta", Init::Const(std::f64::consts::FRAC_PI_2))?;
        let k = vb.get_with_hints(1, "k", Init::Const(1.0))?;
        Ok(Self { b, theta, k })
    }

    pub fn forward(&self, xs: &ComplexTensor) -> Result<ComplexTensor> {
        let mag = xs.mag()?;
        let abs_phase = xs.phase()?.abs()?;
        let mag_gate =
            candle_nn::ops::sigmoid(&mag.broadcast_sub(&self.b)?.broadcast_mul(&self.k)?)?;
        let phase_gate = candle_nn::ops::sigmoid(
            &self
                .theta
                .broadcast_sub(&abs_phase)?
                .broadcast_mul(&self.k)?,
        )?;
        xs.scale(&mag_gate.mul(&phase_gate)?)
    }

    #[cfg(any(test, doctest))]
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        3
    }
}

impl fmt::Display for Worelu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "worelu(b,θ,k=learnable)")
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Var};
    use candle_nn::{VarBuilder, VarMap};

    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_tensor_close(values: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(values.len(), expected.len());
        for (a, e) in values.iter().zip(expected.iter()) {
            assert_close(*a, *e, tol);
        }
    }

    fn vb_f64() -> (VarMap, Device) {
        (VarMap::new(), Device::Cpu)
    }

    #[test]
    fn crelu_applies_relu_componentwise() -> Result<()> {
        let xs = ComplexTensor::new(
            Tensor::from_vec(vec![-1.0f64, 0.5, 1.0, 0.0], 4, &Device::Cpu)?,
            Tensor::from_vec(vec![2.0f64, -0.3, 0.0, -1.0], 4, &Device::Cpu)?,
        )?;
        let ys = ComplexActivation::CReLU.forward(&xs)?;
        assert_eq!(ys.real.to_vec1::<f64>()?, vec![0.0, 0.5, 1.0, 0.0]);
        assert_eq!(ys.imag.to_vec1::<f64>()?, vec![2.0, 0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn polar_cgelu_preserves_phase() -> Result<()> {
        // For any z≠0, arg(Φ(|z|)·z) == arg(z) because Φ(|z|) ∈ (0,1).
        let xs = ComplexTensor::new(
            Tensor::from_vec(vec![3.0f64, -2.0], 2, &Device::Cpu)?,
            Tensor::from_vec(vec![4.0f64, 1.0], 2, &Device::Cpu)?,
        )?;
        let ys = ComplexActivation::CGELU.forward(&xs)?;
        let re = ys.real.to_vec1::<f64>()?;
        let im = ys.imag.to_vec1::<f64>()?;
        // arg ratio im/re should equal input's im/re for each element.
        assert_close(re[0] / 3.0, im[0] / 4.0, 1e-12);
        assert_close(re[1] / -2.0, im[1] / 1.0, 1e-12);
        Ok(())
    }

    #[test]
    fn cardioid_zeros_negative_real_axis_and_scales_forward() -> Result<()> {
        // Positive real axis: gate = 1.0 (full pass-through).
        let xs = ComplexTensor::new(
            Tensor::from_vec(vec![1.0f64], 1, &Device::Cpu)?,
            Tensor::from_vec(vec![0.0f64], 1, &Device::Cpu)?,
        )?;
        let ys = ComplexActivation::Cardioid.forward(&xs)?;
        assert_close(ys.real.to_vec1::<f64>()?[0], 1.0, 1e-12);

        // Negative real axis: gate = 0.
        let xs_neg = ComplexTensor::new(
            Tensor::from_vec(vec![-1.0f64], 1, &Device::Cpu)?,
            Tensor::from_vec(vec![0.0f64], 1, &Device::Cpu)?,
        )?;
        let ys_neg = ComplexActivation::Cardioid.forward(&xs_neg)?;
        assert_close(ys_neg.real.to_vec1::<f64>()?[0], 0.0, 1e-12);
        Ok(())
    }

    #[test]
    fn make_complex_activation_accepts_all_names_and_aliases() -> Result<()> {
        // Stateless variants.
        assert_eq!(make_complex_activation("crelu")?, ComplexActivation::CReLU);
        assert_eq!(make_complex_activation("cgelu")?, ComplexActivation::CGELU);
        assert_eq!(make_complex_activation("polar_gelu")?, ComplexActivation::CGELU);
        assert_eq!(make_complex_activation("cardioid")?, ComplexActivation::Cardioid);
        // Learnable variants now parse in the unified enum (they
        // previously errored under the 2-enum split).
        assert_eq!(make_complex_activation("modrelu")?, ComplexActivation::ModReLU);
        assert_eq!(
            make_complex_activation("hybrid_cardioid_gelu")?,
            ComplexActivation::HybridCardioidGelu
        );
        assert_eq!(make_complex_activation("lpma")?, ComplexActivation::Lpma);
        assert_eq!(
            make_complex_activation("cswish_phase")?,
            ComplexActivation::CSwishPhase
        );
        assert_eq!(make_complex_activation("worelu")?, ComplexActivation::Worelu);
        // Unknown names still fail.
        assert!(make_complex_activation("unknown").is_err());
        Ok(())
    }

    /// Every variant of the unified [`ComplexActivation`] enum (3
    /// stateless + 5 learnable = 8 total) must produce a non-empty
    /// short name whose `from_name` parses back to the same variant.
    /// Adding a new variant forces the match below to be extended
    /// (via exhaustiveness), which in turn forces `all_variants` to
    /// list the new name — catches silent drift between `name()`,
    /// `from_name`, and the HPO search-space string lists.
    #[test]
    fn every_complex_activation_variant_has_round_trip_name() {
        let all_variants = [
            ComplexActivation::CReLU,
            ComplexActivation::CGELU,
            ComplexActivation::Cardioid,
            ComplexActivation::ModReLU,
            ComplexActivation::HybridCardioidGelu,
            ComplexActivation::Lpma,
            ComplexActivation::CSwishPhase,
            ComplexActivation::Worelu,
        ];
        for variant in all_variants {
            let name = variant.name();
            assert!(!name.is_empty(), "empty name for {variant:?}");
            let reparsed = ComplexActivation::from_name(name)
                .unwrap_or_else(|e| panic!("re-parse '{name}' failed: {e}"));
            assert_eq!(
                reparsed, variant,
                "round-trip mismatch: {variant:?} → '{name}' → {reparsed:?}",
            );
        }
        // Exhaustiveness trap: adding a new variant breaks this match
        // at compile time, forcing `all_variants` above to be updated.
        fn _exhaustive_check(a: ComplexActivation) {
            match a {
                ComplexActivation::CReLU
                | ComplexActivation::CGELU
                | ComplexActivation::Cardioid
                | ComplexActivation::ModReLU
                | ComplexActivation::HybridCardioidGelu
                | ComplexActivation::Lpma
                | ComplexActivation::CSwishPhase
                | ComplexActivation::Worelu => {}
            }
        }
    }

    /// `is_stateless()` must be consistent with which variants accept
    /// `forward()` without needing a `VarBuilder`.
    #[test]
    fn is_stateless_matches_forward_capability() {
        for variant in [
            ComplexActivation::CReLU,
            ComplexActivation::CGELU,
            ComplexActivation::Cardioid,
        ] {
            assert!(variant.is_stateless(), "{variant:?} should be stateless");
        }
        for variant in [
            ComplexActivation::ModReLU,
            ComplexActivation::HybridCardioidGelu,
            ComplexActivation::Lpma,
            ComplexActivation::CSwishPhase,
            ComplexActivation::Worelu,
        ] {
            assert!(!variant.is_stateless(), "{variant:?} should be learnable");
            // Calling `forward` on a learnable variant errors out
            // with a hint to use `ActivationLayer::new` instead.
            let xs = ComplexTensor::new(
                Tensor::from_vec(vec![1.0f64], 1, &Device::Cpu).unwrap(),
                Tensor::from_vec(vec![0.0f64], 1, &Device::Cpu).unwrap(),
            )
            .unwrap();
            let err = variant.forward(&xs).expect_err("learnable forward should error");
            assert!(
                err.to_string().contains("learnable"),
                "error should mention 'learnable', got: {err}"
            );
        }
    }

    #[test]
    fn modrelu_new_constructs_and_forwards() -> Result<()> {
        let (varmap, dev) = vb_f64();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &dev);
        let layer = ModReLU::new(3, vb)?;
        assert_eq!(layer.parameter_count(), 3);
        assert_eq!(layer.to_string(), "modrelu(features=3)");
        Ok(())
    }

    #[test]
    fn modrelu_bias_gradients_match_finite_differences() -> Result<()> {
        let bias_values = vec![-0.2f64, 0.1];
        let xs = ComplexTensor::new(
            Tensor::from_vec(vec![3.0f64, 1.0, 2.0, -1.0], (2, 2), &Device::Cpu)?,
            Tensor::from_vec(vec![4.0f64, 2.0, 2.0, 2.0], (2, 2), &Device::Cpu)?,
        )?;
        let var_bias =
            Var::from_tensor(&Tensor::from_vec(bias_values.clone(), 2, &Device::Cpu)?)?;
        let layer = ModReLU::from_bias(var_bias.as_tensor().clone())?;
        let loss = layer.forward(&xs)?.mag_sq()?.sum_all()?;
        let analytical = loss
            .backward()?
            .get(&var_bias)
            .cloned()
            .ok_or_else(|| candle_msg("missing modrelu bias gradient"))?
            .to_vec1::<f64>()?;

        let eps = 1e-6;
        for idx in 0..bias_values.len() {
            let mut pp = bias_values.clone();
            pp[idx] += eps;
            let lp = ModReLU::from_bias(Tensor::from_vec(pp, 2, &Device::Cpu)?)?
                .forward(&xs)?
                .mag_sq()?
                .sum_all()?
                .to_scalar::<f64>()?;
            let mut pm = bias_values.clone();
            pm[idx] -= eps;
            let lm = ModReLU::from_bias(Tensor::from_vec(pm, 2, &Device::Cpu)?)?
                .forward(&xs)?
                .mag_sq()?
                .sum_all()?
                .to_scalar::<f64>()?;
            let numerical = (lp - lm) / (2.0 * eps);
            let rel_err =
                (analytical[idx] - numerical).abs() / analytical[idx].abs().max(numerical.abs());
            assert!(rel_err < 1e-5, "bias[{idx}] grad mismatch: {rel_err:.2e}");
        }
        Ok(())
    }

    #[test]
    fn hybrid_cardioid_gelu_reduces_to_pure_cardioid_when_lambda_is_one() -> Result<()> {
        let tensors: HashMap<String, Tensor> = [(
            "lambda".to_string(),
            Tensor::from_vec(vec![1.0f64], 1, &Device::Cpu)?,
        )]
        .into_iter()
        .collect();
        let vb = VarBuilder::from_tensors(tensors, DType::F64, &Device::Cpu);
        let layer = HybridCardioidGelu::new(vb)?;
        let xs = ComplexTensor::new(
            Tensor::from_vec(vec![1.0f64, -1.0, 0.0], 3, &Device::Cpu)?,
            Tensor::from_vec(vec![0.0f64, 0.0, 2.0], 3, &Device::Cpu)?,
        )?;
        let ys = layer.forward(&xs)?;
        let cardioid_out = ComplexActivation::Cardioid.forward(&xs)?;
        assert_tensor_close(
            &ys.real.to_vec1::<f64>()?,
            &cardioid_out.real.to_vec1::<f64>()?,
            1e-12,
        );
        assert_tensor_close(
            &ys.imag.to_vec1::<f64>()?,
            &cardioid_out.imag.to_vec1::<f64>()?,
            1e-12,
        );
        Ok(())
    }

    #[test]
    fn hybrid_cardioid_gelu_parameters_flow_gradients() -> Result<()> {
        let (varmap, dev) = vb_f64();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &dev);
        let layer = HybridCardioidGelu::new(vb)?;
        let xs = ComplexTensor::new(
            Tensor::randn(0f64, 1.0, (2, 3), &dev)?,
            Tensor::randn(0f64, 1.0, (2, 3), &dev)?,
        )?;
        let loss = layer.forward(&xs)?.mag_sq()?.sum_all()?;
        let grads = loss.backward()?;
        let lambda = varmap
            .data()
            .lock()
            .expect("poisoned")
            .iter()
            .find(|(k, _)| k.ends_with("lambda"))
            .map(|(_, v)| v.clone())
            .expect("lambda var must exist");
        let g = grads
            .get(lambda.as_tensor())
            .expect("lambda must receive a gradient");
        let g_val = g.flatten_all()?.to_vec1::<f64>()?[0];
        assert!(g_val.is_finite());
        Ok(())
    }

    #[test]
    fn lpma_reduces_to_magnitude_gate_when_b_phase_is_zero() -> Result<()> {
        let tensors: HashMap<String, Tensor> = [
            (
                "b_mag".to_string(),
                Tensor::from_vec(vec![0.0f64, 0.0], 2, &Device::Cpu)?,
            ),
            (
                "b_phase".to_string(),
                Tensor::from_vec(vec![0.0f64, 0.0], 2, &Device::Cpu)?,
            ),
        ]
        .into_iter()
        .collect();
        let vb = VarBuilder::from_tensors(tensors, DType::F64, &Device::Cpu);
        let layer = Lpma::new(2, vb)?;
        let xs = ComplexTensor::new(
            Tensor::from_vec(vec![3.0f64, 1.0], (1, 2), &Device::Cpu)?,
            Tensor::from_vec(vec![4.0f64, 2.0], (1, 2), &Device::Cpu)?,
        )?;
        let ys = layer.forward(&xs)?;
        // At b_phase=0, φ=0, so output = z · σ(|z|).
        let mag = xs.mag()?.to_vec2::<f64>()?[0].clone();
        let expected_re = [3.0 * sigmoid(mag[0]), 1.0 * sigmoid(mag[1])];
        let expected_im = [4.0 * sigmoid(mag[0]), 2.0 * sigmoid(mag[1])];
        assert_tensor_close(&ys.real.to_vec2::<f64>()?[0], &expected_re, 1e-12);
        assert_tensor_close(&ys.imag.to_vec2::<f64>()?[0], &expected_im, 1e-12);
        Ok(())
    }

    #[test]
    fn lpma_parameter_count_is_two_per_feature() -> Result<()> {
        let (varmap, dev) = vb_f64();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &dev);
        let layer = Lpma::new(4, vb)?;
        assert_eq!(layer.parameter_count(), 8);
        Ok(())
    }

    #[test]
    fn worelu_uses_fast_phase_and_flows_gradients() -> Result<()> {
        let (varmap, dev) = vb_f64();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &dev);
        let layer = Worelu::new(vb)?;
        assert_eq!(layer.parameter_count(), 3);
        let xs = ComplexTensor::new(
            Tensor::randn(0f64, 1.0, (4, 2), &dev)?,
            Tensor::randn(0f64, 1.0, (4, 2), &dev)?,
        )?;
        let loss = layer.forward(&xs)?.mag_sq()?.sum_all()?;
        let grads = loss.backward()?;
        for name in ["b", "theta", "k"] {
            let v = varmap
                .data()
                .lock()
                .expect("poisoned")
                .iter()
                .find(|(k, _)| k.ends_with(name))
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| panic!("{name} var must exist"));
            let g = grads.get(v.as_tensor()).expect("gradient");
            let g_val = g.flatten_all()?.to_vec1::<f64>()?[0];
            assert!(g_val.is_finite(), "grad[{name}] not finite: {g_val}");
        }
        Ok(())
    }

    #[test]
    fn cswish_phase_forward_and_backward_are_finite_at_zero_input() -> Result<()> {
        // Regression for the 20/21-HPO-failures bug: CSwishPhase.forward
        // computed `2·Re·Im / (Re² + Im²)` without an eps floor, so a
        // zero-magnitude input gave 0/0 = NaN in the forward and
        // NaN gradients in the backward. The previous `where_cond`
        // mask was dead after the `mag()` eps-floor fix and couldn't
        // rescue the NaN. This test pins the behaviour at `z = 0`.
        use candle_core::Var;
        let tensors: HashMap<String, Tensor> = [(
            "alpha".to_string(),
            Tensor::from_vec(vec![0.5f32], 1, &Device::Cpu)?,
        )]
        .into_iter()
        .collect();
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let layer = CSwishPhase::new(vb)?;

        let xv = Var::new(&[0.0f32, 0.0, 0.0, 0.0], &Device::Cpu)?;
        let yv = Var::new(&[0.0f32, 0.0, 0.0, 0.0], &Device::Cpu)?;
        let xs = ComplexTensor::new(xv.as_tensor().clone(), yv.as_tensor().clone())?;

        let ys = layer.forward(&xs)?;
        for v in ys.real.to_vec1::<f32>()?.iter().chain(ys.imag.to_vec1::<f32>()?.iter()) {
            assert!(v.is_finite(), "cswish_phase forward NaN at z=0: {v}");
        }

        let loss = ys.real.sqr()?.sum_all()?.add(&ys.imag.sqr()?.sum_all()?)?;
        let grads = loss.backward()?;
        let gx = grads.get(&xv).unwrap().to_vec1::<f32>()?;
        let gy = grads.get(&yv).unwrap().to_vec1::<f32>()?;
        for g in gx.iter().chain(gy.iter()) {
            assert!(g.is_finite(), "cswish_phase backward NaN at z=0: {g}");
        }
        Ok(())
    }

    #[test]
    fn cswish_phase_equals_mod_silu_when_alpha_is_zero() -> Result<()> {
        let tensors: HashMap<String, Tensor> = [(
            "alpha".to_string(),
            Tensor::from_vec(vec![0.0f64], 1, &Device::Cpu)?,
        )]
        .into_iter()
        .collect();
        let vb = VarBuilder::from_tensors(tensors, DType::F64, &Device::Cpu);
        let layer = CSwishPhase::new(vb)?;
        let xs = ComplexTensor::new(
            Tensor::from_vec(vec![3.0f64, -1.0], (1, 2), &Device::Cpu)?,
            Tensor::from_vec(vec![4.0f64, 2.0], (1, 2), &Device::Cpu)?,
        )?;
        let ys = layer.forward(&xs)?;
        let mag = xs.mag()?.to_vec2::<f64>()?[0].clone();
        let expected_re = [3.0 * sigmoid(mag[0]), -1.0 * sigmoid(mag[1])];
        let expected_im = [4.0 * sigmoid(mag[0]), 2.0 * sigmoid(mag[1])];
        assert_tensor_close(&ys.real.to_vec2::<f64>()?[0], &expected_re, 1e-12);
        assert_tensor_close(&ys.imag.to_vec2::<f64>()?[0], &expected_im, 1e-12);
        Ok(())
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

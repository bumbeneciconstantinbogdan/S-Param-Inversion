//! # NRW (Nicolson-Ross-Weir) Algorithm
//!
//! This module implements the NRW method for computing S-parameters from material
//! properties in a rectangular waveguide operating in TE10 mode.
//!
//! ## Physical Model
//!
//! ```text
//!     +---------------------------------+
//!     |         Waveguide (air)         |
//!     |  Port 1                  Port 2 |
//!     |    <-+----------------+- >      |
//!     |      |  MUT (eps_r,mu_r) |      |
//!     |      |   thickness d     |      |
//!     |      +-------------------+      |
//!     |            width a              |
//!     +---------------------------------+
//! ```
//!
//! ## References
//!
//! - Nicolson, A.M. and Ross, G.F. (1970). Measurement of the intrinsic
//!   properties of materials by time-domain techniques.
//! - Weir, W.B. (1974). Automatic measurement of complex dielectric constant
//!   and permeability at microwave frequencies.

use std::f64::consts::PI;

use candle_core::{DType, Result, Tensor};

use sparam_core::complex::Complex64;
use sparam_core::complex_tensor::ComplexTensor;
use sparam_core::constants::{EPSILON_0, MU_0};

/// Speed of light in vacuum in meters per second.
pub(crate) const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Rectangular waveguide configuration for NRW calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaveguideConfig {
    d: f64,
    a: f64,
}

impl WaveguideConfig {
    /// Creates a validated waveguide configuration.
    pub fn new(d: f64, a: f64) -> Result<Self> {
        validate_positive_parameter("sample thickness d", d)?;
        validate_positive_parameter("waveguide width a", a)?;
        Ok(Self { d, a })
    }

    /// Sample thickness in meters.
    #[must_use]
    pub const fn sample_thickness(self) -> f64 {
        self.d
    }

    /// Waveguide width in meters.
    #[must_use]
    pub const fn width(self) -> f64 {
        self.a
    }

    /// TE10 cutoff frequency in hertz.
    #[must_use]
    pub fn cutoff_frequency(self) -> f64 {
        SPEED_OF_LIGHT / (2.0 * self.a)
    }

    /// Transverse wave number for TE10 mode.
    #[must_use]
    pub fn kt(self) -> f64 {
        PI / self.a
    }
}

use sparam_core::tensor_ops::{as_complex_f64, atan2_tensor, ensure_same_device};

fn validate_positive_parameter(name: &str, value: f64) -> candle_core::Result<()> {
    sparam_core::validation::validate_positive_f64(name, value)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))
}

fn ensure_same_device_sparameters(
    frequencies: &Tensor,
    s11: &ComplexTensor,
    s21: &ComplexTensor,
) -> Result<()> {
    ensure_same_device(frequencies.device(), "frequencies", s11.device(), "S11")?;
    ensure_same_device(s11.device(), "S11", s21.device(), "S21")
}

fn propagation_constant_from_transmission_factor(
    propagation: &ComplexTensor,
    sample_thickness: f64,
) -> Result<ComplexTensor> {
    let d = sample_thickness;

    // Imaginary part: ln|P| / d — fully tensor-based.
    let beta_imag = propagation.mag()?.log()?.affine(1.0 / d, 0.0)?;

    let phase = atan2_tensor(&propagation.imag.neg()?, &propagation.real)?;
    let beta_real = phase.affine(1.0 / d, 0.0)?;

    Ok(ComplexTensor::new_unchecked(beta_real, beta_imag))
}

/// Scalar NRW forward model: computes S11/S21 from material properties at a
/// single frequency on the CPU.  This is the single source-of-truth implementation;
/// the tensor-based [`nrw_direct_with_config`] dispatches per-element through the
/// same physics, and the data generation module calls this directly for grid generation.
pub fn nrw_direct_scalar(
    d: f64,
    a: f64,
    freq: f64,
    eps_r: Complex64,
    mu_r: Complex64,
) -> (Complex64, Complex64) {
    let omega = 2.0 * PI * freq;
    let k0_sq = omega.powi(2) * EPSILON_0 * MU_0;
    let kt_sq = (PI / a).powi(2);
    let beta1e = (k0_sq - kt_sq).sqrt();
    let beta1s = Complex64::new(k0_sq, 0.0)
        .mul(eps_r)
        .mul(mu_r)
        .sub(Complex64::new(kt_sq, 0.0))
        .sqrt();
    let ze = omega * MU_0 / beta1e;
    let zs = Complex64::new(omega * MU_0, 0.0).mul(mu_r).div(beta1s);
    let gamma = zs
        .sub(Complex64::new(ze, 0.0))
        .div(zs.add(Complex64::new(ze, 0.0)));
    let propagation = Complex64::new(0.0, -1.0)
        .mul(beta1s)
        .mul(Complex64::new(d, 0.0))
        .exp();
    let one = Complex64::new(1.0, 0.0);
    let gamma_sq = gamma.mul(gamma);
    let propagation_sq = propagation.mul(propagation);
    let denom = one.sub(gamma_sq.mul(propagation_sq));
    let s11 = gamma.mul(one.sub(propagation_sq)).div(denom);
    let s21 = propagation.mul(one.sub(gamma_sq)).div(denom);
    (s11, s21)
}

/// Scalar NRW forward model for non-magnetic materials (μ_r = 1 + 0j).
pub fn nrw_direct_scalar_non_magnetic(
    d: f64,
    a: f64,
    freq: f64,
    eps_prime: f64,
    eps_double_prime: f64,
) -> (Complex64, Complex64) {
    let eps_r = Complex64::new(eps_prime, -eps_double_prime);
    let mu_r = Complex64::new(1.0, 0.0);
    nrw_direct_scalar(d, a, freq, eps_r, mu_r)
}

/// Computes the NRW forward model using a validated waveguide configuration.
pub fn nrw_direct_with_config(
    config: &WaveguideConfig,
    frequencies: &Tensor,
    eps_r: &ComplexTensor,
    mu_r: &ComplexTensor,
) -> Result<(ComplexTensor, ComplexTensor)> {

    let frequencies = frequencies.to_dtype(DType::F64)?;
    let eps_r = as_complex_f64(eps_r)?;
    let mu_r = as_complex_f64(mu_r)?;

    let omega = frequencies.affine(2.0 * PI, 0.0)?;
    let k0_sq = omega.sqr()?.affine(EPSILON_0 * MU_0, 0.0)?;
    let kt_sq = config.kt().powi(2);

    let beta1e = k0_sq.affine(1.0, -kt_sq)?.sqrt()?;

    let eps_mu = eps_r.mul(&mu_r)?;
    let beta1s_sq = ComplexTensor::new_unchecked(
        eps_mu.real.broadcast_mul(&k0_sq)?.affine(1.0, -kt_sq)?,
        eps_mu.imag.broadcast_mul(&k0_sq)?,
    );
    let beta1s = beta1s_sq.sqrt()?;

    let omega_mu0 = omega.affine(MU_0, 0.0)?;
    let ze = ComplexTensor::from_real(omega_mu0.broadcast_div(&beta1e)?)?;
    let zs = mu_r.scale(&omega_mu0)?.div(&beta1s)?;

    let gamma = zs.sub(&ze)?.div(&zs.add(&ze)?)?;
    let p = ComplexTensor::new_unchecked(
        beta1s.imag.affine(config.sample_thickness(), 0.0)?,
        beta1s.real.affine(-config.sample_thickness(), 0.0)?,
    )
    .exp()?;

    let gamma_sq = gamma.mul(&gamma)?;
    let p_sq = p.mul(&p)?;
    let one = ComplexTensor::from_real(gamma.real.ones_like()?)?;
    let denom = one.sub(&gamma_sq.mul(&p_sq)?)?;

    let s11 = gamma.mul(&one.sub(&p_sq)?)?.div(&denom)?;
    let s21 = p.mul(&one.sub(&gamma_sq)?)?.div(&denom)?;

    Ok((s11, s21))
}

/// Whether to use the stable (`u = √((|z|+a)/2)`, `v = b/(2u)`)
/// or the conventional principal-branch (`u`, `v = sgn(b)·√((|z|−a)/2)`)
/// sqrt for the NRW forward.  The stable form fixes the `1/√0`
/// backward-gradient singularity at the vacuum boundary; toggle
/// to the principal form via the `SPARAM_NRW_SQRT=principal`
/// environment variable to reproduce the unstable behaviour for
/// methodology study M10.
fn use_principal_sqrt() -> bool {
    std::env::var("SPARAM_NRW_SQRT")
        .map(|s| s.eq_ignore_ascii_case("principal"))
        .unwrap_or(false)
}

/// Conventional principal complex sqrt — algebraically equivalent
/// to [`complex_sqrt_pos_real`] on the forward pass but with a
/// `1/√0` backward gradient at the vacuum boundary
/// (`Im(z) → 0` with `Re(z) > 0`).
fn complex_sqrt_principal(real: &Tensor, imag: &Tensor) -> Result<(Tensor, Tensor)> {
    use sparam_core::complex_tensor::COMPLEX_STABILITY_EPS;
    let mag_sq = real.sqr()?.broadcast_add(&imag.sqr()?)?;
    let mag = mag_sq.affine(1.0, COMPLEX_STABILITY_EPS)?.sqrt()?;
    let u_arg = mag.broadcast_add(real)?.affine(0.5, COMPLEX_STABILITY_EPS)?;
    let u = u_arg.sqrt()?;
    // Principal-branch v: requires the second sqrt that becomes
    // 1/√0 in the backward when Im(z) → 0 and Re(z) > 0.
    let v_arg = mag.broadcast_sub(real)?.affine(0.5, COMPLEX_STABILITY_EPS)?;
    let v_mag = v_arg.clamp(0.0, f64::INFINITY)?.sqrt()?;
    let ones = v_mag.ones_like()?;
    let neg_ones = ones.neg()?;
    let imag_sign = imag.ge(0.0)?.where_cond(&ones, &neg_ones)?;
    let v = imag_sign.broadcast_mul(&v_mag)?;
    Ok((u, v))
}

/// Dispatches between [`complex_sqrt_pos_real`] (default, stable
/// gradients near vacuum) and [`complex_sqrt_principal`]
/// (conventional principal-branch, used by methodology study M10
/// to reproduce the gradient instability).  The default is
/// always stable; setting `SPARAM_NRW_SQRT=principal` switches
/// for the duration of the process.
fn complex_sqrt_pos_real_dispatch(
    real: &Tensor,
    imag: &Tensor,
) -> Result<(Tensor, Tensor)> {
    if use_principal_sqrt() {
        complex_sqrt_principal(real, imag)
    } else {
        complex_sqrt_pos_real(real, imag)
    }
}

/// Numerically-stable principal complex sqrt for inputs with `Re(z) ≥ 0`.
///
/// The conventional principal sqrt computes `imag = sgn(b)·√((|z|−a)/2)`,
/// which has a `1/√0` backward gradient when `|z| = a` (purely real
/// positive input). That singularity NaNs the physics-loss backprop
/// near the vacuum boundary `ε_r ≈ 1 + 0j`, which is exactly the
/// regime the dense-near-one training samples target. The fix used
/// here is the algebraically equivalent but well-conditioned form
///
/// ```text
/// u = √((|z| + a) / 2)        // always > 0 for a ≥ 0
/// v = b / (2·u)                // bounded for a ≥ 0
/// ```
///
/// `2uv = b` is an identity for the principal sqrt, so once `u` is
/// known we can recover `v` without a second `sqrt`. Both branches
/// have analytic gradients with no `1/√0` factors.
///
/// Inputs must satisfy `Re(z) ≥ 0`; the only NRW intermediate that
/// hits this helper is `β₁ₛ² = ε_r·k₀² − kt²`, which stays in the
/// right half-plane for any physical `ε_r` (Re(ε_r) ≥ 1).
fn complex_sqrt_pos_real(real: &Tensor, imag: &Tensor) -> Result<(Tensor, Tensor)> {
    use sparam_core::complex_tensor::COMPLEX_STABILITY_EPS;
    let mag_sq = real.sqr()?.broadcast_add(&imag.sqr()?)?;
    let mag = mag_sq.affine(1.0, COMPLEX_STABILITY_EPS)?.sqrt()?;
    let u_arg = mag.broadcast_add(real)?.affine(0.5, COMPLEX_STABILITY_EPS)?;
    let u = u_arg.sqrt()?;
    let two_u = u.affine(2.0, 0.0)?;
    let v = imag.broadcast_div(&two_u)?;
    Ok((u, v))
}

/// NRW direct problem for non-magnetic materials (μ_r = 1 + 0j).
///
/// Specialised, per-batch-optimised form of
/// [`nrw_direct_with_config`] that avoids the general path's
/// redundant work when μ_r is known to be unity:
///   * no `ones_like` allocation for `mu_r`;
///   * no `ε_r · μ_r` complex multiply (eps_mu ≡ eps_r);
///   * no `μ_r · ω·μ₀` complex scale (numerator of Z_s is real);
///   * no `1` complex tensor materialisation — `1 − z` is done as
///     `(z.real.affine(-1, 1), z.imag.neg())`;
///   * constant folding: `ω` intermediate elided, `k₀²` and `ω·μ₀`
///     computed directly from `frequencies`.
///
/// The single `sqrt` on `β₁ₛ²` is routed through
/// [`complex_sqrt_pos_real`] so the backward pass stays bounded when
/// the model's predicted ε_r approaches the vacuum boundary
/// (purely-real-positive `β₁ₛ²`). Forward values match the standard
/// principal-sqrt path to f64 round-off; only the gradient differs.
///
/// These micro-optimisations matter because this function runs once
/// per training batch through [`sparam_training::losses::
/// physics_forward_loss`].
pub fn nrw_direct_non_magnetic_with_config(
    config: &WaveguideConfig,
    frequencies: &Tensor,
    eps_r: &ComplexTensor,
) -> Result<(ComplexTensor, ComplexTensor)> {
    use sparam_core::complex_tensor::COMPLEX_STABILITY_EPS;
    ensure_same_device(frequencies.device(), "frequencies", eps_r.device(), "eps_r")?;

    let frequencies = frequencies.to_dtype(DType::F64)?;
    let eps_r = as_complex_f64(eps_r)?;

    let kt_sq = config.kt().powi(2);
    let d = config.sample_thickness();

    // k₀² = (2πf)² · ε₀μ₀, ω·μ₀ = 2π·μ₀·f — folded to one affine each.
    let k0_sq = frequencies
        .sqr()?
        .affine((2.0 * PI).powi(2) * EPSILON_0 * MU_0, 0.0)?;
    let omega_mu0 = frequencies.affine(2.0 * PI * MU_0, 0.0)?;

    // β₁ₑ = √(k₀² - kt²) — real scalar tensor
    let beta1e = k0_sq.affine(1.0, -kt_sq)?.sqrt()?;

    // μ_r = 1+0j ⇒ β₁ₛ² = ε_r·k₀² − kt² (no ε_r·μ_r multiply).
    // Stable sqrt: bounded backward at the vacuum boundary where the
    // standard principal-sqrt formula has a `1/√0` gradient term.
    let beta1s_sq_real = eps_r.real.broadcast_mul(&k0_sq)?.affine(1.0, -kt_sq)?;
    let beta1s_sq_imag = eps_r.imag.broadcast_mul(&k0_sq)?;
    let (beta1s_real, beta1s_imag) =
        complex_sqrt_pos_real_dispatch(&beta1s_sq_real, &beta1s_sq_imag)?;
    let beta1s = ComplexTensor::new_unchecked(beta1s_real, beta1s_imag);

    // Z_e = ω·μ₀ / β₁ₑ (real).
    let ze = ComplexTensor::from_real(omega_mu0.broadcast_div(&beta1e)?)?;
    // Z_s = ω·μ₀ / β₁ₛ — real numerator, so
    // (a, 0) / (c, d) = a·(c, −d) / (c² + d²). Saves 2 of the 4
    // scalar muls `ComplexTensor::div` would run on a full complex
    // numerator.
    let zs = {
        let mag_sq = beta1s
            .real
            .sqr()?
            .broadcast_add(&beta1s.imag.sqr()?)?;
        let inv = mag_sq.affine(1.0, COMPLEX_STABILITY_EPS)?.recip()?;
        let num_real = omega_mu0.broadcast_mul(&beta1s.real)?;
        let num_imag = omega_mu0.broadcast_mul(&beta1s.imag)?.neg()?;
        ComplexTensor::new_unchecked(
            num_real.broadcast_mul(&inv)?,
            num_imag.broadcast_mul(&inv)?,
        )
    };

    // Γ = (Z_s − Z_e) / (Z_s + Z_e)
    let gamma = zs.sub(&ze)?.div(&zs.add(&ze)?)?;

    // P = exp(−j·β₁ₛ·d)
    let p = ComplexTensor::new_unchecked(
        beta1s.imag.affine(d, 0.0)?,
        beta1s.real.affine(-d, 0.0)?,
    )
    .exp()?;

    let gamma_sq = gamma.mul(&gamma)?;
    let p_sq = p.mul(&p)?;

    // `1 − z` without materialising the `1` tensor.
    let one_minus_p_sq = ComplexTensor::new_unchecked(
        p_sq.real.affine(-1.0, 1.0)?,
        p_sq.imag.neg()?,
    );
    let one_minus_gamma_sq = ComplexTensor::new_unchecked(
        gamma_sq.real.affine(-1.0, 1.0)?,
        gamma_sq.imag.neg()?,
    );
    let gamma_sq_p_sq = gamma_sq.mul(&p_sq)?;
    let denom = ComplexTensor::new_unchecked(
        gamma_sq_p_sq.real.affine(-1.0, 1.0)?,
        gamma_sq_p_sq.imag.neg()?,
    );

    let s11 = gamma.mul(&one_minus_p_sq)?.div(&denom)?;
    let s21 = p.mul(&one_minus_gamma_sq)?.div(&denom)?;

    Ok((s11, s21))
}

/// Computes the NRW inverse model using a validated waveguide configuration.
pub fn nrw_inverse_with_config(
    config: &WaveguideConfig,
    frequencies: &Tensor,
    s11: &ComplexTensor,
    s21: &ComplexTensor,
) -> Result<(ComplexTensor, ComplexTensor)> {
    ensure_same_device_sparameters(frequencies, s11, s21)?;

    let frequencies = frequencies.to_dtype(DType::F64)?;
    let s11 = as_complex_f64(s11)?;
    let s21 = as_complex_f64(s21)?;

    let v1 = s21.add(&s11)?;
    let v2 = s21.sub(&s11)?;
    let one = ComplexTensor::from_real(v1.real.ones_like()?)?;
    let x_num = one.sub(&v1.mul(&v2)?)?;
    let x_den_base = v1.sub(&v2)?;
    // Epsilon stabilisation — applied to BOTH the real and imaginary
    // parts. A previous version added epsilon only to the real part,
    // which left `1/x_den` vulnerable to explosion when `x_den` was
    // purely imaginary and small (e.g. an s11 with Re(s11) ≈ 0,
    // Im(s11) at machine-epsilon scale — a valid but rare case). For
    // small x, |z + ε(1+i)|² = |z|² + 2ε(Re(z)+Im(z)) + 2ε², which is
    // bounded below by 2·ε² even when both original components are
    // zero, so the subsequent `x_num / x_den` stays finite.
    let x_den = ComplexTensor::new_unchecked(
        x_den_base.real.affine(1.0, 1e-16)?,
        x_den_base.imag.affine(1.0, 1e-16)?,
    );
    let x = x_num.div(&x_den)?;

    let gamma_root = x.mul(&x)?.sub(&one)?.sqrt()?;
    let gamma_plus = x.add(&gamma_root)?;
    let gamma_minus = x.sub(&gamma_root)?;
    let gamma_plus_outside_unit_disk = gamma_plus.mag()?.affine(1.0, -1.0)?.ge(f64::EPSILON)?;
    let gamma = ComplexTensor::new_unchecked(
        gamma_plus_outside_unit_disk.where_cond(&gamma_minus.real, &gamma_plus.real)?,
        gamma_plus_outside_unit_disk.where_cond(&gamma_minus.imag, &gamma_plus.imag)?,
    );

    let propagation = v1.sub(&gamma)?.div(&one.sub(&gamma.mul(&v1)?)?)?;
    let beta1s =
        propagation_constant_from_transmission_factor(&propagation, config.sample_thickness())?;

    let omega = frequencies.affine(2.0 * PI, 0.0)?;
    let k0_sq = omega.sqr()?.affine(EPSILON_0 * MU_0, 0.0)?;
    let kt_sq = config.kt().powi(2);
    let beta1e = k0_sq.affine(1.0, -kt_sq)?.sqrt()?;

    let impedance_ratio = one.add(&gamma)?.div(&one.sub(&gamma)?)?;
    let mu_r = impedance_ratio
        .mul(&beta1s)?
        .div(&ComplexTensor::from_real(beta1e)?)?;

    let beta1s_sq = beta1s.mul(&beta1s)?;
    let epsilon_num =
        ComplexTensor::new_unchecked(beta1s_sq.real.affine(1.0, kt_sq)?, beta1s_sq.imag);
    let epsilon_r = epsilon_num.div(&mu_r.scale(&k0_sq)?)?;

    Ok((epsilon_r, mu_r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use num_complex::Complex64;
    use serde::Deserialize;
    use std::process::Command;
    use std::time::Instant;

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
        mu_r_real: Vec<f64>,
        mu_r_imag: Vec<f64>,
        s11_real: Vec<f64>,
        s11_imag: Vec<f64>,
        s21_real: Vec<f64>,
        s21_imag: Vec<f64>,
    }

    #[derive(Debug, Deserialize)]
    struct PythonBenchmarkRoot {
        benchmark: PythonBenchmark,
    }

    #[derive(Debug, Deserialize)]
    struct PythonBenchmark {
        median_us: f64,
        mean_us: f64,
    }

    #[derive(Debug, Deserialize)]
    struct PythonInverseParityRoot {
        inverse_parity: PythonInverseParityCase,
    }

    #[derive(Debug, Deserialize)]
    struct PythonInverseParityCase {
        a: f64,
        d: f64,
        frequencies_hz: Vec<f64>,
        s11_real: Vec<f64>,
        s11_imag: Vec<f64>,
        s21_real: Vec<f64>,
        s21_imag: Vec<f64>,
        eps_r_real: Vec<f64>,
        eps_r_imag: Vec<f64>,
        mu_r_real: Vec<f64>,
        mu_r_imag: Vec<f64>,
    }

    #[derive(Debug, Deserialize)]
    struct PythonInverseFailureRoot {
        inverse_failure: PythonInverseFailureCase,
    }

    #[derive(Debug, Deserialize)]
    struct PythonInverseFailureCase {
        a: f64,
        d: f64,
        frequencies_hz: Vec<f64>,
        s11_real: Vec<f64>,
        s11_imag: Vec<f64>,
        s21_real: Vec<f64>,
        s21_imag: Vec<f64>,
        eps_r_real_is_finite: Vec<bool>,
        eps_r_imag_is_finite: Vec<bool>,
        mu_r_real_is_finite: Vec<bool>,
        mu_r_imag_is_finite: Vec<bool>,
    }

    fn benchmark_inputs(
        batch_size: usize,
        device: &Device,
    ) -> Result<(Tensor, ComplexTensor, ComplexTensor)> {
        let frequency_span = 12.4e9 - 8.2e9;
        let frequency_values: Vec<f64> = if batch_size == 1 {
            vec![8.2e9]
        } else {
            (0..batch_size)
                .map(|index| 8.2e9 + frequency_span * index as f64 / (batch_size - 1) as f64)
                .collect()
        };
        let eps_real_values = vec![2.5f64; batch_size];
        let eps_imag_values = vec![-0.15f64; batch_size];
        let mu_real_values = vec![1.0f64; batch_size];
        let mu_imag_values = vec![0.0f64; batch_size];

        let frequencies = Tensor::new(frequency_values.as_slice(), device)?;
        let eps_r = ComplexTensor::new(
            Tensor::new(eps_real_values.as_slice(), device)?,
            Tensor::new(eps_imag_values.as_slice(), device)?,
        )?;
        let mu_r = ComplexTensor::new(
            Tensor::new(mu_real_values.as_slice(), device)?,
            Tensor::new(mu_imag_values.as_slice(), device)?,
        )?;

        Ok((frequencies, eps_r, mu_r))
    }

    fn median_us(mut samples: Vec<f64>) -> f64 {
        samples.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
        let mid = samples.len() / 2;
        if samples.len().is_multiple_of(2) {
            (samples[mid - 1] + samples[mid]) * 0.5
        } else {
            samples[mid]
        }
    }

    fn nrw_direct_reference(
        d: f64,
        a: f64,
        frequency: f64,
        eps_r: Complex64,
        mu_r: Complex64,
    ) -> (Complex64, Complex64) {
        let omega = 2.0 * PI * frequency;
        let k0_sq = omega.powi(2) * EPSILON_0 * MU_0;
        let kt_sq = (PI / a).powi(2);
        let beta1e = (k0_sq - kt_sq).sqrt();
        let beta1s =
            (Complex64::new(k0_sq, 0.0) * eps_r * mu_r - Complex64::new(kt_sq, 0.0)).sqrt();
        let ze = omega * MU_0 / beta1e;
        let zs = Complex64::new(omega * MU_0, 0.0) * mu_r / beta1s;
        let gamma = (zs - ze) / (zs + ze);
        let p = (Complex64::new(0.0, -1.0) * beta1s * d).exp();
        let one = Complex64::new(1.0, 0.0);
        let denom = one - gamma * gamma * p * p;
        let s11 = gamma * (one - p * p) / denom;
        let s21 = p * (one - gamma * gamma) / denom;
        (s11, s21)
    }

    #[test]
    fn test_nrw_direct_vacuum_sample() -> Result<()> {
        let device = Device::Cpu;
        let d = 1.5e-3;
        let a = 22.86e-3;
        let frequencies = Tensor::new(&[10.0e9f64], &device)?;
        let eps_r = ComplexTensor::from_real(Tensor::new(&[1.0f64], &device)?)?;
        let mu_r = ComplexTensor::from_real(Tensor::new(&[1.0f64], &device)?)?;

        let (s11, s21) =
            nrw_direct_with_config(&WaveguideConfig::new(d, a)?, &frequencies, &eps_r, &mu_r)?;

        assert!(s11.real.to_vec1::<f64>()?[0].abs() < 1e-12);
        assert!(s11.imag.to_vec1::<f64>()?[0].abs() < 1e-12);

        let omega = 2.0 * PI * 10.0e9;
        let beta1e = (omega.powi(2) * EPSILON_0 * MU_0 - (PI / a).powi(2)).sqrt();
        let expected_s21 = (Complex64::new(0.0, -1.0) * beta1e * d).exp();
        assert!((s21.real.to_vec1::<f64>()?[0] - expected_s21.re).abs() < 1e-12);
        assert!((s21.imag.to_vec1::<f64>()?[0] - expected_s21.im).abs() < 1e-12);

        Ok(())
    }

    #[test]
    fn test_nrw_direct_matches_reference_for_batched_inputs() -> Result<()> {
        let device = Device::Cpu;
        let d = 1.5e-3;
        let a = 22.86e-3;
        let frequencies = Tensor::new(&[8.2e9f64, 10.0e9, 12.4e9], &device)?;
        let eps_r = ComplexTensor::new(
            Tensor::new(&[2.5f64, 3.2, 4.0], &device)?,
            Tensor::new(&[-0.15f64, -0.30, -0.45], &device)?,
        )?;
        let mu_r = ComplexTensor::from_real(Tensor::new(&[1.0f64], &device)?)?;

        let (s11, s21) =
            nrw_direct_with_config(&WaveguideConfig::new(d, a)?, &frequencies, &eps_r, &mu_r)?;
        let s11_real = s11.real.to_vec1::<f64>()?;
        let s11_imag = s11.imag.to_vec1::<f64>()?;
        let s21_real = s21.real.to_vec1::<f64>()?;
        let s21_imag = s21.imag.to_vec1::<f64>()?;

        let refs = [
            nrw_direct_reference(
                d,
                a,
                8.2e9,
                Complex64::new(2.5, -0.15),
                Complex64::new(1.0, 0.0),
            ),
            nrw_direct_reference(
                d,
                a,
                10.0e9,
                Complex64::new(3.2, -0.30),
                Complex64::new(1.0, 0.0),
            ),
            nrw_direct_reference(
                d,
                a,
                12.4e9,
                Complex64::new(4.0, -0.45),
                Complex64::new(1.0, 0.0),
            ),
        ];

        for (
            index,
            (
                (expected_s11, expected_s21),
                (actual_s11_re, actual_s11_im, actual_s21_re, actual_s21_im),
            ),
        ) in refs
            .iter()
            .zip(
                s11_real
                    .iter()
                    .zip(s11_imag.iter())
                    .zip(s21_real.iter().zip(s21_imag.iter()))
                    .map(|((s11_re, s11_im), (s21_re, s21_im))| {
                        (*s11_re, *s11_im, *s21_re, *s21_im)
                    }),
            )
            .enumerate()
        {
            assert!(
                (actual_s11_re - expected_s11.re).abs() < 1e-10,
                "s11 real mismatch at {index}: {actual_s11_re} vs {}",
                expected_s11.re
            );
            assert!(
                (actual_s11_im - expected_s11.im).abs() < 1e-10,
                "s11 imag mismatch at {index}: {actual_s11_im} vs {}",
                expected_s11.im
            );
            assert!(
                (actual_s21_re - expected_s21.re).abs() < 1e-10,
                "s21 real mismatch at {index}: {actual_s21_re} vs {}",
                expected_s21.re
            );
            assert!(
                (actual_s21_im - expected_s21.im).abs() < 1e-10,
                "s21 imag mismatch at {index}: {actual_s21_im} vs {}",
                expected_s21.im
            );
        }

        Ok(())
    }


    #[test]
    fn test_waveguide_config_validation_and_helpers() -> Result<()> {
        let config = WaveguideConfig::new(1.5e-3, 22.86e-3)?;
        assert!((config.sample_thickness() - 1.5e-3).abs() < f64::EPSILON);
        assert!((config.width() - 22.86e-3).abs() < f64::EPSILON);
        assert!((config.cutoff_frequency() - (SPEED_OF_LIGHT / (2.0 * 22.86e-3))).abs() < 1e-12);
        assert!((config.kt() - (PI / 22.86e-3)).abs() < 1e-12);

        let error = WaveguideConfig::new(0.0, 22.86e-3)
            .err()
            .expect("zero thickness should fail");
        assert!(format!("{error}").contains("sample thickness d"));

        let error = WaveguideConfig::new(1.0, -1.0)
            .err()
            .expect("negative width should fail");
        assert!(format!("{error}").contains("waveguide width a"));

        Ok(())
    }

    #[test]
    fn test_nrw_direct_non_magnetic_matches_general_case() -> Result<()> {
        let device = Device::Cpu;
        let frequencies = Tensor::new(&[8.2e9f64, 10.0e9, 12.4e9], &device)?;
        let eps_r = ComplexTensor::new(
            Tensor::new(&[2.5f64, 3.2, 4.0], &device)?,
            Tensor::new(&[-0.15f64, -0.30, -0.45], &device)?,
        )?;
        let mu_r = ComplexTensor::from_real(Tensor::new(&[1.0f64], &device)?)?;

        let config = WaveguideConfig::new(1.5e-3, 22.86e-3)?;
        let (general_s11, general_s21) =
            nrw_direct_with_config(&config, &frequencies, &eps_r, &mu_r)?;
        let (non_mag_s11, non_mag_s21) =
            nrw_direct_non_magnetic_with_config(&config, &frequencies, &eps_r)?;

        for (actual, expected) in general_s11
            .real
            .to_vec1::<f64>()?
            .iter()
            .zip(non_mag_s11.real.to_vec1::<f64>()?.iter())
        {
            assert!((actual - expected).abs() < 1e-12);
        }
        for (actual, expected) in general_s11
            .imag
            .to_vec1::<f64>()?
            .iter()
            .zip(non_mag_s11.imag.to_vec1::<f64>()?.iter())
        {
            assert!((actual - expected).abs() < 1e-12);
        }
        for (actual, expected) in general_s21
            .real
            .to_vec1::<f64>()?
            .iter()
            .zip(non_mag_s21.real.to_vec1::<f64>()?.iter())
        {
            assert!((actual - expected).abs() < 1e-12);
        }
        for (actual, expected) in general_s21
            .imag
            .to_vec1::<f64>()?
            .iter()
            .zip(non_mag_s21.imag.to_vec1::<f64>()?.iter())
        {
            assert!((actual - expected).abs() < 1e-12);
        }

        Ok(())
    }


    #[test]
    fn test_nrw_inverse_round_trip_recovers_original_material_properties() -> Result<()> {
        let device = Device::Cpu;
        let frequencies = Tensor::new(&[8.2e9f64, 10.0e9, 12.4e9], &device)?;
        let eps_r = ComplexTensor::new(
            Tensor::new(&[2.5f64, 3.2, 4.0], &device)?,
            Tensor::new(&[-0.15f64, -0.30, -0.45], &device)?,
        )?;
        let mu_r = ComplexTensor::new(
            Tensor::new(&[1.0f64, 1.05, 1.1], &device)?,
            Tensor::new(&[0.0f64, -0.02, -0.05], &device)?,
        )?;

        let config = WaveguideConfig::new(1.5e-3, 22.86e-3)?;
        let (s11, s21) = nrw_direct_with_config(&config, &frequencies, &eps_r, &mu_r)?;
        let (recovered_eps_r, recovered_mu_r) =
            nrw_inverse_with_config(&config, &frequencies, &s11, &s21)?;

        for (actual, expected) in recovered_eps_r
            .real
            .to_vec1::<f64>()?
            .iter()
            .zip(eps_r.real.to_vec1::<f64>()?.iter())
        {
            assert!((actual - expected).abs() < 1e-8);
        }
        for (actual, expected) in recovered_eps_r
            .imag
            .to_vec1::<f64>()?
            .iter()
            .zip(eps_r.imag.to_vec1::<f64>()?.iter())
        {
            assert!((actual - expected).abs() < 1e-8);
        }
        for (actual, expected) in recovered_mu_r
            .real
            .to_vec1::<f64>()?
            .iter()
            .zip(mu_r.real.to_vec1::<f64>()?.iter())
        {
            assert!((actual - expected).abs() < 1e-8);
        }
        for (actual, expected) in recovered_mu_r
            .imag
            .to_vec1::<f64>()?
            .iter()
            .zip(mu_r.imag.to_vec1::<f64>()?.iter())
        {
            assert!((actual - expected).abs() < 1e-8);
        }

        Ok(())
    }


    #[test]
    fn test_rust_benchmark_not_slower_than_python() -> Result<()> {
        if cfg!(debug_assertions) {
            eprintln!("Skipping Python performance comparison in debug builds.");
            return Ok(());
        }

        let output = match Command::new("python3")
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .args([
                "benches/nrw_python_baseline.py",
                "--mode",
                "benchmark",
                "--batch-size",
                "4096",
                "--warmup",
                "50",
                "--iterations",
                "200",
            ])
            .output()
        {
            Ok(output) => output,
            Err(error) => {
                eprintln!(
                    "Skipping Python performance comparison: failed to start python3: {error}"
                );
                return Ok(());
            }
        };
        if !output.status.success() {
            eprintln!(
                "Skipping Python performance comparison: Python benchmark failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return Ok(());
        }

        let python_result: PythonBenchmarkRoot = serde_json::from_slice(&output.stdout)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let config = WaveguideConfig::new(1.5e-3, 22.86e-3)?;
        let device = Device::Cpu;
        let (frequencies, eps_r, mu_r) = benchmark_inputs(4096, &device)?;

        for _ in 0..50 {
            let _ = nrw_direct_with_config(&config, &frequencies, &eps_r, &mu_r)?;
        }

        let mut samples = Vec::with_capacity(200);
        for _ in 0..200 {
            let start = Instant::now();
            let _ = nrw_direct_with_config(&config, &frequencies, &eps_r, &mu_r)?;
            samples.push(start.elapsed().as_secs_f64() * 1e6);
        }

        let rust_median = median_us(samples);
        let speedup = python_result.benchmark.median_us / rust_median;
        println!(
            "Python median: {:.2} us, mean: {:.2} us; Rust median: {:.2} us; speedup: {:.2}x",
            python_result.benchmark.median_us,
            python_result.benchmark.mean_us,
            rust_median,
            speedup,
        );
        // Defends the manuscript's headline 5.8× speedup claim with a
        // conservative lower bound: the actual ratio fluctuates with
        // hardware (NumPy SIMD, AVX2/AVX-512 availability), so the
        // assertion holds at 2.0× rather than the headline number.
        // A failure here likely means the Python baseline got faster
        // (e.g. NumPy upgrade) and the manuscript number needs to be
        // re-measured before the next paper revision.
        assert!(
            speedup >= 2.0,
            "Rust speedup over Python regressed below 2.0x (manuscript claims 5.8x): {:.2}x",
            speedup
        );

        Ok(())
    }

    #[test]
    fn test_nrw_direct_minimum_throughput() -> Result<()> {
        let device = Device::Cpu;
        let config = WaveguideConfig::new(1.5e-3, 22.86e-3)?;
        let batch_size = 1024;
        let freq = 8.2e9;

        let frequencies = Tensor::from_vec(vec![freq; batch_size], batch_size, &device)?;
        let eps_real: Vec<f64> = (0..batch_size).map(|i| 1.0 + (i as f64 % 99.0)).collect();
        let eps_imag: Vec<f64> = (0..batch_size).map(|i| -(i as f64 % 50.0)).collect();
        let eps_r = ComplexTensor::new(
            Tensor::from_vec(eps_real, batch_size, &device)?,
            Tensor::from_vec(eps_imag, batch_size, &device)?,
        )?;
        let mu_r = ComplexTensor::new(
            Tensor::ones(batch_size, DType::F64, &device)?,
            Tensor::zeros(batch_size, DType::F64, &device)?,
        )?;

        let iterations = 100;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = nrw_direct_with_config(&config, &frequencies, &eps_r, &mu_r)?;
        }
        let elapsed = start.elapsed();

        let throughput = (iterations * batch_size) as f64 / elapsed.as_secs_f64();
        assert!(
            throughput > 1_000_000.0,
            "NRW direct throughput {throughput:.0} samples/s below 1M minimum"
        );
        Ok(())
    }
}

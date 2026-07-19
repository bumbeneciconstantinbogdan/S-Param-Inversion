//! Complex-valued tensor wrapper for Candle tensors.
//!
//! [`ComplexTensor`] wraps a pair of Candle [`Tensor`]s (real and imaginary)
//! and provides element-wise arithmetic, magnitude, phase, and
//! interleave/deinterleave helpers needed by the complex MLP training pipeline.

use candle_core::{DType, Device, Result, Shape, Tensor};

/// Numerical-stability floor for complex ops that would otherwise hit
/// `sqrt(0)` / `1/0` singularities (`mag`, `div`, `fast_complex_phase`,
/// activation zero-guards). `1e-12` stays above F32 subnormals while
/// keeping F64 physics parity within the `1e-10` golden tolerance.
pub const COMPLEX_STABILITY_EPS: f64 = 1e-12;

/// A wrapper for complex numbers represented by a pair of real and imaginary tensors.
#[derive(Clone, Debug)]
pub struct ComplexTensor {
    /// Real part tensor.
    pub real: Tensor,
    /// Imaginary part tensor.
    pub imag: Tensor,
}

impl ComplexTensor {
    /// Creates a new [`ComplexTensor`] from real and imaginary parts, validating
    /// that shapes, devices, and dtypes match.
    pub fn new(real: Tensor, imag: Tensor) -> Result<Self> {
        #[cold]
        #[inline(never)]
        fn shape_err(r: &Shape, i: &Shape) -> candle_core::Error {
            candle_core::Error::Msg(format!("Shape mismatch: real {r:?} vs imag {i:?}"))
        }
        #[cold]
        #[inline(never)]
        fn device_err(r: &Device, i: &Device) -> candle_core::Error {
            candle_core::Error::Msg(format!("Device mismatch: real {r:?} vs imag {i:?}"))
        }
        #[cold]
        #[inline(never)]
        fn dtype_err(r: DType, i: DType) -> candle_core::Error {
            candle_core::Error::Msg(format!("DType mismatch: real {r:?} vs imag {i:?}"))
        }

        if real.shape() != imag.shape() {
            return Err(shape_err(real.shape(), imag.shape()));
        }
        if !real.device().same_device(imag.device()) {
            return Err(device_err(real.device(), imag.device()));
        }
        if real.dtype() != imag.dtype() {
            return Err(dtype_err(real.dtype(), imag.dtype()));
        }
        Ok(Self { real, imag })
    }

    /// Unchecked constructor for hot paths when metadata is known to match.
    #[inline]
    pub fn new_unchecked(real: Tensor, imag: Tensor) -> Self {
        Self { real, imag }
    }

    /// Creates a complex tensor filled with zeros.
    pub fn zeros(shape: &Shape, dtype: DType, device: &Device) -> Result<Self> {
        Ok(Self::new_unchecked(
            Tensor::zeros(shape, dtype, device)?,
            Tensor::zeros(shape, dtype, device)?,
        ))
    }

    /// Creates a complex tensor from a real tensor with zero imaginary part.
    pub fn from_real(real: Tensor) -> Result<Self> {
        let imag = Tensor::zeros_like(&real)?;
        Ok(Self::new_unchecked(real, imag))
    }

    /// Creates a complex tensor from an imaginary tensor with zero real part.
    pub fn from_imag(imag: Tensor) -> Result<Self> {
        let real = Tensor::zeros_like(&imag)?;
        Ok(Self::new_unchecked(real, imag))
    }

    /// Creates a new [`ComplexTensor`] from a `(real, imag)` pair (validated).
    pub fn from_pair(pair: (Tensor, Tensor)) -> Result<Self> {
        Self::new(pair.0, pair.1)
    }

    /// Returns references as a `(real, imag)` pair.
    #[inline]
    pub fn as_pair(&self) -> (&Tensor, &Tensor) {
        (&self.real, &self.imag)
    }

    /// Consumes `self` and returns owned `(real, imag)` tensors.
    #[inline]
    pub fn into_pair(self) -> (Tensor, Tensor) {
        (self.real, self.imag)
    }

    /// Returns a cloned pair of `(real, imag)` tensors.
    pub fn to_pair(&self) -> (Tensor, Tensor) {
        (self.real.clone(), self.imag.clone())
    }

    /// Returns the tensor shape (same for real and imag).
    #[inline]
    pub fn shape(&self) -> &Shape {
        self.real.shape()
    }

    /// Returns the device (same for real and imag).
    #[inline]
    pub fn device(&self) -> &Device {
        self.real.device()
    }

    /// Returns the dtype (same for real and imag).
    #[inline]
    pub fn dtype(&self) -> DType {
        self.real.dtype()
    }

    /// Complex addition: `C = A + B`.
    #[inline]
    pub fn add(&self, other: &Self) -> Result<Self> {
        Ok(Self::new_unchecked(
            self.real.broadcast_add(&other.real)?,
            self.imag.broadcast_add(&other.imag)?,
        ))
    }

    /// Complex subtraction: `C = A - B`.
    #[inline]
    pub fn sub(&self, other: &Self) -> Result<Self> {
        Ok(Self::new_unchecked(
            self.real.broadcast_sub(&other.real)?,
            self.imag.broadcast_sub(&other.imag)?,
        ))
    }

    /// Complex multiplication: `C = A * B`.
    #[inline]
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let r = self
            .real
            .broadcast_mul(&other.real)?
            .broadcast_sub(&self.imag.broadcast_mul(&other.imag)?)?;
        let i = self
            .real
            .broadcast_mul(&other.imag)?
            .broadcast_add(&self.imag.broadcast_mul(&other.real)?)?;
        Ok(Self::new_unchecked(r, i))
    }

    /// Complex division: `C = A / B` with [`COMPLEX_STABILITY_EPS`] added
    /// to the denominator for numerical stability.
    #[inline]
    pub fn div(&self, other: &Self) -> Result<Self> {
        let ar_br = self.real.broadcast_mul(&other.real)?;
        let ai_bi = self.imag.broadcast_mul(&other.imag)?;
        let ai_br = self.imag.broadcast_mul(&other.real)?;
        let ar_bi = self.real.broadcast_mul(&other.imag)?;

        let denom = other.real.sqr()?.broadcast_add(&other.imag.sqr()?)?;
        let denom = denom.affine(1.0, COMPLEX_STABILITY_EPS)?;
        let inv_denom = denom.recip()?;

        let r = ar_br.broadcast_add(&ai_bi)?.broadcast_mul(&inv_denom)?;
        let i = ai_br.broadcast_sub(&ar_bi)?.broadcast_mul(&inv_denom)?;

        Ok(Self::new_unchecked(r, i))
    }

    /// Returns the complex conjugate: `A* = A_r - jA_i`.
    #[inline]
    pub fn conj(&self) -> Result<Self> {
        Ok(Self::new_unchecked(self.real.clone(), self.imag.neg()?))
    }

    /// Returns the complex conjugate, consuming `self` to avoid cloning the real part.
    pub fn into_conj(self) -> Result<Self> {
        Ok(Self::new_unchecked(self.real, self.imag.neg()?))
    }

    /// Negation: `-z = -A_r - jA_i`.
    #[inline]
    pub fn neg(&self) -> Result<Self> {
        Ok(Self::new_unchecked(self.real.neg()?, self.imag.neg()?))
    }

    /// Scale by a real-valued tensor (broadcasted).
    #[inline]
    pub fn scale(&self, scalar: &Tensor) -> Result<Self> {
        Ok(Self::new_unchecked(
            self.real.broadcast_mul(scalar)?,
            self.imag.broadcast_mul(scalar)?,
        ))
    }

    /// Returns the magnitude squared: `|A|^2 = A_r^2 + A_i^2`.
    #[inline]
    pub fn mag_sq(&self) -> Result<Tensor> {
        self.real.sqr()?.broadcast_add(&self.imag.sqr()?)
    }

    /// Ensure both tensors are contiguous for better cache/SIMD performance.
    pub fn contiguous(&self) -> Result<Self> {
        Ok(Self::new_unchecked(
            self.real.contiguous()?,
            self.imag.contiguous()?,
        ))
    }

    /// Check if both parts are contiguous.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.real.is_contiguous() && self.imag.is_contiguous()
    }

    /// `|A| = sqrt(A_r² + A_i² + ε)`. The ε floor is required because
    /// Candle's `sqrt` backward `grad / (2·√x)` is Inf at `x = 0`,
    /// which poisons the whole gradient via `Inf × 0 = NaN` downstream.
    #[inline]
    pub fn mag(&self) -> Result<Tensor> {
        (self.mag_sq()? + COMPLEX_STABILITY_EPS)?.sqrt()
    }

    /// Returns the principal complex square root.
    #[must_use = "this returns a new ComplexTensor without modifying the original"]
    pub fn sqrt(&self) -> Result<Self> {
        let magnitude = self.mag()?;
        let real = magnitude
            .broadcast_add(&self.real)?
            .affine(0.5, 0.0)?
            .clamp(0.0, f64::INFINITY)?
            .sqrt()?;
        let imag_mag = magnitude
            .broadcast_sub(&self.real)?
            .affine(0.5, 0.0)?
            .clamp(0.0, f64::INFINITY)?
            .sqrt()?;
        let ones = imag_mag.ones_like()?;
        let neg_ones = ones.neg()?;
        let imag_sign = self.imag.ge(0.0)?.where_cond(&ones, &neg_ones)?;
        let imag = imag_sign.broadcast_mul(&imag_mag)?;
        Ok(Self::new_unchecked(real, imag))
    }

    /// Returns the complex exponential: `e^z = e^x(cos y + j sin y)`.
    #[must_use = "this returns a new ComplexTensor without modifying the original"]
    pub fn exp(&self) -> Result<Self> {
        let real_exp = self.real.exp()?;
        let real = real_exp.broadcast_mul(&self.imag.cos()?)?;
        let imag = real_exp.broadcast_mul(&self.imag.sin()?)?;
        Ok(Self::new_unchecked(real, imag))
    }

    /// Returns the principal phase angle `arg(z) = atan2(imag, real)` in radians.
    ///
    /// Delegates to [`fast_complex_phase`].
    #[must_use = "this returns a new Tensor without modifying the original"]
    pub fn phase(&self) -> Result<Tensor> {
        fast_complex_phase(self)
    }
}

/// Branchless polynomial approximation of `atan2(imag, real)`,
/// differentiable and device-resident. Max absolute error ≈ 0.0038 rad
/// (≈ 0.22°); see Rajan et al. 2006, IEEE SPM, eq. (7).
pub fn fast_complex_phase(z: &ComplexTensor) -> Result<Tensor> {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    let x = &z.real;
    let y = &z.imag;

    let a = x.abs()?;
    let b = y.abs()?;

    let u = a.minimum(&b)?;
    let v = a.maximum(&b)?;

    let denom = v.affine(1.0, COMPLEX_STABILITY_EPS)?;
    let r = u.broadcast_div(&denom)?;

    // θ_base = r · ( π/4 + 0.273 · (1 − r) )
    let one_minus_r = r.affine(-1.0, 1.0)?;
    let poly = one_minus_r.affine(0.273, FRAC_PI_4)?;
    let theta_base = r.broadcast_mul(&poly)?;

    // Quadrant folds based on signs of x, y and |x| < |y|.
    let mask_ab = a.lt(&b)?;
    let pi_2_minus_base = theta_base.affine(-1.0, FRAC_PI_2)?;
    let theta1 = mask_ab.where_cond(&pi_2_minus_base, &theta_base)?;

    let mask_x_neg = x.lt(0.0)?;
    let pi_minus_theta1 = theta1.affine(-1.0, PI)?;
    let theta2 = mask_x_neg.where_cond(&pi_minus_theta1, &theta1)?;

    let mask_y_neg = y.lt(0.0)?;
    let neg_theta2 = theta2.affine(-1.0, 0.0)?;
    let theta_final = mask_y_neg.where_cond(&neg_theta2, &theta2)?;

    Ok(theta_final)
}

impl std::ops::Add<&ComplexTensor> for &ComplexTensor {
    type Output = Result<ComplexTensor>;

    fn add(self, rhs: &ComplexTensor) -> Self::Output {
        self.add(rhs)
    }
}

impl std::ops::Sub<&ComplexTensor> for &ComplexTensor {
    type Output = Result<ComplexTensor>;

    fn sub(self, rhs: &ComplexTensor) -> Self::Output {
        self.sub(rhs)
    }
}

impl std::ops::Mul<&ComplexTensor> for &ComplexTensor {
    type Output = Result<ComplexTensor>;

    fn mul(self, rhs: &ComplexTensor) -> Self::Output {
        self.mul(rhs)
    }
}

impl std::ops::Div<&ComplexTensor> for &ComplexTensor {
    type Output = Result<ComplexTensor>;

    fn div(self, rhs: &ComplexTensor) -> Self::Output {
        self.div(rhs)
    }
}

impl std::ops::Neg for &ComplexTensor {
    type Output = Result<ComplexTensor>;

    fn neg(self) -> Self::Output {
        self.neg()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::f64::consts::PI;

    #[test]
    fn all_operations() -> Result<()> {
        let device = Device::Cpu;

        let z1 = ComplexTensor::new(
            Tensor::new(&[3.0f32], &device)?,
            Tensor::new(&[4.0f32], &device)?,
        )?;
        let z2 = ComplexTensor::new(
            Tensor::new(&[1.0f32], &device)?,
            Tensor::new(&[2.0f32], &device)?,
        )?;

        let add = z1.add(&z2)?;
        assert_eq!(add.real.to_vec1::<f32>()?, &[4.0]);
        assert_eq!(add.imag.to_vec1::<f32>()?, &[6.0]);

        let sub = z1.sub(&z2)?;
        assert_eq!(sub.real.to_vec1::<f32>()?, &[2.0]);
        assert_eq!(sub.imag.to_vec1::<f32>()?, &[2.0]);

        let mul = z1.mul(&z2)?;
        assert_eq!(mul.real.to_vec1::<f32>()?, &[-5.0]);
        assert_eq!(mul.imag.to_vec1::<f32>()?, &[10.0]);

        let div = z1.div(&z2)?;
        assert!((div.real.to_vec1::<f32>()?[0] - 2.2).abs() < 1e-5);
        assert!((div.imag.to_vec1::<f32>()?[0] - (-0.4)).abs() < 1e-5);

        let neg = z1.neg()?;
        assert_eq!(neg.real.to_vec1::<f32>()?, &[-3.0]);
        assert_eq!(neg.imag.to_vec1::<f32>()?, &[-4.0]);

        let conj = z1.conj()?;
        assert_eq!(conj.real.to_vec1::<f32>()?, &[3.0]);
        assert_eq!(conj.imag.to_vec1::<f32>()?, &[-4.0]);

        let z1_clone = z1.clone();
        let into_conj = z1_clone.into_conj()?;
        assert_eq!(into_conj.real.to_vec1::<f32>()?, &[3.0]);
        assert_eq!(into_conj.imag.to_vec1::<f32>()?, &[-4.0]);

        let scalar = Tensor::new(&[2.0f32], &device)?;
        let scaled = z1.scale(&scalar)?;
        assert_eq!(scaled.real.to_vec1::<f32>()?, &[6.0]);
        assert_eq!(scaled.imag.to_vec1::<f32>()?, &[8.0]);

        let mag = z1.mag()?;
        assert!((mag.to_vec1::<f32>()?[0] - 5.0).abs() < 1e-5);

        let mag_sq = z1.mag_sq()?;
        assert_eq!(mag_sq.to_vec1::<f32>()?, &[25.0]);

        let cont = z1.contiguous()?;
        assert!(cont.is_contiguous());

        // Operator overloads
        let op_add = (&z1 + &z2)?;
        assert_eq!(op_add.real.to_vec1::<f32>()?, &[4.0]);

        let op_neg = (-&z1)?;
        assert_eq!(op_neg.real.to_vec1::<f32>()?, &[-3.0]);

        Ok(())
    }

    #[test]
    fn constructor_shape_mismatch() {
        let device = Device::Cpu;
        let r = Tensor::new(&[1.0f32, 2.0], &device).unwrap();
        let i = Tensor::new(&[3.0f32, 4.0, 5.0], &device).unwrap();
        let err = ComplexTensor::new(r, i).err().unwrap();
        assert!(format!("{err}").to_lowercase().contains("shape mismatch"));
    }

    #[test]
    fn edge_cases() -> Result<()> {
        let device = Device::Cpu;

        let zero = ComplexTensor::zeros(&Shape::from_dims(&[2]), DType::F32, &device)?;
        assert_eq!(zero.real.to_vec1::<f32>()?, &[0.0, 0.0]);

        let real_only = ComplexTensor::from_real(Tensor::new(&[5.0f32, -3.0], &device)?)?;
        assert_eq!(real_only.imag.to_vec1::<f32>()?, &[0.0, 0.0]);

        let z = ComplexTensor::new(
            Tensor::new(&[3.0f32], &device)?,
            Tensor::new(&[4.0f32], &device)?,
        )?;
        let double_conj = z.conj()?.conj()?;
        assert_eq!(double_conj.real.to_vec1::<f32>()?, &[3.0]);
        assert_eq!(double_conj.imag.to_vec1::<f32>()?, &[4.0]);

        let z_times_conj = z.mul(&z.conj()?)?;
        assert!((z_times_conj.real.to_vec1::<f32>()?[0] - 25.0).abs() < 1e-5);
        assert!(z_times_conj.imag.to_vec1::<f32>()?[0].abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn sqrt_and_exp() -> Result<()> {
        let device = Device::Cpu;

        let z = ComplexTensor::new(
            Tensor::new(&[3.0f64], &device)?,
            Tensor::new(&[4.0f64], &device)?,
        )?;
        let sqrt_z = z.sqrt()?;
        assert!((sqrt_z.real.to_vec1::<f64>()?[0] - 2.0).abs() < 1e-12);
        assert!((sqrt_z.imag.to_vec1::<f64>()?[0] - 1.0).abs() < 1e-12);

        let imaginary_pi = ComplexTensor::new(
            Tensor::new(&[0.0f64], &device)?,
            Tensor::new(&[PI], &device)?,
        )?;
        let exp_z = imaginary_pi.exp()?;
        assert!((exp_z.real.to_vec1::<f64>()?[0] + 1.0).abs() < 1e-12);
        assert!(exp_z.imag.to_vec1::<f64>()?[0].abs() < 1e-12);

        Ok(())
    }

    #[test]
    fn phase_matches_cardinal_axes() -> Result<()> {
        let device = Device::Cpu;

        // The four cardinal axes + the origin. Origin is an edge case —
        // `atan2(0, 0)` is conventionally 0 in std/f64, and our
        // approximation returns exactly 0 there too (min/max both zero).
        let xs = &[1.0f64,  0.0,  -1.0,  0.0,  0.0];
        let ys = &[0.0f64,  1.0,   0.0, -1.0,  0.0];
        let expected = &[0.0f64, PI / 2.0, PI, -PI / 2.0, 0.0];

        let z = ComplexTensor::new(
            Tensor::new(xs, &device)?,
            Tensor::new(ys, &device)?,
        )?;
        let phase = z.phase()?.to_vec1::<f64>()?;
        for (got, want) in phase.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-2,
                "phase(cardinal) got {got}, want {want}"
            );
        }
        Ok(())
    }

    #[test]
    fn phase_error_bound_matches_paper() -> Result<()> {
        // The Rajan et al. 2006 paper bounds the max absolute error at
        // ~0.0038 rad. Sample a dense grid of points across the whole
        // complex plane (excluding the origin) and confirm.
        let device = Device::Cpu;
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        // 41 × 41 grid on [-3, 3] minus the origin.
        for i in -20..=20 {
            for j in -20..=20 {
                let x = i as f64 * 0.15;
                let y = j as f64 * 0.15;
                if x.abs() < 1e-9 && y.abs() < 1e-9 { continue; }
                xs.push(x);
                ys.push(y);
            }
        }

        let z = ComplexTensor::new(
            Tensor::new(xs.as_slice(), &device)?,
            Tensor::new(ys.as_slice(), &device)?,
        )?;
        let approx = z.phase()?.to_vec1::<f64>()?;

        let max_err = xs.iter()
            .zip(ys.iter())
            .zip(approx.iter())
            .map(|((&x, &y), &got)| (got - y.atan2(x)).abs())
            .fold(0.0f64, f64::max);

        // Paper bound is 0.0038 rad; add a small margin for fp roundoff.
        assert!(
            max_err < 0.004,
            "max |θ̂ − atan2| = {max_err:.6} rad exceeds paper bound 0.0038"
        );
        Ok(())
    }

    /// `mag().sum().backward()` must produce finite gradients even
    /// when every input element is exactly zero — the eps floor keeps
    /// sqrt's backward from emitting Inf.
    #[test]
    fn mag_backward_is_finite_at_zero() -> Result<()> {
        use candle_core::Var;
        let device = Device::Cpu;

        let xv = Var::new(&[0.0f32, 0.0, 0.0, 0.0], &device)?;
        let yv = Var::new(&[0.0f32, 0.0, 0.0, 0.0], &device)?;
        let z = ComplexTensor::new(xv.as_tensor().clone(), yv.as_tensor().clone())?;
        let loss = z.mag()?.sum_all()?;
        let grads = loss.backward()?;

        let gx = grads.get(&xv).unwrap().to_vec1::<f32>()?;
        let gy = grads.get(&yv).unwrap().to_vec1::<f32>()?;
        for g in gx.iter().chain(gy.iter()) {
            assert!(g.is_finite());
        }
        Ok(())
    }

    #[test]
    fn phase_is_fully_differentiable() -> Result<()> {
        // The phase path is used inside some complex activations, so the
        // op graph must survive `.backward()`. We don't assert a specific
        // gradient here, only that the backward pass runs end-to-end.
        use candle_core::Var;
        let device = Device::Cpu;

        let xv = Var::new(&[1.0f64, 2.0, -0.5, 0.7], &device)?;
        let yv = Var::new(&[0.3f64, -1.5, 2.0, -0.1], &device)?;
        let z = ComplexTensor::new(xv.as_tensor().clone(), yv.as_tensor().clone())?;
        let loss = z.phase()?.sqr()?.sum_all()?;
        let grads = loss.backward()?;
        assert!(grads.get(&xv).is_some());
        assert!(grads.get(&yv).is_some());
        Ok(())
    }

    #[test]
    fn pair_accessors() -> Result<()> {
        let device = Device::Cpu;
        let r = Tensor::new(&[1.0f32, 2.0], &device)?;
        let i = Tensor::new(&[3.0f32, 4.0], &device)?;
        let c = ComplexTensor::new(r, i)?;

        let (rr, ii) = c.as_pair();
        assert_eq!(rr.to_vec1::<f32>()?, &[1.0, 2.0]);
        assert_eq!(ii.to_vec1::<f32>()?, &[3.0, 4.0]);

        let (r2, i2) = c.clone().into_pair();
        assert_eq!(r2.to_vec1::<f32>()?, &[1.0, 2.0]);
        assert_eq!(i2.to_vec1::<f32>()?, &[3.0, 4.0]);

        Ok(())
    }
}

//! Complex number scalar type for CPU-side computations.
//!
//! [`Complex64`] is a lightweight `(re, im)` scalar used in CPU-only NRW
//! computations where tensor overhead is unnecessary.

/// A lightweight scalar complex number for CPU-only computations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex64 {
    /// Creates a new complex number.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Element-wise addition.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    /// Element-wise subtraction.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }

    /// Complex multiplication.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    /// Complex division.
    #[inline]
    pub fn div(self, other: Self) -> Self {
        let denominator = other.re * other.re + other.im * other.im;
        Self::new(
            (self.re * other.re + self.im * other.im) / denominator,
            (self.im * other.re - self.re * other.im) / denominator,
        )
    }

    /// Returns the magnitude (absolute value).
    #[inline]
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Returns the principal complex square root.
    #[inline]
    pub fn sqrt(self) -> Self {
        let magnitude = self.abs();
        let real = ((magnitude + self.re) * 0.5).max(0.0).sqrt();
        let imag_magnitude = ((magnitude - self.re) * 0.5).max(0.0).sqrt();
        let imag_sign = if self.im.is_sign_negative() {
            -1.0
        } else {
            1.0
        };
        Self::new(real, imag_sign * imag_magnitude)
    }

    /// Returns the complex exponential.
    #[inline]
    pub fn exp(self) -> Self {
        let magnitude = self.re.exp();
        Self::new(magnitude * self.im.cos(), magnitude * self.im.sin())
    }

    /// Returns the complex natural logarithm.
    #[inline]
    pub fn ln(self) -> Self {
        Self::new(self.abs().ln(), self.im.atan2(self.re))
    }
}

impl std::ops::Add for Complex64 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl std::ops::Sub for Complex64 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(rhs)
    }
}

impl std::ops::Mul for Complex64 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

impl std::ops::Div for Complex64 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(rhs)
    }
}

impl std::ops::Neg for Complex64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn basic_ops() {
        let z1 = Complex64::new(3.0, 4.0);
        let z2 = Complex64::new(1.0, 2.0);

        assert_eq!(z1 + z2, Complex64::new(4.0, 6.0));
        assert_eq!(z1 - z2, Complex64::new(2.0, 2.0));
        assert_eq!(z1 * z2, Complex64::new(-5.0, 10.0));

        let div = z1 / z2;
        assert!((div.re - 2.2).abs() < 1e-12);
        assert!((div.im + 0.4).abs() < 1e-12);
        assert!((z1.abs() - 5.0).abs() < 1e-12);

        let sqrt = z1.sqrt();
        assert!((sqrt.re - 2.0).abs() < 1e-12);
        assert!((sqrt.im - 1.0).abs() < 1e-12);
    }

    #[test]
    fn exp_and_ln() {
        let exp = Complex64::new(0.0, PI).exp();
        assert!((exp.re + 1.0).abs() < 1e-12);
        assert!(exp.im.abs() < 1e-12);

        let ln = Complex64::new(1.0, 1.0).ln();
        assert!((ln.re - (2.0_f64.sqrt()).ln()).abs() < 1e-12);
        assert!((ln.im - PI / 4.0).abs() < 1e-12);
    }

    #[test]
    fn negation() {
        let z = Complex64::new(3.0, -4.0);
        assert_eq!(-z, Complex64::new(-3.0, 4.0));
    }
}

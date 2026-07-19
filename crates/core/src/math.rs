//! Pure numeric utility functions.

/// Linearly interpolate between `a` and `b` by factor `t` in `[0, 1]`.
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Clamp a value to the range `[lo, hi]`.
#[inline]
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lerp_endpoints() {
        assert!((lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-15);
        assert!((lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-15);
        assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn clamp_within() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-1.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
    }
}

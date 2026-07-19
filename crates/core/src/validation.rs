//! Input validation helpers for configuration parameters.
//!
//! These primitives consolidate the duplicated `is_finite`, `> 0`, `>= 0`
//! checks that previously lived in every module.

use crate::error::{CoreError, msg_error};

/// Validates that `value` is **finite and strictly positive** (> 0).
pub fn validate_positive_f64(name: &str, value: f64) -> Result<(), CoreError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(msg_error(format!(
            "{name} must be finite and > 0, got {value}"
        )));
    }
    Ok(())
}

/// Validates that `value` is **finite and non-negative** (>= 0).
pub fn validate_non_negative_f64(name: &str, value: f64) -> Result<(), CoreError> {
    if !value.is_finite() || value < 0.0 {
        return Err(msg_error(format!(
            "{name} must be finite and >= 0, got {value}"
        )));
    }
    Ok(())
}

/// Validates that `range` is finite, ordered, and strictly positive.
pub fn validate_positive_range(name: &str, range: (f64, f64)) -> Result<(), CoreError> {
    validate_positive_f64(&format!("{name}.min"), range.0)?;
    validate_positive_f64(&format!("{name}.max"), range.1)?;
    if range.1 < range.0 {
        return Err(msg_error(format!(
            "{name} must be ordered as (min <= max), got ({}, {})",
            range.0, range.1
        )));
    }
    Ok(())
}

/// Validates that `range` is finite, ordered, and non-negative.
pub fn validate_non_negative_range(name: &str, range: (f64, f64)) -> Result<(), CoreError> {
    validate_non_negative_f64(&format!("{name}.min"), range.0)?;
    validate_non_negative_f64(&format!("{name}.max"), range.1)?;
    if range.1 < range.0 {
        return Err(msg_error(format!(
            "{name} must be ordered as (min <= max), got ({}, {})",
            range.0, range.1
        )));
    }
    Ok(())
}

/// Validates that `value` is **strictly positive** (> 0).
pub fn validate_positive_usize(name: &str, value: usize) -> Result<(), CoreError> {
    if value == 0 {
        return Err(msg_error(format!("{name} must be > 0, got 0")));
    }
    Ok(())
}

/// Validates that `tensor` is exactly 2-D with both dimensions > 0.
///
/// Returns `(rows, cols)` on success.
#[cfg(feature = "tensor")]
pub fn validate_2d_tensor(name: &str, tensor: &candle_core::Tensor) -> Result<(usize, usize), CoreError> {
    let dims = tensor.dims();
    if dims.len() != 2 {
        return Err(msg_error(format!(
            "{name} must be 2-D, got shape {dims:?}"
        )));
    }
    if dims[0] == 0 || dims[1] == 0 {
        return Err(msg_error(format!(
            "{name} must have non-empty dimensions, got shape {dims:?}"
        )));
    }
    Ok((dims[0], dims[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_f64_accepts_valid() {
        assert!(validate_positive_f64("x", 1.0).is_ok());
        assert!(validate_positive_f64("x", 0.001).is_ok());
    }

    #[test]
    fn positive_f64_rejects_zero() {
        assert!(validate_positive_f64("x", 0.0).is_err());
    }

    #[test]
    fn positive_f64_rejects_negative() {
        assert!(validate_positive_f64("x", -1.0).is_err());
    }

    #[test]
    fn positive_f64_rejects_nan_inf() {
        assert!(validate_positive_f64("x", f64::NAN).is_err());
        assert!(validate_positive_f64("x", f64::INFINITY).is_err());
    }

    #[test]
    fn non_negative_f64_accepts_zero() {
        assert!(validate_non_negative_f64("x", 0.0).is_ok());
    }

    #[test]
    fn non_negative_f64_rejects_negative() {
        assert!(validate_non_negative_f64("x", -0.001).is_err());
    }

    #[test]
    fn positive_range_accepts_valid() {
        assert!(validate_positive_range("eps_prime", (1.0, 2.0)).is_ok());
    }

    #[test]
    fn positive_range_rejects_reversed() {
        assert!(validate_positive_range("eps_prime", (2.0, 1.0)).is_err());
    }

    #[test]
    fn non_negative_range_accepts_zero_lower() {
        assert!(validate_non_negative_range("eps_double_prime", (0.0, 2.0)).is_ok());
    }

    #[test]
    fn positive_usize_rejects_zero() {
        assert!(validate_positive_usize("n", 0).is_err());
    }

    #[test]
    fn positive_usize_accepts_one() {
        assert!(validate_positive_usize("n", 1).is_ok());
    }

    #[cfg(feature = "tensor")]
    #[test]
    fn validate_2d_rejects_1d() {
        let t = candle_core::Tensor::zeros(&[5], candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        assert!(validate_2d_tensor("t", &t).is_err());
    }

    #[cfg(feature = "tensor")]
    #[test]
    fn validate_2d_accepts_valid() {
        let t = candle_core::Tensor::zeros(&[3, 4], candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        let (r, c) = validate_2d_tensor("t", &t).unwrap();
        assert_eq!((r, c), (3, 4));
    }
}

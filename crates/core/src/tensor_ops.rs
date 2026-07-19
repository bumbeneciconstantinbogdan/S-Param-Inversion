//! Tensor utility operations: atan2, device checks, type conversions.

use std::sync::Once;

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, DType, Device, Layout, Result, Shape, Tensor};

use crate::complex_tensor::ComplexTensor;

#[derive(Debug, Clone, Copy)]
struct Atan2Op;

impl CustomOp2 for Atan2Op {
    fn name(&self) -> &'static str {
        "atan2"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn contiguous_slice<'a, T>(values: &'a [T], layout: &Layout, name: &str) -> Result<&'a [T]> {
            match layout.contiguous_offsets() {
                Some((start, end)) => Ok(&values[start..end]),
                None => Err(candle_core::Error::Msg(format!(
                    "atan2 {name} input must be contiguous"
                ))),
            }
        }

        fn apply_f32(s1: &[f32], l1: &Layout, s2: &[f32], l2: &Layout) -> Result<(CpuStorage, Shape)> {
            let lhs = contiguous_slice(s1, l1, "lhs")?;
            let rhs = contiguous_slice(s2, l2, "rhs")?;
            let values = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(lhs, rhs)| lhs.atan2(*rhs))
                .collect();
            Ok((CpuStorage::F32(values), Shape::from_dims(l1.shape().dims())))
        }

        fn apply_f64(s1: &[f64], l1: &Layout, s2: &[f64], l2: &Layout) -> Result<(CpuStorage, Shape)> {
            let lhs = contiguous_slice(s1, l1, "lhs")?;
            let rhs = contiguous_slice(s2, l2, "rhs")?;
            let values = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(lhs, rhs)| lhs.atan2(*rhs))
                .collect();
            Ok((CpuStorage::F64(values), Shape::from_dims(l1.shape().dims())))
        }

        match (s1, s2) {
            (CpuStorage::F32(lhs), CpuStorage::F32(rhs)) => apply_f32(lhs, l1, rhs, l2),
            (CpuStorage::F64(lhs), CpuStorage::F64(rhs)) => apply_f64(lhs, l1, rhs, l2),
            _ => Err(candle_core::Error::Msg(format!(
                "atan2_tensor supports matching F32/F64 tensors, got {:?} and {:?}",
                s1.dtype(),
                s2.dtype()
            ))),
        }
    }
}

static NON_CPU_ATAN2_WARNING: Once = Once::new();

#[cold]
fn warn_non_cpu_atan2_fallback() {
    NON_CPU_ATAN2_WARNING.call_once(|| {
        eprintln!(
            "warning: atan2_tensor on non-CPU devices copies inputs through host memory; avoid using it on hot paths"
        );
    });
}

/// Asserts that two [`Device`] references refer to the same device.
pub fn ensure_same_device(
    lhs_device: &Device,
    lhs_name: &str,
    rhs_device: &Device,
    rhs_name: &str,
) -> Result<()> {
    if !lhs_device.same_device(rhs_device) {
        return Err(candle_core::Error::Msg(format!(
            "Device mismatch between {lhs_name} and {rhs_name}"
        )));
    }
    Ok(())
}

/// Promotes a [`ComplexTensor`] to `F64` dtype (both real and imaginary parts).
pub fn as_complex_f64(value: &ComplexTensor) -> Result<ComplexTensor> {
    Ok(ComplexTensor::new_unchecked(
        value.real.to_dtype(DType::F64)?,
        value.imag.to_dtype(DType::F64)?,
    ))
}

/// Computes element-wise `atan2(y, x)` for F32 or F64 tensors.
pub fn atan2_tensor(y: &Tensor, x: &Tensor) -> Result<Tensor> {
    ensure_same_device(y.device(), "y", x.device(), "x")?;
    if y.dims() != x.dims() {
        return Err(candle_core::Error::Msg(format!(
            "Shape mismatch between y {:?} and x {:?}",
            y.shape(),
            x.shape()
        )));
    }
    if y.dtype() != x.dtype() {
        return Err(candle_core::Error::Msg(format!(
            "DType mismatch between y {:?} and x {:?}",
            y.dtype(),
            x.dtype()
        )));
    }

    if y.device().is_cpu() {
        let y = y.contiguous()?;
        let x = x.contiguous()?;
        return y.apply_op2_no_bwd(&x, &Atan2Op);
    }

    warn_non_cpu_atan2_fallback();

    match y.dtype() {
        DType::F32 => {
            let y_values = y.flatten_all()?.to_vec1::<f32>()?;
            let x_values = x.flatten_all()?.to_vec1::<f32>()?;
            let values = y_values
                .iter()
                .zip(x_values.iter())
                .map(|(lhs, rhs)| lhs.atan2(*rhs))
                .collect();
            Tensor::from_vec(values, y.shape(), y.device())
        }
        DType::F64 => {
            let y_values = y.flatten_all()?.to_vec1::<f64>()?;
            let x_values = x.flatten_all()?.to_vec1::<f64>()?;
            let values = y_values
                .iter()
                .zip(x_values.iter())
                .map(|(lhs, rhs)| lhs.atan2(*rhs))
                .collect();
            Tensor::from_vec(values, y.shape(), y.device())
        }
        dtype => Err(candle_core::Error::Msg(format!(
            "atan2_tensor supports only F32/F64 tensors, got {dtype:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_device_passes() {
        let cpu = Device::Cpu;
        assert!(ensure_same_device(&cpu, "a", &cpu, "b").is_ok());
    }

    #[test]
    fn as_complex_f64_promotes() {
        let real = Tensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
        let imag = Tensor::ones(&[2], DType::F32, &Device::Cpu).unwrap();
        let ct = ComplexTensor::new_unchecked(real, imag);
        let promoted = as_complex_f64(&ct).unwrap();
        assert_eq!(promoted.real.dtype(), DType::F64);
        assert_eq!(promoted.imag.dtype(), DType::F64);
    }

    #[test]
    fn atan2_tensor_matches_scalar_reference() {
        let y = Tensor::from_vec(vec![0.0_f64, 1.0, -1.0, 1.0], (2, 2), &Device::Cpu).unwrap();
        let x = Tensor::from_vec(vec![1.0_f64, 0.0, 0.0, -1.0], (2, 2), &Device::Cpu).unwrap();

        let actual = atan2_tensor(&y, &x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let expected = [
            0.0_f64,
            std::f64::consts::FRAC_PI_2,
            -std::f64::consts::FRAC_PI_2,
            3.0 * std::f64::consts::FRAC_PI_4,
        ];

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "actual={actual}, expected={expected}"
            );
        }
    }
}

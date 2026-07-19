//! Shared tensor bridge helpers for dataset samples and packed complex features.

use candle_core::{Device, Result, Tensor};

use sparam_core::complex_tensor::ComplexTensor;
use sparam_core::error::candle_msg;

use crate::generation::PermittivitySample;

/// Describes how predictions are encoded in the output tensor.
#[derive(Debug, Clone, Copy)]
pub enum PredictionEncoding {
    RealColumns {
        negate_imag: bool,
    },
    PackedComplex {
        complex_feature_count: usize,
        negate_imag: bool,
    },
}

/// Output of [`crate::workflows::samples_to_tensors`] (and equivalents):
/// the `(features, targets)` tensor pair plus the encoding metadata
/// callers need to interpret the prediction shape later.  Bundled into
/// a struct to retire the 3-tuple `(Tensor, Tensor, PredictionEncoding)`
/// that was being destructured at 13+ call sites.
#[derive(Debug)]
pub struct TensorDataset {
    pub features: Tensor,
    pub targets: Tensor,
    pub encoding: PredictionEncoding,
}

/// Convert permittivity samples into real-valued `(features, targets)` tensors.
pub fn samples_to_real_tensors(samples: &[PermittivitySample]) -> Result<(Tensor, Tensor)> {
    let n = samples.len();
    let mut features = Vec::with_capacity(n * 4);
    let mut targets = Vec::with_capacity(n * 2);

    for sample in samples {
        features.push(sample.s11_real);
        features.push(sample.s11_imag);
        features.push(sample.s21_real);
        features.push(sample.s21_imag);
        targets.push(sample.eps_prime);
        targets.push(sample.eps_double_prime);
    }

    Ok((
        Tensor::from_vec(features, &[n, 4], &Device::Cpu)?,
        Tensor::from_vec(targets, &[n, 2], &Device::Cpu)?,
    ))
}

/// Convert permittivity samples into complex-valued feature/target tensors.
pub fn samples_to_complex_tensors(
    samples: &[PermittivitySample],
) -> Result<(ComplexTensor, ComplexTensor)> {
    let n = samples.len();
    let mut feature_real = Vec::with_capacity(n * 2);
    let mut feature_imag = Vec::with_capacity(n * 2);
    let mut target_real = Vec::with_capacity(n);
    let mut target_imag = Vec::with_capacity(n);

    for sample in samples {
        feature_real.push(sample.s11_real);
        feature_real.push(sample.s21_real);
        feature_imag.push(sample.s11_imag);
        feature_imag.push(sample.s21_imag);
        target_real.push(sample.eps_prime);
        target_imag.push(sample.eps_double_prime);
    }

    let features = ComplexTensor::new(
        Tensor::from_vec(feature_real, &[n, 2], &Device::Cpu)?,
        Tensor::from_vec(feature_imag, &[n, 2], &Device::Cpu)?,
    )?;
    let targets = ComplexTensor::new(
        Tensor::from_vec(target_real, &[n, 1], &Device::Cpu)?,
        Tensor::from_vec(target_imag, &[n, 1], &Device::Cpu)?,
    )?;

    Ok((features, targets))
}

/// Convert permittivity samples into packed complex `(features, targets)` tensors.
pub fn samples_to_packed_complex_tensors(
    samples: &[PermittivitySample],
) -> Result<(Tensor, Tensor)> {
    let (features, targets) = samples_to_complex_tensors(samples)?;
    Ok((
        pack_complex_features(&features)?,
        pack_complex_features(&targets)?,
    ))
}

/// Pack a complex tensor into half-half real columns: `[re0, re1, …, im0, im1, …]`.
///
/// Matches the block layout `ComplexLinear::forward` consumes, so the
/// unpack step on the hot path is two zero-copy `narrow`s rather than a
/// `reshape` + `narrow` dance.
pub fn pack_complex_features(values: &ComplexTensor) -> Result<Tensor> {
    let dims = values.real.dims();
    if dims.len() != 2 {
        return Err(candle_msg(format!(
            "complex tensor bridge expects 2D tensors shaped (batch, features), got {:?}",
            dims
        )));
    }
    if values.imag.dims() != dims {
        return Err(candle_msg(format!(
            "complex tensor bridge expected matching real/imag shapes, got {:?} vs {:?}",
            dims,
            values.imag.dims()
        )));
    }

    Tensor::cat(&[&values.real, &values.imag], 1)
}

/// Unpack half-half real columns back into a complex tensor. Both parts
/// are zero-copy `narrow` views over the packed storage.
pub fn unpack_complex_features(
    values: &Tensor,
    complex_feature_count: usize,
) -> Result<ComplexTensor> {
    let dims = values.dims();
    if dims.len() != 2 {
        return Err(candle_msg(format!(
            "complex tensor bridge expects 2D tensors shaped (batch, scalar_features), got {:?}",
            dims
        )));
    }

    let expected_scalar_features = complex_feature_count * 2;
    if dims[1] != expected_scalar_features {
        return Err(candle_msg(format!(
            "complex tensor bridge expected {expected_scalar_features} scalar features for {complex_feature_count} complex features, got {}",
            dims[1]
        )));
    }

    let real = values.narrow(1, 0, complex_feature_count)?;
    let imag = values.narrow(1, complex_feature_count, complex_feature_count)?;
    Ok(ComplexTensor::new_unchecked(real, imag))
}

/// Dispatch on `model_type` to produce the matching `(features, targets)`
/// tensors plus the encoding metadata. Real returns `(N, 4)` interleaved
/// features and `(N, 2)` real targets; Complex returns half-half packed
/// `(N, 4)` features and `(N, 2)` packed targets.
pub fn samples_to_tensors(
    model_type: sparam_models::ModelType,
    samples: &[PermittivitySample],
    negate_real_imag: bool,
) -> Result<TensorDataset> {
    match model_type {
        sparam_models::ModelType::Real => {
            let (features, targets) = samples_to_real_tensors(samples)?;
            Ok(TensorDataset {
                features,
                targets,
                encoding: PredictionEncoding::RealColumns {
                    negate_imag: negate_real_imag,
                },
            })
        }
        sparam_models::ModelType::Complex => {
            let (features, targets) = samples_to_packed_complex_tensors(samples)?;
            Ok(TensorDataset {
                features,
                targets,
                encoding: PredictionEncoding::PackedComplex {
                    complex_feature_count: 1,
                    negate_imag: false,
                },
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PermittivitySample {
        PermittivitySample {
            s11_real: 1.0,
            s11_imag: 2.0,
            s21_real: 3.0,
            s21_imag: 4.0,
            eps_prime: 5.0,
            eps_double_prime: 6.0,
            is_dense_patch: false,
        }
    }

    #[test]
    fn real_samples_are_converted_with_expected_shapes() -> Result<()> {
        let (features, targets) = samples_to_real_tensors(&[sample()])?;

        assert_eq!(features.dims(), &[1, 4]);
        assert_eq!(targets.dims(), &[1, 2]);
        assert_eq!(features.to_vec2::<f64>()?, vec![vec![1.0, 2.0, 3.0, 4.0]]);
        assert_eq!(targets.to_vec2::<f64>()?, vec![vec![5.0, 6.0]]);
        Ok(())
    }

    #[test]
    fn complex_bridge_round_trips() -> Result<()> {
        // Half-half layout: [re_block | im_block], so the two complex
        // features `(1+2i, 3+4i)` pack as `[1, 3, 2, 4]` (not the old
        // interleaved `[1, 2, 3, 4]`).
        let (features, targets) = samples_to_complex_tensors(&[sample()])?;
        let packed_features = pack_complex_features(&features)?;
        let packed_targets = pack_complex_features(&targets)?;
        let unpacked_features = unpack_complex_features(&packed_features, 2)?;
        let unpacked_targets = unpack_complex_features(&packed_targets, 1)?;

        assert_eq!(
            packed_features.to_vec2::<f64>()?,
            vec![vec![1.0, 3.0, 2.0, 4.0]]
        );
        assert_eq!(packed_targets.to_vec2::<f64>()?, vec![vec![5.0, -6.0]]);
        assert_eq!(
            unpacked_features.real.to_vec2::<f64>()?,
            features.real.to_vec2::<f64>()?
        );
        assert_eq!(
            unpacked_features.imag.to_vec2::<f64>()?,
            features.imag.to_vec2::<f64>()?
        );
        assert_eq!(
            unpacked_targets.real.to_vec2::<f64>()?,
            targets.real.to_vec2::<f64>()?
        );
        assert_eq!(
            unpacked_targets.imag.to_vec2::<f64>()?,
            targets.imag.to_vec2::<f64>()?
        );
        Ok(())
    }
}

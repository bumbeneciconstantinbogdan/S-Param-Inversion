//! Feature-scaling transforms for the data pipeline.
//!
//! Provides [`IdentityScaler`], [`StandardScaler`] (zero-mean, unit-variance),
//! and [`MinMaxScaler`] ([0, 1] range), all conforming to the [`Scaler`] trait.
//! The type-erased [`ScalerRef`] allows callers to operate on a scaler without
//! knowing its concrete type.

use std::borrow::Cow;

use candle_core::{DType, Device, Result, Tensor};
use sparam_core::error::candle_msg;

/// Common interface for feature-scaling transforms used by the data pipeline.
pub trait Scaler {
    /// Fit scaler statistics from a batched feature tensor of shape `(batch_size, num_features)`.
    fn fit(&mut self, data: &Tensor) -> Result<()>;

    /// Transform a batched feature tensor using the fitted scaler statistics.
    fn transform(&self, data: &Tensor) -> Result<Tensor>;

    /// Reverse a previous `transform()` operation.
    fn inverse_transform(&self, data: &Tensor) -> Result<Tensor>;

    /// Fit and immediately transform the same batch.
    fn fit_transform(&mut self, data: &Tensor) -> Result<Tensor> {
        self.fit(data)?;
        self.transform(data)
    }
}

/// Concrete reference to a fitted scaler implementation.
///
/// This avoids trait-object dispatch on hot data-loading and HPO paths while
/// still supporting the project's current scaler variants.
#[derive(Debug, Clone, Copy)]
pub enum ScalerRef<'a> {
    Identity(&'a IdentityScaler),
    Standard(&'a StandardScaler),
    MinMax(&'a MinMaxScaler),
}

impl<'a> ScalerRef<'a> {
    #[inline]
    pub fn transform(&self, data: &Tensor) -> Result<Tensor> {
        match self {
            Self::Identity(scaler) => scaler.transform(data),
            Self::Standard(scaler) => scaler.transform(data),
            Self::MinMax(scaler) => scaler.transform(data),
        }
    }

    #[inline]
    pub fn inverse_transform(&self, data: &Tensor) -> Result<Tensor> {
        match self {
            Self::Identity(scaler) => scaler.inverse_transform(data),
            Self::Standard(scaler) => scaler.inverse_transform(data),
            Self::MinMax(scaler) => scaler.inverse_transform(data),
        }
    }
}

/// No-op scaler useful for tests and already-normalized tensors.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityScaler;

impl IdentityScaler {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Scaler for IdentityScaler {
    fn fit(&mut self, _data: &Tensor) -> Result<()> {
        Ok(())
    }

    fn transform(&self, data: &Tensor) -> Result<Tensor> {
        Ok(data.clone())
    }

    fn inverse_transform(&self, data: &Tensor) -> Result<Tensor> {
        Ok(data.clone())
    }
}

impl<'a> From<&'a IdentityScaler> for ScalerRef<'a> {
    fn from(scaler: &'a IdentityScaler) -> Self {
        Self::Identity(scaler)
    }
}

impl<'a> From<&'a StandardScaler> for ScalerRef<'a> {
    fn from(scaler: &'a StandardScaler) -> Self {
        Self::Standard(scaler)
    }
}

impl<'a> From<&'a MinMaxScaler> for ScalerRef<'a> {
    fn from(scaler: &'a MinMaxScaler) -> Self {
        Self::MinMax(scaler)
    }
}

fn validate_feature_range(feature_range: (f64, f64)) -> Result<()> {
    if !feature_range.0.is_finite()
        || !feature_range.1.is_finite()
        || feature_range.0 >= feature_range.1
    {
        return Err(candle_msg(format!(
            "MinMaxScaler feature range must be finite and ordered as (min < max), got ({}, {})",
            feature_range.0, feature_range.1
        )));
    }

    Ok(())
}

/// Bundle of fitted feature + target scalers, sized for caching across
/// requests (cheap to clone — `Tensor` fields are `Arc`-shared).
#[derive(Debug, Clone)]
pub struct FittedScalers {
    pub feature: StandardScaler,
    pub target: StandardScaler,
}

/// Standard score normalization with per-feature mean and population standard deviation.
#[derive(Debug, Clone, Default)]
pub struct StandardScaler {
    mean: Option<Tensor>,
    std: Option<Tensor>,
}

impl StandardScaler {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mean: None,
            std: None,
        }
    }

    pub fn from_stats(mean: Tensor, std: Tensor) -> Result<Self> {
        validate_feature_matrix(&mean, "StandardScaler", "from_stats")?;
        validate_feature_matrix(&std, "StandardScaler", "from_stats")?;
        validate_floating_dtype(mean.dtype(), "StandardScaler", "from_stats")?;
        validate_floating_dtype(std.dtype(), "StandardScaler", "from_stats")?;

        if mean.dims() != std.dims() {
            return Err(candle_msg(format!(
                "StandardScaler::from_stats expects matching mean/std shapes, got {:?} vs {:?}",
                mean.dims(),
                std.dims()
            )));
        }

        Ok(Self {
            mean: Some(mean.to_device(&Device::Cpu)?.to_dtype(DType::F64)?),
            std: Some(replace_zero_with_one(
                &std.to_device(&Device::Cpu)?.to_dtype(DType::F64)?,
            )?),
        })
    }

    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.mean.is_some() && self.std.is_some()
    }

    pub fn cloned_stats(&self) -> Result<(Tensor, Tensor)> {
        let (mean, std) = self.fitted_stats()?;
        Ok((mean.clone(), std.clone()))
    }

    pub fn with_negated_columns(&self, columns: &[usize]) -> Result<Self> {
        let (mean, std) = self.cloned_stats()?;
        let shape = mean.dims().to_vec();
        let feature_count = shape[1];
        let mut mean_values = mean.flatten_all()?.to_vec1::<f64>()?;

        for &column in columns {
            if column >= feature_count {
                return Err(candle_msg(format!(
                    "StandardScaler::with_negated_columns expected a column index below {feature_count}, got {column}"
                )));
            }
            mean_values[column] = -mean_values[column];
        }

        let mean = Tensor::from_vec(mean_values, shape.as_slice(), &Device::Cpu)?;
        Self::from_stats(mean, std)
    }

    fn fitted_stats(&self) -> Result<(&Tensor, &Tensor)> {
        self.mean.as_ref().zip(self.std.as_ref()).ok_or_else(|| {
            candle_msg("StandardScaler must be fitted before transform or inverse_transform")
        })
    }
}

impl Scaler for StandardScaler {
    fn fit(&mut self, data: &Tensor) -> Result<()> {
        validate_feature_matrix(data, "StandardScaler", "fit")?;
        validate_floating_dtype(data.dtype(), "StandardScaler", "fit")?;

        let data = data.to_device(&Device::Cpu)?.to_dtype(DType::F64)?;
        let mean = data.mean_keepdim(0)?;
        let centered = data.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(0)?;
        let std = replace_zero_with_one(&variance.sqrt()?)?;

        self.mean = Some(mean);
        self.std = Some(std);
        Ok(())
    }

    fn transform(&self, data: &Tensor) -> Result<Tensor> {
        let (_, feature_count) = validate_feature_matrix(data, "StandardScaler", "transform")?;
        validate_floating_dtype(data.dtype(), "StandardScaler", "transform")?;

        let (mean, std) = self.fitted_stats()?;
        validate_feature_count(feature_count, mean, "StandardScaler", "transform")?;

        let mean = align_stats_to_input(mean, data)?;
        let std = align_stats_to_input(std, data)?;
        data.broadcast_sub(mean.as_ref())?.broadcast_div(std.as_ref())
    }

    fn inverse_transform(&self, data: &Tensor) -> Result<Tensor> {
        let (_, feature_count) =
            validate_feature_matrix(data, "StandardScaler", "inverse_transform")?;
        validate_floating_dtype(data.dtype(), "StandardScaler", "inverse_transform")?;

        let (mean, std) = self.fitted_stats()?;
        validate_feature_count(feature_count, mean, "StandardScaler", "inverse_transform")?;

        let mean = align_stats_to_input(mean, data)?;
        let std = align_stats_to_input(std, data)?;
        data.broadcast_mul(std.as_ref())?.broadcast_add(mean.as_ref())
    }
}

/// Min-max normalization with configurable output range.
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    data_min: Option<Tensor>,
    data_max: Option<Tensor>,
    data_range: Option<Tensor>,
    feature_range: (f64, f64),
    target_span: f64,
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl MinMaxScaler {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            data_min: None,
            data_max: None,
            data_range: None,
            feature_range: (0.0, 1.0),
            target_span: 1.0,
        }
    }

    pub fn with_feature_range(min: f64, max: f64) -> Result<Self> {
        let feature_range = (min, max);
        validate_feature_range(feature_range)?;
        Ok(Self {
            feature_range,
            target_span: max - min,
            ..Self::new()
        })
    }

    #[must_use]
    pub fn feature_range(&self) -> (f64, f64) {
        self.feature_range
    }

    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.data_min.is_some() && self.data_max.is_some() && self.data_range.is_some()
    }

    fn fitted_stats(&self) -> Result<(&Tensor, &Tensor, &Tensor)> {
        self.data_min
            .as_ref()
            .zip(self.data_max.as_ref())
            .zip(self.data_range.as_ref())
            .map(|((data_min, data_max), data_range)| (data_min, data_max, data_range))
            .ok_or_else(|| {
                candle_msg("MinMaxScaler must be fitted before transform or inverse_transform")
            })
    }
}

impl Scaler for MinMaxScaler {
    fn fit(&mut self, data: &Tensor) -> Result<()> {
        validate_feature_matrix(data, "MinMaxScaler", "fit")?;
        validate_floating_dtype(data.dtype(), "MinMaxScaler", "fit")?;
        validate_feature_range(self.feature_range)?;

        let data = data.to_device(&Device::Cpu)?.to_dtype(DType::F64)?;
        let data_min = data.min_keepdim(0)?;
        let data_max = data.max_keepdim(0)?;
        let data_range = replace_zero_with_one(&data_max.broadcast_sub(&data_min)?)?;

        self.data_min = Some(data_min);
        self.data_max = Some(data_max);
        self.data_range = Some(data_range);
        Ok(())
    }

    fn transform(&self, data: &Tensor) -> Result<Tensor> {
        let (_, feature_count) = validate_feature_matrix(data, "MinMaxScaler", "transform")?;
        validate_floating_dtype(data.dtype(), "MinMaxScaler", "transform")?;

        let (data_min, _data_max, data_range) = self.fitted_stats()?;
        validate_feature_count(feature_count, data_min, "MinMaxScaler", "transform")?;

        let data_min = align_stats_to_input(data_min, data)?;
        let data_range = align_stats_to_input(data_range, data)?;

        data.broadcast_sub(data_min.as_ref())?
            .broadcast_div(data_range.as_ref())?
            .affine(self.target_span, self.feature_range.0)
    }

    fn inverse_transform(&self, data: &Tensor) -> Result<Tensor> {
        let (_, feature_count) =
            validate_feature_matrix(data, "MinMaxScaler", "inverse_transform")?;
        validate_floating_dtype(data.dtype(), "MinMaxScaler", "inverse_transform")?;

        let (data_min, _data_max, data_range) = self.fitted_stats()?;
        validate_feature_count(feature_count, data_min, "MinMaxScaler", "inverse_transform")?;

        let data_min = align_stats_to_input(data_min, data)?;
        let data_range = align_stats_to_input(data_range, data)?;
        let normalized = data.affine(
            1.0 / self.target_span,
            -self.feature_range.0 / self.target_span,
        )?;

        normalized
            .broadcast_mul(data_range.as_ref())?
            .broadcast_add(data_min.as_ref())
    }
}

fn replace_zero_with_one(values: &Tensor) -> Result<Tensor> {
    let zero_mask = values.eq(0.0)?;
    zero_mask.where_cond(&values.ones_like()?, values)
}

fn align_stats_to_input<'a>(stats: &'a Tensor, input: &Tensor) -> Result<Cow<'a, Tensor>> {
    if stats.dtype() == input.dtype() && stats.device().same_device(input.device()) {
        Ok(Cow::Borrowed(stats))
    } else {
        Ok(Cow::Owned(stats.to_device(input.device())?.to_dtype(input.dtype())?))
    }
}

fn validate_feature_matrix(
    data: &Tensor,
    scaler_name: &str,
    operation: &str,
) -> Result<(usize, usize)> {
    let dims = data.dims();
    if dims.len() != 2 {
        return Err(candle_msg(format!(
            "{scaler_name}::{operation} expects a 2D tensor shaped (batch_size, num_features), got {:?}",
            dims
        )));
    }
    if dims[0] == 0 || dims[1] == 0 {
        return Err(candle_msg(format!(
            "{scaler_name}::{operation} expects non-empty batch and feature dimensions, got {:?}",
            dims
        )));
    }
    Ok((dims[0], dims[1]))
}

fn validate_feature_count(
    feature_count: usize,
    stats: &Tensor,
    scaler_name: &str,
    operation: &str,
) -> Result<()> {
    let expected = stats.dims()[1];
    if feature_count != expected {
        return Err(candle_msg(format!(
            "{scaler_name}::{operation} expected {expected} features based on fitted statistics, got {feature_count}"
        )));
    }

    Ok(())
}

fn validate_floating_dtype(dtype: DType, scaler_name: &str, operation: &str) -> Result<()> {
    if matches!(
        dtype,
        DType::U8 | DType::U32 | DType::I16 | DType::I32 | DType::I64
    ) {
        return Err(candle_msg(format!(
            "{scaler_name}::{operation} expects a floating-point tensor, got {dtype:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_standard_scaler_fit_computes_population_mean_and_safe_std() -> Result<()> {
        let data = Tensor::from_vec(
            vec![1.0f64, 10.0, 3.0, 10.0, 5.0, 10.0],
            (3, 2),
            &Device::Cpu,
        )?;
        let mut scaler = StandardScaler::new();

        scaler.fit(&data)?;

        assert!(scaler.is_fitted());
        let mean = scaler
            .mean
            .as_ref()
            .expect("mean should be stored after fit")
            .flatten_all()?
            .to_vec1::<f64>()?;
        let std = scaler
            .std
            .as_ref()
            .expect("std should be stored after fit")
            .flatten_all()?
            .to_vec1::<f64>()?;

        assert_close(mean[0], 3.0);
        assert_close(mean[1], 10.0);
        assert_close(std[0], (8.0f64 / 3.0).sqrt());
        assert_close(std[1], 1.0);

        Ok(())
    }

    #[test]
    fn test_standard_scaler_transform_and_inverse_transform_round_trip() -> Result<()> {
        let data = Tensor::from_vec(
            vec![1.0f64, 10.0, 3.0, 10.0, 5.0, 10.0],
            (3, 2),
            &Device::Cpu,
        )?;
        let mut scaler = StandardScaler::new();

        let transformed = scaler.fit_transform(&data)?;
        let recovered = scaler.inverse_transform(&transformed)?;
        let transformed_values = transformed.flatten_all()?.to_vec1::<f64>()?;
        let recovered_values = recovered.flatten_all()?.to_vec1::<f64>()?;
        let original_values = data.flatten_all()?.to_vec1::<f64>()?;

        assert_close(transformed_values[0], -1.224_744_871_391_589);
        assert_close(transformed_values[1], 0.0);
        assert_close(transformed_values[2], 0.0);
        assert_close(transformed_values[3], 0.0);
        assert_close(transformed_values[4], 1.224_744_871_391_589);
        assert_close(transformed_values[5], 0.0);

        for (actual, expected) in recovered_values.iter().zip(original_values.iter()) {
            assert_close(*actual, *expected);
        }

        Ok(())
    }

    #[test]
    fn test_standard_scaler_transforms_new_batches_with_fitted_statistics() -> Result<()> {
        let train = Tensor::from_vec(
            vec![1.0f64, 10.0, 3.0, 10.0, 5.0, 10.0],
            (3, 2),
            &Device::Cpu,
        )?;
        let batch = Tensor::from_vec(vec![7.0f64, 10.0, -1.0, 10.0], (2, 2), &Device::Cpu)?;
        let mut scaler = StandardScaler::new();

        scaler.fit(&train)?;
        let transformed = scaler.transform(&batch)?;
        let values = transformed.flatten_all()?.to_vec1::<f64>()?;

        assert_eq!(transformed.dims(), &[2, 2]);
        assert_close(values[0], 2.449_489_742_783_178);
        assert_close(values[1], 0.0);
        assert_close(values[2], -2.449_489_742_783_178);
        assert_close(values[3], 0.0);

        Ok(())
    }

    #[test]
    fn test_standard_scaler_rejects_unfitted_or_invalid_input_shapes() -> Result<()> {
        let vector = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], 3, &Device::Cpu)?;
        let data = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)?;
        let scaler = StandardScaler::new();
        let mut fit_scaler = StandardScaler::new();

        let fit_error = fit_scaler
            .fit(&vector)
            .expect_err("1D tensors should be rejected");
        let transform_error = scaler
            .transform(&data)
            .expect_err("transform without fit should be rejected");

        assert!(fit_error.to_string().contains("2D tensor"));
        assert!(transform_error.to_string().contains("must be fitted"));

        Ok(())
    }

    #[test]
    fn test_standard_scaler_with_negated_columns_matches_refit() -> Result<()> {
        let train = Tensor::from_vec(
            vec![1.0f64, 10.0, 3.0, 14.0, 5.0, 18.0],
            (3, 2),
            &Device::Cpu,
        )?;
        let negated_train = Tensor::from_vec(
            vec![1.0f64, -10.0, 3.0, -14.0, 5.0, -18.0],
            (3, 2),
            &Device::Cpu,
        )?;
        let mut base = StandardScaler::new();
        let mut refit = StandardScaler::new();

        base.fit(&train)?;
        refit.fit(&negated_train)?;

        let derived = base.with_negated_columns(&[1])?;
        let derived_values = derived.transform(&negated_train)?.flatten_all()?.to_vec1::<f64>()?;
        let refit_values = refit.transform(&negated_train)?.flatten_all()?.to_vec1::<f64>()?;

        for (actual, expected) in derived_values.iter().zip(refit_values.iter()) {
            assert_close(*actual, *expected);
        }

        Ok(())
    }

    #[test]
    fn test_minmax_scaler_default_range_round_trips_and_handles_constant_columns() -> Result<()> {
        let data = Tensor::from_vec(
            vec![1.0f64, 10.0, 3.0, 10.0, 5.0, 10.0],
            (3, 2),
            &Device::Cpu,
        )?;
        let mut scaler = MinMaxScaler::new();

        let transformed = scaler.fit_transform(&data)?;
        let recovered = scaler.inverse_transform(&transformed)?;
        let transformed_values = transformed.flatten_all()?.to_vec1::<f64>()?;
        let recovered_values = recovered.flatten_all()?.to_vec1::<f64>()?;
        let original_values = data.flatten_all()?.to_vec1::<f64>()?;

        assert!(scaler.is_fitted());
        assert_eq!(scaler.feature_range(), (0.0, 1.0));
        assert_close(transformed_values[0], 0.0);
        assert_close(transformed_values[1], 0.0);
        assert_close(transformed_values[2], 0.5);
        assert_close(transformed_values[3], 0.0);
        assert_close(transformed_values[4], 1.0);
        assert_close(transformed_values[5], 0.0);

        for (actual, expected) in recovered_values.iter().zip(original_values.iter()) {
            assert_close(*actual, *expected);
        }

        Ok(())
    }

    #[test]
    fn test_minmax_scaler_supports_custom_feature_range() -> Result<()> {
        let train = Tensor::from_vec(vec![1.0f64, 10.0, 5.0, 10.0], (2, 2), &Device::Cpu)?;
        let batch = Tensor::from_vec(
            vec![1.0f64, 10.0, 3.0, 10.0, 5.0, 10.0],
            (3, 2),
            &Device::Cpu,
        )?;
        let mut scaler = MinMaxScaler::with_feature_range(-1.0, 1.0)?;

        scaler.fit(&train)?;
        let transformed = scaler.transform(&batch)?;
        let values = transformed.flatten_all()?.to_vec1::<f64>()?;

        assert_eq!(scaler.feature_range(), (-1.0, 1.0));
        assert_close(values[0], -1.0);
        assert_close(values[1], -1.0);
        assert_close(values[2], 0.0);
        assert_close(values[3], -1.0);
        assert_close(values[4], 1.0);
        assert_close(values[5], -1.0);

        Ok(())
    }

    #[test]
    fn test_minmax_scaler_extrapolates_outside_fitted_feature_range() -> Result<()> {
        let train = Tensor::from_vec(vec![1.0f64, 5.0], (2, 1), &Device::Cpu)?;
        let batch = Tensor::from_vec(vec![0.0f64, 6.0], (2, 1), &Device::Cpu)?;
        let mut scaler = MinMaxScaler::new();

        scaler.fit(&train)?;
        let transformed = scaler.transform(&batch)?;
        let values = transformed.flatten_all()?.to_vec1::<f64>()?;

        assert_close(values[0], -0.25);
        assert_close(values[1], 1.25);

        Ok(())
    }

    #[test]
    fn test_minmax_scaler_rejects_invalid_range_and_unfitted_or_mismatched_inputs() -> Result<()> {
        let invalid_range = MinMaxScaler::with_feature_range(1.0, 1.0)
            .expect_err("equal feature range bounds should be rejected");
        let train = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)?;
        let wrong_features = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], (1, 3), &Device::Cpu)?;
        let scaler = MinMaxScaler::new();
        let mut fitted_scaler = MinMaxScaler::new();

        fitted_scaler.fit(&train)?;

        let unfitted_error = scaler
            .transform(&train)
            .expect_err("transform without fit should be rejected");
        let mismatch_error = fitted_scaler
            .transform(&wrong_features)
            .expect_err("feature-count mismatch should be rejected");

        assert!(invalid_range.to_string().contains("feature range"));
        assert!(unfitted_error.to_string().contains("must be fitted"));
        assert!(mismatch_error.to_string().contains("expected 2 features"));

        Ok(())
    }
}

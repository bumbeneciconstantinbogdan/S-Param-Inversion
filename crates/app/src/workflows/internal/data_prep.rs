//! Data preparation: tensor conversion, splitting, scaling, model construction,
//! and data loader helpers.

use candle_core::{DType, Result, Tensor};

use sparam_core::error::candle_msg;
use sparam_data::loader::{BatchSize, DataLoader};
use sparam_data::scaling::{FittedScalers, Scaler, StandardScaler};
use sparam_data::tensor_bridges::{PredictionEncoding, TensorDataset, samples_to_tensors};
use sparam_data::generation::PermittivitySample;
use sparam_models::ModelType;

pub(crate) struct PreparedTensorSplits {
    pub(crate) train_features: Tensor,
    pub(crate) train_targets: Tensor,
    pub(crate) val_features: Tensor,
    pub(crate) val_targets: Tensor,
    pub(crate) test_features: Tensor,
    pub(crate) test_targets: Tensor,
    pub(crate) feature_scaler: StandardScaler,
    pub(crate) target_scaler: StandardScaler,
    pub(crate) encoding: PredictionEncoding,
}

impl PreparedTensorSplits {
    pub(crate) fn dtype(&self) -> DType {
        self.train_features.dtype()
    }
}

pub struct PreparedEvaluationData {
    pub features: Tensor,
    pub targets: Tensor,
    pub feature_scaler: StandardScaler,
    pub target_scaler: StandardScaler,
    pub encoding: PredictionEncoding,
}

pub(crate) fn parse_batch_size(batch_size: &str) -> Result<BatchSize> {
    match batch_size.to_ascii_uppercase().as_str() {
        "ALL" => Ok(BatchSize::All),
        value => value
            .parse::<usize>()
            .map(BatchSize::Fixed)
            .map_err(|error| candle_msg(format!("invalid batch size '{value}': {error}"))),
    }
}

pub(crate) fn build_loader(
    features: &Tensor,
    targets: &Tensor,
    batch_size: BatchSize,
    feature_scaler: &StandardScaler,
    target_scaler: &StandardScaler,
    shuffle: bool,
    seed: u64,
) -> Result<DataLoader> {
    // Transform once up front, hand already-scaled tensors to the
    // loader. Replaces the old per-batch `(x − μ) / σ` allocations.
    let scaled_features = feature_scaler.transform(features)?;
    let scaled_targets = target_scaler.transform(targets)?;
    let loader = DataLoader::new(scaled_features, scaled_targets, batch_size)?
        .with_shuffle(shuffle)
        .with_seed(seed);
    Ok(loader)
}

pub(crate) fn prepare_splits_from_samples(
    train: &[PermittivitySample],
    val: &[PermittivitySample],
    test: &[PermittivitySample],
    model_type: ModelType,
    real_dtype: DType,
    negate_real_imag: bool,
) -> Result<PreparedTensorSplits> {
    let train_ds = tensors_from_samples(model_type, train, real_dtype, negate_real_imag)?;
    let val_ds = tensors_from_samples(model_type, val, real_dtype, negate_real_imag)?;
    let test_ds = tensors_from_samples(model_type, test, real_dtype, negate_real_imag)?;
    let (feature_scaler, target_scaler) =
        fit_standard_scalers(&train_ds.features, &train_ds.targets)?;

    Ok(PreparedTensorSplits {
        train_features: train_ds.features,
        train_targets: train_ds.targets,
        val_features: val_ds.features,
        val_targets: val_ds.targets,
        test_features: test_ds.features,
        test_targets: test_ds.targets,
        feature_scaler,
        target_scaler,
        encoding: train_ds.encoding,
    })
}

pub(crate) fn fit_standard_scalers(
    train_features: &Tensor,
    train_targets: &Tensor,
) -> Result<(StandardScaler, StandardScaler)> {
    let mut feature_scaler = StandardScaler::new();
    feature_scaler.fit(train_features)?;
    let mut target_scaler = StandardScaler::new();
    target_scaler.fit(train_targets)?;
    Ok((feature_scaler, target_scaler))
}

pub fn prepare_evaluation_data(
    eval_samples: &[PermittivitySample],
    scaler_samples: &[PermittivitySample],
    model_type: ModelType,
    real_dtype: DType,
    negate_real_imag: bool,
    pre_fitted: Option<&FittedScalers>,
) -> Result<PreparedEvaluationData> {
    let eval_ds = tensors_from_samples(model_type, eval_samples, real_dtype, negate_real_imag)?;

    let (feature_scaler, target_scaler) = if let Some(fitted) = pre_fitted {
        // Cache hit: skip the StandardScaler::fit on `scaler_samples`.
        // `StandardScaler::clone` is cheap (Tensor fields are Arc-shared).
        (fitted.feature.clone(), fitted.target.clone())
    } else {
        let scaler_ds =
            tensors_from_samples(model_type, scaler_samples, real_dtype, negate_real_imag)?;
        fit_standard_scalers(&scaler_ds.features, &scaler_ds.targets)?
    };

    Ok(PreparedEvaluationData {
        features: eval_ds.features,
        targets: eval_ds.targets,
        feature_scaler,
        target_scaler,
        encoding: eval_ds.encoding,
    })
}

/// Wraps [`samples_to_tensors`] and casts Real tensors to the
/// caller-requested dtype. Complex tensors stay at their native dtype
/// (f64 after packing).
fn tensors_from_samples(
    model_type: ModelType,
    samples: &[PermittivitySample],
    real_dtype: DType,
    negate_real_imag: bool,
) -> Result<TensorDataset> {
    let ds = samples_to_tensors(model_type, samples, negate_real_imag)?;
    if matches!(model_type, ModelType::Real) {
        Ok(TensorDataset {
            features: ds.features.to_dtype(real_dtype)?,
            targets: ds.targets.to_dtype(real_dtype)?,
            encoding: ds.encoding,
        })
    } else {
        Ok(ds)
    }
}
